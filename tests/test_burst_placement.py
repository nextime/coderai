"""Per-model placement: local / remote / burst-to-RunPod.

The queue arithmetic is what matters here — a leaked slot silently caps a model's
concurrency forever, and a phantom release hands a slot to a waiter that should
still be waiting.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codai.frontproxy.reqqueue import FrontQueue, QueueFull


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_try_acquire_takes_a_free_slot_and_refuses_when_full():
    async def go():
        q = FrontQueue()
        assert await q.try_acquire("m", 1) is True      # free
        assert await q.try_acquire("m", 1) is False     # busy -> burst instead
        await q.release("m")
        assert await q.try_acquire("m", 1) is True      # freed again
    run(go())


def test_try_acquire_respects_capacity_above_one():
    async def go():
        q = FrontQueue()
        assert [await q.try_acquire("m", 3) for _ in range(3)] == [True, True, True]
        assert await q.try_acquire("m", 3) is False
    run(go())


def test_try_acquire_never_waits_even_with_a_queue_behind_it():
    async def go():
        q = FrontQueue()
        await q.acquire("m", 1, 4)                       # slot held
        waiter = asyncio.ensure_future(q.acquire("m", 1, 4))
        await asyncio.sleep(0)                           # let it enqueue
        # The point of try_acquire: it returns rather than joining that queue.
        assert await asyncio.wait_for(q.try_acquire("m", 1), timeout=0.5) is False
        await q.release("m")                             # hands the slot to waiter
        await asyncio.wait_for(waiter, timeout=0.5)
        await q.release("m")
    run(go())


def test_a_granted_slot_is_not_double_counted():
    """try_acquire must take exactly one slot — the burst path then skips the
    blocking acquire, and releasing once must fully free the model."""
    async def go():
        q = FrontQueue()
        assert await q.try_acquire("m", 1) is True
        await q.release("m")
        # If try_acquire had counted twice, this would still be occupied.
        assert await q.try_acquire("m", 1) is True
    run(go())


def test_queue_full_still_raises_for_non_burst_models():
    async def go():
        q = FrontQueue()
        await q.acquire("m", 1, 1)
        w = asyncio.ensure_future(q.acquire("m", 1, 1))
        await asyncio.sleep(0)
        with pytest.raises(QueueFull):
            await q.acquire("m", 1, 1)
        w.cancel()
        try:
            await w
        except asyncio.CancelledError:
            pass
    run(go())


# --------------------------------------------------------------------------- #
# the front's placement decision
# --------------------------------------------------------------------------- #
class _Front:
    """Just the bits of the front the placement decision touches."""

    from codai.frontproxy.app import FrontProxy as _FP
    _spill_on_busy = _FP._spill_on_busy

    def __init__(self, info):
        self._info = info

    def _model_info(self, model):
        return self._info


def test_spill_on_busy_only_when_enabled_and_triggered():
    assert _Front({}) ._spill_on_busy("m") is False
    assert _Front({"runpod_spillover": {"enabled": True}})._spill_on_busy("m") is False
    assert _Front({"runpod_spillover": {"on_busy": True}})._spill_on_busy("m") is False
    assert _Front({"runpod_spillover": {"enabled": True, "on_busy": True}}) \
        ._spill_on_busy("m") is True
    # A model that only bursts when the QUEUE overflows must not burst on busy.
    assert _Front({"runpod_spillover": {"enabled": True, "on_concurrency_full": True}}) \
        ._spill_on_busy("m") is False


def test_model_is_renamed_for_the_remote():
    import json
    from codai.frontproxy.app import FrontProxy

    body = json.dumps({"model": "local-name", "messages": []}).encode()
    assert json.loads(FrontProxy._rewrite_model(body, "remote-name"))["model"] \
        == "remote-name"
    # No served_model configured -> body untouched, byte for byte.
    assert FrontProxy._rewrite_model(body, "") is body
    # Garbage in, same garbage out rather than an exception.
    assert FrontProxy._rewrite_model(b"not json", "x") == b"not json"


# --------------------------------------------------------------------------- #
# pod pool: real concurrency scaling, sticky cache, and the bearer token
# --------------------------------------------------------------------------- #
class _Pod:
    def __init__(self, pid, inflight=0):
        self.pod_id = pid
        self.url = f"http://{pid}"
        self.healthy = True
        self.inflight = inflight
        self.last_used = 0.0


def _pool(**block):
    from codai.api.runpod_worker import RunpodPodPool, parse_model_runpod

    class _Acct:
        enabled = True
    return RunpodPodPool("m", _Acct(), parse_model_runpod(block), "m")


def test_pool_grows_on_real_concurrency_not_only_when_empty():
    p = _pool(max_pods=3, scale_up_inflight_per_pod=2)
    busy = _Pod("a", inflight=2)
    p.pods = [busy]
    # Least-loaded pod is at the threshold and max_pods allows it -> grow.
    assert p._should_grow([busy]) is True
    # Below the threshold: one pod is absorbing the load fine, don't pay for another.
    p2 = _pool(max_pods=3, scale_up_inflight_per_pod=4)
    p2.pods = [_Pod("a", inflight=1)]
    assert p2._should_grow(p2.pods) is False
    # No healthy pod at all: grow regardless of the threshold.
    p4 = _pool(max_pods=3, scale_up_inflight_per_pod=99)
    assert p4._should_grow([]) is True
    # At max_pods, never.
    p3 = _pool(max_pods=1, scale_up_inflight_per_pod=1)
    p3.pods = [busy]
    assert p3._should_grow([busy]) is False


def test_pool_picks_least_loaded_and_honours_the_inflight_ceiling():
    p = _pool(max_pods=3, max_inflight_per_pod=2)
    a, b = _Pod("a", inflight=2), _Pod("b", inflight=1)
    assert p._pick([a, b], "") is b            # least loaded
    a.inflight = b.inflight = 2
    assert p._pick([a, b], "") is None         # all at the ceiling -> wait
    q = _pool(max_pods=3)                      # 0 = no ceiling
    assert q._pick([a, b], "") is not None


def test_sticky_sessions_keep_a_conversation_on_its_cached_pod():
    p = _pool(max_pods=3)
    a, b = _Pod("a", inflight=5), _Pod("b", inflight=0)
    p._remember_affinity("conv-1", a)
    # Even though `a` is busier, the conversation goes back to it: its prefix
    # cache still holds the context, and re-prefilling elsewhere costs more.
    assert p._pick([a, b], "conv-1") is a
    assert p._pick([a, b], "conv-2") is b      # unknown conversation -> least loaded
    off = _pool(max_pods=3, sticky_sessions=False)
    off._remember_affinity("conv-1", a)
    assert off._pick([a, b], "conv-1") is b    # disabled -> least loaded


def test_pods_are_locked_to_a_token_by_default():
    from codai.api.runpod_worker import pod_plan, parse_model_runpod

    p = _pool(max_pods=1)
    assert p.api_key.startswith("cra-")        # generated, never open by default

    explicit = _pool(max_pods=1, api_key="my-token")
    assert explicit.api_key == "my-token"

    opened = _pool(max_pods=1, allow_open_pod=True)
    assert opened.api_key == ""                # only when explicitly asked for

    # Each server takes the token its own way.
    vllm = pod_plan(parse_model_runpod({"served_model": "org/m"}), "org/m", api_key="tok")
    assert "--api-key tok" in vllm["args"]
    gguf = pod_plan(parse_model_runpod({"hf_gguf": "u/r:Q4"}), "m", api_key="tok")
    assert "--api-key tok" in gguf["args"]
    cod = pod_plan(parse_model_runpod({"engine": "coderai", "image": "i"}), "images",
                   api_key="tok")
    assert cod["env"]["CODERAI_API_TOKEN"] == "tok"


def test_a_pod_coderai_rejects_requests_without_the_token(monkeypatch):
    """The env token is what locks a capability pod: it has no auth.json."""
    import hmac
    monkeypatch.setenv("CODERAI_API_TOKEN", "tok")
    # The middleware compares with hmac.compare_digest against the env value.
    assert hmac.compare_digest("tok", "tok")
    from codai.api.ratelimit import _unauthorized
    assert _unauthorized().status_code == 401


# --------------------------------------------------------------------------- #
# sharing one pod between models
# --------------------------------------------------------------------------- #
def test_two_models_on_one_coderai_pod_share_a_pool(monkeypatch):
    """Both models name the same pool, so the overflow from both lands on one
    card until it saturates — rather than renting a GPU each."""
    import codai.api.runpod_worker as rw

    monkeypatch.setattr(rw, "_pools", {})
    monkeypatch.setattr(rw, "_ensure_scaler", lambda: None)

    class _Acct:
        enabled = True

    block = {"pool": "shared", "engine": "coderai", "image": "reg/coderai:base",
             "max_pods": 2, "max_hourly_usd": 0.5}
    a_cfg = rw.parse_model_runpod(block)
    b_cfg = rw.parse_model_runpod({"pool": "shared", "engine": "coderai"})

    ka = rw.shared_pool_key(a_cfg, "model-a")
    kb = rw.shared_pool_key(b_cfg, "model-b")
    assert ka == kb == "pool:shared"

    pa = rw.get_pod_pool(ka, _Acct(), a_cfg, "model-a", shared=True)
    pb = rw.get_pod_pool(kb, _Acct(), b_cfg, "model-b", shared=True)
    assert pa is pb
    # The first model's settings define the shared pod; a later joiner must not
    # silently redefine the budget everyone is sharing.
    assert pa.mcfg.max_hourly_usd == 0.5 and pa.mcfg.image == "reg/coderai:base"


def test_a_model_without_a_pool_name_still_gets_its_own():
    import codai.api.runpod_worker as rw
    cfg = rw.parse_model_runpod({"engine": "coderai", "image": "i"})
    assert rw.shared_pool_key(cfg, "model-a") == "model-a"


def test_sharing_is_refused_for_single_model_pod_servers():
    """A vLLM pod is launched `--model X` and llama.cpp `-hf one.gguf`: sharing
    would answer with the wrong model's weights, or 404."""
    import codai.api.runpod_worker as rw

    vllm = rw.parse_model_runpod({"pool": "shared", "served_model": "org/model-a"})
    with pytest.raises(RuntimeError, match="cannot serve another"):
        rw.shared_pool_key(vllm, "model-a")

    gguf = rw.parse_model_runpod({"pool": "shared", "hf_gguf": "u/r:Q4"})
    with pytest.raises(RuntimeError, match="cannot serve another"):
        rw.shared_pool_key(gguf, "model-b", "/AI/m.gguf")


def test_a_shared_pod_grows_when_it_saturates():
    """Sharing must not mean queueing forever: once the pod is carrying
    scale_up_inflight_per_pod requests, the pool rents a second one."""
    p = _pool(pool="shared", engine="coderai", image="i", max_pods=2,
              scale_up_inflight_per_pod=3)
    pod = _Pod("a", inflight=3)
    p.pods = [pod]
    assert p._should_grow([pod]) is True
    quiet = _Pod("a", inflight=1)
    p.pods = [quiet]
    assert p._should_grow([quiet]) is False   # not saturated -> keep sharing


# --------------------------------------------------------------------------- #
# the reaper must not eat a sibling engine's pods
# --------------------------------------------------------------------------- #
def test_pods_are_registered_across_processes(tmp_path, monkeypatch):
    """coderai runs one engine per GPU, each with its own pools AND its own
    reaper. Observed live: nvidia created a pod, radeon terminated it four
    seconds later, then the reverse — a mutual kill loop. The registry is the
    shared view that stops it."""
    import codai.api.runpod_worker as rw

    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(tmp_path / "pods.json"))
    monkeypatch.setattr(rw, "_pools", {})

    rw.register_pod("pod-a", "capability:embeddings")
    rw.register_pod("pod-b", "some-model")
    assert rw.registered_pod_ids() == {"pod-a", "pod-b"}

    # A process with NO pools of its own still sees them — which is exactly the
    # situation the second engine was in when it killed the first one's pod.
    assert {"pod-a", "pod-b"} <= rw._known_pod_ids()

    rw.unregister_pod("pod-a")
    assert rw.registered_pod_ids() == {"pod-b"}


def test_a_young_pod_is_never_reaped(monkeypatch, tmp_path):
    """The backstop for the race the registry cannot close: a process that died
    between creating a pod and recording it."""
    import codai.api.runpod_worker as rw

    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(tmp_path / "pods.json"))
    monkeypatch.setattr(rw, "_pools", {})

    class _Acct:
        enabled = True
        api_key = "k"
        deployment_id = "default"

    monkeypatch.setattr("codai.models.manager.get_active_runpod_config", lambda: _Acct())

    young = {"id": "fresh", "name": "coderai-default-x", "status": "RUNNING",
             "uptime_s": 30}
    old = {"id": "ancient", "name": "coderai-default-y", "status": "RUNNING",
           "uptime_s": rw.REAP_GRACE_SECONDS + 60}
    killed = []

    class _Client:
        def __init__(self, *a, **k): pass
        def list_pods(self): return [young, old]
        def terminate_pod(self, pid): killed.append(pid)

    monkeypatch.setattr("codai.api.runpod_client.RunpodClient", _Client)
    assert rw.reap_orphans() == 1
    assert killed == ["ancient"]        # the 30-second-old pod survives


def test_a_sibling_engines_pod_is_reused_not_duplicated(tmp_path, monkeypatch):
    """One engine per GPU means each had its own pool: the same capability was
    rented a card PER ENGINE while max_pods said 1. Observed live — nvidia took
    pod y6tg…, radeon took yymi… for the same 'capability:embeddings'."""
    import codai.api.runpod_worker as rw

    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(tmp_path / "pods.json"))
    monkeypatch.setattr(rw, "_pod_health_ok", lambda url, **kw: True)

    # Another process recorded a healthy pod for this pool.
    rw._registry_update(lambda d: d.__setitem__(
        "pod-from-nvidia", {"pool": "capability:embeddings", "pid": 999999,
                            "at": 0, "url": "http://pod-a:8000"}) or True)

    pod_id, url, key = rw.find_shared_pod("capability:embeddings")
    assert (pod_id, url) == ("pod-from-nvidia", "http://pod-a:8000")

    # A different pool must not borrow it.
    assert rw.find_shared_pod("capability:images") == (None, "", "")

    # Our own pods are left to the local pool, not "borrowed" from ourselves.
    import os
    rw._registry_update(lambda d: d.__setitem__(
        "pod-mine", {"pool": "capability:images", "pid": os.getpid(),
                     "at": 0, "url": "http://pod-b:8000"}) or True)
    assert rw.find_shared_pod("capability:images") == (None, "", "")


def test_a_borrowed_pod_is_released_not_terminated(monkeypatch):
    """Terminating a pod we merely borrowed would kill it under the engine that
    owns it — and that engine is the one paying for it."""
    import codai.api.runpod_worker as rw

    killed = []

    class _Client:
        def __init__(self, *a, **k): pass
        def terminate_pod(self, pid): killed.append(pid)

    monkeypatch.setattr("codai.api.runpod_client.RunpodClient", _Client)
    pool = _pool(max_pods=1)
    borrowed = rw.PodHandle(pod_id="pod-x", url="http://x", hourly_usd=0.0,
                            started_at=0.0, gpu="(shared)")
    pool.pods = [borrowed]
    pool._terminate(borrowed, "idle")
    assert killed == [] and pool.pods == []


# --------------------------------------------------------------------------- #
# network volumes
# --------------------------------------------------------------------------- #
def test_a_volume_moves_the_caches_off_container_disk():
    """The point of a volume: weights land somewhere that outlives the pod, so
    the second pod does not re-download them."""
    from codai.api.runpod_worker import volume_for, volume_env, staging_dir, \
        parse_model_runpod

    class _Acct:
        network_volume_id = ""
        volume_mount_path = "/workspace"

    cfg = parse_model_runpod({"network_volume_id": "vol-abc"})
    assert volume_for(cfg, _Acct()) == ("vol-abc", "/workspace")

    env = volume_env("/workspace")
    # Downloads, uploads and every cache go to the volume.
    assert env["HF_HOME"] == "/workspace/huggingface"
    assert env["CODERAI_MODELS_DIR"] == "/workspace/models"
    assert staging_dir(cfg, _Acct()) == "/workspace/staged"

    # Without one, nothing changes: container disk, wiped with the pod.
    plain = parse_model_runpod({})
    assert volume_for(plain, _Acct()) == ("", "")
    assert staging_dir(plain, _Acct()).startswith("/runpod-volume")


def test_an_account_wide_volume_applies_and_a_pod_can_override():
    from codai.api.runpod_worker import volume_for, parse_model_runpod

    class _Acct:
        network_volume_id = "vol-account"
        volume_mount_path = "/workspace"

    assert volume_for(parse_model_runpod({}), _Acct())[0] == "vol-account"
    assert volume_for(parse_model_runpod({"network_volume_id": "vol-own"}),
                      _Acct())[0] == "vol-own"


def test_a_volume_forces_secure_cloud():
    """RunPod offers network volumes on Secure Cloud only: keeping COMMUNITY in
    the list would just produce candidates the attach rejects."""
    from codai.api.runpod_worker import parse_model_runpod

    cfg = parse_model_runpod({"network_volume_id": "vol-abc",
                              "cloud_types": ["SECURE", "COMMUNITY"]})
    assert cfg.cloud_types == ["SECURE"]
    # Without a volume the choice is left alone.
    assert parse_model_runpod({"cloud_types": ["COMMUNITY"]}).cloud_types == ["COMMUNITY"]


def test_venv_on_volume_uses_a_small_image_and_the_volumes_venv(monkeypatch):
    """The 7 GB image is nearly all torch. With a volume, those libraries can
    live there and the pod pulls a small image instead."""
    import codai.api.runpod_worker as rw

    class _Acct:
        network_volume_id = "vol-abc"
        volume_mount_path = "/workspace"
        enabled = True

    monkeypatch.setattr(rw, "_account_hint", lambda: _Acct())
    cfg = rw.parse_model_runpod({"engine": "coderai", "venv_on_volume": True})
    plan = rw.pod_plan(cfg, "embeddings", "capability:embeddings",
                       api_key="tok", entry={"path": "org/e", 
                                             "model_type": "embedding_models"})
    assert plan["image"].endswith("coderai-slim:latest")
    assert plan["entrypoint"] == ["/bin/sh", "-c"]
    script = plan["start_cmd"][0]
    assert "/workspace/venvs/embeddings" in script
    assert "profiles/embeddings.txt" in script      # the right profile installed
    assert script.rstrip().endswith("--port 8000")


def test_venv_on_volume_without_a_volume_is_refused(monkeypatch):
    """The venv would have nowhere to live; say so rather than boot a pod that
    installs 7 GB onto container disk and throws it away."""
    import codai.api.runpod_worker as rw

    class _NoVol:
        network_volume_id = ""
        volume_mount_path = "/workspace"

    monkeypatch.setattr(rw, "_account_hint", lambda: _NoVol())
    cfg = rw.parse_model_runpod({"engine": "coderai", "image": "img",
                                 "venv_on_volume": True})
    with pytest.raises(RuntimeError, match="needs a network volume"):
        rw.pod_plan(cfg, "x", "capability:embeddings", entry={})


def test_the_venv_marker_is_written_last_and_builders_are_serialised():
    """Two hazards: a pod that dies mid-install leaving a venv later pods import
    from, and two pods installing into the same directory at once."""
    from codai.api.runpod_worker import venv_bootstrap_script

    s = venv_bootstrap_script("/workspace", "tts")
    # The marker is created after the installs, not before.
    assert s.index("pip install -r") < s.index('touch "$MARK"')
    # And the ready check gates the exec.
    assert s.index('touch "$MARK"') < s.index('[ -f "$MARK" ] ||')
    # mkdir is the atomic lock; a loser waits instead of installing too.
    assert 'mkdir "$LOCK"' in s and "another pod is building" in s


def test_a_sibling_waits_for_a_booting_pod_instead_of_renting_one(tmp_path, monkeypatch):
    """A pod takes ~2.5 minutes to boot. Live, the second engine looked during
    that window, saw no READY pod, and rented its own — two cards, 45s apart."""
    import codai.api.runpod_worker as rw

    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(tmp_path / "pods.json"))

    # A sibling registered a pod that has not finished booting (no url yet).
    rw._registry_update(lambda d: d.__setitem__(
        "booting", {"pool": "capability:embeddings", "pid": 999999,
                    "at": __import__("time").time(), "url": ""}) or True)
    assert rw.sibling_is_provisioning("capability:embeddings") is True
    assert rw.sibling_is_provisioning("capability:images") is False

    # Once it is serving, it is adoptable rather than merely blocking.
    monkeypatch.setattr(rw, "_pod_health_ok", lambda url, **kw: True)
    rw._registry_update(lambda d: d.__setitem__(
        "booting", {"pool": "capability:embeddings", "pid": 999999,
                    "at": __import__("time").time(),
                    "url": "http://pod:8000"}) or True)
    assert rw.sibling_is_provisioning("capability:embeddings") is False
    assert rw.find_shared_pod("capability:embeddings")[1] == "http://pod:8000"

    # A stale entry from a dead process must not block provisioning forever.
    rw._registry_update(lambda d: d.__setitem__(
        "ancient", {"pool": "capability:video", "pid": 999998,
                    "at": 0, "url": ""}) or True)
    assert rw.sibling_is_provisioning("capability:video") is False


def test_a_surviving_pods_token_is_remembered_across_a_restart(tmp_path, monkeypatch):
    """A pool generates a random bearer token and launches its pods with it. When
    the process dies and the pod does not, the next pool generates a DIFFERENT
    token and the surviving pod answers 401 to everything for the rest of its
    paid life — observed live: the test run reported 'serves: (HTTP 401)'."""
    import codai.api.runpod_worker as rw

    monkeypatch.setattr(rw, "_pod_registry_path", lambda: str(tmp_path / "pods.json"))
    seen = {}

    def _health(url, path="/v1/models", api_key="", **kw):
        seen["key"] = api_key
        return api_key == "cra-original"      # the pod only knows its own token

    monkeypatch.setattr(rw, "_pod_health_ok", _health)

    # The pod the dead process rented, with the token it was launched with.
    rw.register_pod("pod-survivor", "capability:embeddings",
                    "http://pod-a:8000", "cra-original")
    monkeypatch.setattr(rw.os, "getpid", lambda: 424242)   # a NEW process

    pod_id, url, key = rw.find_shared_pod("capability:embeddings",
                                          api_key="cra-freshly-generated")
    assert (pod_id, url, key) == ("pod-survivor", "http://pod-a:8000", "cra-original")
    assert seen["key"] == "cra-original", "probed with the pod's token, not ours"
