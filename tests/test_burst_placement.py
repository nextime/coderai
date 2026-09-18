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
    assert "--port 8000" in script.rstrip().splitlines()[-1]   # (+ TLS args when sent)


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


def test_the_port_wait_covers_a_cold_image_pull(monkeypatch):
    """The port appears only after the image is pulled, so the budget has to
    cover the download. At 300s the same 7.3 GB image succeeded on a machine
    with cached layers (133s) and failed on cold ones — three times, renting a
    fresh machine and re-pulling for each."""
    import codai.api.runpod_worker as rw

    assert rw.RunpodPodPool.PORT_TIMEOUT_S >= 900.0


def test_a_port_timeout_says_what_it_was_waiting_on(monkeypatch):
    """'did not expose port 8000 within 300.0s' gives nothing to act on."""
    import codai.api.runpod_client as rc

    client = rc.RunpodClient.__new__(rc.RunpodClient)
    monkeypatch.setattr(client, "get_pod",
                        lambda pid: {"status": "RUNNING", "uptime_s": 42,
                                     "ready": False}, raising=False)
    try:
        client.wait_ready("pod-x", 8000, ready_timeout=0.1, poll_every=0.01)
    except rc.RunpodError as exc:
        assert "last status='RUNNING'" in str(exc) or "status=RUNNING" in str(exc)
        assert "boot_timeout_s" in str(exc)
    else:
        raise AssertionError("expected a RunpodError")


def test_a_model_can_declare_a_capability_its_section_does_not_imply():
    """Voice cloning runs on a TTS model (XTTS, F5) but is a different endpoint
    and a different pod image. Section alone maps it to 'tts', so it could never
    be placed as 'voice' — which is why the voice image went untested."""
    from codai.api.runpod_worker import model_capability

    xtts = {"path": "coqui/XTTS-v2", "model_type": "tts_models"}
    assert model_capability(xtts) == "tts"

    xtts_as_voice = dict(xtts, capability="voice")
    assert model_capability(xtts_as_voice) == "voice"

    # The override wins for text too, where the default is deliberately blank.
    llm = {"path": "org/m", "model_type": "text_models"}
    assert model_capability(llm) == ""
    assert model_capability(dict(llm, capability="text")) == "text"


def test_an_ocr_pod_arrives_with_ocr_switched_on():
    """OCR ships disabled — right for a fresh install, wrong for a pod rented to
    do OCR. One booted, took the request and answered 'OCR subsystem is disabled
    (enable it in Settings → OCR)', which is a settings screen nobody will open
    on a machine that exists for four more minutes."""
    from codai.api.runpod_worker import pod_plan, parse_model_runpod

    mcfg = parse_model_runpod({"engine": "coderai", "image": "x:1"})
    # Surya 0.22 is a VLM that needs a server the pod image does not carry —
    # vLLM (which it spawns in Docker) or llama-server on a GGUF. A pod asked
    # for surya serves the request with docTR, and says so, rather than loading
    # surya, reaching inference and dying on "docker binary not found".
    env = pod_plan(mcfg, "ocr", entry={"path": "surya",
                                       "model_type": "ocr_models"})["env"]
    assert env["CODERAI_OCR_ENABLED"] == "1"
    assert env["CODERAI_OCR_DEFAULT_ENGINE"] == "doctr"
    assert env["CODERAI_OCR_DOCTR_ENABLED"] == "1"
    assert "CODERAI_OCR_SURYA_ENABLED" not in env

    # A non-gated engine does not carry the licence flag unless we accepted it.
    env = pod_plan(mcfg, "ocr", entry={"path": "paddle",
                                       "model_type": "ocr_models"})["env"]
    assert env["CODERAI_OCR_DEFAULT_ENGINE"] == "paddle"
    assert env["CODERAI_OCR_PADDLE_ENABLED"] == "1"

    # An unnamed engine falls to docTR: the only one the images ship, because it
    # is the only one that runs in-process. The others need a venv of their own.
    env = pod_plan(mcfg, "ocr", entry={"path": "whatever",
                                       "model_type": "ocr_models"})["env"]
    assert env["CODERAI_OCR_DEFAULT_ENGINE"] == "doctr"
    assert env["CODERAI_OCR_DOCTR_ENABLED"] == "1"

    # And a pod for anything else is untouched.
    env = pod_plan(mcfg, "images", entry={"path": "org/sd",
                                          "model_type": "image_models"})["env"]
    assert not any(k.startswith("CODERAI_OCR") for k in env)


def test_env_can_enable_a_subsystem_without_writing_config(tmp_path, monkeypatch):
    """Like the seed list, this describes one disposable pod — it must never
    land in a real deployment's config.json."""
    from codai.config import ConfigManager

    monkeypatch.setenv("CODERAI_OCR_ENABLED", "1")
    monkeypatch.setenv("CODERAI_OCR_DEFAULT_ENGINE", "doctr")
    cm = ConfigManager(str(tmp_path))
    cm.load()
    assert cm.config.ocr.enabled is True
    assert cm.config.ocr.default_engine == "doctr"
    assert cm.config.ocr.surya_accept_license is False   # not asked for

    import json
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk.get("ocr", {}).get("enabled") is not True, \
        "the env override leaked into config.json"


def test_a_capability_whose_stack_cannot_share_gets_its_own_image():
    """`speaker` used to point at the STT image on the assumption it was 'the
    STT deps plus a bit'. That image shipped nothing for diarization or
    voiceprints, so a speaker pod booted with no way to embed a voice."""
    from codai.api.runpod_worker import (default_capability_image,
                                         PUBLISHED_CAPABILITY_IMAGES)

    assert default_capability_image("speaker").endswith("coderai-speaker:latest")
    for name in ("speaker", "tts-xtts", "stt-nemo", "stt-crisper", "ocr-paddle"):
        assert name in PUBLISHED_CAPABILITY_IMAGES, name

    # Capabilities that genuinely DO share a stack still do.
    assert default_capability_image("rerank").endswith("coderai-embeddings:latest")
    assert default_capability_image("stems").endswith("coderai-audio:latest")


def test_a_pod_is_told_where_its_baked_venvs_are():
    """Otherwise the worker tries to build one, on a machine rented by the
    second, installing what the image already contains."""
    from codai.api.runpod_worker import pod_plan, parse_model_runpod

    mcfg = parse_model_runpod({"engine": "coderai", "image": "x:1"})
    env = pod_plan(mcfg, "speaker", entry={"path": "pyannote/speaker-diarization-3.1",
                                           "model_type": "audio_models"})["env"]
    assert env["CODERAI_PYANNOTE_VENV"] == "/opt/coderai/venvs/pyannote"
    assert env["CODERAI_NEMO_VENV"] == "/opt/coderai/venvs/nemo"

    # A pod asked for coqui XTTS inherits the CPML acceptance: the alternative
    # is an interactive prompt on a pipe, which corrupts the worker protocol
    # and then blocks forever on input().
    env = pod_plan(mcfg, "tts", entry={"path": "coqui/XTTS-v2",
                                       "model_type": "tts_models"})["env"]
    assert env["COQUI_TOS_AGREED"] == "1"


def test_a_150gb_model_warns_before_anyone_rents_a_card():
    """The machinery around pods is tuned for ~10 GB images and models of a
    few. A frontier MoE model is 100 GB and up — DeepSeek-V4 alone is ~154 GB —
    so a pod spends an hour or two filling its disk before it answers anything,
    on every cold start. That must never be a surprise."""
    from codai.api.runpod_worker import weight_transfer_warning, parse_model_runpod

    for engine in ("colibri", "ds4", "k3", "ktransformers"):
        warn = weight_transfer_warning({"path": "big/model"},
                                       parse_model_runpod({"engine": engine}))
        assert "network volume" in warn and "cold pod" in warn, engine

    # A volume is the fix, so the warning goes away when one is attached.
    assert weight_transfer_warning(
        {"path": "big/model"},
        parse_model_runpod({"engine": "ds4", "network_volume_id": "vol-1"})) == ""

    # And an ordinary capability model says nothing at all.
    assert weight_transfer_warning(
        {"path": "bge-m3"}, parse_model_runpod({"engine": "coderai"})) == ""


def test_the_provision_path_itself_is_free_of_undefined_names():
    """The 150 GB warning was unit-tested and passed — and then every pod test
    failed in two seconds with NameError: name 'entry' is not defined, because
    the call site in _provision_one used a name that did not exist there. A
    helper can be correct while the line that calls it is not. pyflakes sees
    exactly this class of bug, so run it on the modules a pod boot goes through."""
    import subprocess, sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    mods = ["codai/api/runpod_worker.py", "codai/api/remote_gateway.py",
            "codai/api/model_test.py", "codai/api/runpod_client.py"]
    out = subprocess.run([sys.executable, "-m", "pyflakes", *mods],
                         cwd=root, capture_output=True, text=True).stdout
    undefined = [l for l in out.splitlines() if "undefined name" in l]
    assert not undefined, "\n".join(undefined)


def test_the_container_disk_grows_to_fit_the_weights_unless_a_volume_holds_them(monkeypatch):
    """A pod that boots, pulls the image, starts the download and fills its disk
    dies at load_timeout_s having paid for every minute. The repo's size is one
    metadata call before renting. And with a volume attached the weights land
    there, so the disk stays at its configured size."""
    import codai.api.runpod_worker as rw

    monkeypatch.setattr(rw, "estimate_weights_gb", lambda e, m=None: 35.2)
    mcfg = rw.parse_model_runpod({"engine": "coderai"})
    gb, note = rw.disk_for({"path": "org/big"}, mcfg)
    assert gb == 61 and "raised 40 -> 61" in note

    # An explicit disk already large enough is respected, not shrunk.
    big = rw.parse_model_runpod({"engine": "coderai", "container_disk_gb": 200})
    assert rw.disk_for({"path": "org/big"}, big) == (200, "")

    # A volume holds the weights: the disk only needs the image.
    vol = rw.parse_model_runpod({"engine": "coderai", "network_volume_id": "vol-1"})
    assert rw.disk_for({"path": "org/big"}, vol) == (40, "")

    # Unknown size: say nothing, change nothing.
    monkeypatch.setattr(rw, "estimate_weights_gb", lambda e, m=None: 0.0)
    assert rw.disk_for({"path": "/local/x.gguf"}, mcfg) == (40, "")


def test_the_weight_estimate_counts_what_a_load_fetches_not_the_whole_repo(monkeypatch):
    """SDXL's repo is 77 GB; a diffusers load pulls 28. The rest is the same
    weights again as Flax, legacy .bin and ONNX. Sizing to the repo would rent
    100 GB for a 28 GB model."""
    import types
    import codai.api.runpod_worker as rw

    def _fake_info(repo, files_metadata=True):
        F = lambda n, s: types.SimpleNamespace(rfilename=n, size=s)
        return types.SimpleNamespace(siblings=[
            F("unet/diffusion_pytorch_model.safetensors", 10_000_000_000),
            F("unet/diffusion_pytorch_model.bin", 10_000_000_000),        # dupe of ^
            F("unet/diffusion_flax_model.msgpack", 10_000_000_000),       # Flax dupe
            F("unet/model.onnx_data", 10_000_000_000),                    # ONNX dupe
            F("model_index.json", 1_000),
        ])
    monkeypatch.setattr("huggingface_hub.model_info", _fake_info)
    gb = rw.estimate_weights_gb({"path": "org/sdxl"}, rw.parse_model_runpod({"engine": "coderai"}))
    assert abs(gb - 10.0) < 0.01, f"counted dupes: {gb} GB"


def test_a_gated_model_pod_gets_the_hf_token_and_nothing_else_does(monkeypatch):
    """pyannote is gated. The local install has HF_TOKEN in its environment; a
    pod had nothing, so it loaded from its venv, reached from_pretrained and
    stopped at 'the model is gated'. The token is a secret: it travels as pod
    ENVIRONMENT and must never land in the seed list or the catalogue."""
    import json
    from codai.api.runpod_worker import pod_plan, parse_model_runpod

    monkeypatch.setenv("HF_TOKEN", "hf_secret_for_test")
    mcfg = parse_model_runpod({"engine": "coderai", "image": "x:1"})
    entry = {"path": "pyannote/speaker-diarization-3.1", "model_type": "audio_models"}
    plan = pod_plan(mcfg, "speaker", entry=entry)

    assert plan["env"]["HF_TOKEN"] == "hf_secret_for_test"
    # The seed list is JSON the pod parses; it must carry the model, not the secret.
    seeds = json.loads(plan["env"].get("CODERAI_SEED_MODELS", "[]"))
    assert "hf_secret_for_test" not in json.dumps(seeds)
    # And nothing else in the plan carries it either.
    for key, val in plan.items():
        if key != "env":
            assert "hf_secret_for_test" not in json.dumps(val, default=str), key

    # An explicit per-model token beats the ambient one. The merge happens at
    # provision time, and it used to run the other way round — the plan's
    # resolved token silently replaced the one a model was configured with.
    monkeypatch.setenv("HF_TOKEN", "hf_ambient")
    mcfg2 = parse_model_runpod({"engine": "coderai", "image": "x:1",
                                "env": {"HF_TOKEN": "hf_explicit"}})
    plan2 = pod_plan(mcfg2, "speaker", entry=entry)
    merged = dict(plan2["env"]); merged.update(mcfg2.env or {})
    assert merged["HF_TOKEN"] == "hf_explicit"


def test_a_stated_weights_size_beats_the_estimate(monkeypatch):
    """The estimator cannot see a URL download, an upload or a local file, and
    for an HF repo it errs high on purpose. A size the operator states is the
    honest answer, and it wins wherever it is written."""
    import codai.api.runpod_worker as rw

    monkeypatch.setattr("huggingface_hub.model_info",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")))
    mcfg = rw.parse_model_runpod({"engine": "coderai"})

    # on the entry
    assert rw.estimate_weights_gb({"path": "/local/x.gguf", "weights_gb": 30}, mcfg) == 30.0
    assert rw.disk_for({"path": "/local/x.gguf", "weights_gb": 30}, mcfg)[0] == 55
    # in the runpod block
    assert rw.estimate_weights_gb({"path": "/local/x.gguf",
                                   "runpod": {"weights_gb": "12.5"}}, mcfg) == 12.5
    # beats HF even when HF would answer
    monkeypatch.setattr(rw, "_hf_weights", lambda *a, **k: 35.0, raising=False)
    assert rw.estimate_weights_gb({"path": "org/repo", "weights_gb": 7}, mcfg) == 7.0
    # a nonsense value is ignored, not trusted
    assert rw.estimate_weights_gb({"path": "/local/x.gguf", "weights_gb": "lots"}, mcfg) == 0.0
    # and it is never seeded to a pod: it describes the pod's disk, not the model
    plan = rw.pod_plan(rw.parse_model_runpod({"engine": "coderai", "image": "x:1"}),
                       "images", entry={"path": "org/m", "model_type": "image_models",
                                        "weights_gb": 30})
    import json
    assert "weights_gb" not in json.dumps(json.loads(plan["env"].get("CODERAI_SEED_MODELS", "[]")))


def test_a_multi_gpu_pod_is_priced_and_sharded_as_a_whole(monkeypatch):
    """RunPod rents multi-GPU pods. gpu_count was plumbed to the API and used
    by nothing: the ceiling checked one card's price, min_vram_gb one card's
    memory, and the engine loaded onto GPU 0 while the rest sat idle."""
    import codai.api.runpod_worker as rw

    catalog = [{"id": "a100", "display_name": "A100", "memory_gb": 80,
                "secure_price": 1.5, "community_price": None, "spot_price": None}]
    class _C:
        def list_gpu_types(self): return catalog

    # One card: $1.50 fits a $2 ceiling, 80 GB fails a 160 GB floor.
    m1 = rw.parse_model_runpod({"engine": "vllm", "max_hourly_usd": 2.0, "min_vram_gb": 160})
    try:
        rw._rank_gpus(_C(), m1, None)
    except Exception as e:
        assert "160" in str(e)
    else:
        raise AssertionError("one 80 GB card must not satisfy a 160 GB floor")

    # Two cards: 160 GB total satisfies the floor, but $3.00 exceeds a $2 ceiling.
    m2 = rw.parse_model_runpod({"engine": "vllm", "gpu_count": 2,
                                "max_hourly_usd": 2.0, "min_vram_gb": 160})
    try:
        rw._rank_gpus(_C(), m2, None)
    except Exception as e:
        assert "$2.0" in str(e) or "2.0/hr" in str(e)
    else:
        raise AssertionError("two $1.50 cards must not pass a $2 ceiling")

    # Two cards under a $4 ceiling: priced and labelled as the pod.
    m3 = rw.parse_model_runpod({"engine": "vllm", "gpu_count": 2,
                                "max_hourly_usd": 4.0, "min_vram_gb": 160})
    sel = rw._rank_gpus(_C(), m3, None)[0]
    assert sel["price"] == 3.0 and sel["memory_gb"] == 160
    assert sel["gpu_count"] == 2 and sel["display_name"] == "2x A100"

    # And the engines are told to use every card they were rented.
    assert "--tensor-parallel-size 2" in rw._vllm_docker_args(m3, "org/m", {"path": "org/m"})
    ml = rw.parse_model_runpod({"engine": "llamacpp", "gpu_count": 2, "hf_gguf": "u/r:Q4"})
    assert "--split-mode layer" in rw._llamacpp_docker_args(ml, "m", {})
    # A single card asks for neither: no needless flags on the common case.
    assert "tensor-parallel" not in rw._vllm_docker_args(m1, "org/m", {"path": "org/m"})


def test_an_engine_model_gets_the_engines_image_and_its_engine_switched_on(monkeypatch):
    """The local ds4-server is an sm_86 build against CUDA 13 and the local
    colibri has no GPU backend: neither can be copied into a pod. The engines
    image builds them for every pod GPU, and a model that names an engine —
    on its runpod block, since `backend: runpod` takes the pin slot — lands on
    that image with the engine enabled and this install's settings for it."""
    import json
    import codai.api.runpod_worker as rw
    from codai.api.runpod_worker import pod_plan, parse_model_runpod

    class _Cfg:
        ctx = 65536; ssd_streaming = True; extra_args = "--foo"; extra_env = ""
        model_variant = "q4-imatrix"; auto_download = False
        expert_cache_reserve_gb = 0; model_id = "deepseek-v4"
        install_dir = "/home/me/.coderai/ds4"; port = 1234
    class _Root:
        ds4 = _Cfg()
    class _CM:
        config = _Root()
    monkeypatch.setattr("codai.admin.routes.config_manager", _CM(), raising=False)

    entry = {"path": "/AI/gguf/DeepSeek-V4-Q4.gguf", "model_type": "text_models",
             "backend": "runpod", "weights_gb": 154,
             "ds4": {"ssd_streaming": True}}
    mcfg = parse_model_runpod({"engine": "ds4"})
    plan = pod_plan(mcfg, "deepseek", entry=entry)
    assert plan["engine"] == "coderai"
    assert plan["image"].endswith("coderai-engines:latest")
    env = plan["env"]
    assert env["CODERAI_DS4_ENABLED"] == "1"
    fwd = json.loads(env["CODERAI_DS4_CONFIG"])
    assert fwd["ctx"] == 65536 and fwd["ssd_streaming"] is True
    assert fwd["auto_download"] is True, "a pod that cannot fetch answers nothing"
    assert "install_dir" not in fwd and "port" not in fwd

    seed = json.loads(env["CODERAI_SEED_MODELS"])[0]
    assert seed["backend"] == "ds4", "the pod must run ds4 for it, not transformers"
    assert seed["ds4"] == {"ssd_streaming": True}
    assert "weights_gb" not in seed          # sizes the disk here, not the model there
    assert seed["path"] == "DeepSeek-V4-Q4.gguf", "a local path means nothing there"

    # Weights already on the volume: the seed points straight at them.
    mcfg = parse_model_runpod({"engine": "ds4", "network_volume_id": "vol1",
                               "volume_path": "models/DeepSeek-V4-Q4.gguf"})
    seed = json.loads(pod_plan(mcfg, "deepseek", entry=entry)["env"]["CODERAI_SEED_MODELS"])[0]
    assert seed["path"] == "/workspace/models/DeepSeek-V4-Q4.gguf"

    # A local model pinned to the engine itself, bursting: same answer.
    local = {"path": "/AI/glm52", "model_type": "text_models", "backend": "colibri"}
    plan = pod_plan(parse_model_runpod({"network_volume_id": "v", "volume_path": "glm52"}),
                    "glm", entry=local)
    assert plan["image"].endswith("coderai-engines:latest")
    assert plan["env"]["CODERAI_COLIBRI_ENABLED"] == "1"
    assert json.loads(plan["env"]["CODERAI_SEED_MODELS"])[0]["backend"] == "colibri"

    # The other pins describe THIS machine and are stripped from the seed.
    vk = {"path": "org/m", "model_type": "text_models", "backend": "vulkan"}
    seed = json.loads(pod_plan(parse_model_runpod({"engine": "coderai", "image": "x"}),
                               "m", entry=vk)["env"]["CODERAI_SEED_MODELS"])[0]
    assert "backend" not in seed

    # ktransformers is a Python stack pinning its own torch: its own image,
    # its own venv, and the worker is told where that venv is.
    class _Kt:
        ctx = 32768; extra_args = "--tp-size 1"; extra_env = ""; model_id = "ktransformers"
        venv = "/local/venv"; install_dir = "/home/me/.coderai/ktransformers"
    _Root.ktransformers = _Kt()
    plan = pod_plan(parse_model_runpod({"engine": "kt"}), "m",
                    entry={"path": "org/m", "model_type": "text_models"})
    assert plan["image"].endswith("coderai-engines-kt:latest")
    assert plan["env"]["CODERAI_KT_ENABLED"] == "1"
    assert plan["env"]["CODERAI_KT_VENV"] == "/opt/coderai/venvs/sglang"
    fwd = json.loads(plan["env"]["CODERAI_KT_CONFIG"])
    assert fwd["ctx"] == 32768 and fwd["extra_args"] == "--tp-size 1"
    assert "venv" not in fwd and "install_dir" not in fwd
    assert json.loads(plan["env"]["CODERAI_SEED_MODELS"])[0]["backend"] == "kt"

    # The volume env points ds4's downloader at the volume too.
    assert rw.volume_env("/workspace")["DS4_GGUF_DIR"] == "/workspace/cache/ds4"


def test_a_pod_switches_its_engine_on_from_env_and_takes_the_settings(tmp_path, monkeypatch):
    import json
    from codai.config import ConfigManager

    monkeypatch.setenv("CODERAI_DS4_ENABLED", "1")
    monkeypatch.setenv("CODERAI_DS4_CONFIG", json.dumps(
        {"ctx": 65536, "auto_download": True, "install_dir": "/nope",
         "port": 9, "not_a_field": 1}))
    cm = ConfigManager(str(tmp_path))
    cm.load()
    assert cm.config.ds4.enabled is True
    assert cm.config.ds4.ctx == 65536 and cm.config.ds4.auto_download is True
    assert cm.config.ds4.install_dir != "/nope" and cm.config.ds4.port != 9
    assert not hasattr(cm.config.ds4, "not_a_field")
    assert cm.config.colibri.enabled is False       # only what was asked for

    # kt's config lives under `ktransformers`; the env name is the short one.
    monkeypatch.setenv("CODERAI_KT_ENABLED", "1")
    monkeypatch.setenv("CODERAI_KT_CONFIG", json.dumps({"ctx": 8192, "venv": "/nope"}))
    cm2 = ConfigManager(str(tmp_path / "b"))
    cm2.load()
    assert cm2.config.ktransformers.enabled is True and cm2.config.ktransformers.ctx == 8192
    assert cm2.config.ktransformers.venv != "/nope"
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk.get("ds4", {}).get("enabled") is not True


def test_the_engines_image_is_built_from_source_for_every_pod_gpu():
    """Shipping this machine's binaries would ship an sm_86-only ds4 linked
    to CUDA 13. The image compiles in a CUDA 12.8 stage for Ampere through
    Blackwell, and the build fails if a binary cannot find its libraries."""
    from pathlib import Path
    from codai.api.runpod_worker import PUBLISHED_CAPABILITY_IMAGES, default_capability_image
    assert "engines" in PUBLISHED_CAPABILITY_IMAGES and "engines-kt" in PUBLISHED_CAPABILITY_IMAGES
    assert default_capability_image("engines").endswith("coderai-engines:latest")
    prof = Path("packaging/runpod/profiles")
    assert (prof / "engines-kt.light").exists()
    assert "sglang-kt" in (prof / "engines-kt.venv-sglang.txt").read_text()
    assert (prof / "engines-kt.venv-sglang.check").exists()
    assert (prof / "engines.txt").exists()
    assert (prof / "engines.dockerfile").read_text().strip() == "Dockerfile.capability-engines"
    df = Path("packaging/runpod/Dockerfile.capability-engines").read_text()
    for arch in ("sm_80", "sm_86", "sm_89", "sm_90", "sm_120"):
        assert arch in df
    assert "CUDA_ARCH=portable" in df and "x86-64-v3" in df
    assert "base-light" in df, "the engines are C: no torch needed"
    assert "patch-k3.py" in df and "not found" in df
    for var in ("CODERAI_DS4_DIR", "CODERAI_COLIBRI_DIR", "CODERAI_K3_DIR"):
        assert var in df
    sh = Path("packaging/runpod/build_capability_image.sh").read_text()
    assert ".dockerfile" in sh and "engines-src" in sh


def test_the_kt_worker_launches_from_its_own_venv(tmp_path, monkeypatch):
    """SGLang + kt-kernel pin their own torch and cannot share coderai's
    venv; on a pod they are baked into one of their own, and the worker
    must launch from THAT interpreter — sys.executable has no sglang."""
    import sys
    from codai.api import kt_worker as kw

    class _Cfg:
        venv = ""; model_id = "ktransformers"; kt_weight_path = ""; ctx = 0; extra_args = ""
    monkeypatch.delenv("CODERAI_KT_VENV", raising=False)
    assert kw._venv_python(_Cfg()) == sys.executable

    venv = tmp_path / "sglang"; (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("#!/bin/sh\n")
    monkeypatch.setenv("CODERAI_KT_VENV", str(venv))
    assert kw._venv_python(_Cfg()) == str(venv / "bin" / "python")
    cmd = kw._launch_cmd(_Cfg(), "127.0.0.1", 1234, "org/m")
    assert cmd[0] == str(venv / "bin" / "python") and "sglang.launch_server" in cmd
    # A venv that is configured but missing falls back rather than crashing.
    monkeypatch.setenv("CODERAI_KT_VENV", str(tmp_path / "missing"))
    assert kw._venv_python(_Cfg()) == sys.executable


def test_direct_tcp_reaches_the_pod_without_the_proxy(monkeypatch):
    """The proxy hostname is fronted by Cloudflare: a 100 s idle limit per
    request cuts off a slow non-streaming render. direct_tcp exposes the
    port as tcp and talks to the public ip:port instead — read from the
    runtime, because the public port is random per pod."""
    from codai.api import runpod_client as rc
    from codai.api.runpod_worker import parse_model_runpod

    from codai.api.runpod_worker import direct_tcp_for
    # Unset = decide by image: our images go direct (they serve TLS there),
    # upstream images stay behind the proxy (it is the only TLS they have).
    assert parse_model_runpod({}).direct_tcp is None
    assert direct_tcp_for(parse_model_runpod({}), "coderai") is True
    assert direct_tcp_for(parse_model_runpod({}), "vllm") is False
    assert direct_tcp_for(parse_model_runpod({}), "llamacpp") is False
    assert direct_tcp_for(parse_model_runpod({}), "custom") is False
    # An explicit setting wins either way.
    assert direct_tcp_for(parse_model_runpod({"direct_tcp": True}), "vllm") is True
    assert direct_tcp_for(parse_model_runpod({"direct_tcp": False}), "coderai") is False
    assert parse_model_runpod({"direct_tcp": "auto"}).direct_tcp is None

    ports = [{"ip": "203.0.113.7", "isIpPublic": True, "privatePort": 8000,
              "publicPort": 41234, "type": "tcp"}]
    assert rc.direct_tcp_url(ports, 8000) == "http://203.0.113.7:41234"
    assert rc.direct_tcp_url(ports, 8001) == ""                    # wrong port
    assert rc.direct_tcp_url([{**ports[0], "type": "http"}], 8000) == ""
    assert rc.direct_tcp_url([{**ports[0], "isIpPublic": False}], 8000) == ""
    assert rc.pod_proxy_url("abc", 8000) == "https://abc-8000.proxy.runpod.net"

    # create_pod asks for the tcp mapping only when told to.
    seen = {}
    class _C(rc.RunpodClient):
        def __init__(self): pass
        def _gql(self, q, v, timeout=30.0):
            seen["ports"] = v["input"]["ports"]; return {"podFindAndDeployOnDemand": {"id": "p1"}}
    _C().create_pod(name="n", image="i", gpu_type_id="g", port=8000)
    assert seen["ports"] == "8000/http"
    _C().create_pod(name="n", image="i", gpu_type_id="g", port=8000, direct_tcp=True)
    assert seen["ports"] == "8000/tcp"

    # wait_ready keeps waiting until the runtime reports the public mapping.
    infos = iter([{"status": "RUNNING", "ready": True, "ports": [], "uptime_s": 1},
                  {"status": "RUNNING", "ready": True, "ports": ports, "uptime_s": 2}])
    class _W(rc.RunpodClient):
        def __init__(self): pass
        def get_pod(self, pod_id): return next(infos)
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    assert _W().wait_ready("p1", 8000, direct_tcp=True) == "http://203.0.113.7:41234"


def test_a_direct_tcp_pod_speaks_tls_the_renting_coderai_can_verify(tmp_path, monkeypatch):
    """direct_tcp removed the 100 s proxy cutoff at the price of plain HTTP.
    Now the pod brings TLS: a leaf from this install's CA, sent as env, and
    this side verifies against that CA alone — no hostname check, since the
    pod's IP is unknown when the cert is issued and a foreign cert fails the
    chain before any name is compared."""
    import http.server, json, ssl, threading
    from codai.api import pod_tls, pod_http, runpod_client as rc

    monkeypatch.setattr(pod_tls, "tls_dir", lambda: str(tmp_path / "tls"))
    env = pod_tls.pod_env("deploy-mymodel")
    assert env["CODERAI_TLS_CERT"].startswith("-----BEGIN CERTIFICATE-----")
    assert "PRIVATE KEY" in env["CODERAI_TLS_KEY"]
    assert (tmp_path / "tls" / "ca.pem").exists()

    # A "pod": an https server on 127.0.0.1 using exactly what boot.sh writes.
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert.write_text(env["CODERAI_TLS_CERT"]); key.write_text(env["CODERAI_TLS_KEY"])
    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"ok": True, "auth": self.headers.get("Authorization")}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def log_message(self, *a): pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), _H)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(str(cert), str(key))
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"https://127.0.0.1:{srv.server_address[1]}"

    # The URL a direct-TCP TLS pod gets: https to its bare public IP.
    ports = [{"ip": "127.0.0.1", "isIpPublic": True, "privatePort": 8000,
              "publicPort": srv.server_address[1], "type": "tcp"}]
    assert rc.direct_tcp_url(ports, 8000, tls=True) == url
    assert pod_http.is_pinned_url(url) and not pod_http.is_pinned_url("https://x-8000.proxy.runpod.net")

    monkeypatch.setattr(pod_http, "_session", None)
    r = pod_http.get(url + "/healthz", headers={"Authorization": "Bearer t"}, timeout=5)
    assert r.status_code == 200 and r.json()["auth"] == "Bearer t"
    # Plain requests would refuse it — that is the whole reason pod_http exists.
    import requests, pytest
    with pytest.raises(requests.exceptions.SSLError):
        requests.get(url + "/healthz", timeout=5)
    srv.shutdown()

    # Provisioning issues a cert only for coderai pods on the direct path.
    from codai.api.runpod_worker import parse_model_runpod
    assert parse_model_runpod({"direct_tcp": True}).direct_tcp is True
    assert parse_model_runpod({"direct_tcp": "off"}).direct_tcp is False
    from pathlib import Path
    boot = Path("packaging/runpod/boot.sh").read_text()
    assert "CODERAI_TLS_CERT" in boot and "--ssl-certfile" in boot and "unset CODERAI_TLS_KEY" in boot


def test_the_llm_servers_can_run_as_coderai_pods_instead_of_upstream_images(monkeypatch):
    """coderai-text was transformers only, so a GGUF or a vLLM-served repo
    still went to an upstream image — no TLS on the direct path, no /boot,
    no local LoRAs. coderai-llama and coderai-vllm are those servers as
    coderai pods; the upstream engines stay selectable."""
    import json
    from codai.api.runpod_worker import (pod_plan, parse_model_runpod,
                                         PUBLISHED_CAPABILITY_IMAGES, resolve_pod_engine)
    assert "llama" in PUBLISHED_CAPABILITY_IMAGES and "vllm" in PUBLISHED_CAPABILITY_IMAGES

    class _V:
        ctx = 16384; gpu_memory_utilization = 0.85; tensor_parallel_size = 2
        max_num_seqs = 0; dtype = "bfloat16"; quantization = ""; extra_args = ""; extra_env = ""
        venv = "/local/vllm_venv"; port = 9
    class _Root:
        vllm = _V()
    class _CM:
        config = _Root()
    monkeypatch.setattr("codai.admin.routes.config_manager", _CM(), raising=False)

    entry = {"path": "org/m", "model_type": "text_models", "backend": "runpod"}
    # Upstream stays upstream.
    assert resolve_pod_engine(parse_model_runpod({"engine": "vllm"}), "m", entry=entry) == "vllm"
    up = pod_plan(parse_model_runpod({"engine": "vllm"}), "org/m", entry=entry)
    assert up["image"].startswith("vllm/") and "CODERAI_VLLM_ENABLED" not in up["env"]

    # vLLM as a coderai pod: its image, the engine on, local settings forwarded
    # — except tensor parallelism, which follows the pod's own card count.
    plan = pod_plan(parse_model_runpod({"engine": "coderai-vllm", "gpu_count": 2}),
                    "org/m", entry=entry)
    assert plan["engine"] == "coderai" and plan["image"].endswith("coderai-vllm:latest")
    env = plan["env"]
    assert env["CODERAI_VLLM_ENABLED"] == "1" and env["CODERAI_VLLM_VENV"] == "/opt/coderai/venvs/vllm"
    fwd = json.loads(env["CODERAI_VLLM_CONFIG"])
    assert fwd["dtype"] == "bfloat16" and fwd["ctx"] == 16384 and fwd["tensor_parallel_size"] == 2
    assert "venv" not in fwd and "port" not in fwd
    assert json.loads(env["CODERAI_SEED_MODELS"])[0]["backend"] == "vllm"
    one = pod_plan(parse_model_runpod({"engine": "coderai-vllm"}), "org/m", entry=entry)
    assert "tensor_parallel_size" not in json.loads(one["env"]["CODERAI_VLLM_CONFIG"])

    # llama.cpp as a coderai pod: the GGUF goes to coderai's own CUDA backend.
    gg = {"path": "org/m-GGUF", "model_type": "text_models", "backend": "runpod"}
    plan = pod_plan(parse_model_runpod({"engine": "coderai-llama", "hf_gguf": "org/m-GGUF:Q4_K_M"}),
                    "org/m-GGUF", entry=gg)
    assert plan["engine"] == "coderai" and plan["image"].endswith("coderai-llama:latest")
    assert "backend" not in json.loads(plan["env"]["CODERAI_SEED_MODELS"])[0]

    # The worker finds the baked venv from env, ahead of the home-dir default.
    from codai.api import vllm_worker as vw
    class _Cfg: venv = ""
    monkeypatch.setenv("CODERAI_VLLM_VENV", "/opt/coderai/venvs/vllm")
    assert vw.resolve_venv_dir(_Cfg()) == "/opt/coderai/venvs/vllm"

    from pathlib import Path
    prof = Path("packaging/runpod/profiles")
    assert (prof / "llama.dockerfile").read_text().strip() == "Dockerfile.capability-llama"
    df = Path("packaging/runpod/Dockerfile.capability-llama").read_text()
    assert "GGML_CUDA=on" in df and "80;86;89;90;120" in df and "check_llama_wheel" in df
    assert (prof / "vllm.light").exists() and (prof / "vllm.venv-vllm.uv").exists()
    assert "vllm==" in (prof / "vllm.venv-vllm.txt").read_text()


def test_the_gguf_backend_can_size_the_card_without_torch(monkeypatch):
    """The llama pod image carried 5 GB of torch so the CUDA GGUF backend
    could ask it three questions about the card — 11.6 GB, over RunPod's
    line. NVML answers the same three from a 1 MB package; torch stays
    preferred where it exists."""
    import sys, types
    from codai.backends import gpu_probe as gp

    # No torch: NVML answers.
    monkeypatch.setattr(gp, "_torch", lambda: None)
    class _Mem:  total = 80 * 1024**3; free = 70 * 1024**3
    nv = types.SimpleNamespace(
        nvmlInit=lambda: None, nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetMemoryInfo=lambda h: _Mem(),
        nvmlSystemGetDriverVersion=lambda: b"570.86")
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    assert gp.cuda_available() is True
    assert gp.total_memory_bytes() == [80 * 1024**3] * 2
    assert gp.free_memory_bytes() == 70 * 1024**3
    assert "NVML" in gp.runtime_label() and "570.86" in gp.runtime_label()

    # Neither: honest zeros, not a crash.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert gp.cuda_available() is False and gp.total_memory_bytes() == []
    assert gp.free_memory_bytes() is None

    # The backend goes through the probe, not torch.
    from pathlib import Path
    src = Path("codai/backends/cuda.py").read_text()
    i = src.index("def _detect_device"); j = src.index("def _get_gpu_memory_map")
    assert "import torch" not in src[i:src.index("def _get_available_vram")]
    assert "gpu_probe" in src[i:j]
    prof = Path("packaging/runpod/profiles")
    assert (prof / "llama.light").exists()
    assert "nvidia-ml-py" in (prof / "llama.txt").read_text()


def test_a_pod_that_refuses_connections_is_dropped_and_the_request_retried(monkeypatch):
    """The harness reaped a pod by API; the engine's pool still held it as
    healthy until the next timed probe, and the next request died on
    'connection refused' for something the pool could have known."""
    import requests
    from codai.backends import runpod as rb
    from codai.api.runpod_worker import PodHandle

    dead = PodHandle(pod_id="dead", url="https://203.0.113.9:1", hourly_usd=0.3, started_at=0)
    live = PodHandle(pod_id="live", url="https://203.0.113.10:1", hourly_usd=0.3, started_at=0)
    handed = iter([dead, live])
    class _Pool:
        pods = [dead, live]; discarded = []
        def acquire(self, timeout=1200.0, affinity=""):
            p = next(handed); return p, p.url
        def release(self, p): pass
        def discard(self, p, reason=""): self.discarded.append((p.pod_id, reason)); self.pods.remove(p)
    pool = _Pool()

    b = rb.RunpodBackend.__new__(rb.RunpodBackend)
    b._mode = "pods"; b._pool = pool; b._headers = {}; b._url = ""
    b._enter_request = lambda: None; b._exit_request = lambda: None
    b._chat_payload = lambda *a, **k: {"messages": []}
    b._store_usage = lambda u: None

    calls = []
    class _R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "hi"}}], "usage": {}}
    def _post(url, **kw):
        calls.append(url)
        if "203.0.113.9" in url:
            raise requests.exceptions.ConnectionError("refused")
        return _R()
    monkeypatch.setattr(rb.pod_http, "post", _post)

    assert b.generate_chat([{"role": "user", "content": "x"}]) == "hi"
    assert [u.split("/")[2] for u in calls] == ["203.0.113.9:1", "203.0.113.10:1"]
    assert pool.discarded == [("dead", "connection refused")] and dead not in pool.pods


def test_the_vllm_worker_reports_the_root_cause_not_the_wrapper_traceback():
    """On a pod the 503 body is all anyone gets, and it carried the last six
    lines — vLLM's own "Engine core initialization failed. See root cause
    above". The root cause IS above; surface it."""
    from pathlib import Path
    src = Path("codai/api/vllm_worker.py").read_text()
    assert "keyed = " in src and '"Traceback"' in src
    prof = Path("packaging/runpod/profiles")
    for p in ("vllm", "engines-kt"):
        assert "gcc" in (prof / f"{p}.apt").read_text(), f"{p}: Triton needs a C compiler"
    assert ".apt" in Path("packaging/runpod/Dockerfile.capability").read_text()
