"""The orchestrator drives; the pod answers. Simulated end to end.

An earlier version of this had pods calling in, on the belief that a pod could
not be addressed. It can -- coderai already proxies inference to pods -- and
``wait_ready(..., direct_tcp=True)`` returns an ip:port that bypasses the
RunPod proxy, so a 160 MB adapter meets neither the Cloudflare idle limit nor
its body cap. Driving is both smaller and free of the one setting that could
strand a fleet: a wrong callback URL meant pods that booted, could not phone
home, and billed until reaped.

What is tested here is the part a unit test of either side would miss: the
sequence. Dataset once, work pushed, status polled, adapter pulled, average
pushed back, and every failure ending with the pods released.
"""

import json
import os
import threading
from http.server import ThreadingHTTPServer

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file

from codai.api import loras
from codai.api import text_lora_fleet as flt
from codai.api import text_lora_rounds as tlr
from codai.api import text_lora_trainer as tlt


class FakePod:
    """A pod's HTTP API, with the training replaced by a value we choose.

    The real runner is exercised by its own image probe; what matters here is
    that the orchestrator speaks the protocol correctly, so this answers like
    a pod and records what it was asked.
    """

    def __init__(self, value=1.0, fail=False):
        self.value = value
        self.fail = fail
        self.dataset = b""
        self.started = []
        self.start_adapters = []
        self.state = {"state": "idle", "round": -1, "step": 0,
                      "examples_trained": 0, "error": "", "adapter": ""}
        self.secret = "s3cr3t"
        self._srv = None
        self.port = 0

    def start(self):
        pod = self

        from http.server import BaseHTTPRequestHandler

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _ok(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _auth(self):
                sent = (self.headers.get("Authorization") or "")[7:].strip()
                if sent != pod.secret:
                    self._ok({"error": "bad credential"}, 401)
                    return False
                return True

            def do_GET(self):
                if not self._auth():
                    return
                if self.path == "/status":
                    return self._ok(pod.state)
                if self.path == "/adapter":
                    if pod.state["state"] != "done":
                        return self._ok({"error": "not done"}, 404)
                    import io as _io
                    buf = pod._adapter_bytes()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(buf)))
                    self.end_headers()
                    self.wfile.write(buf)
                    return
                self._ok({"error": "nope"}, 404)

            def do_PUT(self):
                if not self._auth():
                    return
                n = int(self.headers.get("Content-Length") or 0)
                pod.start_adapters.append(self.rfile.read(n))
                self._ok({"ok": True})

            def do_POST(self):
                if not self._auth():
                    return
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n)
                if self.path == "/dataset":
                    pod.dataset = raw
                    return self._ok({"ok": True, "bytes": len(raw)})
                if self.path == "/round":
                    if pod.state["state"] == "training":
                        return self._ok({"error": "busy"}, 409)
                    payload = json.loads(raw)
                    pod.started.append(payload)
                    if pod.fail:
                        pod.state.update(state="error", error="CUDA OOM",
                                         round=payload["round"])
                    else:
                        pod.state.update(state="done", round=payload["round"],
                                         examples_trained=len(payload["indices"]),
                                         adapter="x")
                    return self._ok({"ok": True}, 202)
                self._ok({"error": "nope"}, 404)

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self._srv.server_address[1]
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.port}"

    def _adapter_bytes(self):
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "a.safetensors")
        save_file({"lora_A.weight": torch.full((2, 2), float(self.value))}, p)
        with open(p, "rb") as fh:
            return fh.read()

    def stop(self):
        if self._srv:
            self._srv.shutdown()


class FakeRunpod:
    """create/wait_ready/terminate, with the pods we started."""

    def __init__(self, pods):
        self.pods = pods
        self.created = []
        self.terminated = []

    def create_pod(self, **kw):
        self.created.append(kw)
        return f"pod{len(self.created)}"

    def wait_ready(self, pod_id, port, ready_timeout=900.0, poll_every=5.0,
                   direct_tcp=False, tls=False):
        assert direct_tcp, "a 160 MB pull must not go through the RunPod proxy"
        idx = int(pod_id.replace("pod", "")) - 1
        return self.pods[idx].url

    def terminate_pod(self, pod_id, attempts=4):
        self.terminated.append(pod_id)


@pytest.fixture
def two_pods(monkeypatch, tmp_path):
    monkeypatch.setattr(loras, "_loras_dir", lambda: str(tmp_path), raising=False)
    pods = [FakePod(value=0.0), FakePod(value=4.0)]
    for p in pods:
        p.url = p.start()
    client = FakeRunpod(pods)
    spec = flt.TrainerPodSpec(image="img", gpu_types=["NVIDIA A40"],
                              hourly_usd=0.45, max_pod_hourly_usd=0.90)
    fleet = flt.TrainerFleet(client, "job-p", spec)
    for pod in pods:
        pass
    yield {"pods": pods, "client": client, "fleet": fleet, "tmp": tmp_path,
           "spec": spec}
    for p in pods:
        p.stop()


def _wire(fleet, pods):
    """Rent, resolve addresses, and let the fakes learn the minted secrets.

    The fleet mints a bearer per pod, so the test takes ITS value rather than
    imposing one -- which is also the only way to notice if it stopped minting.
    """
    fleet.provision([0, 1])
    for shard in (0, 1):
        assert fleet.pods[shard].secret, "the fleet minted no credential"
        pods[shard].secret = fleet.pods[shard].secret
        fleet.address(shard)
    return fleet


# ------------------------------------------------------------- the direction --

def test_the_pod_is_rented_with_a_direct_tcp_mapping(two_pods):
    """Through the proxy a 160 MB body meets Cloudflare's cap and a 100 s idle
    limit; direct TCP meets neither. wait_ready asserts it."""
    _wire(two_pods["fleet"], two_pods["pods"])
    assert two_pods["client"].created[0]["direct_tcp"] is True


def test_nothing_tells_the_pod_our_address(two_pods):
    """The whole point of driving: there is no callback URL to get wrong, so no
    fleet can boot, fail to phone home, and bill until reaped."""
    _wire(two_pods["fleet"], two_pods["pods"])
    env = two_pods["client"].created[0]["env"]
    assert not any("URL" in k or "http" in str(v).lower()
                   for k, v in env.items()), env
    assert "CODERAI_TRAIN_SECRET" in env


def test_the_pod_runs_its_baked_runner_with_no_start_command(two_pods):
    """Baked, so the image tag pins it and a pod needs nothing from outside."""
    _wire(two_pods["fleet"], two_pods["pods"])
    assert two_pods["client"].created[0].get("start_cmd") in (None, [])


# ------------------------------------------------------------- the sequence --

def test_the_dataset_is_pushed_once_per_pod(two_pods):
    f = _wire(two_pods["fleet"], two_pods["pods"])
    ds = two_pods["tmp"] / "d.jsonl"
    ds.write_bytes(b'{"messages": [], "answer": "a"}\\n')
    for shard in (0, 1):
        flt.push_dataset(f.pods[shard], str(ds))
    assert all(p.dataset.startswith(b'{"messages"') for p in two_pods["pods"])


def test_a_round_is_pushed_and_the_adapter_pulled(two_pods):
    f = _wire(two_pods["fleet"], two_pods["pods"])
    pod = f.pods[0]
    flt.start_round(pod, {"round": 0, "indices": [1, 2, 3], "settings": {},
                          "base": "b"})
    st = flt.poll_status(pod)
    assert st["state"] == "done" and st["examples_trained"] == 3
    dest = str(two_pods["tmp"] / "pulled.safetensors")
    flt.fetch_adapter(pod, dest)
    assert load_file(dest)["lora_A.weight"].flatten()[0].item() == 0.0


def test_a_partial_pull_is_never_left_where_an_average_would_read_it(two_pods):
    """Renamed at the end, so a transfer cut halfway cannot be averaged as
    though it were an adapter."""
    import inspect
    src = inspect.getsource(flt.fetch_adapter)
    assert ".part" in src and "os.replace" in src


def test_a_busy_pod_refuses_a_second_round(two_pods):
    f = _wire(two_pods["fleet"], two_pods["pods"])
    pod = f.pods[0]
    two_pods["pods"][0].state["state"] = "training"
    with pytest.raises(flt.PodBusy):
        flt.start_round(pod, {"round": 1, "indices": [1], "settings": {}})


def test_an_unreachable_pod_is_reported_not_raised(two_pods):
    """One failed poll on a busy machine is not a dead pod; the silence window
    decides that."""
    f = _wire(two_pods["fleet"], two_pods["pods"])
    two_pods["pods"][0].stop()
    st = flt.poll_status(f.pods[0], timeout=2)
    assert st["state"] == "unreachable"


def test_the_average_is_pushed_back_for_the_next_round(two_pods):
    f = _wire(two_pods["fleet"], two_pods["pods"])
    avg = str(two_pods["tmp"] / "avg.safetensors")
    save_file({"lora_A.weight": torch.full((2, 2), 2.0)}, avg)
    flt.push_start_adapter(f.pods[0], avg)
    assert len(two_pods["pods"][0].start_adapters) == 1
    assert len(two_pods["pods"][0].start_adapters[0]) > 0


def test_a_failed_round_is_visible_in_status(two_pods):
    pods = [FakePod(fail=True)]
    pods[0].url = pods[0].start()
    try:
        client = FakeRunpod(pods)
        fleet = flt.TrainerFleet(client, "job-f",
                                 flt.TrainerPodSpec(image="i", gpu_types=["g"]))
        fleet.provision([0])
        pods[0].secret = fleet.pods[0].secret
        fleet.address(0)
        flt.start_round(fleet.pods[0], {"round": 0, "indices": [1],
                                        "settings": {}})
        st = flt.poll_status(fleet.pods[0])
        assert st["state"] == "error" and "OOM" in st["error"]
    finally:
        pods[0].stop()


def test_credentials_are_required_by_the_pod(two_pods):
    f = _wire(two_pods["fleet"], two_pods["pods"])
    f.pods[0].secret = "wrong"
    st = flt.poll_status(f.pods[0])
    assert st["state"] == "unreachable" and st.get("http") == 401


def test_a_pod_that_never_publishes_a_port_is_released_not_awaited(two_pods):
    """It is billing and it will never be useful."""
    class Never(FakeRunpod):
        def wait_ready(self, *a, **kw):
            raise RuntimeError("Pod cannot be reached and more waiting will not help")

    client = Never(two_pods["pods"])
    fleet = flt.TrainerFleet(client, "job-n",
                             flt.TrainerPodSpec(image="i", gpu_types=["g"]))
    fleet.provision([0])
    with pytest.raises(RuntimeError):
        fleet.address(0)
    assert fleet.release(0, "never published a port") is True
    assert client.terminated == ["pod1"]


# ------------------------------------------------------------- the self-test --
#
# "Can coderai make a test?" -- yes, and it is the cheapest thing in the whole
# path: one pod, one round, a handful of examples. It proves rent, boot,
# dataset push, tokenizer, encode, train, pull, average and release for a few
# minutes of GPU instead of discovering a broken stage hours into a real run.

def test_a_test_run_is_one_pod_one_round_and_a_few_examples():
    """Not a different code path -- the SAME one with everything turned down,
    or the test would prove something other than what a real run does."""
    import inspect
    src = inspect.getsource(loras)
    i = src.index("if req.test_run:")
    block = src[i:i + 900]
    assert "pods=1" in block and "rounds=1" in block
    assert "encoded[:n]" in block, "it must shrink the work, not the pipeline"


def test_a_test_run_shortens_the_leash():
    """A smoke test that can bill for a day is not a smoke test."""
    import inspect
    src = inspect.getsource(loras)
    i = src.index("if req.test_run:")
    assert "pod_max_lifetime_s" in src[i:i + 900]


def test_the_timeline_names_the_stage_that_failed():
    """"It failed" after four minutes and $0.06 is not a diagnosis. Eight
    stages look identical from outside unless each is timed."""
    st = flt.Stages()
    st.done("rent", pods=1)
    st.failed("reachable", "Pod cannot be reached")
    d = st.as_dict()
    assert d["failed_at"] == "reachable"
    assert [r["stage"] for r in d["stages"]] == ["rent", "reachable"]
    assert d["stages"][0]["ok"] is True and d["stages"][1]["ok"] is False


def test_the_timeline_names_the_slowest_stage():
    """On a first run the answer to "why twenty minutes" is one stage, and
    rarely the one anyone guessed -- usually the image pull or the weights."""
    import time as _t
    st = flt.Stages()
    st.done("rent")
    _t.sleep(0.05)
    st.done("reachable")
    assert st.as_dict()["slowest"] == "reachable"


def test_a_clean_timeline_has_no_failure():
    st = flt.Stages()
    st.done("rent")
    st.done("adapter")
    assert st.as_dict()["failed_at"] is None


def test_a_job_starts_below_its_minimum_pods_and_grows():
    """REVERSED, because the reason for the refusal is gone.

    min_pods was a gate: with `shard = index % max_pods` fixed at the start,
    running a sixteen-shard job on two pods left fourteen sixteenths of the
    data untouched, so refusing was the only honest answer and releasing what
    it had rented was the right follow-through.

    Round-first slicing changed the premise -- each round's content depends on
    (seed, round, rounds) alone and is split across whatever cohort is present
    -- so a run on one pod trains the whole dataset, more slowly and with less
    averaging. Enforcing the gate after that was destroying booted, paid-for
    pods to prevent something that can no longer happen, and it was why a
    large fleet could never be assembled: the early arrivals were always
    thrown away while the rest were hunted.

    min_pods is now the target the fleet grows toward between rounds.
    """
    import inspect
    src = inspect.getsource(loras._text_train_distributed)
    i = src.index("floor = max(0, int(req.min_pods or 0))")
    block = src[i:i + 1800]
    assert "len(ready) < floor" in block
    assert "starting on " in block and "grows toward" in block
    # No raise in that branch any more -- the quoted old message in the
    # comment does not count.
    raises = [l for l in block.splitlines() if "raise " in l]
    assert raises == [], raises
    # Zero pods is still fatal, and the release is still the finally.
    assert "no pod became reachable" in src
    assert "release_all" in src[src.index("finally:"):]
