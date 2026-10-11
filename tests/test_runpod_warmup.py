""""Ready" has to mean the model is loaded, not that the port is open.

A vLLM pod answers /v1/models the moment its process starts — minutes before it
can serve anything, because the checkpoint still has to be downloaded (every
pod re-downloads it when there is no network volume) and the KV cache built on
first use. So the pool called the pod ready, every client read that as usable,
and the entire model load landed inside somebody's first request.

Measured on the Digesta orchestrator: one OCR page took 807s end to end, of which
~400s was the pull and ~400s the weights, while the client timed out at 180-300s
and concluded the serving was broken. It was not: nobody had ever waited long
enough. A warm-up moves that wait to where it is visible — a pod that is not
ready yet — and the pod bills through the load either way.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                         # noqa: E402


class _Reply:
    def __init__(self, status=200, text="{}"):
        self.status_code, self.text = status, text


def _pool(monkeypatch, reply=None, boom=None, warmup=True, budget=900):
    import threading
    pool = rw.RunpodPodPool.__new__(rw.RunpodPodPool)
    pool._cv = threading.Condition()
    pool.model_key = "datalab-to/surya-ocr-2"
    pool.served = "datalab-to/surya-ocr-2"
    pool.api_key = "tok"
    cfg = rw.RunpodModelConfig()
    cfg.warmup_on_boot = warmup
    cfg.warmup_timeout_s = budget
    pool.mcfg = cfg
    seen = {}

    def _post(url, json=None, headers=None, timeout=None, **kw):
        seen["url"], seen["json"] = url, json
        seen["headers"], seen["timeout"] = headers or {}, timeout
        if boom:
            raise boom
        return reply or _Reply()
    from codai.api import pod_http
    monkeypatch.setattr(pod_http, "post", _post)
    return pool, seen


def test_the_warmup_asks_the_served_model_for_one_token(monkeypatch):
    """Cheap on purpose: the point is to make the weights resident, not to
    generate anything."""
    pool, seen = _pool(monkeypatch)
    assert pool._warmup("p1", "https://10.0.0.2:8000") is True
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["json"]["model"] == "datalab-to/surya-ocr-2"
    assert seen["json"]["max_tokens"] == 1


def test_it_carries_the_pods_bearer_token(monkeypatch):
    pool, seen = _pool(monkeypatch)
    pool._warmup("p1", "https://10.0.0.2:8000")
    assert seen["headers"]["Authorization"] == "Bearer tok"


def test_the_budget_is_its_own_setting(monkeypatch):
    """Answering /v1/models and having the weights resident are different
    milestones; a checkpoint re-downloaded per pod is the slow one."""
    pool, seen = _pool(monkeypatch, budget=1234)
    pool._warmup("p1", "https://10.0.0.2:8000")
    assert seen["timeout"] == 1234


def test_a_failed_warmup_does_not_reject_the_pod(monkeypatch):
    """We have already paid to boot it, and it may well serve — refusing it over
    a diagnostic request would be worse than the latency this removes."""
    pool, _ = _pool(monkeypatch, boom=OSError("connection reset"))
    assert pool._warmup("p1", "https://10.0.0.2:8000") is False


def test_an_http_error_is_reported_and_tolerated(monkeypatch, capsys):
    pool, _ = _pool(monkeypatch, reply=_Reply(status=400, text="bad request"))
    assert pool._warmup("p1", "https://10.0.0.2:8000") is False
    assert "serving anyway" in capsys.readouterr().out


def test_it_can_be_switched_off(monkeypatch):
    """An engine that does not take chat completions, or an operator who would
    rather have the pod available sooner."""
    pool, seen = _pool(monkeypatch, warmup=False)
    assert pool._warmup("p1", "https://10.0.0.2:8000") is False
    assert seen == {}                      # nothing was sent


def test_a_zero_budget_switches_it_off_too(monkeypatch):
    pool, seen = _pool(monkeypatch, budget=0)
    assert pool._warmup("p1", "https://10.0.0.2:8000") is False
    assert seen == {}


def test_a_successful_warmup_says_how_long_the_load_took(monkeypatch, capsys):
    """The number nobody had: it is the difference between a slow pod and a
    hung request."""
    pool, _ = _pool(monkeypatch)
    pool._warmup("p1", "https://10.0.0.2:8000")
    out = capsys.readouterr().out
    assert "weights resident after" in out


def test_provisioning_warms_up_before_calling_the_pod_ready():
    """The ordering is the whole feature: the wait has to land before 'ready',
    not inside the first request."""
    import inspect
    src = inspect.getsource(rw.RunpodPodPool._provision_one)
    assert "_warmup" in src
    assert src.index("_warmup") < src.index("ready for")


def test_the_defaults_warm_up(monkeypatch):
    """A pod that is reachable but cannot yet answer is the case that cost an
    evening, so this is on unless it is turned off."""
    cfg = rw.RunpodModelConfig()
    assert cfg.warmup_on_boot is True
    assert cfg.warmup_timeout_s >= 600
