"""A proxy that swallows logprobs does not lose a number — it invents one.

Surya asks every OCR request for `logprobs: true` (SURYA_INFERENCE_LOGPROBS
defaults to True in surya 0.22) and turns the per-token log-probabilities into
each line's confidence: `_mean_token_prob` in
surya/inference/backends/openai_client.py averages exp(logprob) over the tokens.
When the field comes back empty, surya does NOT report "unknown" — it falls back
to a literal 1.0 (surya/recognition/__init__.py:248 and :302, layout/__init__.py:97).

So dropping logprobs on the way through coderai had the worst possible failure
mode for a judicial-document pipeline: every line of every page comes back
marked perfectly confident, including the ones the model guessed at, and nothing
downstream can tell which page needs a human. The field had been hardcoded to
None in every response builder since the chat endpoint was written.
"""

import asyncio
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.backends.runpod import RunpodBackend                   # noqa: E402
from codai.backends.vllm import VllmBackend                       # noqa: E402
from codai.models.manager import ModelManager                     # noqa: E402

# One token with logprob 0 (p=1.0) and one with ln(0.5): a mean of 0.75, which is
# distinguishable from the 1.0 that a dropped field produces.
LOGPROBS = {"content": [{"token": "A", "logprob": 0.0, "bytes": [65], "top_logprobs": []},
                        {"token": "B", "logprob": -0.6931471805599453,
                         "bytes": [66], "top_logprobs": []}]}


class _Reply:
    #: A real response reports its code, and the backend reads it so that a
    #: server's own explanation of a 4xx survives the trip up.
    status_code = 200
    ok = True
    reason = "OK"
    url = "http://stub/v1/chat/completions"

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _answer(monkeypatch, module, seen, with_logprobs=True):
    """Capture the outbound payload and answer like a vLLM server."""
    def _post(url, json=None, headers=None, timeout=None, **kw):
        seen.append(json)
        choice = {"message": {"role": "assistant", "content": "hello"},
                  "finish_reason": "stop", "index": 0}
        if with_logprobs:
            choice["logprobs"] = LOGPROBS
        return _Reply({"choices": [choice],
                       "usage": {"prompt_tokens": 3, "completion_tokens": 2,
                                 "total_tokens": 5}})
    monkeypatch.setattr(module, "post", _post, raising=False)
    return seen


# ---------------------------------------------------------------- vLLM proxy #
def _vllm_backend(monkeypatch, seen, **kw):
    import requests
    backend = VllmBackend()
    backend._url = "http://pod:8000"
    _answer(monkeypatch, requests, seen, **kw)
    return backend


def test_vllm_asks_the_server_for_logprobs(monkeypatch):
    seen: list = []
    backend = _vllm_backend(monkeypatch, seen)
    backend.generate_chat([{"role": "user", "content": "x"}], logprobs=True)
    assert seen[0]["logprobs"] is True


def test_vllm_passes_top_logprobs_through(monkeypatch):
    seen: list = []
    backend = _vllm_backend(monkeypatch, seen)
    backend.generate_chat([{"role": "user", "content": "x"}], logprobs=True, top_logprobs=5)
    assert seen[0]["top_logprobs"] == 5


def test_vllm_does_not_ask_when_the_client_did_not(monkeypatch):
    """logprobs cost tokens on the wire and time on the server; an unasked-for
    request must look exactly as it did before."""
    seen: list = []
    backend = _vllm_backend(monkeypatch, seen)
    backend.generate_chat([{"role": "user", "content": "x"}])
    assert "logprobs" not in seen[0]
    assert "top_logprobs" not in seen[0]


def test_vllm_hands_back_what_the_server_measured(monkeypatch):
    seen: list = []
    backend = _vllm_backend(monkeypatch, seen)
    backend.generate_chat([{"role": "user", "content": "x"}], logprobs=True)
    assert backend.get_last_logprobs() == LOGPROBS


def test_a_second_call_without_logprobs_does_not_report_the_first_ones(monkeypatch):
    """Stale logprobs would be read as THIS page's confidence."""
    seen: list = []
    backend = _vllm_backend(monkeypatch, seen)
    backend.generate_chat([{"role": "user", "content": "x"}], logprobs=True)
    _answer(monkeypatch, __import__("requests"), seen, with_logprobs=False)
    backend.generate_chat([{"role": "user", "content": "y"}])
    assert backend.get_last_logprobs() is None


def test_the_content_still_comes_back(monkeypatch):
    seen: list = []
    backend = _vllm_backend(monkeypatch, seen)
    assert backend.generate_chat([{"role": "user", "content": "x"}],
                                 logprobs=True) == "hello"


# -------------------------------------------------------------- RunPod proxy #
def _runpod_backend(monkeypatch, seen, **kw):
    from codai.api import pod_http
    backend = RunpodBackend.__new__(RunpodBackend)
    backend._last_usage = {}
    backend._last_logprobs = None
    backend._served_name = "datalab-to/surya-ocr-2"
    backend._headers = {}
    backend._inflight = 0
    # (base, release, bearer): the acquire hands out the POD's token with the
    # address, since pods in one pool can carry different ones.
    monkeypatch.setattr(backend, "_acquire_endpoint",
                        lambda: ("http://pod/v1", lambda: None, "pod-tok"),
                        raising=False)
    _answer(monkeypatch, pod_http, seen, **kw)
    return backend


def test_the_pod_is_asked_too(monkeypatch):
    """Surya attaches to the POD's coderai, so this is the hop that matters for
    the Digesta install: Aruba has no GPU and never serves the VLM itself."""
    seen: list = []
    backend = _runpod_backend(monkeypatch, seen)
    backend.generate_chat([{"role": "user", "content": "x"}], logprobs=True)
    assert seen[0]["logprobs"] is True
    assert backend.get_last_logprobs() == LOGPROBS


def test_a_runpod_request_without_logprobs_is_unchanged(monkeypatch):
    seen: list = []
    backend = _runpod_backend(monkeypatch, seen)
    backend.generate_chat([{"role": "user", "content": "x"}])
    assert "logprobs" not in seen[0]


# ------------------------------------------------------------- the plumbing #
class _Backend:
    """A backend that accepts logprobs, as the proxies now do."""

    def __init__(self):
        self.seen = {}

    def generate_chat(self, messages, max_tokens=None, temperature=0.7, top_p=1.0,
                      stop=None, tools=None, response_format=None,
                      enable_thinking=False, repeat_penalty=1.0, presence_penalty=0.0,
                      frequency_penalty=0.0, logprobs=None, top_logprobs=None):
        self.seen = {"logprobs": logprobs, "top_logprobs": top_logprobs}
        return "ok"

    def get_last_logprobs(self):
        return LOGPROBS


class _OldBackend:
    """A backend written before any of this: no logprobs, no accessor."""

    def __init__(self):
        self.calls = 0

    def generate_chat(self, messages, max_tokens=None, temperature=0.7, top_p=1.0,
                      stop=None, tools=None, response_format=None,
                      enable_thinking=False, repeat_penalty=1.0, presence_penalty=0.0,
                      frequency_penalty=0.0):
        self.calls += 1
        return "ok"


def _manager(backend):
    mgr = ModelManager.__new__(ModelManager)
    mgr.backend = backend
    return mgr


def test_the_manager_forwards_the_request():
    backend = _Backend()
    assert _manager(backend).generate_chat([{"role": "user", "content": "x"}],
                                           logprobs=True, top_logprobs=3) == "ok"
    assert backend.seen == {"logprobs": True, "top_logprobs": 3}


def test_a_backend_that_cannot_do_logprobs_is_called_once_and_still_answers():
    """The kwargs ladder negotiates by catching TypeError, which would call a
    backend twice for a kwarg it never had. The signature is read instead."""
    backend = _OldBackend()
    assert _manager(backend).generate_chat([{"role": "user", "content": "x"}],
                                           logprobs=True) == "ok"
    assert backend.calls == 1


def test_an_old_backend_reports_no_logprobs_rather_than_failing():
    assert _manager(_OldBackend()).get_last_logprobs() is None


def test_the_manager_reads_them_back():
    assert _manager(_Backend()).get_last_logprobs() == LOGPROBS


# ------------------------------------------------------- streamed accumulation #
def test_streamed_logprobs_are_accumulated_into_one_object():
    """A stream carries logprobs a few tokens per chunk; a caller reading them
    after the stream wants the whole list, as the buffered path gives it."""
    backend = VllmBackend()
    backend._last_logprobs = None
    backend._collect_logprobs({"content": [LOGPROBS["content"][0]]})
    backend._collect_logprobs({"content": [LOGPROBS["content"][1]]})
    assert backend.get_last_logprobs() == LOGPROBS


def test_a_chunk_without_logprobs_adds_nothing():
    backend = VllmBackend()
    backend._last_logprobs = None
    backend._collect_logprobs(None)
    backend._collect_logprobs({})
    backend._collect_logprobs({"content": []})
    assert backend.get_last_logprobs() is None


# --------------------------------------------------- what surya does with them #
def test_the_mean_token_probability_is_recoverable():
    """Surya's own arithmetic, over what the proxy now returns: a confidence that
    was measured (0.75) rather than the 1.0 it assumes when the field is empty."""
    import math
    probs = [math.exp(t["logprob"]) for t in LOGPROBS["content"]]
    assert abs(sum(probs) / len(probs) - 0.75) < 1e-9
