"""The window a proxied model reports must be the one it is served with.

2026-10-22, production: `qwen38` is served from a rented pod launched with
`--max-model-len 32768`, `/v1/models` published 32768 -- and every answer's
`usage.context_size` said **2048**. Two numbers for one model, from one process.

The cause is a precedence, not a typo. A pod-served model's window lives in its
`runpod` block (`ctx`, which becomes the pod's `--max-model-len`), while the
backend also accepts a generic `n_ctx` kwarg -- and that kwarg carries the
NODE-WIDE fallback `vulkan.n_ctx` when the model's entry sets no `n_ctx` of its
own. Its default is 2048. The kwarg was applied last, so a global default for
local llama.cpp models silently became the published window of every pod-served
model on the install.

Reporting 2048 was the visible half. The dangerous half is auto-compaction,
which compacts to a percentage of exactly this number: enabling it would have
crushed every conversation to ~1300 tokens to fit a window sixteen times
larger, and the summaries would have looked like a model that could not
remember anything.
"""
import types

import pytest

from codai.backends.runpod import RunpodBackend
from codai.backends.vllm import VllmBackend


# --------------------------------------------------------------- pods (proxy) #

def _runpod(monkeypatch, block_ctx, **load_kwargs):
    """A RunpodBackend loaded with a runpod block of ``block_ctx`` tokens."""
    b = RunpodBackend.__new__(RunpodBackend)
    b._ctx = 32768
    b._model_id = b._served_name = "qwen38"
    b._acct = None
    b._headers = {}
    b._pool = None
    mcfg = types.SimpleNamespace(
        ctx=block_ctx, served_model="", mode="serverless", endpoint_id="e1",
        min_pods=0, max_pods=1)
    monkeypatch.setattr("codai.api.runpod_worker.model_runpod_block",
                        lambda name: {}, raising=False)
    monkeypatch.setattr("codai.api.runpod_worker.parse_model_runpod",
                        lambda blk: mcfg, raising=False)
    monkeypatch.setattr("codai.api.runpod_worker.serverless_base_url",
                        lambda acct, cfg: "https://pod:1/v1", raising=False)
    monkeypatch.setattr("codai.api.runpod_worker.auth_headers",
                        lambda acct: {}, raising=False)
    b.load_model("qwen38", **load_kwargs)
    return b


def test_the_pods_own_ctx_beats_the_node_wide_default(monkeypatch):
    """THE BUG: `vulkan.n_ctx` (default 2048) arrived as n_ctx and overwrote the
    32768 the pod is actually launched with."""
    b = _runpod(monkeypatch, 32768, n_ctx=2048)
    assert b.get_context_size() == 32768


def test_a_model_with_no_pod_ctx_still_takes_the_kwarg(monkeypatch):
    """The fallback is a fallback, not dead code: with nothing configured for the
    pod, the only number available is the one passed in."""
    b = _runpod(monkeypatch, 0, n_ctx=8192)
    assert b.get_context_size() == 8192


def test_with_neither_the_backend_keeps_its_own_default(monkeypatch):
    b = _runpod(monkeypatch, 0)
    assert b.get_context_size() == 32768


def test_a_junk_kwarg_cannot_shrink_the_window(monkeypatch):
    """A list or a string reaching n_ctx must not silently become the window --
    and must not raise inside the request path either."""
    for junk in ([], "", "nonsense", None, [4096]):
        b = _runpod(monkeypatch, 65536, n_ctx=junk)
        assert b.get_context_size() == 65536


# ------------------------------------------------------------- vLLM (in-process)

def _vllm(monkeypatch, block, **load_kwargs):
    from codai.config import VllmConfig
    b = VllmBackend.__new__(VllmBackend)
    b._cfg = VllmConfig()
    b._model_id = "m"
    b._svc_key = None
    b._url = None
    b._ctx = int(b._cfg.ctx)
    b._served_name = "m"
    b._last_usage = {}
    b._last_logprobs = None
    b._last_tool_calls = None
    b._last_reasoning = None
    monkeypatch.setattr(b, "_resolve_model_dir", lambda name: "/models/m")
    monkeypatch.setattr("codai.backends.overrides.model_block",
                        lambda name, key: dict(block))
    monkeypatch.setattr("codai.api.vllm_worker.resolve_service_key",
                        lambda cfg, path: (path, "svc"), raising=False)
    monkeypatch.setattr("codai.api.vllm_worker.ensure_service",
                        lambda cfg, **kw: "http://127.0.0.1:41721", raising=False)
    b.load_model("m", **load_kwargs)
    return b


def test_the_models_vllm_block_beats_the_node_wide_default(monkeypatch):
    b = _vllm(monkeypatch, {"ctx": 131072}, n_ctx=2048)
    assert b.get_context_size() == 131072


def test_a_per_model_n_ctx_reaches_the_launch_as_well(monkeypatch):
    """Reporting a window the server was not started with is the same bug in
    reverse, so an explicit n_ctx has to change what is launched too."""
    b = _vllm(monkeypatch, {}, n_ctx=65536)
    assert b.get_context_size() == 65536
    assert int(b._cfg.ctx) == 65536        # ...and that is what vLLM is given


def test_the_served_window_wins_once_the_server_has_been_asked(monkeypatch):
    """The configured number is a request; the served one can refuse a request.
    vLLM reports its own on /v1/models, and once that answer is in hand it is
    the truth -- measured on an install where `ctx: 18432` was configured and
    `--max-model-len 32768` was served."""
    b = _vllm(monkeypatch, {"ctx": 18432})
    assert b.get_context_size() == 18432
    b._ctx_len = 32768                      # as context_length() caches it
    assert b.get_context_size() == 32768


def test_an_unknown_served_window_is_not_mistaken_for_zero(monkeypatch):
    b = _vllm(monkeypatch, {"ctx": 32768})
    b._ctx_len = None                       # context_length() tried and failed
    assert b.get_context_size() == 32768
