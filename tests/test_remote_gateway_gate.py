"""The gate that decides whether the gateway looks at a request at all.

`RemoteGatewayMiddleware.__call__` short-circuits on `any_remote_configured()`
before it ever calls `resolve_target`. So anything that gate cannot see is
served LOCALLY, whatever `model_placement` would have said — and the two
disagreeing is invisible from outside, because every page that reports where a
model runs asks `model_placement` directly.

Reported from production and reproduced on 0.3.80: `/v1/embeddings` for a
RunPod-backed bge-m3 spent 312 seconds downloading from HuggingFace onto a
GPU-less orchestrator and then returned 500, while
`/v1/models/test/state?model=bge-m3` answered `"placement":"pod"`.

The gate knew about capability endpoints, capability pods and `service_url`.
It did not know about a per-model `backend`. An install whose remote models are
ALL per-model pods therefore answered "nothing is remote".

OCR was unaffected, which is why this survived a release: the OCR manager calls
`placement.choose` itself instead of going through this middleware.

The invariant: if `model_placement` would send a model away from here, the gate
must say so too.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import remote_gateway as rg                      # noqa: E402


class _CM:
    def __init__(self, data):
        self.models_data = data


@pytest.fixture(autouse=True)
def _no_capability_remotes(monkeypatch):
    """The install this broke on: no capability endpoints, no capability pods."""
    monkeypatch.setattr(rg, "capability_endpoints", lambda *a, **k: [])
    monkeypatch.setattr(rg, "capability_pods", lambda *a, **k: [])
    rg._ANY_CACHE[0] = 0.0          # the gate is cached for 5s; start cold
    yield
    rg._ANY_CACHE[0] = 0.0


def _with_models(monkeypatch, data):
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager", _CM(data), raising=False)


POD_ENTRY = {"id": "BAAI/bge-m3", "alias": "bge-m3", "backend": "runpod",
             "capability": "embeddings",
             "runpod": {"mode": "pods", "engine": "coderai",
                        "image": "ghcr.io/nextime/coderai-embeddings:latest"}}


def test_a_per_model_pod_counts_as_a_remote(monkeypatch):
    """THE regression. This returned False, so the middleware handed every
    request to the local app and a GPU-less box loaded bge-m3 itself."""
    _with_models(monkeypatch, {"embedding_models": [POD_ENTRY]})
    assert rg._any_remote_configured_uncached() is True


def test_the_gate_agrees_with_model_placement(monkeypatch):
    """The invariant, stated directly: anything placed away from here must be
    visible to the gate, or the gate silently overrules the placement."""
    _with_models(monkeypatch, {"embedding_models": [POD_ENTRY]})
    assert rg.model_placement("bge-m3")[0] == "pod"
    assert rg._any_remote_configured_uncached() is True


def test_a_pod_in_any_section_counts(monkeypatch):
    """Embeddings was the one that broke, but the gate walks every section and
    a text or image model on a pod is the same situation."""
    for section in ("text_models", "image_models", "video_models",
                    "embedding_models", "tts_models"):
        rg._ANY_CACHE[0] = 0.0
        _with_models(monkeypatch, {section: [dict(POD_ENTRY)]})
        assert rg._any_remote_configured_uncached() is True, section


def test_a_host_backed_model_counts(monkeypatch):
    _with_models(monkeypatch, {"text_models": [
        {"id": "m", "backend": "host", "host": {"url": "http://box:8000"}}]})
    assert rg._any_remote_configured_uncached() is True


def test_a_host_entry_with_a_list_of_hosts_counts(monkeypatch):
    _with_models(monkeypatch, {"text_models": [
        {"id": "m", "backend": "host", "host": {"hosts": ["http://a", "http://b"]}}]})
    assert rg._any_remote_configured_uncached() is True


def test_a_service_url_still_counts(monkeypatch):
    _with_models(monkeypatch, {"text_models": [
        {"id": "m", "service_url": "http://elsewhere:8000"}]})
    assert rg._any_remote_configured_uncached() is True


# ------------------------------------------- and what must NOT count as remote #
def test_a_purely_local_install_still_says_no(monkeypatch):
    """The gate exists to keep the hot path free on an install with nothing
    remote. Making it answer True for everything would buffer every body."""
    _with_models(monkeypatch, {"text_models": [{"id": "m", "path": "/models/m"}],
                               "embedding_models": [{"id": "e"}]})
    assert rg._any_remote_configured_uncached() is False


def test_a_runpod_block_without_the_backend_is_a_leftover(monkeypatch):
    """Disabling a model's pod by flipping `backend` must actually disable it:
    the block is left behind so the settings survive, and `model_placement`
    reads the backend too."""
    e = dict(POD_ENTRY, backend="local")
    _with_models(monkeypatch, {"embedding_models": [e]})
    assert rg.model_placement("bge-m3")[0] != "pod"
    assert rg._any_remote_configured_uncached() is False


def test_a_backend_without_its_block_is_nothing_to_route_to(monkeypatch):
    e = {"id": "m", "backend": "runpod"}
    _with_models(monkeypatch, {"text_models": [e]})
    assert rg._any_remote_configured_uncached() is False


def test_a_host_backend_with_neither_url_nor_hosts_does_not_count(monkeypatch):
    _with_models(monkeypatch, {"text_models": [
        {"id": "m", "backend": "host", "host": {}}]})
    assert rg._any_remote_configured_uncached() is False


def test_junk_in_models_data_cannot_break_the_hot_path(monkeypatch):
    _with_models(monkeypatch, {"text_models": ["not a dict", None, 7],
                               "whatever": "not a list"})
    assert rg._any_remote_configured_uncached() is False


def test_no_config_manager_at_all_is_not_an_error(monkeypatch):
    import codai.admin.routes as routes
    monkeypatch.setattr(routes, "config_manager", None, raising=False)
    assert rg._any_remote_configured_uncached() is False
