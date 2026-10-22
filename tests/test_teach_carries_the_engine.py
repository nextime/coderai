"""A taught model must reach the pod with the engine that can serve it.

2026-10-22, digesta-alpha on a fresh pod:

  [boot +0.8s] seeded models registered: qwen38-awq, diogene-alpha,
               digesta-alpha, qwen38-vision          <- correct
  ...
  Resolving model: digesta-alpha
  [register] digesta-alpha registered in text_models <- teach, overwriting it
  [lora] added 'digesta-alpha' to the running engine — no reload
  Loading model on demand: digesta-alpha
  Warning: No GPU backend detected.
  Error loading model digesta-alpha: Unknown backend: cpu

seed_model_env strips `backend` unless it is told the engine: a RunPod model
carries `backend: "runpod"`, which is THIS install's placement and means
nothing on a pod. The seed path has passed the engine since the same fault bit
diogene-alpha; teach_model did not, so a model seeded correctly at boot was
re-registered -- which happens whenever an adapter must be sent -- and the good
entry was replaced by a backend-less one. It worked until something taught it,
which is why it looked intermittent.
"""
import json

import pytest

from codai.api import remote_gateway as rg
from codai.api.runpod_worker import engine_pod_backend, parse_model_runpod


ENTRY = {
    "id": "digesta-alpha",
    "backend": "runpod",
    "loras": [{"name": "digesta-alpha", "path": "sha256:ab", "weight": 1.0}],
    "runpod": {"mode": "pods", "engine": "coderai-vllm",
               "hf_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4",
               "served_model": "digesta-alpha"},
}


def test_the_pool_engine_resolves_to_the_pods_backend():
    mcfg = parse_model_runpod(ENTRY["runpod"])
    assert engine_pod_backend(ENTRY, mcfg) == "vllm"


@pytest.fixture
def taught(monkeypatch):
    """Capture the payload teach_model posts to the pod."""
    sent = {}

    class _R:
        ok = True
        status_code = 200
        text = ""
    def _post(url, data=None, headers=None, timeout=None):
        sent["url"] = url
        sent["body"] = json.loads(data)
        return _R()
    monkeypatch.setattr("codai.api.pod_http.post", _post)
    monkeypatch.setattr("codai.models.manager._model_entry_for",
                        lambda name: ENTRY if name == "digesta-alpha" else {})
    rg._REGISTERED.clear()
    return sent


def test_a_taught_model_carries_its_engine(taught):
    """THE BUG: this arrived with no `backend` at all, and the pod -- which has
    no local GPU backend -- fell through to "cpu"."""
    assert rg.teach_model("https://pod:1", "k", "digesta-alpha") is True
    rows = taught["body"]
    assert rows, "nothing was registered"
    row = next(r for r in rows if (r.get("id") or r.get("alias")) == "digesta-alpha")
    assert row["backend"] == "vllm"


def test_the_adapter_still_travels(taught):
    rg.teach_model("https://pod:1", "k", "digesta-alpha")
    row = next(r for r in taught["body"]
               if (r.get("id") or r.get("alias")) == "digesta-alpha")
    assert row["loras"], "the LoRA must still be registered with it"
    assert row["path"] == "cyankiwi/Qwen3.8-27B-AWQ-INT4"


def test_a_model_with_no_pod_engine_is_unchanged(taught, monkeypatch):
    """`backend: runpod` alone says nothing a pod can use, so it is still
    stripped -- the pod uses its own default, as before."""
    entry = {"id": "plain", "backend": "runpod", "path": "org/plain",
             "runpod": {"mode": "pods"}}
    monkeypatch.setattr("codai.models.manager._model_entry_for",
                        lambda name: entry if name == "plain" else {})
    rg._REGISTERED.clear()
    rg.teach_model("https://pod:1", "k", "plain")
    row = taught["body"][0]
    assert "backend" not in row


def test_it_posts_to_the_register_endpoint(taught):
    rg.teach_model("https://pod:1", "k", "digesta-alpha")
    assert taught["url"].endswith("/v1/models/register")
