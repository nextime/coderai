"""Siblings sharing one checkpoint must be sized by the card that was ASKED for.

Measured 2026-11-01 on zeiss. Three image cards all carry
``path: unsloth/Krea-2-Turbo`` -- Krea-2-Turbo (GGUF transformer, 14.0 GB),
Krea-2-Edit (14.5 GB) and Krea-2-RealDream (civitai fp8 transformer, 19.0 GB).
A RealDream request produced:

    [vram-est] key='image:unsloth/Krea-2-Turbo' -> 9.68 GB
    Image model needs 9.7 GB, 14.2 GB free — no eviction needed
    Measured footprint for 'image:Krea-2-RealDream': 17.22 GB
      (of which 4.03 GB was offloaded to host RAM this load)
    torch.OutOfMemoryError: Tried to allocate 12.00 MiB ... 8.00 MiB free

Admission sized Turbo, the loader loaded RealDream, and the 3 GB gap spilled to
host RAM -- a quantized model with part of itself on the CPU fails every
request. ``request_model`` returns the alias key and the resolved path
separately and only the path was passed to the loader, so the loader's idea of
"what was requested" was the shared checkpoint.
"""
import inspect

import codai.api.images as images


def test_the_loader_takes_a_model_key():
    sig = inspect.signature(images._load_diffusers_pipeline)
    assert "model_key" in sig.parameters, (
        "the loader must be told which CARD was requested; the path cannot "
        "distinguish siblings that share a checkpoint"
    )
    assert sig.parameters["model_key"].default is None, (
        "optional, for the preload path which has no key yet"
    )


def test_the_requested_id_comes_from_the_key_not_the_path():
    src = inspect.getsource(images._load_diffusers_pipeline)
    head = src[:src.index("def _cfg(")]
    assert "_requested_id = model_name" in head
    assert "if model_key:" in head, (
        "_requested_id must prefer model_key; by the time the request path "
        "reaches the loader, model_name is already the checkpoint"
    )
    assert 'model_key.split(":", 1)[1]' in head, (
        "the bare id is needed, since the sizing lookup re-prefixes 'image:'"
    )


def test_both_request_call_sites_pass_the_key():
    src = inspect.getsource(images)
    calls = [b for b in src.split("_load_diffusers_pipeline,")[1:]]
    assert len(calls) == 2, "expected the two request-path call sites"
    for c in calls:
        assert "model_key" in c[:200], (
            "a call site that omits the key silently reintroduces the bug"
        )


def test_sizing_tells_the_siblings_apart():
    """The key is what separates them -- the path cannot."""
    from codai.models.manager import MultiModelManager
    m = MultiModelManager.__new__(MultiModelManager)
    m.config = {
        "image:Krea-2-Turbo": {"path": "unsloth/Krea-2-Turbo",
                               "used_vram_gb": 14.0},
        "image:Krea-2-RealDream": {"path": "unsloth/Krea-2-Turbo",
                                   "used_vram_gb": 19.0},
    }
    m._measured_vram_gb = {}
    turbo = m._get_model_used_vram_gb("image:Krea-2-Turbo")
    dream = m._get_model_used_vram_gb("image:Krea-2-RealDream")
    assert turbo > 0 and dream > 0
    assert dream > turbo, (
        "RealDream is the larger card; sizing it as Turbo is what spilled "
        "4.03 GB to host RAM"
    )
