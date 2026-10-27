"""A configured GGUF component that is not on disk should say so.

diffusers reports a missing single-file checkpoint as

    Invalid `pretrained_model_name_or_path` provided. Please set it to a
    valid URL.

which reads as "your repo id is wrong" rather than "that file is not here".
Measured on LTX-2.5, whose entry pointed at ~/.cache/huggingface on the HOST
— a path the container does not mount. The model had every other thing it
needed.
"""
import pytest

from codai.models import hf_loading


def test_a_missing_file_is_named_as_a_missing_file(tmp_path, monkeypatch):
    missing = str(tmp_path / "not-here" / "model-Q4_K_M.gguf")
    cfg = {"path": "Lightricks/LTX-2.5-Diffusers",
           "component_quantization": {"transformer": missing}}
    monkeypatch.setattr(hf_loading, "_discover_components",
                        lambda name: {"transformer": ["diffusers",
                                                      "LTX2VideoTransformer3DModel"]})
    with pytest.raises(RuntimeError) as err:
        hf_loading.build_gguf_pipeline_components("LTX-2.5", cfg, "bfloat16")
    msg = str(err.value)
    assert "no such file" in msg and missing in msg
    # And it points at the usual cause rather than leaving it a mystery.
    assert "mounted volume" in msg


def test_an_http_url_is_not_checked_on_disk(monkeypatch):
    """Remote GGUFs are a supported form; only local paths are probed."""
    import inspect

    src = inspect.getsource(hf_loading.build_gguf_pipeline_components)
    assert 'startswith(("http://", "https://"))' in src
