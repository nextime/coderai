"""A meta-tensor failure must say WHICH parameters were missing.

`Cannot copy out of meta tensor; no data!` means some parameters were never
materialised. For a GGUF that almost always means the checkpoint does not
contain them -- and the message names none of them, which sends the reader
hunting for a corrupt file.

Measured 2026-10-27 on LTX-2.3-10Eros and -PinkCherry: diffusers'
convert_ltx2_transformer_to_diffusers maps their 4444 GGUF tensors onto
exactly the 4186 the class expects, and precisely 12 remain -- all in one
conditioning branch. Both fine-tunes, identically. The checkpoints are fine;
they were exported without that branch.
"""
import pytest

import codai.models.hf_loading as H


def test_a_non_meta_error_is_left_alone():
    # Only this one failure shape gets the treatment; everything else keeps
    # its own message.
    assert H._missing_tensor_hint(ValueError("something else"),
                                  "transformer", "/x.gguf", {}) == ""
    assert H._missing_tensor_hint(RuntimeError("CUDA out of memory"),
                                  "transformer", "/x.gguf", {}) == ""


def test_a_meta_error_it_cannot_analyse_degrades_to_nothing():
    # Best effort: an unreadable path, no converter, no config -- the original
    # error stands rather than this raising on top of it.
    out = H._missing_tensor_hint(
        RuntimeError("Cannot copy out of meta tensor; no data!"),
        "transformer", "/nonexistent/x.gguf", {"path": "/nonexistent"})
    assert out == ""


def test_it_never_offers_to_invent_the_weights():
    """The refusal has to stay a refusal.

    Filling in parameters a checkpoint does not contain is a substitution, and
    the whole point of the surrounding error is that a silent substitution is
    worse than a failure. The hint must therefore not promise a flag that
    papers over it.
    """
    import inspect
    src = inspect.getsource(H._missing_tensor_hint)
    assert "inventing them" in src
    for promise in ("allow_missing", "zero-init", "initialise them explicitly"):
        assert promise not in src, promise


def test_the_component_dir_resolver_tolerates_anything():
    assert H._component_dir({}, "transformer") is None
    assert H._component_dir({"path": "/nonexistent"}, "transformer") is None
    assert H._component_dir(None, "transformer") is None


def test_the_refusal_still_names_the_component_and_path():
    # The hint is an addition to the existing message, not a replacement:
    # whatever else happens, the caller still learns which component and which
    # file could not be loaded.
    import inspect
    src = inspect.getsource(H.build_gguf_pipeline_components)
    assert 'problems.append(f"{name} ({path}): {e}' in src
    assert "_missing_tensor_hint" in src
