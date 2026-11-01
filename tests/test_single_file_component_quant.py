# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""bitsandbytes cannot quantize a single-file component as it loads.

``from_single_file`` given a bnb config takes the PRE-quantized path: it
expects the checkpoint to carry bnb metadata already. A plain community
fine-tune carries none, so the load reads the whole file and dies with

    Supplied state dict for transformer_blocks.0.attn.to_gate.weight does not
    contain `bitsandbytes__*` and possibly other `quantized_stats` components

-- a tensor name instead of a cause. Measured 2026-11-01 on Krea-2-RealDream
(``civitai:3396368:3304483``, 12.9 GB), whose docstring had promised this
worked. It is refused up front now, naming what does work.
"""

import pytest

from codai.models import hf_loading


SRC = open("codai/models/hf_loading.py").read()


def _cfg(source, quant):
    return {"component_models": {"transformer": source},
            "component_quantization": quant,
            "path": "unsloth/Krea-2-Turbo"}


@pytest.mark.parametrize("source", [
    "civitai:3396368:3304483",
    "/AI/models/realDream_krea2V3.safetensors",
])
@pytest.mark.parametrize("mode", ["4bit", "8bit"])
def test_bnb_on_a_single_file_component_is_refused(source, mode):
    with pytest.raises(RuntimeError) as ei:
        hf_loading.build_component_models(
            "unsloth/Krea-2-Turbo", _cfg(source, {"transformer": mode}), None)
    msg = str(ei.value)
    assert "single file" in msg
    assert mode in msg
    assert "transformer" in msg


def test_the_refusal_names_what_does_work():
    """A refusal that does not say what to do instead just moves the dead end."""
    with pytest.raises(RuntimeError) as ei:
        hf_loading.build_component_models(
            "unsloth/Krea-2-Turbo",
            _cfg("civitai:3396368:3304483", {"transformer": "4bit"}), None)
    msg = str(ei.value)
    assert "GGUF" in msg, "the GGUF route is the one that quantizes on load"
    assert "offload" in msg


def test_it_refuses_before_reading_the_file_and_before_the_class_lookup():
    """12.9 GB read and then an error about a tensor name is the behaviour being
    replaced, so the check must precede the load. It must also precede the class
    lookup: that lookup `continue`s when the installed diffusers has no class
    for the component, which skipped the refusal entirely and is how this test
    caught it."""
    head = SRC.index("def build_component_models")
    body = SRC[head:SRC.index("def ", head + 10)]
    refusal = body.index("from_single_file cannot do")
    lookup = body.index("cls = getattr(diffusers, cls_name, None)")
    load = body.index("cls.from_single_file(")
    assert refusal < lookup < load


def test_an_unquantized_single_file_component_is_untouched():
    """The ordinary case -- a single-file component with no quantization asked
    for -- must not be caught by this."""
    cfg = _cfg("civitai:3396368:3304483", {})
    # No RuntimeError about single files; it fails later or succeeds, either way
    # not here. diffusers may be absent in the test environment, which returns
    # ({}, '') -- also acceptable, and not a refusal.
    try:
        hf_loading.build_component_models("unsloth/Krea-2-Turbo", cfg, None)
    except RuntimeError as exc:
        assert "from_single_file cannot do" not in str(exc)


def test_a_repo_component_may_still_be_quantized():
    """The capability is real for a repo; only the single-file path lacks it."""
    cfg = _cfg("prithivMLmods/Some-Transformer", {"transformer": "4bit"})
    try:
        hf_loading.build_component_models("unsloth/Krea-2-Turbo", cfg, None)
    except RuntimeError as exc:
        assert "from_single_file cannot do" not in str(exc)


def test_the_docstring_no_longer_promises_it():
    """The false promise is why the config change was tried at all."""
    head = SRC.index("def build_component_models")
    doc = SRC[head:SRC.index('"""', SRC.index('"""', head) + 3)]
    assert "does NOT apply to a component that is a single FILE" in doc
