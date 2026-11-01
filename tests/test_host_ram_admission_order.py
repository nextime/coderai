# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""An admission check must run before the allocation it admits.

The host-RAM check sat about 150 lines AFTER build_component_models, and a
single-file component is loaded to CPU first. So it counted the memory of the
very load it was deciding on. Measured 2026-11-01 on Krea-2-RealDream:

    09:07:54  component 'transformer': 256 per-tensor weight scales applied
    09:07:54  Loading 'image:unsloth/Krea-2-Turbo': needs ~6.8 GB host RAM
              (9.7 GB footprint against 4.8 GB free VRAM spills 4.8 GB to the
              host, plus 2.0 GB to stage it), budget is 0.0 GB (-6.1 GB left
              under the 43 GB server RAM cap)
    09:07:56  HostRamExhausted → 400

Its 12.84 GB fp8 transformer was already resident. The fleet read 49.1 GB
against a 43 GB cap and the load was refused for wanting 2 GB to stage the
rest. Minutes later the same fleet measured 3.99 GB held with models loaded and
45.13 GB available, so nothing was being retained: the 49 GB was this load,
counted against itself.

Two further sizing errors are in the same two lines and have the same cause as
the VRAM ones: `9.7 GB footprint` is Krea-2-Turbo's size, because `model_name`
is the shared checkpoint by then, and `4.8 GB free VRAM` is the figure with
this load's own reservation subtracted, on a card the next load measured at
25.3 GB free.
"""

SRC = open("codai/api/images.py").read()


def test_the_host_ram_check_precedes_the_component_load():
    """The whole point: nothing of this model may be resident when its budget
    is decided."""
    check = SRC.index("# Host-RAM admission")
    build = SRC.index("build_component_models(model_name, _mc, dtype)")
    assert check < build, (
        "the host-RAM check must run before any component is loaded to CPU")


def test_it_also_precedes_the_gguf_and_quant_component_building():
    """Those stage weights too."""
    check = SRC.index("# Host-RAM admission")
    for probe in ("build_gguf_pipeline_components(", "build_pipeline_quant_config("):
        assert check < SRC.index(probe), probe


def test_it_sizes_the_requested_card_not_the_shared_checkpoint():
    """Krea-2-RealDream shares unsloth/Krea-2-Turbo and is twice its size."""
    i = SRC.index("# Host-RAM admission")
    body = SRC[i:SRC.index("ensure_host_ram_for", i) + 200]
    req = body.index('f"image:{_requested_id}"')
    path = body.index('f"image:{model_name}"')
    assert req < path


def test_an_empty_candidate_is_not_looked_up():
    i = SRC.index("# Host-RAM admission")
    body = SRC[i:SRC.index("ensure_host_ram_for", i) + 200]
    assert "if _k and _k in _mmm.config" in body


def test_the_requested_id_is_still_passed_to_the_sizer():
    i = SRC.index("# Host-RAM admission")
    assert "_requested_id)" in SRC[i:SRC.index("ensure_host_ram_for", i) + 240]


def test_the_check_is_raised_not_swallowed():
    """A refusal is the point of it; catching it would admit every load."""
    i = SRC.index("# Host-RAM admission")
    body = SRC[i:SRC.index("build_pipeline_quant_config(")]
    # The only try/except in the block guards the IMPORT, not the call.
    assert body.count("try:") == 1
    assert "except Exception:\n        _mmm = None" in body
    call = body.index("_mmm.ensure_host_ram_for")
    assert "try:" not in body[body.index("if _mmm is not None:"):call]


def test_the_requested_id_is_assigned_before_the_check_uses_it():
    """It is captured before `model_name` is rewritten to the checkpoint path;
    if the check moved above that assignment it would read a NameError."""
    assign = SRC.index("_requested_id = model_name")
    check = SRC.index("# Host-RAM admission")
    assert assign < check
