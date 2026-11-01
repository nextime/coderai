# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""One checkpoint, two cards, two sizes -- admission must use the one asked for.

Krea-2-Turbo and Krea-2-RealDream share `unsloth/Krea-2-Turbo` as their path
and are nothing like the same size: Turbo serves a Q4_K_M GGUF transformer
(9.7 GB estimated), RealDream a 12.84 GiB fp8 one (20.45 GB estimated, 17.22 GB
measured).

By the time the admission check runs, `model_name` has been rewritten from the
alias to that shared path, so looking it up first sized RealDream as Turbo.
Measured 2026-11-01:

    Image model needs 9.7 GB, 13.6 GB free — no eviction needed
    Measured footprint 'image:Krea-2-RealDream': 17.22 GB (4.60 GB offloaded)
    image generation error: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR

No eviction ran, accelerate spilled 4.60 GB to host RAM, and a quantized model
with part of itself on the CPU fails every request -- which is the outcome the
offload refusal exists to avoid, reached by a different route.

`requested_id` was already being passed and is only a disambiguation hint: it
cannot override an explicit key naming the other card, which is why the fix had
to be in the key lookup.
"""

SRC = open("codai/api/images.py").read()


def _block():
    i = SRC.index("# Needed VRAM for this model")
    return SRC[i:SRC.index("if _need_gb > 0:", i)]


def test_the_requested_id_is_tried_before_the_rewritten_path():
    body = _block()
    req = body.index("_requested_id}")
    path = body.index("image:{model_name}")
    assert req < path, "the shared checkpoint must not be sized first"


def test_both_spellings_of_each_are_tried():
    """With and without the `image:` prefix, since typed entries are keyed one
    way and bare ids the other."""
    body = _block()
    for probe in ('f"image:{_requested_id}"', "_requested_id,",
                  'f"image:{model_name}"', "model_name)"):
        assert probe in body, probe


def test_an_empty_candidate_is_not_looked_up():
    """_requested_id can be blank; `"" in config` would be a pointless lookup
    and `None` a TypeError on some mappings."""
    body = _block()
    assert "if _k and _k in _mmm.config" in body


def test_the_requested_id_hint_is_still_passed():
    """Belt and braces: the key chooses the card, the hint disambiguates
    same-path siblings inside the sizer."""
    i = SRC.index("_need_gb = _mmm._get_model_used_vram_gb")
    assert "requested_id=_requested_id" in SRC[i:i + 260]


def test_the_rewrite_that_caused_it_is_still_there():
    """The alias -> path rewrite is correct and load depends on it; this test
    exists so that if it ever moves, the ordering above is re-examined rather
    than silently becoming a no-op."""
    assert "model_name = _cfg_path" in SRC
    assert SRC.index("model_name = _cfg_path") < SRC.index(
        "# Needed VRAM for this model")
