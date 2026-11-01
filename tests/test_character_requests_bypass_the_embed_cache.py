# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The second character request on a cached pipeline failed at once.

The IP-Adapter rewrites the UNet's attention processors and the shape of what
cross-attention receives. Prompt embeddings encoded for one arrangement and
replayed into the other reach the processor as something it does not expect.

Measured 2026-11-01, on two models in a row:

    CyberRealistic-Pony  cold 768x1344 in 119 s, OK
                         warm failed after 1 s, bare 500
    pornmaster-pro-pony  cold 768x1344 in  75 s, OK
                         warm failed at once, bare 500

and in the log, immediately before each:

    Prompt embed cache HIT for model 'StableDiffusionXLPipeline#…'
    …
    AttributeError: 'tuple' object has no attribute 'shape'
      in diffusers/models/attention_processor.py

The client saw a bare 500 because the exception escapes through the Starlette
middleware rather than the handler, so nothing named a cause.

Z-Image is excluded from this cache for the same shape of reason -- embeddings
that are not safe to reuse -- and a character request is now excluded too.
"""

SRC = open("codai/api/images.py").read()


def _block():
    i = SRC.index("_embed_cacheable =")
    return SRC[max(0, i - 1800):i + 200]


def test_a_character_request_is_not_cacheable():
    assert '_embed_cacheable = "ZImage" not in _pipe_cls and not _has_char_refs' \
        in SRC


def test_all_three_kinds_of_reference_count():
    """Profiles, inline references and environment profiles all attach an
    IP-Adapter; any one of them changes the attention path."""
    body = _block()
    for field in ("character_profiles", "character_references",
                  "environment_profiles"):
        assert field in body, field


def test_it_is_decided_from_the_request_not_the_pipeline_state():
    """`pipeline._coderai_ip_state` is set LATER in this function, so reading it
    here would have let the cold request store a value that only the warm one
    could trip over -- which is exactly the sequence that failed."""
    body = _block()
    assert "_coderai_ip_state" not in body
    assert "getattr(request, 'character_profiles'" in body


def test_the_zimage_exclusion_is_kept():
    """It is a different reason for the same decision and must not be lost."""
    assert '"ZImage" not in _pipe_cls' in SRC


def test_the_flag_gates_both_the_read_and_the_write():
    """Reading a stale entry is what failed; writing one nothing will read is
    merely wasteful. Both are covered by the one flag."""
    i = SRC.index("_embed_cacheable =")
    after = SRC[i:i + 2600]
    assert "if _embed_cacheable else None" in after, "the read is gated"
    assert "and _embed_cacheable" in after, "the write is gated"
