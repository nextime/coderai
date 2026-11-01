"""An image card that states no precision must not be handed 'f32'.

Measured 2026-11-01 on zeiss. PonyRealism-Uncensored's config has no
``precision`` key at all, yet the loader logged

    Using precision: f32 (torch.float32)

with no "(auto: ...)" suffix -- proof the auto branch never ran. The cause was
codai/main.py's translated-kwargs builder defaulting the key to 'f32', so by
the time codai/api/images.py read it the key was always set and its
card-aware half-width policy (_auto_precision) was dead code for every image
model. The result:

    Measured footprint ...: 14.98 GB            (7.49 GB at half width)
    IP-Adapter: aligned 72 module(s) to torch.float32
    OutOfMemoryError: Tried to allocate 1008.00 MiB ... 306.00 MiB is free

Double the weights and a float32 IP-Adapter, and the VAE decode had nowhere to
go. Every other consumer of the key reads it as ``get('precision') or 'bf16'``.
"""
import inspect
import re

import codai.main as cmain
import codai.api.images as images


def _image_branch():
    src = inspect.getsource(cmain)
    i = src.index('elif model_type == "image":')
    return src[i:src.index('elif model_type == "audio":', i)]


def test_the_image_branch_supplies_no_precision_default():
    branch = _image_branch()
    m = re.search(r"kwargs\['precision'\] = model_cfg\.get\('precision'([^)]*)\)",
                  branch)
    assert m, "the image branch must still set the key"
    assert m.group(1).strip() == "", (
        "a default here pre-empts the loader's own card-aware policy; "
        f"found default {m.group(1)!r}"
    )


def test_the_loader_only_auto_resolves_when_the_key_is_absent():
    """Which is why the default above mattered: a set key skips auto."""
    src = inspect.getsource(images._load_diffusers_pipeline)
    assert "if not _precision_cfg or _precision_cfg.lower() == 'auto':" in src


def test_auto_picks_half_width_on_a_small_card():
    precision, why = images._auto_precision()
    assert why, "every branch must say why, so the log shows it ran"
    assert precision in ("bf16", "f32")


def test_an_f32_card_reserves_the_wider_adapter():
    class _Req:
        character_profiles = ["x"]
    half = images._character_ref_vram_gb(_Req(), {"precision": "bf16"})
    full = images._character_ref_vram_gb(_Req(), {"precision": "f32"})
    assert full == half * 2.0, (
        "diffusers loads the adapter at the pipeline's dtype, so an f32 "
        "pipeline takes twice the half-width figure"
    )


def test_an_unknown_precision_reserves_the_half_width_figure():
    class _Req:
        character_profiles = ["x"]
    assert images._character_ref_vram_gb(_Req(), {}) == images._IP_ADAPTER_VRAM_GB
    assert images._character_ref_vram_gb(_Req(), None) == images._IP_ADAPTER_VRAM_GB


def test_a_plain_request_still_reserves_nothing_whatever_the_precision():
    class _Req:
        prompt = "a cat"
    assert images._character_ref_vram_gb(_Req(), {"precision": "f32"}) == 0.0
