"""A character request must reserve the IP-Adapter's VRAM, not borrow it.

Measured 2026-11-01 on zeiss. All three image failures of the run were
character requests, and two of them were CUDA OOM by a hair:

    CyberRealistic-Pony:  tried to allocate 96.00 MiB, 50.00 MiB free
    Krea-2-RealDream:     tried to allocate 12.00 MiB,  8.00 MiB free

Both had been admitted as fitting. A character request loads an IP-Adapter on
top of the pipeline -- the adapter weights plus the CLIP image encoder it
projects from, about 2 GB for SDXL -- and ``extra_vram_gb`` counted LoRAs only,
so the adapter came out of whatever slack was left.
"""
import inspect

import codai.api.images as images
from codai.api.characters import character_groups


class _Req:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_a_plain_request_reserves_nothing():
    assert images._character_ref_vram_gb(_Req(prompt="a cat")) == 0.0


def test_an_empty_list_reserves_nothing():
    assert images._character_ref_vram_gb(
        _Req(character_profiles=[], characters=[])) == 0.0


def test_every_spelling_reserves_the_adapter():
    for attr in ("characters", "character_profiles", "character_references"):
        got = images._character_ref_vram_gb(_Req(**{attr: ["x"]}))
        assert got == images._IP_ADAPTER_VRAM_GB, attr


def test_the_fields_are_the_ones_the_reader_reads():
    """A name this helper checks and the resolver does not would be dead; a name
    the resolver reads and this does not is an unreserved adapter."""
    mine = set(inspect.getsource(images._character_ref_vram_gb)
               .split('for _attr in (')[1].split(')')[0]
               .replace('"', '').replace(' ', '').strip(',').split(','))
    theirs = {m for m in ("characters", "character_profiles",
                          "character_references")
              if f'getattr(request, "{m}"' in
              inspect.getsource(character_groups)}
    assert mine == theirs, f"helper reads {mine}, resolver reads {theirs}"


def test_admission_adds_it_to_the_lora_reservation():
    src = inspect.getsource(images)
    assert "extra_vram_gb=_lora_extra_gb + _char_extra_gb," in src, (
        "the character reservation must reach request_model; on its own the "
        "figure changes nothing"
    )


def test_the_reservation_covers_the_measured_shortfalls():
    """Either OOM would have been avoided by reserving the adapter."""
    for free_mib, wanted_mib in ((50.0, 96.0), (8.0, 12.0)):
        short_gb = (wanted_mib - free_mib) / 1024.0
        assert images._IP_ADAPTER_VRAM_GB > short_gb
