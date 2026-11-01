"""Every sizing decision in request_model must name the card that was asked for.

Measured 2026-11-01 on zeiss. Three image cards share
``path: unsloth/Krea-2-Turbo``; an edit on Krea-2-RealDream produced

    [vram-est] key='image:Krea-2-RealDream' source=used_vram_gb -> 9.68 GB
      used_vram_gb=14.0
      component_quantization={'transformer': '…krea2_turbo-Q4_K_M.gguf', …}
      quant_mult=0.640
    Loading 'image:Krea-2-RealDream': need 9.7 GB VRAM, have 7.3 GB free
    ...
    CUDA out of memory. Tried to allocate 12.00 MiB ... 8.00 MiB is free

The key was right and the CONFIG was Krea-2-Turbo's: 14.0 GB with a GGUF
transformer, 9.68 GB after its quantization multiplier, for a model that
measures 17.22 GB with a civitai fp8 transformer. _get_model_used_vram_gb can
tell the siblings apart, but its disambiguation only runs when it is given the
requested id, and two calls inside request_model omitted it -- including the
one the "need N GB, have M GB free" decision is made on.
"""
import re

import codai.models.manager as mgr
from codai.models.manager import MultiModelManager


def _request_model_body():
    import inspect
    src = inspect.getsource(mgr)
    start = src.index("    def request_model(self, requested_model")
    m = re.search(r"\n    def (?!request_model)", src[start + 10:])
    return src[start:start + 10 + m.start()]


def test_every_sizing_call_in_request_model_passes_the_requested_id():
    body = _request_model_body()
    lines = body.splitlines()
    missing = []
    for i, line in enumerate(lines):
        if "_get_model_used_vram_gb(" in line:
            window = " ".join(lines[i:i + 3])
            if "_requested_id" not in window:
                missing.append(line.strip())
    assert not missing, (
        "a sizing call without the requested id is sized by whichever "
        "same-path sibling registered the key: " + "; ".join(missing)
    )


def test_the_disambiguation_needs_the_requested_id_to_fire():
    """Why omitting it is silent rather than an error."""
    import inspect
    src = inspect.getsource(MultiModelManager._get_model_used_vram_gb)
    assert "if requested_id:" in src
    i = src.index("if requested_id:")
    assert "alias" in src[i:i + 400], (
        "the reliable discriminator is an entry whose alias IS the requested id"
    )


def test_siblings_are_sized_by_their_own_entry():
    m = MultiModelManager.__new__(MultiModelManager)
    m._measured_vram_gb = {}
    m.model_aliases = {}          # _config_for_model resolves aliases through it
    m.config = {
        "image:unsloth/Krea-2-Turbo": {
            "alias": "Krea-2-Turbo", "path": "unsloth/Krea-2-Turbo",
            "used_vram_gb": 14.0,
        },
        "image:Krea-2-RealDream": {
            "alias": "Krea-2-RealDream", "path": "unsloth/Krea-2-Turbo",
            "used_vram_gb": 19.0,
        },
    }
    turbo = m._get_model_used_vram_gb(
        "image:unsloth/Krea-2-Turbo", "unsloth/Krea-2-Turbo",
        requested_id="Krea-2-Turbo")
    dream = m._get_model_used_vram_gb(
        "image:unsloth/Krea-2-Turbo", "unsloth/Krea-2-Turbo",
        requested_id="Krea-2-RealDream")
    assert dream > turbo, (
        f"RealDream ({dream:.2f}) must not be sized as Turbo ({turbo:.2f}); "
        "the shared path is the same string for both"
    )


def test_the_alias_match_is_not_undone_by_the_path_fallback():
    """The bug under the bug: the right answer was found, then overwritten.

    The fallback loop looks its candidate up with _config_for_model, which
    resolves an alias through the alias map to the shared PATH -- landing on
    whichever sibling registered it. So on 2026-11-01 the log printed

        [vram-est] sizing 'Krea-2-RealDream' by its own config

    and the estimate was still Krea-2-Turbo's 9.68 GB.
    """
    import inspect
    src = inspect.getsource(MultiModelManager._get_model_used_vram_gb)
    assert "_sized_by_alias" in src
    i = src.index("for _cand in")
    assert "_sized_by_alias" in src[i:i + 80], (
        "the path fallback must be skipped once the alias settled it"
    )


def test_the_alias_entry_is_kept_when_it_is_already_current():
    """`_c is not cfg` must not leave _sized_by_alias false and re-open the loop."""
    m = MultiModelManager.__new__(MultiModelManager)
    m._measured_vram_gb = {}
    m.model_aliases = {}
    dream = {"alias": "Krea-2-RealDream", "path": "unsloth/Krea-2-Turbo",
             "used_vram_gb": 19.0}
    m.config = {
        "image:Krea-2-RealDream": dream,
        "image:unsloth/Krea-2-Turbo": {
            "alias": "Krea-2-Turbo", "path": "unsloth/Krea-2-Turbo",
            "used_vram_gb": 14.0},
    }
    got = m._get_model_used_vram_gb(
        "image:Krea-2-RealDream", "unsloth/Krea-2-Turbo",
        requested_id="Krea-2-RealDream")
    turbo = m._get_model_used_vram_gb(
        "image:unsloth/Krea-2-Turbo", "unsloth/Krea-2-Turbo",
        requested_id="Krea-2-Turbo")
    assert got > turbo, f"RealDream {got:.2f} must exceed Turbo {turbo:.2f}"
