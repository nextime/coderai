"""A VRAM estimate came out three times under the truth, downward.

Measured on zeiss 2026-10-30, from the serving-failure watch:

    Auto: Qwen/Qwen3.5-9B is quantized — trying full-GPU regardless of the
          3.1 GB estimate (5.6 GB free)
    Error loading model Qwen/Qwen3.5-9B: … needs about 3.1 GB of VRAM to
          itself; 5.6 GB is free after evicting what could be evicted.

which reads like a bug in eviction — 5.6 is more than 3.1 — and is not. The
estimate was wrong, and wrong in the dangerous direction: too SMALL, so the
loader believed it fitted three times over, attempted a full-GPU load of a model
needing ~6.5 GB, and failed.

The chain, from the debug line:

    source=used_vram_gb -> 3.13 GB | used_vram_gb=7.795 measured_cfg=8.67
    load_in_4bit=True prequantized=False quant_mult=0.300

* `measured_vram_gb=8.67` (the real 4-bit footprint) was correctly DISCARDED:
  it was recorded under a different quantization signature. That guard exists
  because a stale measurement once forced CPU offload on a card the model fit.
* It fell back to `used_vram_gb=7.795`, which is a STORAGE-size baseline that
  the quant factors are applied on top of: 7.795 x 0.300 + reserve = 3.13.
* But the checkpoint is 19.31 GB of bf16 on disk with `quantization_config:
  None` — verified, so `prequantized=False` is right and the 0.300 multiplier
  is right. 7.795 was simply not the size of anything: the post-download
  backfill writes `cached_size * 1.2`, and 6.5 GB downloaded x 1.2 = 7.8.
  The figure described a partial download.

So the field can hold a number nobody checked, and the factors then scale it
down again. The checkpoint on disk is measurable fact, available at the moment
of the estimate, and is now used as a floor -- upward only, so a configured
value that is already larger stays the operator's.
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.models.manager import MultiModelManager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MGR = io.open(os.path.join(ROOT, 'codai/models/manager.py'), encoding='utf-8').read()


def _bare():
    """A manager without __init__: only the pure helper is exercised."""
    return MultiModelManager.__new__(MultiModelManager)


# ── the helper reads what is really on disk ──────────────────────────────────

def test_a_local_weights_file_is_measured(tmp_path):
    # Sparse: the helper reads st_size, so a 2 GB apparent file costs no disk.
    # Writing the bytes for real failed with OSError on a host whose root
    # filesystem was 98% full, which is a silly way for a size test to fail.
    f = tmp_path / "model.safetensors"
    with open(f, "wb") as fh:
        fh.truncate(2_000_000_000)
    gb = MultiModelManager._disk_weights_gb(_bare(), "m", "", {"path": str(f)})
    assert 1.9 < gb < 2.1, gb


def test_an_unknown_model_gets_no_opinion():
    """0.0 means "no opinion", which must leave the configured value alone --
    the alternative is a floor invented from nothing."""
    assert MultiModelManager._disk_weights_gb(
        _bare(), "nope/not-a-real-model-anywhere", "", {}) == 0.0


def test_it_never_raises(tmp_path):
    for cfg in ({}, {"path": None}, {"path": "/no/such/file.safetensors"},
                {"path": str(tmp_path)}, {"path": 12345}):
        assert MultiModelManager._disk_weights_gb(_bare(), "k", "", cfg) >= 0.0


def test_a_directory_is_not_mistaken_for_weights(tmp_path):
    assert MultiModelManager._disk_weights_gb(
        _bare(), "k", "", {"path": str(tmp_path)}) == 0.0


def test_an_unrecognised_extension_is_ignored(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_bytes(b"\0" * 1000)
    assert MultiModelManager._disk_weights_gb(
        _bare(), "k", "", {"path": str(f)}) == 0.0


# ── the floor, and its direction ─────────────────────────────────────────────

def test_the_floor_only_moves_the_estimate_up():
    """The field's purpose is that a configured number stays put. A floor that
    could lower it would break exactly that."""
    i = MGR.index('_floor = (self._disk_weights_gb(')
    block = MGR[i:i + 1200]
    assert '_floor > _explicit_val * 1.25' in block, (
        "the floor no longer requires the disk size to EXCEED the "
        "configured value, so it can shrink an operator's figure")


def test_the_margin_tolerates_an_honest_difference():
    """A configured value a little under the disk size is ordinary (weights vs
    checkpoint, tokenizer files, a shard counted twice). Only a gross mismatch
    is treated as a bad number."""
    i = MGR.index('_floor = (self._disk_weights_gb(')
    assert '* 1.25' in MGR[i:i + 1200]


def test_it_says_what_it_did_and_why():
    i = MGR.index('_floor = (self._disk_weights_gb(')
    block = MGR[i:i + 1600]
    assert 'well under the' in block and 'weights on disk' in block
    assert 'before the' in block and 'download finished' in block


def test_the_source_is_distinguishable_in_the_debug_line():
    """`source=` is how this subsystem is read; a floor that reported itself as
    an ordinary used_vram_gb estimate would be invisible."""
    assert '"used_vram_gb(disk-floor)"' in MGR


def test_the_factors_are_still_applied_to_the_floor():
    """The disk size is a STORAGE size: a 4-bit load of a 19 GB bf16 checkpoint
    needs ~5.8 GB, not 19. Skipping the factors would over-estimate by 3x and
    evict the whole card."""
    i = MGR.index('"used_vram_gb(disk-floor)"')
    assert '_apply_factors(_floor)' in MGR[i:i + 300]


def test_the_ordinary_path_is_unchanged():
    i = MGR.index('return _dbg_est("used_vram_gb",')
    # from_disk=False since 2026-10-31: the storage->load precision ratio
    # belongs to a baseline measured from the bytes on disk, and applying it
    # to a configured VRAM figure halved one that was already in the right
    # units (17.0 GB declared -> 2.75 GB estimated -> 5.64 GB measured).
    assert '_apply_factors(_explicit_val, from_disk=False)' in MGR[i:i + 240]


def test_the_measured_path_still_wins_and_still_skips_factors():
    """A real measurement already reflects quant/precision/offload. This change
    must not touch that, nor the signature guard that discards a measurement
    taken under different quantization."""
    assert 'Ignoring measured_vram_gb' in MGR
    assert '_dbg_est("measured"' in MGR


# ── the real case, if the checkpoint is present ──────────────────────────────

def test_the_incident_model_now_sizes_from_its_checkpoint():
    gb = MultiModelManager._disk_weights_gb(
        _bare(), "Qwen3.5-9B", "Qwen/Qwen3.5-9B", {"path": "Qwen/Qwen3.5-9B"})
    if not gb:
        pytest.skip("Qwen/Qwen3.5-9B is not in this host's HF cache")
    assert gb > 15.0, f"expected a ~19 GB bf16 checkpoint, got {gb:.2f}"
    assert gb > 7.795 * 1.25, "the floor would not have fired on the real case"
    # 4-bit of ~19 GB is ~5.8 GB, which is MORE than the 5.6 GB that was free --
    # so the load is refused or makes room, instead of being attempted and
    # failing on a 3.13 GB estimate.
    assert gb * 0.30 > 5.6 * 0.95


# ---------------------------------------------------------------------------
# NOT for diffusers pipelines: the repo on disk is not a VRAM lower bound
# ---------------------------------------------------------------------------
#
# Found the same night the floor was added, in production, by the floor itself
# misfiring. FLUX.2-klein-4B declares `used_vram_gb = 17.0` and really occupies
# 17.24 GB -- the eviction log states the measured figure -- while its cached
# repo is ~23.7 GB, because a diffusers cache holds every component and
# routinely duplicate formats and variants of which one subset loads. The floor
# raised the estimate to 25.61 GB, past the 24 GB card, so each load announced
# "this model needs more than the whole card", evicted the resident fleet and
# gained nothing:
#
#     [vram-est] key='image:black-forest-labs/FLUX.2-klein-4B'
#                source=used_vram_gb(disk-floor) -> 25.61 GB | used_vram_gb=17.0
#     Loading 'image:black-forest-labs/FLUX.2-klein-4B': need 25.6 GB VRAM,
#                have 0.0 GB free — evicting models to make room
#
# The two errors are not symmetric. The under-estimate this floor prevents made
# a text model attempt a load that could not fit. An over-estimate on an image
# pipeline costs OTHER models their residency, every time, and the model it was
# "protecting" was already sized correctly.

def test_the_floor_is_gated_to_single_file_models():
    i = MGR.index('_floor = (self._disk_weights_gb(')
    block = MGR[max(0, i - 1800):i + 400]
    assert '_single_file' in block
    assert '"image_models", "video_models"' in block
    assert 'if _single_file else 0.0' in block


def _gate_block() -> str:
    """The whole rationale for the gate, however long it grows.

    Measured from the comment that opens it rather than a fixed number of
    characters back from the code: a second exemption added later pushed the
    diffusers counter-example out of an 1800-character window, failing this
    test while the rationale it guards was still there, word for word.
    """
    i = MGR.index('_floor = (self._disk_weights_gb(')
    start = MGR.rindex('ONLY for a single-file', 0, i)
    return MGR[start:i]


def test_the_gate_records_why_a_repo_size_is_not_a_bound():
    block = _gate_block()
    assert 'diffusers' in block
    assert 'duplicate' in block or 'variants' in block
    assert '17.24' in block and '25.61' in block, (
        'the measured counter-example is what stops this being re-added')


def test_text_models_keep_the_floor():
    """The Qwen3.5-9B case this exists for must still be covered."""
    i = MGR.index('_floor = (self._disk_weights_gb(')
    block = MGR[max(0, i - 1800):i + 400]
    assert '("text", "vision")' in block


# ---------------------------------------------------------------------------
# NOR for one of two cards over the same weights file
# ---------------------------------------------------------------------------
#
# The second misfire, and the same shape as the first: the floor sized the
# SMALL member of a shared-weights pair by the file both of them point at.
#
#     [vram-est] sizing 'Wan2.2-I2V-A14B-Lightning' by its own config
#                (shares a weights file with 'Wan2.2-I2V-A14B-Diffusers')
#     used_vram_gb=22.000 GB for 'Wan-AI/Wan2.2-I2V-A14B-Diffusers' is well
#                under the 126.18 GB of weights on disk — sizing against the
#                disk instead
#
# i.e. the floor undid, in the next breath, the disambiguation the same
# function had just applied. Two entries over one checkpoint differ for real --
# lisa and lisa-32k are that 26B GGUF at n_ctx 178000 and 32768, measured
# 51.97 GB and 14.33 GB -- so for the smaller one the file on disk is not a
# bound at all. What the floor catches is a half-downloaded checkpoint, and no
# sibling entry declares a size for one of those.

def test_a_shared_weights_sibling_disables_the_floor():
    from codai.models.manager import multi_model_manager as m
    saved = dict(m.config or {})
    calls = []
    orig = m._disk_weights_gb
    try:
        m.config.clear()
        m.config.update({
            "big": {"path": "shared/repo", "used_vram_gb": 151.42, "alias": None},
            "small": {"path": "shared/repo", "used_vram_gb": 22.0,
                      "alias": "accelerated"},
        })
        m._disk_weights_gb = lambda *a, **k: (calls.append(a) or 126.18)
        small = m._get_model_used_vram_gb("shared/repo", requested_id="accelerated")
    finally:
        m._disk_weights_gb = orig
        m.config.clear()
        m.config.update(saved)
    # 22 GB plus whatever reserve the estimator adds -- nowhere near the 126 GB
    # checkpoint, and nowhere near the 151 GB its sibling declares.
    assert small < 126.18 / 2, (
        f"the small card was sized by the shared checkpoint ({small:.2f} GB)")
    assert small > 22.0, f"the configured figure was lost entirely ({small:.2f})"


def test_the_exemption_needs_a_declared_size_not_just_a_shared_path():
    """A sibling that declares nothing cannot be the one saying the file is not
    a bound — a half-downloaded checkpoint is exactly a path with no second
    opinion on it."""
    block = _gate_block() + MGR[MGR.index('_floor = (self._disk_weights_gb('):][:200]
    i = MGR.index("_mypath = str((cfg.get('path')")
    gate = MGR[i:i + 1200]
    assert "used_vram_gb" in gate, "the sibling's declared size is not consulted"
    assert "_single_file = False" in gate
