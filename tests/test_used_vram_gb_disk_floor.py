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
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\0" * 2_000_000_000)
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
    i = MGR.index('_floor = self._disk_weights_gb(')
    block = MGR[i:i + 1200]
    assert '_floor > _explicit_val * 1.25' in block, (
        "the floor no longer requires the disk size to EXCEED the "
        "configured value, so it can shrink an operator's figure")


def test_the_margin_tolerates_an_honest_difference():
    """A configured value a little under the disk size is ordinary (weights vs
    checkpoint, tokenizer files, a shard counted twice). Only a gross mismatch
    is treated as a bad number."""
    i = MGR.index('_floor = self._disk_weights_gb(')
    assert '* 1.25' in MGR[i:i + 1200]


def test_it_says_what_it_did_and_why():
    i = MGR.index('_floor = self._disk_weights_gb(')
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
    assert '_apply_factors(_explicit_val)' in MGR[i:i + 200]


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
