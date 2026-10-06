"""Sizing a LongCat load by what it keeps on the card, not by its directory.

The generic estimator sums a model's whole HF cache entry. LongCat's entry holds the
bf16 DiT (51 GB, stored fp32), the INT8 DiT built beside it (13.6 GB) and an fp32
UMT5-XXL text encoder (22 GB) — so it asked a 24 GB card for 74.3 GB and evicted every
other model on the box before each generation. Only one DiT loads, each component is
cast from its own stored dtype, and under offload the DiT and encoder take turns.
"""
import json
import pathlib
import struct

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def M():
    from codai.models.manager import multi_model_manager
    return multi_model_manager


def _safetensors(path, dtype, elems):
    """A header-only safetensors file: the estimator reads the header, not the data."""
    header = {"w": {"dtype": dtype, "shape": [elems], "data_offsets": [0, 0]}}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def _checkpoint(tmp_path, dit_gb=4.0, int8_gb=1.0, enc_gb=2.0, vae_gb=0.1):
    """A cache layout like the real one, with file SIZES standing in for weights."""
    snap = tmp_path / "hub" / "models--meituan-longcat--LongCat-Video" / "snapshots" / "r1"
    for sub, gb, dtype in (("dit", dit_gb, "F32"), ("base_model_int8", int8_gb, "I8"),
                           ("text_encoder", enc_gb, "F32"), ("vae", vae_gb, "F32")):
        d = snap / sub
        d.mkdir(parents=True)
        _safetensors(d / "model.safetensors", dtype, 8)
        # The size on disk is what gets scaled; pad a sibling to the target.
        (d / "weights.bin").write_bytes(b"")
        with open(d / "weights.bin", "wb") as fh:
            fh.truncate(int(gb * 1e9))
    return snap


@pytest.fixture
def cfg_for(tmp_path, monkeypatch):
    _checkpoint(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    for clear in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(clear, raising=False)

    def make(**over):
        base = {"backend": "longcat", "path": "meituan-longcat/LongCat-Video"}
        base.update(over)
        return base
    return make


def test_a_non_longcat_model_is_left_to_the_generic_path(M, cfg_for):
    assert M._longcat_resident_gb(cfg_for(backend="diffusers"), "video:x", "", 2.0) == 0.0


def test_the_int8_dit_is_not_scaled(M, cfg_for):
    """It is already at its load size on disk. Scaling it by load/storage would double
    it, and the whole point of the variant is that it fits."""
    gb = M._longcat_resident_gb(cfg_for(variant="int8", offload_strategy="model"),
                                "video:x", "", 2.0)
    # max(int8 1.0, encoder 2.0 * 2/4 = 1.0) + vae 0.05
    assert gb == pytest.approx(1.05, abs=0.02)


def test_the_bf16_dit_is_cast_down_from_fp32_storage(M, cfg_for):
    """4 GB of fp32 on disk is 2 GB of bf16 on the card."""
    gb = M._longcat_resident_gb(cfg_for(variant="bf16", offload_strategy="model"),
                                "video:x", "", 2.0)
    # max(dit 4*0.5 = 2.0, encoder 1.0) + vae 0.05
    assert gb == pytest.approx(2.05, abs=0.02)


def test_only_the_chosen_dit_is_counted(M, cfg_for):
    """Both DiTs sit in the same snapshot; summing them is most of the 74 GB error."""
    int8 = M._longcat_resident_gb(cfg_for(variant="int8"), "video:x", "", 2.0)
    bf16 = M._longcat_resident_gb(cfg_for(variant="bf16"), "video:x", "", 2.0)
    assert int8 < bf16
    # int8 1.0 + encoder 1.0 + vae 0.05 — the bf16 DiT's 4 GB is absent
    assert int8 == pytest.approx(2.05, abs=0.02)


def test_offload_makes_the_peak_a_max_not_a_sum(M, cfg_for):
    """They take turns on the card, so the encoder is not added to the DiT."""
    with_off = M._longcat_resident_gb(cfg_for(variant="bf16", offload_strategy="model"),
                                      "video:x", "", 2.0)
    resident = M._longcat_resident_gb(cfg_for(variant="bf16"), "video:x", "", 2.0)
    assert with_off < resident
    assert resident == pytest.approx(with_off + 1.0, abs=0.02)   # + the encoder


def test_offload_none_is_treated_as_no_offload(M, cfg_for):
    explicit = M._longcat_resident_gb(cfg_for(variant="int8", offload_strategy="none"),
                                      "video:x", "", 2.0)
    unset = M._longcat_resident_gb(cfg_for(variant="int8"), "video:x", "", 2.0)
    assert explicit == unset


def test_a_missing_checkpoint_falls_back_to_the_generic_path(M, tmp_path, monkeypatch):
    """0.0 means "I cannot size this" — it must not claim a model needs nothing."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    cfg = {"backend": "longcat", "path": "nobody/nothing"}
    assert M._longcat_resident_gb(cfg, "video:x", "", 2.0) == 0.0


@pytest.mark.parametrize("dtype,width", [("F32", 4.0), ("BF16", 2.0), ("F16", 2.0),
                                         ("I8", 1.0), ("F8_E4M3", 1.0)])
def test_stored_dtype_widths_are_read_from_the_header(M, tmp_path, dtype, width):
    _safetensors(tmp_path / "model.safetensors", dtype, 16)
    assert M._safetensors_storage_bpe(str(tmp_path)) == width


def test_a_folder_with_no_safetensors_reports_unknown(M, tmp_path):
    assert M._safetensors_storage_bpe(str(tmp_path)) == 0.0
