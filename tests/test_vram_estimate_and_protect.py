"""A 7B embedder was admitted on 2.75 GB, took 5.6 GB, and OOMed on its first batch.

Measured on zeiss 2026-10-31, during a customer test run, with the card already
holding a 26B GGUF in the sibling engine:

    [vram-est] key='embedding:Alibaba-NLP/gme-Qwen2-VL-7B-Instruct'
               source=used_vram_gb -> 2.75 GB | used_vram_gb=17.0
               load_in_4bit=True quant_mult=0.300 prec(load2/disk4)=0.50
    Measured footprint for '…gme-Qwen2-VL-7B-Instruct': 5.64 GB total
    [embeddings] encode OOM (CUDA out of memory. Tried to allocate 1.02 GiB.
               GPU 0 has a total capacity of 23.56 GiB of which 954 …)
    Active model '…gme-Qwen2-VL-7B-Instruct' is busy — waiting for it to go
               idle before eviction…

Two defects, one after the other.

`used_vram_gb` is a VRAM figure, and by this file's own convention an
UNQUANTIZED one -- 17.0 x 0.300 for 4-bit = 5.1 GB, which is what the loader
then measured. The storage->load precision ratio was applied on top of it
because the checkpoint's config.json declares float32, halving a number that
was already in the right units. That ratio belongs to a baseline derived from
the bytes on disk, nowhere else.

Then the recovery: the eviction pass called to free scratch space for the
resident model fell through to its "evict the active model" fallback, which IS
that model, and waited for it to go idle. It cannot go idle -- it is blocked on
the memory the pass was called to free -- so the budget was spent on a model it
must not touch while 17.8 GB sat idle in the sibling engine, and the
cross-engine releaser that would have asked for it runs only afterwards.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.models.manager import MultiModelManager, multi_model_manager as MGR


# ── the estimate ─────────────────────────────────────────────────────────────

def _estimate(cfg: dict, key="embedding:x", path="org/model-7B"):
    saved = dict(MGR.config or {})
    try:
        MGR.config.clear()
        MGR.config.update({path: dict(cfg, path=path)})
        return MGR._get_model_used_vram_gb(path)
    finally:
        MGR.config.clear()
        MGR.config.update(saved)


def test_a_configured_vram_figure_is_not_halved_by_the_disk_dtype():
    """The incident: 17.0 GB declared, 4-bit asked for, fp32 on disk."""
    gb = _estimate({"used_vram_gb": 17.0, "load_in_4bit": True,
                    "precision": "bf16"})
    # 17.0 x 0.30 = 5.1 plus the runtime reserve. The bug produced 2.75.
    assert gb > 4.5, f"estimate {gb:.2f} GB is the halved figure again"
    assert gb < 9.0, f"estimate {gb:.2f} GB looks like no quantization at all"


def test_the_quantization_multiplier_still_applies_to_it():
    """It is the UNQUANTIZED footprint, so 4-bit must still shrink it --
    the opposite error, measured earlier the same night, turned 7.0 into 1.13."""
    full = _estimate({"used_vram_gb": 17.0})
    quant = _estimate({"used_vram_gb": 17.0, "load_in_4bit": True})
    assert quant < full / 2, f"4-bit did not reduce the figure ({quant} vs {full})"


def test_the_precision_ratio_still_applies_to_a_disk_derived_baseline():
    """Where the baseline IS the bytes on disk, normalizing storage precision to
    load precision is exactly right; this only stops it reaching a figure that
    was never about disk."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "codai/models/manager.py"),
        encoding="utf-8").read()
    i = src.index("def _apply_factors(")
    assert "from_disk: bool = True" in src[i:i + 200]
    body = src[i:i + 1800]
    assert "if from_disk:" in body and "prec_factor" in body
    # The two disk paths (a local weights file, an HF cache scan) keep it.
    assert src.count("_apply_factors(weights_gb)") == 2


def test_only_the_configured_path_opts_out():
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "codai/models/manager.py"),
        encoding="utf-8").read()
    assert src.count("from_disk=False") == 1
    i = src.index("from_disk=False")
    assert "_explicit_val" in src[i - 120:i + 40]


# ── the eviction that waited for its own caller ──────────────────────────────

def test_the_protected_key_is_never_waited_on():
    m = MultiModelManager.__new__(MultiModelManager)
    import threading
    m._cv = threading.Condition()
    m.models = {"a": object(), "keep": object()}
    m._keep_resident = set()
    m._loading_reservations = {}
    m._last_used = {}
    m._measured_vram_gb = {}
    m._model_sizes = {}
    m.config = {}
    m._offload_cfg = None
    m.model_aliases = {}
    m._active_in_vram = "keep"
    m._external_vram_releasers = []
    waited = []
    freed = []
    m._lru_order = lambda: ["keep", "a"]
    m._is_key_busy = lambda k: k == "keep"
    m._wait_until_idle = lambda k, *a, **kw: waited.append(k) or False
    m._get_free_vram_gb = lambda *a, **kw: 0.0
    m._get_total_vram_gb = lambda: 24.0
    m._total_vram_gb = lambda: 24.0
    m._evict_one = lambda k, reason="": freed.append(k)
    m._size_label = lambda k: "x"
    # Deliberately NOT skipped on an AttributeError: a collaborator this needs
    # and does not have is something to add here, not something to step over.
    m._evict_models_for_vram(2.0, protect="keep")
    assert "keep" not in waited, "waited for the model the room was for"
    assert "keep" not in freed, "evicted the model the room was for"


def test_the_source_never_waits_on_the_protected_model():
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "codai/models/manager.py"),
        encoding="utf-8").read()
    i = src.index("def _evict_models_for_vram(")
    body = src[i:i + 6000]
    assert "protect: str = \"\"" in body
    # Pass 1 skips it outright…
    assert "if protect and key == protect:" in body
    # …and the active-model fallback does not fire for it.
    assert "self.active_in_vram != protect" in body


def test_the_embedding_retry_protects_the_model_it_is_making_room_for():
    emb = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "codai/api/embeddings.py"),
        encoding="utf-8").read()
    i = emb.index("encode OOM")
    body = emb[i:i + 1400]
    assert "protect=model_key" in body
