"""The image path's eviction must be reachable, and must ask for the peak.

Two facts, both measured on 2026-11-01 from one failure: CyberRealistic-Pony
loaded onto a card still holding a 17.22 GB Krea-2-RealDream and died
mid-generation asking for 96 MiB with 50 MiB free.

1. The proactive-eviction block sat BELOW the ``--no-ram`` branch, which loads
   and returns. On any install running --no-ram -- the mode that also refuses
   to offload, so eviction is its only remedy -- the block was dead code. No
   eviction line appears anywhere in that run's log.

2. Admission compared free VRAM against the WEIGHTS: 8.6 GB needed, 9.0 GB
   free, "no eviction needed". What fitted was the weights; the activations,
   cuBLAS workspace and VAE decode had 0.4 GB between them.
"""
import inspect

import codai.api.images as images
from codai.models.manager import multi_model_manager


def _loader_source():
    src = inspect.getsource(images)
    i = src.index("# Proactive VRAM eviction before first load attempt")
    return src, i


def test_eviction_is_reachable_in_no_ram_mode():
    src, evict_at = _loader_source()
    no_ram_at = src.index("# --no-ram mode: load directly on GPU")
    assert evict_at < no_ram_at, (
        "the proactive-eviction block must come BEFORE the --no-ram branch; "
        "that branch returns, so below it the block never runs"
    )


def test_admission_asks_for_weights_plus_headroom():
    src, evict_at = _loader_source()
    end = src.index("# Standard loading path (with OOM fallback)")
    block = src[evict_at:end]
    assert "_evict_headroom_gb" in block, (
        "admission must add the generation headroom to the weights before "
        "deciding; comparing weights to free VRAM admits a model whose "
        "weights fit and whose decode does not"
    )
    assert "_free_gb < _want_gb" in block, (
        "the comparison must be against weights+headroom, not the weights"
    )


def test_the_headroom_would_have_evicted_realdream():
    """The arithmetic of the measured failure, through the real helper."""
    need, free = 8.6, 9.0
    head = multi_model_manager._evict_headroom_gb(need, key="")
    assert head > 0.0
    assert free >= need, "the weights did fit -- that is why it was admitted"
    assert free < need + head, (
        "weights+headroom must NOT fit in 9.0 GB, or admission would still "
        "wave this load through"
    )
