"""Eviction frees 10% more than asked, so an estimate cannot land on the edge.

Every size that reaches eviction is an estimate. One that lands a few hundred MB
short does not fail honestly — the load is admitted, starts allocating, and dies
on a tiny buffer. Measured on zeiss across 2026-10-30/31, each on a load that
eviction had just reported as satisfied:

    CUDA out of memory. Tried to allocate  20.00 MiB … 8.69 MiB is free
    CUDA out of memory. Tried to allocate 190.00 MiB … 18.69 MiB is free
    CUDA out of memory. Tried to allocate  12.00 MiB … 8.69 MiB is free
    CUDA out of memory. Tried to allocate   9.00 GiB … 4.08 GiB is free

The last of those is a genuinely too-large request; the first three are the edge.
10% of the request costs at most one more idle model and puts the edge out of
reach.

The headroom is what eviction TARGETS, never what it reports. The shortfall
warning measures against what the caller asked for, so freeing enough for the
load but not the whole headroom is a success and stays silent — otherwise every
successful eviction would print a warning, and the one real warning would be
lost among them.
"""

import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MGR = io.open(os.path.join(ROOT, 'codai/models/manager.py'), encoding='utf-8').read()


def _fn():
    # Matched without the rest of the signature: `protect` was added later,
    # and an anchor that pins the whole parameter list fails on a change
    # that has nothing to do with the headroom this file is about.
    i = MGR.index('    def _evict_models_for_vram(self, needed_gb: float')
    return MGR[i:MGR.index('\n    def ', i + 10)]


def test_the_headroom_fraction_is_ten_percent():
    from codai.models.manager import _EVICT_HEADROOM_FRACTION
    assert _EVICT_HEADROOM_FRACTION == 0.10


def test_the_target_is_the_request_plus_headroom():
    """The headroom is now RESOLVED rather than computed inline, because it is
    configurable per model: a VAE decode asks for about 1 GB in one allocation
    whatever the checkpoint weighs, and a tenth of a 7.49 GB model is not that
    (measured 2026-11-01, CyberRealistic-Pony, 110 MiB free at the decode).
    The shape of the target is unchanged: the request plus the headroom."""
    body = _fn()
    assert 'target_gb = needed_gb + _headroom' in body
    assert '_headroom = self._evict_headroom_gb(needed_gb, for_key or protect)' \
        in body, "the per-model resolver is what makes it configurable"


def test_every_enough_yet_check_uses_the_padded_target():
    """A pass still comparing against the bare request would stop early and
    leave the edge case exactly where it was."""
    body = _fn()
    checks = re.findall(r'_get_free_vram_gb\(\)\s*(?:>=|<)\s*(\w+)', body)
    assert checks, 'no free-VRAM comparisons found — did the function move?'
    assert all(c == 'target_gb' for c in checks), checks


def test_the_shortfall_warning_still_measures_the_request():
    """Warning on the padded target would fire on every successful eviction."""
    body = _fn()
    i = body.index('_free_after = self._get_free_vram_gb()')
    assert 'if _free_after < needed_gb:' in body[i:i + 200]


def test_the_whole_card_verdict_still_uses_the_request():
    """'needs more than the whole card' must compare the real requirement; with
    the headroom a model needing exactly the card's size would be mislabelled."""
    body = _fn()
    assert 'needed_gb > _total_gb' in body


def test_a_zero_or_negative_request_still_returns_early():
    body = _fn()
    assert body.index('if needed_gb <= 0:') < body.index('target_gb =')


def test_the_reason_is_recorded_with_its_evidence():
    body = _fn()
    assert '20.00 MiB' in body and '8.69 MiB' in body, (
        'the measurements that justify the margin are what stop it being '
        'tuned away as arbitrary')


def test_the_arithmetic():
    """10% of the request, not of the card and not a flat constant: a flat
    margin is too small for a 30 GB model and too large for a 0.1 GB one."""
    from codai.models.manager import _EVICT_HEADROOM_FRACTION as H
    for need, want in ((10.0, 11.0), (2.62, 2.882), (0.1, 0.11), (30.9, 33.99)):
        assert abs(need * (1.0 + H) - want) < 0.001, (need, want)


def test_the_fraction_is_still_the_floorless_default():
    """A model that configures nothing gets exactly what it got before."""
    from codai.models import manager as M
    m = M.MultiModelManager.__new__(M.MultiModelManager)
    m.config = {}
    assert m._evict_headroom_gb(10.0) == 10.0 * M._EVICT_HEADROOM_FRACTION
