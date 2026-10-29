# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Whether a model fits on the GPU is learned once, not rediscovered.

Sizing knows what the weights weigh. Whether every component plus the
activation peak fits AT ONCE is only answered by trying.

Measured 2026-10-29: LTX-2.5 was sized at 22.1 GB against 25.0 GB free, chose
a resident load, and OOMed on transformer, vocoder and prompt_enhancer. Its
real footprint is 23.28 GB and resident needs all of it simultaneously. The
attempt, the OOM and the teardown took ~16 minutes to arrive at exactly the
CPU-offload placement that would otherwise have been chosen immediately -- and
without remembering, every single load pays that again.

Unknown must still mean try: a model never attempted, or a card that has since
changed, should get its chance at the GPU rather than being written off.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Mgr:
    """The two methods under test, on the real class."""

    def __init__(self, config):
        self.config = config
        self.model_aliases = {}
        self.model_registered_types = {}

    @staticmethod
    def _bind():
        from codai.models.manager import MultiModelManager
        for m in ("record_resident_fits", "resident_known_not_to_fit",
                  "_config_for_model_key", "_resolve_config_key",
                  "_config_for_model", "_registered_types_for"):
            setattr(_Mgr, m, getattr(MultiModelManager, m))


_Mgr._bind()


def _mgr(entry=None):
    cfg = {"video:LTX-2.5": dict(entry or {})}
    return _Mgr(cfg)


def test_an_unattempted_model_is_still_tried():
    """No verdict recorded = try the card."""
    m = _mgr()
    assert m.resident_known_not_to_fit("video:LTX-2.5") is False


def test_a_proven_failure_is_remembered():
    m = _mgr()
    m.record_resident_fits("video:LTX-2.5", False)
    assert m.config["video:LTX-2.5"]["measured_resident_fits"] is False
    assert m.resident_known_not_to_fit("video:LTX-2.5") is True


def test_a_model_that_fits_is_never_doubted():
    m = _mgr()
    m.record_resident_fits("video:LTX-2.5", True)
    assert m.resident_known_not_to_fit("video:LTX-2.5") is False


def test_a_later_success_overrides_an_earlier_failure():
    """A bigger card, a smaller quantisation, fewer co-resident models -- the
    verdict must not be a one-way door."""
    m = _mgr()
    m.record_resident_fits("video:LTX-2.5", False)
    assert m.resident_known_not_to_fit("video:LTX-2.5") is True
    m.record_resident_fits("video:LTX-2.5", True)
    assert m.resident_known_not_to_fit("video:LTX-2.5") is False


def test_an_unknown_model_does_not_raise():
    m = _mgr()
    assert m.resident_known_not_to_fit("video:never-seen") is False
    m.record_resident_fits("video:never-seen", False)   # must not raise


def test_the_loader_consults_and_records():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/api/video.py"), encoding="utf-8").read()
    assert "_resident_known_bad(model_name)" in src, (
        "the cache-hit path no longer consults what was learned, so a model "
        "proven not to fit pays the failed attempt on every load")
    assert src.count("_record_resident_fits(model_name,") >= 2, (
        "both outcomes must be recorded: writing only the failure means a "
        "model that fits is re-proven every time, and writing only the success "
        "never stops the expensive retry")
