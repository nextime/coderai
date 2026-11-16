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

"""An index means nothing unless both sides encoded the same dataset.

The orchestrator sends INDICES into its own ordering; the pod resolves them
against an ordering it built itself from the uploaded file. If the two
disagree in length, the indices refer to different examples -- and the pod
used to drop the out-of-range ones silently:

    mine = [_cache["examples"][i] for i in payload.get("indices") or []
            if i < len(_cache["examples"])]

So a pod whose encoding was short trained whatever survived the filter and
reported THAT as `examples_trained`, which is indistinguishable from a pod
that simply did less work. Measured on the first exchange that ever closed:
297 examples averaged where about 2,000 had been handed out, and nothing
anywhere said why -- because the pod's own count never left the pod.

Two halves, and both are needed. The pod refuses rather than filters, and it
reports its encoded count so this side can say which pod disagrees and by
how much. Trainer-pod stdout is not collected, so a message the pod only
prints is a message nobody reads.
"""

import inspect

from codai.api import loras


RUNNER = open("packaging/runpod/text_lora_pod_runner.py").read()

#: The runner with comments stripped. The old filter is QUOTED in the comment
#: that explains why it went -- checking the raw text for it therefore finds
#: the explanation and reports the bug as present, which is how this test
#: failed the first time it ran.
RUNNER_CODE = "\n".join(
    l for l in RUNNER.splitlines() if not l.lstrip().startswith("#"))


def test_the_pod_refuses_instead_of_filtering():
    assert "if i < len(_cache" not in RUNNER_CODE, (
        "the silent filter is back; an out-of-range index must be fatal")
    i = RUNNER.index("_out = [i for i in _want if i >= _have]")
    block = RUNNER[i:i + 900]
    assert "raise RuntimeError" in block
    assert "past the end of this pod's" in block
    # it names BOTH counts, or the message cannot be acted on
    assert "{_have} examples" in block
    assert "max(_want)" in block


def test_the_pod_reports_its_own_encoded_count():
    """The number that was missing. Pod stdout is not collected, so it has to
    travel in /status."""
    assert '"encoded": 0' in RUNNER
    assert "_set(encoded=len(enc))" in RUNNER
    i = RUNNER.index("_set(state=\"training\"")
    assert "encoded=_have" in RUNNER[i:i + 200]
    assert "asked=len(mine)" in RUNNER[i:i + 200]


def test_the_orchestrator_compares_the_counts():
    src = inspect.getsource(loras._text_train_distributed)
    assert 'status.get("encoded")' in src
    i = src.index('_penc = int(status.get("encoded") or 0)')
    block = src[i:i + 1200]
    assert "!= len(coord.encoded)" in block
    # once per shard, not once per poll
    assert "_warned_encode" in block
    assert "do not mean" in block


def test_the_disagreement_reaches_the_job_record():
    """A client polling progress must be able to see it, not just the log."""
    src = inspect.getsource(loras._text_train_distributed)
    i = src.index('_penc = int(status.get("encoded") or 0)')
    block = src[i:i + 1200]
    assert "last_error=" in block


def test_a_matching_count_says_nothing():
    """It must not warn on the normal case, or the warning is noise."""
    src = inspect.getsource(loras._text_train_distributed)
    i = src.index('_penc = int(status.get("encoded") or 0)')
    block = src[i:i + 300]
    # guarded on BOTH a non-zero report and an actual difference
    assert "if _penc and _penc != len(coord.encoded)" in block
