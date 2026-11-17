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

"""A pod says `done` until something starts it working, dispatch included.

`POST /round` starts the pod's worker in a thread and answers 202 without
touching the pod's state. So for a moment after dispatch `/status` still
carries the PREVIOUS block's terminal state: `done`, the same round number,
and that block's example count. The orchestrator's done-branch matches on
exactly those three, and by then the shard holds its NEXT block -- so it
fetched the same adapter again and closed a block no gradient step had seen.

Gating the branch on holding a block closed the handout half of this. This is
the rest of it, and it is closed from both ends:

- the pod sets a state synchronously in the handler, so the window does not
  exist for a pod running a current image;
- the orchestrator refuses to collect from a shard it has just dispatched to
  until it has SEEN that shard working, so the window is also closed for a
  pod running the image already in production.

Stalling is an acceptable failure here and miscounting is not: a shard that
never reports working is simply absent from its exchange, which the round
deadline and `min_fraction` already handle, while a miscount is paid for in
full and reported as success.
"""

import inspect

import pytest

from codai.api import loras


# --------------------------------------------------------------------- #
# the orchestrator's half
# --------------------------------------------------------------------- #

def _src():
    return inspect.getsource(loras._text_train_distributed)


def test_a_shard_is_marked_on_dispatch():
    src = _src()
    assert "_dispatched.add(shard)" in src, (
        "without this the stale `done` from before the dispatch is collected "
        "and closes the block that was just handed out")
    # Right where the work is handed out, not somewhere hopeful.
    assert src.index("flt.start_round(pod, _payload)") < \
        src.index("_dispatched.add(shard)")


def test_the_done_branch_refuses_a_just_dispatched_shard():
    src = _src()
    branch = src.split('if state == "done"')[1][:300]
    assert "_dispatched" in branch, branch


def test_seeing_the_pod_work_clears_the_mark():
    """Otherwise the first collection is the only one a pod ever gets, and a
    fast pod contributing twice to an exchange is the whole design."""
    src = _src()
    assert "_dispatched.discard(shard)" in src
    working = src.split('if state in (')[1][:900]
    assert "discard" in working, working


def test_the_synchronous_pod_state_is_treated_as_working():
    """The pod-side fix names a new state; the orchestrator has to know it."""
    src = _src()
    working = src.split('if state in (')[1][:200]
    assert '"starting"' in working, working


# --------------------------------------------------------------------- #
# the pod's half
# --------------------------------------------------------------------- #

def _pod_src():
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "packaging", "runpod",
                        "text_lora_pod_runner.py")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_the_round_handler_sets_a_state_before_it_answers():
    """The window is the gap between the 202 and the worker's first `_set`."""
    src = _pod_src()
    handler = src.split('if self.path == "/round":')[1].split(
        "self._json(404")[0]
    assert "_set(" in handler, (
        "answering 202 without touching the state leaves `/status` reporting "
        "the previous block as done")
    assert handler.index("_set(") < handler.index("_worker.start()"), \
        "it has to be set BEFORE the thread races the next poll"


def test_the_handler_claims_the_new_round_number():
    """A state that kept the old round number would still match the branch."""
    src = _pod_src()
    handler = src.split('if self.path == "/round":')[1].split(
        "self._json(404")[0]
    assert "round=" in handler, handler


def test_a_busy_pod_is_still_refused():
    """Two rounds in flight hand back an adapter from the wrong one."""
    src = _pod_src()
    handler = src.split('if self.path == "/round":')[1].split(
        "self._json(404")[0]
    assert "409" in handler
    assert "starting" in handler.split("409")[0], \
        "the new state has to count as busy too, or two rounds can start"
