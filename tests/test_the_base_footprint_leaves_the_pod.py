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

"""What the card is holding has to reach the orchestrator, not the pod's log.

A training pod's stdout is collected NOWHERE: there is no public pod-log API
and the orchestrator never reads it. So the fact that decides whether a fleet
can train at all -- did the base actually quantize, did it fit on the card --
was written for nobody.

It matters because the training loop turns an OutOfMemoryError into a SKIPPED
example. A base that silently stayed in full precision, or that
`device_map="auto"` spilled onto the CPU, OOMs on every long example and the
pod still answers `done`. Measured: a fleet trained 4 of every 16 examples it
was asked for and reported 100%, and the cause could not be established from
outside the pod at all.

The whole chain is tested here, because it broke in the middle twice: the
trainer collected it, the pod's progress callback dropped every field but
`step`, and the orchestrator never looked.
"""

import inspect
import os
import types

import pytest

from codai.api import loras
from codai.api import text_lora_trainer as tlt


def _runner_src():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "packaging", "runpod",
                           "text_lora_pod_runner.py"), encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------- #
# 1. the trainer collects it, as data
# --------------------------------------------------------------------- #

class _P:
    def __init__(self, dtype, device, n=1000, size=1):
        self.dtype = dtype
        self.device = device
        self._n = n
        self._s = size

    def numel(self):
        return self._n

    def element_size(self):
        return self._s


class _M:
    def __init__(self, params):
        self._p = params

    def parameters(self):
        return self._p


def test_a_quantized_base_is_reported_as_quantized():
    """bnb keeps 4-bit weights as uint8 blocks, so the dtype is the tell."""
    got = tlt.describe_loaded_base(_M([_P("torch.uint8", "cuda:0")]))
    assert got["quantized_4bit"] is True
    assert "4-bit=yes" in got["line"]
    assert got["cpu_params"] == 0


def test_an_unquantized_base_says_so_in_capitals():
    """This is the line that ends a run early instead of four hours late."""
    got = tlt.describe_loaded_base(_M([_P("torch.bfloat16", "cuda:0")]))
    assert got["quantized_4bit"] is False
    assert "NOT QUANTIZED" in got["line"], got["line"]


def test_parameters_on_the_cpu_are_counted():
    """`device_map="auto"` spills rather than failing, which reads as success."""
    got = tlt.describe_loaded_base(_M([
        _P("torch.uint8", "cuda:0"), _P("torch.uint8", "cpu"),
        _P("torch.uint8", "cpu")]))
    assert got["cpu_params"] == 2
    assert got["devices"]["cpu"] == 2


def test_a_base_wholly_on_one_device_counts_no_spill():
    """One device is not a spill, however that device is named."""
    got = tlt.describe_loaded_base(_M([_P("torch.uint8", "cpu")] * 3))
    assert got["cpu_params"] == 0, "nothing was split, so nothing spilled"


def test_the_size_is_measured_in_bytes():
    got = tlt.describe_loaded_base(_M([
        _P("torch.uint8", "cuda:0", n=2_000_000_000, size=1)]))
    assert got["param_bytes"] == 2_000_000_000
    assert got["param_gb"] == 2.0


# --------------------------------------------------------------------- #
# 2. the trainer hands it out before the first step
# --------------------------------------------------------------------- #

def test_it_is_reported_through_the_progress_callback():
    """Before the first step: if every example OOMs the run still ends, but
    if the pod wedges the return value never arrives -- and that is exactly
    when the footprint is what one wants."""
    src = inspect.getsource(tlt.run_training)
    assert "describe_loaded_base(model)" in src
    assert "base=_footprint" in src
    assert src.index("describe_loaded_base(model)") < src.index("for i, idx in"), \
        "it must be reported before training, not after"


def test_it_is_also_in_the_return_value():
    src = inspect.getsource(tlt.run_training)
    assert '"base": _footprint' in src
    assert '"asked": len(encoded)' in src, \
        "asked beside trained, so the ratio needs no second source"


def test_the_report_failing_does_not_stop_the_run():
    src = inspect.getsource(tlt.run_training)
    tail = src.split("describe_loaded_base(model)")[1][:400]
    assert "except Exception" in tail, \
        "instrumentation must never be why a run refuses to start"


# --------------------------------------------------------------------- #
# 3. the pod passes it on instead of dropping it
# --------------------------------------------------------------------- #

def test_the_progress_callback_keeps_more_than_step():
    """It was `lambda **kw: _set(step=...)`, which silently discarded the
    footprint the moment the trainer started sending it."""
    src = _runner_src()
    assert 'on_progress=lambda **kw: _set(step=' not in src, \
        "the lambda dropped every field but step"
    assert 'fields["base"] = kw["base"]' in src, src[:0] or "no base passthrough"


def test_the_done_state_carries_it_too():
    src = _runner_src()
    done = src.split('_set(state="done"')[1][:300]
    assert "base=" in done, done


def test_status_declares_the_field():
    src = _runner_src()
    assert '"base": {}' in src, "a missing default makes the field absent"
    assert "base, error}" in src, "the documented surface has to match"


# --------------------------------------------------------------------- #
# 4. and the orchestrator actually reads it
# --------------------------------------------------------------------- #

def test_the_orchestrator_logs_the_footprint():
    """`/status` is only a channel if something listens on it. The pod has
    always reported `asked` and nothing ever compared it."""
    src = inspect.getsource(loras._text_train_distributed)
    assert 'status.get("base")' in src
    assert "_said_base" in src


def test_it_is_said_once_per_shard_not_once_per_poll():
    """A 20-second poll over a four-hour run would print it 720 times."""
    src = inspect.getsource(loras._text_train_distributed)
    assert "_said_base.add(shard)" in src
    assert "shard not in _said_base" in src


def test_an_unquantized_base_is_called_out_by_the_orchestrator():
    src = inspect.getsource(loras._text_train_distributed)
    assert 'quantized_4bit") is False' in src, \
        "a falsy check would also fire when the pod never reported it"
    assert "UNQUANTIZED" in src


def test_a_cpu_spill_is_called_out_by_the_orchestrator():
    src = inspect.getsource(loras._text_train_distributed)
    assert 'cpu_params' in src
    assert "did not fit on the card" in src


def test_the_loaded_phase_counts_as_working():
    """The pod reports `loaded` between the base landing and the first step;
    an unknown state there would be read as a stale `done`."""
    src = inspect.getsource(loras._text_train_distributed)
    working = src.split("if state in (")[1][:260]
    assert '"loaded"' in working, working
