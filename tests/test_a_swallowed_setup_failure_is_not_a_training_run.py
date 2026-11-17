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

"""Gradient checkpointing failing quietly produces a run that trains nothing.

`model.gradient_checkpointing_enable()` and `enable_input_require_grads()` sat
in a bare `except Exception: pass`. Both failure modes are silent and neither
is survivable:

- without checkpointing, a 27B base at the 8192-token cap cannot hold its own
  activations on one 48GB card, so every long example raises OOM -- and the
  training loop SKIPS an OOM rather than raising it, so the round reports
  "done" having trained whatever short examples happened to fit;
- without `enable_input_require_grads`, the checkpointed segments take no
  gradient at their input. That costs no memory at all and simply trains
  nothing.

Either way the pod answers `state=done` and the operator can only infer the
problem from the averaging weights, which is how two 16-pod runs were paid for
before anybody could say why they were undertrained.
"""

import inspect
import sys
import types

import pytest

from codai.api import text_lora_trainer as tlt


# --------------------------------------------------------------------- #
# driven, not read: the real `load_base_and_adapter` against fake libs
# --------------------------------------------------------------------- #

class _Model:
    """A base model that fails to enable whichever half the test chooses."""

    def __init__(self, *, fail=""):
        self._fail = fail
        self.config = types.SimpleNamespace(use_cache=True)
        self.enabled = []

    def gradient_checkpointing_enable(self, **kw):
        if self._fail == "checkpointing":
            raise ValueError("this model does not support checkpointing")
        self.enabled.append("checkpointing")

    def enable_input_require_grads(self):
        if self._fail == "require_grads":
            raise AttributeError("no input embeddings to hook")
        self.enabled.append("require_grads")

    def named_modules(self):
        return []

    def parameters(self):
        return []

    def print_trainable_parameters(self):
        pass


@pytest.fixture()
def fake_libs(monkeypatch):
    """Stub transformers/peft/torch just enough to reach the setup calls."""
    made = {}

    def _from_pretrained(base, **kw):
        return made["model"]

    tf = types.ModuleType("transformers")
    tf.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=_from_pretrained)
    peft = types.ModuleType("peft")
    peft.get_peft_model = lambda m, c: m
    peft.prepare_model_for_kbit_training = lambda m: m
    peft.LoraConfig = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setattr(tlt, "bnb_config", lambda s: None)
    monkeypatch.setattr(tlt, "linear_module_names", lambda m: ["q_proj"])
    monkeypatch.setattr(tlt, "select_target_modules", lambda l: ["q_proj"])
    monkeypatch.setattr(tlt, "describe_selection", lambda l, t: "1 module")
    monkeypatch.setattr(tlt, "lora_config", lambda s, t: object())
    return made


def _load(fake_libs, *, fail=""):
    fake_libs["model"] = _Model(fail=fail)
    return tlt.load_base_and_adapter("some/base", tlt.QLoraSettings())


def test_a_working_setup_still_loads(fake_libs):
    """The guard must not have been replaced by a refusal to run at all."""
    model, _line = _load(fake_libs)
    assert model.enabled == ["checkpointing", "require_grads"]
    assert model.config.use_cache is False, "cache and checkpointing conflict"


def test_checkpointing_failing_ends_the_load(fake_libs):
    """It used to `pass`, and the run then OOM-skipped its way to 'done'."""
    with pytest.raises(RuntimeError) as got:
        _load(fake_libs, fail="checkpointing")
    assert "gradient checkpointing" in str(got.value)


def test_require_grads_failing_ends_the_load(fake_libs):
    """Silent and free: the segments simply take no gradient."""
    with pytest.raises(RuntimeError):
        _load(fake_libs, fail="require_grads")


def test_the_failure_names_the_base_and_the_consequence(fake_libs):
    """Two bases train in one install, and an OOM here is SKIPPED not raised,
    so the reader cannot be expected to infer either part."""
    with pytest.raises(RuntimeError) as got:
        _load(fake_libs, fail="checkpointing")
    msg = str(got.value)
    assert "some/base" in msg, msg
    assert "OOM" in msg, msg
    assert "skipped" in msg, msg


def test_the_original_error_is_kept(fake_libs):
    """`from exc`: the library's own message is the actionable part."""
    with pytest.raises(RuntimeError) as got:
        _load(fake_libs, fail="checkpointing")
    assert isinstance(got.value.__cause__, ValueError)
    assert "does not support checkpointing" in str(got.value.__cause__)


# --------------------------------------------------------------------- #
# and the skip itself stays reported
# --------------------------------------------------------------------- #

def test_an_oom_skip_is_still_counted():
    """The guard is right to skip; it must not be right quietly."""
    src = inspect.getsource(tlt.run_training)
    assert '"skipped_oom": progress.skipped_oom' in src, \
        "the count has to leave the trainer or nobody can see the cause"


def test_the_guard_only_catches_oom():
    """Anything broader would swallow real failures as 'did not fit'."""
    calls = []

    class _Oom(Exception):
        pass

    fake_torch = types.SimpleNamespace(
        OutOfMemoryError=_Oom,
        cuda=types.SimpleNamespace(empty_cache=lambda: calls.append("freed")))
    import sys as _s
    orig = _s.modules.get("torch")
    _s.modules["torch"] = fake_torch
    try:
        with pytest.raises(tlt.OomSkipped):
            tlt.step_with_oom_guard(lambda: (_ for _ in ()).throw(_Oom("no room")))
        assert calls == ["freed"], "the skip has to give the memory back"
        # A non-OOM failure is a real failure and must propagate untouched.
        with pytest.raises(ValueError):
            tlt.step_with_oom_guard(
                lambda: (_ for _ in ()).throw(ValueError("a real bug")))
    finally:
        if orig is None:
            _s.modules.pop("torch", None)
        else:
            _s.modules["torch"] = orig

# --------------------------------------------------------------------- #
# and what the card is actually holding
# --------------------------------------------------------------------- #

def test_the_loaded_base_is_described(fake_libs, capsys):
    """A base that silently failed to quantize, or spilled to the CPU, shows
    up only as a pod that trains a fraction of its slice -- because the loop
    turns an OOM into a skip. So setup has to say what it loaded."""
    _load(fake_libs)
    said = capsys.readouterr().out
    assert "base loaded:" in said, said
    assert "GB of parameters" in said, said
    assert "4-bit=" in said, said


def test_an_unquantized_base_says_so_loudly(fake_libs):
    """`4-bit=NO` is the line that would have ended a four-hour run early."""
    import inspect
    src = inspect.getsource(tlt.load_base_and_adapter)
    assert "the base is NOT quantized" in src, src[-900:]


def test_a_base_split_onto_the_cpu_is_a_warning(fake_libs, capsys):
    """device_map='auto' spills rather than failing, which reads as success."""
    import types as _t

    class _P:
        def __init__(self, dev):
            self.device = dev
            self.dtype = "torch.uint8"

        def numel(self):
            return 1000

        def element_size(self):
            return 1

    model = fake_libs["model"] = _Model()
    model.parameters = lambda: [_P("cuda:0"), _P("cpu"), _P("cpu")]
    tlt.load_base_and_adapter("some/base", tlt.QLoraSettings())
    said = capsys.readouterr().out
    assert "did not fit on the card" in said, said
    assert "SKIPS an OOM" in said, said


def test_the_report_failing_does_not_stop_training(fake_libs, capsys):
    """Instrumentation must never be the reason a run refuses to start."""
    model = fake_libs["model"] = _Model()

    def _boom():
        raise RuntimeError("no parameters here")
    model.parameters = _boom

    got = tlt.load_base_and_adapter("some/base", tlt.QLoraSettings())
    assert got is not None
    assert "could not describe the loaded base" in capsys.readouterr().out
