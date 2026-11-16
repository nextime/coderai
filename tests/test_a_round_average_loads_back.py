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

"""A round's average has to load back into the next round's model.

This is the whole mechanism of :mod:`codai.api.text_lora_rounds`: pods train a
round, the orchestrator averages their adapters, and the average is pushed back
as the NEXT round's starting point. If that last step cannot load, a
distributed run cannot pass round 1 -- which is exactly what happened in
production, on every pod, in a 16-round job whose round 0 looked perfect:

    train(shard 3): FAILED -- RuntimeError: start.safetensors has 512 key(s)
    this model has no place for, e.g. ['base_model.model.model.layers.0.mlp.
    down_proj.lora_A.weight', ...]

512 of 512, on an adapter the same function had written minutes earlier. The
cause is that PEFT uses two key spellings: the FILE has no adapter name in its
keys and the LIVE module does, so comparing one against the other matches
nothing. These tests use real PEFT on a two-layer model rather than reading the
source, because the source text looked entirely reasonable.
"""

import os

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")


def _tiny_peft_model():
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)
            self.out = nn.Linear(8, 2)

        def forward(self, x):
            return self.out(self.lin(x))

    return get_peft_model(M(), LoraConfig(r=2, lora_alpha=4,
                                          target_modules=["lin"]))


def test_the_file_and_the_live_module_spell_the_keys_differently(tmp_path):
    """The fact the broken check assumed away, pinned so it cannot drift."""
    from safetensors.torch import load_file

    m = _tiny_peft_model()
    m.save_pretrained(str(tmp_path))
    on_disk = list(load_file(str(tmp_path / "adapter_model.safetensors")))
    live = [k for k in m.state_dict() if "lora" in k]

    assert on_disk and live
    assert all(".default." not in k for k in on_disk)
    assert all(".default." in k for k in live)
    # So a by-name comparison finds NOTHING -- all of them, not some.
    own = dict(m.state_dict())
    assert [k for k in on_disk if k not in own] == on_disk


def test_a_saved_adapter_loads_back_through_peft(tmp_path):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    m = _tiny_peft_model()
    with torch.no_grad():                 # something to recognise afterwards
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.fill_(0.25)
    m.save_pretrained(str(tmp_path))
    incoming = load_file(str(tmp_path / "adapter_model.safetensors"))

    fresh = _tiny_peft_model()
    res = set_peft_model_state_dict(fresh, incoming)
    assert list(getattr(res, "unexpected_keys", []) or []) == []
    got = [p for n, p in fresh.named_parameters() if "lora_B" in n]
    assert got and all(float(p.detach().abs().mean()) == pytest.approx(0.25)
                       for p in got)


def test_run_training_starts_from_the_average_it_was_given(tmp_path,
                                                           monkeypatch):
    """The real call path: ``run_training(init_adapter=...)``.

    The base load and the training loop are stubbed -- neither is what broke,
    and a 27B 4-bit base does not fit in a test -- but the init_adapter branch
    itself runs, on a file written the way a round's average is written.
    """
    from safetensors.torch import save_file
    from codai.api import text_lora_trainer as tlt

    donor = _tiny_peft_model()
    donor.save_pretrained(str(tmp_path / "donor"))
    from safetensors.torch import load_file
    avg = load_file(str(tmp_path / "donor" / "adapter_model.safetensors"))
    start = str(tmp_path / "start.safetensors")
    save_file(avg, start)

    target = _tiny_peft_model()
    monkeypatch.setattr(tlt, "load_base_and_adapter",
                        lambda base, settings: (target, "stub"))
    # Stop at the statement AFTER the init_adapter branch: the loop needs an
    # optimiser over a real base and has its own tests.
    marker = RuntimeError("reached the training loop")

    def _boom(*a, **k):
        raise marker
    monkeypatch.setattr(tlt, "Progress", _boom)

    with pytest.raises(RuntimeError) as exc:
        tlt.run_training(base="stub", encoded=[], out_dir=str(tmp_path / "o"),
                         settings=tlt.QLoraSettings(), resume=False,
                         init_adapter=start)
    # It got PAST the load -- the old code raised "has N key(s) this model has
    # no place for" here instead, and that is the regression.
    assert exc.value is marker


def test_an_adapter_from_a_different_target_set_is_still_fatal(tmp_path,
                                                               monkeypatch):
    """The check the old code was trying to make, kept working.

    Starting from random weights because a key was quietly skipped looks
    exactly like a round that trained badly, so a genuine mismatch must raise.
    """
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from codai.api import text_lora_trainer as tlt

    class Other(nn.Module):
        def __init__(self):
            super().__init__()
            self.elsewhere = nn.Linear(8, 8)

        def forward(self, x):
            return self.elsewhere(x)

    donor = get_peft_model(Other(), LoraConfig(r=2, lora_alpha=4,
                                               target_modules=["elsewhere"]))
    donor.save_pretrained(str(tmp_path / "donor"))
    from safetensors.torch import load_file
    start = str(tmp_path / "start.safetensors")
    save_file(load_file(str(tmp_path / "donor" / "adapter_model.safetensors")),
              start)

    monkeypatch.setattr(tlt, "load_base_and_adapter",
                        lambda base, settings: (_tiny_peft_model(), "stub"))
    monkeypatch.setattr(tlt, "load_progress",
                        lambda *a, **k: pytest.fail("got past the load"))
    monkeypatch.setattr(tlt, "Progress",
                        lambda *a, **k: pytest.fail("got past the load"))

    with pytest.raises(RuntimeError) as exc:
        tlt.run_training(base="stub", encoded=[], out_dir=str(tmp_path / "o"),
                         settings=tlt.QLoraSettings(), resume=False,
                         init_adapter=start)
    assert "no place for" in str(exc.value)
    assert os.path.basename(start) in str(exc.value) or start in str(exc.value)
