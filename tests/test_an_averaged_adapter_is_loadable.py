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

"""A distributed run must leave behind something a loader accepts.

The averaging path wrote `adapter_model.safetensors` and nothing else, so
the directory had no `adapter_config.json` -- and without one, neither PEFT
nor vLLM will load it. The first distributed run to reach the end shipped
exactly that: 318 MB of correct tensors in a directory that is not an
adapter. Measured against `diogene-alpha`, which the single-pod path built
properly and which serves.

The config is DERIVED FROM THE TENSORS on purpose: the pod chooses its own
target modules by listing the base model's Linear layers and keeping the
language projections, which this side cannot reproduce without loading the
base model. The keys are the only truthful record of what was adapted.
"""

import json
import os
import struct

import pytest

from codai.api import text_lora_rounds as tlr


# Shapes from the real 512-tensor adapter the 16-pod run produced, cut down
# to two layers: lora_A is [r, in], lora_B is [out, r].
def _write_fake_adapter(path, *, rank=16, layers=2, prefix="base_model.model."):
    hdr = {}
    offset = 0
    for i in range(layers):
        for mod, (fan_in, fan_out) in {
            "self_attn.q_proj": (5120, 5120),
            "mlp.down_proj": (17408, 5120),
        }.items():
            base = f"{prefix}model.layers.{i}.{mod}"
            for name, shape in ((f"{base}.lora_A.weight", [rank, fan_in]),
                                (f"{base}.lora_B.weight", [fan_out, rank])):
                n = shape[0] * shape[1] * 4
                hdr[name] = {"dtype": "F32", "shape": shape,
                             "data_offsets": [offset, offset + n]}
                offset += n
    blob = json.dumps(hdr).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * offset)
    return path


def test_the_config_names_every_adapted_module(tmp_path):
    """CORRECTED. This asserted the fully-qualified module PATHS.

    It was a faithful description of the tensors and the reason the adapter
    this file is named for could not be loaded. PEFT and vLLM resolve
    `target_modules` by matching each entry against the SUFFIX of the base's
    module names, so a path only ever matches if the orchestrator guessed the
    base's whole module tree -- and `diogene-alpha`, which this file measures
    itself against and which actually serves, lists the seven leaf names.
    Measured live: coderai resolved the name, the gateway taught the pod, the
    pod's model list came back without it, generation answered 500.
    """
    p = _write_fake_adapter(str(tmp_path / "adapter_model.safetensors"))
    cfg = tlr.adapter_config_for(p, base="Qwen/Qwen3.8-27B", alpha=32)

    assert cfg["target_modules"] == ["down_proj", "q_proj"], (
        "the adapted modules are the LEAF names a loader matches, "
        "deduplicated across layers")
    assert not any("." in m for m in cfg["target_modules"]), (
        "a module path is not a target_modules entry")


def test_the_rank_is_read_off_the_tensors(tmp_path):
    p = _write_fake_adapter(str(tmp_path / "a.safetensors"), rank=64)
    cfg = tlr.adapter_config_for(p, base="b", alpha=128)
    assert cfg["r"] == 64, "lora_A is [r, in_features]: the rank is the rows"
    assert cfg["lora_alpha"] == 128


def test_the_rank_is_not_confused_with_a_narrow_layer(tmp_path):
    """A module narrower than the rank must not become the rank."""
    p = str(tmp_path / "a.safetensors")
    hdr = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
           {"dtype": "F32", "shape": [32, 8], "data_offsets": [0, 1024]}}
    blob = json.dumps(hdr).encode("utf-8")
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * 1024)
    assert tlr.adapter_config_for(p, base="b", alpha=64)["r"] == 32


def test_a_missing_alpha_does_not_scale_the_adapter_to_nothing(tmp_path):
    """alpha=0 would make the adapter a no-op; PEFT's convention is alpha==r."""
    p = _write_fake_adapter(str(tmp_path / "a.safetensors"), rank=16)
    for missing in (None, 0):
        cfg = tlr.adapter_config_for(p, base="b", alpha=missing)
        assert cfg["lora_alpha"] == 16


def test_the_adapter_name_prefix_is_tolerated(tmp_path):
    """Live state dicts carry `lora_A.default.weight`, saved ones do not."""
    p = str(tmp_path / "a.safetensors")
    hdr = {}
    off = 0
    for key in ("base_model.model.model.layers.0.mlp.up_proj"
                ".lora_A.default.weight",
                "base_model.model.model.layers.0.mlp.up_proj"
                ".lora_B.default.weight"):
        hdr[key] = {"dtype": "F32", "shape": [16, 32],
                    "data_offsets": [off, off + 2048]}
        off += 2048
    blob = json.dumps(hdr).encode("utf-8")
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * off)
    cfg = tlr.adapter_config_for(p, base="b", alpha=32)
    # The leaf name, for the reason given in
    # `test_the_config_names_every_adapted_module`. What this test is for is
    # the `.default.` infix, which must still be stripped.
    assert cfg["target_modules"] == ["up_proj"]


def test_the_required_keys_match_a_working_adapter(tmp_path):
    """Against the shape of `diogene-alpha`, which actually serves."""
    p = _write_fake_adapter(str(tmp_path / "a.safetensors"))
    cfg = tlr.adapter_config_for(p, base="Qwen/Qwen3.8-27B",
                                 alpha=32, dropout=0.0)
    for key in ("peft_type", "task_type", "r", "lora_alpha", "lora_dropout",
                "target_modules", "base_model_name_or_path", "bias"):
        assert key in cfg, f"a loader reads {key}"
    assert cfg["peft_type"] == "LORA"
    assert cfg["task_type"] == "CAUSAL_LM"
    assert cfg["bias"] == "none"
    assert cfg["base_model_name_or_path"] == "Qwen/Qwen3.8-27B"


def test_something_that_is_not_an_adapter_is_refused(tmp_path):
    """Better a loud failure than a config describing no tensors."""
    p = str(tmp_path / "a.safetensors")
    hdr = {"some.random.weight": {"dtype": "F32", "shape": [2, 2],
                                  "data_offsets": [0, 16]}}
    blob = json.dumps(hdr).encode("utf-8")
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * 16)
    with pytest.raises(ValueError, match="not a .*LoRA|no lora_A"):
        tlr.adapter_config_for(p, base="b", alpha=16)


def test_the_file_is_written_where_a_loader_looks(tmp_path):
    p = _write_fake_adapter(str(tmp_path / "adapter_model.safetensors"))
    dest = tlr.write_adapter_config(str(tmp_path), p, base="b", alpha=32)

    assert os.path.basename(dest) == "adapter_config.json"
    assert os.path.isfile(dest)
    with open(dest, encoding="utf-8") as fh:
        back = json.load(fh)
    assert back["r"] == 16
    assert not os.path.exists(dest + ".part"), "no half-written file left"


def test_the_distributed_run_writes_one(tmp_path):
    """The call site, not just the helper: this is what was missing."""
    import inspect
    from codai.api import loras
    src = inspect.getsource(loras._text_train_distributed)
    assert "write_adapter_config" in src, (
        "the averaging path must write adapter_config.json; without it the "
        "run produces a directory no loader accepts")
