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

"""A VRAM measurement is only valid for the quantization it was taken under.

Change component_quantization and the stored number describes a model that no
longer exists. That would be harmless if the measurement were a hint, but
force_vram_update makes it WIN over the configured size, so a stale figure
goes on deciding placement.

Measured 2026-10-29: Wan2.2-TI2V-5B carried measured_vram_gb = 40.0 from a
load where only its text encoder was quantized -- its map named text_encoder
and vae, and vae is conv-only so bitsandbytes skipped it, leaving an fp32
18.63 GB transformer to load whole. Its components are 18.63 + 10.58 + 2.63 GB
on disk; quantized properly the same pipeline is roughly 10 GB. Without
invalidation, 40.0 would have kept forcing CPU offload on a 25.3 GB card until
some unlucky load happened to re-measure it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.models.manager import MultiModelManager


BASE = {"precision": "bf16", "load_in_4bit": True,
        "component_quantization": {"text_encoder": "4bit", "vae": "4bit"}}


def _sig(cfg):
    return MultiModelManager._quant_signature(None, cfg)


def test_adding_a_component_changes_the_signature():
    """The real change: the transformer joins the quantized set."""
    after = dict(BASE, component_quantization={
        "text_encoder": "4bit", "vae": "4bit", "transformer": "4bit"})
    assert _sig(BASE) != _sig(after)


def test_the_global_flag_changes_the_signature():
    assert _sig(BASE) != _sig(dict(BASE, load_in_4bit=False))


def test_precision_changes_the_signature():
    assert _sig(BASE) != _sig(dict(BASE, precision="fp16"))


def test_an_unrelated_edit_does_not():
    """Offload is a runtime placement decision, not a weight-size one; it must
    not throw away a perfectly good measurement."""
    assert _sig(BASE) == _sig(dict(BASE, offload_strategy="model",
                                   used_vram_gb=14.0, load_mode="on-request"))


def test_the_manager_stamps_and_checks_the_signature():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/models/manager.py"),
               encoding="utf-8").read()
    assert '_persist("measured_quant_sig"' in src, (
        "the measurement is stored without recording which quantization it "
        "describes, so it can never be invalidated")
    assert "_sig_then != _sig_now" in src, (
        "a measurement from a different quantization is still trusted")


def test_a_measurement_with_no_signature_is_still_trusted():
    """Entries measured before this existed must not all be discarded at
    once -- that would re-measure every model on one upgrade."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/models/manager.py"),
               encoding="utf-8").read()
    block = src.split("_sig_then = cfg.get(\"measured_quant_sig\")", 1)[1][:400]
    assert "_sig_then and" in block, (
        "an entry without a recorded signature would be invalidated too")


def test_the_signature_does_not_depend_on_a_model_name():
    """The two call sites do not have the same name to hand -- the reader has
    a resolved path, the writer only a config key. Hashing a name they derive
    differently would make every comparison a mismatch and throw away every
    measurement."""
    import inspect
    sig = inspect.signature(MultiModelManager._quant_signature)
    assert list(sig.parameters) == ["self", "cfg"], (
        f"_quant_signature takes {list(sig.parameters)}; a model-name "
        f"parameter reintroduces the mismatch")


def test_both_call_sites_bind_what_they_pass():
    """record_vram_delta has no `resolved_name`. Passing it there raised
    NameError inside an enclosing try, so the signature silently never
    persisted and the invalidation could never fire -- verified live
    2026-10-29, measured_quant_sig was None after a real load."""
    import ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "codai/models/manager.py"),
               encoding="utf-8").read()
    tree = ast.parse(src)
    lines = src.splitlines()
    checked = 0
    for i, line in enumerate(lines, 1):
        if "_quant_signature(" not in line or "def _quant_signature" in line:
            continue
        passed = [v for v in ("cfg", "cfgd") if f"({v})" in line]
        if not passed:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not (node.lineno < i <= (node.end_lineno or 0)):
                continue
            if node.name.startswith("_persist"):
                continue
            bound = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                     and isinstance(n.ctx, ast.Store)}
            bound |= {a.arg for a in node.args.args}
            for v in passed:
                assert v in bound, (
                    f"{node.name}() passes '{v}' to _quant_signature but never "
                    f"binds it — NameError, swallowed, signature never written")
            checked += 1
            break
    assert checked >= 2, f"expected both call sites, checked {checked}"
