"""Per-model vLLM speed settings: they reach the engine, and saving keeps them.

Two separate things are pinned here.

THE FLAGS. Decode on one card is bound by how fast the weights can be read, so
the only levers worth having are the ones that avoid work rather than speed it
up: prefill a bigger batch per step, reuse a prefix already computed, and draft
tokens from the prompt. Each is a flag vLLM only honours if it was launched with
it, so each is asserted on the launch command rather than on the config object.

Prefix caching is a tri-state on purpose. vLLM's own default is per MODEL -- on
for an ordinary decoder, off for a hybrid-attention one, where the feature is
supported but kept opt-in upstream -- so "leave it alone" has to be expressible
and has to be the default. A flag passed unasked would overrule that judgement
for every model at once.

THE SAVE. The model page used to send `vllm: {}` for any model whose backend was
not literally "vllm", and the save route reads an empty block as "this model has
no vLLM settings" and deletes the stored one. A pod-served model is exactly the
case that was hit: its card was hidden, so merely opening the model and saving it
dropped its block -- kv-cache dtype, cache floors and all -- with no error and no
visible change until the next pod booted differently. The rule now is that a
block is only sent when its card was on screen to be edited; otherwise the key is
left out, which the route ignores.
"""
import json
import pathlib
import re
import shutil
import subprocess
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = ROOT / "codai/admin/templates/models.html"


# --------------------------------------------------------------------------
# the launch command
# --------------------------------------------------------------------------

def _cfg(**kw):
    """A vLLM config carrying only what a test sets, like a real override."""
    base = dict(ctx=0, gpu_memory_utilization=0, tensor_parallel_size=0,
                pipeline_parallel_size=0, distributed_executor_backend="",
                max_num_seqs=0, dtype="", quantization="",
                max_num_batched_tokens=0, prefix_caching="auto",
                speculative="", speculative_tokens=0,
                tool_call_parser="off", reasoning_parser="off",
                return_token_ids=False, extra_args="")
    base.update(kw)
    return types.SimpleNamespace(**base)


def _cmd(**kw):
    from codai.api import vllm_worker
    return vllm_worker._launch_cmd("py", _cfg(**kw), "127.0.0.1", 8000,
                                   "org/model", served_name="m", model_config={})


def _flag_value(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def test_prefill_batch_is_passed_when_configured():
    """vLLM's default is 2048, so a 60k prompt is ~30 sequential passes."""
    assert _flag_value(_cmd(max_num_batched_tokens=8192),
                       "--max-num-batched-tokens") == "8192"


def test_a_callers_own_prefill_batch_still_wins():
    """A side job sizing itself down must not be overruled by the config."""
    from codai.api import vllm_worker
    cmd = vllm_worker._launch_cmd("py", _cfg(max_num_batched_tokens=8192),
                                  "127.0.0.1", 8000, "org/model",
                                  served_name="m", model_config={},
                                  max_num_batched_tokens=2048)
    assert _flag_value(cmd, "--max-num-batched-tokens") == "2048"


def test_prefill_batch_is_absent_by_default():
    assert "--max-num-batched-tokens" not in _cmd()


@pytest.mark.parametrize("value", ["auto", "", None])
def test_prefix_caching_unset_passes_no_flag(value):
    """The model's own default must survive; silence is not "off"."""
    cmd = _cmd(prefix_caching=value)
    assert "--enable-prefix-caching" not in cmd
    assert "--no-enable-prefix-caching" not in cmd


def test_prefix_caching_on_is_forced_explicitly():
    """The only way to get it on a hybrid model, where vLLM defaults it off."""
    assert "--enable-prefix-caching" in _cmd(prefix_caching="on")


def test_prefix_caching_off_is_forced_explicitly():
    cmd = _cmd(prefix_caching="off")
    assert "--no-enable-prefix-caching" in cmd
    assert "--enable-prefix-caching" not in cmd


def test_ngram_needs_no_draft_model_and_no_json_from_the_operator():
    cmd = _cmd(speculative="ngram")
    body = json.loads(_flag_value(cmd, "--speculative-config"))
    assert body["method"] == "ngram"
    assert body["num_speculative_tokens"] >= 1


def test_ngram_draft_length_is_configurable():
    cmd = _cmd(speculative="ngram", speculative_tokens=3)
    assert json.loads(_flag_value(cmd, "--speculative-config"))[
        "num_speculative_tokens"] == 3


def test_a_real_draft_model_can_be_configured_as_json():
    raw = '{"method": "eagle", "model": "org/eagle-head", "num_speculative_tokens": 4}'
    body = json.loads(_flag_value(_cmd(speculative=raw), "--speculative-config"))
    assert body["model"] == "org/eagle-head"


def test_unparseable_speculative_config_does_not_stop_the_engine_booting():
    """An argparse error about JSON would name neither the model nor the field."""
    assert "--speculative-config" not in _cmd(speculative="{not json")


def test_no_speculative_flag_by_default():
    assert "--speculative-config" not in _cmd()


def test_blank_quantization_lets_vllm_detect_the_checkpoint():
    """Auto-detect picks the fastest kernel; a wrong name picks a slower one."""
    assert "--quantization" not in _cmd()
    assert _flag_value(_cmd(quantization="awq"), "--quantization") == "awq"


# --------------------------------------------------------------------------
# the settings travel: per-model override, and on to a pod
# --------------------------------------------------------------------------

def test_the_levers_are_overridable_per_model():
    from codai.backends.overrides import VLLM_FIELDS, apply
    from codai.config import VllmConfig
    out = apply(VllmConfig(), {"max_num_batched_tokens": 8192, "prefix_caching": "on",
                         "speculative": "ngram", "speculative_tokens": 4,
                         "quantization": "awq"}, VLLM_FIELDS, "vllm")
    assert out.max_num_batched_tokens == 8192
    assert out.prefix_caching == "on"
    assert out.speculative == "ngram"
    assert out.speculative_tokens == 4
    assert out.quantization == "awq"


def test_a_model_cannot_override_what_is_not_whitelisted():
    from codai.backends.overrides import VLLM_FIELDS, apply
    from codai.config import VllmConfig
    out = apply(VllmConfig(), {"install_dir": "/somewhere/else"}, VLLM_FIELDS, "vllm")
    assert not hasattr(out, "install_dir")


def test_the_levers_reach_a_pod():
    """They describe how to RUN the model, so they belong where it runs."""
    from codai.api.runpod_worker import _ENGINE_FORWARDED_FIELDS
    forwarded = _ENGINE_FORWARDED_FIELDS["vllm"]
    for field in ("max_num_batched_tokens", "prefix_caching",
                  "speculative", "speculative_tokens", "quantization"):
        assert field in forwarded, f"{field} would stay on the orchestrator"


def test_the_admin_save_route_accepts_the_levers():
    src = (ROOT / "codai/admin/routes.py").read_text(encoding="utf-8")
    block = re.search(r'for _blk, _fields in \(\("vllm",(.*?)\)\),', src, re.S)
    assert block, "the per-model engine block whitelist moved"
    for field in ("max_num_batched_tokens", "prefix_caching", "speculative"):
        assert f'"{field}"' in block.group(1), f"{field} is dropped on save"


# --------------------------------------------------------------------------
# the save no longer wipes the block
# --------------------------------------------------------------------------

needs_node = pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")


@pytest.fixture(scope="module")
def html():
    return PAGE.read_text(encoding="utf-8")


def _fn(html, name):
    m = re.search(r"^function %s\(.*?^}" % re.escape(name), html, re.S | re.M)
    assert m, f"{name}() is gone from the page"
    return m.group(0)


def _collect(html, tmp_path, backend, engine, fields=None):
    """Run the page's own _collectCluster() against a stubbed form."""
    const = re.search(r"^const _VLLM_POD_ENGINES = .*?;", html, re.S | re.M)
    assert const, "_VLLM_POD_ENGINES is gone from the page"
    values = dict(fields or {})
    values["cfg-backend"] = backend
    values["cfg-rp-engine"] = engine
    js = tmp_path / "collect.js"
    js.write_text(
        "const VALUES = %s;\n"
        "function parse(){}\n"
        "const document = { getElementById: id =>"
        " ({ value: VALUES[id] === undefined ? '' : VALUES[id],"
        "    checked: !!VALUES[id] }) };\n"
        % json.dumps(values)
        + const.group(0) + "\n"
        + _fn(html, "_vllmBlockApplies") + "\n"
        + _fn(html, "_nodesToLines") + "\n"
        + _fn(html, "_collectCluster") + "\n"
        + "console.log(JSON.stringify(_collectCluster()));",
        encoding="utf-8")
    proc = subprocess.run(["node", str(js)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr.strip()[:800]
    return json.loads(proc.stdout)


@needs_node
def test_saving_a_pod_served_model_does_not_delete_its_vllm_block(html, tmp_path):
    """The defect: an empty block is a destructive statement, not a no-op.

    A model served by a vLLM pod has `backend: runpod`, so the card was hidden
    and the collector emitted `vllm: {}` -- which the save route turns into
    "delete the block". That silently dropped --kv-cache-dtype fp8 and the KV
    cache floors, and nothing changed until the next pod booted in FP16.
    """
    out = _collect(html, tmp_path, "runpod", "coderai-vllm",
                   {"cfg-vllm-extra": "--kv-cache-dtype fp8",
                    "cfg-vllm-kv-conc": "3"})
    assert out.get("vllm"), "the vLLM block came back empty and would be deleted"
    assert out["vllm"]["extra_args"] == "--kv-cache-dtype fp8"
    assert out["vllm"]["kv_min_concurrency"] == 3


@needs_node
def test_a_block_whose_card_was_never_shown_is_left_alone(html, tmp_path):
    """Not sent at all, rather than sent empty: the route then keeps it."""
    out = _collect(html, tmp_path, "cuda", "auto")
    assert "vllm" not in out, "an unseen block must not be sent, even empty"


@needs_node
def test_the_levers_are_collected_from_the_page(html, tmp_path):
    out = _collect(html, tmp_path, "runpod", "coderai-vllm",
                   {"cfg-vllm-mnbt": "8192", "cfg-vllm-prefix": "on",
                    "cfg-vllm-spec": "ngram", "cfg-vllm-spec-tokens": "5",
                    "cfg-vllm-quant": "awq"})
    blk = out["vllm"]
    assert blk["max_num_batched_tokens"] == 8192
    assert blk["prefix_caching"] == "on"
    assert blk["speculative"] == "ngram"
    assert blk["speculative_tokens"] == 5
    assert blk["quantization"] == "awq"


@needs_node
def test_a_local_vllm_model_still_gets_the_card(html, tmp_path):
    out = _collect(html, tmp_path, "vllm", "auto", {"cfg-vllm-tp": "2"})
    assert out["vllm"]["tensor_parallel_size"] == 2


@needs_node
def test_a_pod_on_another_engine_does_not_get_the_vllm_card(html, tmp_path):
    """A GGUF on llama.cpp has no vLLM settings to edit, so none are sent."""
    out = _collect(html, tmp_path, "runpod", "coderai-llama")
    assert "vllm" not in out
