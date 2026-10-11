"""`extra_args` on a model's runpod block has to reach the pod's command line.

It has been accepted there — and whitelisted per engine, so it survives a save —
since pods shipped, and it went nowhere: parsed, stored, never printed. Setting it
LOOKED like tuning and did nothing, which is worse than refusing it would have been,
because the pod boots, serves, and quietly ignores the decision.

The two Digesta OCR VLMs are what noticed. A page-reading model wants prefix caching
(its prompt is identical on every page of every document) and the Qwen-VL pixel bounds
that Surya's own launcher sets, and on a rented pod neither could be expressed: the
only way in was `docker_args`, which replaces the entire generated command and so
means re-deriving --model, --served-model-name, --max-model-len and the api key by
hand, to be kept in step forever after.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import runpod_worker as rw                  # noqa: E402


def _mcfg(**kw):
    cfg = rw.RunpodModelConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_a_vllm_pod_gets_the_operators_flags():
    args = rw._vllm_docker_args(
        _mcfg(ctx=18432, extra_args="--enable-prefix-caching"),
        "datalab-to/surya-ocr-2", {})
    assert "--model datalab-to/surya-ocr-2" in args
    assert "--max-model-len 18432" in args
    assert args.endswith("--enable-prefix-caching")


def test_the_extra_flags_come_last_so_they_win():
    """On any command line the later flag wins; a tuning knob that landed first
    would be overridden by our own default and look like it had done nothing."""
    args = rw._vllm_docker_args(
        _mcfg(ctx=4096, extra_args="--max-model-len 9000"),
        "org/model", {})
    assert args.index("--max-model-len 4096") < args.index("--max-model-len 9000")


def test_quoted_json_survives():
    """--mm-processor-kwargs takes a JSON object; splitting and rejoining it on
    whitespace would hand vLLM three broken fragments."""
    kw = '--mm-processor-kwargs {"min_pixels":3136,"max_pixels":6291456}'
    args = rw._vllm_docker_args(_mcfg(extra_args=kw), "org/model", {})
    assert kw in args


def test_a_llamacpp_pod_gets_them_too():
    args = rw._llamacpp_docker_args(
        _mcfg(ctx=8192, extra_args="--flash-attn", hf_gguf="org/repo:Q4_K_M"),
        "org/repo:Q4_K_M", {})
    assert args.endswith("--flash-attn")


def test_nothing_configured_changes_nothing():
    """The flag is additive: an install that never set it must get the exact command
    line it got before."""
    plain = rw._vllm_docker_args(_mcfg(ctx=1024), "org/model", {})
    assert plain == plain.strip()
    assert "  " not in plain


def test_the_api_key_still_lands_on_a_tuned_pod():
    """pod_plan appends --api-key only when it is not already there; a trailing
    extra_args must not confuse that check into leaving the pod WIDE OPEN."""
    plan = rw.pod_plan(_mcfg(ctx=1024, extra_args="--enable-prefix-caching"),
                       "org/model", model_key="org/model", api_key="secrettoken",
                       entry={})
    assert "--api-key secrettoken" in plan["args"]
    assert "--enable-prefix-caching" in plan["args"]


def test_docker_args_still_replaces_everything():
    """The full-override escape hatch is unchanged — and because it replaces the
    line, extra_args is not appended to it either."""
    plan = rw.pod_plan(_mcfg(docker_args="--model x --port 8000",
                             extra_args="--enable-prefix-caching"),
                       "org/model", model_key="org/model", entry={})
    assert plan["args"].startswith("--model x --port 8000")
    assert "--enable-prefix-caching" not in plan["args"]
