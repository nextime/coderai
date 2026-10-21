"""vLLM refuses a request carrying `tools` unless it was launched for it.

Reported from production:

    500 {"detail": "Generation error: the pod answered 500: Generation error:
         400 Bad Request … \"auto\" tool choice requires
         --enable-auto-tool-choice and --tool-call-parser to be set"}

A 400 at REQUEST time, from a server that started cleanly, for a capability the
catalogue says the model has — and naming two flags the caller has no way to
set. coderai passed `tools` through to vLLM (correctly: the remote decides what
it supports) and never launched it able to accept them.

A tool call is TEXT the model emits in its own trained format, so the parser
belongs to the model FAMILY rather than to the deployment, which is what makes
a default possible at all.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import vllm_worker as vw                       # noqa: E402


def _cfg(parser="auto"):
    return type("C", (), {"tool_call_parser": parser})()


@pytest.fixture(autouse=True)
def _no_template_probe(monkeypatch):
    """These tests cover the NAME table alone, so the template probe is off:
    it would otherwise reach the network for every repo id. What the template
    decides when it IS readable is covered in
    tests/test_vllm_qwen_tool_parser_choice.py."""
    monkeypatch.setattr(vw, "_chat_template_text", lambda *a, **k: "")


@pytest.mark.parametrize("path,want", [
    # By name only. This model's own template actually mandates XML tool calls,
    # so in production it resolves to qwen3_xml -- see the test file named above.
    ("cyankiwi/Qwen3.8-27B-AWQ-INT4", "hermes"),
    ("Qwen/Qwen2.5-7B-Instruct", "hermes"),
    ("meta-llama/Llama-3.3-70B-Instruct", "llama3_json"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    ("deepseek-ai/DeepSeek-V3", "deepseek_v3"),
    ("zai-org/GLM-4.6", "glm45"),
])
def test_the_parser_is_picked_from_the_model_family(path, want):
    assert vw._tool_parser_for(_cfg(), {}, path) == want


def test_a_longer_family_name_wins():
    """qwen3-coder emits a different syntax from the rest of the Qwen line."""
    assert vw._tool_parser_for(_cfg(), {}, "Qwen/Qwen3-Coder-30B") == "qwen3_coder"


def test_an_explicit_setting_beats_the_guess():
    assert vw._tool_parser_for(_cfg("mistral"), {}, "Qwen/Qwen3-8B") == "mistral"


def test_off_really_means_off():
    """A family guessed wrongly parses tool calls into nonsense, and silently
    producing wrong tool ARGUMENTS is worse than refusing the request."""
    assert vw._tool_parser_for(_cfg("off"), {}, "Qwen/Qwen3-8B") == ""
    assert vw._tool_args(_cfg("off"), {}, "Qwen/Qwen3-8B") == []


def test_the_model_entry_is_read_too():
    """A local path says nothing; the entry's id or alias does."""
    assert vw._tool_parser_for(_cfg(), {"id": "qwen38-awq"}, "/models/blob-1234") \
        == "hermes"


def test_both_flags_or_neither():
    """vLLM refuses to START with --enable-auto-tool-choice and no parser, and
    a pod that will not boot is worse than one that cannot do tool calls."""
    args = vw._tool_args(_cfg(), {}, "Qwen/Qwen3-8B")
    assert args == ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    assert vw._tool_args(_cfg(), {}, "some/unknown-model-7b") == []


def test_an_unknown_family_says_so(capsys):
    """Otherwise the symptom lands on the CLIENT as a 400 at request time."""
    vw._tool_args(_cfg(), {}, "some/unknown-model-7b")
    out = capsys.readouterr().out
    assert "no tool-call parser known" in out
    assert "vllm.tool_call_parser" in out


def test_the_launch_carries_the_flags():
    cmd = vw._launch_cmd("py", _cfg(), "127.0.0.1", 1, "Qwen/Qwen3-8B",
                         served_name="qwen3", model_config={})
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"


def test_a_models_own_vllm_block_can_override_it():
    from codai.backends.overrides import VLLM_FIELDS, apply
    from codai.config import VllmConfig
    assert "tool_call_parser" in VLLM_FIELDS
    out = apply(VllmConfig(), {"tool_call_parser": "llama3_json"}, VLLM_FIELDS, "vllm")
    assert out.tool_call_parser == "llama3_json"
    # …and the default is the family guess, not "off"
    assert VllmConfig().tool_call_parser == "auto"


def test_that_block_travels_to_a_pod():
    """A pod that does not get it serves the model differently from here."""
    import json
    from codai.api.runpod_worker import seed_model_env
    seed = json.loads(seed_model_env(
        {"id": "m", "path": "x", "vllm": {"tool_call_parser": "hermes"}},
        "m", "vendor/repo"))[0]
    assert seed["vllm"] == {"tool_call_parser": "hermes"}
