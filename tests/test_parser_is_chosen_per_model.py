"""Which parser reads a model's tool calls is a property OF THAT MODEL.

The admin model dialog has had a "Chat template parser" dropdown all along; it
is stored on the model entry and even travels to rented pods in
seed_model_env's keep-list -- and nothing read it, so choosing one changed
nothing at all. Selection was a name guess, and a name cannot tell Qwen2.5's
JSON-in-<tool_call> from current Qwen3's <function=/<parameter= XML.

Precedence, strongest first: the per-model setting, the global setting, the
model's own chat template, the name.
"""
import json

import pytest

from codai.models import chat_style
from codai.models.parser import (GLMParser, GemmaParser, LlamaParser,
                                 MistralParser, ModelParserDispatcher,
                                 QwenParser, ApexBig50Parser)


@pytest.fixture
def entry(monkeypatch):
    """A models.json entry for 'the-model', plus a readable template."""
    state = {"entry": {}, "template": "", "global": "auto"}

    def _entry_for(name):
        return state["entry"] if name in ("the-model", state["entry"].get("path")) else {}
    monkeypatch.setattr("codai.models.manager._model_entry_for", _entry_for)
    monkeypatch.setattr(chat_style, "chat_template_text",
                        lambda path: state["template"])

    class _C:
        pass
    cfg = _C()
    cfg.config = _C()
    monkeypatch.setattr("codai.admin.routes.config_manager", cfg, raising=False)

    def _set(**kw):
        state.update(kw)
        cfg.config.parser = state["global"]
        chat_style.forget()
    _set()
    return _set


XML = "<tool_call>\n<function={{ t.name }}>\n<parameter=k>v</parameter>\n</function>"
JSON_TPL = '<tool_call>\n{"name": "{{ t.name }}", "arguments": {}}\n</tool_call>'
GLM_TPL = "<tool_call>{{ t.name }}<arg_key>k</arg_key><arg_value>v</arg_value>"


def _picked(name="the-model"):
    return type(ModelParserDispatcher(model_name=name).parser).__name__


# ------------------------------------------------------- the per-model setting #

def test_the_models_own_setting_wins(entry):
    """THE BUG: this dropdown existed, was stored, shipped to pods -- and did
    nothing."""
    entry(entry={"parser": "mistral", "path": "org/some-qwen-model"})
    assert _picked() == "MistralParser"


def test_every_value_the_dropdown_offers_resolves(entry):
    for want, cls in (("llama", "LlamaParser"), ("mistral", "MistralParser"),
                      ("chatml", "QwenParser"), ("phi", "PhiParser"),
                      ("gemma", "GemmaParser"), ("qwen", "QwenParser")):
        entry(entry={"parser": want, "path": "org/m"})
        assert _picked() == cls, want


def test_a_style_id_or_class_name_also_works(entry):
    """A setting written by hand in models.json, or copied from the vLLM side."""
    for want, cls in (("qwen3_xml", "QwenParser"), ("glm45", "GLMParser"),
                      ("GLMParser", "GLMParser"), ("deepseek_v3", "DeepSeekParser")):
        entry(entry={"parser": want, "path": "org/m"})
        assert _picked() == cls, want


def test_auto_means_detect(entry):
    entry(entry={"parser": "auto", "path": "org/llama-3.3-70b"})
    assert _picked("org/llama-3.3-70b") == "LlamaParser"


def test_an_unknown_value_falls_back_to_detection(entry):
    """A typo must not silently disable tool calls."""
    entry(entry={"parser": "nonesuch", "path": "org/llama-3.3-70b"})
    assert _picked("org/llama-3.3-70b") == "LlamaParser"


def test_a_plain_alias_now_reaches_its_template(entry):
    """_family_name only resolves an alias for an ADAPTER, so an alias like
    "lisa" named no family and got the catch-all. Its entry still says where
    the weights are, so its template can be read."""
    entry(entry={"path": "org/private-build"}, template=GLM_TPL)
    assert _picked() == "GLMParser"


def test_the_global_setting_applies_when_the_model_says_nothing(entry):
    entry(entry={"path": "org/m"}, **{"global": "glm"})
    assert _picked() == "GLMParser"


def test_a_models_setting_beats_the_global_one(entry):
    entry(entry={"parser": "qwen", "path": "org/m"}, **{"global": "glm"})
    assert _picked() == "QwenParser"


# ------------------------------------------------------------- the template #

def test_a_nameless_model_is_read_from_its_template(entry):
    """Nothing in "the-model" names a family; before this it got the catch-all."""
    entry(entry={"path": "org/private-build"}, template=XML)
    assert _picked() == "QwenParser"


def test_the_template_distinguishes_glm(entry):
    entry(entry={"path": "org/private-build"}, template=GLM_TPL)
    assert _picked() == "GLMParser"


def test_an_unreadable_template_still_gives_the_catch_all(entry):
    entry(entry={"path": "org/private-build"}, template="")
    assert _picked() == "ApexBig50Parser"


def test_a_name_that_names_a_family_keeps_it(entry):
    """A per-family parser carries more than the tool syntax -- gemma's token
    repair, Qwen's repetition guard -- so one template marker must not
    displace a family that matched by name."""
    entry(entry={"path": "google/gemma-4-31B"}, template=JSON_TPL)
    assert _picked("google/gemma-4-31B") == "GemmaParser"


def test_an_adapter_is_read_from_the_base_it_adapts(entry):
    """A LoRA has no template of its own."""
    entry(entry={"loras": [{"name": "a", "path": "sha256:ab"}],
                 "runpod": {"hf_repo": "org/private-build"}}, template=XML)
    assert _picked() == "QwenParser"


# ---------------------------------------------------------------- the cache #

def test_the_template_is_read_once_per_model(monkeypatch):
    reads = []
    monkeypatch.setattr(chat_style, "_from_repo",
                        lambda p: reads.append(p) or XML)
    chat_style.forget()
    assert chat_style.tool_call_style("org/m") == "qwen_xml"
    assert chat_style.tool_call_style("org/m") == "qwen_xml"
    assert reads == ["org/m"]


def test_styles_map_to_both_sides():
    """One table, so the two parser choices cannot drift apart."""
    for style, (vllm_name, coderai_name) in chat_style.STYLES.items():
        assert vllm_name and coderai_name, style
    assert chat_style.STYLES["qwen_xml"][0] == "qwen3_xml"
    assert chat_style.STYLES["hermes_json"][0] == "hermes"
