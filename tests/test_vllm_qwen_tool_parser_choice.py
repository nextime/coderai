"""hermes vs qwen3_xml: the model's template decides, not its name.

2026-10-21, Qwen3.8-27B on a pod: every tool-using request logged
`hermes_tool_parser ... json.decoder.JSONDecodeError: Expecting value` and the
text it choked on was

    <tool_call>
    <function=agenda_udienze>
    <parameter=dal>2026-10-21</parameter>

vLLM reads that with `qwen3_xml`; `hermes` expects JSON inside <tool_call> and
cannot. Buffered, coderai's own parser rescued the call from the text; STREAMED,
vLLM had already consumed those tokens and the call was simply gone.

A name cannot tell the two apart -- Qwen2.5 and the Hermes finetunes really are
hermes -- so the template is read instead.
"""
import json

import pytest

from codai.api import vllm_worker as vw


XML_TPL = ("{%- for tool in tools %}<tool_call>\n<function={{ tool.name }}>\n"
           "<parameter=key>value</parameter>\n</function>\n</tool_call>{%- endfor %}")
JSON_TPL = ('{%- for tc in message.tool_calls %}<tool_call>\n'
            '{"name": "{{ tc.name }}", "arguments": {{ tc.arguments }}}\n</tool_call>{%- endfor %}')


class _Cfg:
    tool_call_parser = "auto"


def _model_dir(tmp_path, template=None, tokenizer_template=None):
    d = tmp_path / "model"
    d.mkdir()
    if template is not None:
        (d / "chat_template.jinja").write_text(template)
    if tokenizer_template is not None:
        (d / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": tokenizer_template}))
    return str(d)


def test_an_xml_template_picks_qwen3_xml(tmp_path):
    """THE BUG: this answered 'hermes' and every call raised JSONDecodeError."""
    path = _model_dir(tmp_path, XML_TPL)
    assert vw._tool_parser_for(_Cfg(), {"id": "qwen3.8-27b"}, path) == "qwen3_xml"


def test_a_json_template_stays_on_hermes(tmp_path):
    path = _model_dir(tmp_path, JSON_TPL)
    assert vw._tool_parser_for(_Cfg(), {"id": "qwen2.5-7b"}, path) == "hermes"


def test_the_template_may_live_in_tokenizer_config(tmp_path):
    path = _model_dir(tmp_path, None, XML_TPL)
    assert vw._tool_parser_for(_Cfg(), {"id": "qwen3-32b"}, path) == "qwen3_xml"


def test_no_template_leaves_the_name_table_alone(tmp_path):
    """Unreadable is not evidence: a model with no template on disk keeps the
    answer it had before, rather than being switched on a failed read."""
    path = _model_dir(tmp_path)
    assert vw._tool_parser_for(_Cfg(), {"id": "qwen2.5-7b"}, path) == "hermes"


def test_an_explicit_setting_still_wins(tmp_path):
    cfg = _Cfg()
    cfg.tool_call_parser = "hermes"
    path = _model_dir(tmp_path, XML_TPL)
    assert vw._tool_parser_for(cfg, {"id": "qwen3.8-27b"}, path) == "hermes"


def test_off_still_means_off(tmp_path):
    cfg = _Cfg()
    cfg.tool_call_parser = "off"
    assert vw._tool_parser_for(cfg, {"id": "qwen3.8-27b"},
                               _model_dir(tmp_path, XML_TPL)) == ""


def test_other_families_are_untouched(tmp_path):
    """Only the hermes answer is second-guessed; a template probe must not
    reroute deepseek or glm, whose parsers were never in question."""
    path = _model_dir(tmp_path, XML_TPL)
    assert vw._tool_parser_for(_Cfg(), {"id": "deepseek-v3"}, path) == "deepseek_v3"
    assert vw._tool_parser_for(_Cfg(), {"id": "glm-4.6"}, path) == "glm45"


def test_qwen3_coder_keeps_its_own_entry(tmp_path):
    path = _model_dir(tmp_path, XML_TPL)
    assert vw._tool_parser_for(_Cfg(), {"id": "qwen3-coder-30b"}, path) == "qwen3_coder"


def test_the_launch_flags_carry_it(tmp_path):
    path = _model_dir(tmp_path, XML_TPL)
    assert vw._tool_args(_Cfg(), {"id": "qwen3.8-27b"}, path) == [
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml"]


def test_a_missing_model_path_is_survivable():
    assert vw._chat_template_text("") == ""
    assert vw._emits_qwen_xml_calls("") is False


def test_an_unreadable_repo_does_not_stop_the_launch(monkeypatch):
    """A parser choice must never raise: a pod that will not boot is worse
    than one that reads tool calls with the wrong parser."""
    def _boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr("codai.api.embeddings._fetch_repo_file", _boom)
    __import__("codai.models.chat_style", fromlist=["forget"]).forget()
    assert vw._chat_template_text("some/repo") == ""
    assert vw._emits_qwen_xml_calls("some/repo") is False
    assert vw._tool_parser_for(_Cfg(), {"id": "qwen2.5-7b"}, "some/repo") == "hermes"


def test_a_repo_template_is_fetched_not_merely_looked_up_in_the_cache(monkeypatch):
    """On a rented pod nothing is cached the first time, so a cache-only
    lookup would report "no template" for every model."""
    seen = []
    def _fetch(repo, name):
        seen.append(name)
        return None
    monkeypatch.setattr("codai.api.embeddings._fetch_repo_file", _fetch)
    __import__("codai.models.chat_style", fromlist=["forget"]).forget()
    vw._chat_template_text("org/qwen3.8-27b")
    assert seen == ["chat_template.jinja", "tokenizer_config.json"]
