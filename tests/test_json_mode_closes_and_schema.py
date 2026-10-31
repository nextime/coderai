"""JSON mode returned JSON that would not parse.

Measured on zeiss 2026-10-31: 21 script generations ended `finish_reason=length`
after writing 6000 tokens of well-formed-but-unfinished JSON for a job needing
about 800. One ran 8.5 minutes. The client reported them as "broken JSON after
1493 s" and "second run had no scenes".

Nothing was broken in the usual sense. `response_format: {"type":"json_object"}`
is honoured — llama.cpp applies its generic JSON grammar — but that grammar
constrains SYNTAX only. It never requires the root object to close, so an array
of scenes can grow until the token budget ends mid-token. `max_tokens` was the
caller's own 6000, the prompt was 2414 tokens and the model's n_ctx is 32000, so
no limit was misconfigured and nothing was truncated by the server.

Two fixes, neither of which touches the caller's max_tokens:

1. A SCHEMA bounds the shape, so the object must close. llama-cpp-python accepts
   only `type: json_object` and reads a sibling `schema` key, while OpenAI nests
   the schema under `json_schema` — so the shapes are translated, not forwarded.
   Forwarding OpenAI's form verbatim would silently DISABLE the grammar
   (`_grammar_for_response_format` returns None for any other type), which is
   worse than not supporting it.

2. When the budget is hit anyway, the open structures are closed so the caller
   gets valid JSON holding the items that did arrive. `finish_reason` stays
   `length`, so a caller that checks still learns the answer was cut short.
"""

import io
import json
import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = io.open(os.path.join(ROOT, 'codai/backends/vulkan.py'), encoding='utf-8').read()


@pytest.fixture(scope='module')
def closer():
    """The helper, lifted out so the test does not need a loaded model."""
    i = SRC.index('    def close_truncated_json(text: str):')
    j = SRC.index('    def _render_chat_prompt(self, messages):')
    fn = textwrap.dedent(SRC[i:j]).replace('@staticmethod\n', '')
    ns = {}
    exec(fn, ns)
    return ns['close_truncated_json']


# ── closing a truncated document ─────────────────────────────────────────────

def test_the_real_shape_a_cut_scene_array(closer):
    out = closer('{"title":"x","scenes":[{"text":"one"},{"text":"two"},{"te')
    assert out is not None
    d = json.loads(out)
    assert len(d['scenes']) == 2, 'the half-written scene should be dropped'
    assert d['title'] == 'x'


def test_a_string_cut_mid_word_is_closed(closer):
    out = closer('{"title":"Perche i polpi sono intellige')
    assert json.loads(out)['title'].startswith('Perche')


def test_a_trailing_comma_is_removed(closer):
    assert len(json.loads(closer('{"a":[{"x":1},'))['a']) == 1


def test_deep_nesting_is_closed(closer):
    assert json.loads(closer('{"a":{"b":{"c":[1,2,3'))['a']['b']['c'] == [1, 2]


def test_valid_json_is_left_alone(closer):
    assert closer('{"a": 1}') is None


def test_prose_is_left_alone(closer):
    """A model that ignored JSON mode entirely must not be "repaired" into
    something that parses but means nothing."""
    assert closer('Sure! Here is your script:') is None


def test_empty_is_left_alone(closer):
    assert closer('') is None and closer('   ') is None


def test_an_unverifiable_repair_is_not_applied(closer):
    """The function parses its own output before returning it; if it cannot be
    made to parse, the original is left for the caller to see."""
    assert closer('{"a": }') is None


def test_it_never_invents_content(closer):
    out = closer('{"title":"x","scenes":[{"text":"one"},{"text":"two"},{"te')
    assert 'null' not in out and 'two"},{' not in out.replace('{"text":"two"}', '')


# ── gating ───────────────────────────────────────────────────────────────────

def test_the_repair_only_runs_for_a_json_request():
    i = SRC.index('JSON response hit max_tokens mid-structure')
    block = SRC[max(0, i - 1200):i]
    assert "kwargs.get('response_format')" in block


def test_the_repair_only_runs_on_a_length_finish():
    i = SRC.index('JSON response hit max_tokens mid-structure')
    block = SRC[max(0, i - 1200):i]
    assert "_fr == 'length'" in block


def test_finish_reason_is_not_rewritten():
    """The caller must still be able to tell the answer was cut short."""
    i = SRC.index('JSON response hit max_tokens mid-structure')
    block = SRC[max(0, i - 1400):i + 600]
    assert "finish_reason" not in block.split('_fixed = ')[1].split('return content')[0] \
        or "stays 'length'" in block


def test_a_failure_in_the_repair_cannot_fail_the_request():
    i = SRC.index('JSON response hit max_tokens mid-structure')
    block = SRC[max(0, i - 1200):i + 900]
    assert 'except Exception' in block and 'close-up skipped' in block


# ── schema support ───────────────────────────────────────────────────────────

def test_both_response_format_shapes_are_accepted():
    i = SRC.index("_rf.get('type') in ('json_object', 'json_schema')")
    assert i > 0


def test_openais_nested_schema_is_translated_not_forwarded():
    """llama-cpp-python returns NO grammar for any type but json_object, so
    forwarding OpenAI's `json_schema` verbatim would disable the constraint."""
    i = SRC.index("_rf.get('type') in ('json_object', 'json_schema')")
    block = SRC[i:i + 900]
    assert "_js.get('schema')" in block
    assert "{'type': 'json_object', 'schema': _schema}" in block


def test_llamacpps_own_shape_still_works():
    i = SRC.index("_rf.get('type') in ('json_object', 'json_schema')")
    block = SRC[i:i + 900]
    assert "_rf.get('schema')" in block


def test_a_schemaless_json_object_request_is_unchanged():
    i = SRC.index("_rf.get('type') in ('json_object', 'json_schema')")
    block = SRC[i:i + 900]
    assert "else {'type': 'json_object'})" in block


def test_why_the_translation_matters_is_recorded():
    i = SRC.index("_rf.get('type') in ('json_object', 'json_schema')")
    head = SRC[max(0, i - 1800):i]
    assert 'finish_reason=length' in head
    assert 'silently disables the' in head or 'silently DISABLE' in head
