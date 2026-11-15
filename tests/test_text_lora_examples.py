"""What a text LoRA actually learns, and how much of it.

Three decisions live in this module and each is a correctness problem:

* the loss is on the ANSWER, not the prompt;
* exclusion ranges over the answer's CHARACTERS must become token labels, and
  an off-by-one token teaches one character of the thing the mask exists to
  exclude -- while looking exactly like it worked;
* an example's weight SCALES its loss and never duplicates the example.

The mask tests include a range whose end lands exactly on a token boundary.
That case is the whole point: on ordinary text the half-open and inclusive
readings of a range agree, so a test built from a real example passes either
way and proves nothing.
"""

import json

import pytest

from codai.api import text_lora_dataset as tld


# --------------------------------------------------------------- a tokenizer --

class Tok:
    """Character-per-token, so char ranges and token spans line up exactly and
    a boundary case can be constructed on purpose."""

    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        out = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        return out + ("[assistant]" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [ord(c) for c in text]
        if not return_offsets_mapping:
            return {"input_ids": ids}
        return {"input_ids": ids,
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class ChunkTok(Tok):
    """Two characters per token: a range ending mid-token, and one ending
    exactly on a boundary, become expressible."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [(i, min(i + 2, len(text))) for i in range(0, len(text), 2)]
        ids = [1000 + i for i, _ in enumerate(spans)]
        if not return_offsets_mapping:
            return {"input_ids": ids}
        return {"input_ids": ids, "offset_mapping": spans}


MSGS = [{"role": "user", "content": "q"}]


def _ex(answer, mask=(), weight=1.0):
    return tld.Example(messages=MSGS, answer=answer,
                       mask=[tuple(m) for m in mask], weight=weight, line=1)


# ------------------------------------------------------- the loss placement --

def test_the_prompt_is_never_learned():
    enc, skip = tld.encode(Tok(), _ex("ab"))
    assert skip is None
    prompt_len = len(enc.input_ids) - 3          # 2 answer chars + eos
    assert set(enc.labels[:prompt_len]) == {-100}
    assert enc.labels[prompt_len:] == [ord("a"), ord("b"), 99]


def test_stopping_is_learned():
    """A model that never learns EOS runs to the token limit on every request."""
    enc, _ = tld.encode(Tok(), _ex("ab"))
    assert enc.labels[-1] == Tok.eos_token_id


def test_the_answer_is_tokenized_on_its_own():
    """Offsets from a single pass over prompt+answer are relative to the joined
    string, which would shift every mask range by the prompt's length."""
    enc, _ = tld.encode(Tok(), _ex("abc", mask=[[0, 1]]))
    tail = enc.labels[-4:]                        # a, b, c, eos
    assert tail == [-100, ord("b"), ord("c"), 99]


# ------------------------------------------------------------- the mask ------

def test_an_excluded_range_leaves_the_loss():
    enc, _ = tld.encode(Tok(), _ex("abcdef", mask=[[2, 4]]))
    answer = enc.labels[-7:-1]
    assert answer == [ord("a"), ord("b"), -100, -100, ord("e"), ord("f")]
    assert enc.n_masked == 2


def test_two_disjoint_ranges():
    enc, _ = tld.encode(Tok(), _ex("abcdef", mask=[[0, 1], [4, 6]]))
    assert enc.labels[-7:-1] == [-100, ord("b"), ord("c"), ord("d"), -100, -100]


def test_a_range_ending_exactly_on_a_token_boundary_excludes_no_more():
    """THE case. [0,4) over 2-char tokens covers exactly the first two tokens;
    reading `end` as inclusive would eat a third. On text where a range ends
    mid-token both readings agree, so only this shape can tell them apart."""
    enc, _ = tld.encode(ChunkTok(), _ex("abcdefgh", mask=[[0, 4]]))
    answer = [l for l in enc.labels if l != -100 or True][-5:-1]   # 4 answer tokens
    assert answer == [-100, -100, 1002, 1003], answer


def test_a_token_straddling_the_start_is_excluded_whole():
    """Conservative on purpose: that token usually carries the space before the
    excluded passage, and losing it costs nothing, while keeping it would teach
    the passage's first characters."""
    enc, _ = tld.encode(ChunkTok(), _ex("abcdefgh", mask=[[3, 6]]))
    answer = enc.labels[-5:-1]
    # token 1 spans [2,4) and overlaps [3,6) -> out; token 2 spans [4,6) -> out
    assert answer == [1000, -100, -100, 1003], answer


def test_an_empty_range_masks_nothing():
    enc, _ = tld.encode(Tok(), _ex("abc", mask=[[1, 1]]))
    assert enc.n_masked == 0


def test_a_fully_masked_answer_is_skipped_not_trained_on_nothing():
    """With every answer token excluded, the trailing EOS is STILL a trained
    position -- so an example that teaches only "stop immediately after the
    prompt" sails past a naive "is anything left?" check and trains empty
    answers. It has to be skipped on the answer's labels, not the sequence's."""
    enc, skip = tld.encode(Tok(), _ex("abc", mask=[[0, 3]]))
    assert enc is None
    assert "empty answer" in skip.reason


def test_a_range_past_the_end_is_refused_not_clamped():
    """It means the mask was computed against different text. Clamping would
    mask the wrong characters and say nothing."""
    with pytest.raises(tld.DatasetError, match="runs past the answer"):
        tld.parse_example({"messages": MSGS, "answer": "abc", "mask": [[1, 9]]}, line=7)


@pytest.mark.parametrize("bad,msg", [
    ([[2, 1]], "not ordered"),
    ([[-1, 2]], "not ordered"),
    ([[1]], "must be"),
    ("nope", "must be a list"),
    ([["a", "b"]], "must be integers"),
])
def test_a_malformed_mask_names_the_line(bad, msg):
    with pytest.raises(tld.DatasetError) as e:
        tld.parse_example({"messages": MSGS, "answer": "abcd", "mask": bad}, line=42)
    assert "line 42" in str(e.value) and msg in str(e.value)


# ------------------------------------------------------------ the weight -----

def test_the_weight_scales_and_never_duplicates():
    examples = [_ex("ab", weight=w) for w in (1.0, 0.3333, 0.1667)]
    out, st = tld.prepare(Tok(), examples)
    assert len(out) == 3, "one encoded example per input, whatever the weight"
    assert [round(e.weight, 4) for e in out] == [1.0, 0.3333, 0.1667]


def test_a_weight_above_one_also_scales():
    """The case that used to cost a silent 0.5: an integer number of copies at
    full scale, with the fraction dropped."""
    out, _ = tld.prepare(Tok(), [_ex("ab", weight=2.5)])
    assert len(out) == 1 and out[0].weight == 2.5


def test_the_divisor_is_fixed_not_the_batch_weight_sum():
    """Normalising by the batch would re-scale the weights away: a batch of all
    0.1667 would train identically to a batch of all 1.0."""
    assert tld.loss_scale(1.0, 4) == pytest.approx(0.25)
    assert tld.loss_scale(0.1667, 4) == pytest.approx(0.041675)
    # the ratio between two weights survives, which is all they mean
    assert (tld.loss_scale(1.0, 4) / tld.loss_scale(0.3333, 4)
            == pytest.approx(1.0 / 0.3333))


def test_a_zero_weight_is_dropped_not_trained_at_zero():
    rows = [{"messages": MSGS, "answer": "ab", "weight": 0},
            {"messages": MSGS, "answer": "cd", "weight": 1}]
    kept = list(_jsonl(rows))
    assert [e.answer for e in kept] == ["cd"]


def test_a_negative_weight_is_refused():
    with pytest.raises(tld.DatasetError, match="negative"):
        tld.parse_example({"messages": MSGS, "answer": "ab", "weight": -1}, line=3)


# ------------------------------------------------- over-long, and reporting --

def test_an_over_long_example_is_skipped_with_its_size():
    enc, skip = tld.encode(Tok(), _ex("a" * 50), max_len=20)
    assert enc is None
    assert skip.n_tokens > 20 and "over the 20 limit" in skip.reason
    assert "prompt +" in skip.reason, "say which half is big"


def test_the_answer_is_never_truncated_to_fit():
    """Truncating teaches a cut-off answer, which is worse than not training on
    the example at all -- and it would be invisible."""
    long = _ex("a" * 50)
    enc, skip = tld.encode(Tok(), long, max_len=20)
    assert enc is None and skip is not None


def test_skips_are_returned_not_merely_logged():
    """A silently shrinking dataset makes 'it ran' indistinguishable from 'it
    ran on a third of the data'."""
    examples = [_ex("ab"), _ex("c" * 100), _ex("de")]
    out, st = tld.prepare(Tok(), examples, max_len=30)
    assert st.examples == 2 and len(st.skipped) == 1
    assert "SKIPPED" in st.describe()
    assert st.trained_tokens == 2 * 3          # two answer chars + eos, twice


def test_stats_count_what_the_mask_removed():
    out, st = tld.prepare(Tok(), [_ex("abcdef", mask=[[0, 2]])])
    assert st.masked_tokens == 2 and st.trained_tokens == 5   # 4 chars + eos


# --------------------------------------------------------------- loading -----

def _jsonl(rows, tmp=None, **kw):
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return tld.load_jsonl(path, **kw)


def test_a_dataset_whose_fields_are_named_differently_is_mapped_not_rewritten():
    """A 100 MB JSONL should not be regenerated to be trainable, and a renaming
    pass is a chance to corrupt it."""
    rows = [{"msgs": MSGS, "risposta": "ab", "maschera": [[0, 1]], "peso": 0.5}]
    got = list(_jsonl(rows, field_map={"messages": "msgs", "answer": "risposta",
                                       "mask": "maschera", "weight": "peso"}))
    assert len(got) == 1
    assert got[0].answer == "ab" and got[0].weight == 0.5
    assert got[0].mask == [(0, 1)]


def test_an_answer_given_as_a_message_object_is_accepted():
    rows = [{"messages": MSGS, "answer": {"role": "assistant", "content": "ab"}}]
    assert list(_jsonl(rows))[0].answer == "ab"


def test_a_missing_answer_names_the_line():
    rows = [{"messages": MSGS, "answer": "ok"}, {"messages": MSGS}]
    with pytest.raises(tld.DatasetError, match="line 2"):
        list(_jsonl(rows))


def test_broken_json_names_the_line():
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"messages": MSGS, "answer": "a"}) + "\n")
        fh.write("{not json\n")
    with pytest.raises(tld.DatasetError, match="line 2"):
        list(tld.load_jsonl(path))


def test_the_masked_text_can_be_read_back_for_a_check():
    ex = _ex("Il danno. Esito: x.", mask=[[10, 19]])
    assert ex.masked_text() == ["Esito: x."]
