"""A decision example is prompt-only and scored at the answer letter.

The decision objective (rizzo-flow / Spark-X2.5 style) does not learn an answer
STRING -- it matches the model's probability over a few answer-letter tokens to
a target distribution, read at the final prompt position. So the encoding must:

* append NOTHING after the prompt -- no answer tokens, no EOS -- or the scored
  "final position" would be the EOS, not the letter;
* map each candidate letter to exactly ONE token id, because the loss reads one
  logit per letter and the server scores one logprob per letter; a letter that
  is two tokens breaks the contract and must fail loudly;
* refuse a dataset whose records and whose objective disagree, rather than
  silently training the wrong thing.
"""

import pytest

from codai.api import text_lora_dataset as tld


class Tok:
    """Character-per-token: a letter 'A' is the single token ord('A')."""

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


MSGS = [{"role": "user", "content": "pick"}]


def _dex(decision, weight=1.0):
    return tld.Example(messages=MSGS, answer="", weight=weight, line=7,
                       decision=dict(decision))


# ------------------------------------------------------------- the encoding --

def test_no_answer_or_eos_is_appended():
    enc, skip = tld.encode_decision(Tok(), _dex({"A": 0.7, "B": 0.3}))
    assert skip is None
    prompt = Tok().apply_chat_template(MSGS, add_generation_prompt=True)
    # Exact equality IS the "nothing appended" proof: no answer token, no EOS.
    assert enc.input_ids == [ord(c) for c in prompt]
    assert enc.labels == []                       # nothing supervised by position
    assert enc.objective == "decision"


def test_letters_map_to_single_token_ids_and_normalise():
    enc, _ = tld.encode_decision(Tok(), _dex({"A": 3.0, "B": 1.0}))
    assert enc.letter_token_ids == [ord("A"), ord("B")]
    assert enc.target_probs == pytest.approx([0.75, 0.25])


def test_the_scored_position_is_the_last_prompt_token():
    enc, _ = tld.encode_decision(Tok(), _dex({"A": 1.0}))
    # The last input token is the start of the assistant turn the model answers
    # from; n_trained is the single scored position.
    assert enc.n_trained == 1
    assert enc.n_masked == 0


def test_a_multitoken_letter_is_refused():
    # Under char-per-token, a two-character 'letter' is two tokens.
    with pytest.raises(tld.DatasetError) as e:
        tld.encode_decision(Tok(), _dex({"AB": 1.0}))
    assert "one" in str(e.value).lower()


def test_two_letters_sharing_a_token_are_refused():
    class Collide(Tok):
        def __call__(self, text, add_special_tokens=False,
                     return_offsets_mapping=False):
            # every letter maps to the same id
            if not return_offsets_mapping:
                return {"input_ids": [5] * len(text)}
            return {"input_ids": [5] * len(text),
                    "offset_mapping": [(i, i + 1) for i in range(len(text))]}
    with pytest.raises(tld.DatasetError):
        tld.encode_decision(Collide(), _dex({"A": 0.5, "B": 0.5}))


def test_a_prompt_over_the_limit_is_skipped_not_truncated():
    enc, skip = tld.encode_decision(Tok(), _dex({"A": 1.0}), max_len=3)
    assert enc is None and skip is not None
    assert "over the 3" in skip.reason


# --------------------------------------------------------------- parsing -----

def test_parse_reads_a_decision_record():
    ex = tld.parse_example({"messages": MSGS, "decision": {"A": 0.7, "B": 0.3}})
    assert ex.decision == {"A": 0.7, "B": 0.3}
    assert ex.answer == ""


def test_an_all_zero_decision_target_is_refused():
    with pytest.raises(tld.DatasetError):
        tld.parse_example({"messages": MSGS, "decision": {"A": 0.0, "B": 0.0}})


def test_a_negative_decision_probability_is_refused():
    with pytest.raises(tld.DatasetError):
        tld.parse_example({"messages": MSGS, "decision": {"A": -1.0, "B": 2.0}})


# ---------------------------------------------- objective/dataset agreement --

def test_a_decision_record_under_an_sft_job_fails_loudly():
    exs = [tld.parse_example({"messages": MSGS, "decision": {"A": 1.0}}, line=1)]
    with pytest.raises(tld.DatasetError):
        tld.prepare(Tok(), exs, objective="sft")


def test_an_sft_record_under_a_decision_job_fails_loudly():
    exs = [tld.parse_example({"messages": MSGS, "answer": "hello"}, line=1)]
    with pytest.raises(tld.DatasetError):
        tld.prepare(Tok(), exs, objective="decision")


def test_prepare_counts_decision_examples():
    exs = [tld.parse_example({"messages": MSGS, "decision": {"A": 0.6, "B": 0.4}},
                             line=1)]
    enc, st = tld.prepare(Tok(), exs, objective="decision")
    assert st.examples == 1 and st.decision_examples == 1
    assert enc[0].objective == "decision"
