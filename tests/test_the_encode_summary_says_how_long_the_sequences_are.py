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

"""The encode summary omitted the one length that decides whether a step fits.

`trained_tokens` counts answer positions that contribute to the loss and
`masked_tokens` counts answer positions excluded from it. The PROMPT is in
neither. So for a set whose prompts run to thousands of tokens the summary
read `7308 examples, 2868079 trained tokens, 34542 masked away` -- 397 tokens
an example -- while the sequences actually fed to the model were about
5,000 tokens each.

That mattered: 397 tokens makes an out-of-memory theory absurd, and ~5,000
tokens over a 152k vocab makes it arithmetic. The summary now states the
sequence length, because that is the number a reader is using it for.
"""

import pytest

from codai.api import text_lora_dataset as tld


class _Tok:
    """The character-per-token fake the dataset tests already use.

    Borrowed rather than reinvented: a tokeniser fake that does not return
    `offset_mapping` cannot encode at all, and one char per token makes the
    sequence length countable by hand.
    """

    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        out = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        return out + ("[assistant]" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False):
        ids = [ord(c) for c in text]
        if not return_offsets_mapping:
            return {"input_ids": ids}
        return {"input_ids": ids,
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def _example(prompt_chars, answer_chars, line=1):
    """One example with a prompt of `prompt_chars` and an answer of
    `answer_chars`, each character being exactly one token."""
    return tld.Example(
        messages=[{"role": "user", "content": "p" * prompt_chars}],
        answer="a" * answer_chars,
        mask=[], weight=1.0, line=line)


def _stats(*examples):
    _enc, st = tld.prepare(_Tok(), list(examples), max_len=8192)
    return st


# ------------------------------------------------------------------ #

def test_the_summary_states_the_sequence_length():
    st = _stats(_example(50, 5))
    said = st.describe()
    assert "per sequence" in said, said


def test_a_long_prompt_shows_up_even_though_it_trains_nothing():
    """The whole point: prompt tokens are in neither existing counter."""
    short = _stats(_example(5, 5))
    long = _stats(_example(500, 5))
    assert long.trained_tokens == short.trained_tokens, \
        "the answer is the same, so the old counters cannot tell these apart"
    assert long.masked_tokens == short.masked_tokens
    assert long.seq_tokens > short.seq_tokens * 10, \
        "but the sequence the model forwards is vastly longer"
    assert "per sequence" in long.describe()


def test_the_longest_sequence_is_reported_not_only_the_mean():
    """A mean hides the example that OOMs; the maximum is what blows up."""
    st = _stats(_example(10, 5, line=1), _example(1000, 5, line=2))
    assert st.seq_tokens_max > st.seq_tokens // st.examples, \
        "the tail has to be visible separately from the average"
    assert str(st.seq_tokens_max) in st.describe()


def test_the_length_counts_the_whole_sequence():
    """Prompt + answer + eos, which is what one forward pass costs."""
    st = _stats(_example(20, 4))
    # "[user]" + 20 p + "[assistant]" + 4 a + eos, one token per character.
    expected = len("[user]") + 20 + len("[assistant]") + 4 + 1
    assert st.seq_tokens == expected, (st.seq_tokens, expected)
    assert st.seq_tokens_max == expected


def test_an_empty_pass_says_nothing_about_length():
    """No examples, no division by zero and no invented number."""
    st = tld.Stats()
    said = st.describe()
    assert "per sequence" not in said, said
    assert "0 examples" in said


def test_the_old_counters_are_unchanged():
    """This adds a number; it must not move the two already there."""
    st = _stats(_example(20, 7))
    assert st.trained_tokens > 0
    assert f"{st.trained_tokens} trained tokens" in st.describe()
    assert f"{st.masked_tokens} masked away" in st.describe()


def test_skips_are_still_reported_loudly():
    st = _stats(_example(20, 7))
    st.skipped.append(tld.Skipped(line=9, n_tokens=99999, reason="too long"))
    assert "1 SKIPPED" in st.describe()
