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

"""Examples for text (causal-LM) LoRA training: what is learned and how much.

Three things make a supervised fine-tune of a chat model different from "train
on the string", and each one is a correctness problem rather than a nicety.

**The loss belongs on the answer, not the prompt.** The prompt is given at
inference; learning to produce it teaches nothing and spends the budget on the
wrong tokens. Prompt positions are labelled ``-100``.

**Some of the answer must not be learned either.** A dataset that knows which
passages are unreliable carries them as EXCLUSION RANGES, and a trainer that
ignores them teaches exactly the values the curator decided to exclude. The
ranges are over CHARACTERS of the answer and the model sees TOKENS, so they have
to be mapped after tokenization -- and a mapping that is off by one token
teaches one character of the excluded text while looking like it worked. The
convention is half-open, ``[start, end)``, the same as a Python slice, and a
token overlapping a range AT ALL leaves the loss: over-excluding costs a little
signal, under-excluding costs the thing the mask was for.

**An example's weight scales its loss; it never duplicates the example.**
A weight says what a lesson is worth, so a 40-token answer and a 4000-token one
with the same weight must count the same. Duplicating an example to weight it
multiplies GPU hours for one lesson, and a weight of 2.5 either loses the 0.5 or
needs a fractional copy of a thing that cannot be fractional. Scaling is exact
for every weight, including those above 1.

Nothing here imports torch: this is the part that decides what is learned, and
it is worth being able to test without a GPU in the room.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

#: Canonical field names. A dataset that calls them something else is mapped at
#: load time (``field_map``) rather than rewritten -- a 100 MB JSONL should not
#: have to be regenerated to be trainable, and a renaming pass is a chance to
#: corrupt it.
FIELDS = ("messages", "answer", "mask", "weight", "decision")


class DatasetError(ValueError):
    """A malformed example, named with its line number.

    Loudly, and with the line: a trainer that skips what it cannot parse
    silently trains on a smaller dataset than anyone thinks, and "it ran" is
    indistinguishable from "it ran on a third of the data".
    """


@dataclasses.dataclass
class Example:
    """One supervised example.

    ``mask`` holds half-open character ranges over ``answer`` that must NOT be
    learned. ``weight`` scales this example's loss.
    """

    messages: List[Dict[str, Any]]
    answer: str
    mask: List[Tuple[int, int]] = dataclasses.field(default_factory=list)
    weight: float = 1.0
    line: int = 0
    #: A decision example instead of a supervised-text one. When set, this maps
    #: candidate answer-letter strings to a target probability, and training is
    #: a soft cross-entropy over those letters' logits at the final prompt
    #: position -- NOT next-token learning of ``answer``. See ``encode_decision``
    #: and the ``decision`` objective in ``text_lora_trainer``.
    decision: Optional[Dict[str, float]] = None

    def masked_text(self) -> List[str]:
        """The excluded substrings, for a log line or an eyeball check."""
        return [self.answer[a:b] for a, b in self.mask]


@dataclasses.dataclass
class Encoded:
    """One example as token ids and labels, ready for a collator."""

    input_ids: List[int]
    labels: List[int]
    weight: float
    n_trained: int          # positions contributing to the loss
    n_masked: int           # answer positions excluded by the mask
    line: int = 0
    #: ``"sft"`` (next-token, uses ``labels``) or ``"decision"`` (soft
    #: cross-entropy over ``letter_token_ids`` with ``target_probs`` at the
    #: final prompt position; ``labels`` is empty). Kept on the encoded example
    #: so the step loop needs no second structure to tell them apart.
    objective: str = "sft"
    letter_token_ids: List[int] = dataclasses.field(default_factory=list)
    target_probs: List[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Skipped:
    """An example that was not encoded, and why.

    Returned rather than logged-and-dropped: a shrinking dataset is a silent
    change to what is being trained, so the caller gets a count it can report
    and refuse to proceed on.
    """

    line: int
    reason: str
    n_tokens: int = 0


def _coerce_mask(raw: Any, answer_len: int, line: int) -> List[Tuple[int, int]]:
    """Validate exclusion ranges: half-open, in order, inside the answer."""
    out: List[Tuple[int, int]] = []
    if raw is None:
        return out
    if not isinstance(raw, (list, tuple)):
        raise DatasetError(f"line {line}: mask must be a list of [start, end]")
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise DatasetError(
                f"line {line}: each mask range must be [start, end], got {item!r}")
        try:
            a, b = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            raise DatasetError(f"line {line}: mask bounds must be integers: {item!r}")
        if a < 0 or b < a:
            raise DatasetError(f"line {line}: mask range [{a}, {b}) is not ordered")
        if b > answer_len:
            # Not clamped. A range past the end means the mask was computed
            # against a different string than the one in this record, and
            # clamping it would mask the wrong characters with no complaint.
            raise DatasetError(
                f"line {line}: mask range [{a}, {b}) runs past the answer "
                f"({answer_len} chars) -- it was computed against different text")
        if b > a:
            out.append((a, b))
    return out


def _coerce_decision(raw: Any, line: int) -> Dict[str, float]:
    """Validate a decision target: a non-empty map of answer-letter -> weight.

    The values are a distribution over the candidate letters; they need not sum
    to one here (``encode_decision`` normalises), but they must be non-negative
    and not all zero -- a target of all-zeros has no gradient and is a dataset
    bug, not a lesson. The keys are the letter STRINGS as they must tokenize at
    inference; that each is a single token is checked at encode time, where the
    tokenizer is in hand.
    """
    if not isinstance(raw, dict) or not raw:
        raise DatasetError(
            f"line {line}: 'decision' must be a non-empty object mapping an "
            f"answer letter to its target probability")
    out: Dict[str, float] = {}
    for letter, p in raw.items():
        if not isinstance(letter, str) or not letter:
            raise DatasetError(
                f"line {line}: decision keys must be non-empty strings: {letter!r}")
        try:
            val = float(p)
        except (TypeError, ValueError):
            raise DatasetError(
                f"line {line}: decision[{letter!r}] must be a number, got {p!r}")
        if val < 0:
            raise DatasetError(
                f"line {line}: decision[{letter!r}] is negative ({val})")
        out[letter] = val
    if sum(out.values()) <= 0:
        raise DatasetError(
            f"line {line}: decision target is all zeros -- no letter carries any "
            f"probability, so there is nothing to learn")
    return out


def parse_example(obj: Dict[str, Any], *, line: int = 0,
                  field_map: Optional[Dict[str, str]] = None) -> Example:
    """One JSONL record to an :class:`Example`, or raise :class:`DatasetError`."""
    fm = {k: k for k in FIELDS}
    fm.update(field_map or {})
    if not isinstance(obj, dict):
        raise DatasetError(f"line {line}: expected a JSON object")

    messages = obj.get(fm["messages"])
    if not isinstance(messages, list) or not messages:
        raise DatasetError(f"line {line}: {fm['messages']!r} must be a non-empty list")
    for m in messages:
        if not isinstance(m, dict) or "role" not in m:
            raise DatasetError(f"line {line}: each message needs a role: {m!r}")

    weight_raw = obj.get(fm["weight"], 1.0)
    try:
        weight = float(1.0 if weight_raw is None else weight_raw)
    except (TypeError, ValueError):
        raise DatasetError(f"line {line}: {fm['weight']!r} must be a number")
    if weight < 0:
        raise DatasetError(f"line {line}: weight {weight} is negative")

    # A decision example carries a letter->probability target and NO supervised
    # answer text: the lesson is the distribution over the candidate letters,
    # not a string to reproduce. `answer`/`mask` are not read in this case.
    decision_raw = obj.get(fm["decision"])
    if decision_raw is not None:
        return Example(messages=list(messages), answer="",
                       weight=weight, line=line,
                       decision=_coerce_decision(decision_raw, line))

    answer = obj.get(fm["answer"])
    if isinstance(answer, dict):           # a message object rather than a string
        answer = answer.get("content")
    if not isinstance(answer, str) or not answer:
        raise DatasetError(f"line {line}: {fm['answer']!r} must be a non-empty string")

    return Example(messages=list(messages), answer=answer,
                   mask=_coerce_mask(obj.get(fm["mask"]), len(answer), line),
                   weight=weight, line=line)


def load_jsonl(path: str, *, field_map: Optional[Dict[str, str]] = None,
               skip_zero_weight: bool = True) -> Iterator[Example]:
    """Stream examples from a JSONL file.

    Streamed, not read whole: these datasets run to hundreds of megabytes and
    the trainer does not need them resident.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception as exc:
                raise DatasetError(f"line {n}: not valid JSON: {exc}")
            ex = parse_example(obj, line=n, field_map=field_map)
            if skip_zero_weight and ex.weight == 0:
                continue      # a weight of zero is "do not teach this", not an error
            yield ex


def token_labels_for_answer(offsets: Sequence[Tuple[int, int]],
                            ids: Sequence[int],
                            mask: Sequence[Tuple[int, int]]) -> List[int]:
    """Answer token labels, with masked ranges replaced by ``-100``.

    A token spanning ``[a, b)`` is excluded when it overlaps any excluded range
    ``[ma, mb)`` -- ``a < mb and b > ma``. Half-open on both sides, so a range
    ending exactly where a token begins does NOT exclude that token, and a range
    is a Python slice of the answer.

    Any overlap excludes the whole token, which is the conservative direction:
    a token straddling the start of an excluded passage usually carries the
    space before it, and losing that costs nothing, while keeping it would teach
    the first characters of what was excluded.
    """
    out: List[int] = []
    for (a, b), tok in zip(offsets, ids):
        excluded = any(a < mb and b > ma for ma, mb in mask)
        out.append(-100 if excluded else tok)
    return out


def encode(tokenizer, ex: Example, *, max_len: int = 8192,
           enable_thinking: bool = False,
           add_eos: bool = True) -> Tuple[Optional[Encoded], Optional[Skipped]]:
    """One example to ids and labels. Exactly one of the two is None.

    The prompt is rendered with the chat template and
    ``add_generation_prompt=True``, so training sees the string inference will
    see up to the point the model starts answering.

    The answer is tokenized SEPARATELY, with offsets, and the ids concatenated.
    That is deliberate: offsets from a single pass over prompt+answer are
    relative to the joined string, and the token at the seam can merge the end
    of the prompt with the start of the answer -- which would put a mask range
    off by the length of the prompt, or mask a token that is half prompt.
    Tokenizing the answer alone makes the offsets mean what the ranges mean.

    An example longer than ``max_len`` is SKIPPED, never truncated. Truncating
    the answer would silently teach a cut-off answer; truncating the prompt
    would teach against a prompt inference will not send. Skipping is reported.
    """
    prompt = tokenizer.apply_chat_template(
        ex.messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    enc = tokenizer(ex.answer, add_special_tokens=False,
                    return_offsets_mapping=True)
    answer_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    answer_labels = token_labels_for_answer(offsets, answer_ids, ex.mask)

    tail_ids: List[int] = []
    tail_labels: List[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if add_eos and eos is not None:
        # Trained, not masked: stopping is part of the answer, and a model that
        # never learns to emit EOS runs to the token limit on every request.
        tail_ids, tail_labels = [eos], [eos]

    ids = list(prompt_ids) + list(answer_ids) + tail_ids
    labels = [-100] * len(prompt_ids) + answer_labels + tail_labels

    if len(ids) > max_len:
        return None, Skipped(line=ex.line, n_tokens=len(ids),
                             reason=(f"{len(ids)} tokens over the {max_len} limit "
                                     f"({len(prompt_ids)} prompt + "
                                     f"{len(answer_ids)} answer)"))

    # Counted over the ANSWER, not the whole sequence: with every answer token
    # excluded, the trailing EOS is still a trained position, so a naive
    # "anything left?" check passes and the example teaches exactly one thing --
    # stop immediately after the prompt. That trains empty answers.
    if not any(l != -100 for l in answer_labels):
        return None, Skipped(line=ex.line, n_tokens=len(ids),
                             reason=("every answer token is excluded by the mask: "
                                     "the only trainable position left is the end "
                                     "of the answer, which teaches an empty answer"))
    n_trained = sum(1 for l in labels if l != -100)
    return Encoded(input_ids=ids, labels=labels, weight=ex.weight,
                   n_trained=n_trained,
                   n_masked=sum(1 for l in answer_labels if l == -100),
                   line=ex.line), None


def _letter_token_id(tokenizer, letter: str, line: int) -> int:
    """The single token id a candidate answer letter must tokenize to.

    The decision objective reads the logit of ONE token per letter at the final
    prompt position, and the serving side scores the SAME single token from the
    model's first generated step. A letter that tokenizes to two ids has no one
    logit to read and no one logprob to score, so it breaks the contract on both
    sides -- a loud failure here is far better than a silently-wrong target.
    """
    ids = tokenizer(letter, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise DatasetError(
            f"line {line}: decision letter {letter!r} tokenizes to {len(ids)} "
            f"tokens ({ids}), not one -- the decision objective scores a single "
            f"answer-letter token, so each candidate must be exactly one token "
            f"for this tokenizer (try the letter the model actually emits, e.g. "
            f"'A' vs ' A')")
    return int(ids[0])


def encode_decision(tokenizer, ex: Example, *, max_len: int = 8192,
                    enable_thinking: bool = False,
                    ) -> Tuple[Optional[Encoded], Optional[Skipped]]:
    """One decision example to a prompt-only encoding. Exactly one of two is None.

    The prompt is rendered with ``add_generation_prompt=True`` and NOTHING is
    appended -- no answer token, no EOS. The scored position is therefore the
    last prompt token: ``logits[:, -1, :]`` is the model's distribution over the
    token it would generate first, which is exactly the answer letter. The loss
    (in ``text_lora_trainer.decision_distillation_loss``) is a soft cross-entropy
    between ``target_probs`` and the softmax over ``letter_token_ids`` there.

    An example longer than ``max_len`` is SKIPPED, never truncated, for the same
    reason as :func:`encode`: a prompt inference will not send teaches nothing.
    """
    prompt = tokenizer.apply_chat_template(
        ex.messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    if len(prompt_ids) > max_len:
        return None, Skipped(line=ex.line, n_tokens=len(prompt_ids),
                             reason=(f"{len(prompt_ids)} prompt tokens over the "
                                     f"{max_len} limit"))

    letters = list((ex.decision or {}).keys())
    token_ids = [_letter_token_id(tokenizer, l, ex.line) for l in letters]
    if len(set(token_ids)) != len(token_ids):
        # Two letters sharing a token id would have their targets summed into one
        # logit, quietly training a different distribution than the record states.
        raise DatasetError(
            f"line {ex.line}: two decision letters map to the same token id "
            f"({dict(zip(letters, token_ids))}) -- their targets cannot be told "
            f"apart at the scored position")
    total = float(sum((ex.decision or {}).values()))
    probs = [float(ex.decision[l]) / total for l in letters]

    return Encoded(input_ids=list(prompt_ids), labels=[], weight=ex.weight,
                   n_trained=1, n_masked=0, line=ex.line,
                   objective="decision", letter_token_ids=token_ids,
                   target_probs=probs), None


def loss_scale(weight: float, accumulation_steps: int) -> float:
    """What to multiply an example's mean loss by before ``backward()``.

    The divisor is the FIXED accumulation count, never the batch's own weight
    sum. Normalising by the batch would re-scale the weights away: a batch of
    examples all weighted 0.1667 would train exactly like a batch all weighted
    1.0, since each would be divided by its own total. The weights are only
    meaningful relative to each other across the whole run.
    """
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    return float(weight) / float(accumulation_steps)


@dataclasses.dataclass
class Stats:
    """What a pass over the dataset actually prepared."""

    examples: int = 0
    decision_examples: int = 0   # of `examples`, those encoded as decisions
    skipped: List[Skipped] = dataclasses.field(default_factory=list)
    trained_tokens: int = 0
    masked_tokens: int = 0
    weight_sum: float = 0.0
    #: SEQUENCE length, which is neither of the two above. `trained_tokens`
    #: counts answer positions in the loss and `masked_tokens` counts answer
    #: positions excluded from it -- the PROMPT is in neither, so a summary
    #: made of those two says nothing about how long the sequences are. That
    #: is the number which decides whether a step fits in VRAM, and leaving
    #: it out is how a set whose examples are ~5,000 tokens long was read as
    #: "397 tokens per example" and an OOM theory dismissed on the strength
    #: of it. One forward of 5,000 tokens over a 152k vocab is several GB of
    #: logits alone.
    seq_tokens: int = 0
    seq_tokens_max: int = 0

    def describe(self) -> str:
        parts = [f"{self.examples} examples",
                 f"{self.trained_tokens} trained tokens",
                 f"{self.masked_tokens} masked away"]
        if self.examples and self.seq_tokens:
            parts.append(f"{self.seq_tokens // max(1, self.examples)} tokens "
                         f"per sequence on average, longest "
                         f"{self.seq_tokens_max}")
        if self.decision_examples:
            parts.append(f"{self.decision_examples} decision example(s)")
        if self.skipped:
            parts.append(f"{len(self.skipped)} SKIPPED")
        return ", ".join(parts)


def prepare(tokenizer, examples: Iterable[Example], *, max_len: int = 8192,
            enable_thinking: bool = False,
            objective: str = "sft") -> Tuple[List[Encoded], Stats]:
    """Encode a whole dataset, collecting what was skipped rather than dropping it.

    ``objective`` selects how every example is encoded: ``"sft"`` supervises the
    answer text (``encode``), ``"decision"`` scores the answer-letter logits
    (``encode_decision``). A ``decision`` record carried under an ``"sft"`` job,
    or an ``"sft"`` record (no ``decision`` key) under a ``"decision"`` job, is a
    configuration error and fails loudly rather than being skipped.
    """
    if objective not in ("sft", "decision"):
        raise ValueError(f"unknown objective {objective!r}; expected 'sft' or 'decision'")
    out: List[Encoded] = []
    st = Stats()
    for ex in examples:
        if objective == "decision":
            if ex.decision is None:
                raise DatasetError(
                    f"line {ex.line}: a decision job needs a 'decision' target on "
                    f"every example, and this one has none")
            enc, skip = encode_decision(tokenizer, ex, max_len=max_len,
                                        enable_thinking=enable_thinking)
        else:
            if ex.decision is not None:
                raise DatasetError(
                    f"line {ex.line}: this example has a 'decision' target but the "
                    f"job objective is 'sft'; set objective='decision' to train on it")
            enc, skip = encode(tokenizer, ex, max_len=max_len,
                               enable_thinking=enable_thinking)
        if skip is not None:
            st.skipped.append(skip)
            continue
        out.append(enc)
        st.examples += 1
        if enc.objective == "decision":
            st.decision_examples += 1
        st.trained_tokens += enc.n_trained
        st.masked_tokens += enc.n_masked
        st.seq_tokens += len(enc.input_ids)
        st.seq_tokens_max = max(st.seq_tokens_max, len(enc.input_ids))
        st.weight_sum += enc.weight
    return out, st


# --------------------------------------------------------------------------- #
# the dataset store
# --------------------------------------------------------------------------- #
#
# Content-addressed, like the adapter blobs, and for the same reason: a client
# that already sent a 119 MB JSONL must not send it again, and it can compute
# the id itself (plain sha256 of the file) to ask before uploading.
#
# Separate from the adapter store on purpose. That one names its files
# ``<hash>.safetensors`` and unpacks tars, because what it holds is weights; a
# training set stored there would be a JSONL called .safetensors, which is the
# kind of small lie that costs someone an afternoon.
#
# Streamed to disk while hashing, never read whole: these files run to hundreds
# of megabytes and the point of a store is that the server does not need them in
# memory to accept them.

def dataset_dir() -> str:
    import os as _os
    from codai.api.loras import _loras_dir
    d = _os.path.join(_loras_dir(), "_datasets")
    _os.makedirs(d, exist_ok=True)
    return d


def dataset_hex(ident: str) -> str:
    """The bare lowercase hex of a dataset id (accepts an optional 'sha256:')."""
    ident = (ident or "").strip()
    if ident.lower().startswith("sha256:"):
        ident = ident[7:]
    ident = ident.lower()
    if not re_fullmatch_hex(ident):
        raise DatasetError(f"not a sha256 id: {ident!r}")
    return ident


def re_fullmatch_hex(s: str) -> bool:
    import re
    return bool(re.fullmatch(r"[0-9a-f]{64}", s or ""))


def dataset_path(ident: str) -> str:
    """Where a dataset with this id lives, whether or not it is there yet."""
    import os as _os
    return _os.path.join(dataset_dir(), f"{dataset_hex(ident)}.jsonl")


def dataset_exists(ident: str) -> bool:
    import os as _os
    try:
        return _os.path.isfile(dataset_path(ident))
    except DatasetError:
        return False


def save_dataset_stream(chunks: Iterable[bytes]) -> Tuple[str, bool, int]:
    """Stream a dataset into the store. Returns ``(hex, existed, bytes)``.

    Written to a temp file and renamed only once the whole body has arrived, so
    a connection that drops at 80 MB leaves nothing a later run can mistake for
    a complete dataset. The id is the sha256 of the bytes as sent, which is what
    a client can compute locally to skip the upload entirely.
    """
    import hashlib
    import os as _os
    import tempfile
    h = hashlib.sha256()
    total = 0
    fd, tmp = tempfile.mkstemp(dir=dataset_dir(), suffix=".part")
    try:
        with _os.fdopen(fd, "wb") as fh:
            for chunk in chunks:
                if not chunk:
                    continue
                h.update(chunk)
                fh.write(chunk)
                total += len(chunk)
        hexd = h.hexdigest()
        final = _os.path.join(dataset_dir(), f"{hexd}.jsonl")
        if _os.path.isfile(final):
            _os.unlink(tmp)
            return hexd, True, total
        _os.replace(tmp, final)
        return hexd, False, total
    except BaseException:
        try:
            _os.unlink(tmp)
        except OSError:
            pass
        raise


def resolve_dataset(dataset: str = "", dataset_id: str = "") -> str:
    """A local path for either form, or raise with what is wrong.

    Two ways in because they answer different situations: a path for an install
    that already has the file on disk, an id for a client on another machine
    that uploaded it.
    """
    import os as _os
    if dataset_id:
        path = dataset_path(dataset_id)
        if not _os.path.isfile(path):
            raise DatasetError(
                f"dataset {dataset_id} is not in the store — upload it to "
                f"POST /v1/loras/datasets first")
        return path
    ds = _os.path.expanduser((dataset or "").strip())
    if not ds:
        raise DatasetError("one of 'dataset' (a path) or 'dataset_id' is required")
    if not _os.path.isfile(ds):
        raise DatasetError(f"dataset file not found: {ds}")
    return ds
