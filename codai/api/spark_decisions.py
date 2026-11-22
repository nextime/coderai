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

"""Score a decision model's answer-letter distribution from one forward pass.

A decision model (rizzo-flow, or a Spark-X2.5 LoRA trained with the ``decision``
objective) answers by the probability it puts on a handful of answer-letter
tokens at the first generated position -- it is read, not chatted with. This
module turns the model's first-token top-logprobs into that distribution, so the
caller gets ``{"A": 0.74, "B": 0.26}`` and an argmax rather than a one-character
string with no confidence attached.

The parsing is a pure function (:func:`distribution_from_logprobs`) so it can be
tested without a model; the endpoint in ``codai/api/text.py`` wraps it around the
ordinary chat path with ``max_tokens=1, logprobs=True``.

The single-token contract is the same one the trainer enforces
(``text_lora_dataset._letter_token_id``): each candidate letter must be one
token, so there is one logprob to read per letter on this side and one logit to
train per letter on the other. A letter that never appears among the returned
top-logprobs is reported with probability 0 -- it is a real answer ("the model
gave this option no mass"), not an error.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _first_token_candidates(logprobs: Any) -> List[Dict[str, Any]]:
    """The ``{token, logprob}`` candidates at the first generated position.

    Accepts the OpenAI chat-logprobs shape ``{"content": [{"token", "logprob",
    "top_logprobs": [...]}, ...]}`` and returns the first content token's
    ``top_logprobs`` (falling back to the token itself when a backend returned no
    alternatives). Returns ``[]`` when there is nothing to read, which the caller
    turns into a clear error rather than a silent empty distribution.
    """
    if not isinstance(logprobs, dict):
        return []
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return []
    first = content[0]
    if not isinstance(first, dict):
        return []
    top = first.get("top_logprobs")
    if isinstance(top, list) and top:
        return [t for t in top if isinstance(t, dict) and "token" in t]
    # No alternatives: the chosen token is the only candidate we have.
    if "token" in first and "logprob" in first:
        return [{"token": first["token"], "logprob": first["logprob"]}]
    return []


def _match_logprob(candidates: List[Dict[str, Any]], letter: str) -> Optional[float]:
    """The logprob a letter was given, matched exactly then by trimmed string.

    A model usually emits the bare letter, but a chat template can prepend a
    space (`" A"`); matching the trimmed token as a fallback recovers that
    without matching a different letter. Returns None when the letter is not
    among the candidates.
    """
    # Exact token first -- unambiguous.
    for c in candidates:
        if c.get("token") == letter:
            return float(c.get("logprob"))
    # Then trimmed, which recovers a leading-space tokenization.
    want = letter.strip()
    for c in candidates:
        tok = c.get("token")
        if isinstance(tok, str) and tok.strip() == want and want != "":
            return float(c.get("logprob"))
    return None


def distribution_from_logprobs(logprobs: Any,
                               letters: List[str]) -> Dict[str, float]:
    """Normalised probabilities over ``letters`` from first-token logprobs.

    Each letter's probability is ``exp(logprob)`` for the matching candidate,
    renormalised across the letters that were found so they sum to 1 -- this is
    the model's choice AMONG the offered options, which is what a decision
    consumer wants (the absolute mass not on any letter is the model answering
    off-menu, a separate question). A letter with no matching candidate gets 0.

    Raises ``ValueError`` only when NONE of the letters were found: that means
    the model put all its first-token mass elsewhere (or ``top_logprobs`` was too
    small to include any of them), and returning an all-zero distribution would
    invite a divide-by-zero or a false argmax downstream.
    """
    if not letters:
        raise ValueError("no candidate letters were given to score")
    cands = _first_token_candidates(logprobs)
    if not cands:
        raise ValueError(
            "the response carried no first-token logprobs to score; the model "
            "or engine did not return logprobs (ask for them, and check the "
            "engine supports top_logprobs)")
    raw: Dict[str, float] = {}
    for letter in letters:
        lp = _match_logprob(cands, letter)
        raw[letter] = math.exp(lp) if lp is not None else 0.0
    total = sum(raw.values())
    if total <= 0:
        raise ValueError(
            f"none of the candidate letters {letters} appeared among the "
            f"model's top {len(cands)} first-token tokens -- raise top_logprobs, "
            f"or check the letters match what the model emits")
    return {k: v / total for k, v in raw.items()}


def summarize(distribution: Dict[str, float]) -> Tuple[Optional[str], float]:
    """``(argmax_letter, confidence)`` for a scored distribution.

    Confidence is the winning letter's own probability -- the calibrated number
    a gate uses to decide whether to trust the small model or fall back to the
    LLM. Empty in, ``(None, 0.0)`` out.
    """
    if not distribution:
        return None, 0.0
    argmax = max(distribution, key=distribution.get)
    return argmax, float(distribution[argmax])
