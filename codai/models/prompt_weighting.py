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

"""The prompt syntax every Stable-Diffusion client already speaks.

A CLIP text encoder takes 77 tokens, two of which are the start and end
markers. Handing a longer prompt to a diffusers pipeline does not fail: the
tokenizer truncates it and the model renders what fits, silently. That is
fatal for the booru-tag prompting style the SDXL fine-tune families are
trained on (Pony, Illustrious, NoobAI and their derivatives), where a prompt
opens with a run of quality tags -- ``score_9, score_8_up, score_7_up, ...``
-- and spends twenty-odd tokens before reaching the subject. Truncated at 77,
the subject is what gets dropped.

Those models also expect two things the plain text path cannot express:

* **weights** -- ``(tag:1.3)`` to emphasise, ``[tag]`` to attenuate, bare
  ``(tag)`` for 1.1 and nesting that multiplies. Without a parser the brackets
  and the colon reach CLIP as literal tokens, i.e. as noise in the prompt.
* **BREAK** -- start the next 75-token window here, so tags meant to describe
  different things are not blended by sharing one window.

This module implements both, and the ``clip_skip`` those fine-tunes are
trained with, by encoding the prompt ourselves: split it into weighted
fragments, tokenize to as many 75-token windows as it needs, encode each
window, scale per token, and concatenate. The result goes to the pipeline as
``prompt_embeds`` instead of ``prompt``, so no caller loses tokens and no
pipeline has to support any of it.

The grammar is A1111's, which ComfyUI, InvokeAI and every prompt a user is
likely to paste also use. One consequence worth stating plainly: a literal
parenthesis in prose now carries a weight of 1.1. ``\\(`` escapes it, and
``prompt_weighting: false`` on a request (or ``prompt_weighting`` in the
model's config) turns the parser off while keeping the long-prompt windowing.

Only the CLIP-encoder families are handled -- SD 1.x/2.x (one encoder) and
SDXL (two, with a pooled vector from the second). Flux, SD3 and Z-Image carry
a T5 with its own much longer limit and their own conventions;
:func:`supports_weighted_embeds` returns False for those and the caller keeps
the pipeline's native path.
"""

import re
from typing import List, Optional, Tuple

#: Tokens per encoded window, excluding the start and end markers that bring
#: each window to CLIP's 77.
WINDOW = 75

#: Parsed fragments use this sentinel, with weight -1, for an explicit BREAK.
BREAK = "BREAK"

_re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

_re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

_ROUND = 1.1
_SQUARE = 1 / 1.1


def parse_prompt_attention(text: str) -> List[list]:
    """``text`` as ``[[fragment, weight], ...]``, plus ``[BREAK, -1]`` markers.

    ``(a)`` weighs 1.1, ``[a]`` weighs 1/1.1, ``(a:1.3)`` weighs exactly 1.3,
    and nesting multiplies, so ``((a))`` is 1.21. A backslash escapes a
    bracket. Unclosed brackets still apply to everything after them, which is
    what a user who forgot the closing one meant.
    """
    res: List[list] = []
    round_pos: List[int] = []
    square_pos: List[int] = []

    def multiply_from(start: int, factor: float) -> None:
        for p in range(start, len(res)):
            res[p][1] *= factor

    for m in _re_attention.finditer(text or ""):
        token = m.group(0)
        weight = m.group(1)
        if token.startswith("\\"):
            res.append([token[1:], 1.0])
        elif token == "(":
            round_pos.append(len(res))
        elif token == "[":
            square_pos.append(len(res))
        elif weight is not None and round_pos:
            multiply_from(round_pos.pop(), float(weight))
        elif token == ")" and round_pos:
            multiply_from(round_pos.pop(), _ROUND)
        elif token == "]" and square_pos:
            multiply_from(square_pos.pop(), _SQUARE)
        else:
            parts = _re_break.split(token)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append([BREAK, -1])
                res.append([part, 1.0])

    for pos in round_pos:
        multiply_from(pos, _ROUND)
    for pos in square_pos:
        multiply_from(pos, _SQUARE)

    if not res:
        return [["", 1.0]]

    # Merge neighbours that ended up with the same weight: fewer, longer
    # fragments tokenize closer to how the model saw the text in training
    # (CLIP's BPE merges across what would otherwise be a fragment boundary).
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1] and BREAK not in (res[i][0], res[i + 1][0]):
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res


def has_weighting_syntax(text: str) -> bool:
    """True when ``text`` uses anything the parser would change.

    Only used for reporting: the encoder runs the same either way.
    """
    if not text:
        return False
    if _re_break.search(text):
        return True
    return any(w != 1.0 for _, w in parse_prompt_attention(text))


# --------------------------------------------------------------------------- #
#  tokenizing to windows
# --------------------------------------------------------------------------- #

def _pad_id(tokenizer) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return 0 if pad is None else int(pad)


def tokenize_weighted(tokenizer, text: str,
                      plain: bool = False) -> Tuple[List[List[int]], List[List[float]]]:
    """``text`` as equal-length windows of token ids and per-token weights.

    Each window is ``WINDOW`` ids long: the last one is padded, and a BREAK
    closes the current window early. The start/end markers are added by
    :func:`_wrap`, not here, so a window can be re-wrapped per encoder.

    ``plain`` skips the weight grammar entirely (every token weighs 1.0) for a
    caller that wants the long-prompt windowing without it.
    """
    fragments = [[text or "", 1.0]] if plain else parse_prompt_attention(text)
    pad = _pad_id(tokenizer)

    windows: List[List[int]] = []
    weights: List[List[float]] = []
    ids: List[int] = []
    ws: List[float] = []

    def close(force: bool) -> None:
        """Push the current window, padded out, if there is one (or ``force``)."""
        nonlocal ids, ws
        if not ids and not force:
            return
        ids = ids + [pad] * (WINDOW - len(ids))
        ws = ws + [1.0] * (WINDOW - len(ws))
        windows.append(ids)
        weights.append(ws)
        ids, ws = [], []

    for fragment, weight in fragments:
        if fragment == BREAK and weight < 0:
            close(force=True)
            continue
        if not fragment:
            continue
        # add_special_tokens=False: the markers belong to the window, not to
        # each fragment. truncation=False because the whole point is to keep
        # what a single window cannot hold.
        chunk = tokenizer(fragment, add_special_tokens=False,
                          truncation=False).input_ids
        for tid in chunk:
            if len(ids) == WINDOW:
                close(force=True)
            ids.append(int(tid))
            ws.append(float(weight))

    close(force=not windows)
    return windows, weights


def _wrap(tokenizer, window: List[int]) -> List[int]:
    """One window with CLIP's start and end markers, i.e. 77 ids."""
    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = _pad_id(tokenizer)
    bos = pad if bos is None else int(bos)
    eos = pad if eos is None else int(eos)
    return [bos] + list(window) + [eos]


# --------------------------------------------------------------------------- #
#  encoding
# --------------------------------------------------------------------------- #

def supports_weighted_embeds(pipeline) -> bool:
    """True when this pipeline's text encoders are the CLIP pair we handle.

    A second encoder that is not CLIP means a T5 (Flux, SD3) whose limit and
    conventions are different, so those keep their native path.
    """
    tok = getattr(pipeline, "tokenizer", None)
    enc = getattr(pipeline, "text_encoder", None)
    if tok is None or enc is None:
        return False
    if "CLIP" not in type(enc).__name__:
        return False
    enc2 = getattr(pipeline, "text_encoder_2", None)
    if enc2 is not None and "CLIP" not in type(enc2).__name__:
        return False
    if enc2 is not None and getattr(pipeline, "tokenizer_2", None) is None:
        return False
    return True


def _is_sdxl(pipeline) -> bool:
    return getattr(pipeline, "text_encoder_2", None) is not None


def _expand_learned_tokens(pipeline, text: str, tokenizer) -> str:
    """``text`` with multi-vector textual inversions written out in full.

    A textual inversion trained on several vectors is registered as ``tok``,
    ``tok_1``, ``tok_2``..., and a prompt only ever says ``tok``. Diffusers
    expands it in ``maybe_convert_prompt`` on its way into the tokenizer, so
    anything encoding a prompt itself has to do the same -- otherwise an
    embedding like ``easynegative`` (eight vectors) contributes exactly one of
    them and the prompt looks like it did nothing.

    It replaces only registered trigger words, so the weight grammar passes
    through untouched and this is safe to run before parsing.
    """
    if not text:
        return text
    convert = getattr(pipeline, "maybe_convert_prompt", None)
    if convert is None:
        return text
    try:
        return convert(text, tokenizer)
    except Exception:
        return text


def _hidden(out, text_encoder, clip_skip: Optional[int], sdxl: bool):
    """The hidden state a diffusers pipeline would have used.

    Mirrors diffusers exactly, because anything else shifts every output:
    SDXL always takes the penultimate layer, so its ``clip_skip`` counts from
    there, while SD 1.x takes the final (already normalised) output and a
    ``clip_skip`` there needs the final layer norm applied by hand.
    """
    if sdxl:
        layer = -2 if not clip_skip else -(int(clip_skip) + 2)
        return out.hidden_states[layer]
    if not clip_skip:
        return out[0]
    emb = out.hidden_states[-(int(clip_skip) + 1)]
    try:
        return text_encoder.text_model.final_layer_norm(emb)
    except AttributeError:
        return emb


def _apply_weights(emb, weights):
    """Scale each token's embedding, keeping the window's overall magnitude.

    Multiplying alone changes the mean of the window, which shows up as a
    brightness/contrast shift that has nothing to do with the emphasis asked
    for. Restoring the original mean afterwards is what A1111 does and what
    the weights were tuned against.
    """
    import torch
    if all(w == 1.0 for w in weights):
        return emb
    w = torch.tensor(weights, dtype=emb.dtype, device=emb.device)
    w = w.unsqueeze(0).unsqueeze(-1)            # (1, tokens, 1)
    before = emb.mean()
    emb = emb * w
    after = emb.mean()
    if float(after) != 0.0:
        emb = emb * (before / after)
    return emb


def encode_weighted_prompt(pipeline, prompt: str, negative_prompt: str = "",
                           clip_skip: Optional[int] = None,
                           plain: bool = False,
                           device=None, dtype=None) -> Optional[dict]:
    """Pipeline kwargs for ``prompt`` with no 77-token limit and real weights.

    Returns ``{'prompt_embeds', 'negative_prompt_embeds'}``, plus the two
    pooled vectors on SDXL -- ready to pass straight to the pipeline instead
    of ``prompt``/``negative_prompt``. Returns None when the pipeline is not
    one of the CLIP families, so the caller can fall back.

    The positive and negative prompts are padded to the same number of
    windows: classifier-free guidance subtracts one from the other, so they
    have to be the same length whatever the user typed.
    """
    import torch

    if not supports_weighted_embeds(pipeline):
        return None

    sdxl = _is_sdxl(pipeline)
    pairs = [(pipeline.tokenizer, pipeline.text_encoder)]
    if sdxl:
        pairs.append((pipeline.tokenizer_2, pipeline.text_encoder_2))

    if device is None:
        device = getattr(pipeline, "device", None)

    # One window count for both sides, per tokenizer. The two SDXL tokenizers
    # are the same BPE with different padding, so they agree in practice, but
    # taking the max keeps a mismatch from truncating anything.
    tokenized = []
    n_windows = 1
    for tokenizer, _ in pairs:
        pos = tokenize_weighted(
            tokenizer, _expand_learned_tokens(pipeline, prompt, tokenizer),
            plain=plain)
        neg = tokenize_weighted(
            tokenizer,
            _expand_learned_tokens(pipeline, negative_prompt, tokenizer),
            plain=plain)
        tokenized.append((tokenizer, pos, neg))
        n_windows = max(n_windows, len(pos[0]), len(neg[0]))

    def _windows_to_embeds(tokenizer, text_encoder, windows, weights):
        """Encode every window and lay them end to end -> (embeds, pooled)."""
        pad = _pad_id(tokenizer)
        embeds = []
        pooled = None
        for i in range(n_windows):
            window = windows[i] if i < len(windows) else [pad] * WINDOW
            w = weights[i] if i < len(weights) else [1.0] * WINDOW
            ids = torch.tensor([_wrap(tokenizer, window)], dtype=torch.long,
                               device=device)
            out = text_encoder(ids, output_hidden_states=True)
            emb = _hidden(out, text_encoder, clip_skip, sdxl)
            # The pooled vector summarises a whole 77-token pass, so there is
            # no meaningful way to pool across windows: SDXL gets the first
            # one, which holds the start of the prompt.
            if pooled is None and sdxl and hasattr(out, "text_embeds"):
                pooled = out.text_embeds
            embeds.append(_apply_weights(emb, [1.0] + list(w) + [1.0]))
        return torch.cat(embeds, dim=1), pooled

    pos_parts, neg_parts = [], []
    pooled_pos = pooled_neg = None
    with torch.no_grad():
        for (tokenizer, pos, neg), (_, text_encoder) in zip(tokenized, pairs):
            emb, pooled = _windows_to_embeds(tokenizer, text_encoder, *pos)
            pos_parts.append(emb)
            if pooled is not None:
                pooled_pos = pooled
            emb, pooled = _windows_to_embeds(tokenizer, text_encoder, *neg)
            neg_parts.append(emb)
            if pooled is not None:
                pooled_neg = pooled

    # SDXL conditions on both encoders' states concatenated on the feature
    # axis (768 + 1280); SD 1.x has only the one.
    prompt_embeds = torch.cat(pos_parts, dim=-1) if len(pos_parts) > 1 else pos_parts[0]
    negative_embeds = torch.cat(neg_parts, dim=-1) if len(neg_parts) > 1 else neg_parts[0]

    if dtype is not None:
        prompt_embeds = prompt_embeds.to(dtype)
        negative_embeds = negative_embeds.to(dtype)

    out = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_embeds,
    }
    if sdxl and pooled_pos is not None and pooled_neg is not None:
        if dtype is not None:
            pooled_pos = pooled_pos.to(dtype)
            pooled_neg = pooled_neg.to(dtype)
        out["pooled_prompt_embeds"] = pooled_pos
        out["negative_pooled_prompt_embeds"] = pooled_neg
    return out


def window_count(tokenizer, prompt: str, plain: bool = False) -> int:
    """How many 75-token windows ``prompt`` needs. For logging and tests."""
    windows, _ = tokenize_weighted(tokenizer, prompt, plain=plain)
    return len(windows)
