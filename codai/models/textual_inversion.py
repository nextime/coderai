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

"""Learned prompt tokens: ``easynegative`` and the rest of the embeddings.

A textual inversion is a handful of vectors trained to sit in CLIP's embedding
table under a word of their own. Writing that word in a prompt is how half the
Stable-Diffusion ecosystem's negative prompts are built -- ``easynegative``,
``bad-hands-5``, ``FastNegativeV2`` -- and without the embedding loaded the
word is just an unknown token, so the prompt silently asks for nothing.

Three ways to get one loaded, in the order a request resolves them:

* ``embeddings: ["easynegative"]`` on the request, for an ad-hoc one.
* ``textual_inversion`` on the model's config, for the ones a model always
  wants.
* every file in ``image.embeddings_dir``, discovered by filename -- which is
  how A1111 works, and what a prompt copied from a model card assumes.

A reference is a bare name (matched against the directory), a path to a
``.safetensors`` / ``.pt`` / ``.bin`` file, or a HuggingFace repo id. Appending
``:token`` renames the trigger word: ``my-emb.safetensors:myneg``.

Two details decide whether this actually works:

* **SDXL keeps two text encoders**, so an SDXL-format embedding holds a
  ``clip_l`` and a ``clip_g`` tensor that have to go into the matching
  tokenizer/encoder pair. Loading only the first gives the token an embedding
  in one encoder and noise in the other.
* **A multi-vector embedding occupies several tokens**, registered as
  ``tok``, ``tok_1``, ``tok_2``... Diffusers expands the prompt through
  ``maybe_convert_prompt`` before tokenizing; anything encoding a prompt
  itself has to do the same or only the first vector is ever used. That is
  handled in :mod:`codai.models.prompt_weighting`.
"""

import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

_log = logging.getLogger(__name__)

#: What an embedding file looks like on disk.
SUFFIXES = (".safetensors", ".pt", ".bin", ".ckpt")

#: Marks the tokens already loaded into a pipeline, so a second request does
#: not try to register the same trigger word again (diffusers raises on that).
_LOADED_ATTR = "_coderai_textual_inversions"


def _split_ref(ref: str) -> Tuple[str, Optional[str]]:
    """``"file.safetensors:token"`` -> (``"file.safetensors"``, ``"token"``).

    A Windows drive letter or an ``hf://`` style prefix would be mangled by a
    naive split, so only a colon in the last path segment counts.
    """
    ref = str(ref).strip()
    head, sep, tail = ref.rpartition(":")
    if sep and tail and "/" not in tail and os.sep not in tail and len(head) > 1:
        return head, tail
    return ref, None


def _token_for(ref: str, explicit: Optional[str]) -> str:
    """The trigger word: the one asked for, else the filename, as A1111 does."""
    if explicit:
        return explicit
    base = os.path.basename(ref.rstrip("/"))
    for suffix in SUFFIXES:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def discover(directory: Optional[str]) -> Dict[str, str]:
    """``{token: path}`` for every embedding file in ``directory``.

    Not recursive and not sorted by anything meaningful: a flat directory of
    trigger words is the convention these files arrive in.
    """
    out: Dict[str, str] = {}
    if not directory:
        return out
    directory = os.path.expanduser(directory)
    try:
        names = os.listdir(directory)
    except OSError as exc:
        _log.warning("embeddings_dir %s cannot be read: %s", directory, exc)
        return out
    for name in sorted(names):
        if not name.endswith(SUFFIXES):
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            out[_token_for(name, None)] = path
    return out


def resolve(ref: str, directory: Optional[str] = None) -> Tuple[str, str]:
    """``(source, token)`` for one reference, or raise ``FileNotFoundError``.

    ``source`` is a path that exists, or a HuggingFace repo id left for
    diffusers to fetch. A bare name is looked up in ``directory`` first,
    because that is what a prompt copied off a model card means by it.
    """
    raw, explicit = _split_ref(ref)
    token = _token_for(raw, explicit)

    if os.path.isfile(os.path.expanduser(raw)):
        return os.path.expanduser(raw), token

    if directory:
        # Looked up by the name of the FILE, not by the trigger word: with
        # "easynegative:myneg" the word is myneg and the file is still
        # easynegative.safetensors.
        name = _token_for(raw, None)
        found = discover(directory)
        if name in found:
            return found[name], token
        # a name given with its extension, or a file sitting beside the others
        candidate = os.path.join(os.path.expanduser(directory),
                                 os.path.basename(raw))
        if os.path.isfile(candidate):
            return candidate, token

    if "/" in raw and not raw.startswith((".", "/", "~")):
        return raw, token          # a repo id; diffusers resolves and caches it

    raise FileNotFoundError(
        f"textual inversion {ref!r} not found: no such file, nothing by that "
        f"name in the embeddings directory"
        + (f" ({directory})" if directory else " (image.embeddings_dir is not set)")
        + ", and it is not a HuggingFace repo id")


def loaded_tokens(pipeline) -> set:
    """Trigger words already registered on this pipeline."""
    return set(getattr(pipeline, _LOADED_ATTR, None) or ())


def _load_one(pipeline, source: str, token: str) -> None:
    """Register ``token`` on the pipeline, both encoders included on SDXL."""
    sdxl = getattr(pipeline, "text_encoder_2", None) is not None
    state = None
    if sdxl and os.path.isfile(source) and source.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            state = load_file(source)
        except Exception as exc:
            _log.debug("cannot inspect %s as safetensors (%s)", source, exc)

    if state is not None and "clip_g" in state and "clip_l" in state:
        # The SDXL layout: one tensor per text encoder, and they are not
        # interchangeable -- 768 wide for clip_l, 1280 for clip_g.
        pipeline.load_textual_inversion(
            state["clip_l"], token=token,
            text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
        pipeline.load_textual_inversion(
            state["clip_g"], token=token,
            text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2)
    else:
        pipeline.load_textual_inversion(source, token=token)


def apply(pipeline, refs: Iterable[str] = (), directory: Optional[str] = None,
          strict: bool = False) -> List[str]:
    """Load ``refs`` (plus everything in ``directory``) onto ``pipeline``.

    Returns the trigger words now usable in a prompt. Already-loaded ones are
    skipped, so this is cheap to call on every request -- which it is, because
    a pipeline outlives the request that loaded it and a later request must
    find the same tokens there.

    ``strict`` raises on a reference that cannot be resolved or loaded: that is
    for the ones a caller named explicitly, where quietly rendering without the
    embedding is the failure mode worth avoiding. Directory sweeps are never
    strict -- one unreadable file should not break every request.
    """
    if not hasattr(pipeline, "load_textual_inversion"):
        if strict and list(refs):
            raise RuntimeError("this pipeline cannot load textual inversions")
        return []

    already = loaded_tokens(pipeline)
    # (source, token, asked for by name) -- the last flag decides whether a
    # failure is this request's problem or just a note in the log.
    wanted: List[Tuple[str, str, bool]] = []

    for ref in refs or ():
        if not str(ref).strip():
            continue
        source, token = resolve(str(ref), directory)   # may raise, by design
        wanted.append((source, token, True))

    named = {token for _, token, _ in wanted}
    for token, path in discover(directory).items():
        if token not in named:
            wanted.append((path, token, False))

    added = []
    for source, token, explicit in wanted:
        if token in already:
            continue
        try:
            _load_one(pipeline, source, token)
        except Exception as exc:
            if strict and explicit:
                raise RuntimeError(
                    f"textual inversion {token!r} could not be loaded: {exc}") from exc
            _log.warning("textual inversion %r (%s) not loaded: %s",
                         token, source, exc)
            continue
        already.add(token)
        added.append(token)

    setattr(pipeline, _LOADED_ATTR, already)
    if added:
        _log.info("Textual inversions loaded: %s", ", ".join(sorted(added)))
    return sorted(already)
