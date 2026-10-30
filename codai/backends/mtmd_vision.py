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

"""How many tokens a picture is worth to a GGUF vision model.

A GGUF text model plus an ``mmproj`` projector answers image questions through
llama.cpp's mtmd handler. The handler decides how many tokens an image becomes
from the image's own size, and for some projectors that number is far too small
to carry a picture -- llama.cpp says so itself, on every load:

    load_hparams: projector: qwen3vl_merger
    load_hparams: Qwen-VL models require at minimum 1024 image tokens to
                  function correctly on grounding tasks
    load_hparams: if you encounter problems with accuracy, try adding
                  --image-min-tokens 1024

Nothing set that floor, and ``mtmd_context_params.image_min_tokens`` stayed at
its default. What that looked like from outside, measured on Qwen3.8-27B with
its own matching projector loaded and reporting ``has vision encoder``:

* a 448x448 picture added **16** tokens to the prompt (61 -> 77), where the
  same image through the HF path added 156;
* two completely different pictures -- a yellow field with a blue triangle and
  a green square, and three white circles on black -- produced byte-identical
  descriptions at temperature 0, neither resembling either image ("a series of
  exclamation marks");
* the same prompt with NO image answered differently, so the pictures were
  reaching the model; only their content was not.

Sixteen tokens cannot describe a photograph, so the model answered from the
prompt and sounded perfectly confident doing it. That is the worst failure
shape there is: a 200 with a fluent wrong answer, which no caller can tell from
a right one.

So the floor is set here: from the card when it says, and otherwise from what
the projector itself needs.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

#: Projector families that need a minimum, and what llama.cpp says it is.
#: Keyed by substring of ``clip.projector_type`` as the GGUF records it.
#: Only families with a documented floor belong here -- a projector that is
#: happy with what the image gives it must not be forced to upscale, which
#: costs prefill time for nothing.
PROJECTOR_MIN_TOKENS = {
    "qwen3vl": 1024,     # qwen3vl_merger — llama.cpp ggml-org/llama.cpp#16842
    "qwen2vl": 1024,     # same merger lineage, same grounding requirement
    "qwen25vl": 1024,
}

#: Guards the brief window in which the upstream default-params factory is
#: wrapped (see :class:`ImageTokenFloorHandler`). Loads are serialised in
#: practice; this makes it true rather than likely.
_PARAMS_LOCK = threading.Lock()


def projector_type(mmproj_path: str) -> str:
    """``clip.projector_type`` from an mmproj GGUF, lowercased, or "".

    Read from the file rather than guessed from its name: an mmproj is often
    renamed to sit beside the model it serves, and the name of the base model
    says nothing about which merger was exported.
    """
    try:
        from gguf import GGUFReader
        path = os.path.expanduser(str(mmproj_path or ""))
        if not os.path.isfile(path):
            return ""
        reader = GGUFReader(path)
        field = reader.fields.get("clip.projector_type")
        if field is None:
            return ""
        val = field.parts[field.data[0]] if field.data else None
        if val is None:
            return ""
        try:
            return bytes(val).decode("utf-8", "replace").strip().lower()
        except (TypeError, ValueError):
            return str(val).strip().lower()
    except Exception:
        # No gguf reader, an unreadable file, or a metadata layout this does
        # not know: fall back to no floor, which is how it behaved before.
        return ""


def min_image_tokens(mmproj_path: str, configured=None) -> tuple:
    """``(min_tokens, why)`` for this projector.

    A card's own ``image_min_tokens`` always wins, including an explicit 0 for
    an operator who wants the old behaviour back.
    """
    if configured is not None:
        try:
            want = int(configured)
        except (TypeError, ValueError):
            want = 0
        return max(0, want), "set on the model's card"
    proj = projector_type(mmproj_path)
    if not proj:
        return 0, ""
    key = proj.replace("-", "").replace("_", "").replace(".", "")
    for name, floor in PROJECTOR_MIN_TOKENS.items():
        if name in key:
            return floor, (f"{proj} needs at least {floor} image tokens to "
                           f"describe a picture")
    return 0, ""


def handler(mmproj_path: str, *, use_gpu: bool = True, verbose: bool = False,
            image_min_tokens: int = 0, image_max_tokens: int = 0):
    """An mtmd chat handler that honours an image-token floor.

    Returns the plain upstream handler when no floor applies, so the ordinary
    case runs upstream code with nothing of ours in the way.
    """
    from llama_cpp.llama_chat_format import MTMDChatHandler
    if not (image_min_tokens or image_max_tokens):
        return MTMDChatHandler(clip_model_path=mmproj_path, verbose=verbose,
                               use_gpu=use_gpu)
    return ImageTokenFloorHandler(
        clip_model_path=mmproj_path, verbose=verbose, use_gpu=use_gpu,
        image_min_tokens=image_min_tokens, image_max_tokens=image_max_tokens)


class _LazyBase:
    """Deferred base so importing this module never imports llama_cpp."""


def _make_handler_class():
    from llama_cpp.llama_chat_format import MTMDChatHandler

    class ImageTokenFloorHandler(MTMDChatHandler):
        """:class:`MTMDChatHandler` with ``image_min/max_tokens`` applied.

        Upstream builds its context from
        ``mtmd_cpp.mtmd_context_params_default()`` and sets four fields on the
        result; the token bounds are not among them and there is no hook in
        between. Rather than copy that method -- which would silently stop
        tracking any field upstream adds -- this wraps the default-params
        factory for the duration of the one call, so the struct upstream
        receives already carries the bounds and everything else upstream does
        to it still happens.
        """

        def __init__(self, *args, image_min_tokens: int = 0,
                     image_max_tokens: int = 0, **kwargs):
            self._image_min_tokens = max(0, int(image_min_tokens or 0))
            self._image_max_tokens = max(0, int(image_max_tokens or 0))
            super().__init__(*args, **kwargs)

        def _init_mtmd_context(self, llama_model):
            if getattr(self, "mtmd_ctx", None) is not None:
                return
            mtmd = self._mtmd_cpp
            original = mtmd.mtmd_context_params_default

            def _with_bounds():
                params = original()
                if self._image_min_tokens:
                    params.image_min_tokens = self._image_min_tokens
                if self._image_max_tokens:
                    params.image_max_tokens = self._image_max_tokens
                return params

            with _PARAMS_LOCK:
                mtmd.mtmd_context_params_default = _with_bounds
                try:
                    return super()._init_mtmd_context(llama_model)
                finally:
                    mtmd.mtmd_context_params_default = original

    return ImageTokenFloorHandler


def __getattr__(name):
    # ImageTokenFloorHandler is built on first use so that `import
    # mtmd_vision` costs nothing on a server with no vision model.
    if name == "ImageTokenFloorHandler":
        cls = _make_handler_class()
        globals()[name] = cls
        return cls
    raise AttributeError(name)
