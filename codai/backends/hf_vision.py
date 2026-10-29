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

"""Vision for a transformers-served chat model.

A vision-language checkpoint holds two things: a text decoder and a vision
tower that turns pixels into the embeddings the decoder reads. ``AutoModelFor-
CausalLM`` loads only the first of them. Measured on transformers 5.12 with
``Qwen/Qwen3.5-9B``, whose config carries a ``vision_config`` and an
``image_token_id``::

    AutoModelForCausalLM      -> Qwen3_5ForCausalLM            (text only)
    AutoModelForImageTextToText -> Qwen3_5ForConditionalGeneration

The engine asked for the first class, so the tower was never instantiated and
the weights for it were never read. The model then answered every picture with
a description of nothing -- "a blank white image with no visible content" --
because the request had reached it as the literal text ``[image_url content]``:
with no projector loaded the chat path has nowhere to put an image, so it
flattens multipart content to a placeholder. The model was not wrong; it was
answering the prompt it was given.

Two halves, both here:

* :func:`auto_model_class` picks the class from the checkpoint's own config, so
  a vision checkpoint loads its tower. It is used by every load path in the
  CUDA backend, which is why it must never raise -- an unreadable config falls
  back to the text class and the model keeps working as it did before.
* :func:`build_inputs` renders a chat with its images through the model's
  ``AutoProcessor``, which is the only component that knows how many image
  placeholder tokens this family expands one picture into.

What a caller sends is normalised by :func:`codai.api.text._normalize_vision_content`
before it gets here, so images arrive as ``{"type": "image_url", "image_url":
{"url": ...}}`` with either an http(s) link or a ``data:`` URI.
"""

import base64
import binascii
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

#: How long to wait for an image the caller referenced by URL. A chat request
#: is already waiting on this, so it is short on purpose.
_FETCH_TIMEOUT = 20

_DATA_URI = re.compile(r"^data:(?P<mime>[^;,]*)(?P<b64>;base64)?,(?P<data>.*)$",
                       re.IGNORECASE | re.DOTALL)


def _config_for(model_name: str):
    """The checkpoint's config, or None when it cannot be read.

    ``trust_remote_code`` matches what the loaders use: a config that needs its
    own code to be read at all would otherwise report "no vision" and the model
    would load without its tower again.
    """
    try:
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        print(f"[vision] {model_name}: its config could not be read "
              f"({type(exc).__name__}: {str(exc)[:160]})", flush=True)
        return None


def vision_report(model_name: str, config=None) -> dict:
    """Why this checkpoint is, or is not, loaded with a vision tower.

    Every step is reported because every one of them used to fail silently.
    Measured on zeiss, 2026-10-29: after the vision loading shipped, an image
    sent to ``Qwen3.5-9B`` was refused with "served without a vision tower" —
    and the engine logged nothing at all about why, because
    :func:`is_vision_checkpoint` answered False and said nothing. A decision
    this consequential has to leave a trace.

    Keys: ``config`` (read or not), ``vision_config``, ``image_token_id``,
    ``cls`` (the class that would load it, or None), ``reason``.
    """
    out = {"config": False, "vision_config": False, "image_token_id": False,
           "cls": None, "reason": ""}
    cfg = config if config is not None else _config_for(model_name)
    if cfg is None:
        out["reason"] = "its config could not be read"
        return out
    out["config"] = True
    out["cfg_class"] = type(cfg).__name__
    out["vision_config"] = getattr(cfg, "vision_config", None) is not None
    out["image_token_id"] = getattr(cfg, "image_token_id", None) is not None
    if not out["vision_config"] and not out["image_token_id"]:
        out["reason"] = ("its config has neither a vision_config nor an "
                         "image_token_id — it is a text checkpoint")
        return out
    # The config says there is a tower; the installed transformers must also
    # have a class that loads it. Without this check a brand-new family would
    # pick the vision class and fail the load outright, which is worse than
    # serving it as text.
    try:
        from transformers import AutoModelForImageTextToText
        # BOTH arguments. transformers' `_LazyAutoMapping.get` is not dict.get:
        # its signature is `get(self, key, default)` and `default` is
        # REQUIRED. Calling it with one argument raises
        #
        #   TypeError: _LazyAutoMapping.get() missing 1 required positional
        #   argument: 'default'
        #
        # which this function caught and reported as "no vision class" — for
        # every checkpoint, always. So every vision model loaded as text and
        # then had its pictures refused by the guard added for exactly that
        # case. Measured on zeiss, 2026-10-29, against Qwen3.5-9B.
        cls = AutoModelForImageTextToText._model_mapping.get(type(cfg), None)
    except Exception as exc:
        out["reason"] = (f"the image-text-to-text mapping could not be read "
                         f"({type(exc).__name__}: {str(exc)[:120]})")
        return out
    if cls is None:
        out["reason"] = (f"this transformers has no image-text-to-text class "
                         f"for {type(cfg).__name__} ({_tf_version()})")
        return out
    out["cls"] = cls
    return out


def _tf_version() -> str:
    try:
        import transformers
        return f"transformers {transformers.__version__}"
    except Exception:
        return "transformers version unknown"


def is_vision_checkpoint(model_name: str, config=None) -> bool:
    """Whether this checkpoint carries a vision tower.

    Decided from the config, not from the model id: ``Qwen3.5-9B`` has no
    "vl"/"vision" in its name and is a vision model, while plenty of models
    with "vision" in the name are not image-text-to-text at all. The two
    signals are a nested ``vision_config`` and an ``image_token_id`` -- a
    decoder that has somewhere to put an image.

    See :func:`vision_report` for the same decision with its reasons.
    """
    return vision_report(model_name, config)["cls"] is not None


def auto_model_class(model_name: str):
    """``(class, is_vision)`` for loading this checkpoint.

    Never raises. The text class is the fallback for every uncertainty, so a
    checkpoint this function cannot make sense of loads exactly as it did
    before vision existed here.
    """
    from transformers import AutoModelForCausalLM
    try:
        report = vision_report(model_name)
        if report["cls"] is not None:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText, True
        # Only worth a line when the checkpoint LOOKS like a vision model and
        # is not being loaded as one: that is the case an operator has to be
        # able to see, and the one that used to be silent.
        if report.get("vision_config") or report.get("image_token_id"):
            print(f"[vision] {model_name} looks like a vision checkpoint "
                  f"(vision_config={report['vision_config']}, "
                  f"image_token_id={report['image_token_id']}, "
                  f"config={report.get('cfg_class')}) but "
                  f"{report['reason']} — loading it as text", flush=True)
    except Exception as exc:
        print(f"[vision] {model_name}: could not decide on a model class "
              f"({type(exc).__name__}: {str(exc)[:160]}) — loading it as text",
              flush=True)
    return AutoModelForCausalLM, False


def load_processor(model_name: str):
    """The checkpoint's ``AutoProcessor``, or None.

    The processor -- not the tokenizer -- owns the image preprocessing and the
    placeholder expansion, so without it a vision model cannot be prompted with
    a picture even though its tower is loaded. A model whose processor cannot be
    built is therefore not vision-capable, and the backend says so rather than
    answering blind.
    """
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        print(f"[vision] no AutoProcessor for {model_name}: {exc}")
        return None
    # A tokenizer-only "processor" cannot take images. transformers happily
    # returns one for some text checkpoints, and it would pass every duck-type
    # check until the first picture arrived.
    if not hasattr(proc, "image_processor") and not hasattr(proc, "feature_extractor"):
        print(f"[vision] {model_name}: AutoProcessor has no image processor")
        return None
    return proc


def _decode_data_uri(url: str):
    m = _DATA_URI.match(url)
    if not m:
        return None
    data = m.group("data")
    if m.group("b64"):
        try:
            return base64.b64decode(data, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"image data URI is not valid base64: {exc}")
    from urllib.parse import unquote_to_bytes
    return unquote_to_bytes(data)


def load_image(url: str):
    """One ``image_url`` as a PIL image in RGB.

    Accepts a ``data:`` URI, an http(s) link and a local path, which is the
    same set the llama.cpp multimodal path accepts -- a client cannot be
    expected to know which backend is serving the model it asked for.
    """
    from PIL import Image

    raw: Optional[bytes] = None
    if url.startswith("data:"):
        raw = _decode_data_uri(url)
    elif url.startswith(("http://", "https://")):
        import requests
        resp = requests.get(url, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        raw = resp.content
    elif os.path.exists(url):
        with open(url, "rb") as fh:
            raw = fh.read()
    if raw is None:
        raise ValueError(f"unsupported image reference: {url[:64]!r}")
    img = Image.open(io.BytesIO(raw))
    # Convert while the file handle is still open: PIL is lazy, and a converted
    # copy is also what detaches the image from the BytesIO we are about to
    # drop. RGB because a palette or RGBA image reaches the tower as the wrong
    # channel count.
    return img.convert("RGB")


def has_images(messages) -> bool:
    """Whether any turn carries a picture."""
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "input_image", "image"):
                return True
    return False


def _part_url(part: Dict[str, Any]) -> Optional[str]:
    t = part.get("type")
    holder = part.get("image_url") if t == "image_url" else part.get("image")
    if holder is None:
        holder = part.get("url")
    url = holder.get("url") if isinstance(holder, dict) else holder
    return url if isinstance(url, str) and url else None


def to_processor_messages(messages) -> Tuple[List[dict], List[Any]]:
    """``(messages, images)`` in the shape a processor chat template wants.

    Images are replaced by ``{"type": "image"}`` markers and handed over
    separately, in the order they appear, because that is the contract every
    HF image-text-to-text processor shares: the template emits one placeholder
    run per marker and ``processor(text=…, images=[…])`` fills them in order.
    A tool/assistant turn keeps its text; roles are passed through untouched so
    the model's own template decides what to do with them.
    """
    out: List[dict] = []
    images: List[Any] = []
    for m in messages or []:
        if isinstance(m, dict):
            role = m.get("role") or "user"
            content = m.get("content")
        else:
            role = getattr(m, "role", "user") or "user"
            content = getattr(m, "content", None)
        if not isinstance(content, list):
            out.append({"role": role,
                        "content": [{"type": "text", "text": "" if content is None else str(content)}]})
            continue
        parts: List[dict] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append({"type": "text", "text": str(part)})
                continue
            if part.get("type") in ("image_url", "input_image", "image"):
                url = _part_url(part)
                if not url:
                    continue
                images.append(load_image(url))
                parts.append({"type": "image"})
            elif "text" in part:
                parts.append({"type": "text", "text": str(part.get("text") or "")})
        if not parts:
            parts = [{"type": "text", "text": ""}]
        out.append({"role": role, "content": parts})
    return out, images


def build_inputs(processor, messages, add_generation_prompt: bool = True):
    """Tokenized model inputs for a chat that contains pictures.

    Rendered in two steps -- template to text, then ``processor(text, images)``
    -- rather than the one-step ``apply_chat_template(tokenize=True)``, because
    the two-step form is the one every family in transformers 5 supports and it
    keeps the rendered prompt available for logging. The one-step form is the
    fallback for a processor whose template refuses to render without its
    images (some remote-code processors do).
    """
    msgs, images = to_processor_messages(messages)
    try:
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
        return processor(text=[text], images=images or None, return_tensors="pt"), images
    except Exception as exc:
        print(f"[vision] two-step prompt render failed ({exc}); using the "
              f"processor's own tokenizing template")
        inputs = processor.apply_chat_template(
            _embed_images(msgs, images), add_generation_prompt=add_generation_prompt,
            tokenize=True, return_dict=True, return_tensors="pt")
        return inputs, images


def _embed_images(msgs: List[dict], images: List[Any]) -> List[dict]:
    """``msgs`` with each image marker carrying its own picture.

    The one-step template reads the picture out of the content part itself,
    while the two-step form takes them as a separate list. Same order either
    way, so the markers are filled from the front.
    """
    pool = list(images)
    out = []
    for m in msgs:
        parts = []
        for part in m.get("content") or []:
            if part.get("type") == "image" and pool:
                parts.append({"type": "image", "image": pool.pop(0)})
            else:
                parts.append(part)
        out.append({**m, "content": parts})
    return out
