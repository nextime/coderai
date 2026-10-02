# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""olmOCR-2 engine — the AllenAI document VLM, driven over an OpenAI chat endpoint.

Unlike paddle/docTR/Surya-local, olmOCR-2 (``allenai/olmOCR-2-7B-1025``, a Qwen2.5-VL-7B
fine-tune, Apache-2.0) is a *vision language model*: one page image in, the whole page's
text out, in reading order, with equations as LaTeX and tables as HTML. It has no text
detector, so it returns NO per-line bounding boxes — pick it for transcription fidelity on
hard documents (old scans, maths, multi-column, tables), and paddle/Surya when you need
boxes or layout regions.

Three ways to serve it (``ocr.olmocr_serve``):

- ``model``   — through coderai's OWN model manager: register the checkpoint in
  models.json as a vision model (GGUF+mmproj on llama.cpp, or an HF/qwenvl entry) and put
  its id in ``ocr.olmocr_model_id``. Gets VRAM accounting, eviction, quantisation and the
  thermal governor for free, and is the only mode that works on a Vulkan/ROCm box.
- ``vllm``    — coderai's vLLM backend serves ``ocr.olmocr_model`` on its own instance
  (continuous batching, CUDA only), exactly like the Surya-2 path.
- ``server``  — attach to an OpenAI-compatible server already running the model
  (``ocr.olmocr_server_url``): llama-server, a remote vLLM, another coderai.

The prompt is olmOCR's own "no-anchoring v4 YAML" prompt, so the model answers in its
trained format: a YAML front matter block (primary_language, is_rotation_valid,
rotation_correction, is_table, is_diagram) followed by the page as markdown. The front
matter is parsed into ``OcrPage.meta`` and stripped from ``text``; when the model reports
the page as rotated we re-render it turned and ask once more (what the olmOCR toolkit's
pipeline does).
"""

import base64
import io
import json
import threading
from typing import Optional

from codai.ocr.base import OcrEngine, OcrPage, OcrError


#: olmocr.prompts.build_no_anchoring_v4_yaml_prompt() — kept verbatim: the model was
#: trained on this wording, so paraphrasing it costs accuracy. Mirrored here rather than
#: importing olmocr, which would drag the whole toolkit (and its pinned deps) in.
OLMOCR_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the "
    "primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)

#: The model card's rendering rule: longest side 1288 px. Off-size pages still work, but
#: this is what it was trained and benchmarked on.
DEFAULT_LONGEST_SIDE = 1288

VALID_MODES = ("model", "vllm", "server")

DEFAULT_MODEL = "allenai/olmOCR-2-7B-1025-FP8"

_FRONT_MATTER_KEYS = (
    "primary_language", "is_rotation_valid", "rotation_correction",
    "is_table", "is_diagram",
)


# ---------------------------------------------------------------------------
# front matter
# ---------------------------------------------------------------------------

def _coerce(value: str):
    v = (value or "").strip().strip('"').strip("'")
    lo = v.lower()
    if lo in ("true", "yes"):
        return True
    if lo in ("false", "no"):
        return False
    if lo in ("none", "null", "~", ""):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_front_matter(content: str) -> (dict, str):
    """Split olmOCR's reply into (metadata, body).

    The reply normally opens with a ``---`` fenced YAML block. Models drift: the fence may
    be wrapped in a markdown code block, the opening fence may be missing entirely, or the
    block may not be there at all. Anything we cannot read as front matter stays in the
    body — losing page text to a parse detail would be much worse than a missing flag.
    """
    s = (content or "").strip()
    if s.startswith("```"):
        # ```markdown\n---\n…  — drop the fence line, and a trailing fence if it closes.
        nl = s.find("\n")
        s = s[nl + 1:] if nl != -1 else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()

    lines = s.splitlines()
    start = 0
    if lines and lines[0].strip() == "---":
        start = 1
    elif lines and lines[0].split(":", 1)[0].strip() in _FRONT_MATTER_KEYS:
        start = 0          # fence omitted but the first key is one of ours
    else:
        return {}, s

    meta, end = {}, None
    for i in range(start, len(lines)):
        line = lines[i]
        if line.strip() == "---":
            end = i
            break
        if not line.strip():
            if start == 0:
                end = i      # unfenced block ends at the first blank line
                break
            continue
        if ":" not in line:
            if start == 0:
                end = i
                break
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        if key:
            meta[key] = _coerce(val)

    if not meta:
        return {}, s
    body = "\n".join(lines[(end + 1) if end is not None else len(lines):]).strip()
    return meta, body


# ---------------------------------------------------------------------------
# image prep
# ---------------------------------------------------------------------------

def prepare_image(image, longest_side: int = DEFAULT_LONGEST_SIDE) -> str:
    """Return a ``data:image/png;base64,…`` URI of ``image`` scaled to ``longest_side``.

    Only ever scales DOWN past the target: upsampling a small crop buys nothing and costs
    vision tokens.
    """
    from PIL import Image

    img = image.convert("RGB")
    target = max(64, int(longest_side or DEFAULT_LONGEST_SIDE))
    w, h = img.size
    longest = max(w, h)
    if longest > target:
        scale = target / float(longest)
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

_loop = None
_loop_lock = threading.Lock()


def _own_loop():
    """A module-owned event loop, used only when no host loop was handed to us.

    ``recognize_image`` runs in a worker thread (the pool calls it through
    ``asyncio.to_thread``), so reaching coderai's async model path needs a loop. The pool
    passes the app's own loop — this is the fallback for direct/unit-test calls."""
    import asyncio
    global _loop
    with _loop_lock:
        if _loop is not None and not _loop.is_closed():
            return _loop
        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, name="olmocr-loop",
                         daemon=True).start()
        return _loop


class OlmOcrEngine(OcrEngine):
    name = "olmocr"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._base = ""              # OpenAI base URL for the http modes
        self._host_loop = None       # set by the pool: the app's event loop

    # -- config ------------------------------------------------------------

    def _mode(self) -> str:
        mode = (getattr(self.cfg, "olmocr_serve", "model") or "model").strip().lower()
        if mode not in VALID_MODES:
            raise OcrError(
                f"invalid ocr.olmocr_serve '{mode}' (use {'/'.join(VALID_MODES)})",
                status=400)
        return mode

    def _model_name(self) -> str:
        return (getattr(self.cfg, "olmocr_model", "") or DEFAULT_MODEL).strip()

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        mode = self._mode()
        self._base = ""
        if mode == "model":
            if not (getattr(self.cfg, "olmocr_model_id", "") or "").strip():
                raise OcrError(
                    "olmOCR 'model' mode needs ocr.olmocr_model_id — the id of a VISION "
                    "model in models.json holding an olmOCR-2 checkpoint (GGUF+mmproj or "
                    "HF). Set it, or switch ocr.olmocr_serve to vllm/server.",
                    status=400)
            return
        if mode == "server":
            url = (getattr(self.cfg, "olmocr_server_url", "") or "").strip()
            if not url:
                raise OcrError(
                    "olmOCR 'server' mode needs ocr.olmocr_server_url (an OpenAI-compatible "
                    "endpoint already serving olmOCR-2).", status=400)
            self._base = url.rstrip("/")
            if not self._base.endswith("/v1"):
                self._base += "/v1"
            return
        # vllm: boot (or reuse) coderai's vLLM backend on the olmOCR checkpoint.
        from codai.api import vllm_worker
        from codai.models.manager import get_active_vllm_config
        vcfg = get_active_vllm_config()
        if vcfg is None:
            raise OcrError("olmOCR vllm mode needs the vLLM backend configured", status=400)
        model = self._model_name()
        base = vllm_worker.ensure_service(vcfg, model_path=model, served_name=model,
                                          gpu_memory_utilization=self._vlm_gmu())
        self._base = base.rstrip("/") + "/v1"

    def cleanup(self) -> None:
        # Nothing of ours holds VRAM: in 'model' mode the model manager owns the weights,
        # in 'vllm' mode the shared service does (stopped by the manager's releaser), and
        # 'server' mode is someone else's process.
        self._loaded = False

    def vram_gb(self) -> float:
        # Nothing is held PER INSTANCE: an instance is one in-flight page against a shared
        # server. In 'model' mode the model manager already accounts for the weights,
        # 'server' mode is another process entirely, and the 'vllm' instance is accounted
        # for once through prelaunch_vram_gb() and released by the manager's stop estimate.
        return 0.0

    def prelaunch_vram_gb(self) -> float:
        """vLLM claims gpu_memory_utilization × the whole card before it will serve, and
        refuses to start when that much is not free — so it has to be freed BEFORE the
        load, not after it (see Surya's crash loop)."""
        return self._vllm_need() if self._mode() == "vllm" else 0.0

    def _vlm_gmu(self):
        """The OCR-side share of the card for this engine's vLLM (0/None = the backend's)."""
        try:
            v = float(getattr(self.cfg, "vlm_gpu_memory_utilization", 0.0) or 0.0)
        except Exception:
            v = 0.0
        return v or None

    def _vllm_need(self) -> float:
        try:
            from codai.api.vllm_worker import planned_vram_gb
            from codai.models.manager import get_active_vllm_config
            vcfg = get_active_vllm_config()
            return planned_vram_gb(vcfg, self._vlm_gmu()) if vcfg is not None else 0.0
        except Exception:
            return 0.0

    # -- inference ---------------------------------------------------------

    def recognize_image(self, image) -> OcrPage:
        w, h = image.size
        page = OcrPage(index=0, width=int(w), height=int(h))

        content = self._ask(image)
        meta, body = parse_front_matter(content)

        rotation = meta.get("rotation_correction") or 0
        valid = meta.get("is_rotation_valid")
        if (valid is False or rotation) and bool(getattr(self.cfg, "olmocr_retry_rotation", True)):
            try:
                turned = int(rotation) % 360
            except Exception:
                turned = 0
            if turned:
                # PIL rotates counter-clockwise; olmOCR reports the correction to APPLY,
                # so expand=True and the same sign the toolkit uses.
                retry_img = image.rotate(-turned, expand=True)
                retry = self._ask(retry_img)
                rmeta, rbody = parse_front_matter(retry)
                if rbody.strip():
                    meta, body = rmeta, rbody
                    meta["rotation_applied"] = turned
                    rw, rh = retry_img.size
                    page.width, page.height = int(rw), int(rh)

        page.text = body
        page.meta = {k: v for k, v in meta.items() if v is not None}
        # No detector, so no lines/regions/tables boxes — the text IS the result.
        return page

    def _ask(self, image) -> str:
        """One page → the model's raw reply text."""
        data_uri = prepare_image(
            image, int(getattr(self.cfg, "olmocr_longest_side", DEFAULT_LONGEST_SIDE) or
                       DEFAULT_LONGEST_SIDE))
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": OLMOCR_PROMPT},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }]
        max_tokens = int(getattr(self.cfg, "olmocr_max_tokens", 4096) or 4096)
        temperature = float(getattr(self.cfg, "olmocr_temperature", 0.1) or 0.0)

        if self._mode() == "model":
            return self._ask_local(messages, max_tokens, temperature)
        return self._ask_http(messages, max_tokens, temperature)

    def _ask_local(self, messages, max_tokens, temperature) -> str:
        """Through coderai's own in-process chat handler (model mode)."""
        import asyncio

        from codai.api.text import chat_completions
        from codai.pydantic.textrequest import ChatCompletionRequest

        model_id = (getattr(self.cfg, "olmocr_model_id", "") or "").strip()
        req = ChatCompletionRequest(
            model=model_id, messages=messages, stream=False,
            temperature=temperature, max_tokens=max_tokens,
        )
        loop = self._host_loop
        if loop is None or loop.is_closed():
            loop = _own_loop()
        fut = asyncio.run_coroutine_threadsafe(chat_completions(req, None), loop)
        try:
            result = fut.result(timeout=float(getattr(self.cfg, "olmocr_timeout", 300.0) or 300.0))
        except Exception as e:
            raise OcrError(f"olmOCR model '{model_id}' call failed: {e}", status=500)
        content = _content_of(result)
        if not content.strip():
            raise OcrError(f"olmOCR model '{model_id}' returned no text", status=500)
        return content

    def _ask_http(self, messages, max_tokens, temperature) -> str:
        """Against an OpenAI-compatible server (vllm/server modes)."""
        import requests

        payload = {
            "model": self._model_name(),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        key = (getattr(self.cfg, "olmocr_api_key", "") or "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = self._base.rstrip("/") + "/chat/completions"
        try:
            r = requests.post(url, json=payload, headers=headers,
                              timeout=float(getattr(self.cfg, "olmocr_timeout", 300.0) or 300.0))
        except Exception as e:
            raise OcrError(f"olmOCR server {url} unreachable: {e}", status=503)
        if r.status_code >= 400:
            raise OcrError(f"olmOCR server {url} returned HTTP {r.status_code}: "
                           f"{r.text[:300]}", status=502)
        try:
            content = _content_of(r.json())
        except Exception as e:
            raise OcrError(f"olmOCR server returned unparseable JSON: {e}", status=502)
        if not content.strip():
            raise OcrError("olmOCR server returned no text", status=502)
        return content


def _content_of(result) -> str:
    """Assistant message content out of a chat-completions result, whatever shape it
    arrives in (dict, pydantic model, JSONResponse)."""
    r = None
    if isinstance(result, dict):
        r = result
    elif hasattr(result, "model_dump"):
        try:
            r = result.model_dump()
        except Exception:
            r = None
    if r is None and hasattr(result, "body"):
        try:
            r = json.loads(result.body)
        except Exception:
            r = None
    if not isinstance(r, dict):
        return ""
    choices = r.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    msg = first.get("message")
    if isinstance(msg, dict):
        c = msg.get("content")
        if isinstance(c, list):      # multipart reply — join its text parts
            return "".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c or ""
    return first.get("text") or ""
