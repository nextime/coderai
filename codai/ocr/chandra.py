# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""Chandra OCR engine — Datalab's document VLM, driven over an OpenAI chat endpoint.

Chandra (``datalab-to/chandra-ocr-2``) comes from the same lab as Surya and is its
successor in spirit: one page image in, the whole page out as HTML. Unlike Surya it is
NOT a detector plus a recogniser — it is a single vision model, so (like olmOCR) we talk
to it over an OpenAI chat endpoint and nothing of its own runs here.

What makes it worth having alongside Surya-2 and olmOCR:

- **It returns layout boxes.** Its ``ocr_layout`` prompt asks for one ``<div>`` per layout
  block carrying ``data-bbox`` (normalised 0-1000) and ``data-label`` from a fixed label
  set. So it gives boxes AND labelled regions, which olmOCR cannot, and it gives them in
  one pass, which Surya needs a detector for.
- **A different family.** Chandra 2 is a Qwen3-VL fine-tune with its own tokenizer, where
  Surya-2 has its own 65k ``WordLevel`` vocabulary. That matters for the unreadable-glyph
  problem (see ``codai.ocr.base.UNREADABLE``): running a page that loses its accents in
  Surya through Chandra says whether the loss is the document's pixels or that one model.
- **Tables, maths, chemistry, forms.** Its prompt constrains the output to a known tag set
  with ``<math>`` KaTeX, ``<chem>`` SMILES, colspan/rowspan tables and marked checkboxes.

Three ways to serve it (``ocr.chandra_serve``), the same three as olmOCR:

- ``model``   — through coderai's OWN model manager: register the checkpoint in models.json
  as a vision model and put its id in ``ocr.chandra_model_id``. Gets VRAM accounting,
  eviction and the cluster/RunPod dispatcher for free, and is the only mode that works on a
  box with no GPU of its own — which is also the box whose whole job is to rent one.
- ``vllm``    — coderai's vLLM backend serves ``ocr.chandra_model`` on its own instance.
- ``server``  — attach to an OpenAI-compatible server already running it
  (``ocr.chandra_server_url``).

LICENSE: the code is Apache-2.0 but the WEIGHTS are a modified OpenRAIL-M — free for
research, personal use and companies under $2M funding/revenue, and explicitly not for use
competing with Datalab's own API. That is a narrower grant than Surya's GPL, so this engine
is license-gated the same way: ``ocr.chandra_accept_license`` must be set deliberately.
"""

import base64
import io
import json
import threading
from html.parser import HTMLParser
from typing import List, Optional

from codai.ocr.base import (OcrEngine, OcrPage, OcrLine, OcrRegion, OcrError,
                            logprobs_of, mean_token_prob, vlm_serve_limits)


DEFAULT_MODEL = "datalab-to/chandra-ocr-2"

VALID_MODES = ("model", "vllm", "server")

#: chandra/settings.py: MAX_OUTPUT_TOKENS. A dense page genuinely needs this much —
#: the model emits HTML, which is far more verbose than the text it carries.
DEFAULT_MAX_TOKENS = 12384

#: chandra/settings.py: BBOX_SCALE. data-bbox coordinates are normalised to this.
BBOX_SCALE = 1000

#: Labels the ``ocr_layout`` prompt tells the model to use. Kept so a label the model
#: invented can be told from one it was asked for, rather than both arriving as "a label".
LAYOUT_LABELS = (
    "Caption", "Footnote", "Equation-Block", "List-Group", "Page-Header", "Page-Footer",
    "Image", "Section-Header", "Table", "Text", "Complex-Block", "Code-Block", "Form",
    "Table-Of-Contents", "Figure", "Chemical-Block", "Diagram", "Blank-Page",
)

#: Labels whose block is page furniture rather than content.
_FURNITURE = {"Page-Header", "Page-Footer"}


# ---------------------------------------------------------------------------
# the prompts
# ---------------------------------------------------------------------------
# Mirrored VERBATIM from chandra/prompts.py rather than importing chandra, which would
# drag torch, transformers, bs4 and markdownify into the main venv for two string
# constants. The model was trained on this exact wording — including the Python repr of
# the tag and attribute lists, which is why they are formatted the way they are.

ALLOWED_TAGS = [
    "math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr", "td", "p", "th",
    "div", "pre", "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li", "input", "a", "span",
    "img", "hr", "tbody", "small", "caption", "strong", "thead", "big", "code", "chem",
]

ALLOWED_ATTRIBUTES = [
    "class", "colspan", "rowspan", "display", "checked", "type", "border", "value",
    "style", "href", "alt", "align", "data-bbox", "data-label",
]

PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()

OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page

{PROMPT_ENDING}
""".strip()

OCR_PROMPT = f"""
OCR this image to HTML.

{PROMPT_ENDING}
""".strip()

PROMPT_MAPPING = {"ocr_layout": OCR_LAYOUT_PROMPT, "ocr": OCR_PROMPT}


# ---------------------------------------------------------------------------
# image prep
# ---------------------------------------------------------------------------

def scale_to_fit(img, max_size=(3072, 2048), min_size=(1792, 28), grid_size: int = 28):
    """Chandra's own input scaling (chandra/model/util.py), reimplemented.

    Two things matter and neither is optional. The grid: a Qwen-family vision tower
    tokenises in 28x28 patches, so dimensions are snapped to multiples of 28 — handing it
    an off-grid image makes the server do this itself, differently. And the ceiling:
    3072x2048 is 6,291,456 pixels, which is also Surya-2's cap, and it is reached by an A4
    page at about 255 DPI (see ``OcrConfig.chandra_dpi``). Above that the page is scaled
    DOWN before the model sees it, so rasterising higher only costs time.

    The refinement loop is chandra's: after snapping to the grid the block count can creep
    back over the ceiling, so blocks are shed one at a time, taking the one that distorts
    the aspect ratio less.
    """
    from PIL import Image

    width, height = img.size
    if width <= 0 or height <= 0:
        return img

    original_ar = width / height
    current_pixels = width * height
    max_pixels = max_size[0] * max_size[1]
    min_pixels = min_size[0] * min_size[1]

    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    w_blocks = max(1, round((width * scale) / grid_size))
    h_blocks = max(1, round((height * scale) / grid_size))

    while (w_blocks * h_blocks * grid_size * grid_size) > max_pixels:
        if w_blocks == 1 and h_blocks == 1:
            break
        if w_blocks == 1:
            h_blocks -= 1
            continue
        if h_blocks == 1:
            w_blocks -= 1
            continue
        ar_w_loss = abs(((w_blocks - 1) / h_blocks) - original_ar)
        ar_h_loss = abs((w_blocks / (h_blocks - 1)) - original_ar)
        if ar_w_loss < ar_h_loss:
            w_blocks -= 1
        else:
            h_blocks -= 1

    new_width, new_height = w_blocks * grid_size, h_blocks * grid_size
    if (new_width, new_height) == (width, height):
        return img
    return img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)


def prepare_image(image) -> (str, tuple):
    """``(data URI, (w, h))`` for the image as the MODEL will see it.

    The size is returned because the boxes come back normalised against it, and
    de-normalising them against the pre-scaling size would put every box in the wrong
    place on any page that was scaled.
    """
    img = scale_to_fit(image.convert("RGB"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return uri, img.size


# ---------------------------------------------------------------------------
# degenerate output
# ---------------------------------------------------------------------------

def detect_repeat_token(text: str, base_max_repeats: int = 4, window_size: int = 500,
                        cut_from_end: int = 0, scaling_factor: float = 3.0) -> bool:
    """Whether the reply ends in a repetition loop (chandra/model/util.py).

    A VLM that loses its place on a page does not fail — it repeats the last thing it
    saw until it runs out of tokens, and the result arrives looking like a successful
    transcription. Chandra's own client treats this as a retryable error, and so do we:
    shorter repeated sequences need more repetitions before they count, because a real
    document legitimately repeats short strings (table cells, dotted leaders).
    """
    if cut_from_end > 0:
        text = text[:-cut_from_end]
    for seq_len in range(1, window_size // 2 + 1):
        candidate = text[-seq_len:]
        if not candidate:
            continue
        max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))
        repeat_count = 0
        pos = len(text) - seq_len
        if pos < 0:
            continue
        while pos >= 0:
            if text[pos:pos + seq_len] == candidate:
                repeat_count += 1
                pos -= seq_len
            else:
                break
        if repeat_count > max_repeats:
            return True
    return False


# ---------------------------------------------------------------------------
# parsing the reply
# ---------------------------------------------------------------------------

class _Blocks(HTMLParser):
    """Split a reply into its TOP-LEVEL ``<div>`` blocks, keeping each one's inner HTML.

    bs4 would be three lines of this, and is not a dependency of the main venv — adding
    one for a tag split that stdlib does is not a trade worth making. Depth tracking is
    the whole job: layout blocks nest (a table block contains divs), and only the outermost
    carries the page-level ``data-bbox``.
    """

    #: Tags the model emits unclosed, which must not push the depth counter.
    VOID = {"br", "hr", "img", "input", "meta", "link"}

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.blocks: List[dict] = []
        self._depth = 0
        self._attrs = {}
        self._parts: List[str] = []
        #: Content that arrived outside any div — a model that ignored the block
        #: instruction still transcribed the page, and dropping it would be the
        #: worst possible reading of "the output was not in the expected shape".
        self.loose: List[str] = []

    def handle_starttag(self, tag, attrs):
        text = self.get_starttag_text() or ""
        if tag == "div":
            if self._depth == 0:
                self._attrs = {k: v for k, v in attrs}
                self._parts = []
                self._depth = 1
                return
            self._depth += 1
        self._emit(text)

    def handle_startendtag(self, tag, attrs):
        self._emit(self.get_starttag_text() or "")

    def handle_endtag(self, tag):
        if tag == "div" and self._depth > 0:
            self._depth -= 1
            if self._depth == 0:
                self.blocks.append({"attrs": self._attrs,
                                    "html": "".join(self._parts)})
                self._attrs, self._parts = {}, []
                return
        if tag in self.VOID:
            return
        self._emit(f"</{tag}>")

    def handle_data(self, data):
        self._emit(data)

    def handle_entityref(self, name):
        self._emit(f"&{name};")

    def handle_charref(self, name):
        self._emit(f"&#{name};")

    def _emit(self, text):
        if self._depth > 0:
            self._parts.append(text)
        elif text.strip():
            self.loose.append(text)

    def close(self):
        super().close()
        # An unclosed final div still holds a block's worth of page.
        if self._depth > 0 and self._parts:
            self.blocks.append({"attrs": self._attrs, "html": "".join(self._parts)})
            self._attrs, self._parts, self._depth = {}, [], 0


def html_to_text(html: str) -> str:
    """Tag-stripped text of a block, with structure turned into newlines.

    ``<math>``/``<chem>`` content is LaTeX and SMILES and survives the strip as-is, which
    is what a text consumer wants: losing the formula entirely would be worse than having
    it unrendered.
    """
    import html as _htmlmod
    import re
    if not html:
        return ""
    s = re.sub(r"(?i)<br\s*/?>", "\n", html)
    # Cells get a SPACE, not a newline: </tr> already ends the row, and without this
    # a table flattens to "VoceImporto" — two columns welded into one word, which is
    # worse than useless to anything indexing or searching the text.
    s = re.sub(r"(?i)</(td|th)>", " ", s)
    s = re.sub(r"(?i)</(p|div|li|tr|h[1-6]|pre|caption)>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = _htmlmod.unescape(s)
    return "\n".join(ln.strip() for ln in s.splitlines() if ln.strip()).strip()


def parse_bbox(value, width: int, height: int, scale: int = BBOX_SCALE
               ) -> Optional[List[float]]:
    """``"x0 y0 x1 y1"`` normalised to ``scale`` → pixel box, or None.

    None rather than a full-page box when it cannot be read: a wrong box pointing at the
    whole page looks like a real answer, and anything drawing on it or cropping by it would
    be silently wrong. A missing box is at least visibly missing.
    """
    try:
        parts = str(value or "").replace(",", " ").split()
        nums = [float(p) for p in parts]
    except Exception:
        return None
    if len(nums) != 4 or scale <= 0:
        return None
    sx, sy = width / float(scale), height / float(scale)
    x0, y0, x1, y1 = nums
    box = [max(0.0, x0 * sx), max(0.0, y0 * sy),
           min(x1 * sx, float(width)), min(y1 * sy, float(height))]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def parse_page(content: str, width: int, height: int, index: int = 0,
               include_furniture: bool = True,
               conf: Optional[float] = None) -> OcrPage:
    """Chandra's reply → an :class:`OcrPage`.

    One top-level div is one layout block, and becomes all three of the things coderai's
    result types carry: a line (its text), a region (its label and box) and — when it is a
    table — a table entry keeping the HTML, because a table flattened to text has lost the
    only thing that made it a table.
    """
    page = OcrPage(index=index, width=int(width), height=int(height))
    score = float(conf) if isinstance(conf, (int, float)) else 0.0
    parser = _Blocks()
    parser.feed(content or "")
    parser.close()

    texts, labels = [], {}
    for blk in parser.blocks:
        attrs = blk["attrs"]
        label = (attrs.get("data-label") or "").strip() or "block"
        if label == "Blank-Page":
            continue
        if not include_furniture and label in _FURNITURE:
            continue
        box = parse_bbox(attrs.get("data-bbox"), width, height)
        text = html_to_text(blk["html"])
        labels[label] = labels.get(label, 0) + 1
        if box is not None:
            page.regions.append(OcrRegion(label=label, bbox=box, conf=score))
        if not text:
            continue
        # ``score`` is the mean token probability of the whole reply, because
        # chandra generates the ENTIRE page in one call -- unlike surya, which
        # calls per block and so has a genuinely per-block figure. The same number
        # is therefore put on every line, and ``meta.confidence_scope`` says
        # "page" so nobody reads it as per-line precision it does not have. 0.0
        # when the server reported no logprobs, and ``confidence_measured`` says
        # which, because a 0.0 that means "not measured" must not be mistaken for
        # a 0.0 that means "unsure".
        page.lines.append(OcrLine(text=text, bbox=box or [0.0, 0.0, 0.0, 0.0],
                                  conf=score))
        texts.append(text)
        if label == "Table":
            page.tables.append({"bbox": box, "html": blk["html"]})

    # A reply with no blocks at all is still a transcription — keep it.
    if not texts and parser.loose:
        loose = html_to_text("".join(parser.loose))
        if loose:
            texts.append(loose)
            page.lines.append(OcrLine(text=loose, bbox=[0.0, 0.0, 0.0, 0.0],
                                      conf=score))
    if not texts and not parser.blocks:
        loose = html_to_text(content or "")
        if loose:
            texts.append(loose)
            page.lines.append(OcrLine(text=loose, bbox=[0.0, 0.0, 0.0, 0.0],
                                      conf=score))

    page.text = "\n\n".join(texts)
    if labels:
        page.meta["labels"] = labels
    page.meta["confidence"] = round(score, 4)
    page.meta["confidence_measured"] = isinstance(conf, (int, float))
    page.meta["confidence_scope"] = "page"
    return page


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

_loop = None
_loop_lock = threading.Lock()


def _own_loop():
    """A module-owned event loop, for the in-process model path when no host loop was
    handed to us (direct and unit-test calls; the pool passes the app's own)."""
    import asyncio
    global _loop
    with _loop_lock:
        if _loop is not None and not _loop.is_closed():
            return _loop
        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, name="chandra-loop",
                         daemon=True).start()
        return _loop


class ChandraEngine(OcrEngine):
    name = "chandra"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._base = ""              # OpenAI base URL for the http modes
        self._host_loop = None       # set by the pool: the app's event loop

    # -- config ------------------------------------------------------------

    def _mode(self) -> str:
        mode = (getattr(self.cfg, "chandra_serve", "model") or "model").strip().lower()
        if mode not in VALID_MODES:
            raise OcrError(
                f"invalid ocr.chandra_serve '{mode}' (use {'/'.join(VALID_MODES)})",
                status=400)
        return mode

    def _model_name(self) -> str:
        return (getattr(self.cfg, "chandra_model", "") or DEFAULT_MODEL).strip()

    def _prompt(self) -> str:
        """``ocr_layout`` unless the operator turned layout off.

        Layout is the default because it is the reason to pick this engine over olmOCR:
        without it the reply is one unlocated blob and the boxes coderai's result type
        carries are all empty.
        """
        want_layout = bool(getattr(self.cfg, "chandra_layout", True))
        return PROMPT_MAPPING["ocr_layout" if want_layout else "ocr"]

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        if not bool(getattr(self.cfg, "chandra_accept_license", False)):
            raise OcrError(
                "Chandra is license-gated: its WEIGHTS are a modified OpenRAIL-M "
                "(free for research, personal use and companies under $2M "
                "funding/revenue, and not for use competing with Datalab's own API). "
                "Set ocr.chandra_accept_license = true to use it.",
                status=400)
        mode = self._mode()
        self._base = ""
        if mode == "model":
            if not (getattr(self.cfg, "chandra_model_id", "") or "").strip():
                raise OcrError(
                    "Chandra 'model' mode needs ocr.chandra_model_id — the id of a "
                    "VISION model in models.json holding a Chandra checkpoint (it must "
                    "be served over HTTP: backend \"runpod\", a service_url, or the "
                    "local vLLM backend). Set it, or switch ocr.chandra_serve to "
                    "vllm/server.", status=400)
            return
        if mode == "server":
            url = (getattr(self.cfg, "chandra_server_url", "") or "").strip()
            if not url:
                raise OcrError(
                    "Chandra 'server' mode needs ocr.chandra_server_url (an "
                    "OpenAI-compatible endpoint already serving Chandra).", status=400)
            self._base = url.rstrip("/")
            if not self._base.endswith("/v1"):
                self._base += "/v1"
            return
        # vllm: boot (or reuse) coderai's vLLM backend on the Chandra checkpoint.
        from codai.api import vllm_worker
        from codai.models.manager import get_active_vllm_config
        vcfg = get_active_vllm_config()
        if vcfg is None:
            raise OcrError("Chandra vllm mode needs the vLLM backend configured",
                           status=400)
        model = self._model_name()
        base = vllm_worker.ensure_service(vcfg, model_path=model, served_name=model,
                                          gpu_memory_utilization=self._vlm_gmu(),
                                          **vlm_serve_limits(self.cfg))
        self._base = base.rstrip("/") + "/v1"

    def cleanup(self) -> None:
        # Nothing of ours holds VRAM: 'model' mode is the model manager's, 'vllm' mode is
        # the shared service's (stopped by the manager's releaser), 'server' mode is
        # another process entirely.
        self._loaded = False

    def vram_gb(self) -> float:
        # Per INSTANCE this is one in-flight page against a shared server, so nil. The
        # real cost is declared once through prelaunch_vram_gb().
        return 0.0

    def prelaunch_vram_gb(self) -> float:
        """vLLM claims gpu_memory_utilization × the whole card before it will serve and
        refuses to start when that much is not free, so it has to be freed BEFORE the
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
            from codai.api.vllm_worker import prelaunch_free_gb
            from codai.models.manager import get_active_vllm_config
            vcfg = get_active_vllm_config()
            return prelaunch_free_gb(vcfg, self._vlm_gmu()) if vcfg is not None else 0.0
        except Exception:
            return 0.0

    # -- inference ---------------------------------------------------------

    def recognize_image(self, image) -> OcrPage:
        src_w, src_h = image.size
        data_uri, (w, h) = prepare_image(image)
        content, meta, conf = self._ask_with_retries(data_uri)
        page = parse_page(
            content, w, h,
            include_furniture=bool(getattr(self.cfg, "chandra_include_furniture", True)),
            conf=conf)
        # The boxes are the model's, normalised against the image the model saw; the page
        # dimensions reported are the same ones, so a client scaling boxes back onto its
        # own render has a consistent pair. The source size is reported too, since it is
        # the one the caller rasterised and the only way to see that scaling happened.
        if (src_w, src_h) != (w, h):
            meta["source_size"] = [int(src_w), int(src_h)]
        page.meta.update(meta)
        return page

    def _ask_with_retries(self, data_uri) -> (str, dict, "Optional[float]"):
        """The page, re-asked while the reply is degenerate.

        A repetition loop is not an error the server reports — it answers 200 with a
        plausible-looking page. Chandra's own client raises the temperature on each retry
        (a greedy decode that fell into a loop will fall into the same one again) and so do
        we, capped the same way.
        """
        base_temp = float(getattr(self.cfg, "chandra_temperature", 0.0) or 0.0)
        top_p = float(getattr(self.cfg, "chandra_top_p", 0.1) or 0.1)
        max_retries = max(0, int(getattr(self.cfg, "chandra_max_retries", 2) or 0))
        meta = {}
        content, conf = "", None
        for attempt in range(max_retries + 1):
            if attempt == 0:
                temp, p = base_temp, top_p
            else:
                temp, p = min(base_temp + 0.2 * attempt, 0.8), 0.95
            content, conf = self._ask(data_uri, temp, p)
            if not detect_repeat_token(content):
                if attempt:
                    meta["retries"] = attempt
                return content, meta, conf
        # Out of retries, and the last reply still loops. Returned anyway, flagged: a
        # looping transcription usually has a correct beginning, and the flag is what lets
        # a caller decide rather than be handed a silent mess.
        meta["retries"] = max_retries
        meta["repeat_detected"] = True
        return content, meta, conf

    def _ask(self, data_uri, temperature: float, top_p: float):
        """One page → ``(raw reply text, mean token probability or None)``."""
        messages = [{
            "role": "user",
            "content": [
                # Image FIRST, then the instruction: that is the order chandra's own
                # client builds (chandra/model/vllm.py), and for a VLM the order the
                # model was trained on is part of the prompt.
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": self._prompt()},
            ],
        }]
        max_tokens = int(getattr(self.cfg, "chandra_max_tokens", DEFAULT_MAX_TOKENS)
                         or DEFAULT_MAX_TOKENS)
        if self._mode() == "model":
            return self._ask_model(messages, max_tokens, temperature, top_p)
        return self._ask_http(messages, max_tokens, temperature, top_p)

    def _want_logprobs(self) -> bool:
        """Whether to ask the server to report per-token logprobs.

        They are what makes ``conf`` a measurement rather than a guess (see
        base.mean_token_prob). Cheap — a float per token — and the only reason to
        turn it off is a server that refuses the field outright."""
        return bool(getattr(self.cfg, "chandra_logprobs", True))

    def _ask_model(self, messages, max_tokens, temperature, top_p):
        """Model mode: ask the dispatcher which server should take this page.

        The in-process path is right for a model this machine serves and wrong for one
        served elsewhere — it cannot see which cluster node is least busy, and it bypasses
        the front, where burst-to-RunPod lives. So the choice is made per page.
        """
        model_id = (getattr(self.cfg, "chandra_model_id", "") or "").strip()
        try:
            from codai.models import placement
            lease = placement.choose(
                model_id, spill_after=placement.spill_threshold(model_id))
        except Exception:
            return self._ask_local(messages, max_tokens, temperature, top_p)
        try:
            if lease.kind == "local":
                return self._ask_local(messages, max_tokens, temperature, top_p)
            return self._ask_http(messages, max_tokens, temperature, top_p,
                                  base=lease.base, api_key=lease.api_key,
                                  model_name=model_id)
        finally:
            lease.release()

    def _ask_local(self, messages, max_tokens, temperature, top_p):
        """Through coderai's own in-process chat handler (model mode)."""
        import asyncio

        from codai.api.text import chat_completions
        from codai.pydantic.textrequest import ChatCompletionRequest

        model_id = (getattr(self.cfg, "chandra_model_id", "") or "").strip()
        req = ChatCompletionRequest(
            model=model_id, messages=messages, stream=False,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        )
        loop = self._host_loop
        if loop is None or loop.is_closed():
            loop = _own_loop()
        fut = asyncio.run_coroutine_threadsafe(chat_completions(req, None), loop)
        try:
            result = fut.result(
                timeout=float(getattr(self.cfg, "chandra_timeout", 300.0) or 300.0))
        except Exception as e:
            raise OcrError(f"Chandra model '{model_id}' call failed: {e}", status=500)
        content = _content_of(result)
        if not content.strip():
            raise OcrError(f"Chandra model '{model_id}' returned no text", status=500)
        # No confidence from this path: coderai's in-process chat handler does not
        # surface per-token logprobs. Reported as "not measured" rather than 0.0
        # pretending to be one.
        return content, None

    def _ask_http(self, messages, max_tokens, temperature, top_p, base: str = "",
                  api_key: str = "", model_name: str = ""):
        """Against an OpenAI-compatible server (vllm/server modes).

        ``base``/``api_key``/``model_name`` override the configured server for one page,
        which is how model mode reaches the upstream the dispatcher chose.
        """
        # pod_http, not requests: in model mode ``base`` can be a RunPod pod on the
        # direct-TCP path — https to a bare IP serving a certificate signed by THIS
        # install's own CA, which no public CA knows and which was issued before the
        # address was. Plain requests fails it twice over with CERTIFICATE_VERIFY_FAILED.
        # pod_http is a drop-in that hands every other URL straight to requests.
        from codai.api import pod_http

        payload = {
            "model": model_name or self._model_name(),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if self._want_logprobs():
            payload["logprobs"] = True
        headers = {"Content-Type": "application/json"}
        key = api_key or (getattr(self.cfg, "chandra_api_key", "") or "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = (base or self._base).rstrip("/") + "/chat/completions"
        try:
            r = pod_http.post(
                url, json=payload, headers=headers,
                timeout=float(getattr(self.cfg, "chandra_timeout", 300.0) or 300.0))
        except Exception as e:
            raise OcrError(f"Chandra server {url} unreachable: {e}", status=503)
        if r.status_code >= 400:
            raise OcrError(f"Chandra server {url} returned HTTP {r.status_code}: "
                           f"{r.text[:300]}", status=502)
        try:
            body = r.json()
            content = _content_of(body)
        except Exception as e:
            raise OcrError(f"Chandra server returned unparseable JSON: {e}", status=502)
        if not content.strip():
            raise OcrError("Chandra server returned no text", status=502)
        # None when the server declined logprobs, which the page then reports as
        # confidence_measured: false. A server that ignores the field is a
        # configuration, not an error.
        return content, mean_token_prob(logprobs_of(body))


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
