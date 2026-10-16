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

"""OCR engine interface + shared result types + input loading.

Every OCR engine (PaddleOCR, docTR, Surya, olmOCR-2) subclasses :class:`OcrEngine` and returns
:class:`OcrPage` objects. The rest of the subsystem (manager, API, detection,
extraction) speaks only these types, so engines stay interchangeable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import contextvars
import threading
from typing import Any, List, Optional


class OcrError(RuntimeError):
    """Raised for OCR failures the API should surface (missing deps, bad input, …).

    ``status`` maps to the HTTP status the endpoint should return (503 when an engine's
    optional dependency is not installed, 400 for bad input, 500 otherwise).
    """

    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Unreadable glyphs
# ---------------------------------------------------------------------------

#: What a VLM OCR engine writes where it cannot identify a character.
#:
#: It may be the model's own answer OR a byte-level accident, and the text alone
#: cannot tell you which. An earlier version of this comment claimed otherwise —
#: that ``à`` was a single token, so one marker had to mean the model chose it.
#: That was wrong, and measurement settled it: surya-ocr-2's vocabulary is
#: BYTE-level, and an accented letter is two tokens.
#:
#:     'à' utf8=[195, 160]  ids=[2131, 2258]  pieces=['Ã', 'ł']
#:     decode([2131])       = U+FFFD          <- a lone lead byte, ONE marker
#:     decode([2131, 2104]) = 'è'
#:
#: So one marker per accented letter is exactly what a byte-level failure looks
#: like, and ``vocab['à'] = 2160`` does not mean the letter is token 2160 — in a
#: byte-mapped vocabulary that key is the glyph standing for byte 0xA0, which
#: alone also decodes to one marker.
#:
#: Either way the loss is silent: ``più`` becomes ``pi?`` and indexes, searches
#: and reads wrong ever after. For Italian judicial documents that is real, so it
#: gets counted and located rather than passed through unremarked — a known
#: unknown beats a quiet corruption. To find out WHOSE fault a given document's
#: markers are, turn on ``OcrConfig.surya_token_trace``, which reports the model's
#: own bytes at each break (see docs/ocr.md).
#:
#: Too few pixels can also cause it — a diacritic is ~3 pixels tall at 200 DPI
#: (see ``OcrConfig.surya_dpi``) — but that has been RULED OUT for the Italian
#: case: a crisp synthetic page at 250 DPI lost every accent under surya and none
#: under chandra.
UNREADABLE = "\ufffd"


def unreadable_offsets(text: str) -> List[int]:
    """Character offsets of every unreadable-glyph marker in ``text``."""
    return [i for i, ch in enumerate(text or "") if ch == UNREADABLE]


def count_unreadable(text: str) -> int:
    """How many characters the engine could not identify."""
    return (text or "").count(UNREADABLE)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class OcrLine:
    """A recognised text line (or word), with its polygon bounding box."""
    text: str
    bbox: List[float]              # [x0, y0, x1, y1] in pixels
    conf: float = 0.0

    def to_dict(self) -> dict:
        out = {"text": self.text, "bbox": self.bbox, "conf": round(self.conf, 4)}
        n = count_unreadable(self.text)
        if n:
            # Omitted when zero: a clean result stays exactly what it always was.
            out["unreadable"] = n
        return out


@dataclass
class OcrRegion:
    """A layout region (title/text/table/figure/seal/…) from layout analysis.

    Used both for structure output and — in the ``layout`` stamp/signature detection
    mode — to flag likely stamps/seals/figures without a separate detector.
    """
    label: str
    bbox: List[float]             # [x0, y0, x1, y1] in pixels
    conf: float = 0.0

    def to_dict(self) -> dict:
        return {"label": self.label, "bbox": self.bbox, "conf": round(self.conf, 4)}


@dataclass
class OcrPage:
    """OCR result for a single page/image."""
    index: int
    text: str = ""
    width: int = 0
    height: int = 0
    lines: List[OcrLine] = field(default_factory=list)
    regions: List[OcrRegion] = field(default_factory=list)
    tables: List[Any] = field(default_factory=list)     # list of {bbox, html?, cells?}
    #: Engine-reported page metadata, when the engine has any (olmOCR's front matter:
    #: primary_language, is_rotation_valid, rotation_correction, is_table, is_diagram).
    #: Omitted from the response when empty, so engines that report none are unchanged.
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "index": self.index,
            "text": self.text,
            "width": self.width,
            "height": self.height,
            "lines": [l.to_dict() for l in self.lines],
            "regions": [r.to_dict() for r in self.regions],
            "tables": self.tables,
        }
        if self.meta:
            # The byte-level trace is reported ONCE, document-wide, by
            # unreadable_report(). Repeating it on every page would put the same
            # findings in the response twice.
            meta = {k: v for k, v in self.meta.items() if k != "unreadable_trace"}
            if meta:
                out["meta"] = meta
        n = count_unreadable(self.text)
        if n:
            out["unreadable"] = n
        return out


# ---------------------------------------------------------------------------
# Engine interface
# ---------------------------------------------------------------------------

class OcrEngine(ABC):
    """Base class for a dedicated OCR engine.

    Subclasses import their heavy dependency LAZILY inside :meth:`load` (never at module
    import time) and raise :class:`OcrError` with ``status=503`` if it is missing, so the
    server runs fine without any OCR library installed.
    """

    name: str = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load models into memory (called once per instance, lazily). Idempotent."""

    @abstractmethod
    def recognize_image(self, image) -> OcrPage:
        """OCR a single page image (a PIL.Image.Image) → :class:`OcrPage`."""

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    def cleanup(self) -> None:
        """Free this instance's VRAM/resources. SYNC (may be called from the model
        manager's VRAM-eviction thread). Overridden per engine."""
        self._loaded = False

    def lanes(self) -> int:
        """How many pages ONE instance may serve at once.

        One by default: an engine is only safe to drive concurrently if it says
        so. The in-process engines (chandra, olmOCR) get their concurrency from
        the instance count instead, each instance being a slot rather than a
        process, so they have no reason to raise this.
        """
        return 1

    def vram_gb(self) -> float:
        """Rough VRAM footprint (GB) of one loaded instance, for eviction accounting."""
        return 1.0

    def prelaunch_vram_gb(self) -> float:
        """VRAM (GB) that must be FREE before :meth:`load` is even attempted.

        Zero for engines that load weights into whatever room is left (paddle/docTR) —
        they are evicted for after the fact, once their real footprint is known. Non-zero
        for the ones that boot a server which claims a fixed share of the card up front
        and refuses to start otherwise (the VLM engines on vLLM): for those, evicting
        after the load attempt is too late, because the load attempt is what fails.

        This is the total for the WHOLE POOL, not per instance: the engines that declare
        it share one server between all their instances, so the pool asks for it once. (A
        pool of 24 surya workers multiplied per instance would have asked for 200 GB.)
        An engine that declares it is NOT also evicted for by :meth:`vram_gb` after the
        load — the room has already been made."""
        return 0.0


# ---------------------------------------------------------------------------
# Input loading (images + PDF) → list of PIL pages
# ---------------------------------------------------------------------------

# Extensions we accept directly as raster images.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".ppm"}


#: Per-REQUEST engine options, for the one page being read right now.
#:
#: A ContextVar rather than an argument because an engine is anything with
#: `recognize_image(image)` -- the fallback chain and several tests drive
#: duck-typed ones -- so widening that signature would break every engine to
#: serve the two that can use it. A ContextVar also survives the hop into
#: `asyncio.to_thread`, which copies the context, so it reaches a pooled
#: subprocess engine on a worker thread and a lane inside it.
#:
#: Why per request at all: one global cap cannot serve a public deployment.
#: A four-page contract and a fifty-page judgment need different budgets, and
#: the caller is the only one who knows which it is sending.
PAGE_OPTIONS = contextvars.ContextVar("ocr_page_options", default=None)


def page_options() -> dict:
    """Options the caller set for this request, or {}."""
    try:
        return dict(PAGE_OPTIONS.get() or {})
    except Exception:
        return {}


def ink_coverage(image, threshold: int = 200) -> float:
    """Fraction of a page's pixels that are not background, 0.0-1.0.

    The cheapest question nobody was asking: **is there anything on this page?**

    Three outcomes are otherwise indistinguishable in a short OCR result -- a
    page the model could not read, a reply that was cut off, and a page with
    nothing on it. The first needs a better model, the second a bigger budget,
    and the third needs the document fixed upstream, so confusing them sends the
    operator looking in the wrong place. Measured on a real one (2026-10-13): a
    four-page court PDF returned 47 characters and was reported as a model
    failure; every page was a ~900-byte CCITT Group 4 image -- what an EMPTY
    page costs -- carrying 0.13% ink, all of it a diagonal scan edge and a
    diagonal watermark. The OCR was perfect. The document was blank.

    A greyscale histogram over an image already in memory, so it costs nothing
    worth measuring.
    """
    try:
        grey = image.convert("L")
        hist = grey.histogram()
        total = float(max(1, image.width * image.height))
        return float(sum(hist[:max(0, min(256, int(threshold)))])) / total
    except Exception:
        return -1.0


def load_pages(data: bytes, filename: str = "", content_type: str = "",
               dpi: int = 200, policy=None) -> List["Any"]:
    """Decode ``data`` (an image or a PDF) into a list of RGB ``PIL.Image`` pages.

    PDFs are rasterised at ``dpi`` via pypdfium2 (preferred) or pdf2image/poppler.
    Raises :class:`OcrError` (status 400) on undecodable input, 503 if PDF support is
    needed but unavailable.
    """
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover - Pillow is a base dep
        raise OcrError(f"Pillow (PIL) is required for OCR: {e}", status=503)

    import io

    name = (filename or "").lower()
    is_pdf = (content_type or "").lower() == "application/pdf" or name.endswith(".pdf") \
        or data[:5] == b"%PDF-"

    if is_pdf:
        return _rasterize_pdf(data, dpi, policy)

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise OcrError(f"could not decode image: {e}", status=400)

    # Multi-frame images (multi-page TIFF) → one page per frame.
    pages = []
    n = getattr(img, "n_frames", 1)
    for i in range(n):
        try:
            img.seek(i)
        except Exception:
            pass
        pages.append(img.convert("RGB"))
    return pages


#: pdfium is NOT thread-safe: concurrent calls corrupt its state and surface as
#: "PDFium: Data format error" on documents that are perfectly valid. Until 0.3.65
#: this was invisible because `load_pages` ran on the event loop, which serialised
#: every rasterisation by accident; moving rendering into a thread (so a long PDF
#: stops stalling the loop) removed that accident and broke OCR for every install
#: doing more than one document at a time. Rendering still happens OFF the loop --
#: it just happens one call at a time.
#:
#: Held per CALL, not per document, so documents interleave page by page instead of
#: queueing behind each other's full length.
_PDFIUM_LOCK = threading.RLock()

#: Measured ratio of real resident bytes to the RGB pixel arithmetic for a decoded
#: page -- decode transients plus allocator slack. See :func:`page_estimate` for the
#: measurements this comes from. Rounded UP from the worst observed 1.56.
PAGE_OVERHEAD = 1.6


def page_estimate(data: bytes, filename: str = "", content_type: str = "",
                  dpi: int = 200) -> "tuple[int, int]":
    """``(num_pages, estimated_resident_bytes)`` for ``data`` WITHOUT rasterising it.

    The point is to know what a document will cost before paying it, so admission
    control can refuse or queue it rather than discover the size by running out of
    memory. pypdfium2 reports each page's point size from the page tree alone, which
    is cheap; scaling that by ``dpi`` gives the pixel area, and RGB is 3 bytes a pixel.

    The figure is what a page actually COSTS resident: RGB at 3 bytes a pixel, times
    :data:`PAGE_OVERHEAD` for the decode transients and allocator slack that the
    pixel arithmetic alone does not predict.

    That factor is measured, not guessed. Holding N A4 pages at 250 DPI and reading
    RSS in a fresh process (40-page PDF, pypdfium2):

        pages held     pixel arithmetic     measured RSS     ratio
            40              691 MB            +974 MB        1.41
            32              553 MB            +768 MB        1.39
             8              138 MB            +215 MB        1.56

    So the pixel count under-predicts by 40-55%, and a budget built on it would
    admit about half again as much as it meant to. Under-charging defeats the gate
    at exactly the moment it matters, which is why the factor rounds up.

    Falls back to a size-derived guess when the document cannot be inspected: an
    unknown document still has to be charged for SOMETHING, and charging it nothing
    is how an unbounded one slips through.
    """
    name = (filename or "").lower()
    is_pdf = ((content_type or "").lower() == "application/pdf"
              or name.endswith(".pdf") or data[:5] == b"%PDF-")
    _BYTES_PER_PIXEL = 3           # decoded RGB

    if is_pdf:
        try:
            import pypdfium2 as pdfium
            with _PDFIUM_LOCK:
                pdf = pdfium.PdfDocument(data)
                scale = max(0.5, dpi / 72.0)
                n = len(pdf)
                # get_page_size() reads the size from the page tree by index.
                # `pdf[i].get_size()` LOADS the page object to ask it the same
                # question: 593 ms for a 47-page document against 1.9 ms, for
                # identical numbers -- and it was being paid under the lock, so
                # every document in the install queued behind it.
                try:
                    sizes = [pdf.get_page_size(i) for i in range(n)]
                except Exception:
                    sizes = [pdf[i].get_size() for i in range(n)]
                total = sum(int(w * scale) * int(h * scale) for w, h in sizes)
            return n, int(total * _BYTES_PER_PIXEL * PAGE_OVERHEAD)
        except Exception:
            # Unreadable page tree, or no pypdfium2. A PDF page is ~A4 far more often
            # than not; assume that rather than zero.
            a4 = int(int(8.27 * dpi) * int(11.69 * dpi)
                     * _BYTES_PER_PIXEL * PAGE_OVERHEAD)
            return 1, max(a4, len(data) * 4)

    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data))        # header only; no .load()
        n = max(1, int(getattr(img, "n_frames", 1)))
        w, h = img.size
        return n, int(w * h * _BYTES_PER_PIXEL * n * PAGE_OVERHEAD)
    except Exception:
        return 1, max(len(data) * 4, 1)


def pdf_page_count(data: bytes) -> int:
    """Pages in a PDF, or 0 if it cannot be read. Cheap: no rendering."""
    try:
        import pypdfium2 as pdfium
        with _PDFIUM_LOCK:
            return len(pdfium.PdfDocument(data))
    except Exception:
        return 0


def split_pdf_page(data: bytes, index: int) -> bytes:
    """One page of ``data`` as a PDF of its own.

    What makes a pod-side pipeline worth having: the page travels as a PDF the
    pod renders itself, instead of as a bitmap this box rendered. Measured on a
    47-page document: 1.2 ms to split against ~105 ms to rasterise, and ~474 KB
    on the wire against ~1.06 MB of base64 PNG.

    `import_pages` copies the objects the page needs, so a text document's pages
    each carry the fonts they reference -- the saving is proportional for scans
    (where a page is mostly its own image) and smaller for font-heavy text. The
    CPU saving is the point; the bandwidth is a bonus.
    """
    import io
    try:
        import pypdfium2 as pdfium
    except Exception as e:
        raise OcrError(f"splitting a PDF needs pypdfium2: {e}", status=503)
    try:
        with _PDFIUM_LOCK:
            src = pdfium.PdfDocument(data)
            out = pdfium.PdfDocument.new()
            out.import_pages(src, [int(index)])
            buf = io.BytesIO()
            out.save(buf)
        return buf.getvalue()
    except Exception as e:
        raise OcrError(f"failed to split PDF page {index} (pypdfium2): {e}",
                       status=400)


def iter_pdf_page_pdfs(data: bytes):
    """Yield each page of ``data`` as a one-page PDF, opening the source once.

    The per-page form re-parses the document every time (3.0 ms/page measured on
    a 47-page file); this pays that once (0.7 ms) and then only copies. Lazy, so
    a document the caller abandons stops costing at the page it stopped on.
    """
    import io
    try:
        import pypdfium2 as pdfium
    except Exception as e:
        raise OcrError(f"splitting a PDF needs pypdfium2: {e}", status=503)
    try:
        with _PDFIUM_LOCK:
            src = pdfium.PdfDocument(data)
            n = len(src)
    except Exception as e:
        raise OcrError(f"failed to read PDF (pypdfium2): {e}", status=400)
    for i in range(n):
        try:
            with _PDFIUM_LOCK:
                out = pdfium.PdfDocument.new()
                out.import_pages(src, [i])
                buf = io.BytesIO()
                out.save(buf)
            page = buf.getvalue()
        except Exception as e:
            raise OcrError(f"failed to split PDF page {i} (pypdfium2): {e}",
                           status=400)
        yield page


def render_pages(data: bytes, filename: str = "", content_type: str = "",
                 dpi: int = 200, raster_workers: int = 0, chunk: int = 8,
                 policy=None):
    """Yield RGB ``PIL.Image`` pages ONE AT A TIME instead of returning them all.

    Same decoding rules as :func:`load_pages` — this is that function with the list
    turned inside out, so a caller can hold a window of pages rather than the whole
    document. For a 40-page PDF at 250 DPI that is the difference between ~720 MB
    resident for the life of the request and ~18 MB per page in flight.

    A generator, so nothing is rendered until it is asked for and an abandoned
    document stops costing at the page the caller stopped on.
    """
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover - Pillow is a base dep
        raise OcrError(f"Pillow (PIL) is required for OCR: {e}", status=503)

    import io

    name = (filename or "").lower()
    is_pdf = ((content_type or "").lower() == "application/pdf"
              or name.endswith(".pdf") or data[:5] == b"%PDF-")

    if is_pdf:
        # A worker process when there is one (pdfium is safe one document per
        # process); otherwise here, under the lock. The fallback is a correct
        # render, so a pool that cannot start costs throughput, never an outage.
        if raster_workers != 1:
            try:
                from codai.ocr.rasterize import iter_pdf_pages
                pooled = iter_pdf_pages(data, dpi, chunk=chunk,
                                        raster_workers=raster_workers,
                                        policy=policy)
            except Exception:
                pooled = None
            if pooled is not None:
                yield from pooled
                return
        yield from _render_pdf_pages(data, dpi, policy)
        return

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise OcrError(f"could not decode image: {e}", status=400)

    n = getattr(img, "n_frames", 1)
    for i in range(n):
        try:
            img.seek(i)
        except Exception:
            pass
        yield img.convert("RGB")


def _page_or_render(page, dpi: float, scale: float, policy):
    """One page as a PIL image: its own bytes when it may be, else rendered.

    The in-process twin of the rasteriser worker's fast path. Both call the same
    guards (``page_image.extract_embedded``) so the two never drift -- a page
    passed through in one and rendered in the other would make a pool outage
    change what the model is shown, not just how fast it arrives.
    """
    from codai.ocr.workers import page_image
    got = None
    if policy is not None:
        got = policy.extract(page, dpi)
    elif page_image.DEFAULT.passthrough:
        got = page_image.DEFAULT.extract(page, dpi)
    if got is not None:
        import io as _io
        from PIL import Image as _Image
        mime, raw, _size = got
        return page_image.attach(_Image.open(_io.BytesIO(raw)), mime, raw)
    return page.render(scale=scale).to_pil().convert("RGB")


def render_pdf_range(data: bytes, dpi: int, start: int, stop: int,
                     policy=None):
    """Yield in-process RGB pages for ``[start, stop)`` of a PDF, under the lock.

    The fallback for a pooled read whose pool stopped serving midway: re-rendering
    from page zero would redo work the caller already has, and failing a document
    that is half read is the worst of the three options.
    """
    try:
        import pypdfium2 as pdfium
    except Exception:
        pdfium = None
    if pdfium is None:
        # No pypdfium2 at all: the whole-document fallback owns this case.
        for i, img in enumerate(_rasterize_pdf(data, dpi, policy)):
            if start <= i < stop:
                yield img
        return
    try:
        with _PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(data)
            scale = max(0.5, dpi / 72.0)
            n = len(pdf)
    except Exception as e:
        raise OcrError(f"failed to rasterise PDF (pypdfium2): {e}", status=400)
    for i in range(max(0, start), min(n, stop)):
        try:
            with _PDFIUM_LOCK:
                img = _page_or_render(pdf[i], dpi, scale, policy)
        except Exception as e:
            raise OcrError(f"failed to rasterise PDF (pypdfium2): {e}", status=400)
        yield img


def _render_pdf_pages(data: bytes, dpi: int, policy=None):
    """Yield one rasterised RGB page at a time (pypdfium2, else pdf2image)."""
    try:
        import pypdfium2 as pdfium
    except Exception:
        pdfium = None

    if pdfium is not None:
        try:
            with _PDFIUM_LOCK:
                pdf = pdfium.PdfDocument(data)
                scale = max(0.5, dpi / 72.0)
                n = len(pdf)
        except Exception as e:
            raise OcrError(f"failed to rasterise PDF (pypdfium2): {e}", status=400)
        for i in range(n):
            try:
                # The yield is OUTSIDE the lock: the caller does OCR between pages,
                # and holding pdfium while an engine reads a page would serialise
                # the whole subsystem behind one document.
                with _PDFIUM_LOCK:
                    img = _page_or_render(pdf[i], dpi, scale, policy)
            except Exception as e:
                raise OcrError(f"failed to rasterise PDF (pypdfium2): {e}", status=400)
            yield img
        return

    # pdf2image has no per-page API that avoids the whole document, so this path
    # keeps the old behaviour: correctness first, and pypdfium2 is the shipped dep.
    for im in _rasterize_pdf(data, dpi, policy):
        yield im


def _rasterize_pdf(data: bytes, dpi: int, policy=None) -> List["Any"]:
    """Rasterise a PDF to a list of RGB PIL pages (pypdfium2 first, then pdf2image)."""
    from PIL import Image  # noqa: F401  (imported for type parity / already validated)

    # Preferred: pypdfium2 (self-contained, no system poppler).
    try:
        import pypdfium2 as pdfium
    except Exception:
        pdfium = None

    if pdfium is not None:
        try:
            with _PDFIUM_LOCK:
                pdf = pdfium.PdfDocument(data)
                scale = max(0.5, dpi / 72.0)
                pages = []
                for i in range(len(pdf)):
                    pages.append(_page_or_render(pdf[i], dpi, scale, policy))
            return pages
        except Exception as e:
            raise OcrError(f"failed to rasterise PDF (pypdfium2): {e}", status=400)

    # Fallback: pdf2image (needs system poppler).
    try:
        from pdf2image import convert_from_bytes
    except Exception:
        raise OcrError(
            "PDF OCR needs 'pypdfium2' (pip install pypdfium2) or 'pdf2image' + poppler",
            status=503,
        )
    try:
        return [im.convert("RGB") for im in convert_from_bytes(data, dpi=dpi)]
    except Exception as e:
        raise OcrError(f"failed to rasterise PDF (pdf2image): {e}", status=400)


def mean_token_prob(logprobs) -> Optional[float]:
    """Mean ``exp(logprob)`` over a reply's tokens, or None if none were reported.

    The measured confidence a VLM engine can actually offer. It is the same
    computation surya's own client does (``_mean_token_prob`` in
    ``surya/inference/backends/openai_client.py``), so a number from here is
    comparable with the one surya reports rather than a second, differently
    defined "confidence".

    ``None``, never a default, when the server reported nothing. A 0.0 that means
    "not measured" is indistinguishable from a 0.0 that means "the model was
    certain it was unsure", and that confusion already produced one confidently
    wrong verdict in this subsystem (see ``UNREADABLE``). Callers must be able to
    tell the two apart, which is why every engine that uses this also reports
    whether the figure was measured.

    Accepts the OpenAI shape — ``choices[0].logprobs.content`` — as dicts or as
    model objects, since the client library returns either depending on version.
    """
    import math
    if not logprobs:
        return None
    total, n = 0.0, 0
    for entry in logprobs:
        lp = (entry.get("logprob") if isinstance(entry, dict)
              else getattr(entry, "logprob", None))
        if not isinstance(lp, (int, float)):
            continue
        try:
            total += math.exp(float(lp))
        except (OverflowError, ValueError):
            continue
        n += 1
    return (total / n) if n else None


def logprobs_of(result) -> Optional[list]:
    """``choices[0].logprobs.content`` out of a chat-completions reply, or None."""
    try:
        choices = (result or {}).get("choices") or []
        if not choices:
            return None
        lp = (choices[0] or {}).get("logprobs") or {}
        content = lp.get("content")
        return content or None
    except Exception:
        return None


def vlm_serve_limits(cfg) -> dict:
    """The serving limits a VLM OCR engine imposes on its own vLLM instance, as kwargs
    for ``vllm_worker.ensure_service``.

    Without these the engine inherits the vLLM BACKEND's settings, which are sized for
    serving an LLM: ``ctx=18432`` here, and the activation peak vLLM profiles for that
    batch size was 9.4 GB of the 11.1 GB footprint that left surya-ocr-2 unable to start
    on a 24 GB card at all. Chunked prefill is on, so capping the tokens per batch cuts
    the peak without capping the context a page may use. 0/absent = inherit."""
    out = {}
    for key, arg in (("vlm_max_num_batched_tokens", "max_num_batched_tokens"),
                     ("vlm_max_num_seqs", "max_num_seqs"),
                     ("vlm_max_model_len", "max_model_len")):
        try:
            v = int(getattr(cfg, key, 0) or 0)
        except Exception:
            v = 0
        if v > 0:
            out[arg] = v
    return out


#: At most this many individual markers are located in a document-level report.
#: A page of pure noise could otherwise return thousands, and the count already
#: tells the caller how bad it is.
UNREADABLE_MAX_SPOTS = 100

#: Characters of surrounding text kept with each located marker, each side.
UNREADABLE_CONTEXT = 24


def unreadable_report(pages: "List[OcrPage]") -> dict:
    """Count and locate every character the engine could not identify.

    Returns ``{}`` when the document is clean, so the key is simply absent from a
    response that has nothing to report.

    Each located marker carries the confidence of the line it sits in, which is
    the finest granularity available: surya reports one mean token probability per
    block (``PolygonBox.confidence``), not per character, so a per-character score
    would have to be invented. ``conf`` is therefore the line's, and is the signal
    worth acting on — a low-confidence line with an unreadable glyph is a line to
    re-read at a higher DPI, while a high-confidence one is more likely a glyph
    that genuinely is not in the model's repertoire.
    """
    total = 0
    per_page = []
    spots = []
    for page in pages or []:
        n_page = count_unreadable(getattr(page, "text", ""))
        total += n_page
        if n_page:
            per_page.append({"page": page.index, "count": n_page})
        for li, line in enumerate(getattr(page, "lines", []) or []):
            text = getattr(line, "text", "") or ""
            for off in unreadable_offsets(text):
                if len(spots) >= UNREADABLE_MAX_SPOTS:
                    break
                lo = max(0, off - UNREADABLE_CONTEXT)
                spots.append({
                    "page": page.index,
                    "line": li,
                    "offset": off,
                    "conf": round(float(getattr(line, "conf", 0.0) or 0.0), 4),
                    "bbox": getattr(line, "bbox", None),
                    "context": text[lo:off + UNREADABLE_CONTEXT + 1],
                })
    if not total:
        return {}
    report = {"count": total, "pages": per_page, "spots": spots}
    if total > len(spots):
        report["truncated"] = total - len(spots)
    # The byte-level diagnosis, when the engine produced one (surya with
    # ``ocr.surya_token_trace`` on). This is the part that says WHOSE fault the
    # loss is: a marker in the text is equally consistent with the model emitting
    # a broken byte sequence and with our serving stack mis-assembling a correct
    # one, because surya-ocr-2's vocabulary is byte-level and either byte of an
    # accented letter decodes alone to exactly one marker. The tokens tell them
    # apart; the text never can.
    traces = []
    for page in pages or []:
        trace = (getattr(page, "meta", {}) or {}).get("unreadable_trace")
        if trace:
            entry = dict(trace)
            entry["page"] = page.index
            traces.append(entry)
    if traces:
        report["trace"] = traces
        # One line a human can act on without reading the findings: the whole
        # point is to stop guessing whose fault this is.
        if all(t.get("conclusive") for t in traces):
            blamed = ("the model" if any(t.get("findings") for t in traces)
                      else "the serving path")
            report["blame"] = blamed
        else:
            report["blame"] = "unknown"
    return report
