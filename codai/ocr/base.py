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

Every OCR engine (PaddleOCR, docTR, Surya) subclasses :class:`OcrEngine` and returns
:class:`OcrPage` objects. The rest of the subsystem (manager, API, detection,
extraction) speaks only these types, so engines stay interchangeable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
# Result types
# ---------------------------------------------------------------------------

@dataclass
class OcrLine:
    """A recognised text line (or word), with its polygon bounding box."""
    text: str
    bbox: List[float]              # [x0, y0, x1, y1] in pixels
    conf: float = 0.0

    def to_dict(self) -> dict:
        return {"text": self.text, "bbox": self.bbox, "conf": round(self.conf, 4)}


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

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "width": self.width,
            "height": self.height,
            "lines": [l.to_dict() for l in self.lines],
            "regions": [r.to_dict() for r in self.regions],
            "tables": self.tables,
        }


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


# ---------------------------------------------------------------------------
# Input loading (images + PDF) → list of PIL pages
# ---------------------------------------------------------------------------

# Extensions we accept directly as raster images.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".ppm"}


def load_pages(data: bytes, filename: str = "", content_type: str = "", dpi: int = 200) -> List["Any"]:
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
        return _rasterize_pdf(data, dpi)

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


def _rasterize_pdf(data: bytes, dpi: int) -> List["Any"]:
    """Rasterise a PDF to a list of RGB PIL pages (pypdfium2 first, then pdf2image)."""
    from PIL import Image  # noqa: F401  (imported for type parity / already validated)

    # Preferred: pypdfium2 (self-contained, no system poppler).
    try:
        import pypdfium2 as pdfium
    except Exception:
        pdfium = None

    if pdfium is not None:
        try:
            pdf = pdfium.PdfDocument(data)
            scale = max(0.5, dpi / 72.0)
            pages = []
            for i in range(len(pdf)):
                page = pdf[i]
                bitmap = page.render(scale=scale)
                pages.append(bitmap.to_pil().convert("RGB"))
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
