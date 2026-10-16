#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.
"""How a page becomes the bytes an OCR engine is given.

Measured on the orchestrator, A4 at 200 DPI (3.4-3.9 Mpix), best of three:

    pdfium page.render()              21-46 ms
    to_pil().convert("RGB")            6-8 ms
    PNG encode, PIL default (level 6) 86-353 ms     1846 KB
    PNG encode, level 1 (lossless)       123 ms     2117 KB
    JPEG q95                             8.4 ms      596 KB
    extract the page's embedded JPEG      0.1 ms      147 KB

So "rasterisation" is mostly not rasterisation: three quarters to nine tenths of
it is zlib. And on the surya path it was paid TWICE per page -- once to hand the
page to the isolated-venv worker over the pipe, once inside surya's own client
to build the data URI -- which is about 0.8 s of CPU per page against roughly
5 s on the GPU. That is the difference between a pod's cores keeping 48 pages in
flight and becoming the ceiling themselves.

Two ways out, and this module is both:

* **Do not encode what is already encoded.** A scanned page is usually one
  full-page DCTDecode (JPEG) image object, and its stream can be handed to the
  model exactly as the scanner wrote it: no render, no encode, no resample. The
  guards in :func:`extract_embedded` are the whole difficulty -- see there.

* **When we must encode, do not pay for level 6.** zlib level 1 is lossless and
  2.6x faster; the extra ~15 % on the wire is not the scarce thing. JPEG is
  38x faster again but lossy, so it stays opt-in.

SELF-CONTAINED on purpose: stdlib + PIL, and pypdfium2 only inside
:func:`extract_embedded`. The rasteriser worker and this module's callers in the
main process both need it, and the rasteriser must not ``import codai.ocr``
(377 MB via litellm). ``codai/ocr/workers/ocr_worker.py`` may not import even
this -- it runs in the engine's own venv -- so the carrier it needs travels as a
protocol field, not as shared code.
"""

import io

MIME_PNG = "image/png"
MIME_JPEG = "image/jpeg"

#: ``ocr.page_image_format``.
FORMATS = ("auto", "png", "jpeg")

#: zlib level for the lossless path. 1 rather than PIL's 6: 123 ms against
#: 323 ms on the same page, for 15 % more bytes.
PNG_LEVEL = 1

#: Attribute carrying "these exact bytes already are this image".
_CARRIER = "_coderai_page_bytes"


# --------------------------------------------------------------------------- #
# the carrier
# --------------------------------------------------------------------------- #

def attach(img, mime: str, raw: bytes):
    """Record that ``raw`` (of type ``mime``) already encodes ``img``.

    Stored on the image object, with the size it was true for. Any PIL transform
    -- ``resize``, ``convert``, even ``convert`` to the mode it already has --
    returns a NEW object, which carries nothing, so a page that is scaled or
    recoloured on its way to the model silently stops being passed through
    instead of silently being sent at the wrong size. Failing closed is the
    whole design: the cost of a missed passthrough is milliseconds, the cost of
    a wrong one is a page read at the wrong resolution or the wrong rotation.
    """
    try:
        setattr(img, _CARRIER, (mime, raw, tuple(img.size)))
    except Exception:
        pass            # a PIL build that forbids attributes: encode as usual
    return img


def carried(img):
    """``(mime, raw)`` if this image is already encoded, else ``None``."""
    got = getattr(img, _CARRIER, None)
    if not got:
        return None
    try:
        mime, raw, size = got
    except Exception:
        return None
    if not raw or tuple(img.size) != tuple(size):
        return None     # transformed in place; the bytes are no longer the image
    return mime, raw


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #

def encode(img, fmt: str = "auto", quality: int = 95):
    """``(mime, bytes)`` for one page, by policy.

    ``auto``  pass the carried bytes through when there are any, else PNG level 1.
    ``png``   always re-encode as PNG. Mostly for a controlled comparison: it is
              the only setting under which two runs of the same document produce
              byte-identical requests regardless of what the PDF held.
    ``jpeg``  lossy, opt-in. 8 ms and a third of the bytes; whether it costs
              accuracy is a question about a corpus, not about a codec, so it is
              not a default.
    """
    fmt = (fmt or "auto").strip().lower()
    if fmt not in FORMATS:
        fmt = "auto"
    if fmt != "png":
        got = carried(img)
        if got is not None:
            return got
    buf = io.BytesIO()
    if fmt == "jpeg":
        q = 95
        try:
            q = max(1, min(100, int(quality)))
        except Exception:
            pass
        # JPEG cannot hold RGBA or P; a page that is not already RGB/L is
        # converted rather than refused.
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=q)
        return MIME_JPEG, buf.getvalue()
    img.save(buf, format="PNG", compress_level=PNG_LEVEL)
    return MIME_PNG, buf.getvalue()


def data_uri(mime: str, raw: bytes) -> str:
    import base64
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


# --------------------------------------------------------------------------- #
# passing the scan through
# --------------------------------------------------------------------------- #

#: pdfium's object types. Hard-coded rather than imported from
#: ``pypdfium2.raw`` so this module stays importable without the binding.
_OBJ_IMAGE = 3


def extract_embedded(page, dpi: int, min_dpi: float = 0.0,
                     max_dpi_factor: float = 3.0):
    """``(mime, raw, (w, h))`` if this page IS one embedded JPEG, else ``None``.

    Returning ``None`` is always safe -- the caller renders, which is what it
    did before this existed. Every guard below is a way a passed-through stream
    would have been a WRONG page rather than a slower one, and each was checked
    against a real document:

    * **One image and nothing else.** A single object of type IMAGE. A scan with
      a digital stamp, a signature annotation or an OCR text layer has other
      objects, and shipping only the image would silently drop them.
    * **DCTDecode alone.** That filter *is* a JPEG and needs no transcoding. The
      other common scan filters do: FlateDecode is raw pixels (a test PDF here
      returns a ``789c`` zlib stream PIL cannot open at all), CCITTFaxDecode and
      JBIG2Decode are bitonal fax, JPXDecode is JPEG 2000.
    * **Grey or RGB, 8 or 24 bpp.** A CMYK or YCCK JPEG out of a PDF decodes
      inverted in most readers unless the Adobe marker is honoured, and the
      model is not the place to find that out.
    * **Axis-aligned and unrotated.** The page's ``/Rotate`` must be 0 and the
      image matrix must be a positive scale with no rotation or skew; otherwise
      the bytes are a sideways page, which reads as gibberish rather than as an
      error.
    * **It covers the page.** 98 % in both axes, anchored at the origin, or the
      model is being shown a crop.
    * **At least the TARGET DPI, and at most ``max_dpi_factor`` x it.** This
      guard was wrong when it shipped and the error is worth recording: it used
      an absolute floor (150 DPI) instead of the DPI the page would have been
      rendered at. A 150-DPI scan then went to a 250-DPI engine at its native
      size -- a 40 % linear resolution drop in what the model sees -- and
      surya-ocr-2 lost Italian accents on exactly those pages, which is a
      regression reported from production within hours of the change.

      The justification for passing a scan through was never "fewer pixels", it
      was "the same pixels, without the work". A coarser image is not a saving,
      it is a different page. So the floor is the target itself, and
      ``min_dpi`` only RAISES it; above ``max_dpi_factor`` x the target,
      rendering down is cheaper than shipping the whole thing for the engine to
      scale.
    """
    try:
        if int(page.get_rotation() or 0) % 360 != 0:
            return None
        objs = list(page.get_objects())
        if len(objs) != 1 or getattr(objs[0], "type", None) != _OBJ_IMAGE:
            return None
        obj = objs[0]
        if [str(f) for f in obj.get_filters()] != ["DCTDecode"]:
            return None
        md = obj.get_metadata()
        if int(md.bits_per_pixel) not in (8, 24):
            return None
        if int(md.colorspace) not in (1, 2):     # DEVICEGRAY, DEVICERGB
            return None
        m = obj.get_matrix()
        if abs(m.b) > 1e-6 or abs(m.c) > 1e-6 or m.a <= 0 or m.d <= 0:
            return None
        pw, ph = page.get_size()
        if pw <= 0 or ph <= 0:
            return None
        left, bottom, right, top = obj.get_bounds()
        if (right - left) / pw < 0.98 or (top - bottom) / ph < 0.98:
            return None
        if abs(left) / pw > 0.02 or abs(bottom) / ph > 0.02:
            return None
        # The floor is what we WOULD have rendered; min_dpi only raises it.
        # 0 (the default) therefore means "as detailed as the target", which is
        # the only setting under which passing a scan through cannot change what
        # the model sees.
        floor = float(dpi)
        if min_dpi and float(min_dpi) > floor:
            floor = float(min_dpi)
        # 2 % of slack, because pdfium derives the effective DPI from the
        # image's own bounding box rather than the page: a plain 150-DPI scan of
        # an A4 page reports 150.0 across and 149.957 down, and an exact floor
        # would reject it over four hundredths of a dot.
        lo = floor * 0.98
        hi = float(dpi) * float(max_dpi_factor)
        for eff in (float(md.horizontal_dpi), float(md.vertical_dpi)):
            if not (lo <= eff <= hi):
                return None
        raw = bytes(obj.get_data(decode_simple=False))
        if raw[:3] != b"\xff\xd8\xff":
            return None              # filter said JPEG, bytes disagree
        return MIME_JPEG, raw, (int(md.width), int(md.height))
    except Exception:
        return None                  # any pdfium surprise = render it


# --------------------------------------------------------------------------- #
# the policy, as one value
# --------------------------------------------------------------------------- #

class Policy:
    """``ocr.page_image_*`` as one object, so adding a knob is not a new argument
    on six functions and a protocol field on two workers.

    The default IS the shipped behaviour: pass a scan through, otherwise encode
    losslessly and cheaply. ``Policy()`` is therefore what every caller that has
    no config in reach should use, and no call site needs to know the defaults.
    """

    __slots__ = ("fmt", "jpeg_quality", "min_dpi", "max_dpi_factor")

    def __init__(self, fmt: str = "auto", jpeg_quality: int = 95,
                 min_dpi: float = 0.0, max_dpi_factor: float = 3.0):
        fmt = (fmt or "auto").strip().lower()
        self.fmt = fmt if fmt in FORMATS else "auto"
        try:
            self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        except Exception:
            self.jpeg_quality = 95
        try:
            self.min_dpi = max(0.0, float(min_dpi))
        except Exception:
            self.min_dpi = 0.0
        try:
            self.max_dpi_factor = max(1.0, float(max_dpi_factor))
        except Exception:
            self.max_dpi_factor = 3.0

    # -- predicates ------------------------------------------------------- #
    @property
    def passthrough(self) -> bool:
        """Whether an already-encoded page may be sent as it is."""
        return self.fmt != "png"

    # -- adapters --------------------------------------------------------- #
    @classmethod
    def from_cfg(cls, cfg):
        """From an ``OcrConfig``-shaped object, tolerating one that predates it."""
        g = lambda k, d: getattr(cfg, k, d) if cfg is not None else d  # noqa: E731
        return cls(g("page_image_format", "auto"),
                   g("page_image_jpeg_quality", 95),
                   g("page_image_min_dpi", 0.0),
                   g("page_image_max_dpi_factor", 3.0))

    def as_dict(self) -> dict:
        """For the rasteriser worker's ``open`` frame."""
        return {"fmt": self.fmt, "jpeg_quality": self.jpeg_quality,
                "min_dpi": self.min_dpi, "max_dpi_factor": self.max_dpi_factor}

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(d.get("fmt", "auto"), d.get("jpeg_quality", 95),
                   d.get("min_dpi", 0.0), d.get("max_dpi_factor", 3.0))

    def encode(self, img):
        """``(mime, bytes)`` under this policy."""
        return encode(img, self.fmt, self.jpeg_quality)

    def extract(self, page, dpi):
        """The page's own bytes, or None when they may not be used."""
        if not self.passthrough:
            return None
        return extract_embedded(page, dpi, self.min_dpi, self.max_dpi_factor)

    def __repr__(self):
        return (f"Policy(fmt={self.fmt!r}, jpeg_quality={self.jpeg_quality}, "
                f"min_dpi={self.min_dpi}, max_dpi_factor={self.max_dpi_factor})")


#: The default, as a shared value for call sites that have no config.
DEFAULT = Policy()
