"""A caller can ask for a scan at the resolution the scanner wrote it.

A scanned page that is one embedded JPEG is handed to the model as-is: 0.1 ms
and 147 KB, against a render plus a PNG encode at 21-46 ms and ~1.8 MB. But the
passthrough only applied while the page's own resolution sat between the target
DPI and ``ocr.page_image_max_dpi_factor`` (3.0) times it. A 600-DPI scan read at
a 200-DPI target squeaks in; a 1200-DPI one was rendered DOWN to 200, throwing
away six times the linear detail the file actually held — and the only way to
change that was an install-wide config value, which is no use to a caller who
knows that *this* document is a high-resolution scan.

``native_dpi=true`` on the request removes the ceiling. The FLOOR stays, and
that is not an oversight: passing a page through below the target DPI shows the
model a coarser image than it would have rendered, which is a different page
rather than a faster one. That exact mistake shipped once — an absolute 150-DPI
floor instead of the target — and surya-ocr-2 lost Italian accents on every
150-DPI scan sent to a 250-DPI engine, reported from production within hours.
"""
import json

import pytest

from codai.api import ocr as ocr_api
from codai.ocr.workers import page_image


# --------------------------------------------------------------------------- #
#  a stand-in for one pdfium page holding a single scanned JPEG
# --------------------------------------------------------------------------- #

class _Meta:
    def __init__(self, dpi, w, h):
        self.bits_per_pixel = 24
        self.colorspace = 2          # DEVICERGB
        self.width = w
        self.height = h
        self.horizontal_dpi = float(dpi)
        self.vertical_dpi = float(dpi)


class _Matrix:
    a, b, c, d = 1.0, 0.0, 0.0, 1.0


class _ImageObj:
    type = 3                         # pdfium's IMAGE

    def __init__(self, dpi, page_pt):
        self._dpi = dpi
        self._pw, self._ph = page_pt
        w = int(self._pw / 72.0 * dpi)
        h = int(self._ph / 72.0 * dpi)
        self._md = _Meta(dpi, w, h)

    def get_filters(self):
        return ["DCTDecode"]

    def get_metadata(self):
        return self._md

    def get_matrix(self):
        return _Matrix()

    def get_bounds(self):
        return (0.0, 0.0, self._pw, self._ph)

    def get_data(self, decode_simple=False):
        return b"\xff\xd8\xff" + b"scan bytes"


class _Page:
    """An A4 page whose only content is one scanned JPEG at `dpi`."""

    def __init__(self, dpi):
        self._size = (595.0, 842.0)      # A4 in points
        self._obj = _ImageObj(dpi, self._size)

    def get_rotation(self):
        return 0

    def get_objects(self):
        return [self._obj]

    def get_size(self):
        return self._size


def _extract(scan_dpi, target_dpi, **policy_kw):
    pol = page_image.Policy(**policy_kw) if policy_kw else page_image.Policy()
    return pol.extract(_Page(scan_dpi), target_dpi)


# --------------------------------------------------------------------------- #
#  the stand-in is faithful: the default behaviour is reproduced
# --------------------------------------------------------------------------- #

def test_a_scan_at_the_target_resolution_is_passed_through():
    got = _extract(200, 200)
    assert got is not None
    mime, raw, size = got
    assert mime == page_image.MIME_JPEG
    assert raw.startswith(b"\xff\xd8\xff")


def test_a_scan_coarser_than_the_target_is_never_passed_through():
    """The accents regression. 150 DPI into a 250-DPI engine is a 40 % linear
    resolution drop in what the model sees."""
    assert _extract(150, 250) is None


def test_a_scan_far_finer_than_the_target_is_rendered_down_by_default():
    """1200 DPI against a 200-DPI target is 35 megapixels; the default ceiling
    says render 3 rather than ship 35 for the engine to scale."""
    assert _extract(1200, 200) is None


def test_the_default_ceiling_is_three_times_the_target():
    assert _extract(600, 200) is not None      # exactly 3x: still in
    assert _extract(601, 200) is None


# --------------------------------------------------------------------------- #
#  native_dpi removes the ceiling and only the ceiling
# --------------------------------------------------------------------------- #

def test_native_dpi_passes_a_high_resolution_scan_through_untouched():
    pol = page_image.Policy.from_cfg(None, {"native_dpi": True})
    got = pol.extract(_Page(1200), 200)
    assert got is not None, "the point of the flag"
    _mime, _raw, (w, h) = got
    assert w > 9000 and h > 13000, "at its own resolution, not the target's"


def test_native_dpi_does_not_lower_the_floor():
    """Removing the ceiling must not quietly reintroduce the coarse-page bug."""
    pol = page_image.Policy.from_cfg(None, {"native_dpi": True})
    assert pol.extract(_Page(150), 250) is None


def test_native_dpi_is_an_infinite_ceiling_not_a_large_one():
    pol = page_image.Policy.from_cfg(None, {"native_dpi": True})
    assert pol.max_dpi_factor == float("inf")


def test_native_dpi_overrides_a_configured_force_png():
    """`page_image_format: png` means "always re-encode", which is the one
    setting that makes passthrough impossible. Asking for the original bytes
    and being given a re-encode would be an answer to a different question."""
    class Cfg:
        page_image_format = "png"
    pol = page_image.Policy.from_cfg(Cfg(), {"native_dpi": True})
    assert pol.passthrough is True
    assert pol.extract(_Page(1200), 200) is not None


def test_config_still_decides_when_the_request_says_nothing():
    class Cfg:
        page_image_format = "jpeg"
        page_image_jpeg_quality = 80
        page_image_min_dpi = 120.0
        page_image_max_dpi_factor = 2.0
    pol = page_image.Policy.from_cfg(Cfg(), {})
    assert (pol.fmt, pol.jpeg_quality, pol.min_dpi, pol.max_dpi_factor) == \
        ("jpeg", 80, 120.0, 2.0)


def test_each_field_can_be_overridden_per_request():
    class Cfg:
        page_image_format = "auto"
        page_image_jpeg_quality = 95
        page_image_min_dpi = 0.0
        page_image_max_dpi_factor = 3.0
    pol = page_image.Policy.from_cfg(Cfg(), {
        "page_image_format": "jpeg", "page_image_jpeg_quality": 70,
        "page_image_min_dpi": 300.0, "page_image_max_dpi_factor": 8.0})
    assert (pol.fmt, pol.jpeg_quality, pol.min_dpi, pol.max_dpi_factor) == \
        ("jpeg", 70, 300.0, 8.0)


def test_an_unbounded_policy_survives_the_rasteriser_frame():
    """The policy reaches the rasteriser worker as JSON in its `open` frame, so
    an infinite ceiling has to round-trip rather than become 3.0 again."""
    pol = page_image.Policy.from_cfg(None, {"native_dpi": True})
    back = page_image.Policy.from_dict(json.loads(json.dumps(pol.as_dict())))
    assert back.max_dpi_factor == float("inf")
    assert back.extract(_Page(1200), 200) is not None


# --------------------------------------------------------------------------- #
#  the request field
# --------------------------------------------------------------------------- #

def test_the_request_field_reaches_the_page_options():
    assert ocr_api._page_opts(None, None, True) == {"native_dpi": True}


def test_saying_nothing_sets_nothing():
    """Omitted must mean "use the configured value", not "false"."""
    assert ocr_api._page_opts(None, None) == {}


def test_native_dpi_false_is_an_explicit_choice_not_a_silence():
    assert ocr_api._page_opts(None, None, False) == {"native_dpi": False}


def test_a_bad_page_image_format_is_refused_rather_than_ignored():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        ocr_api._page_opts(None, None, None, "webp")
    assert e.value.status_code == 400


def test_a_ceiling_below_one_is_refused():
    """Below 1.0 the ceiling would sit under the floor: a page passed through
    at a LOWER resolution than it would have been rendered at."""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        ocr_api._page_opts(None, None, None, None, 0.5)
    assert e.value.status_code == 400


def test_both_endpoints_take_the_field():
    """A batch is how a caller sends real work; a flag the batch endpoint does
    not accept is a flag that never applied to anything that mattered, and
    FastAPI drops an unknown form field without a word."""
    import inspect
    for fn in (ocr_api.create_ocr, ocr_api.create_ocr_batch):
        params = inspect.signature(fn).parameters
        assert "native_dpi" in params, fn.__name__
        assert "page_image_format" in params, fn.__name__
        assert "max_dpi_factor" in params, fn.__name__


# --------------------------------------------------------------------------- #
#  and it has to survive the trip to a pod
# --------------------------------------------------------------------------- #

def test_the_flag_is_forwarded_to_a_pod_that_renders_the_page_itself():
    """On the pooled path the page travels as a one-page PDF and the POD
    rasterises it. A flag that is not forwarded works locally and is silently
    ignored the moment a pool is in use."""
    from codai.ocr.manager import _page_post_fields
    fields, _h = _page_post_fields("k", 200, {"native_dpi": True}, None, "surya")
    assert fields.get("native_dpi") == "true"


def test_the_pod_is_not_told_about_a_flag_nobody_set():
    from codai.ocr.manager import _page_post_fields
    fields, _h = _page_post_fields("k", 200, {}, None, "surya")
    assert "native_dpi" not in fields
    assert "page_image_format" not in fields
    assert "max_dpi_factor" not in fields


def test_the_other_page_image_overrides_are_forwarded_too():
    from codai.ocr.manager import _page_post_fields
    fields, _h = _page_post_fields(
        "k", 200, {"page_image_format": "jpeg",
                   "page_image_max_dpi_factor": 8.0}, None, "chandra")
    assert fields["page_image_format"] == "jpeg"
    assert float(fields["max_dpi_factor"]) == 8.0
    assert fields["engine"] == "chandra"


def test_the_engines_read_the_requests_policy_not_only_the_installs():
    """Three engines build the policy when they encode a page for the model.
    Each had the install's config hardcoded, so a per-request override would
    have stopped at the rasteriser."""
    for path in ("codai/ocr/subprocess_engine.py", "codai/ocr/chandra.py",
                 "codai/ocr/olmocr.py"):
        src = open(path).read()
        assert "Policy.from_cfg(self.cfg, page_options())" in src.replace("\n", " ").replace("  ", " ") \
            or "page_options()" in src, path
        assert "Policy.from_cfg(self.cfg)" not in src, \
            f"{path} still ignores the request"
