"""How a page becomes bytes: passthrough, encoding, and every guard that says no.

The measurements behind this (A4 at 200 DPI, 3.4-3.9 Mpix): render 21-46 ms,
PNG level 6 86-353 ms, PNG level 1 123 ms, JPEG q95 8.4 ms, passing the PDF's
own JPEG through 0.1 ms. The point of the tests is not the speed -- it is that
every way a passthrough could show the model the WRONG page is refused.
"""

import base64
import io

import pytest
from PIL import Image

from codai.ocr.workers import page_image as pi


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _rgb(w=40, h=60, colour=(220, 200, 180)):
    return Image.new("RGB", (w, h), colour)


def _jpeg_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _scan_pdf(px_w=1268, px_h=1752, pt_w=609.0, pt_h=842.0):
    """A one-page PDF that IS one full-page JPEG, as a scanner writes it."""
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(pt_w, pt_h)
    img = pdfium.PdfImage.new(pdf)
    buf = io.BytesIO(_jpeg_bytes(Image.new("RGB", (px_w, px_h), (240, 230, 210))))
    img.load_jpeg(buf)
    matrix = pdfium.PdfMatrix().scale(pt_w, pt_h)
    img.set_matrix(matrix)
    page.insert_obj(img)
    page.gen_content()
    out = io.BytesIO()
    pdf.save(out)
    return out.getvalue()


# --------------------------------------------------------------------------- #
# the carrier
# --------------------------------------------------------------------------- #

def test_carried_bytes_are_returned_untouched():
    img = _rgb()
    raw = _jpeg_bytes(img)
    pi.attach(img, pi.MIME_JPEG, raw)
    assert pi.carried(img) == (pi.MIME_JPEG, raw)
    assert pi.encode(img) == (pi.MIME_JPEG, raw)


def test_a_transform_drops_the_carrier():
    """The property the whole design rests on: PIL returns a NEW object."""
    img = _rgb()
    pi.attach(img, pi.MIME_JPEG, _jpeg_bytes(img))
    for other in (img.resize((20, 30)), img.convert("RGB"), img.convert("L"),
                  img.copy(), img.rotate(90, expand=True)):
        assert pi.carried(other) is None


def test_an_in_place_size_change_invalidates_the_carrier():
    """A carrier whose size no longer matches is a wrong page, not a fast one."""
    img = _rgb()
    pi.attach(img, pi.MIME_JPEG, _jpeg_bytes(img))
    img._coderai_page_bytes = (pi.MIME_JPEG, b"\xff\xd8\xffstale", (1, 1))
    assert pi.carried(img) is None
    mime, raw = pi.encode(img)
    assert mime == pi.MIME_PNG       # re-encoded rather than sent stale


def test_no_carrier_is_not_an_error():
    assert pi.carried(_rgb()) is None


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #

def test_auto_encodes_lossless_png_when_there_is_nothing_to_pass_through():
    mime, raw = pi.encode(_rgb(), "auto")
    assert mime == pi.MIME_PNG
    assert Image.open(io.BytesIO(raw)).format == "PNG"


def test_png_refuses_to_pass_a_scan_through():
    """`png` exists so a comparison run is byte-identical whatever the PDF held."""
    img = _rgb()
    pi.attach(img, pi.MIME_JPEG, _jpeg_bytes(img))
    mime, raw = pi.encode(img, "png")
    assert mime == pi.MIME_PNG
    assert Image.open(io.BytesIO(raw)).format == "PNG"


def test_jpeg_is_opt_in_and_honours_quality():
    mime, big = pi.encode(_rgb(400, 400), "jpeg", quality=95)
    assert mime == pi.MIME_JPEG
    _, small = pi.encode(_rgb(400, 400), "jpeg", quality=20)
    assert len(small) < len(big)


def test_jpeg_converts_a_mode_it_cannot_hold():
    mime, raw = pi.encode(Image.new("RGBA", (20, 20)), "jpeg")
    assert mime == pi.MIME_JPEG
    assert Image.open(io.BytesIO(raw)).mode == "RGB"


def test_png_is_written_at_level_one():
    """123 ms against 323 ms on a real page, and lossless either way."""
    img = Image.effect_noise((300, 300), 60).convert("RGB")
    assert pi.PNG_LEVEL == 1
    want = io.BytesIO()
    img.save(want, format="PNG", compress_level=1)
    assert pi.encode(img, "auto")[1] == want.getvalue()
    # ...and it really is the same picture.
    assert Image.open(io.BytesIO(pi.encode(img, "auto")[1])).tobytes() \
        == img.tobytes()


def test_an_unknown_format_falls_back_to_auto():
    assert pi.encode(_rgb(), "webp-ish")[0] == pi.MIME_PNG
    assert pi.Policy("nonsense").fmt == "auto"


def test_data_uri_declares_what_it_carries():
    uri = pi.data_uri(pi.MIME_JPEG, b"\xff\xd8\xffxx")
    assert uri.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == b"\xff\xd8\xffxx"


# --------------------------------------------------------------------------- #
# the policy
# --------------------------------------------------------------------------- #

def test_the_default_policy_is_auto_and_passes_through():
    assert pi.DEFAULT.fmt == "auto"
    assert pi.DEFAULT.passthrough is True
    assert pi.Policy("png").passthrough is False


def test_policy_from_a_config_that_predates_these_knobs():
    class _Old:
        dpi = 200
    p = pi.Policy.from_cfg(_Old())
    assert (p.fmt, p.jpeg_quality, p.min_dpi, p.max_dpi_factor) == ("auto", 95, 150.0, 3.0)


def test_policy_from_the_real_config():
    from codai.config import OcrConfig
    cfg = OcrConfig()
    cfg.page_image_format = "jpeg"
    cfg.page_image_jpeg_quality = 80
    p = pi.Policy.from_cfg(cfg)
    assert (p.fmt, p.jpeg_quality) == ("jpeg", 80)


def test_policy_survives_the_worker_protocol():
    p = pi.Policy("jpeg", 70, 120.0, 2.0)
    back = pi.Policy.from_dict(p.as_dict())
    assert (back.fmt, back.jpeg_quality, back.min_dpi,
            back.max_dpi_factor) == ("jpeg", 70, 120.0, 2.0)


def test_policy_clamps_nonsense():
    p = pi.Policy("auto", 9999, -5, 0.1)
    assert p.jpeg_quality == 100 and p.min_dpi == 0.0 and p.max_dpi_factor == 1.0
    p = pi.Policy("auto", "x", "y", "z")
    assert (p.jpeg_quality, p.min_dpi, p.max_dpi_factor) == (95, 150.0, 3.0)


# --------------------------------------------------------------------------- #
# extraction: the guards
# --------------------------------------------------------------------------- #

def test_a_full_page_scan_is_passed_through():
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument(_scan_pdf())
    got = pi.extract_embedded(pdf[0], 200)
    assert got is not None
    mime, raw, size = got
    assert mime == pi.MIME_JPEG
    assert raw[:3] == b"\xff\xd8\xff"
    assert size == (1268, 1752)
    assert Image.open(io.BytesIO(raw)).size == (1268, 1752)


def test_a_scan_coarser_than_the_floor_is_rendered_instead():
    """75 DPI of information upscaled to 200 is not the same as 200 DPI of it."""
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument(_scan_pdf(px_w=634, px_h=876))      # ~75 DPI
    assert pi.extract_embedded(pdf[0], 200) is None


def test_a_scan_far_above_the_target_is_rendered_down():
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument(_scan_pdf(px_w=6760, px_h=9344))     # ~800 DPI
    assert pi.extract_embedded(pdf[0], 200) is None
    # ...but it IS passed through when the operator raises the bound.
    pdf = pdfium.PdfDocument(_scan_pdf(px_w=6760, px_h=9344))
    assert pi.extract_embedded(pdf[0], 200, max_dpi_factor=5.0) is not None
    # And 3x exactly is inside the bound, not outside it.
    pdf = pdfium.PdfDocument(_scan_pdf(px_w=5072, px_h=7008))      # 600 DPI
    assert pi.extract_embedded(pdf[0], 200) is not None


def test_the_floor_tolerates_pdfiums_own_arithmetic():
    """A 150-DPI A4 scan reports 149.957 down, because pdfium measures the
    image's bounding box and not the page. An exact floor rejected it."""
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument(_scan_pdf())
    md = next(iter(pdf[0].get_objects())).get_metadata()
    assert md.vertical_dpi < 150.0
    assert pi.extract_embedded(pdf[0], 200, min_dpi=150.0) is not None


def test_a_page_with_anything_else_on_it_is_rendered():
    """A stamp, a signature or a text layer must not be dropped silently."""
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument(_scan_pdf())
    page = pdf[0]
    path = pdfium.raw.FPDFPageObj_CreateNewRect(10, 10, 50, 50)
    pdfium.raw.FPDFPage_InsertObject(page, path)
    pdfium.raw.FPDFPage_GenerateContent(page)
    assert pi.extract_embedded(page, 200) is None


def test_a_rotated_page_is_rendered():
    """The bytes are a sideways page; the model reads that as gibberish."""
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument(_scan_pdf())
    pdf[0].set_rotation(90)
    assert pi.extract_embedded(pdf[0], 200) is None


def test_a_cropped_image_is_rendered():
    pdfium = pytest.importorskip("pypdfium2")
    data = _scan_pdf()
    pdf = pdfium.PdfDocument(data)
    page = pdf[0]
    obj = next(iter(page.get_objects()))
    obj.set_matrix(pdfium.PdfMatrix().scale(300.0, 400.0))   # covers a corner
    page.gen_content()
    assert pi.extract_embedded(page, 200) is None


def test_a_flate_image_is_rendered():
    """FlateDecode is raw pixels, not a codec the model can be handed."""
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(609.0, 842.0)
    obj = pdfium.PdfImage.new(pdf)
    obj.set_bitmap(pdfium.PdfBitmap.new_native(
        1268, 1752, pdfium.raw.FPDFBitmap_BGR))
    obj.set_matrix(pdfium.PdfMatrix().scale(609.0, 842.0))
    page.insert_obj(obj)
    page.gen_content()
    assert [str(f) for f in obj.get_filters()] != ["DCTDecode"]
    assert pi.extract_embedded(page, 200) is None


def test_extraction_never_raises():
    class _Hostile:
        def get_rotation(self):
            raise RuntimeError("pdfium said no")
    assert pi.extract_embedded(_Hostile(), 200) is None


def test_the_png_policy_never_extracts():
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument(_scan_pdf())
    assert pi.Policy("png").extract(pdf[0], 200) is None
    assert pi.Policy("auto").extract(pdf[0], 200) is not None


# --------------------------------------------------------------------------- #
# the whole way through: render paths and the engine pipe
# --------------------------------------------------------------------------- #

def test_render_pages_passes_a_scan_through_in_process():
    from codai.ocr.base import render_pages
    pages = list(render_pages(_scan_pdf(), filename="s.pdf", dpi=200,
                              raster_workers=1))
    assert len(pages) == 1
    got = pi.carried(pages[0])
    assert got is not None and got[0] == pi.MIME_JPEG
    assert pages[0].size == (1268, 1752)     # native, NOT dpi-derived


def test_render_pages_renders_a_scan_the_policy_refuses():
    from codai.ocr.base import render_pages
    pages = list(render_pages(_scan_pdf(), filename="s.pdf", dpi=200,
                              raster_workers=1, policy=pi.Policy("png")))
    assert pi.carried(pages[0]) is None
    assert pages[0].size == (1692, 2339)     # 609x842 pt at 200 DPI


def test_the_rasteriser_pool_carries_the_bytes_across_the_pipe():
    """The pooled path and the in-process path must agree, or a pool outage
    would change what the model is shown rather than how fast it arrives."""
    from codai.ocr import rasterize
    rasterize.reset_pool()
    try:
        gen = rasterize.iter_pdf_pages(_scan_pdf(), 200, chunk=4, raster_workers=2)
        if gen is None:
            pytest.skip("no rasteriser pool in this environment")
        pages = list(gen)
    finally:
        rasterize.reset_pool()
    assert len(pages) == 1
    got = pi.carried(pages[0])
    assert got is not None and got[0] == pi.MIME_JPEG
    assert pages[0].size == (1268, 1752)
    # And the pixels are still readable through the lazy open.
    pages[0].load()


def test_the_engine_pipe_sends_the_page_once_encoded():
    """`recognize_image` used to `convert("RGB").save(PNG)` every page, which
    both re-encoded a scan and dropped its bytes for surya to encode again."""
    from codai.ocr import subprocess_engine

    sent = {}

    class _Engine(subprocess_engine.SubprocessOcrEngine):
        def __init__(self):
            self.cfg = type("C", (), {"page_image_format": "auto"})()
            self._mux = False
            self.name = "fake"

        def _rpc(self, msg):
            sent.update(msg)
            return {"ok": True, "page": {}}

        def _page_from_dict(self, d):
            return d

        class _Lock:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        _lock = _Lock()

    img = _rgb(100, 100)
    raw = _jpeg_bytes(img)
    pi.attach(img, pi.MIME_JPEG, raw)
    _Engine().recognize_image(img)
    assert sent["mime"] == pi.MIME_JPEG
    assert base64.b64decode(sent["image"]) == raw


def test_the_engine_pipe_encodes_a_rendered_page_as_png():
    from codai.ocr import subprocess_engine

    sent = {}

    class _Engine(subprocess_engine.SubprocessOcrEngine):
        def __init__(self):
            self.cfg = type("C", (), {})()
            self._mux = False
            self.name = "fake"

        def _rpc(self, msg):
            sent.update(msg)
            return {"ok": True, "page": {}}

        def _page_from_dict(self, d):
            return d

        class _Lock:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        _lock = _Lock()

    _Engine().recognize_image(_rgb(100, 100))
    assert sent["mime"] == pi.MIME_PNG
    assert Image.open(io.BytesIO(base64.b64decode(sent["image"]))).format == "PNG"


def test_chandra_sends_a_scan_as_it_arrived():
    from codai.ocr.chandra import prepare_image, scale_to_fit
    # A size chandra's own block grid leaves alone: anything it has to round
    # (1200x1600 becomes 1204x1596) is a new object and loses the bytes, which
    # is the next test.
    assert scale_to_fit(_rgb(1204, 1596)).size == (1204, 1596)
    img = _rgb(1204, 1596)
    raw = _jpeg_bytes(img)
    pi.attach(img, pi.MIME_JPEG, raw)
    uri, size = prepare_image(img)
    assert uri.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == raw
    assert size == (1204, 1596)


def test_chandra_reports_the_size_it_actually_sent():
    """A page chandra scales is a new object, so the carrier goes and the size
    it returns is the scaled one -- the boxes are normalised against it."""
    from codai.ocr.chandra import prepare_image, scale_to_fit
    img = _rgb(1201, 1601)
    pi.attach(img, pi.MIME_JPEG, _jpeg_bytes(img))
    uri, size = prepare_image(img)
    assert size == scale_to_fit(_rgb(1201, 1601)).size
    if size != (1201, 1601):
        assert uri.startswith("data:image/png;base64,")


def test_olmocr_sends_a_scan_as_it_arrived_but_not_a_scaled_one():
    from codai.ocr.olmocr import prepare_image, DEFAULT_LONGEST_SIDE
    small = _rgb(400, 300)
    raw = _jpeg_bytes(small)
    pi.attach(small, pi.MIME_JPEG, raw)
    assert base64.b64decode(prepare_image(small).split(",", 1)[1]) == raw

    big = _rgb(4000, 2000)
    pi.attach(big, pi.MIME_JPEG, _jpeg_bytes(big))
    uri = prepare_image(big)
    assert uri.startswith("data:image/png;base64,")
    out = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert max(out.size) == DEFAULT_LONGEST_SIDE


def test_surya_tells_its_worker_how_to_encode():
    from codai.ocr.surya import SuryaEngine
    from codai.config import OcrConfig
    cfg = OcrConfig()
    cfg.page_image_format = "jpeg"
    cfg.page_image_jpeg_quality = 70
    opts = SuryaEngine(cfg)._worker_opts()
    assert opts["page_image_format"] == "jpeg"
    assert opts["page_image_jpeg_quality"] == 70


# --------------------------------------------------------------------------- #
# surya's own client, which PNG'd every page a second time
# --------------------------------------------------------------------------- #

def _fake_surya(monkeypatch):
    """Stand in for surya.inference.backends.openai_client.

    The real one is in an isolated venv that this interpreter does not have, so
    the thing under test is the patch, against the shape of the module it
    patches: ``encode_image_b64`` and ``_build_messages``, both module-level,
    with the mime type hard-coded into the data URI.
    """
    import sys
    import types
    oc = types.ModuleType("surya.inference.backends.openai_client")

    def encode_image_b64(image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _build_messages(image, prompt):
        return [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{encode_image_b64(image)}"}},
            {"type": "text", "text": prompt}]}]

    oc.encode_image_b64 = encode_image_b64
    oc._build_messages = _build_messages
    for name, mod in (("surya", types.ModuleType("surya")),
                      ("surya.inference", types.ModuleType("surya.inference")),
                      ("surya.inference.backends",
                       types.ModuleType("surya.inference.backends")),
                      ("surya.inference.backends.openai_client", oc)):
        monkeypatch.setitem(sys.modules, name, mod)
    return oc


def _worker_module():
    from codai.ocr.workers import ocr_worker
    return ocr_worker


def test_surya_sends_a_scan_without_encoding_it_again(monkeypatch):
    oc = _fake_surya(monkeypatch)
    w = _worker_module()
    monkeypatch.setattr(w, "_IMAGE_FORMAT", ["auto", 95])
    img = _rgb(1200, 1600)
    raw = _jpeg_bytes(img)
    w._carry(img, pi.MIME_JPEG, raw)
    w._install_image_encoder()
    try:
        assert base64.b64decode(oc.encode_image_b64(img)) == raw
        msg = oc._build_messages(img, "read it")
        uri = msg[0]["content"][0]["image_url"]["url"]
        # The mime in that URI was a literal "image/png". A JPEG announced as a
        # PNG works until a server believes the label.
        assert uri.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(uri.split(",", 1)[1]) == raw
    finally:
        oc._coderai_image_patched = False


def test_surya_encodes_a_rendered_page_at_level_one(monkeypatch):
    oc = _fake_surya(monkeypatch)
    w = _worker_module()
    monkeypatch.setattr(w, "_IMAGE_FORMAT", ["auto", 95])
    w._install_image_encoder()
    try:
        img = Image.effect_noise((200, 200), 40).convert("RGB")
        got = base64.b64decode(oc.encode_image_b64(img))
        want = io.BytesIO()
        img.save(want, format="PNG", compress_level=1)
        assert got == want.getvalue()
        uri = oc._build_messages(img, "x")[0]["content"][0]["image_url"]["url"]
        assert uri.startswith("data:image/png;base64,")
    finally:
        oc._coderai_image_patched = False


def test_surya_honours_the_jpeg_setting(monkeypatch):
    oc = _fake_surya(monkeypatch)
    w = _worker_module()
    monkeypatch.setattr(w, "_IMAGE_FORMAT", ["jpeg", 60])
    w._install_image_encoder()
    try:
        uri = oc._build_messages(_rgb(200, 200), "x")[0]["content"][0]["image_url"]["url"]
        assert uri.startswith("data:image/jpeg;base64,")
    finally:
        oc._coderai_image_patched = False


def test_surya_png_setting_refuses_a_scans_own_bytes(monkeypatch):
    oc = _fake_surya(monkeypatch)
    w = _worker_module()
    monkeypatch.setattr(w, "_IMAGE_FORMAT", ["png", 95])
    w._install_image_encoder()
    try:
        img = _rgb(200, 200)
        w._carry(img, pi.MIME_JPEG, _jpeg_bytes(img))
        uri = oc._build_messages(img, "x")[0]["content"][0]["image_url"]["url"]
        assert uri.startswith("data:image/png;base64,")
    finally:
        oc._coderai_image_patched = False


def test_a_surya_release_that_renamed_the_encoder_still_reads_pages(monkeypatch):
    """Silent by design: throughput may regress, pages may not fail."""
    import sys
    import types
    oc = types.ModuleType("surya.inference.backends.openai_client")   # no encoder
    for name, mod in (("surya", types.ModuleType("surya")),
                      ("surya.inference", types.ModuleType("surya.inference")),
                      ("surya.inference.backends",
                       types.ModuleType("surya.inference.backends")),
                      ("surya.inference.backends.openai_client", oc)):
        monkeypatch.setitem(sys.modules, name, mod)
    _worker_module()._install_image_encoder()      # must not raise
    assert not hasattr(oc, "encode_image_b64")


def test_the_worker_rejects_a_stale_carrier_too(monkeypatch):
    """The worker's copy of the carrier convention has to fail closed the same
    way page_image's does -- it is the same bug in two processes."""
    w = _worker_module()
    img = _rgb(100, 100)
    w._carry(img, pi.MIME_JPEG, _jpeg_bytes(img))
    assert w._carried(img) is not None
    img._coderai_page_bytes = (pi.MIME_JPEG, b"\xff\xd8\xffstale", (7, 7))
    assert w._carried(img) is None
    assert w._encode_page(img)[0] == pi.MIME_PNG


def test_the_worker_and_the_module_agree_on_the_attribute():
    """Two copies of one convention, in two processes, by necessity: the engine
    venv cannot import coderai. If they ever disagree, a passed-through page is
    silently re-encoded and nobody notices except the clock."""
    w = _worker_module()
    assert w._PAGE_BYTES == pi._CARRIER
    img = _rgb()
    raw = _jpeg_bytes(img)
    pi.attach(img, pi.MIME_JPEG, raw)
    assert w._carried(img) == (pi.MIME_JPEG, raw)
    w._carry(img, pi.MIME_JPEG, raw)
    assert pi.carried(img) == (pi.MIME_JPEG, raw)
