"""A model's standing prompt prefix, and how big one OCR upload may be.

Two unrelated asks, both about a request being taken as meant.

**Prompt prefix.** A model fine-tuned on a quality ladder needs that ladder on
every request, and a client speaking the plain OpenAI image API sends a
sentence and has nowhere to put it. So the prefix belongs to the model, with
the request able to override or switch it off -- and applied at most once, or a
caller who pastes a full prompt gets the ladder twice.

**OCR upload size.** Nothing in the OCR path caps a single page, and a page
over a megabyte is ordinary: a 300 DPI A4 scan is several. What existed was a
hardcoded 100 MB ceiling on a whole document, which a 300 DPI colour filing
passes without being unusual, and no way to raise it. It is now
``ocr.max_upload_mb``. The 1 MB figure that makes a large page fail in the
field is nginx's own ``client_max_body_size`` default, in a reverse proxy in
front -- so the error says so, which is the only way a caller can tell the two
apart.
"""
import io
import types

import pytest
from fastapi import HTTPException

from codai.api import images as images_mod


# --------------------------------------------------------------------------- #
#  the prompt prefix
# --------------------------------------------------------------------------- #

def _req(**kw):
    base = dict(prompt="1girl, red hair", negative_prompt=None,
                prompt_prefix=None, negative_prompt_prefix=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_nothing_configured_leaves_the_prompts_alone():
    p, n = images_mod._effective_prompts(_req(negative_prompt="blurry"), {})
    assert p == "1girl, red hair"
    assert n == "blurry"


def test_the_model_supplies_the_ladder():
    """The case this exists for: a client that only sends a sentence."""
    p, n = images_mod._effective_prompts(
        _req(prompt="a knight in a field"),
        {"prompt_prefix": "score_9, score_8_up, score_7_up",
         "negative_prompt_prefix": "score_4, score_5, score_6"})
    assert p == "score_9, score_8_up, score_7_up, a knight in a field"
    assert n == "score_4, score_5, score_6"


def test_the_negative_prefix_joins_an_existing_negative():
    p, n = images_mod._effective_prompts(
        _req(negative_prompt="blurry, jpeg artifacts"),
        {"negative_prompt_prefix": "score_4, score_5"})
    assert n == "score_4, score_5, blurry, jpeg artifacts"


def test_the_request_overrides_the_model():
    p, _ = images_mod._effective_prompts(
        _req(prompt_prefix="masterpiece"), {"prompt_prefix": "score_9"})
    assert p == "masterpiece, 1girl, red hair"


def test_an_empty_string_switches_it_off_for_this_call():
    """What a caller writing its own complete prompt needs."""
    p, n = images_mod._effective_prompts(
        _req(prompt_prefix="", negative_prompt_prefix="",
             negative_prompt="blurry"),
        {"prompt_prefix": "score_9", "negative_prompt_prefix": "score_4"})
    assert p == "1girl, red hair"
    assert n == "blurry"


def test_it_is_applied_at_most_once():
    """A pasted prompt that already opens with the ladder must not get it
    twice -- repeating it only dilutes the rest of the prompt."""
    prefix = "score_9, score_8_up"
    p, _ = images_mod._effective_prompts(
        _req(prompt=f"{prefix}, 1girl"), {"prompt_prefix": prefix})
    assert p == f"{prefix}, 1girl"
    assert p.count("score_9") == 1


def test_the_already_applied_check_ignores_case():
    p, _ = images_mod._effective_prompts(
        _req(prompt="SCORE_9, score_8_up, 1girl"),
        {"prompt_prefix": "score_9, score_8_up"})
    assert p.lower().count("score_9") == 1


@pytest.mark.parametrize("prefix,expected", [
    ("score_9,",      "score_9, 1girl"),        # no doubled comma
    ("score_9 ,  ",   "score_9, 1girl"),
    ("  score_9  ",   "score_9, 1girl"),
])
def test_the_join_is_tidy(prefix, expected):
    p, _ = images_mod._effective_prompts(
        _req(prompt="1girl"), {"prompt_prefix": prefix})
    assert p == expected


def test_a_prefix_with_an_empty_prompt_is_the_whole_prompt():
    p, _ = images_mod._effective_prompts(
        _req(prompt=""), {"prompt_prefix": "score_9, score_8_up"})
    assert p == "score_9, score_8_up"


def test_a_prefix_carries_the_weight_grammar_through():
    """The prefix is prompt text like any other, so (tag:1.2) in it works."""
    p, _ = images_mod._effective_prompts(
        _req(prompt="1girl"), {"prompt_prefix": "(masterpiece:1.2), score_9"})
    assert p == "(masterpiece:1.2), score_9, 1girl"
    from codai.models.prompt_weighting import parse_prompt_attention
    weights = {t.strip(" ,"): w for t, w in parse_prompt_attention(p)}
    assert weights["masterpiece"] == pytest.approx(1.2)


@pytest.mark.parametrize("cls_name", ["ImageEditRequest", "ImageInpaintRequest",
                                      "ImageOutfitRequest"])
def test_every_endpoint_that_takes_a_prompt_takes_the_prefixes(cls_name):
    cls = getattr(images_mod, cls_name)
    for field in ("prompt_prefix", "negative_prompt_prefix"):
        assert field in cls.model_fields, f"{cls_name} is missing {field}"
        assert cls.model_fields[field].default is None


def test_the_generation_request_takes_them_too():
    from codai.pydantic.imagerequest import ImageGenerationRequest
    r = ImageGenerationRequest(model="m", prompt="p", prompt_prefix="score_9")
    assert r.prompt_prefix == "score_9"
    assert r.negative_prompt_prefix is None


def test_the_prefix_reaches_the_encode_and_the_cache_key_together():
    """If the cache were keyed on the prompt as SENT, two models with different
    prefixes would share one entry and the second would get the first's text."""
    src = open("codai/api/images.py").read()
    head = src.index("async def _generate_with_diffusers")
    body = src[head:head + 9000]
    assert "prompt, neg_prompt = _effective_prompts(request, model_config)" in body
    assert "_embed_cache.get(prompt, neg_prompt, model_id)" in body
    assert "_embed_cache.get(request.prompt" not in body


def test_it_is_configurable_per_model():
    assert '"prompt_prefix", "negative_prompt_prefix",' in \
        open("codai/admin/routes.py").read()
    ui = open("codai/admin/templates/models.html").read()
    assert "cfg-prompt-prefix" in ui and "cfg-negative-prompt-prefix" in ui


def test_nothing_injects_a_quality_ladder_on_its_own():
    """The prefix is the only way tags are added, and it is empty by default:
    an install that configures nothing sends exactly what the caller wrote."""
    for cfg in (None, {}, {"prompt_prefix": "", "negative_prompt_prefix": ""}):
        p, n = images_mod._effective_prompts(_req(), cfg)
        assert p == "1girl, red hair", cfg
        assert n == "", cfg


# --------------------------------------------------------------------------- #
#  OCR upload size
# --------------------------------------------------------------------------- #

def test_the_default_ceiling_is_a_hundred_megabytes():
    from codai.api.ocr import _max_upload_bytes, _DEFAULT_MAX_OCR_MB
    assert _DEFAULT_MAX_OCR_MB == 100
    assert _max_upload_bytes(types.SimpleNamespace()) == 100 * 1024 * 1024


def test_the_ceiling_is_configurable():
    from codai.api.ocr import _max_upload_bytes
    assert _max_upload_bytes(
        types.SimpleNamespace(max_upload_mb=512)) == 512 * 1024 * 1024


def test_zero_means_no_ceiling_here():
    """For an install whose reverse proxy already decides the limit."""
    from codai.api.ocr import _max_upload_bytes
    assert _max_upload_bytes(types.SimpleNamespace(max_upload_mb=0)) == 0


@pytest.mark.parametrize("value", [None, "", "lots", -5])
def test_a_meaningless_ceiling_falls_back_to_the_default(value):
    from codai.api.ocr import _max_upload_bytes
    got = _max_upload_bytes(types.SimpleNamespace(max_upload_mb=value))
    assert got in (0, 100 * 1024 * 1024)


@pytest.mark.parametrize("mb", [1, 2, 5, 20, 99])
def test_a_page_over_a_megabyte_is_under_the_ceiling(mb):
    """The ask, as arithmetic: nothing in this path stops at 1 MB."""
    from codai.api.ocr import _max_upload_bytes
    cap = _max_upload_bytes(types.SimpleNamespace())
    assert mb * 1024 * 1024 < cap


def test_the_413_names_the_setting_and_says_it_was_not_read():
    """A 413 that just says "too large" leaves a caller unable to tell this
    ceiling from a reverse proxy's 1 MB default."""
    src = open("codai/api/ocr.py").read()
    assert "ocr.max_upload_mb" in src
    assert "it was NOT read" in src
    assert "file too large (max 100 MB)" not in src     # the old fixed message


def test_the_code_records_that_this_is_not_a_per_page_limit():
    """The 1 MB figure in the field is nginx's client_max_body_size default,
    and whoever next reads this code should not have to rediscover that."""
    src = open("codai/api/ocr.py").read()
    assert "client_max_body_size" in src


def test_the_bundled_nginx_lets_a_big_body_through():
    """coderai's own nginx must not be the thing stopping a page: its config
    raises the 1 MB default, and the OCR location inherits it."""
    conf = open("packaging/linux/nginx.conf").read()
    assert "client_max_body_size" in conf
    size = conf.split("client_max_body_size", 1)[1].split(";", 1)[0].strip()
    assert size.endswith("m") and int(size[:-1]) >= 1024, size


def test_the_generated_ocr_routing_does_not_lower_it():
    """The fleet's nginx include is written per request path; a
    client_max_body_size there would override the inherited one."""
    from codai import ocr_fleet_workers as fleet
    upstream, locations = fleet.render([9100, 9101])
    assert "location = /v1/ocr {" in locations
    assert "client_max_body_size" not in locations
    assert "client_max_body_size" not in upstream


def test_a_multi_megabyte_single_page_pdf_is_read_as_one_page():
    """End to end on the real reader: a page far over a megabyte is ordinary.

    Built as one page carrying an incompressible image, so the file is big for
    the reason a scan is big rather than by padding.
    """
    pytest.importorskip("pypdfium2")
    import os
    import random

    from PIL import Image

    random.seed(0)
    # ~2480x3508 is A4 at 300 DPI; noise so nothing compresses it away.
    img = Image.frombytes("RGB", (1240, 1754),
                          bytes(random.getrandbits(8)
                                for _ in range(1240 * 1754 * 3)))
    buf = io.BytesIO()
    img.save(buf, "PDF", resolution=150.0)
    data = buf.getvalue()
    assert len(data) > 1024 * 1024, f"only {len(data)} bytes; test is not testing"

    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(data)
    try:
        assert len(doc) == 1
        w, h = doc.get_page_size(0)
        assert w > 0 and h > 0
    finally:
        doc.close()

    from codai.api.ocr import _max_upload_bytes
    assert len(data) < _max_upload_bytes(types.SimpleNamespace())
    assert len(data) > 1024 * 1024
    del os


def test_the_setting_is_exposed_where_an_operator_looks():
    from codai.config import OcrConfig
    assert OcrConfig().max_upload_mb == 100
    assert "max_upload_mb" in open("codai/admin/routes.py").read()
    assert "s-ocr-max-upload-mb" in open(
        "codai/admin/templates/settings.html").read()
