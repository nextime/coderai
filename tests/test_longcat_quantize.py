"""Building INT8 LongCat weights from the model page.

A 13.6B bf16 DiT does not fit a 24 GB card and streams across PCIe every step; the
INT8 copy does fit. It is an ACTION, never a default — it is lossy and slow — and
the bf16 weights are kept, so un-ticking "INT8 DiT" goes straight back to them.

The quantiser itself runs in LongCat's isolated py3.10 venv (it needs upstream's
helpers and torch 2.6), so it is a worker script like training, not in-process.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "longcat_quantize.py"
SERVICE = ROOT / "tools" / "longcat_service.py"
WORKER = ROOT / "codai" / "api" / "longcat_worker.py"
ROUTES = ROOT / "codai" / "admin" / "routes.py"
PAGE = ROOT / "codai" / "admin" / "templates" / "models.html"


def test_the_quantiser_script_exists_and_parses():
    import ast
    assert SCRIPT.is_file()
    ast.parse(SCRIPT.read_text(encoding="utf-8"))


def test_it_never_replaces_the_bf16_weights():
    """It writes a sibling directory; nothing removes or overwrites dit/."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "base_model_int8" in src
    for destructive in ("shutil.rmtree", "os.remove", "os.unlink", "unlink("):
        assert destructive not in src, f"{destructive} has no business here"


def test_it_quantises_on_the_cpu():
    """The whole point is to make a model that does not fit the card — so getting
    there must not need the card."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert 'to("cpu")' in src


def test_an_existing_build_is_not_silently_redone():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "overwrite" in src and "already quantised" in src


def test_the_worker_runs_it_in_the_inference_venv():
    src = WORKER.read_text(encoding="utf-8")
    assert "_QUANTIZE_SCRIPT" in src
    assert "def quantize_int8(" in src
    # ensure_built() is the inference venv; ensure_train_built() is SimpleTuner's.
    body = src[src.index("def quantize_int8("):]
    body = body[:body.index("\ndef ")]
    assert "ensure_built(config)" in body
    assert "ensure_train_built" not in body


def test_the_service_can_load_a_quantised_base_dit():
    """Upstream's load_quantized_dit hardcodes the AVATAR transformer, which is
    wrong for a quantised base checkpoint — so the base family needs its own."""
    src = SERVICE.read_text(encoding="utf-8")
    assert "_load_quantized_base_dit" in src
    assert "LongCatVideoTransformer3DModel(**config)" in src
    # the avatar path must still use upstream's loader
    assert "load_quantized_dit(root, subfolder=LC.INT8_SUBDIR" in src


def test_the_endpoint_routes_longcat_models_to_int8():
    src = ROUTES.read_text(encoding="utf-8")
    assert '_model_backend(model_id) == "longcat"' in src
    assert "_start_longcat_quantize" in src
    # and a model whose weights are absent is refused rather than started
    assert "not downloaded yet" in src


def test_the_status_endpoint_reports_longcat_jobs():
    src = ROUTES.read_text(encoding="utf-8")
    assert "_longcat_quant_jobs" in src


def test_the_page_offers_the_action_next_to_the_int8_switch():
    html = PAGE.read_text(encoding="utf-8")
    assert 'id="cfg-longcat-int8"' in html, "the use-int8 switch is gone"
    assert 'id="cfg-longcat-quant-btn"' in html
    assert "startLongcatQuantize" in html
    # it must read the same hidden field the other quantize button uses
    js = html[html.index("async function startLongcatQuantize"):]
    assert "getElementById('cfg-path')" in js[:600]


def test_quantising_is_not_the_default_anywhere():
    """It must stay opt-in: nothing may turn use_int8 on by itself."""
    for path in (SERVICE, WORKER):
        src = path.read_text(encoding="utf-8")
        assert not re.search(r'use_int8["\']?\s*[:=]\s*True', src), path.name
