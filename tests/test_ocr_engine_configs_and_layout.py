"""Several published ids for one OCR engine, and surya's layout actually run.

Image models have had multi-id for a long time: one checkpoint, several
catalogue entries, each with its own settings and one copy of the weights.
OCR engines were a fixed registry of five singletons reading flat
``ocr.<engine>_*`` fields, so an install could run surya exactly one way.

The reason it mattered here is a lie in the catalogue. ``/v1/ocr/engines``
advertised ``surya: {regions: true, tables: true}`` while the surya worker
built every page with ``"regions": [], "tables": []`` hardcoded — surya ships
a layout predictor and a table recogniser and simply never ran them. A client
that chose surya BECAUSE the catalogue said it returned regions got empty
arrays, no error, and nothing to ask about.

So: ``ocr.surya_layout`` decides whether a worker LOADS the predictors (they
are models, and deciding to have one when a page asks would grow the footprint
past what the VRAM accounting reserved), a named config can turn that on under
its own id beside plain surya, and a per-request ``layout`` chooses whether to
spend the pass. Asking for layout where none is loaded is refused with the
configuration to use instead, because answering 200 with empty regions is the
same lie in a new place.
"""
import pytest

from codai.config import OcrConfig
from codai.ocr import engine_configs as ec
from codai.ocr.manager import (_check_layout_request, _page_post_fields,
                               _returns_shapes)
from codai.ocr.base import OcrError, OcrRegion
from codai.api import ocr as ocr_api


def _function_source(path, name):
    """One function's whole body, by AST.

    Not a byte window: a slice of N characters after the `def` breaks the
    moment the function grows, which is how two of these assertions failed on
    an edit that was correct.
    """
    import ast
    src = open(path).read()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name):
            return ast.get_source_segment(src, node) or ""
    raise AssertionError("no function %r in %s" % (name, path))


def _cfg(**kw):
    return OcrConfig(**kw)


SURYA_LAYOUT = [{"id": "surya-layout", "engine": "surya",
                 "overrides": {"surya_layout": True, "surya_table_rec": True}}]


# --------------------------------------------------------------------------- #
#  resolving a named configuration
# --------------------------------------------------------------------------- #

def test_an_id_resolves_to_its_base_engine_and_its_overrides():
    cfg = _cfg(engine_configs=SURYA_LAYOUT)
    assert ec.resolve(cfg, "surya-layout") == (
        "surya-layout", "surya", {"surya_layout": True, "surya_table_rec": True})


def test_a_bare_engine_name_is_not_a_configuration():
    cfg = _cfg(engine_configs=SURYA_LAYOUT)
    assert ec.resolve(cfg, "surya") == ("surya", "surya", {})


def test_an_unknown_name_is_left_for_the_engine_check_to_refuse():
    """This layer answers "is this a named config", not "is this usable" —
    otherwise an unknown name would produce two different refusals."""
    cfg = _cfg(engine_configs=SURYA_LAYOUT)
    assert ec.resolve(cfg, "nonsense") == ("nonsense", "nonsense", {})


def test_the_overrides_apply_without_touching_the_shared_config():
    cfg = _cfg(engine_configs=SURYA_LAYOUT, surya_layout=False)
    eff = ec.effective(cfg, {"surya_layout": True})
    assert eff.surya_layout is True
    assert cfg.surya_layout is False, "the shared config is every other id's too"


def test_a_typo_in_the_overrides_is_refused_not_ignored():
    """An id that looks configured, serves the base settings and differs from
    its sibling in nothing at all is the quietest possible failure."""
    with pytest.raises(ValueError, match="surya_layoutt"):
        ec.effective(_cfg(), {"surya_layoutt": True})


def test_an_id_equal_to_its_engine_is_not_a_second_configuration():
    cfg = _cfg(engine_configs=[{"id": "surya", "engine": "surya",
                                "overrides": {"surya_layout": True}}])
    assert ec.configured(cfg) == {}


@pytest.mark.parametrize("entry", [
    "not a dict", {}, {"id": "x"}, {"engine": "surya"}, {"id": "", "engine": ""},
])
def test_a_malformed_entry_is_skipped_rather_than_fatal(entry):
    """A bad line must not take the OCR subsystem down; the id simply does not
    resolve, and that is reported where it is asked for."""
    assert ec.configured(_cfg(engine_configs=[entry])) == {}


def test_the_ids_of_an_engine_can_be_listed():
    cfg = _cfg(engine_configs=SURYA_LAYOUT + [
        {"id": "chandra-plain", "engine": "chandra",
         "overrides": {"chandra_layout": False}}])
    assert ec.ids_for(cfg, "surya") == ["surya-layout"]
    assert ec.ids_for(cfg, "chandra") == ["chandra-plain"]
    assert ec.ids_for(cfg, "paddle") == []


def test_a_configuration_describes_itself_for_a_log_line():
    cfg = _cfg(engine_configs=SURYA_LAYOUT)
    got = ec.describe(cfg, "surya-layout")
    assert got.startswith("surya (") and "surya_layout=True" in got


# --------------------------------------------------------------------------- #
#  the catalogue stops lying
# --------------------------------------------------------------------------- #

def test_plain_surya_no_longer_claims_regions_it_does_not_return():
    """The defect, stated as a test: this row said true while the worker
    returned a hardcoded empty list."""
    shapes = _returns_shapes(_cfg())
    assert shapes["surya"]["regions"] is False
    assert shapes["surya"]["tables"] is False
    assert shapes["surya"]["boxes"] is True, "the detector gives boxes regardless"


def test_surya_with_the_layout_predictor_claims_them_truthfully():
    shapes = _returns_shapes(_cfg(surya_layout=True, surya_table_rec=True))
    assert shapes["surya"]["regions"] is True
    assert shapes["surya"]["tables"] is True


def test_tables_need_the_table_recogniser_not_merely_the_layout():
    shapes = _returns_shapes(_cfg(surya_layout=True, surya_table_rec=False))
    assert shapes["surya"]["regions"] is True
    assert shapes["surya"]["tables"] is False


def test_table_recognition_without_layout_claims_nothing():
    """It works from the layout's table regions, so on its own it is inert."""
    shapes = _returns_shapes(_cfg(surya_layout=False, surya_table_rec=True))
    assert shapes["surya"]["tables"] is False


def test_chandra_is_still_read_from_its_own_setting():
    assert _returns_shapes(_cfg(chandra_layout=True))["chandra"]["regions"] is True
    assert _returns_shapes(_cfg(chandra_layout=False))["chandra"]["regions"] is False


# --------------------------------------------------------------------------- #
#  asking for layout where there is none
# --------------------------------------------------------------------------- #

def test_layout_is_refused_on_a_surya_that_has_no_predictor():
    with pytest.raises(OcrError) as e:
        _check_layout_request(_cfg(), "surya", "surya", {"layout": True})
    assert e.value.status == 400
    assert "ocr.surya_layout" in str(e.value)


def test_the_refusal_names_a_configured_id_that_can_serve_it():
    """The useful refusal: not "no", but "ask for this instead"."""
    cfg = _cfg(engine_configs=SURYA_LAYOUT)
    with pytest.raises(OcrError) as e:
        _check_layout_request(cfg, "surya", "surya", {"layout": True})
    assert "surya-layout" in str(e.value)


def test_layout_is_allowed_on_the_configuration_that_loads_it():
    cfg = _cfg(engine_configs=SURYA_LAYOUT)
    eff = ec.effective(cfg, ec.resolve(cfg, "surya-layout")[2])
    _check_layout_request(eff, "surya", "surya-layout", {"layout": True})


def test_not_asking_for_layout_is_never_refused():
    for opts in ({}, {"layout": False}, None):
        _check_layout_request(_cfg(), "surya", "surya", opts or {})


def test_olmocr_cannot_do_layout_and_says_so_rather_than_returning_empty():
    with pytest.raises(OcrError) as e:
        _check_layout_request(_cfg(), "olmocr", "olmocr", {"layout": True})
    assert "olmOCR" in str(e.value) and e.value.status == 400


def test_chandra_with_its_layout_prompt_off_is_refused():
    with pytest.raises(OcrError) as e:
        _check_layout_request(_cfg(chandra_layout=False), "chandra", "chandra",
                              {"layout": True})
    assert "ocr.chandra_layout" in str(e.value)


def test_the_detector_engines_accept_it_since_they_already_do_layout():
    for base in ("paddle", "doctr"):
        _check_layout_request(_cfg(), base, base, {"layout": True})


# --------------------------------------------------------------------------- #
#  the request field, and the trip to a pod
# --------------------------------------------------------------------------- #

def test_the_request_can_ask_for_the_pass_or_decline_it():
    assert ocr_api._page_opts(None, None, layout=True) == {"layout": True}
    assert ocr_api._page_opts(None, None, layout=False) == {"layout": False}
    assert "layout" not in ocr_api._page_opts(None, None)


def test_both_endpoints_take_the_layout_field():
    import inspect
    for fn in (ocr_api.create_ocr, ocr_api.create_ocr_batch):
        assert "layout" in inspect.signature(fn).parameters, fn.__name__


def test_the_choice_is_forwarded_to_a_pod_including_the_negative():
    """`false` has to travel too: on a pool whose configuration loads the
    predictor, declining the pass is the whole saving."""
    f, _h = _page_post_fields("k", 200, {"layout": True}, None, "surya")
    assert f["layout"] == "true"
    f, _h = _page_post_fields("k", 200, {"layout": False}, None, "surya")
    assert f["layout"] == "false"
    f, _h = _page_post_fields("k", 200, {}, None, "surya")
    assert "layout" not in f


def test_the_worker_is_told_at_load_what_to_hold():
    """Capability is decided when the worker is built, not when a page asks:
    a predictor is a model, and acquiring one mid-page is how a footprint
    grows past what the VRAM accounting reserved."""
    import inspect
    from codai.ocr import surya as surya_mod
    src = inspect.getsource(surya_mod.SuryaEngine)
    assert '"layout": bool(getattr(self.cfg, "surya_layout"' in src
    assert '"table_rec"' in src


def test_a_region_carries_the_engines_own_reading_order():
    r = OcrRegion(label="Text", bbox=[0, 0, 1, 1], conf=0.9, order=3)
    assert r.to_dict()["order"] == 3


def test_a_region_from_an_engine_with_no_order_says_so_rather_than_guessing():
    assert OcrRegion(label="Text", bbox=[0, 0, 1, 1]).to_dict()["order"] == -1


def test_the_order_survives_the_worker_protocol():
    """It crosses as JSON and is mapped back by hand; a field left out of that
    mapping reduces the model's reading order to our list order."""
    src = open("codai/ocr/subprocess_engine.py").read()
    assert 'order=int(r.get("order", -1))' in src


def test_the_surya_worker_runs_the_predictors_instead_of_returning_empty():
    src = open("codai/ocr/workers/ocr_worker.py").read()
    assert "def _run_layout" in src
    assert 'page["regions"], page["tables"] = self._run_layout(img)' in src
    # surya's LayoutBox carries `position`, its own reading order.
    body = src[src.index("def _run_layout"):][:3000]
    assert '"position"' in body or "position" in body
    assert "TableRecPredictor" in src, "tables need the table recogniser"


# --------------------------------------------------------------------------- #
#  and a POD has to be able to differ from its sibling
# --------------------------------------------------------------------------- #

def test_a_pool_can_name_the_configuration_its_pods_serve():
    """The pod's OCR settings are derived from the orchestrator's single `ocr`
    config, which is right until two pools serve the same model differently.
    Without `ocr_engine_config` on the entry, a surya-layout pool and a plain
    surya pool would be sent identical env and the second id would be
    decoration."""
    src = open("codai/api/runpod_worker.py").read()
    # On the model ENTRY (a dict). It was first written against `mcfg`, which
    # is a RunpodModelConfig object with no .get, and every pipeline pod's env
    # build raised AttributeError.
    assert 'entry or {}).get("ocr_engine_config")' in src
    head = src.index("if pod_runs_ocr_pipeline(mcfg, image):")
    body = src[head:head + 2600]
    assert "_ec.effective(" in body, "the overrides have to be applied"
    assert body.index("_ocr_cid") < body.index("CODERAI_OCR_ENABLED"), \
        "applied BEFORE the env is derived from it, or it changes nothing"


def test_the_layout_switches_travel_to_a_pod_as_booleans():
    """`bool("0")` is True, so a false sent through the typed loop would turn
    the feature ON and the pod would load two models nobody asked for."""
    worker = open("codai/api/runpod_worker.py").read()
    cfg = open("codai/config.py").read()
    for var in ("CODERAI_OCR_SURYA_LAYOUT", "CODERAI_OCR_SURYA_TABLE_REC"):
        assert var in worker, f"{var} is never sent"
        assert var in cfg, f"{var} is never read on the pod"
    # In the _flag block (read as a string), not the typed one.
    blk = cfg[cfg.index("CODERAI_OCR_SURYA_REPAIR_ACCENTS"):]
    blk = blk[:blk.index("CODERAI_VLLM_RETURN_TOKEN_IDS")]
    assert "CODERAI_OCR_SURYA_LAYOUT" in blk
    assert "CODERAI_OCR_SURYA_TABLE_REC" in blk


# --------------------------------------------------------------------------- #
#  what a POD is told the engine is
# --------------------------------------------------------------------------- #

PIPE_IMG = "ghcr.io/nextime/coderai-vllm-ocr:0.4.98"
PLAIN_IMG = "ghcr.io/nextime/coderai-vllm:latest"


def test_a_pipeline_pod_is_asked_for_the_engine_it_actually_carries():
    """Found by running a page through it, not by reading the code.

    `pod_ocr_engine` rewrote EVERY OCR forward to docTR, on the premise that a
    pod image cannot carry a VLM's model server. `coderai-vllm-ocr` does carry
    one -- that is the whole pod-side pipeline -- and carries no docTR at all,
    so the rewrite asked a pipeline pod for the one engine it cannot have and
    the pod answered "OCR engine 'doctr' is not enabled". Every whole-request
    /v1/ocr forward to a pipeline pod failed that way.
    """
    from codai.api.runpod_worker import pod_ocr_engine
    for eng in ("surya", "chandra", "olmocr"):
        assert pod_ocr_engine(eng, PIPE_IMG) == eng


def test_a_plain_pod_still_falls_back_to_doctr():
    """The original reasoning, still right for an image with no model server."""
    from codai.api.runpod_worker import pod_ocr_engine
    for eng in ("surya", "chandra", "olmocr"):
        assert pod_ocr_engine(eng, PLAIN_IMG) == "doctr"


def test_the_detector_engines_pass_through_either_way():
    from codai.api.runpod_worker import pod_ocr_engine
    for img in (PIPE_IMG, PLAIN_IMG):
        assert pod_ocr_engine("paddle", img) == "paddle"
        assert pod_ocr_engine("doctr", img) == "doctr"


def test_an_unknown_engine_is_still_doctr():
    from codai.api.runpod_worker import pod_ocr_engine
    assert pod_ocr_engine("nonsense", PIPE_IMG) == "doctr"


def test_omitting_the_image_keeps_the_old_answer():
    """Callers that cannot know the image must not change behaviour."""
    from codai.api.runpod_worker import pod_ocr_engine
    assert pod_ocr_engine("surya") == "doctr"


def test_the_gateway_asks_with_the_image():
    """A pod's capability depends on its image, so the question cannot be asked
    without one — that omission is what sent every OCR forward to docTR."""
    src = open("codai/api/remote_gateway.py").read()
    assert "pod_ocr_engine(model, _img)" in src
    head = src.index("if cap == \"ocr\":")
    body = src[head:head + 900]
    assert "pod_image_for_entry" in body


# --------------------------------------------------------------------------- #
#  the answer says WHICH configuration served it
# --------------------------------------------------------------------------- #

def test_the_pod_path_reports_the_configuration_not_the_base_engine():
    """A pod answers with the engine it ran, because the configuration ids are
    an orchestrator-side naming layer it has never heard of. Reported as-is,
    both `surya` and `surya-layout` came back as "surya" and a caller
    comparing them could not tell which had served -- which is most of the
    point of having two ids.

    Verified live 2026-10-25: a page through `engine=surya-layout` answered
    `engine: surya`.
    """
    src = open("codai/ocr/manager.py").read()
    head = src.index("name, pages, blank, stamps, signatures = _out")
    body = src[head:head + 1200]
    assert "_req_id" in body and "name = _req_id" in body
    assert "_req_id != _req_base" in body, \
        "a plain engine name must keep reporting itself"


def test_the_local_path_already_reported_it():
    """Both paths have to agree, or the answer depends on where the page was
    read -- which is configuration, and nothing a client should see."""
    src = open("codai/ocr/manager.py").read()
    head = src.index("async def ocr_pages")
    body = src[head:head + 2000]
    assert "served_as" in body


# --------------------------------------------------------------------------- #
#  a confidence that was never measured does not look like one
# --------------------------------------------------------------------------- #

def test_a_table_cell_carries_no_confidence_unless_one_was_measured():
    """surya's table recogniser does not score cells in the mode we call, so
    every cell came back `conf: 0.0` -- a placeholder wearing a
    measurement's clothes. Same confusion `confidence_measured` prevents for
    pages elsewhere in this subsystem."""
    body = _function_source("codai/ocr/workers/ocr_worker.py", "_run_layout")
    assert 'cell["conf"] = float(conf)' in body
    assert "isinstance(conf, (int, float))" in body, \
        "presence has to decide it, not truthiness: a real 0.0 is a measurement"
    assert '"confidence_measured": bool(measured)' in body
    assert '"conf": float(getattr(c, "confidence", 0.0) or 0.0)' not in body, \
        "the unconditional placeholder is what made every cell read 0.00"


def test_table_markup_is_absent_rather_than_empty():
    """So a caller can tell "this engine returns no HTML" from "this table's
    HTML is empty"."""
    body = _function_source("codai/ocr/workers/ocr_worker.py", "_run_layout")
    assert "if html:" in body and 'entry["html"] = html' in body
