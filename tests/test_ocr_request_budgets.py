"""What `/v1/ocr` and `/v1/ocr/batch` actually accept, and that they are wired.

Two defects this covers, both of the same family -- a thing that silently does
nothing looks exactly like a thing that works:

1. `/v1/ocr`'s route decorator ended up attached to the `_page_opts` HELPER
   rather than to `create_ocr`, because the helper was inserted between the
   decorator and its function. FastAPI registered the helper as the handler, so
   the endpoint existed, answered, and was not the OCR endpoint. Nothing failed
   at import and no test looked at the route table.

2. `/v1/ocr/batch` never took `max_tokens`/`max_continuations` at all. A batch
   is how a caller sends real work, so the budgets were accepted only on the
   path nobody benchmarks -- and FastAPI drops an unknown form field in silence,
   so a caller setting `max_tokens` on every batch was told nothing.

Asserted against the route table and the signatures rather than by calling the
endpoints, so the tests hold without an OCR engine, a model or a GPU.
"""
import inspect

import pytest


def _routes():
    from codai.api import ocr as ocr_api
    return {(r.path, tuple(sorted(r.methods))): r
            for r in ocr_api.router.routes if hasattr(r, "methods")}


def _endpoint(path, method="POST"):
    for (p, methods), r in _routes().items():
        if p == path and method in methods:
            return r.endpoint
    raise AssertionError(f"no {method} {path} route is registered")


def test_the_ocr_route_points_at_the_ocr_handler():
    """The bug was not a missing route — it was a route to the wrong function."""
    fn = _endpoint("/v1/ocr")
    assert fn.__name__ == "create_ocr", (
        f"POST /v1/ocr is handled by {fn.__name__!r}")


def test_the_page_options_helper_is_not_an_endpoint():
    from codai.api import ocr as ocr_api
    handlers = {r.endpoint for r in ocr_api.router.routes if hasattr(r, "endpoint")}
    assert ocr_api._page_opts not in handlers, (
        "_page_opts is registered as an HTTP handler")


def test_the_ocr_handler_takes_an_upload():
    """A helper promoted to a handler by accident would take neither."""
    params = inspect.signature(_endpoint("/v1/ocr")).parameters
    assert "file" in params


@pytest.mark.parametrize("path,upload", [("/v1/ocr", "file"),
                                         ("/v1/ocr/batch", "files")])
@pytest.mark.parametrize("field", ["max_tokens", "max_continuations"])
def test_both_paths_accept_the_per_request_budgets(path, upload, field):
    params = inspect.signature(_endpoint(path)).parameters
    assert upload in params, f"{path} does not take an upload"
    assert field in params, (
        f"{path} does not accept {field} — FastAPI will drop it in silence")


@pytest.mark.parametrize("path", ["/v1/ocr", "/v1/ocr/batch"])
@pytest.mark.parametrize("field", ["max_tokens", "max_continuations"])
def test_a_budget_is_optional_and_defaults_to_the_configuration(path, field):
    """Omitted must mean "use the configured value", never 0: 0 is meaningful
    for both (no cap override; resumption off)."""
    p = inspect.signature(_endpoint(path)).parameters[field]
    assert p.default is not inspect.Parameter.empty
    assert getattr(p.default, "default", "missing") is None, (
        f"{path}:{field} does not default to None")


# --- what the budgets become ------------------------------------------------


def test_nothing_sent_means_nothing_overridden():
    from codai.api.ocr import _page_opts
    assert _page_opts(None, None) == {}


def test_zero_is_carried_rather_than_treated_as_absent():
    from codai.api.ocr import _page_opts
    assert _page_opts(0, 0) == {"max_tokens": 0, "max_continuations": 0}


def test_each_budget_travels_alone():
    from codai.api.ocr import _page_opts
    assert _page_opts(3072, None) == {"max_tokens": 3072}
    assert _page_opts(None, 4) == {"max_continuations": 4}


def test_a_negative_budget_is_clamped_not_passed_on():
    from codai.api.ocr import _page_opts
    assert _page_opts(-5, -1) == {"max_tokens": 0, "max_continuations": 0}


def test_the_batch_path_forwards_them_to_the_document_call():
    """The signature accepting a field proves nothing about it being used.

    Asserted on the ARGUMENTS rather than on the whole call as it was once
    written: ``_page_opts`` has since grown the page-image overrides, and an
    exact-string match on its argument list failed on a change that forwarded
    strictly more than before. What this test is for is that the budgets reach
    the document call at all.
    """
    from codai.api import ocr as ocr_api
    src = inspect.getsource(_endpoint("/v1/ocr/batch"))
    body = src.split("ocr_manager.ocr_document")[1]
    call = body.split("page_opts=_page_opts(")
    assert len(call) > 1, "the batch path accepts the budgets and then drops them"
    args = call[1].split(")")[0]
    for name in ("max_tokens", "max_continuations"):
        assert name in args, f"the batch path drops {name}"
