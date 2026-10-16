# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""`ocr.max_pages`: refuse an over-long document instead of reading it.

Length is the one cost knowable BEFORE any work (page_estimate reads the page
tree) and the one that scales every other cost at once. An operator whose
corpus has a tail of outliers that will never finish inside a client's timeout
is better served refusing them in milliseconds than paying for them twice.

Two properties matter and both are easy to get wrong:

* it is OFF by default -- page count is not a fault, and an install reading
  book scans is as legitimate as one reading two-page letters;
* a refusal is LOUD. This subsystem already carries the trap that a degraded
  page answers 200; a skipped document that looked like an empty read would be
  the same mistake with a bigger blast radius.
"""

import types

import pytest

import codai.ocr.manager as om
from codai.ocr.base import OcrError


def _cfg(**kw):
    """The real config object: the ceiling must hold against the shipped one,
    not a stub that happens to carry only the fields this test thought of."""
    from codai.config import OcrConfig
    cfg = OcrConfig()
    cfg.enabled = True
    cfg.default_engine = "surya"
    cfg.surya_enabled = True
    cfg.surya_accept_license = True
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _mgr(cfg, pages: int):
    mgr = om.OcrManager()
    mgr._cfg = cfg
    return mgr


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _no_real_work(monkeypatch):
    """Nothing past the ceiling check should run in these tests."""
    monkeypatch.setattr(om, "_wait_thermal_safe", lambda: None)

    def _never(*_a, **_k):
        raise AssertionError("the document was read despite the ceiling")

    monkeypatch.setattr(om, "pod_pipeline_model", lambda *_a, **_k: "")
    return _never


async def _read(mgr, pages_claimed, monkeypatch, _never):
    monkeypatch.setattr(om, "page_estimate",
                        lambda *_a, **_k: (pages_claimed, pages_claimed * 1000))
    monkeypatch.setattr(om.OcrManager, "_read_windowed", _never)
    return await mgr.ocr_document(b"%PDF-1.4 x", filename="big.pdf",
                                  content_type="application/pdf")


@pytest.mark.anyio
async def test_a_document_over_the_ceiling_is_refused(monkeypatch, _no_real_work):
    mgr = _mgr(_cfg(max_pages=200), 201)
    with pytest.raises(OcrError) as e:
        await _read(mgr, 201, monkeypatch, _no_real_work)
    assert e.value.status == 413


@pytest.mark.anyio
async def test_the_refusal_names_both_numbers(monkeypatch, _no_real_work):
    """A caller has to be able to act on it: how long, and how long is allowed."""
    mgr = _mgr(_cfg(max_pages=200), 1847)
    with pytest.raises(OcrError) as e:
        await _read(mgr, 1847, monkeypatch, _no_real_work)
    msg = str(e.value)
    assert "1847" in msg and "200" in msg
    assert "NOT read" in msg
    assert "ocr.max_pages" in msg          # the setting, by name


@pytest.mark.anyio
async def test_exactly_the_ceiling_is_allowed(monkeypatch, _no_real_work):
    """`more than 200` means 200 passes. Off-by-one here refuses real work."""
    mgr = _mgr(_cfg(max_pages=200), 200)
    monkeypatch.setattr(om, "page_estimate", lambda *_a, **_k: (200, 200000))
    reached = {}

    async def _ok(self, *a, **k):
        reached["yes"] = True
        return ("surya", [], [], [], [])

    monkeypatch.setattr(om.OcrManager, "_read_windowed", _ok)
    monkeypatch.setattr(om.OcrManager, "_retry_degraded",
                        lambda self, *a, **k: _done(""))
    await mgr.ocr_document(b"%PDF-1.4 x", filename="ok.pdf",
                           content_type="application/pdf")
    assert reached.get("yes")


async def _done(v):
    return v


@pytest.mark.anyio
async def test_zero_means_no_ceiling(monkeypatch, _no_real_work):
    """The shipped default must not refuse anything."""
    mgr = _mgr(_cfg(max_pages=0), 100000)
    monkeypatch.setattr(om, "page_estimate", lambda *_a, **_k: (100000, 10**9))
    reached = {}

    async def _ok(self, *a, **k):
        reached["yes"] = True
        return ("surya", [], [], [], [])

    monkeypatch.setattr(om.OcrManager, "_read_windowed", _ok)
    monkeypatch.setattr(om.OcrManager, "_retry_degraded",
                        lambda self, *a, **k: _done(""))
    await mgr.ocr_document(b"%PDF-1.4 x", filename="huge.pdf",
                           content_type="application/pdf")
    assert reached.get("yes")


@pytest.mark.anyio
async def test_a_nonsense_setting_does_not_refuse_everything(monkeypatch,
                                                             _no_real_work):
    """A bad value must fail OPEN: refusing every document is worse."""
    mgr = _mgr(_cfg(max_pages="not a number"), 10)
    monkeypatch.setattr(om, "page_estimate", lambda *_a, **_k: (10, 10000))
    reached = {}

    async def _ok(self, *a, **k):
        reached["yes"] = True
        return ("surya", [], [], [], [])

    monkeypatch.setattr(om.OcrManager, "_read_windowed", _ok)
    monkeypatch.setattr(om.OcrManager, "_retry_degraded",
                        lambda self, *a, **k: _done(""))
    await mgr.ocr_document(b"%PDF-1.4 x", filename="x.pdf",
                           content_type="application/pdf")
    assert reached.get("yes")


def test_the_default_is_off():
    from codai.config import OcrConfig
    assert OcrConfig().max_pages == 0


def test_the_setting_survives_a_save_and_the_admin_api():
    """A ceiling that is not persisted is a ceiling that lasts one restart."""
    import pathlib
    cfg_src = pathlib.Path("codai/config.py").read_text()
    assert '"max_pages": self.config.ocr.max_pages,' in cfg_src
    routes = pathlib.Path("codai/admin/routes.py").read_text()
    assert '"max_pages": c.ocr.max_pages,' in routes
    assert 'if "max_pages" in d:' in routes
    html = pathlib.Path("codai/admin/templates/settings.html").read_text()
    assert "s-ocr-max-pages" in html
    assert "max_pages: parseInt" in html
