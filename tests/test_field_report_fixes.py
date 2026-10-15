"""The 15 Oct 2026 field report, item by item.

Every test here is named after what a client measured in production, so that a
regression says which report item came back rather than which function changed.
"""

import pathlib
import sys

import pytest
from fastapi import HTTPException

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ===================================================================== P1-b
# `/v1/runpod/scale` answered 404 "'<model>' has no runpod block" for models
# that /v1/runpod/status was reporting with ready pods — and a teardown blocked
# for ten minutes before saying it. Both symptoms are one cause: /v1/runpod is
# not in the gateway's local list, so a control-plane call carrying a `model`
# field was resolved to that model's pod and FORWARDED there. Asking for six
# pods back rented a seventh to ask on, and the 404 came from the pod, which
# has no runpod configuration of its own.
def test_the_runpod_control_plane_is_never_forwarded():
    from codai.api.remote_gateway import resolve_target
    body = b'{"model":"datalab-to/surya-ocr-2","pods":0,"teardown":true}'
    for path in ("/v1/runpod/scale", "/v1/runpod/status", "/v1/runpod/release",
                 "/v1/runpod/scale/abc123"):
        assert resolve_target(path, "POST", "", body, "application/json") is None, path


def test_a_teardown_is_not_forwarded_even_with_remote_forced():
    """`force=remote` is a test lever; it must not turn a teardown into a rental."""
    from codai.api.remote_gateway import resolve_target
    body = b'{"model":"bge-m3","pods":0,"teardown":true}'
    assert resolve_target("/v1/runpod/scale", "POST", "", body,
                          "application/json", force="remote") is None


# ===================================================================== P2-a
# `bge-m3` and `BAAI/bge-m3` each rented their own A40: the pool key was
# whichever name the request carried.
def test_an_alias_and_its_canonical_id_share_one_pool(monkeypatch):
    from codai.api import runpod_worker as rw
    import codai.models.manager as mgr

    entry = {"id": "bge-m3", "path": "BAAI/bge-m3"}
    monkeypatch.setattr(mgr, "_model_entry_for",
                        lambda name: entry if name in ("bge-m3", "BAAI/bge-m3") else None,
                        raising=False)
    mcfg = type("M", (), {"pool": ""})()
    a = rw.shared_pool_key(mcfg, "bge-m3", "BAAI/bge-m3")
    b = rw.shared_pool_key(mcfg, "BAAI/bge-m3", "BAAI/bge-m3")
    assert a == b


def test_a_model_with_no_entry_keeps_its_own_name(monkeypatch):
    from codai.api import runpod_worker as rw
    import codai.models.manager as mgr
    monkeypatch.setattr(mgr, "_model_entry_for", lambda name: None, raising=False)
    mcfg = type("M", (), {"pool": ""})()
    assert rw.shared_pool_key(mcfg, "whatever", "") == "whatever"


def test_a_named_pool_still_wins_over_the_canonical_name(monkeypatch):
    from codai.api import runpod_worker as rw
    mcfg = type("M", (), {"pool": "shared"})()
    monkeypatch.setattr(rw, "resolve_pod_engine", lambda *a, **k: "coderai")
    assert rw.shared_pool_key(mcfg, "anything", "") == "pool:shared"


# ===================================================================== P1-a
# surya returned 200 with empty pages after a lost lease. Three separate
# failures had to line up, and each is pinned here.
class _Page:
    def __init__(self, index=0, text="", meta=None):
        self.index, self.text, self.meta = index, text, meta or {}


def test_a_page_with_ink_and_no_text_is_degraded_not_accepted():
    """We measured the ink ourselves and said there is something on the page."""
    from codai.ocr.manager import OcrManager
    page = _Page(text="", meta={"page_ink": {"ink": 0.2, "blank": False,
                                             "why": "there IS something on it"}})
    assert OcrManager._page_degraded(page) == "unread"


def test_a_page_the_engine_never_answered_for_is_degraded():
    from codai.ocr.manager import OcrManager
    page = _Page(text="", meta={"engine_empty": {"calls": 4}})
    assert OcrManager._page_degraded(page) == "unread"


def test_a_genuinely_blank_page_is_left_alone():
    """Re-reading it on a second engine spends a card to confirm an empty page."""
    from codai.ocr.manager import OcrManager
    page = _Page(text="", meta={"page_ink": {"ink": 0.001, "blank": True}})
    assert OcrManager._page_degraded(page) == ""


def test_a_page_that_read_fine_is_not_degraded():
    from codai.ocr.manager import OcrManager
    assert OcrManager._page_degraded(_Page(text="full of text")) == ""


def test_unread_pages_are_listed_separately_from_blank_ones():
    from codai.ocr.manager import _unread_pages
    pages = [
        _Page(0, "", {"page_ink": {"ink": 0.2, "blank": False, "why": "ink"}}),
        _Page(1, "", {"page_ink": {"ink": 0.001, "blank": True}}),
        _Page(2, "read"),
    ]
    assert [p["page"] for p in _unread_pages(pages)] == [0]


def test_a_wholly_unread_document_is_not_a_200():
    """A 200 with no text is indistinguishable from a blank document, and a
    client that treats an empty read as an unreadable scan writes it off."""
    from codai.api.ocr import _answer
    cfg = type("C", (), {"ocr": type("O", (), {"unread_status": 502})()})()
    out = _answer({"text": "", "num_pages": 2,
                   "unread_pages": [{"page": 0, "why": "ink"},
                                    {"page": 1, "why": "ink"}]}, cfg)
    assert getattr(out, "status_code", 200) == 502


def test_a_partially_read_document_is_still_a_200_with_the_list():
    from codai.api.ocr import _answer
    cfg = type("C", (), {"ocr": type("O", (), {"unread_status": 502})()})()
    out = _answer({"text": "page two read fine", "num_pages": 2,
                   "unread_pages": [{"page": 0, "why": "ink"}]}, cfg)
    assert isinstance(out, dict) and out["unread_pages"]


def test_the_old_behaviour_is_one_setting_away():
    from codai.api.ocr import _answer
    cfg = type("C", (), {"ocr": type("O", (), {"unread_status": 0})()})()
    out = _answer({"text": "", "unread_pages": [{"page": 0, "why": "ink"}]}, cfg)
    assert isinstance(out, dict)


def test_a_clean_document_is_untouched():
    from codai.api.ocr import _answer
    cfg = type("C", (), {"ocr": type("O", (), {"unread_status": 502})()})()
    doc = {"text": "all good", "num_pages": 1}
    assert _answer(doc, cfg) is doc


# ===================================================================== P2-c
def test_a_short_trailing_chunk_can_be_folded_in():
    from codai.api import chunking as ch
    body = ("La corte ritiene il ricorso fondato ai sensi dell'art. 360 c.p.c. " * 9)
    text = f"MOTIVI DELLA DECISIONE\n\n{body}\n\nP.Q.M.\n\nRigetta."
    plain = ch.chunk_document(text, chunk_tokens=800, overlap_tokens=120)
    assert [c["section"] for c in plain["chunks"]] == ["motivazione", "dispositivo"]
    merged = ch.chunk_document(text, chunk_tokens=800, overlap_tokens=120,
                               min_chunk_chars=200)
    assert len(merged["chunks"]) == 1


def test_an_absorbed_section_label_is_reported_never_dropped():
    """`conclusioni` is what a party ASKED FOR and `dispositivo` is what the
    court DECIDED; a merge that hid the label would surface one as the other."""
    from codai.api import chunking as ch
    body = ("La corte ritiene il ricorso fondato ai sensi dell'art. 360 c.p.c. " * 9)
    text = f"MOTIVI DELLA DECISIONE\n\n{body}\n\nP.Q.M.\n\nRigetta."
    merged = ch.chunk_document(text, chunk_tokens=800, overlap_tokens=120,
                               min_chunk_chars=200)
    assert merged["absorbed"][0]["section"] == "dispositivo"
    assert "dispositivo" in merged["chunks"][0]["absorbed_sections"]


def test_merging_is_off_unless_asked():
    """It MOVES a boundary, and a re-chunked corpus whose boundaries moved
    cannot be compared with the index it replaces."""
    from codai.api import chunking as ch
    r = ch.chunk_document("MOTIVI DELLA DECISIONE\n\nx.\n\nP.Q.M.\n\ny.")
    assert r["params"]["min_chunk_chars"] == 0
    assert "absorbed" not in r


def test_a_decorated_heading_sections_under_it_legal_but_not_strict():
    """The ONE deliberate difference between the two templates, pinned.

    It is the SINGLE-WORD anchors that differ (`fatto`, `premessa`,
    `dispositivo`), not the abbreviations: `**P.Q.M.**` sections under both,
    because that pattern is not line-anchored. OCR engines that emit markdown
    produce decorated headings of both kinds."""
    from codai.api import chunking as ch
    text = ("MOTIVI DELLA DECISIONE\n\nIl ricorso e fondato.\n\n"
            "**DISPOSITIVO**\n\nRigetta.")
    loose = ch.chunk_document(text, structure="it-legal")
    strict = ch.chunk_document(text, structure="it-legal-strict")
    assert [c["section"] for c in loose["chunks"]] == ["motivazione", "dispositivo"]
    assert [c["section"] for c in strict["chunks"]] == ["motivazione"]


# ===================================================================== P3-b
@pytest.mark.anyio
async def test_a_read_that_dribbles_is_stopped_by_the_server():
    """An httpx read timeout is per-READ, so a response trickling bytes never
    trips it: one chandra call stayed open nineteen minutes against timeout=420."""
    import asyncio
    from codai.api.ocr import _bounded
    from codai.ocr.base import OcrError

    async def _forever():
        await asyncio.sleep(10)

    cfg = type("C", (), {"ocr": type("O", (), {"request_timeout_s": 0.05})()})()
    with pytest.raises(OcrError) as err:
        await _bounded(cfg, _forever())
    assert err.value.status == 504


@pytest.mark.anyio
async def test_no_ceiling_configured_means_no_ceiling():
    from codai.api.ocr import _bounded
    cfg = type("C", (), {"ocr": type("O", (), {"request_timeout_s": 0})()})()

    async def _quick():
        return {"ok": True}

    assert (await _bounded(cfg, _quick()))["ok"]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_an_abbreviated_heading_sections_under_both_templates():
    """Pinned so the strict template is not "fixed" into matching less: the
    difference is the anchored single words, and nothing else."""
    from codai.api import chunking as ch
    text = "MOTIVI DELLA DECISIONE\n\nIl ricorso e fondato.\n\n**P.Q.M.**\n\nRigetta."
    for tpl in ("it-legal", "it-legal-strict"):
        r = ch.chunk_document(text, structure=tpl)
        assert [c["section"] for c in r["chunks"]] == ["motivazione", "dispositivo"], tpl


# ===================================================================== store
# Getting an adapter from a client into a served name had no API: the last step
# was editing models.json by hand.
def _catalogue(monkeypatch, models=None, saved=None):
    import codai.admin.routes as routes

    class _CM:
        def __init__(self):
            self.models_data = {"text_models": list(models or [])}

        def save_models(self):
            (saved if saved is not None else []).append(True)

    cm = _CM()
    monkeypatch.setattr(routes, "config_manager", cm, raising=False)
    return cm


BASE_MODEL = {"id": "base-awq", "alias": "base", "path": "vendor/Some-27B",
              "backend": "runpod",
              "runpod": {"pool": "shared", "served_model": "base-awq",
                         "hf_repo": "vendor/Some-27B", "engine": "coderai-vllm",
                         "max_hourly_usd": 1.2, "min_pods": 1, "keep_warm": True}}


@pytest.mark.anyio
async def test_an_uploaded_adapter_becomes_a_servable_name(monkeypatch):
    from codai.api import loras
    saved = []
    cm = _catalogue(monkeypatch, [dict(BASE_MODEL)], saved)
    monkeypatch.setattr(loras, "lora_blob_exists", lambda h: True)
    monkeypatch.setattr("codai.api.models_transfer.register_model_runtime",
                        lambda e, s: "added")

    out = await loras.register_lora(loras.LoraRegisterRequest(
        name="tuned-alpha", base="base-awq", adapter="sha256:abc"))
    assert out["state"] == "registered" and saved
    entry = cm.models_data["text_models"][-1]
    assert entry["id"] == "tuned-alpha"
    assert entry["loras"] == [{"name": "tuned-alpha", "path": "sha256:abc",
                               "weight": 1.0}]


@pytest.mark.anyio
async def test_it_joins_the_base_pool_so_no_second_card_is_rented(monkeypatch):
    """The whole economy: the base weights are already on a card."""
    from codai.api import loras
    cm = _catalogue(monkeypatch, [dict(BASE_MODEL)])
    monkeypatch.setattr(loras, "lora_blob_exists", lambda h: True)
    monkeypatch.setattr("codai.api.models_transfer.register_model_runtime",
                        lambda e, s: "added")
    out = await loras.register_lora(loras.LoraRegisterRequest(
        name="tuned-alpha", base="base", adapter="sha256:abc"))
    rp = cm.models_data["text_models"][-1]["runpod"]
    assert out["pool"] == "shared" and rp["pool"] == "shared"
    assert rp["served_model"] == "tuned-alpha"
    assert rp["max_hourly_usd"] == 1.2          # the base's budget, inherited
    # NOT the base's warm floor: an adapter rides the base's pod, it does not
    # ask for one of its own.
    assert "min_pods" not in rp and "keep_warm" not in rp


@pytest.mark.anyio
async def test_bytes_we_do_not_hold_are_refused_before_anything_is_saved(monkeypatch):
    """A pod sent an id nobody holds serves the BASE instead, silently."""
    from codai.api import loras
    saved = []
    _catalogue(monkeypatch, [dict(BASE_MODEL)], saved)
    monkeypatch.setattr(loras, "lora_blob_exists", lambda h: False)
    with pytest.raises(HTTPException) as err:
        await loras.register_lora(loras.LoraRegisterRequest(
            name="tuned-alpha", base="base-awq", adapter="sha256:missing"))
    assert err.value.status_code == 404
    assert "/v1/loras/upload" in str(err.value.detail)
    assert not saved


@pytest.mark.anyio
async def test_an_unknown_base_is_refused(monkeypatch):
    from codai.api import loras
    _catalogue(monkeypatch, [dict(BASE_MODEL)])
    monkeypatch.setattr(loras, "lora_blob_exists", lambda h: True)
    with pytest.raises(HTTPException) as err:
        await loras.register_lora(loras.LoraRegisterRequest(
            name="x", base="nope", adapter="sha256:abc"))
    assert err.value.status_code == 404


@pytest.mark.anyio
async def test_retraining_replaces_the_adapter_rather_than_duplicating_it(monkeypatch):
    from codai.api import loras
    cm = _catalogue(monkeypatch, [dict(BASE_MODEL)])
    monkeypatch.setattr(loras, "lora_blob_exists", lambda h: True)
    monkeypatch.setattr("codai.api.models_transfer.register_model_runtime",
                        lambda e, s: "added")
    for blob in ("sha256:v1", "sha256:v2"):
        out = await loras.register_lora(loras.LoraRegisterRequest(
            name="tuned-alpha", base="base-awq", adapter=blob))
    assert out["state"] == "updated"
    assert len(cm.models_data["text_models"]) == 2
    assert cm.models_data["text_models"][-1]["loras"][0]["path"] == "sha256:v2"


@pytest.mark.anyio
async def test_a_real_model_is_never_overwritten_by_an_adapter(monkeypatch):
    from codai.api import loras
    _catalogue(monkeypatch, [dict(BASE_MODEL)])
    monkeypatch.setattr(loras, "lora_blob_exists", lambda h: True)
    with pytest.raises(HTTPException) as err:
        await loras.register_lora(loras.LoraRegisterRequest(
            name="base-awq", base="base-awq", adapter="sha256:abc"))
    assert err.value.status_code == 409


def test_the_adapter_store_is_never_forwarded():
    """An upload must land in THIS blob store: it is where a pod is sent its
    adapters from, and the id we hand out has to resolve here."""
    from codai.api.remote_gateway import resolve_target
    for path in ("/v1/loras/upload", "/v1/loras/register",
                 "/v1/loras/blob/abc", "/v1/loras"):
        assert resolve_target(path, "POST", "", b'{"name":"x","base":"y"}',
                              "application/json") is None, path
