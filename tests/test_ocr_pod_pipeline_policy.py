# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Going local must be a DECISION, and it must be audible.

Two defects, found together on a production run that had already been fixed
three times:

1. The gate compared the request's engine against the three VLM engine names.
   `/v1/ocr` has always also accepted the MODEL a caller knows about in place
   of the engine driving it -- `resolve_engine` maps the Surya-2 checkpoint to
   'surya', and the catalogue advertises those ids as aliases -- so a caller
   naming the model got no pod-side pipeline at all.

2. Every way of ending up on the local pipeline that was not "a pod said no"
   returned "" or None in SILENCE. The log of the run that prompted this held
   8,278 successful pod reads and not one line explaining why the next four
   thousand documents were rasterised on a GPU-less orchestrator that climbed
   0.8 GB/min to 36 GB while sixteen rented A40s idled.

So: resolve the engine the way a read resolves it, say which way the decision
went every time it is not the pod, and prefer waiting for a pod over taking
the work on.
"""

import asyncio
import types

import pytest

import codai.ocr.manager as om


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _forget_everything_said():
    om._PIPELINE_GATE_SAID.clear()
    om._SAID_AT.clear()
    om._POD_PIPELINE_UNUSABLE.clear()
    om._POD_WAIT_COOLDOWN.clear()
    yield
    om._PIPELINE_GATE_SAID.clear()
    om._SAID_AT.clear()
    om._POD_PIPELINE_UNUSABLE.clear()
    om._POD_WAIT_COOLDOWN.clear()


OCR_IMAGE = "ghcr.io/nextime/coderai-vllm-ocr:latest"


def _cfg(**kw):
    base = {"default_engine": "surya", "surya_serve": "model",
            "surya_model_id": "datalab-to/surya-ocr-2",
            "chandra_serve": "model",
            "chandra_model_id": "datalab-to/chandra-ocr-2",
            "olmocr_serve": "model",
            "olmocr_model_id": "allenai/olmOCR-2-7B-1025-FP8",
            "max_concurrency": 8, "page_window": 4,
            "pod_pipeline": "prefer", "pod_pipeline_wait_s": 60.0,
            # EXPLICIT, not incidental: these tests stub the SYNC transport
            # (`_post_page_to_pod`, `pod_http.post`), so they must say which
            # path they are exercising. When the async default was introduced
            # they silently stopped testing their own stubs and started dialling
            # 10.0.0.1 for real -- 300 s of nothing instead of a failure. A test
            # that quietly measures a different thing is worse than one that
            # breaks.
            "pod_http_async": False}
    base.update(kw)
    return types.SimpleNamespace(**base)


def _ocr_image_entry(monkeypatch):
    from codai.api import runpod_worker as rw
    monkeypatch.setattr(rw, "_model_entry", lambda _mid: {
        "id": "m", "backend": "runpod",
        "runpod": {"mode": "pods", "image": OCR_IMAGE}})


# ---------------------------------------------------------------------------
# 1. The engine name resolves the way a READ resolves it
# ---------------------------------------------------------------------------

def test_the_model_id_names_the_engine_just_like_everywhere_else(monkeypatch):
    """THE defect: documented, accepted by /v1/ocr, and fatal only here."""
    _ocr_image_entry(monkeypatch)
    got = om.pod_pipeline_model(_cfg(), "datalab-to/surya-ocr-2")
    assert got == "datalab-to/surya-ocr-2"


def test_the_model_id_is_matched_case_insensitively(monkeypatch):
    _ocr_image_entry(monkeypatch)
    assert om.pod_pipeline_model(_cfg(), "DataLab-To/Surya-OCR-2")


def test_every_vlm_engine_can_be_named_by_its_model(monkeypatch):
    _ocr_image_entry(monkeypatch)
    cfg = _cfg()
    assert om.pod_pipeline_model(cfg, "datalab-to/chandra-ocr-2") == \
        "datalab-to/chandra-ocr-2"
    assert om.pod_pipeline_model(cfg, "allenai/olmOCR-2-7B-1025-FP8") == \
        "allenai/olmOCR-2-7B-1025-FP8"


def test_resolving_an_alias_is_said_out_loud(monkeypatch, capsys):
    """Proving the fix in a live log needs the alias to be visible."""
    _ocr_image_entry(monkeypatch)
    om.pod_pipeline_model(_cfg(), "datalab-to/surya-ocr-2")
    out = capsys.readouterr().out
    assert "names the 'surya' engine" in out


def test_the_engine_name_itself_still_works(monkeypatch):
    _ocr_image_entry(monkeypatch)
    assert om.pod_pipeline_model(_cfg(), "surya") == "datalab-to/surya-ocr-2"


# ---------------------------------------------------------------------------
# 2. Every silent route now says which it was
# ---------------------------------------------------------------------------

def test_an_engine_with_no_pod_pipeline_says_so(monkeypatch, capsys):
    _ocr_image_entry(monkeypatch)
    assert om.pod_pipeline_model(_cfg(), "doctr") == ""
    out = capsys.readouterr().out
    assert "no pod-side pipeline" in out
    assert "THIS BOX rasterises" in out


def test_an_unknown_engine_name_says_so(monkeypatch, capsys):
    """The shape a model id used to take: unrecognised, silent, local."""
    _ocr_image_entry(monkeypatch)
    assert om.pod_pipeline_model(_cfg(), "some-model-nobody-serves") == ""
    out = capsys.readouterr().out
    assert "not an OCR engine or a model id" in out
    assert "some-model-nobody-serves" in out


def test_a_serve_mode_that_is_not_model_says_so_and_says_the_mode(
        monkeypatch, capsys):
    _ocr_image_entry(monkeypatch)
    assert om.pod_pipeline_model(_cfg(surya_serve="local"), "surya") == ""
    out = capsys.readouterr().out
    assert "serve mode is 'local'" in out
    assert "surya_serve" in out


def test_an_empty_model_id_says_so(monkeypatch, capsys):
    _ocr_image_entry(monkeypatch)
    assert om.pod_pipeline_model(_cfg(surya_model_id=""), "surya") == ""
    out = capsys.readouterr().out
    assert "surya_model_id is " in out


def test_the_reasons_are_rate_limited_not_repeated_per_document(
        monkeypatch, capsys):
    _ocr_image_entry(monkeypatch)
    for _ in range(50):
        om.pod_pipeline_model(_cfg(surya_serve="local"), "surya")
    assert capsys.readouterr().out.count("serve mode is") == 1


def test_a_standing_refusal_keeps_saying_this_box_rasterises(monkeypatch, capsys):
    """A cooldown is minutes; a run is hours. Silence in between looked like
    success -- the documents read here are indistinguishable from the ones the
    pods read unless something says so."""
    om._POD_PIPELINE_UNUSABLE["m"] = (9e18, "the pod answered 503: no model")
    mgr = om.OcrManager()
    out = asyncio.run(mgr._read_on_pods(
        b"%PDF-1.4 x", "d.pdf", "application/pdf", "m", dpi=200,
        page_opts=None, detect=None, cfg=_cfg(), engine_name="surya"))
    assert out is None
    said = capsys.readouterr().out
    assert "still not asking the pods" in said
    assert "no model" in said                      # the pod's own words


# ---------------------------------------------------------------------------
# 3. Try not to go local
# ---------------------------------------------------------------------------

class _LocalLease:
    kind = "local"
    base = "http://127.0.0.1:8000/v1"
    api_key = ""

    def release(self):
        pass


class _PodLease:
    kind = "pod"
    base = "https://10.0.0.1:1234/v1"
    api_key = "k"

    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def _placement(monkeypatch, leases):
    """`choose` hands back each lease in turn, repeating the last for ever."""
    from codai.models import placement
    seen = []

    def _choose(*_a, **_k):
        got = leases[min(len(seen), len(leases) - 1)]
        seen.append(got)
        return got() if callable(got) else got

    monkeypatch.setattr(placement, "choose", _choose)
    monkeypatch.setattr(placement, "spill_threshold", lambda *_a, **_k: 0)
    monkeypatch.setattr(placement, "why_no_candidates",
                        lambda *_a, **_k: "the pool is at its ceiling")
    return seen


def _one_page(monkeypatch):
    monkeypatch.setattr(om, "iter_pdf_page_pdfs", lambda _d: [b"%PDF-page0"])


def _page_read(monkeypatch):
    reply = {"engine": "surya", "pages": [{"text": "ok"}]}

    async def _aok(*_a, **_k):
        return reply
    monkeypatch.setattr(om, "_post_page_to_pod", lambda *_a, **_k: reply)
    monkeypatch.setattr(om, "_apost_page_to_pod", _aok)


async def _read(cfg):
    mgr = om.OcrManager()
    return await mgr._read_on_pods(b"%PDF-1.4 x", "d.pdf", "application/pdf",
                                   "m", dpi=200, page_opts=None, detect=None,
                                   cfg=cfg, engine_name="surya")


@pytest.mark.anyio
async def test_prefer_waits_and_takes_the_pod_that_turns_up(monkeypatch, capsys):
    """A pool with nothing free is usually a pool whose pods are booting."""
    _one_page(monkeypatch)
    _page_read(monkeypatch)
    _placement(monkeypatch, [_LocalLease(), _PodLease()])
    got = await _read(_cfg(pod_pipeline="prefer", pod_pipeline_wait_s=5.0))
    assert got is not None, "waited, a pod appeared, and it still went local"
    assert "became free after" in capsys.readouterr().out


@pytest.mark.anyio
async def test_prefer_falls_back_when_the_wait_runs_out_and_says_so(
        monkeypatch, capsys):
    """Bounded: an optimisation that cannot run must cost latency, not pages."""
    _one_page(monkeypatch)
    _page_read(monkeypatch)
    _placement(monkeypatch, [_LocalLease()])
    got = await _read(_cfg(pod_pipeline="prefer", pod_pipeline_wait_s=0.2))
    assert got is None
    out = capsys.readouterr().out
    assert "waited" in out and "THIS BOX" in out
    assert "memory" in out


@pytest.mark.anyio
async def test_a_timed_out_wait_is_not_paid_again_by_the_next_page(monkeypatch):
    """60s per page is a tax, not a ceiling: once a wait fails, pages go
    straight through for a cooldown."""
    _one_page(monkeypatch)
    _page_read(monkeypatch)
    _placement(monkeypatch, [_LocalLease()])
    cfg = _cfg(pod_pipeline="prefer", pod_pipeline_wait_s=0.3)
    await _read(cfg)
    loop = asyncio.get_running_loop()
    started = loop.time()
    await _read(cfg)
    assert loop.time() - started < 0.2, "the second document waited again"


@pytest.mark.anyio
async def test_require_refuses_the_page_rather_than_rasterising_here(
        monkeypatch):
    _one_page(monkeypatch)
    _page_read(monkeypatch)
    _placement(monkeypatch, [_LocalLease()])
    with pytest.raises(om.OcrError) as exc:
        await _read(_cfg(pod_pipeline="require", pod_pipeline_wait_s=0.1))
    assert exc.value.status == 503
    assert "will not" in str(exc.value) and "rasterise" in str(exc.value)
    assert "ceiling" in str(exc.value)             # WHY, not just "no"


@pytest.mark.anyio
async def test_auto_keeps_the_old_immediate_fallback(monkeypatch):
    _one_page(monkeypatch)
    _page_read(monkeypatch)
    _placement(monkeypatch, [_LocalLease()])
    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await _read(_cfg(pod_pipeline="auto", pod_pipeline_wait_s=30)) is None
    assert loop.time() - started < 0.2


@pytest.mark.anyio
async def test_auto_still_says_which_way_it_went(monkeypatch, capsys):
    _one_page(monkeypatch)
    _page_read(monkeypatch)
    _placement(monkeypatch, [_LocalLease()])
    await _read(_cfg(pod_pipeline="auto"))
    out = capsys.readouterr().out
    assert "placement answered 'local'" in out
    assert "at its ceiling" in out                 # the REASON, from placement


@pytest.mark.anyio
async def test_a_pod_on_the_first_ask_never_waits(monkeypatch):
    _one_page(monkeypatch)
    _page_read(monkeypatch)
    _placement(monkeypatch, [_PodLease()])
    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await _read(_cfg()) is not None
    assert loop.time() - started < 0.2


@pytest.mark.anyio
async def test_the_lease_is_always_returned(monkeypatch):
    """A lease leaked on the local path shrinks the pool for good."""
    _one_page(monkeypatch)
    _page_read(monkeypatch)
    lease = _PodLease()
    _placement(monkeypatch, [lease])
    await _read(_cfg())
    assert lease.released


def test_an_unknown_policy_falls_back_to_the_safe_one():
    """A typo in a config file must not turn the memory guard off."""
    assert om._pod_pipeline_policy(_cfg(pod_pipeline="wait-forever"))[0] == "prefer"
    assert om._pod_pipeline_policy(types.SimpleNamespace())[0] == "prefer"
    # The words an operator might reasonably write for "don't wait".
    assert om._pod_pipeline_policy(_cfg(pod_pipeline="off"))[0] == "auto"


def test_the_shipped_default_prefers_a_pod():
    from codai.config import OcrConfig
    assert OcrConfig().pod_pipeline == "prefer"
    assert OcrConfig().pod_pipeline_wait_s > 0


# ---------------------------------------------------------------------------
# 4. A dead socket is not a verdict on the image
#
# THE defect this file was written for. Twelve times in four hours a pooled
# keep-alive connection a pod had closed while idle failed on its next use:
#
#   pod OCR at https://194.68.245.122:22005/v1/ocr is unreachable:
#   ('Connection aborted.', RemoteDisconnected('Remote end closed connection
#   without response'))
#
# requests raised, `_post_page_to_pod` turned it into OcrError(503), 503 is in
# `_POD_CANNOT_RUN_PIPELINE`, and so one dead TCP connection said "these pods
# cannot run the OCR pipeline" and moved every document onto the orchestrator
# for five minutes at a time.
# ---------------------------------------------------------------------------

class _Flaky:
    """A pod transport that dies the first n times, then answers."""

    def __init__(self, deaths, exc=None):
        self.deaths = deaths
        self.calls = 0
        self.exc = exc or Exception(
            "('Connection aborted.', RemoteDisconnected('Remote end closed "
            "connection without response'))")

    def __call__(self, *_a, **_k):
        self.calls += 1
        if self.calls <= self.deaths:
            raise self.exc
        return types.SimpleNamespace(
            status_code=200, text="",
            json=lambda: {"engine": "surya", "pages": [{"text": "ok"}]})


def _transport(monkeypatch, fn):
    from codai.api import pod_http
    monkeypatch.setattr(pod_http, "post", fn)
    monkeypatch.setattr(pod_http, "get", lambda *_a, **_k:
                        types.SimpleNamespace(status_code=200))


def test_a_dead_pooled_socket_is_retried_on_a_fresh_one(monkeypatch):
    """Nothing was sent, so a second attempt costs a connect and no work."""
    flaky = _Flaky(deaths=1)
    _transport(monkeypatch, flaky)
    got = om._post_page_to_pod("https://10.0.0.1:1234/v1", "k", b"%PDF", "0.pdf",
                               200, {}, None, "surya")
    assert got["engine"] == "surya"
    assert flaky.calls == 2


def test_a_timeout_is_never_retried(monkeypatch):
    """The pod may be ninety seconds into a 130-second read; sending the page
    again would double the work on a card already doing it."""
    flaky = _Flaky(deaths=99, exc=Exception("HTTPSConnectionPool: Read timed out"))
    _transport(monkeypatch, flaky)
    with pytest.raises(om.PodUnreachable):
        om._post_page_to_pod("https://10.0.0.1:1234/v1", "k", b"%PDF", "0.pdf",
                             200, {}, None, "surya")
    assert flaky.calls == 1


def test_an_unreachable_pod_raises_the_transport_error_not_an_answer(monkeypatch):
    _transport(monkeypatch, _Flaky(deaths=99))
    with pytest.raises(om.PodUnreachable) as exc:
        om._post_page_to_pod("https://10.0.0.1:1234/v1", "k", b"%PDF", "0.pdf",
                             200, {}, None, "surya")
    # Still a 503 to anything that only reads the status -- but a DIFFERENT
    # type, which is what stops it being read as "this image cannot do OCR".
    assert exc.value.status == 503
    assert isinstance(exc.value, om.OcrError)


def test_a_pod_that_answers_503_is_still_a_refusal(monkeypatch):
    """The other side of the line: an ANSWER may condemn the image."""
    _transport(monkeypatch, lambda *_a, **_k: types.SimpleNamespace(
        status_code=503, text="model is not served over HTTP by this install",
        json=lambda: {}))
    with pytest.raises(om.OcrError) as exc:
        om._post_page_to_pod("https://10.0.0.1:1234/v1", "k", b"%PDF", "0.pdf",
                             200, {}, None, "surya")
    assert not isinstance(exc.value, om.PodUnreachable)
    assert exc.value.status == 503


@pytest.mark.anyio
async def test_an_unreachable_pod_does_not_set_the_whole_pool_aside(monkeypatch):
    """THE five-minute outage, twelve times over: one dead socket must not take
    the pod-side pipeline away from every document that follows."""
    _one_page(monkeypatch)
    _placement(monkeypatch, [_PodLease()])
    monkeypatch.setattr(om, "_post_page_to_pod", lambda *_a, **_k: (
        _ for _ in ()).throw(om.PodUnreachable("unreachable: aborted", status=503)))
    monkeypatch.setattr(om, "_pod_still_alive", lambda *_a, **_k: True)
    assert await _read(_cfg(pod_pipeline="auto")) is None
    assert not om._POD_PIPELINE_UNUSABLE, \
        "a dead socket set the pool aside, so the next document goes local too"


@pytest.mark.anyio
async def test_a_page_whose_socket_died_is_tried_on_another_pod(monkeypatch):
    _one_page(monkeypatch)
    _placement(monkeypatch, [_PodLease()])
    seen = []

    def _post(base, *_a, **_k):
        seen.append(base)
        if len(seen) < 2:
            raise om.PodUnreachable("unreachable: aborted", status=503)
        return {"engine": "surya", "pages": [{"text": "ok"}]}

    monkeypatch.setattr(om, "_post_page_to_pod", _post)
    monkeypatch.setattr(om, "_pod_still_alive", lambda *_a, **_k: True)
    got = await _read(_cfg(pod_pipeline="auto"))
    assert got is not None, "gave the document to the local pipeline instead"
    assert len(seen) == 2


@pytest.mark.anyio
async def test_a_pod_answering_healthz_is_not_dropped_from_the_pool(monkeypatch):
    """Killing a pod on a reused idle connection cost seven healthy A40s once."""
    _one_page(monkeypatch)
    lease = _PodLease()
    lease.declared_dead = False

    def _dead():
        lease.declared_dead = True
    lease.dead = _dead
    _placement(monkeypatch, [lease])
    monkeypatch.setattr(om, "_post_page_to_pod", lambda *_a, **_k: (
        _ for _ in ()).throw(om.PodUnreachable("unreachable: aborted", status=503)))
    monkeypatch.setattr(om, "_pod_still_alive", lambda *_a, **_k: True)
    await _read(_cfg(pod_pipeline="auto"))
    assert not lease.declared_dead


@pytest.mark.anyio
async def test_a_pod_that_answers_nothing_at_all_is_dropped(monkeypatch):
    """The opposite case, and the reason the probe exists rather than a guess."""
    _one_page(monkeypatch)
    lease = _PodLease()
    lease.declared_dead = False

    def _dead():
        lease.declared_dead = True
    lease.dead = _dead
    _placement(monkeypatch, [lease])
    monkeypatch.setattr(om, "_post_page_to_pod", lambda *_a, **_k: (
        _ for _ in ()).throw(om.PodUnreachable("unreachable: aborted", status=503)))
    monkeypatch.setattr(om, "_pod_still_alive", lambda *_a, **_k: False)
    await _read(_cfg(pod_pipeline="auto"))
    assert lease.declared_dead


@pytest.mark.anyio
async def test_falling_back_on_unreachable_pods_says_the_pool_is_not_set_aside(
        monkeypatch, capsys):
    _one_page(monkeypatch)
    _placement(monkeypatch, [_PodLease()])
    monkeypatch.setattr(om, "_post_page_to_pod", lambda *_a, **_k: (
        _ for _ in ()).throw(om.PodUnreachable(
            "pod OCR at https://10.0.0.1:1234/v1/ocr is unreachable: "
            "('Connection aborted.', RemoteDisconnected(...))", status=503)))
    monkeypatch.setattr(om, "_pod_still_alive", lambda *_a, **_k: True)
    await _read(_cfg(pod_pipeline="auto"))
    out = capsys.readouterr().out
    assert "could not be sent to any pod" in out
    assert "NOT set aside" in out
    assert "Connection aborted" in out             # the transport's own words


def test_an_active_cooldown_is_never_extended_silently():
    """09:10 to 09:26 with nothing in the log: refusals during a cooldown kept
    pushing the expiry out while only the first one printed."""
    assert om._set_aside_pod_pipeline("m", "first") is True
    first = om._POD_PIPELINE_UNUSABLE["m"][0]
    assert om._set_aside_pod_pipeline("m", "second") is False
    assert om._POD_PIPELINE_UNUSABLE["m"][0] == first
    assert om._POD_PIPELINE_UNUSABLE["m"][1] == "first"


def test_the_hint_list_recognises_what_production_actually_logged():
    msg = ("('Connection aborted.', RemoteDisconnected('Remote end closed "
           "connection without response'))")
    assert om._looks_like_a_dead_socket(Exception(msg))
    assert om._looks_like_a_dead_socket(ConnectionResetError("Connection reset by peer"))
    assert not om._looks_like_a_dead_socket(Exception("Read timed out. (read timeout=600)"))
    assert not om._looks_like_a_dead_socket(Exception("that is not a PDF"))
