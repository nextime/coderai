# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Dedicated OCR endpoint.

POST /v1/ocr — OCR an uploaded image or PDF with a purpose-built OCR engine
(PaddleOCR / docTR / Surya), returning faithful text plus per-line boxes and layout.
Those three are NOT vision LLMs. Two document VLMs are available for pages the
detector-based engines read badly: ``engine=olmocr`` (olmOCR-2, no boxes at all) and
``engine=chandra`` (Chandra, which is asked for layout and so returns boxes and labelled
regions as well). Structured-JSON extraction and stamp/signature detection are layered
on in later milestones (fields already accepted, no-op until then).
"""

import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from codai.ocr.base import OcrError
from codai.ocr.manager import ocr_manager
from codai.ocr.schemas import schema_store

router = APIRouter()

# Maximum upload size: 100 MB (multi-page scanned PDFs can be large).
_MAX_OCR_BYTES = 100 * 1024 * 1024

global_args = None


def set_global_args(args):
    """Set global args from coderai."""
    global global_args
    global_args = args


def _ocr_config():
    """Return the live OcrConfig from the admin config manager (or None)."""
    try:
        from codai.admin.routes import config_manager
        if config_manager is not None and getattr(config_manager, "config", None) is not None:
            return config_manager.config.ocr
    except Exception:
        pass
    return None


def _page_opts(max_tokens, max_continuations) -> dict:
    """Per-request engine overrides, or {} when the caller said nothing.

    One global cap cannot serve a public deployment: a four-page contract and a
    fifty-page judgment need different budgets, and only the caller knows which
    it is sending. Omitted means "use the configured value" -- NOT zero, which
    is a meaningful setting for both of these.
    """
    out = {}
    if max_tokens is not None:
        out["max_tokens"] = max(0, int(max_tokens))
    if max_continuations is not None:
        out["max_continuations"] = max(0, int(max_continuations))
    return out


def _refuse_if_full(cfg) -> None:
    """429 when the queue that keeps the pods fed is itself full.

    THE QUEUE IS DELIBERATE. When a pod finishes a page there has to be another
    page already waiting, or the fleet idles between every dispatch and is paid
    for the gap. So the server accepts `ocr.max_concurrency` pages in flight
    PLUS `ocr.max_queued_pages` waiting, and only past that does it refuse.

    What it must not do is accept without limit. A client offering several times
    the capacity does not get an error -- it gets replies that take longer and
    longer, which reads as "the pages are getting slower", not as "stop
    sending". It then does exactly the wrong thing and sends more. Measured
    2026-10-19: a client offered 522 pages at once against a fleet that could
    carry ~250, mean page latency went to 238 s against a 19 s service time,
    and nothing in any reply said the server was over its capacity.

    Refused BEFORE the upload is read, which matters twice: the client learns
    immediately instead of after transferring a document the server was never
    going to look at, and the server does not buffer megabytes it is about to
    discard. The in-flight memory of this endpoint is concurrent requests times
    upload size, and that product is the one that has OOM-killed it before.

    `Retry-After` is the actionable half. A client that honours it finishes
    sooner than one that retries immediately, because its retries stop
    competing with the work already accepted.
    """
    try:
        state = ocr_manager.admission()
    except Exception:
        return                      # never let admission accounting refuse work
    if not state.get("full"):
        return
    raise HTTPException(
        status_code=429,
        detail=(f"{state['queued']} page(s) are already queued for a slot "
                f"(ocr.max_queued_pages = {state['queue_limit']}, with "
                f"{state['running']} of {state['width']} in flight). The queue "
                f"is kept full on purpose so no pod waits for work; this "
                f"request is past it. Retry after the Retry-After interval — "
                f"pushing harder makes every accepted page slower, including "
                f"this one when it is accepted."),
        headers={"Retry-After": "2",
                 "X-OCR-Pages-Running": str(state["running"]),
                 "X-OCR-Pages-Queued": str(state["queued"]),
                 "X-OCR-Page-Width": str(state["width"]),
                 "X-OCR-Queue-Limit": str(state["queue_limit"])})


async def _bounded(cfg, coro):
    """Run an OCR read under the install's wall-clock ceiling, if it set one.

    A client's read timeout cannot bound this. httpx (and requests) time each
    READ, so a response that dribbles a byte at a time never trips one: a
    chandra read stayed open for nineteen minutes against a `timeout=420`.
    Bounding it needs a stopwatch over the whole call, and the server is the
    side that knows what it is doing.
    """
    try:
        limit = float(getattr(cfg.ocr, "request_timeout_s", 0) or 0)
    except Exception:
        limit = 0.0
    if limit <= 0:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=limit)
    except asyncio.TimeoutError:
        raise OcrError(f"the read passed the {limit:g}s ceiling for one request "
                       f"(ocr.request_timeout_s) and was stopped", status=504)


def _answer(result: dict, cfg):
    """The OCR response, with a status code that matches what we found.

    A document whose every readable page came back empty must NOT be a 200. The
    orchestrator has already measured ink on those pages and said there is
    something on them, so "no text" is this run failing, not the document being
    blank — but the two are identical in a 200 whose `text` is empty, and a
    client that treats an empty read as an unreadable scan then consumes the
    document from its queue and writes it off. That happened to roughly a
    thousand documents in one night behind a lost engine lease.

    Only the all-or-nothing case changes status: a document that produced SOME
    text is still a 200, with `unread_pages` naming the ones it did not. Set
    `ocr.unread_status` to 0 to keep the old behaviour.
    """
    try:
        status = int(getattr(cfg.ocr, "unread_status", 502) or 0)
    except Exception:
        status = 502
    unread = result.get("unread_pages") or []
    if not status or not unread:
        return result
    if (result.get("text") or "").strip():
        return result                   # partially read: 200 and a named list
    return JSONResponse(status_code=status, content={
        "error": {
            "message": (f"every page with ink on it came back empty "
                        f"({len(unread)} of {result.get('num_pages', 0)}): the "
                        f"document was not read. This is not a blank document "
                        f"— see unread_pages."),
            "type": "ocr_unread",
        },
        **result,
    })


@router.post("/v1/ocr", summary="OCR an image or PDF with a dedicated OCR engine")
async def create_ocr(
    file: UploadFile = File(...),
    engine: Optional[str] = Form(None),        # paddle|doctr|surya|olmocr|chandra (default from config)
    dpi: Optional[int] = Form(None),           # PDF rasterisation DPI
    lang: Optional[str] = Form(None),          # document language; omit or "auto" to let the engine decide
    structured: Optional[bool] = Form(False),  # structured JSON extraction via text model
    schema_: Optional[str] = Form(None, alias="schema"),  # override extraction schema (JSON string)
    detect: Optional[str] = Form(None),        # off|layout|detector|both
    max_tokens: Optional[int] = Form(None),    # cap on the model's reply for ONE page (engine default if unset)
    max_continuations: Optional[int] = Form(None),  # resumptions allowed for a page that hits the cap
):
    cfg = _ocr_config()
    if cfg is None:
        raise HTTPException(status_code=503, detail="OCR subsystem is not available")
    ocr_manager.configure(cfg)
    _refuse_if_full(cfg)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > _MAX_OCR_BYTES:
        raise HTTPException(status_code=413, detail="file too large (max 100 MB)")

    try:
        result = await _bounded(cfg, ocr_manager.ocr_document(
            data,
            filename=file.filename or "",
            content_type=file.content_type or "",
            engine=engine,
            dpi=dpi,
            detect=detect,
            structured=bool(structured),
            schema=schema_,
            lang=lang,
            page_opts=_page_opts(max_tokens, max_continuations),
        ))
        return _answer(result, cfg)
    except OcrError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")


@router.post("/v1/ocr/batch", summary="OCR many images/PDFs concurrently")
async def create_ocr_batch(
    files: List[UploadFile] = File(...),
    engine: Optional[str] = Form(None),
    dpi: Optional[int] = Form(None),
    lang: Optional[str] = Form(None),
    structured: Optional[bool] = Form(False),
    schema_: Optional[str] = Form(None, alias="schema"),
    detect: Optional[str] = Form(None),
    max_tokens: Optional[int] = Form(None),
    max_continuations: Optional[int] = Form(None),
):
    """Fan a batch of documents across the engine's instance pool. Per-file failures are
    reported inline (not fatal) so one bad scan doesn't sink the batch.

    Takes the same per-request budgets as ``/v1/ocr``, and must: a batch is how
    a caller sends real work, so a budget this endpoint did not accept was a
    budget that never applied to anything that mattered. FastAPI drops an
    unknown form field silently, so a caller could set ``max_tokens`` on every
    batch for a week and never be told it did nothing."""
    cfg = _ocr_config()
    if cfg is None:
        raise HTTPException(status_code=503, detail="OCR subsystem is not available")
    ocr_manager.configure(cfg)
    _refuse_if_full(cfg)

    payloads = []
    for f in files:
        data = await f.read()
        payloads.append((f.filename or "", f.content_type or "", data))

    async def _one(name, ctype, data):
        if not data:
            return {"filename": name, "error": "empty upload"}
        if len(data) > _MAX_OCR_BYTES:
            return {"filename": name, "error": "file too large (max 100 MB)"}
        try:
            res = await ocr_manager.ocr_document(
                data, filename=name, content_type=ctype, engine=engine, dpi=dpi,
                detect=detect, structured=bool(structured), schema=schema_, lang=lang,
                page_opts=_page_opts(max_tokens, max_continuations),
            )
            res["filename"] = name
            return res
        except OcrError as e:
            return {"filename": name, "error": str(e), "status": e.status}
        except Exception as e:
            return {"filename": name, "error": f"OCR failed: {e}"}

    results = await asyncio.gather(*[_one(n, c, d) for (n, c, d) in payloads])
    ok = sum(1 for r in results if "error" not in r)
    return {"num_files": len(results), "num_ok": ok, "results": list(results)}


# ---------------------------------------------------------------------------
# Extraction schema management (schemas are data, user-extensible)
# ---------------------------------------------------------------------------

class SchemaPayload(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = ""
    language: Optional[str] = ""
    type: Optional[str] = "template"          # template | jsonschema
    prompt_hint: Optional[str] = ""
    schema_: dict = Field(default_factory=dict, alias="schema")
    model_config = ConfigDict(extra="allow", populate_by_name=True)


@router.get("/v1/ocr/engines", summary="List the OCR engines this install offers")
async def list_ocr_engines():
    """Which engine names `/v1/ocr` accepts here, and what each one returns.

    A client used to have to know: naming an engine this install had not enabled
    was a 400 with nothing to ask beforehand. Every field is read from this
    install's config, including the ids that also resolve to an engine (so a
    caller can name the model it knows rather than the pipeline driving it) and
    which engines can return boxes or layout regions at all.

    Where an engine's model runs — here, a pinned cluster node, a rented pod — is
    deliberately NOT in the answer: that is configuration, it can change between
    two requests, and a client never has to care.
    """
    cfg = _ocr_config()
    if cfg is None:
        raise HTTPException(status_code=503, detail="OCR subsystem is not available")
    ocr_manager.configure(cfg)
    try:
        return ocr_manager.engine_catalogue()
    except OcrError as e:
        raise HTTPException(status_code=e.status, detail=str(e))


@router.get("/v1/ocr/schemas", summary="List available extraction schemas")
async def list_schemas():
    return {"schemas": schema_store.list()}


@router.get("/v1/ocr/schemas/{name}", summary="Get one extraction schema")
async def get_schema(name: str):
    try:
        return schema_store.get(name)
    except OcrError as e:
        raise HTTPException(status_code=e.status, detail=str(e))


@router.post("/v1/ocr/schemas", summary="Create or update a user extraction schema")
async def save_schema(payload: SchemaPayload):
    name = payload.name
    if not name:
        raise HTTPException(status_code=400, detail="schema 'name' is required")
    data = {
        "name": name,
        "description": payload.description or "",
        "language": payload.language or "",
        "type": payload.type or "template",
        "prompt_hint": payload.prompt_hint or "",
        "schema": getattr(payload, "schema_", None) or getattr(payload, "schema", None) or {},
    }
    try:
        return schema_store.save(name, data)
    except OcrError as e:
        raise HTTPException(status_code=e.status, detail=str(e))


@router.delete("/v1/ocr/schemas/{name}", summary="Delete a user extraction schema")
async def delete_schema(name: str):
    try:
        schema_store.delete(name)
        return {"deleted": name}
    except OcrError as e:
        raise HTTPException(status_code=e.status, detail=str(e))


# ---------------------------------------------------------------------------
# Pipeline step handler (registered as step type "ocr" in custom_pipelines)
# ---------------------------------------------------------------------------

class OcrStepRequest(BaseModel):
    image: str = ""                       # data: URI, base64, or {{stepN.url}} reference
    engine: Optional[str] = None
    dpi: Optional[int] = None
    lang: Optional[str] = None
    detect: Optional[str] = None
    structured: Optional[bool] = False
    schema_: Optional[str] = Field(default=None, alias="schema")
    model_config = ConfigDict(extra="allow", populate_by_name=True)


async def run_ocr_step(request: OcrStepRequest, http_request=None):
    """Pipeline step: OCR an image reference and return {text, structured, ...}."""
    import base64

    cfg = _ocr_config()
    if cfg is None:
        raise HTTPException(status_code=503, detail="OCR subsystem is not available")
    ocr_manager.configure(cfg)

    ref = (getattr(request, "image", "") or "").strip()
    filename = "input.png"
    if ref.startswith("data:"):
        header, encoded = ref.split(",", 1)
        data = base64.b64decode(encoded)
        if "application/pdf" in header:
            filename = "input.pdf"
    elif ref.startswith("http://") or ref.startswith("https://"):
        import urllib.request

        def _fetch(u: str) -> bytes:
            # In a thread: this is an arbitrary URL from the request, so the
            # fetch is as slow as whoever is serving it, and on the loop that
            # was every other request's problem too.
            with urllib.request.urlopen(u) as resp:    # nosec - user-provided pipeline input
                return resp.read()

        data = await asyncio.to_thread(_fetch, ref)
        filename = ref.rsplit("/", 1)[-1] or filename
    else:
        data = base64.b64decode(ref) if ref else b""

    if not data:
        raise HTTPException(status_code=400, detail="ocr step: no image provided")

    schema = getattr(request, "schema_", None) or getattr(request, "schema", None)
    try:
        return await ocr_manager.ocr_document(
            data, filename=filename, engine=request.engine, dpi=request.dpi,
            detect=request.detect, structured=bool(request.structured), schema=schema,
            lang=getattr(request, "lang", None),
        )
    except OcrError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
