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
NOT a vision LLM. Structured-JSON extraction and stamp/signature detection are layered
on in later milestones (fields already accepted, no-op until then).
"""

import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
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


@router.post("/v1/ocr", summary="OCR an image or PDF with a dedicated OCR engine")
async def create_ocr(
    file: UploadFile = File(...),
    engine: Optional[str] = Form(None),        # paddle|doctr|surya (default from config)
    dpi: Optional[int] = Form(None),           # PDF rasterisation DPI
    lang: Optional[str] = Form(None),          # reserved (per-engine language override)
    structured: Optional[bool] = Form(False),  # structured JSON extraction via text model
    schema_: Optional[str] = Form(None, alias="schema"),  # override extraction schema (JSON string)
    detect: Optional[str] = Form(None),        # off|layout|detector|both
):
    cfg = _ocr_config()
    if cfg is None:
        raise HTTPException(status_code=503, detail="OCR subsystem is not available")
    ocr_manager.configure(cfg)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > _MAX_OCR_BYTES:
        raise HTTPException(status_code=413, detail="file too large (max 100 MB)")

    try:
        return await ocr_manager.ocr_document(
            data,
            filename=file.filename or "",
            content_type=file.content_type or "",
            engine=engine,
            dpi=dpi,
            detect=detect,
            structured=bool(structured),
            schema=schema_,
        )
    except OcrError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")


@router.post("/v1/ocr/batch", summary="OCR many images/PDFs concurrently")
async def create_ocr_batch(
    files: List[UploadFile] = File(...),
    engine: Optional[str] = Form(None),
    dpi: Optional[int] = Form(None),
    structured: Optional[bool] = Form(False),
    schema_: Optional[str] = Form(None, alias="schema"),
    detect: Optional[str] = Form(None),
):
    """Fan a batch of documents across the engine's instance pool. Per-file failures are
    reported inline (not fatal) so one bad scan doesn't sink the batch."""
    cfg = _ocr_config()
    if cfg is None:
        raise HTTPException(status_code=503, detail="OCR subsystem is not available")
    ocr_manager.configure(cfg)

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
                detect=detect, structured=bool(structured), schema=schema_,
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
        with urllib.request.urlopen(ref) as resp:      # nosec - user-provided pipeline input
            data = resp.read()
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
        )
    except OcrError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
