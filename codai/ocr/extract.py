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

"""Structured field extraction from OCR text via an existing coderai text model.

This is NOT a separate model — it reuses whatever text LLM is configured
(``ocr.extract_model_id``) through the in-process ``/v1/chat/completions`` handler. The
target schema is not hardcoded: it is resolved from the schema store (named schema,
inline schema, or generic "auto" mode). See :mod:`codai.ocr.schemas`.
"""

import json
from typing import Optional

from codai.ocr.base import OcrError
from codai.ocr.schemas import schema_store


_SYSTEM_PROMPT = (
    "You extract structured data from OCR'd documents. The OCR text may contain "
    "recognition errors; correct them implicitly where obvious. Respond with a SINGLE "
    "valid JSON object ONLY — no prose, no explanations, no code fences. Use null for "
    "fields not present in the document."
)

_AUTO_INSTRUCTION = (
    "No fixed schema was given. Extract the salient fields of this document as a flat "
    "JSON object of key/value pairs (include a \"document_type\" key), plus a \"summary\"."
)


def _build_user_prompt(text: str, schema_obj: Optional[dict]) -> str:
    parts = []
    if schema_obj is None:
        parts.append(_AUTO_INSTRUCTION)
    else:
        stype = schema_obj.get("type", "template")
        hint = schema_obj.get("prompt_hint", "")
        schema_json = json.dumps(schema_obj.get("schema", {}), ensure_ascii=False, indent=2)
        if stype == "jsonschema":
            parts.append(
                "Return a JSON object conforming to this JSON Schema (same property "
                "names and types):"
            )
        else:
            parts.append(
                "Extract the fields following EXACTLY this JSON template (same keys):"
            )
        parts.append(schema_json)
        if hint:
            parts.append(hint)
    parts.append("Document OCR text:\n-----8<-----\n" + text + "\n-----8<-----")
    parts.append("Return only the filled JSON.")
    return "\n\n".join(parts)


def _content_from_result(result) -> str:
    """Pull the assistant message content from a chat_completions return value,
    tolerating dict / pydantic model / JSONResponse shapes."""
    r = None
    if isinstance(result, dict):
        r = result
    elif hasattr(result, "model_dump"):
        try:
            r = result.model_dump()
        except Exception:
            r = None
    if r is None and hasattr(result, "body"):
        try:
            r = json.loads(result.body)
        except Exception:
            r = None
    if r is None and hasattr(result, "__dict__"):
        r = result.__dict__
    if not isinstance(r, dict):
        return ""
    choices = r.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if isinstance(msg, dict):
        return msg.get("content") or ""
    return ""


def _parse_json_lenient(content: str):
    """Best-effort JSON parse: strip code fences, then fall back to the outermost {...}."""
    s = (content or "").strip()
    if s.startswith("```"):
        s = s.split("```", 2)
        s = s[1] if len(s) > 1 else content
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(s[a:b + 1])
        except Exception:
            pass
    return None


def _validate(parsed, schema_obj: Optional[dict]) -> Optional[str]:
    """Validate ``parsed`` against a JSON Schema if applicable. Returns an error string
    or None. Best-effort: silently skips when jsonschema is absent or type != jsonschema."""
    if not schema_obj or schema_obj.get("type") != "jsonschema":
        return None
    try:
        import jsonschema
    except Exception:
        return None
    try:
        jsonschema.validate(parsed, schema_obj.get("schema", {}))
        return None
    except Exception as e:
        return str(getattr(e, "message", e))


async def extract_fields(text: str, cfg, schema: Optional[str] = None) -> dict:
    """Run field extraction. ``schema`` is a per-request spec (name, inline JSON, or
    auto); falls back to ``cfg.extract_schema``. Returns {"schema": name, "fields": {...}}
    or, on parse failure, {"_parse_error", "_raw"}. Raises OcrError(400) if not configured.
    """
    if not getattr(cfg, "extract_enabled", False):
        raise OcrError("structured extraction is disabled (set ocr.extract_enabled)", status=400)
    model_id = getattr(cfg, "extract_model_id", "") or ""
    if not model_id:
        raise OcrError("no extraction model set (ocr.extract_model_id)", status=400)

    spec = schema if (schema is not None and schema != "") else getattr(cfg, "extract_schema", "")
    schema_obj, _ = schema_store.resolve(spec)
    schema_name = (schema_obj or {}).get("name") or ("auto" if schema_obj is None else "inline")

    if not (text or "").strip():
        return {"schema": schema_name, "fields": None, "_note": "empty OCR text"}

    from codai.api.text import chat_completions
    from codai.pydantic.textrequest import ChatCompletionRequest

    req = ChatCompletionRequest(
        model=model_id,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(text, schema_obj)},
        ],
        stream=False,
        temperature=0.0,
        max_tokens=int(getattr(cfg, "extract_max_tokens", 2048) or 2048),
    )
    try:
        result = await chat_completions(req, None)
    except Exception as e:
        raise OcrError(f"extraction model call failed: {e}", status=500)

    content = _content_from_result(result)
    parsed = _parse_json_lenient(content)
    if parsed is None:
        return {"schema": schema_name, "fields": None,
                "_parse_error": "model did not return valid JSON", "_raw": content}

    out = {"schema": schema_name, "fields": parsed}
    if bool(getattr(cfg, "extract_validate", True)):
        err = _validate(parsed, schema_obj)
        if err:
            out["_validation_error"] = err
    return out
