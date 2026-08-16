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

"""OCR extraction schema store.

Schemas are DATA, not code — so any document type can be supported without touching the
engine. A schema is resolved three ways (in precedence order), which is how we avoid
trying to enumerate "every schema out there":

1. **Inline** — the spec string starts with ``{`` / ``[``: used verbatim (a fill-in
   template or a JSON Schema).
2. **Named** — a bare name (``italian_sentenza``): looked up in the user schema dir
   (``<config_dir>/ocr_schemas/*.json``) first, then the built-in seeds shipped with
   coderai. User files shadow built-ins of the same name.
3. **Auto** — empty / ``auto`` / ``generic``: no schema; the model is asked to extract
   the salient key/value fields itself.

A schema file is JSON:
    {
      "name": "italian_sentenza",
      "description": "Italian judicial decision (sentenza)",
      "language": "it",
      "type": "template" | "jsonschema",   # template = fill-in example; jsonschema = JSON Schema
      "schema": { ... },
      "prompt_hint": "optional extra instruction for the model"
    }
"""

import json
import os
import re
from typing import List, Optional, Tuple

from codai.ocr.base import OcrError

_BUILTIN_DIR = os.path.join(os.path.dirname(__file__), "schemas_builtin")
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_AUTO_NAMES = ("", "auto", "generic")


def _user_dir() -> str:
    """User schema directory: ``<config_dir>/ocr_schemas`` (created on demand)."""
    base = ""
    try:
        from codai.admin.routes import config_manager
        if config_manager is not None:
            base = str(getattr(config_manager, "config_dir", "") or "")
    except Exception:
        base = ""
    if not base:
        base = os.path.expanduser("~/.coderai")
    return os.path.join(base, "ocr_schemas")


def _valid_name(name: str) -> str:
    if not name or not _NAME_RE.match(name):
        raise OcrError(
            "invalid schema name (use letters, digits, '.', '_', '-')", status=400
        )
    return name


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class SchemaStore:
    def list(self) -> List[dict]:
        """Return schema summaries (built-ins + user), user shadowing built-ins."""
        out = {}
        for src, d in (("builtin", _BUILTIN_DIR), ("user", _user_dir())):
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".json"):
                    continue
                data = _read_json(os.path.join(d, fn)) or {}
                name = data.get("name") or fn[:-5]
                out[name] = {
                    "name": name,
                    "description": data.get("description", ""),
                    "language": data.get("language", ""),
                    "type": data.get("type", "template"),
                    "source": src,
                }
        return sorted(out.values(), key=lambda x: x["name"])

    def get(self, name: str) -> dict:
        """Load a named schema (user dir wins over built-in). Raises 400 if missing."""
        _valid_name(name)
        for d in (_user_dir(), _BUILTIN_DIR):
            path = os.path.join(d, f"{name}.json")
            if os.path.isfile(path):
                data = _read_json(path)
                if data is None:
                    raise OcrError(f"schema '{name}' is not valid JSON", status=500)
                return data
        avail = ", ".join(s["name"] for s in self.list()) or "(none)"
        raise OcrError(f"unknown schema '{name}'. Available: {avail}", status=400)

    def save(self, name: str, data: dict) -> dict:
        """Create/overwrite a USER schema (built-ins are read-only)."""
        _valid_name(name)
        if not isinstance(data, dict) or "schema" not in data:
            raise OcrError("schema payload must be an object with a 'schema' field", status=400)
        data.setdefault("name", name)
        data.setdefault("type", "template")
        d = _user_dir()
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data

    def delete(self, name: str) -> None:
        """Delete a USER schema. Built-ins cannot be deleted."""
        _valid_name(name)
        path = os.path.join(_user_dir(), f"{name}.json")
        if not os.path.isfile(path):
            raise OcrError(f"user schema '{name}' not found (built-ins are read-only)", status=404)
        os.remove(path)

    # -- resolution --------------------------------------------------------

    def resolve(self, spec: Optional[str]) -> Tuple[Optional[dict], Optional[str]]:
        """Resolve a spec → (schema_dict_or_None, error_note).

        Returns ``(None, None)`` for auto/generic mode (no schema). For inline or named
        schemas returns the loaded/parsed schema dict.
        """
        s = (spec or "").strip()
        if s.lower() in _AUTO_NAMES:
            return None, None
        if s[:1] in ("{", "["):
            try:
                parsed = json.loads(s)
            except Exception as e:
                raise OcrError(f"inline schema is not valid JSON: {e}", status=400)
            # Wrap a bare schema object into the standard envelope.
            if isinstance(parsed, dict) and "schema" in parsed and (
                    "type" in parsed or "name" in parsed):
                return parsed, None
            inferred = "jsonschema" if isinstance(parsed, dict) and (
                "$schema" in parsed or "properties" in parsed) else "template"
            return {"name": "inline", "type": inferred, "schema": parsed}, None
        return self.get(s), None


schema_store = SchemaStore()
