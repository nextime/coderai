#!/usr/bin/env python3
"""Retrieval chunking — ``POST /v1/chunk``.

Splits a document into the pieces a retrieval index stores, with the document's
structure decided PER REQUEST by a named or inline *structure template*.

NOT the same thing as embedding windowing. ``embed_long_input`` in
``codai/api/embeddings.py`` makes ONE over-long input fit a model's window and
pools the pieces into ONE vector. This produces MANY chunks, each separately
indexable, each carrying its section, parent and references. A caller may use
both: chunk here, then embed each chunk, where windowing is the safety net for a
chunk that still overflows. This endpoint must never pool and never drop text.

WHY STRUCTURE-AWARE. Length-only chunking cuts an argument wherever the
character budget runs out, so a chunk can straddle a section boundary and a
retrieved passage mixes what the parties alleged with what the court held.
Sectioning first and length-chunking INSIDE each section removes that class of
error and gives every chunk a label and a recoverable parent.

A pure CPU text operation: token counts are approximated as characters divided
by ``chars_per_token``, so no tokenizer is loaded, no model is needed and no GPU
or pod is involved. That is deliberate -- it means every image that runs coderai
serves this, including the light-core ones.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

#: Approximation divisor: token counts are characters / this. Keeps the endpoint
#: tokenizer-free, which is what makes it callable with nothing loaded.
DEFAULT_CHARS_PER_TOKEN = 4
DEFAULT_CHUNK_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 120

#: A pattern longer than this, or more sections than this, is refused as
#: `bad_template`. Patterns come from callers; Python's `re` has NO timeout, so
#: these caps are the only bound available here. They are a blunt guard, not a
#: proof: a pathological pattern within the caps can still backtrack badly. Said
#: plainly rather than implied, because the alternative is believing this is
#: safe when it is only bounded in SIZE.
MAX_PATTERN_CHARS = 2000
MAX_TEMPLATE_SECTIONS = 64

#: Paragraphs are separated by a blank line (possibly with whitespace in it).
_PARA_SPLIT = re.compile(r"\n\s*\n")
#: Sentence boundary used to break a paragraph that exceeds the target on its
#: own. Deliberately simple: it is what the production chunker used, and a
#: "better" splitter moves every existing chunk boundary.
_SENT_SPLIT = re.compile(r"(?<=[.;:!?])\s+")


class ChunkError(Exception):
    """A request-level failure, carrying the status and machine-readable code
    the endpoint reports (see docs/chunking.md#errors)."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
#
# `conclusioni` IS ITS OWN LABEL and must never be folded into `dispositivo`.
# In a ruling the dispositivo is what the judge decided; in a filing the
# conclusioni are what a party ASKED FOR. Merging them lets a demand surface as
# if it were a holding, which is the most damaging mistake a legal index can
# make. A template that wants them merged has to say so by giving both patterns
# the same label.
#
# Two marker families, because filings outnumber rulings in a merito corpus:
# with ruling markers alone only 8-10% of 400 measured documents were
# recognised.
#: What may follow a header on its own line. `line_prefix` tolerates decoration
#: BEFORE a header; nothing tolerated it after, so `**FATTO**` was missed while
#: `> fatto` matched -- the end-anchored alternatives (`^\s*fatto\s*$` and
#: friends) require the line to end right after the word, and a trailing `**`
#: defeats that.
#:
#: Widening only the END anchor is PURELY ADDITIVE: every line that matched
#: `\s*$` still matches `[\s*#>_]*$`, so no existing match moves. What changes
#: is which documents get sectioned AT ALL -- one whose headers were markdown-
#: emphasised used to fall back to flat chunking and now gets labels. Those
#: documents' chunks therefore differ from an index built before this, and they
#: need re-chunking; documents already sectioned are unaffected.
#: And the same class BEFORE the word. `^\s*fatto` tolerates only
#: whitespace, so `**FATTO**` failed on the leading asterisks as well as
#: the trailing ones. The patterns that already handled decoration
#: (`premesso che`, `conclusioni`) spell it `^\s*[*#>_ ]*` for exactly this
#: reason; these three simply never did.
_LEAD = r"[\s*#>_]*"
_TRAIL = r"[\s*#>_]*"

IT_LEGAL: Dict[str, Any] = {
    "name": "it-legal",
    "head_label": "intestazione",
    "flags": "im",
    "line_prefix": r"[ \t>*_#]*",
    "merge_adjacent": True,
    "keep_header_line": True,
    "sections": [
        {"label": "premessa",
         "pattern": r"tanto\s+premesso|ci[oò]\s+premesso|^\s*[*#>_ ]*premesso\s+che\b|^" + _LEAD + r"premess[ae]" + _TRAIL + "$"},
        {"label": "fatto",
         "pattern": r"svolgimento\s+del\s+processo|ragioni\s+di\s+fatto|in\s+fatto\b|fatto\s+e\s+diritto|^" + _LEAD + r"fatto" + _TRAIL + r"$|esposizione\s+del\s+fatto"},
        {"label": "motivazione",
         "pattern": r"motivi\s+della\s+decisione|ragioni\s+della\s+decisione|motivazione|considerato\s+in\s+diritto|in\s+diritto\b|motivi\s+in\s+diritto|ragioni\s+in\s+diritto"},
        {"label": "dispositivo",
         "pattern": r"p\.?\s*q\.?\s*m\.?|p\.?\s*t\.?\s*m\.?|per\s+questi\s+motivi|^" + _LEAD + r"dispositivo" + _TRAIL + r"$|così\s+deciso|così\s+provvede"},
        {"label": "conclusioni",
         "pattern": r"^\s*[*#>_ ]*conclusion[ei]\b|\bconclusioni\s*:|voglia\s+(?:l['’]ill\.?mo\s+)?(?:il\s+)?(?:tribunale|giudice|giudicante)|^\s*[*#>_ ]*chiede\b|salvis\s+iuribus"},
    ],
}

#: DEVIATION FROM docs/chunking.md, deliberate and recorded here because it
#: affects byte-identity with the implementation this ports.
#:
#: The doc gives the patterns verbatim and asks for a fidelity case with a
#: markdown-decorated header (`**FATTO**`). Those two requirements contradict
#: each other: the verbatim `^\s*fatto\s*$` cannot match `**FATTO**`. The
#: anchors here are widened to `^[\s*#>_]*fatto[\s*#>_]*$` so the documented
#: case works.
#:
#: The widening is purely additive -- no line that matched before stops
#: matching, so no existing match position moves. What changes is which
#: documents are sectioned AT ALL: one whose headers were emphasised used to
#: fall back to flat chunking and now gets labels. Those documents' chunks
#: differ from an index built before this and must be re-chunked; documents
#: already sectioned are identical.
#:
#: Worth confirming against the production implementation: if it DOES match
#: `**FATTO**`, its patterns differ from the table in the doc, and that
#: difference is what matters for identity rather than this note.
BUILTIN_TEMPLATES: Dict[str, Dict[str, Any]] = {"it-legal": IT_LEGAL}


def _flags_from(spec: str) -> int:
    out = 0
    for ch in (spec or "im"):
        if ch == "i":
            out |= re.IGNORECASE
        elif ch == "m":
            out |= re.MULTILINE
        elif ch == "s":
            out |= re.DOTALL
        elif ch == "x":
            out |= re.VERBOSE
    return out


def resolve_template(structure, registered: Optional[dict] = None) -> Optional[dict]:
    """A template from a name or an inline object.

    Inline beats registered beats built-in. An unknown NAME is an error and
    never a silent fallback to the built-in: a caller who asked for their own
    template and got Italian legal sectioning would index the wrong boundaries
    and have no way to tell.
    """
    if structure in (None, "", False):
        return None
    if isinstance(structure, dict):
        return structure
    name = str(structure).strip()
    for src in (registered or {}, BUILTIN_TEMPLATES):
        if isinstance(src, dict) and name in src:
            tpl = src[name]
            if not isinstance(tpl, dict):
                raise ChunkError("bad_template",
                                 f"registered template {name!r} is not an object")
            return tpl
    raise ChunkError("unknown_template", f"no structure template named {name!r}",
                     status=404)


def compile_template(tpl: dict) -> Tuple[Any, List[str], dict]:
    """(compiled alternation, label-by-group-index, normalised options).

    ONE alternation with named groups, scanned left to right, so the marks come
    out in document order and the first matching alternative wins at a given
    position. Scanning per pattern and merging afterwards changes WHICH header
    claims an overlapping position -- a different segmentation for the same
    document, which is exactly the kind of divergence a port must not introduce.
    """
    if not isinstance(tpl, dict):
        raise ChunkError("bad_template", "structure template must be an object")
    secs = tpl.get("sections")
    if not isinstance(secs, list) or not secs:
        raise ChunkError("bad_template", "template has no sections")
    if len(secs) > MAX_TEMPLATE_SECTIONS:
        raise ChunkError("bad_template",
                         f"template has {len(secs)} sections; the limit is "
                         f"{MAX_TEMPLATE_SECTIONS}")
    prefix = tpl.get("line_prefix")
    prefix = r"[ \t>*_#]*" if prefix is None else str(prefix)
    labels: List[str] = []
    parts: List[str] = []
    for i, sec in enumerate(secs):
        if not isinstance(sec, dict):
            raise ChunkError("bad_template", f"section {i} is not an object")
        pat = sec.get("pattern")
        if not isinstance(pat, str) or not pat:
            raise ChunkError("bad_template", f"section {i} has no pattern")
        if len(pat) > MAX_PATTERN_CHARS:
            raise ChunkError("bad_template",
                             f"section {i} pattern is {len(pat)} chars; the "
                             f"limit is {MAX_PATTERN_CHARS}")
        labels.append(str(sec.get("label") or ""))
        # Group names are positional (g0, g1, ...) because a label may be any
        # string and need not be a valid Python identifier.
        parts.append(f"(?P<g{i}>^{prefix}(?:{pat}))")
    try:
        rx = re.compile("|".join(parts), _flags_from(str(tpl.get("flags") or "im")))
    except re.error as exc:
        raise ChunkError("bad_template", f"pattern does not compile: {exc}")
    opts = {
        "head_label": tpl.get("head_label", "intestazione"),
        "merge_adjacent": bool(tpl.get("merge_adjacent", True)),
        "keep_header_line": bool(tpl.get("keep_header_line", True)),
        "name": str(tpl.get("name") or "") or None,
    }
    return rx, labels, opts


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------
_BINARY_SIGNATURES = (b"PK\x03\x04", b"%PDF-", b"\x7fELF", b"\xd0\xcf\x11\xe0")


def looks_binary(raw) -> Optional[str]:
    """Why this input is not text, or None.

    A chunker that accepts container bytes embeds garbage and the index cannot
    tell afterwards: measured on a production corpus, 0.59% of chunks held
    control characters and 0.04% raw OOXML -- documents whose CONTAINER was
    chunked instead of its extracted text.
    """
    if isinstance(raw, str):
        data = raw.encode("utf-8", "ignore")
        text = raw
    else:
        data = bytes(raw or b"")
        text = data.decode("utf-8", "ignore")
    for sig in _BINARY_SIGNATURES:
        if data.startswith(sig):
            return f"input begins with a {sig!r} container header"
    if "[Content_Types].xml" in text[:4096]:
        return "input looks like an OOXML container, not its extracted text"
    if text:
        bad = sum(1 for c in text if ord(c) < 32 and c not in "\t\n\f\r")
        if bad / max(1, len(text)) > 0.01:
            return (f"{bad} control characters in {len(text)} -- this looks like "
                    f"container bytes rather than extracted text")
    return None


# ---------------------------------------------------------------------------
# Sectioning
# ---------------------------------------------------------------------------
def split_sections(text: str, tpl: Optional[dict]) -> Tuple[List[dict], bool, Optional[str]]:
    """(sections, structural_applied, template_name).

    A section runs from its header's match START to the next header's start when
    ``keep_header_line`` is set, so the header stays inside its own section.
    """
    if tpl is None:
        return [{"label": "", "text": text}], False, None
    rx, labels, opts = compile_template(tpl)
    marks = [(m.start(), labels[int(m.lastgroup[1:])])
             for m in rx.finditer(text) if m.lastgroup]
    if not marks:
        # No header anywhere: flat, and SAID so, rather than inventing one
        # section per paragraph or pretending the template applied.
        return [{"label": "", "text": text}], False, opts["name"]
    out: List[dict] = []
    head = text[:marks[0][0]]
    if head.strip() and opts["head_label"] is not None:
        out.append({"label": str(opts["head_label"] or ""), "text": head})
    for i, (pos, label) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[pos:end] if opts["keep_header_line"] else text[pos:end]
        if opts["merge_adjacent"] and out and out[-1]["label"] == label:
            # Filings routinely open the conclusions twice (CONCLUSIONI, then
            # "Voglia il Tribunale..."). Left separate, one block becomes two
            # and the parent ids multiply, defeating the point of `parent`.
            out[-1]["text"] = out[-1]["text"] + "\n" + body
        else:
            out.append({"label": label, "text": body})
    return out, True, opts["name"]


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
def _pieces(section_text: str, target: int) -> List[str]:
    """A section's paragraphs, with over-long ones broken down. Nothing is ever
    discarded -- a chunker that drops text is worse than one that chunks badly,
    because the loss is invisible in the index."""
    out: List[str] = []
    for para in _PARA_SPLIT.split(section_text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= target:
            out.append(para)
            continue
        # Sentence boundaries, greedily refilled to target.
        cur = ""
        for sent in _SENT_SPLIT.split(para):
            if not sent:
                continue
            if len(sent) > target:
                if cur:
                    out.append(cur)
                    cur = ""
                # A single sentence still over target: fixed slices.
                for i in range(0, len(sent), target):
                    out.append(sent[i:i + target])
                continue
            if cur and len(cur) + len(sent) + 1 > target:
                out.append(cur)
                cur = sent
            else:
                cur = (cur + " " + sent) if cur else sent
        if cur:
            out.append(cur)
    return out


def pack(pieces: List[str], target: int, overlap: int) -> List[str]:
    """Pack pieces into chunks with a carried character tail.

    Reproduced verbatim from the production chunker, including the parts that
    look like they could be improved:

      * paragraphs are joined with "\\n\\n", and the `+ 2` accounts for it;
      * the overlap is a RAW CHARACTER TAIL of the emitted chunk, which is not
        sentence- or token-aligned and may begin mid-word;
      * overlap == 0 starts the next chunk at the piece with no carry.

    "Improving" any of those moves every existing chunk boundary, which makes a
    re-chunked index incomparable with the one it replaces -- and then retrieval
    quality cannot be measured across the change.

    A chunk may therefore reach target + overlap characters.
    """
    chunks: List[str] = []
    cur = ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > target:
            chunks.append(cur)
            cur = (cur[-overlap:] + "\n\n" + p) if overlap else p
        else:
            cur = (cur + "\n\n" + p).strip() if cur else p
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------
_ART_SUFFIX = r"(?:\s*(?:bis|ter|quater|quinquies|sexies))?"
#: The code suffix is MANDATORY, not optional. A bare `art. 360` is ambiguous
#: -- every procedural code has one -- so it is deliberately NOT a reference:
#: this field exists to be exact-matched, and an unqualified number would put
#: three different articles under one key.
_ART_RX = re.compile(
    r"\bart(?:t|icol[oi])?\.?\s*(\d+" + _ART_SUFFIX + r")"
    r"\s*(?:del\s+|della\s+)?(codciv|c\.?c\.?|c\.?p\.?c\.?|c\.?p\.?p\.?|c\.?p\.?|cost\.?)",
    re.IGNORECASE)
#: The year is kept VERBATIM, two digits included: `d.p.r. 115/02` normalises
#: to `d.p.r.115/02`, never to `/2002`. Expanding it would silently split the
#: same act across two keys depending on how the source wrote it.
_ACT_RX = re.compile(
    r"\b(d\.?\s*lgs\.?|d\.?\s*l\.?|d\.?\s*p\.?\s*r\.?|legge|l\.?)\s*"
    r"(?:n\.?\s*)?(\d+)\s*/\s*(\d{2,4})", re.IGNORECASE)
_CODE_MAP = {"codciv": "c.c.", "cc": "c.c.", "cpc": "c.p.c.", "cpp": "c.p.p.",
             "cp": "c.p.", "cost": "cost."}
_KIND_MAP = {"dlgs": "d.lgs.", "dl": "d.l.", "dpr": "d.p.r.", "legge": "l.",
             "l": "l."}


def extract_refs(text: str) -> List[str]:
    """Normalised article and act citations, de-duplicated, FIRST-SEEN ORDER kept.

    An exact-match signal for hybrid search, and the thing dense embeddings
    blur: `art. 2043` and `art. 2054` sit close in vector space and are entirely
    different claims.
    """
    seen: Dict[str, None] = {}
    for m in _ART_RX.finditer(text):
        num = re.sub(r"\s+", "", m.group(1) or "").lower()
        code_raw = re.sub(r"[.\s]", "", (m.group(2) or "")).lower()
        code = _CODE_MAP.get(code_raw, "")
        seen.setdefault(f"art.{num} {code}".strip(), None)
    for m in _ACT_RX.finditer(text):
        kind_raw = re.sub(r"[.\s]", "", (m.group(1) or "")).lower()
        kind = _KIND_MAP.get(kind_raw, kind_raw + ".")
        seen.setdefault(f"{kind}{m.group(2)}/{m.group(3)}", None)
    return list(seen)


# ---------------------------------------------------------------------------
# The whole operation
# ---------------------------------------------------------------------------
def chunk_document(text: str, *, chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
                   overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
                   chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
                   structural: bool = True, structure: Any = "it-legal",
                   refs: Any = "it-legal", include_offsets: bool = False,
                   require_text: bool = True,
                   registered: Optional[dict] = None) -> dict:
    """Chunk `text` and return the response document (see docs/chunking.md)."""
    if not isinstance(text, str):
        text = (text or b"").decode("utf-8", "ignore")
    if require_text:
        why = looks_binary(text)
        if why:
            raise ChunkError("not_text", why)
    # 1. Normalise.
    text = text.replace("\r", "")
    if not text.strip():
        raise ChunkError("empty_input", "input is empty after normalisation")

    target = max(1, int(chunk_tokens) * max(1, int(chars_per_token)))
    overlap = max(0, int(overlap_tokens) * max(1, int(chars_per_token)))

    tpl = resolve_template(structure, registered) if structural else None
    sections, applied, tpl_name = split_sections(text, tpl)

    want_refs = refs not in (None, False, "", "false")
    ref_tpl = None
    if want_refs and isinstance(refs, dict):
        ref_tpl = refs            # reserved: a template may replace the patterns

    chunks: List[dict] = []
    sec_out: List[dict] = []
    cursor = 0
    for parent, sec in enumerate(sections):
        body = sec["text"]
        label = sec["label"] or ""
        sec_out.append({"parent": parent, "section": label or None,
                        "n_chars": len(body)})
        for piece in pack(_pieces(body, target), target, overlap):
            obj: Dict[str, Any] = {
                "ord": len(chunks), "text": piece, "n_chars": len(piece),
                "section": label or None, "parent": parent,
            }
            if want_refs:
                obj["norm_refs"] = extract_refs(piece)
            if include_offsets:
                # Offsets into the NORMALISED text, found forward from the last
                # one so a repeated passage does not resolve to the first copy.
                at = text.find(piece, cursor)
                if at < 0:
                    at = text.find(piece)
                if at >= 0:
                    obj["start"], obj["end"] = at, at + len(piece)
                    cursor = at
            chunks.append(obj)

    return {
        "chunks": chunks,
        "n_chunks": len(chunks),
        "sections": sec_out,
        "structure": tpl_name if applied else None,
        "structural_applied": bool(applied),
        "params": {"chunk_tokens": int(chunk_tokens),
                   "overlap_tokens": int(overlap_tokens),
                   "chars_per_token": int(chars_per_token)},
    }


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------
import os                                                     # noqa: E402

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field             # noqa: E402

router = APIRouter()

#: Refuse an input above this many characters with 413 rather than spending
#: minutes of CPU on it. Generous: a 400-page filing is ~1 M characters.
DEFAULT_MAX_CHARS = 8_000_000


def _cfg():
    try:
        from codai.admin.routes import config_manager
        return getattr(getattr(config_manager, "config", None), "chunking", None)
    except Exception:
        return None


def _setting(name: str, env: str, default, cast=int):
    """Per-request > config > env > default, the convention the OCR and
    embedding settings already use."""
    c = _cfg()
    if c is not None:
        v = getattr(c, name, None)
        if v not in (None, ""):
            try:
                return cast(v)
            except (TypeError, ValueError):
                pass
    raw = (os.environ.get(env) or "").strip()
    if raw:
        try:
            return cast(raw)
        except (TypeError, ValueError):
            pass
    return default


def _bool_setting(name: str, env: str, default: bool) -> bool:
    c = _cfg()
    if c is not None:
        v = getattr(c, name, None)
        if isinstance(v, bool):
            return v
    raw = (os.environ.get(env) or "").strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    return default


def _registered_templates() -> dict:
    c = _cfg()
    t = getattr(c, "templates", None) if c is not None else None
    return t if isinstance(t, dict) else {}


class ChunkRequest(BaseModel):
    text: Optional[str] = None
    chunk_tokens: Optional[int] = Field(None, description="Target size in approximated tokens (chars / chars_per_token).")
    overlap_tokens: Optional[int] = Field(None, description="Carried tail in approximated tokens. 0 = no carry.")
    chars_per_token: Optional[int] = Field(None, description="The approximation divisor; no tokenizer is loaded.")
    structural: Optional[bool] = Field(None, description="false = flat length chunking with no sectioning.")
    structure: Optional[Any] = Field(None, description="Template name, or an inline template object.")
    extract_refs: Optional[Any] = Field(None, description="Reference extractor: a name, an inline object, or false to skip.")
    require_text: Optional[bool] = Field(None, description="Reject binary input instead of chunking container bytes.")
    include_offsets: Optional[bool] = Field(None, description="Add char offsets into the normalised source text.")
    model_config = ConfigDict(extra="allow")


def _resolve_params(text: str, body) -> dict:
    """Merge request fields over the configured defaults."""
    g = (lambda n, e, d, c=int: _setting(n, e, d, c))
    return dict(
        text=text,
        chunk_tokens=(body.chunk_tokens if body.chunk_tokens is not None
                      else g("chunk_tokens", "CODERAI_CHUNK_TOKENS", DEFAULT_CHUNK_TOKENS)),
        overlap_tokens=(body.overlap_tokens if body.overlap_tokens is not None
                        else g("overlap", "CODERAI_CHUNK_OVERLAP", DEFAULT_OVERLAP_TOKENS)),
        chars_per_token=(body.chars_per_token if body.chars_per_token is not None
                         else g("chars_per_token", "CODERAI_CHUNK_CHARS_PER_TOKEN",
                                DEFAULT_CHARS_PER_TOKEN)),
        structural=(body.structural if body.structural is not None
                    else _bool_setting("structural", "CODERAI_CHUNK_STRUCTURAL", True)),
        structure=(body.structure if body.structure is not None
                   else (_setting("structure", "CODERAI_CHUNK_STRUCTURE", "it-legal", str))),
        refs=(body.extract_refs if body.extract_refs is not None else "it-legal"),
        include_offsets=bool(body.include_offsets),
        require_text=(body.require_text if body.require_text is not None
                      else _bool_setting("require_text", "CODERAI_CHUNK_REQUIRE_TEXT", True)),
        registered=_registered_templates(),
    )


def _run(params: dict) -> dict:
    limit = _setting("max_chars", "CODERAI_CHUNK_MAX_CHARS", DEFAULT_MAX_CHARS)
    if len(params["text"]) > limit:
        raise HTTPException(status_code=413, detail={
            "code": "too_large",
            "message": f"input is {len(params['text'])} characters; the limit is "
                       f"{limit} (chunking.max_chars / CODERAI_CHUNK_MAX_CHARS)"})
    try:
        return chunk_document(**params)
    except ChunkError as e:
        # The CODE travels, not just the prose: a caller retrying on 400 needs to
        # tell "your file is a zip" from "your template does not compile".
        raise HTTPException(status_code=e.status,
                            detail={"code": e.code, "message": e.message})


@router.post("/v1/chunk", summary="Split a document into retrieval chunks")
async def create_chunks(request: Request, body: ChunkRequest = None):
    """Chunk a document for a retrieval index.

    Structure-aware by default: the document is sectioned with a template, then
    length-chunked INSIDE each section, so a chunk never straddles a boundary
    and every chunk carries a label and a recoverable `parent`.

    This is NOT embedding windowing (`embed_long_input`), which makes one input
    fit a model window and pools it into ONE vector. This never pools and never
    drops text.

    Pure CPU: token counts are approximated as characters / `chars_per_token`,
    so no tokenizer is loaded and no model or GPU is involved.
    """
    if body is None:
        body = ChunkRequest()
    text = body.text or ""
    if not text:
        raise HTTPException(status_code=400, detail={
            "code": "empty_input",
            "message": "provide 'text' (JSON) or 'file' (multipart)"})
    return _run(_resolve_params(text, body))


@router.post("/v1/chunk/file", summary="Split an uploaded document into chunks")
async def create_chunks_from_file(
    file: UploadFile = File(...),
    chunk_tokens: Optional[int] = Form(None),
    overlap_tokens: Optional[int] = Form(None),
    chars_per_token: Optional[int] = Form(None),
    structural: Optional[bool] = Form(None),
    structure: Optional[str] = Form(None),
    extract_refs: Optional[str] = Form(None),
    require_text: Optional[bool] = Form(None),
    include_offsets: Optional[bool] = Form(None),
):
    """The multipart form of `/v1/chunk`, with the same field names as values --
    matching how `/v1/ocr` takes its options.

    The upload is decoded as UTF-8 text. A PDF or an OOXML container is REFUSED
    (`not_text`) rather than chunked: extract its text first (`/v1/ocr` does
    that for a scan). Chunking a container embeds garbage that the index cannot
    distinguish from content afterwards.
    """
    raw = await file.read()
    why = looks_binary(raw) if (require_text in (None, True)) else None
    if why:
        raise HTTPException(status_code=400, detail={"code": "not_text", "message": why})
    body = ChunkRequest(
        chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
        chars_per_token=chars_per_token, structural=structural,
        structure=structure, require_text=require_text,
        include_offsets=include_offsets,
        extract_refs=(False if (extract_refs or "").strip().lower()
                      in ("0", "false", "no", "off") else (extract_refs or None)))
    return _run(_resolve_params(raw.decode("utf-8", "ignore"), body))


@router.get("/v1/chunk/templates", summary="List the structure templates on offer")
async def list_templates():
    """Which names `/v1/chunk` accepts, and what each one sections on.

    An unknown name is an error rather than a silent fallback to the built-in,
    so a caller needs to be able to see the list.
    """
    reg = _registered_templates()
    out = []
    for name, tpl in list(BUILTIN_TEMPLATES.items()) + list(reg.items()):
        if not isinstance(tpl, dict):
            continue
        out.append({
            "name": name,
            "builtin": name in BUILTIN_TEMPLATES and name not in reg,
            "head_label": tpl.get("head_label"),
            "merge_adjacent": bool(tpl.get("merge_adjacent", True)),
            "labels": [str(s.get("label") or "") for s in (tpl.get("sections") or [])
                       if isinstance(s, dict)],
        })
    return {"templates": out}
