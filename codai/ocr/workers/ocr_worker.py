#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.
#
# Isolated-venv OCR worker. Runs INSIDE a dedicated virtualenv (PaddleOCR or Surya) so
# their conflicting deps (opencv-contrib / pillow<11) never touch the main coderai venv.
# SELF-CONTAINED: imports only stdlib + PIL + the one OCR engine present in this venv.
# Never import coderai here.
#
# Protocol: newline-delimited JSON on stdin/stdout.
#   startup            → worker prints {"ready": true}
#   {"cmd":"load","opts":{...}}   → {"ok":true} | {"ok":false,"error":..,"status":..}
#   {"cmd":"ocr","image":"<b64 png>"} → {"ok":true,"page":{...}} | {"ok":false,...}
#   {"cmd":"ping"}     → {"ok":true}
# argv[1] = engine name ("paddle" | "surya").

import base64
import os
import io
import json
import queue
import sys
import threading


#: stdout is now written by several threads (one per page in flight), and a
#: half-written JSON line is an unparseable reply for a page that actually
#: succeeded. One lock, held for exactly one line.
_OUT_LOCK = threading.Lock()


def _emit(obj):
    line = json.dumps(obj) + "\n"
    with _OUT_LOCK:
        sys.stdout.write(line)
        sys.stdout.flush()


#: Per-PAGE trace state, kept per THREAD.
#:
#: surya's hook is module-level — there is one ``_generate_one`` per process —
#: and the sink used to be a list belonging to whichever worker object installed
#: the hook last. With one page per process that was exact. With several pages in
#: flight in one process it would attribute one page's logprobs, and its accent
#: repairs, to another page entirely: a diagnostic that lies is worse than none,
#: and `surya_repair_accents` would rewrite the wrong text.
_TRACE = threading.local()

#: How many RE-tries of a repeating page coderai allows, or None for surya's
#: own number. One cell, not a per-worker attribute: the hook it feeds is
#: module-level, so every lane in this process shares it.
_RETRY_BUDGET = [None]

#: Our override for surya's per-reply token cap, or 0 for surya's own 2048.
#: Module-level for the same reason the retry budget is: the hook is.
_MAX_TOKENS = [0]

#: How many times a cut-off reply may be resumed. 0 returns it partial.
_CONTINUE_BUDGET = [0]

#: What surya uses when nobody tells it otherwise
#: (`chat_completions_batch(max_tokens_default=2048)`).
_SURYA_DEFAULT_MAX_TOKENS = 2048


def _trace_begin():
    """Start a page's trace. Returns its (calls, repairs) — this thread's own."""
    _TRACE.calls = []
    _TRACE.repairs = []
    _TRACE.attempts = 0
    _TRACE.tokens = []
    _TRACE.continued = 0
    _TRACE.still_truncated = False
    return _TRACE.calls, _TRACE.repairs


def _trace_calls():
    return getattr(_TRACE, "calls", None)


def _attempts_bump():
    """One more call to the model for THIS page. Returns the new count.

    surya retries a page whose generation repeated itself, up to three times at
    rising temperature, and each attempt may run to its token cap. Measured on
    the Digesta orchestrator on 2026-10-13: 2,644 generated tokens per request
    against a page whose text is ~400, and a 25-27% multimodal cache hit rate on
    the pod — i.e. a quarter of all requests were re-sends of an image it had
    already read. None of it was visible from outside: a retried page looks
    exactly like a slow one.
    """
    n = getattr(_TRACE, "attempts", 0) + 1
    _TRACE.attempts = n
    return n


def _attempt_tokens(n: int) -> None:
    """How many tokens THIS attempt generated.

    The number that separates a page the model could not read from a page whose
    reply was cut off. surya-ocr-2 emits block markup -- a bbox and a tag per
    block before any of the text -- so a generation stopped at its cap yields
    markup that parses to a few blocks, or to none at all. "A few" is a page of
    ~71 characters; "none" is a page with no text whatsoever. Both look like a
    model that cannot read, and neither is.
    """
    try:
        lst = getattr(_TRACE, "tokens", None)
        if lst is None:
            lst = _TRACE.tokens = []
        lst.append(int(n or 0))
    except Exception:
        pass


def _attempts():
    return getattr(_TRACE, "attempts", 0)


def _attempt_token_counts():
    return list(getattr(_TRACE, "tokens", None) or ())


def _trace_repairs():
    return getattr(_TRACE, "repairs", None)


def _html_to_text(html: str) -> str:
    """Strip HTML tags → plain text (Surya2 returns block content as HTML)."""
    import re, html as _htmlmod
    if not html:
        return ""
    s = re.sub(r"(?i)<br\s*/?>", "\n", html)
    s = re.sub(r"(?i)</(p|div|li|tr|h[1-6])>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = _htmlmod.unescape(s)
    return "\n".join(ln.strip() for ln in s.splitlines() if ln.strip()).strip()


def _poly_to_xyxy(poly):
    try:
        pts = list(poly)
        if len(pts) == 4 and all(isinstance(p, (int, float)) for p in pts):
            return [float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3])]
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return [min(xs), min(ys), max(xs), max(ys)]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


def _lines_to_text(lines):
    if not lines:
        return ""
    ordered = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    rows = []
    for ln in ordered:
        y0 = ln["bbox"][1]
        placed = False
        for row in rows:
            ry = row[0]["bbox"][1]
            rh = max(1.0, row[0]["bbox"][3] - row[0]["bbox"][1])
            if abs(y0 - ry) <= 0.6 * rh:
                row.append(ln); placed = True; break
        if not placed:
            rows.append([ln])
    out = []
    for row in rows:
        row.sort(key=lambda l: l["bbox"][0])
        out.append(" ".join(l["text"] for l in row if l["text"]))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------

class PaddleWorker:
    def __init__(self, opts):
        self.opts = opts
        self._ocr = None
        self._structure = None

    def load(self):
        from paddleocr import PaddleOCR
        o = self.opts
        lang = o.get("lang", "it")
        use_gpu = bool(o.get("use_gpu", True))
        dev = "gpu" if use_gpu else "cpu"
        extra = {}
        if o.get("det_model_dir"):
            extra["det_model_dir"] = o["det_model_dir"]
        if o.get("rec_model_dir"):
            extra["rec_model_dir"] = o["rec_model_dir"]
        # PaddleOCR's constructor kwargs changed across 2.x→3.x (use_gpu→device,
        # use_angle_cls→use_textline_orientation, show_log removed). Try newest first,
        # peeling to a minimal ctor. It raises ValueError for unknown args (not TypeError).
        attempts = [
            dict(lang=lang, use_textline_orientation=True, device=dev, **extra),   # 3.x
            dict(lang=lang, device=dev, **extra),                                   # 3.x minimal
            dict(lang=lang, use_angle_cls=True, use_gpu=use_gpu, show_log=False, **extra),  # 2.x
            dict(lang=lang),
            dict(),
        ]
        last = None
        for kw in attempts:
            try:
                self._ocr = PaddleOCR(**kw)
                break
            except (TypeError, ValueError) as e:
                last = e
        if self._ocr is None:
            raise last
        # PP-Structure moved to PPStructureV3 / paddlex in 3.x; best-effort only.
        self._structure = None
        if o.get("structure", True):
            for modname, cls in (("paddleocr", "PPStructureV3"), ("paddleocr", "PPStructure")):
                try:
                    import importlib
                    C = getattr(importlib.import_module(modname), cls)
                    self._structure = C()
                    break
                except Exception:
                    continue

    def ocr(self, img):
        import numpy as np
        rgb = np.array(img.convert("RGB"))
        bgr = rgb[:, :, ::-1]
        h, w = rgb.shape[0], rgb.shape[1]
        lines = []
        for box, text, conf in self._run(bgr):
            lines.append({"text": text, "bbox": _poly_to_xyxy(box), "conf": conf})
        regions, tables = [], []
        if self._structure is not None:
            try:
                regions, tables = self._run_structure(bgr)
            except Exception:
                pass
        return {"index": 0, "width": int(w), "height": int(h),
                "lines": lines, "regions": regions, "tables": tables,
                "text": _lines_to_text(lines)}

    def _run(self, bgr):
        out = []
        ocr = self._ocr
        raw = None
        # 3.x uses .predict() (returns dict-like OCRResult per image); 2.x uses .ocr().
        if hasattr(ocr, "predict"):
            try:
                raw = ocr.predict(bgr)
            except Exception:
                raw = None
        if raw is None and hasattr(ocr, "ocr"):
            try:
                raw = ocr.ocr(bgr, cls=True)
            except TypeError:
                try:
                    raw = ocr.ocr(bgr)
                except Exception:
                    raw = None
            except Exception:
                raw = None
        for page in (raw or []):
            if page is None:
                continue
            # dict-like result (3.x OCRResult or a plain dict)
            texts = None
            try:
                texts = page.get("rec_texts")
            except Exception:
                texts = None
            if texts is not None:
                polys = None
                for k in ("dt_polys", "rec_polys"):
                    try:
                        polys = page.get(k)
                    except Exception:
                        polys = None
                    if polys is not None:
                        break
                try:
                    scores = page.get("rec_scores") or []
                except Exception:
                    scores = []
                for i, t in enumerate(texts):
                    box = polys[i] if (polys is not None and i < len(polys)) else [0, 0, 0, 0]
                    conf = float(scores[i]) if i < len(scores) else 0.0
                    out.append((box, str(t), conf))
                continue
            # 2.x nested-list result
            try:
                for item in page:
                    box = item[0]; txt = item[1]
                    if isinstance(txt, (list, tuple)):
                        out.append((box, str(txt[0]), float(txt[1])))
                    else:
                        out.append((box, str(txt), 0.0))
            except Exception:
                continue
        return out

    def _run_structure(self, bgr):
        regions, tables = [], []
        for reg in (self._structure(bgr) or []):
            try:
                label = str(reg.get("type", "region"))
                bbox = [float(v) for v in reg.get("bbox", [0, 0, 0, 0])]
                regions.append({"label": label, "bbox": bbox, "conf": float(reg.get("score", 0.0))})
                if label.lower() == "table":
                    r = reg.get("res")
                    tables.append({"bbox": bbox, "html": r.get("html") if isinstance(r, dict) else None})
            except Exception:
                continue
        return regions, tables


def _construct_tolerant(cls, kwargs):
    try:
        return cls(**kwargs)
    except TypeError:
        k = dict(kwargs)
        for bad in list(k.keys()):
            try:
                return cls(**k)
            except TypeError:
                k.pop(bad, None)
        return cls()


# ---------------------------------------------------------------------------
# Surya
# ---------------------------------------------------------------------------

#: What a model writes where it could not read a character.
UNREADABLE = "\ufffd"

#: At most this many anomalies are reported per page. A page that has gone
#: completely wrong would otherwise return its entire token stream.
TRACE_MAX = 40

#: Tokens of surrounding stream kept with each anomaly, each side.
TRACE_CONTEXT = 2


def _utf8_bad_offsets(buf) -> list:
    """Byte offsets in ``buf`` where the stream is not valid UTF-8.

    A real scan rather than a per-token guess, because per-token is meaningless
    here: surya-ocr-2's vocabulary is byte-level, so ONE character arrives as two
    or three tokens and every one of them is "invalid" in isolation. The first
    version of this checked each token against its immediate successor and duly
    flagged every accent and every em dash — the exact false positive that would
    make the diagnostic worthless.

    So the tokens are reassembled into the byte stream the server decoded, and the
    stream is walked as UTF-8: a lead byte declares its length, the continuations
    must be 0x80-0xBF, and anything else is a genuine break. Both the bytes that
    can never begin a character (0x80-0xC1, 0xF5+) and the sequences that run off
    the end are breaks.
    """
    bad = []
    i, n = 0, len(buf)
    while i < n:
        b = buf[i]
        if b < 0x80:                       # plain ASCII
            i += 1
            continue
        if 0xC2 <= b <= 0xDF:
            need = 1
        elif 0xE0 <= b <= 0xEF:
            need = 2
        elif 0xF0 <= b <= 0xF4:
            need = 3
        else:
            # 0x80-0xC1 is a continuation or an overlong lead; 0xF5+ is out of
            # range. Neither can start a character.
            bad.append(i)
            i += 1
            continue
        if i + need >= n:                  # runs off the end of the stream
            bad.append(i)
            i += 1
            continue
        if not all(0x80 <= buf[i + k] <= 0xBF for k in range(1, need + 1)):
            bad.append(i)
            i += 1
            continue
        i += need + 1
    return bad


def _tok_entry(entry) -> dict:
    """One OpenAI-style logprob entry reduced to what diagnoses a byte problem."""
    if not isinstance(entry, dict):
        entry = {"token": getattr(entry, "token", None),
                 "bytes": getattr(entry, "bytes", None),
                 "logprob": getattr(entry, "logprob", None)}
    raw = entry.get("bytes")
    out = {"token": entry.get("token")}
    if isinstance(raw, (list, tuple)):
        out["bytes"] = [int(b) for b in raw]
    lp = entry.get("logprob")
    if isinstance(lp, (int, float)):
        out["logprob"] = round(float(lp), 4)
    return out


def _ascii_skeleton(text: str) -> str:
    """``text`` with every non-ASCII character dropped.

    The guard that makes the repair provably safe. If two strings have the same
    ASCII skeleton they are the same text differing ONLY in their non-ASCII
    characters -- which is exactly the difference between a reply whose accents
    were destroyed and the same reply with them intact. Any other difference
    (a truncation, a different block, a shifted offset) changes the skeleton and
    the repair is refused.
    """
    return "".join(c for c in (text or "") if ord(c) < 128)


def _reassemble(tokens) -> str:
    """The model's own bytes, decoded strictly, or None if they are not clean.

    ``None`` on anything doubtful: missing bytes, a stream that does not decode,
    or a decode that still contains the marker. The repair must never make a
    reply worse, so every uncertain case leaves the original alone.
    """
    if not tokens:
        return None
    buf = bytearray()
    for t in tokens:
        e = _tok_entry(t)
        raw = e.get("bytes")
        if not isinstance(raw, list) or not raw:
            # EMPTY counts as missing, and getting this wrong once already
            # produced a confidently wrong answer. vLLM 0.27.1 reports
            # ``bytes: []`` for the byte-level tokens of surya-ocr-2's custom
            # vocabulary -- 257 of them on one page, with not a single byte
            # >= 0x80 anywhere in the stream. Accepting that as data built a
            # string with the accents silently absent, found no invalid UTF-8
            # in it, and called the result clean. Absence is not cleanliness.
            return None
        for b in raw:
            buf.append(int(b) & 0xFF)
    if not buf:
        return None
    try:
        text = bytes(buf).decode("utf-8")        # strict: errors are a refusal
    except UnicodeDecodeError:
        return None
    return None if UNREADABLE in text else text


#: vLLM renders a token it cannot JSON-encode as this, when it is launched with
#: ``--return-tokens-as-token-ids``. Those are exactly the tokens that lose
#: surya's accents: byte-level pieces that are not characters on their own.
_TOKEN_ID_PREFIX = "token_id:"


def _token_ids(tokens):
    """The raw vocabulary ids behind a reply, or None if they were not reported.

    This is the way out of the dead end the ``bytes`` field leads to. vLLM 0.27.1
    reports ``bytes: []`` for surya-ocr-2's byte-level tokens and renders them as
    U+FFFD in the text, so neither output carries the character -- but the IDS are
    not lossy, and surya's own tokenizer decodes them correctly
    (``decode([2131, 2104]) == 'e-grave'``, measured). vLLM will report them as
    ``token_id:2131`` when its server is launched with
    ``--return-tokens-as-token-ids``.

    All or nothing: a partial id list would decode to a partial page.
    """
    if not tokens:
        return None
    ids = []
    for t in tokens:
        tok = (t.get("token") if isinstance(t, dict)
               else getattr(t, "token", None))
        if not isinstance(tok, str) or not tok.startswith(_TOKEN_ID_PREFIX):
            return None
        try:
            ids.append(int(tok[len(_TOKEN_ID_PREFIX):]))
        except ValueError:
            return None
    return ids or None


def _decode_ids(ids):
    """``ids`` to text, using the VOCABULARY where the decoder fails.

    This is the whole fix, and it rests on a measured contradiction inside
    surya-ocr-2's own tokenizer:

        id 2160  decode -> U+FFFD      vocabulary piece -> 'a-grave'
        id 2168  decode -> U+FFFD      vocabulary piece -> 'e-grave'
        id 2185  decode -> U+FFFD      vocabulary piece -> 'u-grave'

    The model emits the RIGHT token -- 2160 really is a-grave -- and the decode
    table cannot render it. Every decoder in the chain inherits that: surya's own,
    and vLLM's, which reports ``decoded_token = None`` and therefore ``bytes: []``
    and writes U+FFFD into the text. So the accents were never lost in transport,
    at any resolution, on any vLLM version, with or without --trust-remote-code.
    They were lost in a decode table that disagrees with its own vocabulary.

    The vocabulary is the authority on what a token means, so when the decoder
    answers with the replacement character, the piece is used instead. Per id,
    because a bulk decode cannot be corrected after the fact.

    Imported lazily: the worker must start while the surya venv is still building,
    and a clean page must not pay for loading a tokenizer. Cached, because a
    document is many pages.
    """
    if not ids:
        return None
    tok = getattr(_decode_ids, "_tok", None)
    if tok is None:
        try:
            from transformers import AutoTokenizer
            ckpt = os.environ.get("SURYA_MODEL_CHECKPOINT") or "datalab-to/surya-ocr-2"
            tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
            _decode_ids._vocab = {v: k for k, v in tok.get_vocab().items()}
        except Exception as e:
            print(f"[surya] cannot load the tokenizer for id decoding: {e}",
                  file=sys.stderr)
            _decode_ids._tok = False
            return None
        _decode_ids._tok = tok
    if tok is False:
        return None
    inv = getattr(_decode_ids, "_vocab", None) or {}
    out = []
    try:
        for i in ids:
            piece = tok.decode([i], skip_special_tokens=True)
            if UNREADABLE in piece:
                # The decoder could not render this id. The vocabulary can.
                piece = inv.get(i, piece)
            out.append(piece)
    except Exception as e:
        print(f"[surya] id decoding failed: {e}", file=sys.stderr)
        return None
    return "".join(out)


def repair_content(raw: str, tokens):
    """``(repaired_text, characters_recovered)`` for a reply whose text lost
    characters, or ``(None, 0)`` to leave it alone.

    The measured fault this exists for: surya-ocr-2 emits perfectly good UTF-8 --
    the byte trace confirms the stream decodes cleanly and the model reports ~96%
    confidence -- and the text we read back from ``message.content`` has one
    U+FFFD per accented letter. The correct bytes are right there in the same
    response's logprobs, so the reply can simply be rebuilt from them.

    Guarded three ways, because silently rewriting a model's output is a serious
    thing to do: the reassembled stream must decode strictly, it must contain no
    marker of its own, and its ASCII skeleton must match the original's exactly.
    The last one is what makes this a repair rather than a substitution -- it can
    only ever change non-ASCII characters.
    """
    if not raw or UNREADABLE not in raw:
        return None, 0
    # IDS FIRST. They are the only lossless channel on vLLM 0.27.1, which reports
    # `bytes: []` for precisely the tokens whose characters went missing. The byte
    # route is kept because it needs no server flag and works wherever the bytes
    # are honestly reported.
    fixed = _decode_ids(_token_ids(tokens))
    if fixed is None or UNREADABLE in fixed:
        fixed = _reassemble(tokens)
    if fixed is None:
        return None, 0
    if _ascii_skeleton(fixed) != _ascii_skeleton(raw):
        return None, 0
    return fixed, raw.count(UNREADABLE)


def _trace_report(calls) -> dict:
    """``_trace_findings`` plus the one fact that makes it safe to read.

    An empty finding list has two completely different meanings:

      * logprobs ARRIVED and the byte stream was sound -> the loss happened after
        the model, so the serving stack is at fault. A real verdict.
      * logprobs never arrived -> there was nothing to analyse, and the emptiness
        says nothing at all.

    Reporting a bare list would make those indistinguishable, and the second one
    would read as the first: a confident conclusion drawn from missing data. So
    the envelope always says how many calls were seen and how many carried
    logprobs, and ``conclusive`` is false whenever they did not.

    (surya asks for them by default -- ``SURYA_INFERENCE_LOGPROBS = True`` -- and
    coderai's OCR gateway forwards the body verbatim precisely so they survive.
    But an upstream that declines them is a configuration, not a bug, and has to
    be visible rather than silently inverted into a verdict.)
    """
    calls = list(calls or [])
    with_lp = sum(1 for c in calls if c)
    # How many tokens the server declined to give bytes for. A token with
    # ``bytes: []`` carries no information, and a stream full of them decodes
    # "cleanly" only because there is nothing in it to be wrong -- so this has to
    # be counted and reported, not quietly skipped.
    no_bytes = 0
    for c in calls:
        for t in (c or []):
            e = _tok_entry(t)
            raw = e.get("bytes")
            if not isinstance(raw, list) or not raw:
                no_bytes += 1
    report = {
        "calls": len(calls),
        "with_logprobs": with_lp,
        "tokens_without_bytes": no_bytes,
        "findings": _trace_findings(calls) if with_lp else [],
    }
    if not calls:
        report["conclusive"] = False
        report["why"] = ("no model calls were recorded -- the trace hook did not "
                         "fire, so this says nothing about the cause")
    elif not with_lp:
        report["conclusive"] = False
        report["why"] = ("the model server returned no logprobs, so the bytes "
                         "could not be read -- this says nothing about the cause")
    elif no_bytes:
        # The decisive case on vLLM 0.27.1 + surya-ocr-2: the server answers with
        # logprobs but gives ``bytes: []`` for the very tokens whose characters
        # went missing, so there is nothing to check and nothing to rebuild from.
        # That is still an ANSWER -- it localises the loss inside the server,
        # which produced neither usable bytes nor usable text -- but it is not
        # the same statement as "the model's bytes were sound".
        report["conclusive"] = False
        report["why"] = (
            f"the server reported no bytes for {no_bytes} token(s), including the "
            f"ones whose characters are missing, so the byte stream could not be "
            f"checked. The loss is inside the serving stack -- it produced neither "
            f"usable bytes nor usable text -- but this cannot say whether the "
            f"MODEL's own output was sound, and nothing can be rebuilt from it")
    else:
        report["conclusive"] = True
        if not report["findings"]:
            report["why"] = ("the byte stream decoded cleanly and every token "
                             "reported its bytes, so the characters were lost "
                             "AFTER the model: the fault is in the serving path, "
                             "not the model")
        else:
            kinds = {}
            for f in report["findings"]:
                kinds[f.get("kind")] = kinds.get(f.get("kind"), 0) + 1
            report["kinds"] = kinds
            report["why"] = ("the model's own byte stream is broken at the places "
                             "listed: the model is at fault, not the serving path")
    return report


def _trace_findings(calls) -> list:
    """Where in the model's own token stream the text stopped being readable.

    This is the measurement that separates two faults which look identical in the
    finished text:

      * the MODEL emitted a broken byte sequence -- a lone UTF-8 lead byte with no
        continuation, say -- in which case the text faithfully renders a wrong
        answer and nothing on our side can fix it;
      * the model emitted a CORRECT sequence and the serving stack mis-assembled
        it, in which case the model was right and the bug is ours.

    A single marker per accented letter is equally the signature of both, because
    the vocabulary is byte-level and either byte of an accented letter decodes
    alone to exactly one replacement character. Only the bytes tell them apart.

    Two kinds of finding, and the difference matters:

      ``invalid-utf8``  the reassembled stream does not decode. The server had
                        nothing good to work with, so the model is at fault.
      ``model-emitted`` the stream decodes cleanly and a token IS the replacement
                        character -- the model chose it, or something upstream
                        had already substituted it before we saw a byte.

    What is NOT a finding: a stream that decodes. If the bytes are sound and the
    text still shows markers, the loss happened after this point, which is itself
    the answer -- an empty trace on an unreadable page indicts the stack.
    """
    findings = []
    for call_idx, tokens in enumerate(calls or []):
        if not tokens:
            continue
        entries = [_tok_entry(t) for t in tokens]

        # Reassemble the stream, remembering which token each byte came from.
        buf, owner = bytearray(), []
        for ti, e in enumerate(entries):
            for b in (e.get("bytes") or []):
                buf.append(b & 0xFF)
                owner.append(ti)

        def _add(ti, why, kind):
            if len(findings) >= TRACE_MAX:
                return False
            lo = max(0, ti - TRACE_CONTEXT)
            hi = min(len(entries), ti + TRACE_CONTEXT + 1)
            findings.append({
                "call": call_idx, "index": ti, "kind": kind,
                "token": entries[ti], "why": why, "context": entries[lo:hi],
            })
            return True

        seen = set()
        for off in _utf8_bad_offsets(buf):
            ti = owner[off] if off < len(owner) else len(entries) - 1
            if ti in seen:
                continue
            seen.add(ti)
            if not _add(ti, "byte 0x%02X does not form valid UTF-8 here" % buf[off],
                        "invalid-utf8"):
                return findings

        # Markers the model produced outright. EF BF BD is perfectly valid UTF-8,
        # so the scan above cannot see these.
        for ti, e in enumerate(entries):
            tok = e.get("token")
            if isinstance(tok, str) and UNREADABLE in tok and ti not in seen:
                seen.add(ti)
                if not _add(ti, "the token itself is the replacement character",
                            "model-emitted"):
                    return findings
    return findings



def _at_cap(tokens: int, cap: int) -> bool:
    """Did this reply stop because it ran out of room rather than finishing?

    `finish_reason` would say so directly, but surya's GenerationResult does not
    carry it — it keeps `raw`, `token_count`, `error`, `mean_token_prob` and
    `logprobs` and drops the rest of the completion. The token count against the
    cap is the same signal one step removed, and 98% rather than equality
    because a server may stop a token or two short.
    """
    return bool(cap) and int(tokens or 0) >= int(cap * 0.98)


def _continue_reply(oc, result, item, client, model_name, cap, temperature,
                    top_p, timeout, want_logprobs, budget):
    """Finish a page whose reply hit the cap, by resuming the generation.

    Raising the cap only moves the cliff: a dense enough page overflows any
    fixed number, and what comes back is incomplete block markup that parses to
    a fragment or to nothing (see `_at_cap`'s callers). The only answer that
    holds for every document is to carry on generating.

    Resumption is a **prefilled assistant turn**: the text so far goes back as
    the assistant's own message with `continue_final_message`, so the model
    continues the exact string instead of starting a new answer about the same
    image. A server that does not understand that flag gets a plain "continue"
    instruction instead, and a server that understands neither leaves the page
    partial and honestly marked.

    The logprobs are concatenated with the text, in step. They are not a
    diagnostic here: `surya_repair_accents` rebuilds characters from the model's
    own per-token bytes, so text from a later segment paired with logprobs from
    an earlier one would corrupt exactly the accented characters it exists to
    fix.
    """
    segments = [getattr(result, "raw", "") or ""]
    logprobs = list(getattr(result, "logprobs", None) or [])
    total = int(getattr(result, "token_count", 0) or 0)
    last = total
    rounds = 0
    try:
        prompt = item.prompt or oc.PROMPT_MAPPING[item.prompt_type]
        base_messages = oc._build_messages(oc.scale_to_fit(item.image), prompt)
    except Exception as exc:
        print(f"[surya] cannot resume a cut-off page: {exc}", file=sys.stderr, flush=True)
        return result, 0, True

    attempts = 0
    while attempts < budget and _at_cap(last, cap):
        attempts += 1
        prefix = "".join(segments)
        kwargs = dict(model=model_name, messages=base_messages + [
            {"role": "assistant", "content": prefix}],
            max_tokens=cap, temperature=temperature, top_p=top_p, timeout=timeout,
            extra_body={"continue_final_message": True,
                        "add_generation_prompt": False})
        if want_logprobs:
            kwargs["logprobs"] = True
        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as exc:
            # The flag is a vLLM extension; llama.cpp and others reject it.
            try:
                kwargs.pop("extra_body", None)
                kwargs["messages"] = base_messages + [
                    {"role": "assistant", "content": prefix},
                    {"role": "user", "content": "Continue exactly where you "
                                                "stopped. Do not repeat."}]
                completion = client.chat.completions.create(**kwargs)
            except Exception as exc2:
                print(f"[surya] could not resume a cut-off page "
                      f"({type(exc).__name__}: {exc}; then {exc2}) — returning it "
                      f"partial and marked", file=sys.stderr, flush=True)
                break
        try:
            more = completion.choices[0].message.content or ""
            last = (completion.usage.completion_tokens
                    if getattr(completion, "usage", None) else 0)
        except Exception:
            break
        if not more:
            break
        segments.append(more)
        # Counted only now. `rounds` is reported to the caller as "the page was
        # resumed n times, so it is whole" — an ATTEMPT that failed must not
        # make that claim.
        rounds += 1
        total += int(last or 0)
        if want_logprobs:
            try:
                lp = getattr(completion.choices[0], "logprobs", None)
                content = getattr(lp, "content", None) if lp is not None else None
                for c in (content or []):
                    logprobs.append(c.model_dump() if hasattr(c, "model_dump") else c)
            except Exception:
                # Text without its bytes would mis-align the accent repair, so
                # the repair is given up on rather than fed a mismatch.
                logprobs = []
        print(f"[surya] page resumed after the {cap}-token cap "
              f"(segment {rounds + 1}, {last} more tokens)",
              file=sys.stderr, flush=True)

    if rounds:
        try:
            result.raw = "".join(segments)
            result.token_count = total
            if logprobs:
                result.logprobs = logprobs
        except Exception:
            pass
    return result, rounds, _at_cap(last, cap)


class SuryaWorker:
    def __init__(self, opts):
        self.opts = opts
        self._mode = None
        self._rec = None
        self._det = None
        self._det_proc = None
        self._rec_proc = None
        self._run_ocr = None
        self._langs = [l.strip() for l in str(opts.get("langs", "it")).split(",") if l.strip()] or ["it"]
        self._trace = bool(opts.get("token_trace", False))
        #: Rebuild a reply's text from the model's own bytes when the text we are
        #: handed has lost characters. On by default: the alternative is shipping
        #: text we have MEASURED to be corrupt (see repair_content).
        self._repair = bool(opts.get("repair_accents", True))
        #: Our ceiling on surya's own page retries (None = surya's 3). Written to
        #: the module cell because the hook that reads it is module-level.
        budget = opts.get("max_retries", None)
        _RETRY_BUDGET[0] = None if budget is None else max(0, int(budget))
        #: 0 leaves surya's own 2048 in place.
        _MAX_TOKENS[0] = max(0, int(opts.get("max_tokens", 0) or 0))
        #: Resumptions allowed for one page. 0 = return a cut-off page partial.
        _CONTINUE_BUDGET[0] = max(0, int(opts.get("max_continuations", 2) or 0))
        #: The page's token logprobs and accent repairs live in _TRACE, per
        #: thread — see _trace_begin(). They were per-worker-object until lanes
        #: put several pages in one process.

    def load(self):
        """Load surya, matching the API of whichever release is installed.

        Three layouts exist. They are told apart by what IMPORTS — not by
        catching whatever went wrong and moving on. The old version of this
        caught Exception around each tier, so a real failure inside the right
        tier (a config attribute missing under a newer transformers) was
        swallowed, the loader fell through two tiers, and the error reported
        was from an API that has not existed for a year: "No module named
        surya.ocr". The cause was invisible.
        """
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        try:
            from surya.foundation import FoundationPredictor      # 0.17.x only
        except ImportError:
            FoundationPredictor = None
        if FoundationPredictor is not None:
            # 0.17.x: recognition rides a shared VLM foundation.
            self._rec = RecognitionPredictor(FoundationPredictor())
        else:
            # 0.6–0.16 and 0.22+: no foundation; RecognitionPredictor() alone.
            # (0.22 dropped surya.foundation and takes an optional inference
            # manager instead — the no-arg form is right for both.)
            self._rec = RecognitionPredictor()
        self._det = DetectionPredictor()
        self._mode = "predictor"
        self._install_trace_hook()
        return

    def _install_trace_hook(self):
        """Keep the per-token logprobs surya asks for but then throws away.

        surya's recognition predictor DOES pass them on
        (``BlockOCRResult(..., raw_logprobs=out.logprobs)``) but the model never
        declares the field, and pydantic's default is ``extra="ignore"`` -- so the
        data is constructed and silently dropped. Their own comment calls it
        "phase 2 use", which is presumably why nobody noticed. Verified by
        construction: the attribute does not exist on the result.

        So the value is taken one layer earlier, where it still exists:
        ``_generate_one`` returns the ``GenerationResult`` that carries it. The
        wrapper only reads, never alters, and swallows everything -- a diagnostic
        that can break OCR would be far worse than no diagnostic.
        """
        if not (self._trace or self._repair):
            return
        try:
            from surya.inference.backends import openai_client as _oc
        except Exception as e:
            print(f"[surya] token trace unavailable: {e}", file=sys.stderr)
            return
        _oc._coderai_repair = self._repair
        if getattr(_oc, "_coderai_traced", False):
            # Patched once per process. Nothing to re-point: the sink is the
            # calling THREAD's, so every lane lands in its own page's trace.
            return
        inner = getattr(_oc, "_generate_one", None)
        if inner is None:
            print("[surya] token trace: no _generate_one to hook", file=sys.stderr)
            return

        def _wrapped(*args, **kwargs):
            # surya's vLLM backend never passes `max_tokens_default`, so its
            # 2048 was the only value available -- the same reason `max_retries`
            # needed hooking. It arrives here as a keyword, so it can be raised
            # without touching surya.
            cap = _MAX_TOKENS[0]
            if cap and "max_tokens_default" in kwargs:
                kwargs["max_tokens_default"] = int(cap)
            result = inner(*args, **kwargs)
            # A reply that ran out of room is finished here, before surya parses
            # it: raising the cap only moves the cliff, and incomplete block
            # markup parses to a fragment or to nothing at all.
            try:
                budget = int(_CONTINUE_BUDGET[0] or 0)
                item = args[0] if args else kwargs.get("item")
                eff_cap = int(getattr(item, "max_tokens", 0) or 0) or int(
                    kwargs.get("max_tokens_default") or 0)
                if (budget and item is not None and eff_cap
                        and _at_cap(getattr(result, "token_count", 0), eff_cap)):
                    result, rounds, still = _continue_reply(
                        _oc, result, item,
                        kwargs.get("client"), kwargs.get("model_name"), eff_cap,
                        0.0, kwargs.get("top_p", 0.1), kwargs.get("timeout", 600.0),
                        bool(kwargs.get("request_logprobs_default", True)),
                        budget)
                    _TRACE.continued = getattr(_TRACE, "continued", 0) + rounds
                    _TRACE.still_truncated = bool(still)
            except Exception as exc:
                print(f"[surya] resume failed: {exc}", file=sys.stderr, flush=True)
            try:
                _attempts_bump()
                _attempt_tokens(getattr(result, "token_count", 0))
                sink = _trace_calls()
                if sink is not None:
                    sink.append(getattr(result, "logprobs", None))
                # Repaired HERE, before surya parses the reply, so the blocks, the
                # HTML, the plain text and the confidences downstream are all built
                # from correct characters. Repairing further down would mean fixing
                # the same text in several places and missing one.
                if getattr(_oc, "_coderai_repair", False):
                    fixed, n = repair_content(getattr(result, "raw", ""),
                                              getattr(result, "logprobs", None))
                    if fixed is not None:
                        result.raw = fixed
                        reps = _trace_repairs()
                        if reps is not None:
                            reps.append({"recovered": n})
            except Exception:
                # A repair that can break a page would be worse than the corruption
                # it fixes; the untouched reply is always an acceptable answer.
                pass
            return result

        _oc._generate_one = _wrapped

        # surya decides on its own to re-run a page whose generation repeated
        # itself: up to `max_retries` (3) more calls, each at a higher
        # temperature, each able to run to its token cap. On a page the model
        # genuinely cannot read that is 4 x 2048 tokens of GPU time to arrive at
        # the same unreadable answer — and the repeat detector cannot tell
        # "repeated because the model broke" from "repeated because the page
        # really does say the same thing twenty times", which a tax table does.
        #
        # The retry stays (it does rescue pages), but the budget is ours and the
        # spend is logged. `max_retries` is not reachable from surya's settings:
        # the vLLM backend never passes it, so the default was the only value
        # available. Hooking the decision is the only seam there is.
        inner_retry = getattr(_oc, "_should_retry", None)
        if inner_retry is not None:
            def _wrapped_retry(result, retries, max_retries, *a, **kw):
                budget = _RETRY_BUDGET[0]
                if budget is not None:
                    max_retries = min(int(max_retries or 0), int(budget))
                again = inner_retry(result, retries, max_retries, *a, **kw)
                if again:
                    print(f"[surya] page repeated itself — retry "
                          f"{retries + 1}/{max_retries} (temperature rises; a "
                          f"retried page costs another full generation)",
                          file=sys.stderr, flush=True)
                return again
            _oc._should_retry = _wrapped_retry
        _oc._coderai_traced = True

    def _load_legacy(self):
        # very old (<0.6): the functional run_ocr API. Kept reachable but no
        # longer a silent fallback — a modern install never gets here.
        from surya.ocr import run_ocr
        from surya.model.detection.model import (
            load_model as ldm, load_processor as ldp)
        from surya.model.recognition.model import load_model as lrm
        from surya.model.recognition.processor import load_processor as lrp
        self._run_ocr = run_ocr
        self._det = ldm(); self._det_proc = ldp()
        self._rec = lrm(); self._rec_proc = lrp()
        self._mode = "run_ocr"

    def _predict(self, img):
        # Predictor API drifted across classic surya versions:
        #   0.17.x: rec(images, task_names, det_predictor)   (task_names defaults internally)
        #   0.6.x : rec(images, langs, det_predictor)
        attempts = [
            lambda: self._rec([img]),                                           # 0.20+ (VLM full-page, backend does it)
            lambda: self._rec([img], full_page=True),                           # 0.20+ variant
            lambda: self._rec([img], det_predictor=self._det),                  # 0.17.x (task_names default)
            lambda: self._rec([img], ["ocr_with_boxes"], self._det),            # 0.17.x explicit
            lambda: self._rec([img], [self._langs], self._det),                 # 0.6.x (langs)
        ]
        last = None
        for call in attempts:
            try:
                return call()
            except (TypeError, ValueError, AssertionError) as e:
                last = e
        raise last

    def ocr(self, img):
        img = img.convert("RGB")
        # One page per call, so the trace is started here rather than accumulated:
        # a long document would otherwise report the first page's anomalies on the
        # hundredth. Per-thread, because several pages share this process.
        calls, repairs = _trace_begin()
        if self._mode == "predictor":
            preds = self._predict(img)
        else:
            preds = self._run_ocr([img], [self._langs], self._det, self._det_proc, self._rec, self._rec_proc)
        res = preds[0] if preds else None
        lines = []
        tl = getattr(res, "text_lines", None)
        if tl is None and isinstance(res, dict):
            tl = res.get("text_lines")
        if tl:
            # Classic surya (<=0.17): result.text_lines[].{text,bbox,confidence}
            for t in tl:
                text = getattr(t, "text", None)
                bbox = getattr(t, "bbox", None)
                conf = getattr(t, "confidence", 0.0)
                if text is None and isinstance(t, dict):
                    text = t.get("text"); bbox = t.get("bbox"); conf = t.get("confidence", 0.0)
                if not text:
                    continue
                bb = [float(v) for v in (bbox or [0, 0, 0, 0])][:4]
                if len(bb) < 4:
                    bb = [0.0, 0.0, 0.0, 0.0]
                lines.append({"text": str(text), "bbox": bb, "conf": float(conf or 0.0)})
        else:
            # Surya2 (>=0.20): result.blocks[].{html,polygon,confidence,label,reading_order}
            blocks = getattr(res, "blocks", None)
            if blocks is None and isinstance(res, dict):
                blocks = res.get("blocks")
            for b in sorted(blocks or [], key=lambda x: getattr(x, "reading_order", 0) or 0):
                html = getattr(b, "html", None)
                if html is None and isinstance(b, dict):
                    html = b.get("html")
                text = _html_to_text(html or "")
                if not text:
                    continue
                poly = getattr(b, "polygon", None) or (b.get("polygon") if isinstance(b, dict) else None)
                conf = getattr(b, "confidence", 0.0)
                if isinstance(b, dict):
                    conf = b.get("confidence", conf)
                lines.append({"text": text, "bbox": _poly_to_xyxy(poly or [0, 0, 0, 0]),
                              "conf": float(conf or 0.0)})
        page = {"index": 0, "width": img.width, "height": img.height,
                "lines": lines, "regions": [], "tables": [],
                "text": "\n".join(l["text"] for l in lines)}
        # Only when something actually went unread: a clean page stays byte-identical
        # to what it has always returned.
        if repairs:
            # Said out loud rather than silently: the text the caller gets is not
            # byte-for-byte what the server sent, and anyone comparing the two
            # needs to know why.
            page["accents_repaired"] = {
                "blocks": len(repairs),
                "recovered": sum(int(r.get("recovered", 0)) for r in repairs),
                "how": ("rebuilt from the model's own per-token bytes, which "
                        "decoded cleanly where message.content did not"),
            }
        # A page the model had to be asked for more than once. Reported always,
        # not only under token_trace: this is the difference between a page that
        # was slow and a page that was generated two, three or four times over,
        # and without it the only visible symptom is a batch that produces far
        # fewer completions than the GPU's throughput says it should.
        tries = _attempts()
        toks = _attempt_token_counts()
        cap = int(_MAX_TOKENS[0] or _SURYA_DEFAULT_MAX_TOKENS)
        # Reported whenever there is anything to report -- a retry, or a reply
        # that stopped at its cap. A page cut off mid-markup parses to a few
        # blocks or to none, which is indistinguishable in the finished text
        # from a page the model genuinely could not read. This is the only place
        # the difference is visible.
        capped = [t for t in toks if t >= cap * 0.98]
        resumed = int(getattr(_TRACE, "continued", 0) or 0)
        # Still short of the end after every resumption we were allowed. The
        # caller needs this to refuse a partial page, so it is reported
        # separately from "a segment hit the cap", which resumption may have
        # since fixed.
        incomplete = bool(getattr(_TRACE, "still_truncated", False))
        if tries > 1 or capped or resumed:
            info = {"calls": tries or 1, "tokens": toks, "cap": cap}
            if resumed:
                info["resumed"] = resumed
                info["why"] = (f"the reply ran past the {cap}-token cap and was "
                               f"resumed {resumed} time(s) from where it stopped, "
                               f"so the page is whole")
            if incomplete:
                info["truncated"] = True
                info["complete"] = False
                info["why"] = (
                    f"the reply stopped at the {cap}-token cap and was still "
                    f"unfinished after {resumed} resumption(s), so this page is "
                    f"INCOMPLETE. It is a cut-off reply, NOT a page the model "
                    f"could not read -- raise ocr.surya_max_continuations or "
                    f"ocr.surya_max_tokens.")
            elif capped and not resumed:
                info["truncated"] = True
                info["complete"] = False
                info["why"] = (
                    f"the reply stopped at the {cap}-token cap and resumption is "
                    f"off (ocr.surya_max_continuations = 0), so this page is "
                    f"INCOMPLETE. A cut-off reply, NOT a page the model could "
                    f"not read.")
            elif not resumed:
                info["why"] = ("surya re-ran this page because the generation "
                               "repeated itself; each re-run is a full "
                               "generation at a higher temperature")
            page["model_attempts"] = info
        if self._trace and UNREADABLE in page["text"]:
            try:
                page["unreadable_trace"] = _trace_report(calls)
            except Exception as e:
                # Always a report, never an absence: "the analysis broke" and "the
                # bytes were fine" must not look the same.
                page["unreadable_trace"] = {
                    "conclusive": False,
                    "why": f"trace analysis failed: {e}",
                    "findings": [],
                }
        return page


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class _Lanes:
    """Several pages in flight in ONE worker process.

    The protocol was one request, one reply, strictly in order — so pages in
    flight WAS the process count, and a process costs ~600 MB. In the served
    modes (`surya_serve: model`) that memory buys nothing: the page is rasterised
    here and recognised on a GPU somewhere else, so the worker spends its life
    waiting on a socket. Raising concurrency by forking more of them is how
    Aruba ended up with 64 processes, 37 GB and a nine-minute stall.

    So a lane is a worker OBJECT, not a process: each has its own predictor and
    its own page, and they share the one torch import. In the served modes a
    predictor holds no weights (1.7 MB of tokenizer and config on disk), which is
    what makes this nearly free. Lanes are built on demand, so a quiet install
    pays for exactly one.
    """

    def __init__(self, make, lanes: int, first):
        self._make = make
        self._limit = max(1, int(lanes or 1))
        self._idle = queue.Queue()
        self._idle.put(first)
        self._built = 1
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(self._limit)

    @property
    def limit(self) -> int:
        return self._limit

    def borrow(self):
        """A lane of our own, waiting for one if every lane is busy."""
        self._slots.acquire()
        try:
            try:
                return self._idle.get_nowait()
            except queue.Empty:
                pass
            with self._lock:
                worker = self._make()
                self._built += 1
            return worker
        except BaseException:
            self._slots.release()      # never strand a slot on a failed build
            raise

    def give_back(self, worker) -> None:
        if worker is not None:
            self._idle.put(worker)
        self._slots.release()


def _build(engine: str, opts: dict):
    worker = PaddleWorker(opts) if engine == "paddle" else SuryaWorker(opts)
    worker.load()
    return worker


def _serve_page(lanes: "_Lanes", msg: dict, engine: str) -> None:
    """One page, on a lane of its own, answered with the id it was asked under."""
    rid = msg.get("id")
    worker = None
    try:
        worker = lanes.borrow()
        from PIL import Image
        data = base64.b64decode(msg["image"])
        img = Image.open(io.BytesIO(data)); img.load()
        _emit({"ok": True, "id": rid, "page": worker.ocr(img)})
    except Exception as e:
        _emit({"ok": False, "id": rid, "error": f"{engine} ocr failed: {e}",
               "status": 500})
    finally:
        if worker is not None:
            lanes.give_back(worker)


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "paddle"
    from PIL import Image  # noqa: F401 (ensure Pillow present before READY)
    _emit({"ready": True})

    lanes = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            _emit({"ok": False, "error": f"bad json: {e}"}); continue
        cmd = msg.get("cmd")
        if cmd == "ping":
            _emit({"ok": True}); continue
        if cmd == "load":
            try:
                opts = msg.get("opts") or {}
                # The first lane is loaded HERE, synchronously, so a missing
                # dependency is still a load failure and not a failed page.
                first = _build(engine, opts)
                lanes = _Lanes(lambda: _build(engine, opts),
                               int(opts.get("lanes") or 1), first)
                _emit({"ok": True, "lanes": lanes.limit})
            except Exception as e:
                _emit({"ok": False, "error": f"{engine} load failed: {e}", "status": 503})
            continue
        if cmd == "ocr":
            if lanes is None:
                _emit({"ok": False, "id": msg.get("id"), "error": "not loaded",
                       "status": 500}); continue
            if msg.get("id") is None:
                # No id: the caller is reading replies in order, one at a time.
                worker = None
                try:
                    worker = lanes.borrow()
                    from PIL import Image
                    data = base64.b64decode(msg["image"])
                    img = Image.open(io.BytesIO(data)); img.load()
                    _emit({"ok": True, "page": worker.ocr(img)})
                except Exception as e:
                    _emit({"ok": False, "error": f"{engine} ocr failed: {e}",
                           "status": 500})
                finally:
                    if worker is not None:
                        lanes.give_back(worker)
                continue
            # Answered out of order, under its id, while the loop reads the next.
            threading.Thread(target=_serve_page, args=(lanes, msg, engine),
                             daemon=True).start()
            continue
        if cmd == "shutdown":
            _emit({"ok": True}); return
        _emit({"ok": False, "error": f"unknown cmd {cmd}"})


if __name__ == "__main__":
    main()
