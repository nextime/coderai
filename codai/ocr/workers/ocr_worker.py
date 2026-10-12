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
import io
import json
import sys


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


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
    report = {
        "calls": len(calls),
        "with_logprobs": with_lp,
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
    else:
        report["conclusive"] = True
        if not report["findings"]:
            report["why"] = ("the byte stream decoded cleanly, so the characters "
                             "were lost AFTER the model: the fault is in the "
                             "serving path, not the model")
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
        #: Per-call token logprobs, collected by the hook installed in load().
        self._calls = []
        self._trace = bool(opts.get("token_trace", False))

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
        if not self._trace:
            return
        try:
            from surya.inference.backends import openai_client as _oc
        except Exception as e:
            print(f"[surya] token trace unavailable: {e}", file=sys.stderr)
            return
        if getattr(_oc, "_coderai_traced", False):
            _oc._coderai_sink = self._calls          # re-point at this worker
            return
        inner = getattr(_oc, "_generate_one", None)
        if inner is None:
            print("[surya] token trace: no _generate_one to hook", file=sys.stderr)
            return
        _oc._coderai_sink = self._calls

        def _wrapped(*args, **kwargs):
            result = inner(*args, **kwargs)
            try:
                sink = getattr(_oc, "_coderai_sink", None)
                if sink is not None:
                    sink.append(getattr(result, "logprobs", None))
            except Exception:
                pass
            return result

        _oc._generate_one = _wrapped
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
        # One page per call, so the trace is cleared here rather than accumulated:
        # a long document would otherwise report the first page's anomalies on the
        # hundredth.
        del self._calls[:]
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
        if self._trace and UNREADABLE in page["text"]:
            try:
                page["unreadable_trace"] = _trace_report(self._calls)
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

def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "paddle"
    from PIL import Image  # noqa: F401 (ensure Pillow present before READY)
    _emit({"ready": True})

    worker = None
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
                worker = PaddleWorker(opts) if engine == "paddle" else SuryaWorker(opts)
                worker.load()
                _emit({"ok": True})
            except Exception as e:
                _emit({"ok": False, "error": f"{engine} load failed: {e}", "status": 503})
            continue
        if cmd == "ocr":
            if worker is None:
                _emit({"ok": False, "error": "not loaded", "status": 500}); continue
            try:
                from PIL import Image
                data = base64.b64decode(msg["image"])
                img = Image.open(io.BytesIO(data)); img.load()
                _emit({"ok": True, "page": worker.ocr(img)})
            except Exception as e:
                _emit({"ok": False, "error": f"{engine} ocr failed: {e}", "status": 500})
            continue
        if cmd == "shutdown":
            _emit({"ok": True}); return
        _emit({"ok": False, "error": f"unknown cmd {cmd}"})


if __name__ == "__main__":
    main()
