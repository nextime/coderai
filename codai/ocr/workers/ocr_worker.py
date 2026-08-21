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

    def load(self):
        from surya.detection import DetectionPredictor
        # 0.17.x: RecognitionPredictor(FoundationPredictor()) — a shared VLM foundation.
        try:
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            self._rec = RecognitionPredictor(FoundationPredictor())
            self._det = DetectionPredictor()
            self._mode = "predictor"
            return
        except Exception:
            pass
        # 0.6–0.16: RecognitionPredictor() with no args.
        try:
            from surya.recognition import RecognitionPredictor
            self._rec = RecognitionPredictor()
            self._det = DetectionPredictor()
            self._mode = "predictor"
            return
        except Exception:
            pass
        # very old: functional run_ocr API.
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
        if self._mode == "predictor":
            preds = self._predict(img)
        else:
            preds = self._run_ocr([img], [self._langs], self._det, self._det_proc, self._rec, self._rec_proc)
        res = preds[0] if preds else None
        lines = []
        tl = getattr(res, "text_lines", None)
        if tl is None and isinstance(res, dict):
            tl = res.get("text_lines")
        for t in (tl or []):
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
        return {"index": 0, "width": img.width, "height": img.height,
                "lines": lines, "regions": [], "tables": [], "text": "\n".join(l["text"] for l in lines)}


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
