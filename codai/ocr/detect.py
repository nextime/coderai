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

"""Stamp / signature detection for the OCR subsystem.

Configurable via ``ocr.detect_mode``:

- ``off``       — no detection.
- ``layout``    — reuse the OCR engine's layout regions (PP-Structure etc.); non-text
  regions (figure/seal/stamp) are flagged as candidate stamps. No extra model; coarse,
  and signatures are rarely separable this way.
- ``detector``  — run a small YOLO model (``ocr.detect_model_path``) trained to spot
  signatures/stamps; returns tight boxes + confidence + class.
- ``both``      — run both and merge.

The detector model is loaded once per weights path and reused across requests.
"""

import threading
from typing import Dict, List, Optional

from codai.ocr.base import OcrPage, OcrError


VALID_MODES = ("off", "layout", "detector", "both")

# Class-name → bucket heuristics (English + Italian markers).
_SIGNATURE_MARKERS = ("sign", "signature", "firma", "autograph")
_STAMP_MARKERS = ("stamp", "seal", "timbro", "sigillo", "bollo")
# Layout labels (PP-Structure / docTR / Surya) that plausibly cover a stamp/seal.
_LAYOUT_STAMP_LABELS = ("figure", "seal", "stamp", "image", "graphic")


def resolve_detect_mode(cfg, override: Optional[str]) -> str:
    """Per-request override wins over the configured default; validated."""
    mode = (override or getattr(cfg, "detect_mode", "off") or "off").strip().lower()
    if mode not in VALID_MODES:
        raise OcrError(f"invalid detect mode '{mode}' (use {'/'.join(VALID_MODES)})", status=400)
    return mode


def _classify(label: str) -> str:
    lo = (label or "").lower()
    if any(m in lo for m in _SIGNATURE_MARKERS):
        return "signature"
    if any(m in lo for m in _STAMP_MARKERS):
        return "stamp"
    return "stamp"   # unknown detector classes default to the stamp bucket (label kept)


# ---------------------------------------------------------------------------
# YOLO detector (lazy, cached per weights path)
# ---------------------------------------------------------------------------

_detector_cache: Dict[str, "YoloDetector"] = {}
_cache_lock = threading.Lock()


class YoloDetector:
    def __init__(self, model_path: str, conf: float):
        self.model_path = model_path
        self.conf = conf
        self._model = None
        self._load_lock = threading.Lock()

    def ensure_loaded(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from ultralytics import YOLO
            except Exception as e:
                raise OcrError(
                    "ultralytics is not installed (needed for detector mode). Install: "
                    f"`pip install ultralytics`. Import error: {e}",
                    status=503,
                )
            import os
            if not self.model_path or not os.path.isfile(self.model_path):
                raise OcrError(
                    "stamp/signature detector weights not found. Set ocr.detect_model_path "
                    "to a YOLO .pt file trained on signatures/stamps.",
                    status=400,
                )
            try:
                self._model = YOLO(self.model_path)
            except Exception as e:
                raise OcrError(f"failed to load YOLO detector: {e}", status=500)

    def detect(self, image) -> List[dict]:
        self.ensure_loaded()
        import numpy as np

        arr = np.array(image.convert("RGB"))
        out: List[dict] = []
        try:
            results = self._model.predict(arr, conf=self.conf, verbose=False)
        except Exception as e:
            raise OcrError(f"YOLO detection failed: {e}", status=500)
        for res in results:
            names = getattr(res, "names", {}) or {}
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                try:
                    xyxy = [float(v) for v in b.xyxy[0].tolist()]
                    conf = float(b.conf[0].item()) if b.conf is not None else 0.0
                    cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                    label = str(names.get(cls_id, cls_id))
                    out.append({
                        "bbox": xyxy, "conf": round(conf, 4),
                        "label": label, "type": _classify(label), "source": "detector",
                    })
                except Exception:
                    continue
        return out


# Default detector weight locations (baked in image, then persistent cache, then host).
# A bundled signature detector ships at the first path; stamps are best covered by the
# 'layout' mode or a custom multi-class model via ocr.detect_model_path.
_DEFAULT_DETECTOR_NAME = "signature-yolo.pt"


def _resolve_detector_path(cfg) -> str:
    import os
    if cfg.detect_model_path:
        return cfg.detect_model_path
    cache = os.environ.get("CODERAI_CACHE_DIR") or ("/cache" if os.path.isdir("/cache") else "")
    cands = [f"/opt/coderai/models/ocr/{_DEFAULT_DETECTOR_NAME}"]
    if cache:
        cands.append(os.path.join(cache, "ocr", _DEFAULT_DETECTOR_NAME))
    cands.append(os.path.expanduser(f"~/.coderai/models/ocr/{_DEFAULT_DETECTOR_NAME}"))
    for c in cands:
        if os.path.isfile(c):
            return c
    return ""   # none found → YoloDetector raises a clear 400


def _get_detector(cfg) -> YoloDetector:
    path = _resolve_detector_path(cfg)
    key = f"{path}|{cfg.detect_conf}"
    with _cache_lock:
        det = _detector_cache.get(key)
        if det is None:
            det = YoloDetector(path, float(cfg.detect_conf))
            _detector_cache[key] = det
        return det


# ---------------------------------------------------------------------------
# Layout-based flagging (no extra model)
# ---------------------------------------------------------------------------

def _detect_from_layout(page: OcrPage) -> List[dict]:
    out: List[dict] = []
    for reg in page.regions:
        lo = (reg.label or "").lower()
        hit = (any(m in lo for m in _LAYOUT_STAMP_LABELS)
               or any(m in lo for m in _STAMP_MARKERS)
               or any(m in lo for m in _SIGNATURE_MARKERS))
        if hit:
            out.append({
                "bbox": list(reg.bbox), "conf": round(float(reg.conf), 4),
                "label": reg.label, "type": _classify(reg.label), "source": "layout",
            })
    return out


# ---------------------------------------------------------------------------
# Entry point (runs in a worker thread; sync)
# ---------------------------------------------------------------------------

def detect_page(image, page: OcrPage, cfg, mode: str) -> Dict[str, List[dict]]:
    """Return {'stamps': [...], 'signatures': [...]} for one page under ``mode``."""
    dets: List[dict] = []
    if mode in ("layout", "both"):
        dets.extend(_detect_from_layout(page))
    if mode in ("detector", "both"):
        dets.extend(_get_detector(cfg).detect(image))

    stamps = [d for d in dets if d["type"] == "stamp"]
    signatures = [d for d in dets if d["type"] == "signature"]
    return {"stamps": stamps, "signatures": signatures}
