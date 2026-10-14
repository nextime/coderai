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

import json
import os
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


class _DetectWorker:
    """The YOLO detector, driven in whichever venv actually has ultralytics.

    One process per install, shared by every weights path (the worker caches
    models by path), because loading YOLO twice costs memory for nothing.

    Deliberately NOT a hard dependency: `get()` returns None when no such venv
    exists, and the caller then raises the ordinary "ultralytics is not
    installed" error. A detector that cannot be reached must look like a missing
    detector, not like a broken install.
    """

    _inst = None
    _lock = threading.Lock()

    #: Where a baked venv carrying ultralytics may be. `surya` first: that is
    #: where the pod image puts it, sharing the CPU torch already there.
    _VENVS = ("/opt/coderai/venvs/surya", "/opt/coderai/venvs/detect",
              "/opt/coderai/venvs/ocr-paddle")

    def __init__(self, python: str):
        import subprocess
        worker = os.path.join(os.path.dirname(__file__), "workers",
                              "detect_worker.py")
        self._proc = subprocess.Popen([python, "-I", worker],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=None,
                                      bufsize=0)
        hello = self._reply()
        if not hello.get("ready"):
            self.kill()
            raise OcrError(hello.get("error") or "detector worker not ready",
                           status=503)
        self._io = threading.Lock()

    @classmethod
    def get(cls):
        """The shared worker, or None when no venv carrying ultralytics exists."""
        with cls._lock:
            if cls._inst is not None:
                return cls._inst
            for vdir in cls._VENVS:
                py = os.path.join(vdir, "bin", "python")
                if not os.path.isfile(py):
                    continue
                try:
                    cls._inst = cls(py)
                except Exception as exc:
                    print(f"[ocr] detector venv {vdir} would not start ({exc})",
                          flush=True)
                    continue
                print(f"[ocr] stamp/signature detector runs in {vdir}",
                      flush=True)
                return cls._inst
            return None

    def _reply(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            raise OcrError("detector worker exited", status=500)
        return json.loads(line.decode("utf-8"))

    def detect(self, image, model_path: str, conf: float) -> List[dict]:
        import numpy as np
        rgb = image.convert("RGB")
        raw = np.array(rgb).tobytes()
        hdr = {"cmd": "detect", "model": model_path, "conf": float(conf),
               "w": rgb.width, "h": rgb.height, "nbytes": len(raw)}
        # One page at a time down one pipe: the frame is header-then-bytes, and
        # two callers interleaving would desynchronise it.
        with self._io:
            self._proc.stdin.write(json.dumps(hdr).encode("utf-8") + b"\n")
            self._proc.stdin.write(raw)
            self._proc.stdin.flush()
            r = self._reply()
        if not r.get("ok"):
            raise OcrError(r.get("error") or "YOLO detection failed",
                           status=int(r.get("status", 500)))
        return list(r.get("boxes") or [])

    def kill(self):
        try:
            self._proc.kill()
        except Exception:
            pass


class YoloDetector:
    def __init__(self, model_path: str, conf: float):
        self.model_path = model_path
        self.conf = conf
        self._model = None
        #: Set instead of `_model` when ultralytics is not importable in this
        #: process but a venv carrying it is available (see _DetectWorker).
        self._remote = None
        self._load_lock = threading.Lock()

    def ensure_loaded(self):
        if self._model is not None or self._remote is not None:
            return
        with self._load_lock:
            if self._model is not None or self._remote is not None:
                return
            import os
            if not self.model_path or not os.path.isfile(self.model_path):
                raise OcrError(
                    "stamp/signature detector weights not found. Set ocr.detect_model_path "
                    "to a YOLO .pt file trained on signatures/stamps.",
                    status=400,
                )
            try:
                from ultralytics import YOLO
            except Exception as in_process_error:
                # Not importable HERE is not the same as not installed. On a
                # light-core pod image ultralytics lives beside surya, sharing
                # the CPU torch already in that venv -- which is what makes the
                # detector cost ~30 MB instead of a third torch. Reach it.
                self._remote = _DetectWorker.get()
                if self._remote is None:
                    raise OcrError(
                        "ultralytics is not installed (needed for detector mode) and no "
                        "venv carrying it was found. Install: `pip install ultralytics`, "
                        f"or bake it beside surya. Import error: {in_process_error}",
                        status=503,
                    )
                return
            try:
                self._model = YOLO(self.model_path)
            except Exception as e:
                raise OcrError(f"failed to load YOLO detector: {e}", status=500)

    def detect(self, image) -> List[dict]:
        self.ensure_loaded()
        if self._remote is not None:
            return self._remote.detect(image, self.model_path, self.conf)
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
