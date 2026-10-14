#!/usr/bin/env python3
"""Run the stamp/signature YOLO detector in a venv that has a torch.

``codai/ocr/detect.py`` imports ultralytics IN-PROCESS, which works on any
install whose main venv carries torch. Two places it does not:

* the ``vllm-ocr`` pod image, whose core is LIGHT on purpose —
  ``Dockerfile.capability`` fails the build if torch reaches the main venv, and
  a third torch there would cost ~2.5 GB compressed for a workload that does
  not even want the card (the GPU is serving the model);
* any install that would rather not have torch in its main venv at all.

So ultralytics is installed beside surya, sharing the CPU torch already in that
venv (``profiles/vllm-ocr.venv-surya.txt``), and reached from here. That sharing
is the whole reason the detector costs ~30 MB rather than another torch.

Run as a SCRIPT with the OTHER venv's python, never imported: importing
``codai.ocr`` costs 377 MB and 1.5 s via litellm, and the point is to stay small.

Protocol, newline-delimited JSON on stdin/stdout:

    -> {"cmd": "detect", "model": "/path/to.pt", "conf": 0.35,
        "w": 850, "h": 1100, "nbytes": 2805000}   + <raw RGB bytes>
    <- {"ok": true, "boxes": [{"label": "stamp", "conf": 0.91,
                               "bbox": [x0, y0, x1, y1]}, ...]}

    -> {"cmd": "shutdown"}
    <- {"ok": true}

Raw RGB rather than PNG: the parent already holds the decoded page, and a page
on its way to a detector must not pay a compress/decompress round trip.

A failure answers ``{"ok": false, "error": ..., "status": ...}`` and leaves the
process usable: one bad page must not cost the detector its loaded weights.
"""

import json
import sys


def _reply(obj):
    sys.stdout.buffer.write(json.dumps(obj).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


def _read_exactly(n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            raise EOFError(f"stdin closed after {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def main() -> int:
    try:
        from ultralytics import YOLO
        import numpy as np
    except Exception as e:
        _reply({"ready": False, "error": f"detector deps missing: {e}"})
        return 1
    _reply({"ready": True})

    models = {}          # weights path -> loaded YOLO

    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            break
        try:
            msg = json.loads(line.decode("utf-8"))
        except Exception as e:
            _reply({"ok": False, "error": f"bad request frame: {e}",
                    "status": 500})
            continue
        cmd = msg.get("cmd")

        if cmd == "shutdown":
            _reply({"ok": True})
            return 0

        if cmd != "detect":
            _reply({"ok": False, "error": f"unknown cmd {cmd!r}", "status": 500})
            continue

        try:
            raw = _read_exactly(int(msg.get("nbytes") or 0))
        except Exception as e:
            # The frame is lost; the parent must replace us rather than guess
            # where the next header starts.
            _reply({"ok": False, "error": str(e), "status": 500})
            return 1

        path = str(msg.get("model") or "")
        try:
            model = models.get(path)
            if model is None:
                model = models[path] = YOLO(path)
        except Exception as e:
            _reply({"ok": False, "error": f"failed to load YOLO detector: {e}",
                    "status": 500})
            continue

        try:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                int(msg["h"]), int(msg["w"]), 3)
            results = model.predict(arr, conf=float(msg.get("conf") or 0.35),
                                    verbose=False)
        except Exception as e:
            _reply({"ok": False, "error": f"YOLO detection failed: {e}",
                    "status": 500})
            continue

        boxes = []
        for res in results:
            names = getattr(res, "names", {}) or {}
            bs = getattr(res, "boxes", None)
            if bs is None:
                continue
            for b in bs:
                try:
                    cls = int(b.cls[0])
                    xy = [float(v) for v in b.xyxy[0]]
                    boxes.append({"label": str(names.get(cls, cls)),
                                  "conf": float(b.conf[0]), "bbox": xy})
                except Exception:
                    continue
        _reply({"ok": True, "boxes": boxes})

    return 0


if __name__ == "__main__":
    sys.exit(main())
