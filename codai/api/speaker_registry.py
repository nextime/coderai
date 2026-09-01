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

"""Speaker identification registry — enrolled voiceprints + cosine matching.

A named speaker is one or more embeddings (averaged) produced by a specific
embedding backend (ecapa / wespeaker / …). Identification/verification compares a
query embedding against enrolled speakers of the SAME backend by cosine
similarity — embeddings from different backends are NOT comparable.

Persisted as JSON in the coderai config dir so enrollments survive restarts.
"""

import json
import os
import threading
import time
from typing import Optional

_lock = threading.RLock()


def _store_path() -> str:
    p = os.environ.get("CODERAI_SPEAKER_STORE")
    if p:
        return p
    for d in ("/config/coderai", os.path.expanduser("~/.coderai")):
        if os.path.isdir(d):
            return os.path.join(d, "speakers.json")
    return os.path.join(os.path.expanduser("~/.coderai"), "speakers.json")


def _load() -> dict:
    try:
        with open(_store_path()) as f:
            return json.load(f)
    except Exception:
        return {"speakers": {}}


def _save(data: dict) -> None:
    path = _store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _norm(v):
    import numpy as np
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def _cosine(a, b) -> float:
    import numpy as np
    return float(np.dot(_norm(a), _norm(b)))


def enroll(name: str, embedding: list, backend: str, model: str = None) -> dict:
    """Add a sample for ``name``. Re-enrolling averages with existing samples
    (running mean), improving the voiceprint. One voiceprint per (name, backend)."""
    import numpy as np
    name = (name or "").strip()
    if not name:
        raise ValueError("speaker name is required")
    with _lock:
        data = _load()
        spk = data.setdefault("speakers", {})
        key = f"{name}::{backend}"
        rec = spk.get(key)
        if rec is None:
            rec = {"name": name, "backend": backend, "model": model,
                   "dim": len(embedding), "samples": 0,
                   "embedding": [0.0] * len(embedding)}
        n = rec["samples"]
        # running mean of the raw embeddings
        prev = np.asarray(rec["embedding"], dtype=np.float32)
        cur = np.asarray(embedding, dtype=np.float32)
        if prev.shape != cur.shape:
            prev = np.zeros_like(cur); n = 0
        newmean = (prev * n + cur) / (n + 1)
        rec["embedding"] = newmean.astype(float).tolist()
        rec["samples"] = n + 1
        rec["dim"] = len(rec["embedding"])
        rec["model"] = model or rec.get("model")
        rec["updated"] = None  # timestamp stamped by caller if wanted
        spk[key] = rec
        _save(data)
        return {"name": name, "backend": backend, "samples": rec["samples"],
                "dim": rec["dim"]}


def list_speakers() -> list:
    with _lock:
        data = _load()
    return [{"name": r["name"], "backend": r["backend"], "model": r.get("model"),
             "samples": r.get("samples", 1), "dim": r.get("dim")}
            for r in data.get("speakers", {}).values()]


def delete(name: str, backend: str = None) -> int:
    name = (name or "").strip()
    removed = 0
    with _lock:
        data = _load()
        spk = data.get("speakers", {})
        for key in [k for k, r in spk.items()
                    if r.get("name") == name and (backend is None or r.get("backend") == backend)]:
            spk.pop(key, None)
            removed += 1
        if removed:
            _save(data)
    return removed


def get(name: str, backend: str) -> Optional[dict]:
    with _lock:
        for r in _load().get("speakers", {}).values():
            if r.get("name") == name and r.get("backend") == backend:
                return r
    return None


def _enrolled_for_backend(backend: str) -> list:
    data = _load()
    return [r for r in data.get("speakers", {}).values() if r.get("backend") == backend]


def identify(embedding: list, backend: str, threshold: float = 0.25,
             top_k: int = 3) -> dict:
    """Return the best-matching enrolled speaker for ``embedding`` (same backend).

    ``score`` is cosine similarity in [-1,1]; below ``threshold`` → name "unknown".
    Also returns the top_k ranked candidates."""
    with _lock:
        enrolled = _enrolled_for_backend(backend)
    ranked = sorted(
        ({"name": r["name"], "score": _cosine(embedding, r["embedding"])}
         for r in enrolled),
        key=lambda x: x["score"], reverse=True)
    best = ranked[0] if ranked else None
    name = best["name"] if (best and best["score"] >= threshold) else "unknown"
    return {"name": name, "score": (best["score"] if best else None),
            "threshold": threshold, "candidates": ranked[:top_k],
            "enrolled": len(enrolled), "backend": backend}


def verify(embedding_a: list, embedding_b: list, threshold: float = 0.25) -> dict:
    score = _cosine(embedding_a, embedding_b)
    return {"same_speaker": score >= threshold, "score": score, "threshold": threshold}
