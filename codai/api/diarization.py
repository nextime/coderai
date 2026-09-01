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

"""Speaker diarization via the isolated pyannote.audio worker.

Provides the plumbing shared by the standalone ``/v1/audio/diarization`` endpoint
and the ``diarize=true`` option on ``/v1/audio/transcriptions``:

* :func:`diarize_audio` — start/reuse the (VRAM-eviction-tracked) pyannote worker
  and return speaker turns.
* :func:`merge_speakers` — attach a ``speaker`` label to each ASR segment by
  maximal temporal overlap with the diarization turns.
"""

from typing import List, Optional

from codai.models.manager import multi_model_manager


class _DiarizerHandle:
    """Evictable handle for a running pyannote worker. Its ``cleanup`` stops the
    subprocess, so the manager's LRU eviction frees the worker's VRAM like any
    other loaded model."""

    def __init__(self, model: str, url: str):
        self.model = model
        self.url = url

    def cleanup(self):
        try:
            from codai.api import pyannote_worker
            pyannote_worker.stop_service(self.model)
        except Exception:
            pass


def _get_diarizer(model: Optional[str], config: Optional[dict]) -> _DiarizerHandle:
    """Ensure the pyannote worker is running and registered as an evictable model."""
    from codai.api import pyannote_worker
    resolved = model or pyannote_worker._default_model()
    key = f"diarization:{resolved}"
    cfg = config or {}

    # ensure_service is idempotent: it reuses a healthy worker and re-spawns a dead
    # one (possibly on a new port). Always call it so a worker that died since the
    # handle was cached self-heals — otherwise acquire_stt_backend would hand back a
    # stale handle pointing at the dead worker's port (connection refused).
    url = pyannote_worker.ensure_service(resolved, cfg)

    def _loader():
        return _DiarizerHandle(resolved, url)

    needed = float(cfg.get("used_vram_gb") or 2.0)
    # pyannote is small (~2GB) — default it into the co-resident set so it stays
    # warm alongside the STT models (still last-resort-evictable for a big LLM).
    keep_resident = bool(cfg.get("keep_resident", True))
    handle = multi_model_manager.acquire_stt_backend(
        key, needed, True, _loader, keep_resident=keep_resident)
    # Refresh the URL in case a cached handle points at a re-spawned worker.
    handle.url = url
    return handle


def diarize_audio(audio_path: str, num_speakers=None, min_speakers=None,
                  max_speakers=None, model: Optional[str] = None,
                  config: Optional[dict] = None, identify: bool = False,
                  identify_backend: str = "ecapa", identify_threshold: float = 0.25) -> dict:
    """Return ``{"segments":[{start,end,speaker}], "num_speakers":N}`` for the audio.

    When ``identify`` is set, the worker also embeds one slice per diarized speaker;
    each is matched against the enrolled speaker registry and the diarization
    labels (SPEAKER_00…) are replaced with the recognised names (falling back to
    the pyannote label when unknown). Also returns ``speakers`` = {label: name}."""
    import requests
    handle = _get_diarizer(model, config)
    params = {}
    if num_speakers:
        params["num_speakers"] = int(num_speakers)
    if min_speakers:
        params["min_speakers"] = int(min_speakers)
    if max_speakers:
        params["max_speakers"] = int(max_speakers)
    if identify:
        params["embed"] = identify_backend or "ecapa"
    with open(audio_path, "rb") as f:
        resp = requests.post(f"{handle.url}/diarize", params=params,
                             data=f.read(), timeout=1800,
                             headers={"Content-Type": "application/octet-stream"})
    if resp.status_code != 200:
        raise RuntimeError(f"pyannote worker error {resp.status_code}: {resp.text[:300]}")
    j = resp.json()
    if j.get("error"):
        raise RuntimeError(j["error"])
    segments = j.get("segments") or []
    out = {"segments": segments, "num_speakers": j.get("num_speakers")}
    if identify and j.get("speaker_embeddings"):
        from codai.api import speaker_registry
        label_to_name = {}
        for label, emb in j["speaker_embeddings"].items():
            res = speaker_registry.identify(emb, identify_backend or "ecapa",
                                            threshold=identify_threshold)
            label_to_name[label] = {"name": res["name"], "score": res.get("score")}
        # Relabel segments with recognised names (keep pyannote label when unknown).
        for s in segments:
            m = label_to_name.get(s.get("speaker"))
            if m and m["name"] != "unknown":
                s["speaker_label"] = s["speaker"]
                s["speaker"] = m["name"]
        out["speakers"] = label_to_name
    return out


def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge_speakers(asr_segments: List[dict], diar_segments: List[dict]) -> List[dict]:
    """Attach a ``speaker`` to each ASR segment: the diarization turn it overlaps
    most. ASR segments with no overlap are left unlabeled. Returns a new list."""
    out = []
    for seg in asr_segments or []:
        s0 = float(seg.get("start", 0.0) or 0.0)
        s1 = float(seg.get("end", 0.0) or 0.0)
        best_spk, best_ov = None, 0.0
        for d in diar_segments or []:
            ov = _overlap(s0, s1, float(d.get("start", 0.0)), float(d.get("end", 0.0)))
            if ov > best_ov:
                best_ov, best_spk = ov, d.get("speaker")
        merged = dict(seg)
        if best_spk is not None:
            merged["speaker"] = best_spk
        out.append(merged)
    return out
