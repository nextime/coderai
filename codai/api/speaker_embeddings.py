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

"""Speaker-embedding extraction (voiceprints) via the isolated pyannote venv.

Backends (served by tools/pyannote_service.py's /embed):
* ``ecapa``    — speechbrain ECAPA-TDNN (ungated, 192-d). Default.
* ``pyannote`` — pyannote.audio embedding model (may be gated, uses HF_TOKEN).
* ``wespeaker``— pyannote wespeaker resnet embedding.

Reuses the same worker (and VRAM-eviction registration) as diarization.
"""

from typing import Optional


def get_speaker_embedding(audio_path: str, backend: str = "ecapa",
                          model: Optional[str] = None,
                          config: Optional[dict] = None) -> dict:
    """Return ``{"embedding":[...], "dim":N, "backend":..., "model":...}``."""
    import requests
    from codai.api.diarization import _get_diarizer
    # _get_diarizer ensures the (VRAM-tracked, self-healing) pyannote worker is up
    # and returns a handle with its URL; the worker serves /embed as well.
    handle = _get_diarizer(None, config or {})
    params = {"backend": backend or "ecapa"}
    if model:
        params["model"] = model
    with open(audio_path, "rb") as f:
        resp = requests.post(f"{handle.url}/embed", params=params, data=f.read(),
                             timeout=600, headers={"Content-Type": "application/octet-stream"})
    if resp.status_code != 200:
        raise RuntimeError(f"pyannote worker error {resp.status_code}: {resp.text[:300]}")
    j = resp.json()
    if j.get("error"):
        raise RuntimeError(j["error"])
    return j
