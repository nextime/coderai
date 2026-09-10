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

"""MiniMax-H3 rules shared by both ways of running it.

H3 can run either IN the engine process (when the engine's diffusers is >= 0.40)
or in an isolated venv worker (when it isn't). The arithmetic and the media
handling are identical either way, and they are checkpoint contracts rather than
preferences — so they live here, in a module with NO coderai imports and no
heavy top-level imports, which both sides can load:

  * tools/h3_service.py  imports it as a sibling module inside the isolated venv
  * codai/api/video.py   loads it by path via codai/api/h3_worker.rules()

Everything here is pure arithmetic or media plumbing; diffusers and torch are
imported inside the functions that need them.
"""

import base64
import os
import subprocess
import tempfile

H3_FPS = 24
H3_FRAMES_CHUNK = 17          # video VAE clip length
H3_FRAMES_REMAINDER = 5       # decodable counts are 17n + 5
H3_MIN_SECONDS = 5
H3_MAX_SECONDS = 15
H3_CANVAS_MULTIPLE = 32


def log(msg: str) -> None:
    print(f"[h3] {msg}", flush=True)


# ── geometry ──────────────────────────────────────────────────────────────────

def snap_frames(num_frames: int) -> int:
    """Snap up to the next 17n+5 the video VAE can decode, clamped to 5-15 s."""
    n = max(1, int(num_frames or 0))
    lo = H3_MIN_SECONDS * H3_FPS       # 120
    hi = H3_MAX_SECONDS * H3_FPS       # 360
    n = max(lo, min(hi, n))
    if n % H3_FRAMES_CHUNK != H3_FRAMES_REMAINDER:
        n += (H3_FRAMES_REMAINDER - n % H3_FRAMES_CHUNK) % H3_FRAMES_CHUNK
    while n > hi:                       # snapping may overshoot the 15 s ceiling
        n -= H3_FRAMES_CHUNK
    return n


def snap_axis(value: int, default: int) -> int:
    v = int(value or default)
    v = max(H3_CANVAS_MULTIPLE, v)
    return v - (v % H3_CANVAS_MULTIPLE)


def pick_workflow(body: dict) -> str:
    """Which transformer partition this request needs.

    Loading with no workflow pulls BOTH partitions (61.7 GB each), so a workflow
    is always chosen from the inputs.
    """
    if body.get("references"):
        return "ref2va"
    if body.get("image") or body.get("last_image"):
        return "fl2va"
    return "t2va"


# ── media helpers ─────────────────────────────────────────────────────────────

def _decode_to_file(data: str, suffix: str) -> str:
    """Write a base64 / data-URI payload to a temp file and return its path."""
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, raw)
    os.close(fd)
    return path


def _load_image(data: str):
    from PIL import Image
    path = _decode_to_file(data, ".png")
    try:
        return Image.open(path).convert("RGB")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def build_references(refs: list, temps: list) -> list:
    """Turn the request's reference list into H3 reference dataclasses.

    Order matters: it labels them in the prompt presentation and lays them out on
    the shared rotary clock, so the request order is preserved exactly.
    """
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference)
    kinds = {"image": (MiniMaxH3ImageReference, ".png"),
             "video": (MiniMaxH3VideoReference, ".mp4"),
             "audio": (MiniMaxH3AudioReference, ".wav")}
    out = []
    for ref in refs or []:
        kind = (ref.get("type") or "image").lower()
        if kind not in kinds:
            raise ValueError(f"unknown reference type '{kind}' (image|video|audio)")
        cls, suffix = kinds[kind]
        path = _decode_to_file(ref.get("data") or "", suffix)
        temps.append(path)
        out.append(cls.from_file(path))
    return out


def mux(frames, audio, sampling_rate) -> bytes:
    """Write frames at 24 fps and mux the jointly-generated soundtrack in."""
    from diffusers.utils import export_to_video
    tmpdir = tempfile.mkdtemp(prefix="h3_")
    video_path = os.path.join(tmpdir, "video.mp4")
    export_to_video(frames, video_path, fps=H3_FPS)
    out_path = video_path

    if audio is not None and sampling_rate:
        try:
            import numpy as np
            import soundfile as sf
            wav = audio
            if hasattr(wav, "detach"):
                wav = wav.detach().to("cpu").float().numpy()
            wav = np.asarray(wav)
            while wav.ndim > 2:            # (1, 2, samples) -> (2, samples)
                wav = wav[0]
            if wav.ndim == 2:              # channels-major -> soundfile's (n, ch)
                wav = wav.T
            audio_path = os.path.join(tmpdir, "audio.wav")
            sf.write(audio_path, wav, int(sampling_rate))
            muxed = os.path.join(tmpdir, "muxed.mp4")
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", muxed],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode == 0 and os.path.getsize(muxed):
                out_path = muxed
            else:
                log("audio mux failed; returning the silent video: "
                    + proc.stderr.decode("utf-8", "replace")[-300:])
        except Exception as exc:
            log(f"audio mux skipped ({exc}); returning the silent video")

    with open(out_path, "rb") as fh:
        data = fh.read()
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    return data
