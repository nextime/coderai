import base64
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from codai.api.audio_backends import detect_audio_backends

router = APIRouter()

global_args = None
global_file_path = None


def set_global_args(args):
    global global_args
    global_args = args


def set_global_file_path(path):
    global global_file_path
    global_file_path = path


def _decode_audio(data: str) -> bytes:
    if data.startswith("data:"):
        _, enc = data.split(",", 1)
        return base64.b64decode(enc)
    return base64.b64decode(data)


def _ffmpeg_binary() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(status_code=501, detail="ffmpeg is required for native stem separation")
    return ffmpeg


def _persist_file(path: str, suffix: str, http_request: Request) -> dict:
    from codai.api.urlutils import build_file_url
    data = Path(path).read_bytes()
    if global_file_path:
        os.makedirs(global_file_path, exist_ok=True)
        filename = f"{uuid.uuid4().hex}{suffix}"
        out_path = os.path.join(global_file_path, filename)
        with open(out_path, "wb") as handle:
            handle.write(data)
        return {"url": build_file_url(filename, http_request)}
    return {f"b64_{suffix.lstrip('.')}": base64.b64encode(data).decode("ascii")}


def _run_ffmpeg(command: List[str]):
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "ffmpeg command failed"
        raise HTTPException(status_code=500, detail=detail)


#: demucs stem order; "other" carries everything that is not drums/bass/vocals.
_DEMUCS_STEMS = ("drums", "bass", "other", "vocals")


def _demucs_separate(src: str, workdir: str) -> dict:
    """Run demucs on ``src`` and write one wav per stem into ``workdir``.

    Returns ``{stem_name: path}``. Prefers the in-process ``demucs.api`` (demucs 4.x)
    and falls back to the CLI for builds that don't ship it. Raises ImportError when
    demucs isn't installed at all, so the caller can report a clean 501.
    """
    try:
        from demucs.api import Separator, save_audio
    except ImportError:
        return _demucs_separate_cli(src, workdir)

    separator = Separator()          # default model (htdemucs)
    _origin, stems = separator.separate_audio_file(src)
    out = {}
    for name, tensor in stems.items():
        path = os.path.join(workdir, f"{name}.wav")
        save_audio(tensor, path, samplerate=separator.samplerate)
        out[name] = path
    return out


def _demucs_separate_cli(src: str, workdir: str) -> dict:
    """Fallback path: drive demucs through its CLI (`python -m demucs`)."""
    import sys

    outdir = os.path.join(workdir, "demucs")
    proc = subprocess.run(
        [sys.executable, "-m", "demucs", "-o", outdir, src],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if "No module named" in detail:
            raise ImportError(detail)
        raise HTTPException(status_code=500, detail=f"demucs failed: {detail[-500:]}")

    # demucs writes <outdir>/<model>/<track-name>/<stem>.wav
    out = {}
    for root, _dirs, files in os.walk(outdir):
        for fn in files:
            stem = os.path.splitext(fn)[0]
            if stem in _DEMUCS_STEMS:
                out[stem] = os.path.join(root, fn)
    if not out:
        raise HTTPException(status_code=500, detail="demucs produced no stems")
    return out


def _mix_down(paths: List[str], dest: str) -> str:
    """Sum several stems into one file (used to rebuild the instrumental)."""
    ffmpeg = _ffmpeg_binary()
    if len(paths) == 1:
        _run_ffmpeg([ffmpeg, "-y", "-i", paths[0], dest])
        return dest
    cmd = [ffmpeg, "-y"]
    for p in paths:
        cmd += ["-i", p]
    cmd += ["-filter_complex",
            f"amix=inputs={len(paths)}:duration=longest:normalize=0",
            dest]
    _run_ffmpeg(cmd)
    return dest


def separate_with_provider(audio_bytes: bytes, stem_mode: str, workdir: str) -> dict:
    """Real ML source separation via demucs.

    ``vocals-instrumental`` keeps demucs' vocals stem and sums drums+bass+other back
    into a single instrumental bed — which is what a music dub needs. ``4-stem``
    returns demucs' four stems as-is.
    """
    src = os.path.join(workdir, "input.wav")
    with open(src, "wb") as handle:
        handle.write(audio_bytes)

    try:
        stems = _demucs_separate(src, workdir)
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="ML stem separation needs demucs. Run: pip install demucs "
                   "(or call /v1/audio/stems with fallback_mode=true for the "
                   "best-effort ffmpeg split).")

    mode = stem_mode or "vocals-instrumental"
    if mode == "vocals-instrumental":
        vocals = stems.get("vocals")
        backing = [p for name, p in stems.items() if name != "vocals"]
        if not vocals or not backing:
            raise HTTPException(status_code=500,
                                detail="demucs did not return the expected stems")
        instrumental = _mix_down(sorted(backing),
                                 os.path.join(workdir, "instrumental.wav"))
        artifacts = [
            {"name": "vocals", "path": vocals, "role": "lead-vocal"},
            {"name": "instrumental", "path": instrumental, "role": "backing-track"},
        ]
    elif mode == "4-stem":
        artifacts = [{"name": n, "path": stems[n], "role": n}
                     for n in _DEMUCS_STEMS if n in stems]
        if not artifacts:
            raise HTTPException(status_code=500, detail="demucs returned no stems")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported stem_mode: {mode}")

    return {
        "stem_mode": mode,
        "artifacts": artifacts,
        "engine": "demucs",
        "model": "htdemucs",
        "limitations": [
            "separation quality depends on the source mix",
            "bleed between stems is normal on dense mixes",
        ],
    }


def _split_audio(audio_bytes: bytes, mode: str, workdir: str) -> dict:
    ffmpeg = _ffmpeg_binary()
    src = os.path.join(workdir, "input.wav")
    with open(src, "wb") as handle:
        handle.write(audio_bytes)

    if mode == "vocals-instrumental":
        vocal_path = os.path.join(workdir, "vocals.wav")
        instrumental_path = os.path.join(workdir, "instrumental.wav")
        _run_ffmpeg([
            ffmpeg,
            "-y",
            "-i",
            src,
            "-af",
            "pan=mono|c=0.5*FL+0.5*FR,highpass=f=120",
            vocal_path,
        ])
        _run_ffmpeg([
            ffmpeg,
            "-y",
            "-i",
            src,
            "-af",
            "pan=stereo|c0=FL-0.5*FC|c1=FR-0.5*FC,lowpass=f=14000",
            instrumental_path,
        ])
        return {
            "stem_mode": mode,
            "artifacts": [
                {"name": "vocals", "path": vocal_path, "role": "lead-vocal-estimate"},
                {"name": "instrumental", "path": instrumental_path, "role": "backing-mix-estimate"},
            ],
            "engine": "ffmpeg-mid-side-estimate",
            "limitations": [
                "best-effort heuristic only",
                "works best on center-panned vocals",
                "not equivalent to ML demixing",
            ],
        }

    if mode == "drums-bass-other":
        drums_path = os.path.join(workdir, "drums.wav")
        bass_path = os.path.join(workdir, "bass.wav")
        other_path = os.path.join(workdir, "other.wav")
        _run_ffmpeg([ffmpeg, "-y", "-i", src, "-af", "highpass=f=80,lowpass=f=220", bass_path])
        _run_ffmpeg([ffmpeg, "-y", "-i", src, "-af", "highpass=f=1800", drums_path])
        _run_ffmpeg([ffmpeg, "-y", "-i", src, "-af", "highpass=f=220,lowpass=f=1800", other_path])
        return {
            "stem_mode": mode,
            "artifacts": [
                {"name": "drums", "path": drums_path, "role": "high-frequency-transient-band"},
                {"name": "bass", "path": bass_path, "role": "low-frequency-band"},
                {"name": "other", "path": other_path, "role": "mid-band-residual"},
            ],
            "engine": "ffmpeg-band-split",
            "limitations": [
                "frequency-band approximation only",
                "not isolated stems",
                "bleed between sources is expected",
            ],
        }

    if mode == "4-stem":
        drums_path = os.path.join(workdir, "drums.wav")
        bass_path = os.path.join(workdir, "bass.wav")
        vocals_path = os.path.join(workdir, "vocals.wav")
        other_path = os.path.join(workdir, "other.wav")
        _run_ffmpeg([ffmpeg, "-y", "-i", src, "-af", "highpass=f=80,lowpass=f=220", bass_path])
        _run_ffmpeg([ffmpeg, "-y", "-i", src, "-af", "highpass=f=1800", drums_path])
        _run_ffmpeg([ffmpeg, "-y", "-i", src, "-af", "pan=mono|c=0.5*FL+0.5*FR,highpass=f=120", vocals_path])
        _run_ffmpeg([ffmpeg, "-y", "-i", src, "-af", "highpass=f=220,lowpass=f=1800", other_path])
        return {
            "stem_mode": mode,
            "artifacts": [
                {"name": "vocals", "path": vocals_path, "role": "lead-vocal-estimate"},
                {"name": "drums", "path": drums_path, "role": "high-frequency-transient-band"},
                {"name": "bass", "path": bass_path, "role": "low-frequency-band"},
                {"name": "other", "path": other_path, "role": "mid-band-residual"},
            ],
            "engine": "ffmpeg-hybrid-estimate",
            "limitations": [
                "hybrid heuristic split only",
                "not phase-accurate demixing",
                "use dedicated ML separators for production quality",
            ],
        }

    raise HTTPException(status_code=400, detail=f"Unsupported stem_mode: {mode}")


class AudioStemRequest(BaseModel):
    audio: str
    stem_mode: Optional[str] = "vocals-instrumental"
    response_format: Optional[str] = "url"
    fallback_mode: Optional[bool] = False
    model_config = ConfigDict(extra="allow")


@router.post("/v1/audio/stems", summary="Separate audio into stems")
async def separate_stems(request: AudioStemRequest, http_request: Request = None):
    """Split a track into its component stems (source separation).

    Separates an input clip according to `stem_mode` (e.g. vocals/instrumental, or a
    full 4-stem split). Uses an ML separation provider when available, falling back to
    an ffmpeg-based best-effort split when `fallback_mode` is set. Returns one audio
    output per stem along with the backend and quality tier used.
    """
    try:
        audio_bytes = _decode_audio(request.audio)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio payload: {exc}")

    with tempfile.TemporaryDirectory(prefix="codai-stems-") as workdir:
        backend_info = detect_audio_backends()["separation"]
        if request.fallback_mode:
            result = _split_audio(audio_bytes, request.stem_mode or "vocals-instrumental", workdir)
            quality = "best-effort"
            dependency = "ffmpeg"
            model_name = None
        else:
            result = separate_with_provider(audio_bytes, request.stem_mode or "vocals-instrumental", workdir)
            quality = "ml"
            dependency = "python"
            model_name = result.get("model")
        data = []
        for artifact in result["artifacts"]:
            payload = _persist_file(artifact["path"], ".wav", http_request)
            payload.update({"name": artifact["name"], "role": artifact["role"]})
            data.append(payload)
        return {
            "created": int(time.time()),
            "stem_mode": result["stem_mode"],
            "backend": {
                "engine": result["engine"],
                "model": model_name,
                "quality": quality,
                "dependency": dependency,
                "ml_backend_available": backend_info["available"],
                "preferred_engine": backend_info["engine"],
            },
            "limitations": result["limitations"],
            "data": data,
        }
