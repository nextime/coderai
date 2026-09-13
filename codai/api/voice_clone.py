"""
Voice cloning endpoints.

POST   /v1/audio/clone              — synthesize speech in a cloned voice
GET    /v1/audio/voices             — list saved voice profiles
POST   /v1/audio/voices             — save a named voice profile (ref audio + transcript)
POST   /v1/audio/voices/extract     — extract voice profile from audio or video file
PATCH  /v1/audio/voices/{name}      — update description / replace reference audio
DELETE /v1/audio/voices/{name}      — delete a voice profile
"""

import asyncio
import base64
import io
import json
import os
import subprocess
import threading
import tempfile
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel, ConfigDict

from codai.platform_paths import default_voices_dir

router = APIRouter()

global_args = None
global_file_path = None

# Directory where voice profiles are stored
_VOICES_DIR: Optional[str] = None


def set_global_args(args):
    global global_args, _VOICES_DIR
    global_args = args
    # Store voice profiles alongside output files, or in a default location
    base = getattr(args, 'file_path', None) or str(default_voices_dir())
    _VOICES_DIR = os.path.join(base if os.path.isdir(base) else os.path.dirname(base) if base else str(default_voices_dir().parent), 'voices')
    os.makedirs(_VOICES_DIR, exist_ok=True)


def set_global_file_path(path):
    global global_file_path
    global_file_path = path


def _voices_dir() -> str:
    if _VOICES_DIR:
        return _VOICES_DIR
    d = str(default_voices_dir())
    os.makedirs(d, exist_ok=True)
    return d


def _voice_path(name: str) -> str:
    return os.path.join(_voices_dir(), name)


def _list_voices() -> list:
    d = _voices_dir()
    voices = []
    for entry in os.scandir(d):
        if entry.is_dir():
            meta_path = os.path.join(entry.path, 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                voices.append(meta)
    return sorted(voices, key=lambda v: v.get('created_at', 0))


def _clip_score(path: str) -> float:
    """Rough usability score for a reference clip. 0.0 if unreadable.

    Judged on three things a clone actually suffers from: clipping (a hard
    distortion no engine can undo), level (too quiet means noise dominates), and
    crest factor (speech sits around 4-8; flatter means compressed or clipped).
    Deliberately NOT "loudest wins" — a clipped take has the highest RMS of all
    and is the worst possible prompt."""
    try:
        import numpy as np
        import soundfile as sf
        data, _sr = sf.read(path, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        if not len(data):
            return 0.0
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data ** 2)))
        if rms <= 1e-6 or peak <= 1e-6:
            return 0.0

        # 1. clipping: fraction of samples pinned at full scale. 0.1% is already
        #    audible, 1% is ruinous.
        clipped = float(np.mean(np.abs(data) >= 0.995))
        clip_term = max(0.0, 1.0 - clipped * 300.0)

        # 2. level: best around -20 dBFS RMS (~0.1), falling off either side.
        level_term = float(np.exp(-((np.log10(rms) - np.log10(0.1)) ** 2) / 0.5))

        # 3. crest factor: speech ~4-8. Flat (clipped/compressed) or spiky
        #    (silence with a bang) both score worse.
        crest = peak / rms
        crest_term = float(np.exp(-((crest - 6.0) ** 2) / 32.0))

        return clip_term * level_term * crest_term
    except Exception:
        return 0.0


def _profile_clips(meta: dict) -> list:
    """Every reference clip on a profile, newest first, as [{path, transcript}].

    Profiles used to hold exactly one clip (`audio_file` + `transcript`); that shape
    is still honoured so existing profiles keep working."""
    clips = []
    for c in (meta.get('clips') or []):
        if isinstance(c, dict) and c.get('path') and os.path.isfile(c['path']):
            clips.append(c)
    if not clips and meta.get('audio_file') and os.path.isfile(meta['audio_file']):
        clips.append({'path': meta['audio_file'], 'transcript': meta.get('transcript', '')})
    return clips


def _best_clip(meta: dict) -> Optional[dict]:
    """The clip a clone should prompt on.

    More references only help engines that can average them (see `--engine chain`);
    F5-TTS prompts on ONE, so picking the cleanest take beats picking the first."""
    clips = _profile_clips(meta)
    if not clips:
        return None
    if len(clips) == 1:
        return clips[0]
    scored = sorted(clips, key=lambda c: _clip_score(c['path']), reverse=True)
    best = scored[0]
    print(f"  [voice-clone] {len(clips)} reference clips — using "
          f"{os.path.basename(best['path'])} (cleanest of the set)", flush=True)
    return best


def _save_voice(name: str, audio_bytes: bytes, audio_ext: str, transcript: str, description: str = '') -> dict:
    vdir = _voice_path(name)
    os.makedirs(vdir, exist_ok=True)
    audio_file = os.path.join(vdir, f'ref{audio_ext}')
    with open(audio_file, 'wb') as f:
        f.write(audio_bytes)
    meta = {
        'name': name,
        'description': description,
        'transcript': transcript,
        'audio_file': audio_file,
        'audio_ext': audio_ext,
        'clips': [{'path': audio_file, 'transcript': transcript}],
        'created_at': int(time.time()),
    }
    with open(os.path.join(vdir, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    return meta


def _load_voice(name: str) -> Optional[dict]:
    meta_path = os.path.join(_voice_path(name), 'meta.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def _decode_audio(data: str) -> tuple[bytes, str]:
    """Decode base64 audio data, return (bytes, ext)."""
    if data.startswith('data:'):
        mime, b64 = data.split(',', 1)
        ext = '.' + mime.split('/')[1].split(';')[0]
        return base64.b64decode(b64), ext
    return base64.b64decode(data), '.wav'


def _decode_b64_or_url(data: str) -> bytes:
    if data.startswith('data:'):
        _, b64 = data.split(',', 1)
        return base64.b64decode(b64)
    if data.startswith('http://') or data.startswith('https://'):
        import urllib.request
        with urllib.request.urlopen(data, timeout=60) as r:
            return r.read()
    return base64.b64decode(data)


#: F5-TTS prompts on the reference clip, and longer is NOT better: past roughly
#: 15 s the extra context stops helping and starts costing quality and time (the
#: upstream tooling clips to the same ballpark). Profiles here can hold minutes of
#: audio — the character studio uploads a 90 s clip — so trim at use time.
MAX_REF_SECONDS = float(os.environ.get("CODERAI_F5_MAX_REF_SECONDS", "15"))

_f5_engine = None          # cached F5TTS instance (device -> engine)
_f5_engine_device = None
_f5_lock = threading.Lock()


def _f5_device() -> Optional[str]:
    if global_args:
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
        except Exception:
            pass
    return None


def _get_f5_engine():
    """Return a cached F5TTS engine.

    It used to be constructed per request, which reloaded the whole model (and its
    vocoder) on every single line of dialogue — seconds of latency and a fresh VRAM
    allocation each time. seed-vc next door already keeps a singleton; this brings
    F5 in line. Rebuilt only if the device changes under us."""
    global _f5_engine, _f5_engine_device
    device = _f5_device()
    with _f5_lock:
        if _f5_engine is not None and _f5_engine_device == device:
            return _f5_engine
        from codai.api.hub_compat import install_hub_legacy_kwargs_shim
        install_hub_legacy_kwargs_shim()
        from f5_tts.api import F5TTS
        print(f"  [voice-clone] loading F5-TTS engine (device={device or 'cpu'})", flush=True)
        _f5_engine = F5TTS(device=device)
        _f5_engine_device = device
        return _f5_engine


def release_f5_engine() -> None:
    """Drop the cached engine (VRAM eviction hook)."""
    global _f5_engine, _f5_engine_device
    with _f5_lock:
        if _f5_engine is None:
            return
        _f5_engine = None
        _f5_engine_device = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("  [voice-clone] F5-TTS engine released", flush=True)


# Caching the engine means it holds VRAM between requests, so let the model
# manager reclaim it when a generation needs the card — same contract the LoRA
# base cache and the OCR engines use.
try:
    from codai.models.manager import multi_model_manager as _mmm
    _mmm.register_external_vram_releaser(release_f5_engine)
except Exception:
    pass


def _trim_reference(path: str, temps: list, max_seconds: float = None) -> str:
    """Clip an over-long reference to MAX_REF_SECONDS, returning a path to use.

    Returns the original path when it is already short enough or can't be read."""
    limit = MAX_REF_SECONDS if max_seconds is None else max_seconds
    if limit <= 0:
        return path
    try:
        import soundfile as sf
        info = sf.info(path)
        duration = info.frames / float(info.samplerate or 1)
        if duration <= limit:
            return path
        data, sr = sf.read(path, frames=int(limit * info.samplerate))
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out.close()
        sf.write(out.name, data, sr)
        temps.append(out.name)
        print(f"  [voice-clone] reference trimmed {duration:.0f}s -> {limit:.0f}s "
              f"(F5-TTS gains nothing past that)", flush=True)
        return out.name
    except Exception as exc:
        print(f"  [voice-clone] reference trim skipped: {exc}", flush=True)
        return path


def _f5tts_clone(ref_audio_path: str, ref_text: str, gen_text: str,
                  speed: float = 1.0, seed: Optional[int] = None) -> bytes:
    """Run F5-TTS voice cloning, return WAV bytes."""
    import soundfile as sf

    tts = _get_f5_engine()
    wav, sr, _ = tts.infer(
        ref_file=ref_audio_path,
        ref_text=ref_text,
        gen_text=gen_text,
        speed=speed,
        seed=seed,
        show_info=lambda x: None,
        progress=lambda x, **kw: x,
    )

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format='WAV')
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Watermarking
#
# Cloned speech of real people leaves this server at scale — dubbed videos,
# character dialogue, whole generated scenes. OpenVoice marks its output;
# nothing here did. AudioSeal (MIT, Meta) is the stronger option: an inaudible,
# localised mark that survives the usual re-encoding, plus a detector.
#
# On by default, because a watermark nobody enables protects nobody. Disable per
# request with watermark=false, or globally with CODERAI_AUDIO_WATERMARK=0.
# Degrades gracefully: if audioseal isn't installed the audio is returned
# unchanged with a one-time warning, never an error.
# ---------------------------------------------------------------------------

WATERMARK_DEFAULT = os.environ.get("CODERAI_AUDIO_WATERMARK", "1") not in ("0", "false", "no")
DEFAULT_CHAIN_BASE = os.environ.get("CODERAI_CLONE_BASE_TTS", "hexgrad/Kokoro-82M")

_watermarker = None
_watermark_warned = False
_watermark_lock = threading.Lock()


def _get_watermarker():
    """Cached AudioSeal generator, or None when the package isn't available."""
    global _watermarker, _watermark_warned
    with _watermark_lock:
        if _watermarker is not None:
            return _watermarker
        try:
            from audioseal import AudioSeal
            _watermarker = AudioSeal.load_generator("audioseal_wm_16bits")
            print("  [voice-clone] AudioSeal watermarking active", flush=True)
            return _watermarker
        except Exception as exc:
            if not _watermark_warned:
                _watermark_warned = True
                print(f"  [voice-clone] watermarking unavailable ({exc}) — synthesised "
                      f"audio will NOT be marked. pip install audioseal", flush=True)
            return None


def apply_watermark(wav_bytes: bytes, enabled: Optional[bool] = None) -> bytes:
    """Return the audio with an inaudible watermark, or unchanged if unavailable."""
    if enabled is False or (enabled is None and not WATERMARK_DEFAULT):
        return wav_bytes
    model = _get_watermarker()
    if model is None:
        return wav_bytes
    try:
        import io as _io
        import numpy as np
        import soundfile as sf
        import torch
        data, sr = sf.read(_io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        tensor = torch.from_numpy(np.ascontiguousarray(data))[None, None, :]
        with torch.no_grad():
            marked = model(tensor, sample_rate=sr, alpha=1.0)
        out = (tensor + marked).squeeze().cpu().numpy()
        buf = _io.BytesIO()
        sf.write(buf, out, sr, format="WAV")
        return buf.getvalue()
    except Exception as exc:
        print(f"  [voice-clone] watermarking failed ({exc}) — returning unmarked audio",
              flush=True)
        return wav_bytes


def detect_watermark(wav_bytes: bytes) -> dict:
    """Was this audio produced here? Returns {available, watermarked, confidence}."""
    try:
        import io as _io
        import numpy as np
        import soundfile as sf
        import torch
        from audioseal import AudioSeal
        detector = AudioSeal.load_detector("audioseal_detector_16bits")
        data, sr = sf.read(_io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        tensor = torch.from_numpy(np.ascontiguousarray(data))[None, None, :]
        with torch.no_grad():
            result, _msg = detector.detect_watermark(tensor, sample_rate=sr), None
        score = float(result if isinstance(result, (int, float)) else result[0])
        return {"available": True, "watermarked": score > 0.5, "confidence": round(score, 4)}
    except ImportError:
        return {"available": False, "watermarked": None,
                "detail": "audioseal not installed"}
    except Exception as exc:
        return {"available": True, "watermarked": None, "detail": str(exc)}


# ---------------------------------------------------------------------------
# Cloning engines
#
# F5-TTS clones by CONTINUING a prompt: reference audio + its transcript, one
# model, so prosody and accent come along with the timbre. That is the best
# choice when the reference speaks the target language — and useless without a
# transcript.
#
# The alternatives split the job the way OpenVoice does — synthesise first, then
# move the timbre — which costs some prosody fidelity but needs no transcript and
# lets the language be chosen independently of the reference:
#
#   xtts    XTTS-v2 zero-shot from the clip (17 languages, no transcript)
#   chain   any base TTS -> seed-vc timbre transfer (OpenVoice's architecture,
#           with seed-vc in place of its tone-colour converter)
#   melotts MeloTTS as the base voice of that chain
# ---------------------------------------------------------------------------

CLONE_ENGINES = ("auto", "f5", "xtts", "chain", "melotts")


def _pick_engine(requested: str, ref_text: str, language: str, ref_language: str) -> str:
    """Resolve engine='auto' to a concrete engine, and say why."""
    engine = (requested or "auto").strip().lower()
    if engine not in CLONE_ENGINES:
        raise HTTPException(status_code=400, detail=(
            f"Unknown clone engine '{engine}'. Use one of: {', '.join(CLONE_ENGINES)}"))
    if engine != "auto":
        return engine
    lang = (language or "").strip().lower()[:2]
    ref_lang = (ref_language or "").strip().lower()[:2]
    if not ref_text:
        print("  [voice-clone] auto -> xtts (no transcript for the reference)", flush=True)
        return "xtts"
    if lang and ref_lang and lang != ref_lang:
        print(f"  [voice-clone] auto -> xtts (target '{lang}' differs from the "
              f"reference's '{ref_lang}'; F5 would carry the accent over)", flush=True)
        return "xtts"
    return "f5"


_tts_backends: dict = {}
_tts_backend_lock = threading.Lock()


def _get_tts_backend(model_name: str):
    """Cached TTS backend, for the same reason F5 is cached: these are loaded per
    call otherwise, and a dialog is many calls."""
    with _tts_backend_lock:
        be = _tts_backends.get(model_name)
        if be is None:
            from codai.api.tts_backends import load_backend
            print(f"  [voice-clone] loading TTS backend '{model_name}'", flush=True)
            be = load_backend(model_name, None, {})
            _tts_backends[model_name] = be
        return be


def release_tts_backends() -> None:
    """Drop cached TTS backends (VRAM eviction hook)."""
    with _tts_backend_lock:
        if not _tts_backends:
            return
        _tts_backends.clear()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("  [voice-clone] TTS backends released", flush=True)


def _seedvc_convert(wrapper, source_path: str, target_path: str,
                    diffusion_steps: int = 10):
    """Timbre transfer with seed-vc — the role OpenVoice gives its tone-colour
    converter. Returns (audio, sample_rate)."""
    out = wrapper.convert_voice(
        source=source_path,
        target=target_path,
        diffusion_steps=diffusion_steps,
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        f0_condition=False,
        pitch_shift=0,
        stream_output=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    import numpy as np
    return np.asarray(out).flatten(), 22050      # f0_condition=False -> 22.05 kHz


def _xtts_clone(ref_audio_path: str, gen_text: str, speed: float = 1.0,
                language: str = "en") -> bytes:
    """Zero-shot clone with XTTS-v2 — no transcript needed, language selectable."""
    import io as _io
    import numpy as np
    import soundfile as sf
    from codai.api.tts_backends import load_backend
    backend = _get_tts_backend("coqui/XTTS-v2")
    # XTTS treats a `voice` that is an existing file as the clone source.
    wav, sr = backend.synthesize(gen_text, ref_audio_path, speed or 1.0,
                                 (language or "en")[:2])
    buf = _io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype="float32"), sr, format="WAV")
    return buf.getvalue()


def _base_tts_wav(gen_text: str, speed: float, language: str, model: str,
                  temps: list) -> str:
    """Synthesise the line in a base voice; return a WAV path for conversion."""
    import numpy as np
    import soundfile as sf
    backend = _get_tts_backend(model)
    wav, sr = backend.synthesize(gen_text, "", speed or 1.0, (language or "en")[:2])
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    sf.write(out.name, np.asarray(wav, dtype="float32"), sr)
    temps.append(out.name)
    return out.name


def _chain_clone(ref_audio_path: str, gen_text: str, speed: float, language: str,
                 base_model: str, temps: list) -> bytes:
    """OpenVoice's pattern: base TTS speaks it, seed-vc moves the timbre over.

    seed-vc is a zero-shot converter, so like OpenVoice's tone-colour converter it
    needs only the reference audio — no transcript, and the base voice decides the
    language and delivery."""
    import io as _io
    import numpy as np
    import soundfile as sf
    src = _base_tts_wav(gen_text, speed, language, base_model, temps)
    from codai.api.voice_convert import _get_wrapper
    wrapper = _get_wrapper()
    audio, sr = _seedvc_convert(wrapper, src, ref_audio_path)
    buf = _io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype="float32"), sr, format="WAV")
    return buf.getvalue()


def _save_audio_response(audio_bytes: bytes, http_request: Request) -> dict:
    import uuid
    filename = f"{uuid.uuid4().hex}.wav"
    if global_file_path:
        os.makedirs(global_file_path, exist_ok=True)
        fpath = os.path.join(global_file_path, filename)
        with open(fpath, 'wb') as f:
            f.write(audio_bytes)
        from codai.api.urlutils import build_file_url
        return {"url": build_file_url(filename, http_request)}
    return {"b64_wav": base64.b64encode(audio_bytes).decode()}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class VoiceExtractRequest(BaseModel):
    name: str
    description: str = ''
    audio: Optional[str] = None      # base64/URL audio file
    video: Optional[str] = None      # base64/URL video (audio track extracted)
    transcript: Optional[str] = ''   # optional; auto-transcribed if omitted
    model_config = ConfigDict(extra="allow")


class VoicePatchRequest(BaseModel):
    description: Optional[str] = None
    transcript: Optional[str] = None
    audio: Optional[str] = None      # replace reference audio (base64)
    add_clips: Optional[list] = None  # [{audio: b64, transcript: str}] — extra references
    remove_clips: Optional[list] = None  # 0-based indices into the clip list
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Voice profile management
# ---------------------------------------------------------------------------

@router.get("/v1/audio/voices", summary="List voice profiles")
async def list_voices():
    """List all saved voice profiles."""
    return {"voices": _list_voices()}


@router.post("/v1/audio/voices", summary="Create a voice profile")
async def create_voice(
    name: str = Form(...),
    transcript: str = Form(...),
    description: str = Form(''),
    audio: UploadFile = File(...),
):
    """Save a named voice profile from a reference audio file + transcript."""
    if not name.replace('-', '').replace('_', '').isalnum():
        raise HTTPException(status_code=400, detail="Voice name must be alphanumeric (hyphens/underscores allowed)")

    audio_bytes = await audio.read()
    ext = os.path.splitext(audio.filename)[1] or '.wav'

    # Validate audio is readable
    try:
        import soundfile as sf, io as _io
        sf.info(_io.BytesIO(audio_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    meta = _save_voice(name, audio_bytes, ext, transcript, description)
    return {"created": True, "voice": meta}


@router.delete("/v1/audio/voices/{name}", summary="Delete a voice profile")
async def delete_voice(name: str):
    """Delete a saved voice profile."""
    import shutil
    vdir = _voice_path(name)
    if not os.path.exists(vdir):
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    shutil.rmtree(vdir)
    return {"deleted": True, "name": name}


@router.patch("/v1/audio/voices/{name}", summary="Update a voice profile")
async def patch_voice(name: str, req: VoicePatchRequest):
    """Update description, transcript, or reference audio of a saved voice profile."""
    meta = _load_voice(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    vdir = _voice_path(name)

    if req.description is not None:
        meta['description'] = req.description
    if req.transcript is not None:
        meta['transcript'] = req.transcript
    if req.audio:
        audio_bytes, ext = _decode_audio(req.audio)
        audio_file = os.path.join(vdir, f'ref{ext}')
        # Remove old audio file(s)
        for f in os.listdir(vdir):
            if f.startswith('ref.') or f.startswith('ref'):
                try:
                    os.unlink(os.path.join(vdir, f))
                except Exception:
                    pass
        with open(audio_file, 'wb') as f:
            f.write(audio_bytes)
        meta['audio_file'] = audio_file
        meta['audio_ext'] = ext
        meta['clips'] = [{'path': audio_file, 'transcript': meta.get('transcript', '')}]

    # Extra reference clips. One take is a gamble — a cough, a room, a bad mic and
    # every cloned line inherits it. Several let the clone prompt on the cleanest,
    # and give the averaging engines something to average.
    if req.add_clips:
        clips = _profile_clips(meta)
        for i, c in enumerate(req.add_clips):
            if not isinstance(c, dict) or not c.get('audio'):
                continue
            raw, ext = _decode_audio(c['audio'])
            dest = os.path.join(vdir, f'clip_{int(time.time())}_{i:02d}{ext}')
            with open(dest, 'wb') as fh:
                fh.write(raw)
            clips.append({'path': dest, 'transcript': (c.get('transcript') or '').strip()})
        meta['clips'] = clips
    if req.remove_clips:
        clips = _profile_clips(meta)
        drop = {int(i) for i in req.remove_clips if isinstance(i, (int, float, str)) and str(i).isdigit()}
        kept = []
        for i, c in enumerate(clips):
            if i in drop:
                try:
                    os.unlink(c['path'])
                except OSError:
                    pass
                continue
            kept.append(c)
        meta['clips'] = kept
        if kept and meta.get('audio_file') not in [c['path'] for c in kept]:
            meta['audio_file'] = kept[0]['path']
            meta['transcript'] = kept[0].get('transcript') or meta.get('transcript', '')

    with open(os.path.join(vdir, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    return {"updated": True, "voice": meta, "clips": len(_profile_clips(meta))}


@router.get("/v1/audio/voices/{name}", summary="Get a voice profile")
async def get_voice(name: str):
    """Get a single voice profile metadata."""
    meta = _load_voice(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    return {"voice": meta}


@router.post("/v1/audio/voices/extract", summary="Extract a voice profile from a sample")
async def extract_voice(req: VoiceExtractRequest):
    """
    Extract a voice profile from a source audio or video file.

    - audio: base64/URL audio file (wav, mp3, flac, …)
    - video: base64/URL video file — audio track is extracted automatically
    - transcript: optional; auto-transcribed with Whisper when omitted
    """
    if not req.name.replace('-', '').replace('_', '').isalnum():
        raise HTTPException(status_code=400, detail="Voice name must be alphanumeric (hyphens/underscores allowed)")
    if not req.audio and not req.video:
        raise HTTPException(status_code=400, detail="Provide audio or video")

    temps = []
    try:
        audio_bytes: Optional[bytes] = None
        audio_ext = '.wav'

        if req.video:
            # Extract audio track from video with ffmpeg
            raw = _decode_b64_or_url(req.video)
            in_tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            in_tmp.write(raw)
            in_tmp.close()
            temps.append(in_tmp.name)
            out_tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            out_tmp.close()
            temps.append(out_tmp.name)
            r = subprocess.run(
                ['ffmpeg', '-y', '-i', in_tmp.name, '-vn', '-acodec', 'pcm_s16le',
                 '-ar', '22050', '-ac', '1', out_tmp.name],
                capture_output=True, timeout=120,
            )
            if r.returncode != 0:
                raise HTTPException(status_code=422,
                    detail=f"Audio extraction from video failed: {r.stderr.decode(errors='replace')[:200]}")
            with open(out_tmp.name, 'rb') as f:
                audio_bytes = f.read()
        else:
            audio_bytes, audio_ext = _decode_audio(req.audio)

        # Auto-transcribe if transcript not provided
        transcript = req.transcript or ''
        if not transcript:
            try:
                import whisper
                in_tmp2 = tempfile.NamedTemporaryFile(suffix=audio_ext, delete=False)
                in_tmp2.write(audio_bytes)
                in_tmp2.close()
                temps.append(in_tmp2.name)
                model = whisper.load_model('base')
                result = model.transcribe(in_tmp2.name)
                transcript = result.get('text', '').strip()
            except Exception:
                pass

        # Validate audio
        try:
            import soundfile as sf
            sf.info(io.BytesIO(audio_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")

        meta = _save_voice(req.name, audio_bytes, audio_ext, transcript, req.description)
        return {"created": True, "voice": meta}

    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Voice cloning TTS
# ---------------------------------------------------------------------------

class VoiceCloneRequest(BaseModel):
    text: str                               # text to synthesize
    voice_name: Optional[str] = None        # use a saved voice profile
    ref_audio: Optional[str] = None         # base64 reference audio (if not using saved voice)
    ref_text: Optional[str] = None          # transcript of ref_audio
    speed: Optional[float] = 1.0
    seed: Optional[int] = None
    engine: Optional[str] = "auto"          # auto | f5 | xtts | chain | melotts
    language: Optional[str] = None          # target language (xtts/chain/melotts)
    ref_language: Optional[str] = None      # language the reference speaks
    base_model: Optional[str] = None        # base voice for engine=chain
    watermark: Optional[bool] = None        # override the global watermark setting
    response_format: Optional[str] = "url"
    model_config = ConfigDict(extra="allow")


@router.post("/v1/audio/watermark/detect", summary="Detect a coderai audio watermark")
async def watermark_detect(request: Request):
    """Was this audio synthesised here? Accepts base64/data-URI/URL audio.

    Answers the question the watermark exists for: given a clip in the wild, did it
    come out of this server."""
    body = await request.json()
    data = body.get("audio") or body.get("file") or ""
    if not data:
        raise HTTPException(status_code=400, detail="Provide 'audio' (base64, data URI or URL)")
    try:
        raw = _decode_b64_or_url(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}")
    return detect_watermark(raw)


@router.get("/v1/audio/clone/engines", summary="List voice cloning engines")
async def list_clone_engines():
    """Which cloning engines this install can actually run, and what each needs."""
    def _installed(mod: str) -> bool:
        import importlib.util
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:
            return False

    return {"default": "auto", "engines": [
        {"id": "f5", "installed": _installed("f5_tts"),
         "needs_transcript": True, "cross_lingual": False,
         "detail": "In-context clone: copies timbre AND prosody/accent. Best when "
                   "the reference speaks the target language."},
        {"id": "xtts", "installed": _installed("TTS"),
         "needs_transcript": False, "cross_lingual": True,
         "detail": "XTTS-v2 zero-shot from the clip, 17 languages."},
        {"id": "chain", "installed": _installed("seed_vc"),
         "needs_transcript": False, "cross_lingual": True,
         "detail": f"Base TTS ({DEFAULT_CHAIN_BASE}) then seed-vc timbre transfer — "
                   f"OpenVoice's architecture with seed-vc as the converter."},
        {"id": "melotts", "installed": _installed("melo") or _installed("MeloTTS"),
         "needs_transcript": False, "cross_lingual": True,
         "detail": "Same chain with MeloTTS as the base voice (OpenVoice V2's base)."},
    ], "watermark": {"default_on": WATERMARK_DEFAULT,
                     "available": _installed("audioseal")}}


@router.post("/v1/audio/clone", summary="Clone a voice / synthesize cloned speech")
async def clone_voice(request: VoiceCloneRequest, http_request: Request = None):
    """
    Synthesize speech in a cloned voice using F5-TTS.

    Provide either:
    - voice_name: name of a saved voice profile
    - ref_audio (base64) + ref_text: inline reference audio
    """
    # Resolve reference audio
    ref_audio_path = None
    ref_text = request.ref_text or ''
    temps = []

    try:
        ref_language = request.ref_language or ''
        if request.voice_name:
            meta = _load_voice(request.voice_name)
            if not meta:
                raise HTTPException(status_code=404, detail=f"Voice '{request.voice_name}' not found")
            clip = _best_clip(meta) or {}
            ref_audio_path = clip.get('path') or meta['audio_file']
            ref_text = ref_text or clip.get('transcript') or meta.get('transcript', '')
            ref_language = ref_language or meta.get('language', '')
        elif request.ref_audio:
            audio_bytes, ext = _decode_audio(request.ref_audio)
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            ref_audio_path = tmp.name
            temps.append(ref_audio_path)
        else:
            raise HTTPException(status_code=400, detail="Provide voice_name or ref_audio")

        # A saved profile can hold minutes of audio; every engine here prompts on
        # a short clip.
        ref_audio_path = _trim_reference(ref_audio_path, temps)

        engine = _pick_engine(request.engine, ref_text, request.language, ref_language)
        speed = request.speed or 1.0
        lang = (request.language or ref_language or "en")[:2]

        def _run():
            if engine == "f5":
                if not ref_text:
                    raise HTTPException(status_code=400, detail=(
                        "engine=f5 needs ref_text (the reference transcript). "
                        "Use engine=xtts or engine=chain to clone without one."))
                return _f5tts_clone(ref_audio_path, ref_text, request.text, speed, request.seed)
            if engine == "xtts":
                return _xtts_clone(ref_audio_path, request.text, speed, lang)
            base = request.base_model or ("melotts/MeloTTS" if engine == "melotts"
                                          else DEFAULT_CHAIN_BASE)
            return _chain_clone(ref_audio_path, request.text, speed, lang, base, temps)

        try:
            audio_bytes = await asyncio.get_event_loop().run_in_executor(None, _run)
        except HTTPException:
            raise
        except ImportError as e:
            raise HTTPException(status_code=501, detail=(
                f"engine '{engine}' is not installed: {e}"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Voice cloning failed ({engine}): {e}")

        audio_bytes = apply_watermark(audio_bytes, request.watermark)
        result = _save_audio_response(audio_bytes, http_request)
        return {"created": int(time.time()), "data": [result], "engine": engine}

    finally:
        for t in temps:
            try:
                os.unlink(t)
            except Exception:
                pass
