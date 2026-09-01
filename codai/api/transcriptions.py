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

"""
Audio transcription endpoint for the codai API.
"""

import asyncio
import io
import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from typing import List, Optional

# Maximum upload size: 100 MB
_MAX_AUDIO_BYTES = 100 * 1024 * 1024

# Safe audio extensions (user-supplied extension is NOT trusted for the suffix)
_SAFE_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm', '.mp4'}

# Import from codai modules
from codai.models.manager import multi_model_manager


# Global reference to be set by coderai
global_args = None


def set_global_args(args):
    """Set global args from coderai."""
    global global_args
    global_args = args


# =============================================================================
# Response formatting helpers
# =============================================================================

def _seconds_to_srt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace('.', ',')


def _seconds_to_vtt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def _format_response(fmt: str, text: str, segments: list, words: list = None):
    """Format a transcription result according to the requested response_format.

    ``words`` (when timestamp_granularities[]=word was requested) is surfaced as a
    top-level ``words`` array in verbose_json, OpenAI-style."""
    fmt = (fmt or "json").lower()

    if fmt == "text":
        return PlainTextResponse(text)

    def _spk_prefix(seg):
        spk = seg.get("speaker")
        return f"[{spk}] " if spk else ""

    if fmt == "srt":
        lines = []
        for i, seg in enumerate(segments, 1):
            start = _seconds_to_srt_time(seg.get("start", 0))
            end = _seconds_to_srt_time(seg.get("end", 0))
            lines.append(f"{i}\n{start} --> {end}\n{_spk_prefix(seg)}{seg['text'].strip()}\n")
        srt_body = "\n".join(lines) if lines else f"1\n00:00:00,000 --> 00:00:00,000\n{text}\n"
        return PlainTextResponse(srt_body, media_type="text/plain")

    if fmt == "vtt":
        lines = ["WEBVTT\n"]
        for seg in segments:
            start = _seconds_to_vtt_time(seg.get("start", 0))
            end = _seconds_to_vtt_time(seg.get("end", 0))
            lines.append(f"{start} --> {end}\n{_spk_prefix(seg)}{seg['text'].strip()}\n")
        if not segments:
            lines.append(f"00:00:00.000 --> 00:00:00.000\n{text}\n")
        return PlainTextResponse("\n".join(lines), media_type="text/vtt")

    if fmt == "verbose_json":
        out = {
            "task": "transcribe",
            "language": "unknown",
            "duration": segments[-1].get("end", 0) if segments else 0,
            "text": text,
            "segments": [
                {
                    "id": i,
                    "start": s.get("start", 0),
                    "end": s.get("end", 0),
                    "text": s.get("text", "").strip(),
                    **({"speaker": s["speaker"]} if s.get("speaker") else {}),
                }
                for i, s in enumerate(segments)
            ],
        }
        if words:
            out["words"] = [
                {"word": w.get("word", ""), "start": w.get("start", 0),
                 "end": w.get("end", 0)}
                for w in words
            ]
        return out

    # Default: json. Include segments when they carry speaker labels (diarize=true),
    # so a plain json request still surfaces who-spoke-when.
    if any(s.get("speaker") for s in (segments or [])):
        return {
            "text": text,
            "segments": [
                {"start": s.get("start", 0), "end": s.get("end", 0),
                 "text": s.get("text", "").strip(),
                 **({"speaker": s["speaker"]} if s.get("speaker") else {})}
                for s in segments
            ],
        }
    return {"text": text}


# =============================================================================
# Language selection
# =============================================================================

def _validate_language(model, config: dict, language, target_language):
    """Validate requested language(s) against the model's declared languages.

    A model config may carry ``languages`` (a list of supported ISO codes/names).
    When it does and the caller passes a ``language``/``target_language`` not in
    that list, fail with a clear 400 listing what's allowed. Models that declare
    no languages accept anything (pass-through). ``target_language`` additionally
    requires the model to advertise ``supports_translation``."""
    langs = config.get("languages") or []
    norm = {str(x).strip().lower() for x in langs}
    if language and norm and str(language).strip().lower() not in norm:
        raise HTTPException(
            status_code=400,
            detail=f"Language '{language}' not supported by '{model}'. "
                   f"Supported: {sorted(langs)}")
    if target_language:
        if not config.get("supports_translation"):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' does not support translation "
                       "(target_language). Use a NeMo/Canary model that does.")
        if norm and str(target_language).strip().lower() not in norm:
            raise HTTPException(
                status_code=400,
                detail=f"Target language '{target_language}' not supported by "
                       f"'{model}'. Supported: {sorted(langs)}")


# =============================================================================
# Router and Endpoints
# =============================================================================

router = APIRouter()


@router.post("/v1/audio/transcriptions", summary="Transcribe audio to text")
async def create_transcription(
    model: str = Form(...),
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    target_language: Optional[str] = Form(None),
    diarize: Optional[bool] = Form(False),
    num_speakers: Optional[int] = Form(None),
    timestamp_granularities: Optional[List[str]] = Form(None),
    timestamp_granularities_br: Optional[List[str]] = Form(None,
        alias="timestamp_granularities[]"),
):
    """
    Audio transcription endpoint (OpenAI-compatible).

    ``language`` selects the source language (validated against the model's
    declared languages when it declares any). ``target_language`` requests
    translation into that language for backends that support it (NVIDIA Canary);
    other backends ignore it. ``diarize=true`` additionally runs pyannote speaker
    diarization and tags each returned segment with a ``speaker`` (optionally
    hinted by ``num_speakers``).
    """
    file_content = await file.read()
    if len(file_content) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (max 100 MB)")

    # Register a task so transcription appears in the unified task list, like
    # every other model type. Finished on success or error below.
    from codai.tasks import task_registry
    _tid = task_registry.register(
        "transcription",
        title=(file.filename or "audio")[:80],
        model=model or "",
    )
    task_registry.start(_tid)
    _grans = [g.lower() for g in ((timestamp_granularities or [])
                                  + (timestamp_granularities_br or []))]
    _word_ts = "word" in _grans
    try:
        _resp = await _run_transcription(
            file_content, model, language, prompt, response_format, temperature, file,
            target_language, diarize, num_speakers, _word_ts)
        task_registry.finish(_tid, "done")
        return _resp
    except HTTPException:
        task_registry.finish(_tid, "error")
        raise
    except Exception as e:
        task_registry.finish(_tid, "error", str(e)[:200])
        raise


@router.post("/v1/audio/diarization", summary="Speaker diarization (who spoke when)")
async def create_diarization(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    response_format: Optional[str] = Form("json"),
    identify: Optional[bool] = Form(False),
    identify_backend: Optional[str] = Form("ecapa"),
    identify_threshold: Optional[float] = Form(0.25),
):
    """Standalone speaker diarization via pyannote.audio.

    Returns ``{"segments":[{start,end,speaker}], "num_speakers":N}`` (or srt/vtt).
    ``model`` selects the pyannote pipeline (defaults to the configured/ungated
    one); ``num_speakers`` pins the count, or bound it with min/max_speakers.
    ``identify=true`` matches each diarized speaker against the enrolled registry
    and replaces the labels with recognised names (adds ``speakers`` = {label:name})."""
    file_content = await file.read()
    if len(file_content) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (max 100 MB)")

    from codai.tasks import task_registry
    _tid = task_registry.register(
        "diarization", title=(file.filename or "audio")[:80], model=model or "")
    task_registry.start(_tid)
    fd, tmp_path = tempfile.mkstemp(suffix=".audio")
    with os.fdopen(fd, "wb") as f:
        f.write(file_content)
    try:
        import codai.api.diarization as diarization
        result = await asyncio.to_thread(
            diarization.diarize_audio, tmp_path, num_speakers, min_speakers, max_speakers,
            model, None, bool(identify), identify_backend or "ecapa",
            identify_threshold if identify_threshold is not None else 0.25)
        task_registry.finish(_tid, "done")
        fmt = (response_format or "json").lower()
        if fmt in ("srt", "vtt"):
            segs = [{"start": s["start"], "end": s["end"], "text": "",
                     "speaker": s["speaker"]} for s in result.get("segments", [])]
            return _format_response(response_format, "", segs)
        return result
    except HTTPException:
        task_registry.finish(_tid, "error")
        raise
    except Exception as e:
        task_registry.finish(_tid, "error", str(e)[:200])
        raise HTTPException(status_code=500, detail=f"Diarization error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/v1/audio/speaker-embeddings", summary="Extract a speaker embedding (voiceprint)")
async def create_speaker_embedding(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    backend: Optional[str] = Form("ecapa"),
    window: Optional[float] = Form(None),
    step: Optional[float] = Form(None),
):
    """Speaker-embedding (voiceprint) endpoint.

    Returns a fixed-dim embedding for the speaker in the audio, usable for
    verification / clustering. ``backend``: ``ecapa`` (speechbrain ECAPA-TDNN,
    ungated, default), ``pyannote`` / ``wespeaker`` (pyannote embedding models,
    may need HF_TOKEN). ``model`` overrides the backend's default checkpoint.

    ``window`` (seconds) switches to sliding-window mode: one embedding per window
    of that length, hop ``step`` (default = window), each tagged with start/end —
    instead of a single whole-file embedding."""
    file_content = await file.read()
    if len(file_content) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (max 100 MB)")

    from codai.tasks import task_registry
    _tid = task_registry.register(
        "speaker-embedding", title=(file.filename or "audio")[:80], model=(backend or "ecapa"))
    task_registry.start(_tid)
    fd, tmp_path = tempfile.mkstemp(suffix=".audio")
    with os.fdopen(fd, "wb") as f:
        f.write(file_content)
    try:
        import codai.api.speaker_embeddings as spk
        result = await asyncio.to_thread(
            spk.get_speaker_embedding, tmp_path, backend or "ecapa", model, None,
            window, step)
        task_registry.finish(_tid, "done")
        if result.get("windows") is not None:
            # Sliding-window mode: one embedding per window, each with its span.
            return {
                "object": "list",
                "model": result.get("model"),
                "backend": result.get("backend"),
                "dim": result.get("dim"),
                "window": result.get("window"),
                "step": result.get("step"),
                "count": result.get("count"),
                "data": [{"object": "embedding", "index": i,
                          "start": w.get("start"), "end": w.get("end"),
                          "embedding": w.get("embedding")}
                         for i, w in enumerate(result.get("windows") or [])],
            }
        return {
            "object": "list",
            "model": result.get("model"),
            "backend": result.get("backend"),
            "dim": result.get("dim"),
            "data": [{"object": "embedding", "index": 0,
                      "embedding": result.get("embedding")}],
        }
    except HTTPException:
        task_registry.finish(_tid, "error")
        raise
    except Exception as e:
        task_registry.finish(_tid, "error", str(e)[:200])
        raise HTTPException(status_code=500, detail=f"Speaker-embedding error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def _embed_upload(file, backend, model):
    """Read an upload to a temp file and return its speaker embedding dict."""
    file_content = await file.read()
    if len(file_content) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (max 100 MB)")
    fd, tmp_path = tempfile.mkstemp(suffix=".audio")
    with os.fdopen(fd, "wb") as f:
        f.write(file_content)
    try:
        import codai.api.speaker_embeddings as spk
        return await asyncio.to_thread(
            spk.get_speaker_embedding, tmp_path, backend or "ecapa", model)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/v1/audio/speakers", summary="Enroll a speaker voiceprint")
async def enroll_speaker(
    file: UploadFile = File(...),
    name: str = Form(...),
    backend: Optional[str] = Form("ecapa"),
    model: Optional[str] = Form(None),
):
    """Enroll (or add a sample to) a named speaker from an audio clip. Re-enrolling
    the same name averages the voiceprint. Use the SAME backend for enroll +
    identify (embeddings from different backends aren't comparable)."""
    emb = await _embed_upload(file, backend, model)
    from codai.api import speaker_registry
    try:
        return speaker_registry.enroll(name, emb["embedding"], emb["backend"], emb.get("model"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/v1/audio/speakers", summary="List enrolled speakers")
async def list_speakers_endpoint():
    from codai.api import speaker_registry
    return {"speakers": speaker_registry.list_speakers()}


@router.delete("/v1/audio/speakers/{name}", summary="Delete an enrolled speaker")
async def delete_speaker_endpoint(name: str, backend: Optional[str] = None):
    from codai.api import speaker_registry
    removed = speaker_registry.delete(name, backend)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No enrolled speaker '{name}'")
    return {"deleted": removed, "name": name}


@router.post("/v1/audio/speaker-identify", summary="Identify the speaker in an audio clip")
async def identify_speaker_endpoint(
    file: UploadFile = File(...),
    backend: Optional[str] = Form("ecapa"),
    model: Optional[str] = Form(None),
    threshold: Optional[float] = Form(0.25),
):
    """Return the best-matching enrolled speaker (or "unknown") for the clip."""
    emb = await _embed_upload(file, backend, model)
    from codai.api import speaker_registry
    return speaker_registry.identify(emb["embedding"], emb["backend"],
                                     threshold=threshold if threshold is not None else 0.25)


@router.post("/v1/audio/speaker-verify", summary="Verify a clip against an enrolled speaker")
async def verify_speaker_endpoint(
    file: UploadFile = File(...),
    name: str = Form(...),
    backend: Optional[str] = Form("ecapa"),
    model: Optional[str] = Form(None),
    threshold: Optional[float] = Form(0.25),
):
    """Verify whether the clip is the enrolled speaker ``name`` (same/different)."""
    emb = await _embed_upload(file, backend, model)
    from codai.api import speaker_registry
    rec = speaker_registry.get(name, emb["backend"])
    if rec is None:
        raise HTTPException(status_code=404,
                            detail=f"No enrolled speaker '{name}' for backend '{emb['backend']}'")
    out = speaker_registry.verify(emb["embedding"], rec["embedding"],
                                  threshold=threshold if threshold is not None else 0.25)
    out["name"] = name
    return out


async def _run_transcription(
    file_content: bytes, model: str, language, prompt, response_format, temperature, file,
    target_language=None, diarize=False, num_speakers=None, word_timestamps=False
):
    """Core transcription logic; registered as a task by create_transcription()."""
    # When diarization is requested, keep a temp copy of the audio for pyannote and
    # wrap _format_response so every return path merges speaker labels into the
    # segments. Runs in a thread (pyannote load/inference is blocking).
    _diar_path = None
    if diarize:
        _dfd, _diar_path = tempfile.mkstemp(suffix=".audio")
        with os.fdopen(_dfd, "wb") as _df:
            _df.write(file_content)

    async def _finalize(fmt, text, segments, words=None):
        segs = segments or []
        if diarize:
            try:
                diar = await asyncio.to_thread(
                    diarization.diarize_audio, _diar_path, num_speakers)
                from codai.api.diarization import merge_speakers
                if segs:
                    segs = merge_speakers(segs, diar.get("segments") or [])
                else:
                    # No ASR timing (e.g. whisper-server text-only): surface the raw
                    # speaker turns as segments so who-spoke-when is still returned.
                    segs = diar.get("segments") or []
            except Exception as e:
                print(f"Diarization failed (returning transcription without speakers): {e}")
        return _format_response(fmt, text, segs, words if word_timestamps else None)

    import codai.api.diarization as diarization
    try:
        return await _run_transcription_inner(
            file_content, model, language, prompt, response_format, temperature, file,
            target_language, _finalize, word_timestamps)
    finally:
        if _diar_path:
            try:
                os.unlink(_diar_path)
            except OSError:
                pass


async def _run_transcription_inner(
    file_content: bytes, model: str, language, prompt, response_format, temperature, file,
    target_language, _finalize, word_timestamps=False
):
    """Backend dispatch; calls ``_finalize(fmt, text, segments, words)`` at each
    return point so diarization + word timestamps are handled uniformly."""
    # Check if the requested model maps to a configured whisper-server instance first.
    # Try alias round-robin resolution before direct ID lookup.
    whisper_model_id = multi_model_manager.resolve_whisper_alias_model_id(model)
    whisper_server = (
        multi_model_manager.whisper_servers.get(whisper_model_id)
        if whisper_model_id is not None
        else multi_model_manager.whisper_servers.get(model)
    )
    if whisper_server is not None:
        await asyncio.to_thread(
            multi_model_manager.request_model, requested_model=model, model_type="audio")
        if not whisper_server.is_running():
            # Treat starting the runner as a model load: evict other models for its
            # VRAM and register it in the loaded-model maps (so it's evictable too).
            await asyncio.to_thread(
                multi_model_manager.start_whisper_server, whisper_model_id or model)
        if not whisper_server.is_running():
            raise HTTPException(status_code=500, detail="whisper-server failed to start")
        result = whisper_server.transcribe(
            file_content,
            language=language,
            prompt=prompt
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return await _finalize(response_format, result.get("text", ""), [])

    # Use the manager to resolve the model and manage VRAM
    model_info = await asyncio.to_thread(
        multi_model_manager.request_model,
        requested_model=model,
        model_type="audio"
    )

    # Check if the model was rejected as not allowed
    if model_info.get('error'):
        raise HTTPException(status_code=404, detail=model_info['error'])

    model_name = model_info['model_name']
    model_key = model_info['model_key']
    whisper_model = model_info['model_object']
    audio_config = model_info.get('config') or {}
    # The manager stores the runtime-kwargs dict as `config`, with the full
    # models.json entry nested under `_raw_cfg`. Backend/languages/model_path live
    # there — read from it (falling back to the flat dict for older shapes).
    _raw = audio_config.get('_raw_cfg') if isinstance(audio_config, dict) else None
    stt_cfg = _raw if isinstance(_raw, dict) else (audio_config or {})

    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="Audio transcription not configured. Use --audio-model or --whisper-server."
        )

    cfg_backend = (stt_cfg.get('backend') or '').strip().lower()

    # A whisper.cpp GGUF (or a model configured with the whisper-server backend)
    # cannot be loaded by faster-whisper — handing it a .gguf path or a whisper
    # alias makes faster-whisper raise the opaque "Invalid model size". That only
    # happens when the request reached an engine that has no whisper-server
    # registered for this model (routing should pin it to its owner engine). Fail
    # with an actionable message instead of the misleading size error.
    if cfg_backend == 'whisper-server' or (
            isinstance(model_name, str) and model_name.lower().endswith(".gguf")):
        raise HTTPException(
            status_code=503,
            detail=(f"'{model}' is a whisper-server (whisper.cpp GGUF) model but no "
                    "whisper-server is available on the engine handling this request. "
                    "It must run on its assigned engine.")
        )

    # Validate the requested language against the model's declared languages, if any.
    _validate_language(model, stt_cfg, language, target_language)

    # Determine a safe file extension from the upload's content-type or filename,
    # never trusting the raw user-supplied value for arbitrary suffixes.
    raw_ext = os.path.splitext(file.filename or '')[1].lower()
    safe_ext = raw_ext if raw_ext in _SAFE_EXTENSIONS else '.wav'

    # Save to temp file (needed for some backends)
    with tempfile.NamedTemporaryFile(delete=False, suffix=safe_ext) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name
    
    try:
        # Pluggable STT backends (wav2vec2 / vosk / NVIDIA NeMo-Canary), selected
        # by the model's configured `backend`. These are NOT faster-whisper and
        # must be dispatched before the whisper fallback below.
        import codai.api.stt_backends as stt_backends
        family = stt_backends.resolve_family(model_name, stt_cfg)
        if family in stt_backends.STT_FAMILIES:
            # VRAM accounting: NeMo runs on the 3090 via its isolated worker;
            # wav2vec2 uses this process's GPU (CUDA when available); vosk is CPU.
            uses_gpu = False
            needed_gb = 0.0
            if family == 'nemo':
                uses_gpu = True
                needed_gb = float(stt_cfg.get('used_vram_gb') or 6.0)
            elif family == 'wav2vec2':
                try:
                    import torch
                    uses_gpu = bool(torch.cuda.is_available())
                except Exception:
                    uses_gpu = False
                needed_gb = float(stt_cfg.get('used_vram_gb') or 2.0) if uses_gpu else 0.0

            def _run_backend():
                def _loader():
                    return stt_backends.load_stt_backend(
                        model_name, stt_cfg.get('model_path'), stt_cfg)
                # Evicts other models for `needed_gb` before loading a GPU backend,
                # then registers it so the normal LRU eviction can free it later.
                # keep_resident (models.json) marks the small co-resident set so it
                # isn't swapped out by routine eviction (only as a last resort).
                backend = multi_model_manager.acquire_stt_backend(
                    model_key, needed_gb, uses_gpu, _loader,
                    keep_resident=bool(stt_cfg.get('keep_resident')))
                return backend.transcribe(
                    tmp_path, language=language, prompt=prompt,
                    temperature=temperature, target_language=target_language,
                    word_timestamps=word_timestamps)
            result = await asyncio.to_thread(_run_backend)
            return await _finalize(
                response_format, result.get("text", ""), result.get("segments") or [],
                result.get("words") or [])

        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel

            if whisper_model is None:
                whisper_model = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                )
                multi_model_manager.add_model(model_key, whisper_model)
                multi_model_manager.current_model_key = model_key

            raw_segments, _ = whisper_model.transcribe(
                tmp_path,
                language=language,
                initial_prompt=prompt,
                temperature=temperature,
                word_timestamps=word_timestamps,
            )
            # Materialise the generator so we have all segment data
            segments, words = [], []
            for s in raw_segments:
                segments.append({"start": s.start, "end": s.end, "text": s.text})
                for w in (getattr(s, "words", None) or []):
                    words.append({"word": w.word, "start": w.start, "end": w.end})
            full_text = "".join(s["text"] for s in segments)
            return await _finalize(response_format, full_text.strip(), segments, words)

        except ImportError:
            pass

        # Try whispercpp as fallback
        try:
            import whispercpp

            if whisper_model is None:
                whisper_model = whispercpp.Whisper.from_pretrained(model_name)
                multi_model_manager.add_model(model_key, whisper_model)
                multi_model_manager.current_model_key = model_key

            result = whisper_model.transcribe(tmp_path)

            text = ""
            if hasattr(result, 'text'):
                text = result.text
            elif isinstance(result, dict):
                text = result.get('text', '')
            elif isinstance(result, list):
                for segment in result:
                    if hasattr(segment, 'text'):
                        text += segment.text
                    elif isinstance(segment, dict):
                        text += segment.get('text', '')

            # whispercpp does not expose per-segment timestamps easily
            return await _finalize(response_format, text.strip(), [])

        except ImportError as e:
            raise HTTPException(
                status_code=501,
                detail="Audio transcription not available. Install faster-whisper or whispercpp."
            )
            
    except Exception as e:
        print(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
