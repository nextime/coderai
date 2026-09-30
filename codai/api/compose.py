# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""``POST /v1/video/compose`` — N pieces in, one finished video out.

CoderAI could already make every piece of a short video; nothing turned the
pieces into a reel, because that step needs ffmpeg *and* facts only the server
has — how long the narration it just synthesised actually is, and when each
word falls. A client on another machine cannot know either, so composition has
to live here.

A request is an ordered list of **scenes**. Each scene has narration (text to
synthesise, or audio you already have) and one or more **visuals**; the scene
lasts as long as its narration plus its padding, and its visuals split that time
between them. Everything else — voice, music, captions, overlays, thumbnail — is
defaults you can override per request.

The work is queued and run one render at a time (a render is ffmpeg-bound and
two at once are slower than one after the other), reported through
``GET /v1/video/compose/{id}`` with a stage and a percentage, and stoppable with
``POST /v1/video/compose/{id}/cancel``. A visual that cannot be fetched or
decoded never fails the job: it becomes a gradient and a line in ``warnings``.

The pipeline itself is :mod:`codai.compose.render`; captions are
:mod:`codai.compose.captions`; this module is the API, the queue and the
orchestration between them.

Narration, music and word timings are taken from this server's own endpoints,
called in-process on a loop this module owns (never the loop of the request that
queued the job — that one belongs to one request and may be closed long before
the render needs it). In-process means they run in whichever engine the front
routed the compose request to, so on a multi-engine install the TTS/STT models
must be loadable there; ``docs/video-compose.md`` says so where an operator will
read it.
"""

import asyncio
import base64
import os
import shutil
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from codai.api.loras import _require_api_auth
from codai.compose import captions as cap
from codai.compose import media
from codai.compose import render as rnd

router = APIRouter()


def set_global_file_path(path: str) -> None:
    """Where the composed artefacts are written (``/v1/files`` serves them)."""
    media.set_global_file_path(path)


_MAX_SCENES = 200
_MAX_VISUALS_PER_SCENE = 12
_JOB_HISTORY = 100
_STAGES = ("queued", "narration", "captions", "visuals", "music", "render", "thumbnail")


# =============================================================================
# Request
# =============================================================================

class Canvas(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = 30
    background: str = "#000000"
    model_config = ConfigDict(extra="allow")


class VoiceSpec(BaseModel):
    engine: str = "tts"                     # tts → /v1/audio/speech, clone → /v1/audio/clone
    model: Optional[str] = None
    voice: Optional[str] = None
    voice_name: Optional[str] = None        # saved voice profile (engine=clone)
    language: Optional[str] = None
    speed: float = 1.0
    model_config = ConfigDict(extra="allow")


class Visual(BaseModel):
    type: str = "image"                     # video | image | gradient | color
    src: Optional[str] = None
    trim_start: float = 0.0
    ken_burns: Optional[Any] = None
    colors: Optional[List[str]] = None
    color: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class Scene(BaseModel):
    text: Optional[str] = None
    audio: Optional[str] = None
    duration: Optional[float] = None        # required when there is no narration
    min_duration: float = 0.0
    padding: float = 0.15
    visuals: List[Visual] = Field(default_factory=list)
    voice: Optional[VoiceSpec] = None       # per-scene override (a second speaker)
    model_config = ConfigDict(extra="allow")


class MusicSpec(BaseModel):
    src: Optional[str] = None
    generate: Optional[Dict[str, Any]] = None
    volume: float = 0.14
    loop: bool = True
    duck: bool = True
    fade_in: float = 1.0
    fade_out: float = 2.0
    model_config = ConfigDict(extra="allow")


class CaptionSpec(BaseModel):
    enabled: bool = True
    timing: str = "auto"                    # auto | stt | estimate
    stt_model: Optional[str] = None
    preset: str = "karaoke"
    font: str = "Montserrat"
    font_scale: float = 1.0
    color: str = "#FFFFFF"
    highlight_color: str = "#FFE14D"
    outline_color: str = "#000000"
    position: str = "center"
    max_words_per_line: int = 4
    max_lines: int = 2
    uppercase: bool = False
    model_config = ConfigDict(extra="allow")


class Overlay(BaseModel):
    type: str = "text"                      # text | image
    text: Optional[str] = None
    src: Optional[str] = None
    position: str = "bottom-right"
    opacity: float = 1.0
    size: float = 0.025                     # text height as a fraction of the canvas
    width: float = 0.15                     # image width as a fraction of the canvas
    color: str = "#FFFFFF"
    font: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class ThumbnailSpec(BaseModel):
    at: float = 0.25
    text: Optional[str] = None
    text_color: str = "#FFFFFF"
    highlight_color: str = "#FFE14D"
    font: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class OutputSpec(BaseModel):
    format: str = "mp4"
    video_codec: str = "h264"
    crf: int = 20
    audio_bitrate: str = "192k"
    model_config = ConfigDict(extra="allow")


class ComposeRequest(BaseModel):
    scenes: List[Scene]
    canvas: Canvas = Field(default_factory=Canvas)
    voice: VoiceSpec = Field(default_factory=VoiceSpec)
    visual_fit: str = "cover"               # cover | contain | blur_pad
    transition: Dict[str, Any] = Field(default_factory=lambda: {"type": "none",
                                                               "duration": 0.3})
    music: Optional[MusicSpec] = None
    captions: Optional[CaptionSpec] = None
    overlays: List[Overlay] = Field(default_factory=list)
    thumbnail: Optional[ThumbnailSpec] = None
    output: OutputSpec = Field(default_factory=OutputSpec)
    outputs: List[str] = Field(default_factory=lambda: ["video"])
    is_async: bool = Field(default=True, alias="async")
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# =============================================================================
# Jobs
# =============================================================================

_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()
_cancels: Dict[str, threading.Event] = {}
_queue: List[str] = []                       # job ids waiting, in order
_worker: Optional[threading.Thread] = None
_worker_wake = threading.Event()
_pending: Dict[str, tuple] = {}              # job id -> (request, loop, base_url)


def _now() -> float:
    return time.time()


def _new_job(job_id: str, scenes: int) -> dict:
    rec = {"id": job_id, "status": "queued", "stage": "queued", "progress": 0.0,
           "message": "queued", "warnings": [], "error": None, "result": None,
           "scenes": scenes, "created_at": _now(), "updated_at": _now()}
    with _jobs_lock:
        _jobs[job_id] = rec
        _prune_locked()
    return dict(rec)


def _prune_locked() -> None:
    done = [(jid, r) for jid, r in _jobs.items()
            if r.get("status") in ("done", "failed", "cancelled")]
    excess = len(done) - _JOB_HISTORY
    if excess <= 0:
        return
    done.sort(key=lambda kv: kv[1].get("updated_at") or 0)
    for jid, _ in done[:excess]:
        _jobs.pop(jid, None)
        _cancels.pop(jid, None)


def _update(job_id: str, **fields) -> None:
    with _jobs_lock:
        rec = _jobs.get(job_id)
        if rec is None:
            return
        warn = fields.pop("warning", None)
        if warn:
            rec.setdefault("warnings", []).append(str(warn)[:300])
        rec.update(fields)
        rec["updated_at"] = _now()


def _get(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        rec = _jobs.get(job_id)
        return dict(rec) if rec else None


def _public(rec: dict) -> dict:
    """The document a client polls: the spec's fields, in the spec's order."""
    return {"id": rec["id"], "status": rec["status"], "stage": rec.get("stage"),
            "progress": round(float(rec.get("progress") or 0.0), 1),
            "message": rec.get("message") or "", "warnings": rec.get("warnings") or [],
            "error": rec.get("error"), "result": rec.get("result")}


def cancel_job(job_id: str) -> bool:
    rec = _get(job_id)
    if rec is None:
        return False
    if rec["status"] in ("done", "failed", "cancelled"):
        return True
    ev = _cancels.get(job_id)
    if ev is not None:
        ev.set()
    with _jobs_lock:
        _queue[:] = [j for j in _queue if j != job_id]
    _update(job_id, status="cancelled", message="cancelled by the client",
            stage=rec.get("stage"))
    return True


def _queue_positions() -> None:
    """Tell every waiting job where it is in the line (Motus shows this)."""
    with _jobs_lock:
        waiting = list(_queue)
    for pos, jid in enumerate(waiting, 1):
        rec = _get(jid)
        if rec and rec["status"] == "queued":
            _update(jid, message=f"queued: {pos} ahead of this job"
                    if pos > 1 else "queued: next")


def _ensure_worker() -> None:
    global _worker
    if _worker is not None and _worker.is_alive():
        return
    _worker = threading.Thread(target=_worker_loop, name="compose-worker", daemon=True)
    _worker.start()


def _worker_loop() -> None:
    """One render at a time, in submission order."""
    while True:
        _worker_wake.wait(timeout=5.0)
        _worker_wake.clear()
        while True:
            with _jobs_lock:
                job_id = _queue.pop(0) if _queue else None
            if job_id is None:
                break
            args = _pending.pop(job_id, None)
            rec = _get(job_id)
            if args is None or rec is None or rec["status"] == "cancelled":
                continue
            _queue_positions()
            req, base_url = args
            try:
                # The loop is started on demand: a job whose narration is supplied
                # and whose captions are estimated never needs one.
                _run_job(job_id, req, None, base_url)
            except rnd.Cancelled:
                _update(job_id, status="cancelled", message="cancelled by the client")
            except Exception as exc:                      # never kill the worker
                _update(job_id, status="failed", error=str(exc)[:500],
                        message=f"failed: {exc}"[:300])


def submit(req: ComposeRequest, base_url: str) -> dict:
    job_id = "cmp_" + uuid.uuid4().hex[:10]
    rec = _new_job(job_id, len(req.scenes))
    _cancels[job_id] = threading.Event()
    _pending[job_id] = (req, base_url)
    with _jobs_lock:
        _queue.append(job_id)
    _ensure_worker()
    _worker_wake.set()
    _queue_positions()
    return rec


# =============================================================================
# Validation
# =============================================================================

def _validate(req: ComposeRequest) -> None:
    """Reject a spec that cannot be rendered, naming what is wrong and where."""
    if not req.scenes:
        raise HTTPException(status_code=400, detail="scenes: at least one scene is required")
    if len(req.scenes) > _MAX_SCENES:
        raise HTTPException(status_code=400,
                            detail=f"scenes: at most {_MAX_SCENES} scenes per request")
    c = req.canvas
    if not (16 <= c.width <= 7680 and 16 <= c.height <= 7680):
        raise HTTPException(status_code=400, detail="canvas: width/height out of range")
    if c.width % 2 or c.height % 2:
        raise HTTPException(status_code=400,
                            detail="canvas: width and height must be even (H.264)")
    if not (1 <= c.fps <= 120):
        raise HTTPException(status_code=400, detail="canvas.fps: must be 1..120")
    if (req.visual_fit or "cover").lower() not in ("cover", "contain", "blur_pad"):
        raise HTTPException(status_code=400,
                            detail="visual_fit: expected cover, contain or blur_pad")
    ttype = str((req.transition or {}).get("type") or "none").lower()
    if ttype not in ("none", "fade", "crossfade", "slide", "wipe", "dissolve"):
        raise HTTPException(status_code=400, detail=(
            "transition.type: expected none, fade, crossfade, slide, wipe or dissolve"))
    for i, sc in enumerate(req.scenes):
        has_narration = bool((sc.text or "").strip()) or bool(sc.audio)
        if not has_narration and not (sc.duration and sc.duration > 0):
            raise HTTPException(status_code=400, detail=(
                f"scenes[{i}]: a scene with no text and no audio needs an explicit "
                f"\"duration\""))
        if not sc.visuals:
            raise HTTPException(status_code=400,
                                detail=f"scenes[{i}].visuals: at least one visual is required")
        if len(sc.visuals) > _MAX_VISUALS_PER_SCENE:
            raise HTTPException(status_code=400, detail=(
                f"scenes[{i}].visuals: at most {_MAX_VISUALS_PER_SCENE} per scene"))
        for j, v in enumerate(sc.visuals):
            vt = (v.type or "").lower()
            if vt not in ("video", "image", "gradient", "color"):
                raise HTTPException(status_code=400, detail=(
                    f"scenes[{i}].visuals[{j}].type: expected video, image, gradient "
                    f"or color, got {v.type!r}"))
            if vt in ("video", "image") and not (v.src or "").strip():
                raise HTTPException(status_code=400, detail=(
                    f"scenes[{i}].visuals[{j}].src: required for type {vt!r}"))
    caps = req.captions
    if caps and caps.enabled and (caps.timing or "auto").lower() not in ("auto", "stt", "estimate"):
        raise HTTPException(status_code=400,
                            detail="captions.timing: expected auto, stt or estimate")
    if caps and caps.enabled and (caps.preset or "karaoke").lower() not in cap.PRESETS:
        raise HTTPException(status_code=400, detail=(
            "captions.preset: expected one of " + ", ".join(sorted(cap.PRESETS))))
    unknown = [o for o in req.outputs
               if o not in ("video", "thumbnail", "srt", "vtt", "narration")]
    if unknown:
        raise HTTPException(status_code=400, detail=(
            f"outputs: unknown artefact(s) {unknown}; expected video, thumbnail, "
            f"srt, vtt, narration"))
    if (req.output.format or "mp4").lower() not in ("mp4", "mov", "webm", "mkv"):
        raise HTTPException(status_code=400,
                            detail="output.format: expected mp4, mov, webm or mkv")
    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=501, detail=(
            "composition needs ffmpeg on PATH and this install has none "
            "(the published images ship it)"))
    # Sub-features that need a model: answer 501 now rather than failing a job
    # minutes later.
    if req.music and req.music.generate and not _has_model("audio_gen"):
        raise HTTPException(status_code=501, detail=(
            "music.generate needs an audio-generation model (e.g. musicgen) configured "
            "on this install; supply music.src instead"))
    if caps and caps.enabled and (caps.timing or "auto").lower() == "stt" \
            and not _has_model("audio", caps.stt_model):
        raise HTTPException(status_code=501, detail=(
            f"captions.timing=stt needs a speech-to-text model"
            f"{' called ' + repr(caps.stt_model) if caps.stt_model else ''} on this "
            f"install; use \"estimate\" or \"auto\""))
    if _voice_needed(req) and not _voice_available(req):
        raise HTTPException(status_code=501, detail=(
            "narration needs a TTS model (engine=tts) or a saved voice profile "
            "(engine=clone) on this install; pass per-scene \"audio\" instead"))


def _model_ids(mtype: str) -> List[str]:
    try:
        from codai.models.manager import multi_model_manager
        return [m.id for m in multi_model_manager.list_models()
                if (m.type or "") == mtype]
    except Exception:
        return []


def _has_model(mtype: str, name: str = "") -> bool:
    ids = _model_ids(mtype)
    if not ids:
        return False
    if not name:
        return True
    tail = name.rstrip("/").split("/")[-1].lower()
    return any(name.lower() == i.lower() or i.rstrip("/").split("/")[-1].lower() == tail
               for i in ids)


def _voice_needed(req: ComposeRequest) -> bool:
    return any((sc.text or "").strip() and not sc.audio for sc in req.scenes)


def _voice_available(req: ComposeRequest) -> bool:
    engines = {(sc.voice or req.voice).engine or "tts" for sc in req.scenes}
    if "clone" in engines:
        try:
            from codai.api.voice_clone import _load_voice
            names = {(sc.voice or req.voice).voice_name for sc in req.scenes
                     if ((sc.voice or req.voice).engine or "tts") == "clone"}
            if not all(n and _load_voice(n) for n in names):
                return False
        except Exception:
            return False
    if engines - {"clone"}:
        return _has_model("tts", req.voice.model or "")
    return True


# =============================================================================
# Pieces: narration, music, word timings
# =============================================================================

#: Narration, music and speech recognition are async endpoints, and a render
#: runs in a worker thread. It must NOT borrow the event loop of the request
#: that queued the job: that loop belongs to one request and may be closed long
#: before the render reaches its narration (which is exactly what happens under
#: a test client, and under any server that finishes a request early). So this
#: module owns one long-lived loop in a thread of its own.
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()


def _media_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is not None and not _loop.is_closed():
            return _loop
        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, name="compose-loop",
                         daemon=True).start()
        return _loop


def _await(loop, coro, timeout: float = 1800.0):
    """Run one of the API's coroutines from the render thread."""
    fut = asyncio.run_coroutine_threadsafe(coro, loop or _media_loop())
    return fut.result(timeout=timeout)


def _b64_to_file(b64: str, path: str) -> str:
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    return path


def _synth_scene(loop, text: str, voice: VoiceSpec, out_path: str) -> str:
    """One scene's narration as a wav file, through the ordinary endpoints."""
    engine = (voice.engine or "tts").lower()
    if engine == "clone":
        from codai.api.voice_clone import VoiceCloneRequest, clone_voice
        req = VoiceCloneRequest(text=text, voice_name=voice.voice_name,
                                speed=voice.speed or 1.0, language=voice.language)
        res = _await(loop, clone_voice(req, None))
        # /v1/audio/clone answers {created, data: [{url}|{b64_wav}], engine}: a
        # url when this install has an output directory, inline audio otherwise.
        item = ((res or {}).get("data") or [{}])[0]
        blob = item.get("b64_wav") or (res or {}).get("audio")
        if blob:
            return _b64_to_file(blob, out_path)
        url = item.get("url") or (res or {}).get("url")
        if not url:
            raise rnd.RenderError("voice cloning returned no audio")
        local = media.local_path_for_files_url(url)
        if local:
            return local
        with open(out_path, "wb") as f:
            f.write(media.fetch(url))
        return out_path
    from codai.api.tts import TTSRequest, create_speech
    req = TTSRequest(model=voice.model or "", input=text,
                     voice=voice.voice or "af_sarah", response_format="wav",
                     speed=voice.speed or 1.0)
    if voice.language:
        setattr(req, "language", voice.language)
    if voice.voice_name:
        req.voice_profile = voice.voice_name
    res = _await(loop, create_speech(req, None))
    blob = (res or {}).get("audio")
    if not blob:
        raise rnd.RenderError("TTS returned no audio")
    return _b64_to_file(blob, out_path)


def _generate_music(loop, spec: dict, total: float, out_path: str) -> str:
    from codai.api.audio_gen import AudioGenerationRequest, audio_generate
    req = AudioGenerationRequest(
        model=str(spec.get("model") or ""),
        prompt=str(spec.get("prompt") or "calm background music"),
        duration=float(spec.get("duration") or max(10.0, min(total, 120.0))),
        response_format="b64_wav")
    res = _await(loop, audio_generate(req, None))
    data = (res or {}).get("data") or []
    if not data:
        raise rnd.RenderError("music generation returned nothing")
    item = data[0]
    blob = item.get("b64_wav") or item.get("b64_json")
    if blob:
        return _b64_to_file(blob, out_path)
    url = item.get("url")
    if not url:
        raise rnd.RenderError("music generation returned no audio")
    with open(out_path, "wb") as f:
        f.write(media.fetch(url))
    return out_path


def _stt_words(loop, audio_path: str, model: str, language: Optional[str]) -> List[dict]:
    """Word timestamps for a narration clip (only the timings are used)."""
    from codai.api.transcriptions import _run_transcription
    with open(audio_path, "rb") as f:
        content = f.read()
    res = _await(loop, _run_transcription(
        content, model, language, None, "verbose_json", 0.0,
        os.path.basename(audio_path), word_timestamps=True))
    if hasattr(res, "body"):        # a PlainTextResponse: no timings in it
        return []
    return list((res or {}).get("words") or [])


def _pick_stt_model(caps: CaptionSpec) -> Optional[str]:
    if caps.stt_model and _has_model("audio", caps.stt_model):
        return caps.stt_model
    ids = _model_ids("audio")
    if not ids:
        return None
    # Prefer something whisper-shaped: it is what reports word timings.
    for want in ("whisper", "canary", "wav2vec"):
        for i in ids:
            if want in i.lower():
                return i
    return ids[0]


# =============================================================================
# The render
# =============================================================================

def _run_job(job_id: str, req: ComposeRequest, loop, base_url: str) -> None:
    cancel = _cancels.get(job_id) or threading.Event()
    work = tempfile.mkdtemp(prefix=f"coderai-compose-{job_id}-")
    canvas = req.canvas
    fit = (req.visual_fit or "cover").lower()
    ttype = str((req.transition or {}).get("type") or "none").lower()
    tdur = float((req.transition or {}).get("duration") or 0.3)
    caps = req.captions
    log_lines: List[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        if len(log_lines) > 200:
            del log_lines[:100]

    def stage(name: str, progress: float, message: str) -> None:
        if cancel.is_set():
            raise rnd.Cancelled()
        _update(job_id, stage=name, progress=progress, message=message[:300],
                status="running")

    def warn(msg: str) -> None:
        _update(job_id, warning=msg)
        print(f"[compose {job_id}] WARNING {msg}", flush=True)

    try:
        _update(job_id, status="running", stage="narration", progress=1.0,
                message="starting")

        # ---------------------------------------------------- 1. narration
        n_scenes = len(req.scenes)
        scene_audio: List[Optional[str]] = []
        scene_dur: List[float] = []
        for i, sc in enumerate(req.scenes):
            stage("narration", 2.0 + 26.0 * i / max(1, n_scenes),
                  f"scene {i + 1}/{n_scenes}: narration")
            apath = None
            if sc.audio:
                try:
                    apath = media.resolve(sc.audio, work, f"scene{i}-audio")
                except media.MediaError as exc:
                    warn(f"scene {i}: audio could not be fetched ({exc}); "
                         f"synthesising from text instead")
            if apath is None and (sc.text or "").strip():
                apath = _synth_scene(loop, sc.text, sc.voice or req.voice,
                                     os.path.join(work, f"scene{i}-voice.wav"))
            dur = 0.0
            if apath:
                try:
                    dur = media.duration_of(apath)
                except media.MediaError as exc:
                    warn(f"scene {i}: narration could not be probed ({exc})")
                    apath, dur = None, 0.0
            if apath:
                dur = dur + max(0.0, float(sc.padding or 0.0))
            else:
                dur = float(sc.duration or 0.0)
            dur = max(dur, float(sc.min_duration or 0.0))
            if dur <= 0:
                dur = 3.0
                warn(f"scene {i}: no narration and no duration; used 3.0s")
            scene_audio.append(apath)
            scene_dur.append(round(dur, 3))

        total = round(sum(scene_dur), 3)
        starts, t = [], 0.0
        for d in scene_dur:
            starts.append(round(t, 3))
            t += d

        # ---------------------------------------------------- 2. captions
        events: List[dict] = []
        if caps and caps.enabled:
            stage("captions", 30.0, "timing the words")
            timing = (caps.timing or "auto").lower()
            model = _pick_stt_model(caps) if timing in ("auto", "stt") else None
            if timing == "stt" and not model:
                raise rnd.RenderError("captions.timing=stt but no speech model is configured")
            if timing == "auto" and not model:
                warn("no speech model configured; caption timings are estimated")
            words: List[dict] = []
            for i, sc in enumerate(req.scenes):
                text = (sc.text or "").strip()
                if not text:
                    continue
                stt: List[dict] = []
                if model and scene_audio[i]:
                    try:
                        stt = _stt_words(loop, scene_audio[i], model,
                                         (sc.voice or req.voice).language)
                    except Exception as exc:
                        warn(f"scene {i}: word timestamps failed ({exc}); estimated instead")
                span = max(0.2, scene_dur[i] - max(0.0, float(sc.padding or 0.0)))
                words += (cap.align_words(text, stt, starts[i], span) if stt
                          else cap.estimate_words(text, starts[i], span))
            events = cap.break_lines(words, caps.max_words_per_line, caps.max_lines)
            if caps.font and not media.font_installed(caps.font):
                warn(f"font {caps.font!r} is not installed; captions use the "
                     f"closest available family")

        # ---------------------------------------------------- 3. visuals
        clips: List[str] = []
        clip_durs: List[float] = []
        made = 0
        total_visuals = sum(len(sc.visuals) for sc in req.scenes)
        extra = tdur if ttype != "none" else 0.0
        for i, sc in enumerate(req.scenes):
            n = max(1, len(sc.visuals))
            slot = scene_dur[i] / n
            for j, v in enumerate(sc.visuals):
                stage("visuals", 40.0 + 28.0 * made / max(1, total_visuals),
                      f"scene {i + 1}/{n_scenes}: visual {j + 1}/{n}")
                spec = v.model_dump()
                vt = (spec.get("type") or "image").lower()
                if vt in ("video", "image"):
                    try:
                        spec["_path"] = media.resolve(spec.get("src") or "", work,
                                                      f"s{i}v{j}")
                        spec["_probe"] = media.probe(spec["_path"])
                        if vt == "video" and not spec["_probe"].get("has_video"):
                            raise media.MediaError("no video stream")
                    except (media.MediaError, OSError) as exc:
                        warn(f"scene {i} visual {j}: {exc}; used gradient")
                        spec = {"type": "gradient", "colors": None}
                out = os.path.join(work, f"clip-{i}-{j}.mp4")
                try:
                    rnd.build_visual(spec, out, width=canvas.width, height=canvas.height,
                                     fps=canvas.fps, duration=slot + extra, fit=fit,
                                     index=made, workdir=work, cancel=cancel, log=log)
                except rnd.Cancelled:
                    raise
                except Exception as exc:
                    warn(f"scene {i} visual {j}: render failed ({exc}); used gradient")
                    rnd.build_visual({"type": "gradient"}, out, width=canvas.width,
                                     height=canvas.height, fps=canvas.fps,
                                     duration=slot + extra, fit=fit, index=made,
                                     workdir=work, cancel=cancel, log=log)
                clips.append(out)
                clip_durs.append(slot + extra)
                made += 1

        stage("visuals", 69.0, "joining the clips")
        silent = os.path.join(work, "video.mp4")
        if len(clips) == 1:
            silent = clips[0]
        elif ttype == "none":
            rnd.concat_clips(clips, silent, work, cancel=cancel, log=log)
        else:
            try:
                rnd.xfade_clips(clips, clip_durs, silent, kind=ttype, dur=tdur,
                                fps=canvas.fps, cancel=cancel, log=log)
            except rnd.Cancelled:
                raise
            except Exception as exc:
                warn(f"transition {ttype!r} failed ({exc}); joined without one")
                rnd.concat_clips(clips, silent, work, cancel=cancel, log=log)

        # ---------------------------------------------------- 4. audio
        stage("music", 71.0, "building the soundtrack")
        parts = []
        for i, sc in enumerate(req.scenes):
            p = os.path.join(work, f"narr-{i}.wav")
            if scene_audio[i]:
                rnd.pad_to(scene_audio[i], p, scene_dur[i], cancel=cancel, log=log)
            else:
                rnd.silence(p, scene_dur[i], cancel=cancel, log=log)
            parts.append(p)
        narration = rnd.concat_audio(parts, os.path.join(work, "narration.wav"), work,
                                    cancel=cancel, log=log)
        if any(scene_audio):
            try:
                narration = rnd.loudnorm(narration, os.path.join(work, "narration-ln.wav"),
                                         cancel=cancel, log=log)
            except rnd.Cancelled:
                raise
            except Exception as exc:
                warn(f"loudness normalisation failed ({exc}); narration used as synthesised")

        music_bed = None
        if req.music:
            src = None
            if req.music.src:
                try:
                    src = media.resolve(req.music.src, work, "music")
                except media.MediaError as exc:
                    warn(f"music: {exc}; rendered without music")
            elif req.music.generate:
                stage("music", 73.0, "generating music")
                try:
                    src = _generate_music(loop, req.music.generate, total,
                                          os.path.join(work, "music-gen.wav"))
                except Exception as exc:
                    warn(f"music generation failed ({exc}); rendered without music")
            if src:
                try:
                    music_bed = rnd.build_music_bed(
                        src, os.path.join(work, "music.wav"), total=total,
                        volume=req.music.volume, loop=req.music.loop,
                        fade_in=req.music.fade_in, fade_out=req.music.fade_out,
                        cancel=cancel, log=log)
                except rnd.Cancelled:
                    raise
                except Exception as exc:
                    warn(f"music could not be prepared ({exc}); rendered without music")
        mixed = rnd.mix_narration_music(narration, music_bed,
                                        os.path.join(work, "audio.wav"),
                                        duck=bool(req.music and req.music.duck),
                                        cancel=cancel, log=log)

        # ---------------------------------------------------- 5. final render
        stage("render", 76.0, "encoding")
        ass_path = None
        if events:
            ass_path = os.path.join(work, "captions.ass")
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(cap.build_ass(
                    events, width=canvas.width, height=canvas.height,
                    font=caps.font, font_scale=caps.font_scale, colour=caps.color,
                    highlight=caps.highlight_color, outline_colour=caps.outline_color,
                    position=caps.position, preset=caps.preset,
                    uppercase=caps.uppercase))
        overlays = []
        for k, o in enumerate(req.overlays or []):
            spec = o.model_dump()
            if (spec.get("type") or "text").lower() == "image":
                try:
                    spec["_path"] = media.resolve(spec.get("src") or "", work, f"ov{k}")
                except media.MediaError as exc:
                    warn(f"overlay {k}: {exc}; skipped")
                    continue
            overlays.append(spec)

        fmt = (req.output.format or "mp4").lower()
        final = os.path.join(work, f"{job_id}.{fmt}")
        rnd.final_render(silent, mixed, final, width=canvas.width, height=canvas.height,
                         fps=canvas.fps, total=total, ass_file=ass_path,
                         overlays=overlays, crf=req.output.crf,
                         audio_bitrate=req.output.audio_bitrate,
                         video_codec=req.output.video_codec, cancel=cancel, log=log)

        # ---------------------------------------------------- 6. artefacts
        stage("thumbnail", 95.0, "artefacts")
        want = set(req.outputs or ["video"])
        out_dir = _output_dir()
        result: Dict[str, Any] = {}

        def publish(src: str, name: str) -> dict:
            dst = os.path.join(out_dir, name)
            shutil.copyfile(src, dst)
            return {"url": _file_url(name, base_url), "path": f"/v1/files/{name}",
                    "size": os.path.getsize(dst)}

        vid = publish(final, f"{job_id}.{fmt}")
        info = media.probe(final)
        vid.update({"duration": round(float(info.get("duration") or total), 3),
                    "width": info.get("width") or canvas.width,
                    "height": info.get("height") or canvas.height})
        result["video"] = vid

        if "thumbnail" in want:
            th = req.thumbnail or ThumbnailSpec()
            at = max(0.0, min(0.999, float(th.at if th.at is not None else 0.25))) * total
            tpath = os.path.join(work, f"{job_id}.jpg")
            try:
                rnd.thumbnail(final, tpath, at=at, width=canvas.width,
                              height=canvas.height, text=th.text or "",
                              text_color=th.text_color, highlight_color=th.highlight_color,
                              font=th.font or "", cancel=cancel, log=log)
                result["thumbnail"] = publish(tpath, f"{job_id}.jpg")
            except rnd.Cancelled:
                raise
            except Exception as exc:
                warn(f"thumbnail failed ({exc})")
        if events and ("srt" in want or "vtt" in want):
            paths = cap.write_files(events, work, job_id)
            for kind in ("srt", "vtt"):
                if kind in want:
                    result[kind] = publish(paths[kind], f"{job_id}.{kind}")
        if "narration" in want:
            result["narration"] = publish(narration, f"{job_id}_voice.wav")
        result["scenes"] = [{"index": i, "start": starts[i],
                             "end": round(starts[i] + scene_dur[i], 3)}
                            for i in range(n_scenes)]
        result["duration"] = total

        _update(job_id, status="done", stage="thumbnail", progress=100.0,
                message="done", result=result)
        print(f"[compose {job_id}] done: {total:.1f}s, {n_scenes} scenes, "
              f"{len(clips)} visuals", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _output_dir() -> str:
    d = media.files_dir()
    if not d:
        raise rnd.RenderError("this install has no output directory configured "
                              "(--file-path), so composed video cannot be served")
    os.makedirs(d, exist_ok=True)
    return d


def _file_url(name: str, base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    return f"{base}/v1/files/{name}" if base else f"/v1/files/{name}"


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/v1/video/compose", summary="Compose a finished video from scenes",
             tags=["Video"])
async def compose_video(req: ComposeRequest, http_request: Request,
                        _auth=Depends(_require_api_auth)):
    """Assemble narration, visuals, music and captions into one uploadable video.

    Each scene's length comes from its narration (synthesised here unless the
    scene carries ``audio``) plus its ``padding``, and its visuals share that
    time. With ``"async": true`` (the default) the call returns
    ``202 {id, status}`` and the client polls ``GET /v1/video/compose/{id}``;
    with ``"async": false`` it blocks and returns the finished document, which is
    convenient for tests and a bad idea for a real render.

    A visual that cannot be fetched or decoded is replaced by a gradient and
    reported in ``warnings`` — one bad stock clip does not lose a five-minute
    render."""
    _validate(req)
    try:
        from codai.api.urlutils import get_base_url
        base_url = get_base_url(http_request)
    except Exception:
        base_url = ""
    rec = submit(req, base_url)
    if req.is_async:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=202,
                            content={"id": rec["id"], "status": "queued"})
    # Blocking: wait for the worker to finish this job.
    job_id = rec["id"]
    while True:
        await asyncio.sleep(0.5)
        cur = _get(job_id)
        if cur is None:
            raise HTTPException(status_code=500, detail="job disappeared")
        if cur["status"] in ("done", "failed", "cancelled"):
            if cur["status"] == "failed":
                raise HTTPException(status_code=500,
                                    detail=cur.get("error") or "composition failed")
            return _public(cur)


@router.get("/v1/video/compose/{job_id}", summary="Composition job status", tags=["Video"])
async def compose_status(job_id: str, _auth=Depends(_require_api_auth)):
    """Progress, warnings and — when ``status`` is ``done`` — the artefact URLs."""
    rec = _get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"unknown composition job {job_id!r}")
    return _public(rec)


@router.post("/v1/video/compose/{job_id}/cancel", summary="Cancel a composition job",
             tags=["Video"])
async def compose_cancel(job_id: str, _auth=Depends(_require_api_auth)):
    """Stop a queued or running job (the ffmpeg child is killed, not left to finish)."""
    if not cancel_job(job_id):
        raise HTTPException(status_code=404, detail=f"unknown composition job {job_id!r}")
    return {"id": job_id, "status": "cancelled"}


@router.get("/v1/video/compose", summary="List composition jobs", tags=["Video"])
async def compose_list(_auth=Depends(_require_api_auth)):
    """Every job this front still remembers, newest first — for an admin page or
    a client that lost a job id."""
    with _jobs_lock:
        recs = sorted(_jobs.values(), key=lambda r: r.get("created_at") or 0, reverse=True)
    return {"jobs": [_public(r) for r in recs]}
