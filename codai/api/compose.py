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
#: Lines in one dialogue scene. A scene is a beat, not an act; a longer
#: exchange is several scenes, which is also how it gets its own visuals.
_MAX_SPEAKERS_PER_SCENE = 20
_JOB_HISTORY = 100
_STAGES = ("queued", "narration", "captions", "presenter", "visuals", "music",
           "render", "thumbnail")


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
    # Keep THIS clip's own sound. Every visual is rendered silent (ffmpeg -an),
    # which is right for a stock shot under a voice-over and wrong for footage
    # whose sound IS the content -- a crowd, a demo, a piece to camera that was
    # filmed with its audio. Ignored for image/gradient/colour visuals, which
    # have none, and for a visual that is not the one carrying the scene (only
    # the first audible video in a scene is taken; see _scene_clip_audio).
    keep_audio: bool = False
    # How loud this clip sits when it is mixed UNDER the narration
    # (``audio_mix: "both"``). A per-visual override of the scene's
    # ``clip_volume``.
    volume: Optional[float] = None
    model_config = ConfigDict(extra="allow")


class SpeakerLine(BaseModel):
    """One character's turn in a scene's dialogue.

    A scene has always had one voice and one presenter, which is a monologue.
    Two characters talking needs three things a per-scene override cannot
    give: an ORDER (who speaks when), a voice per turn, and a place in the
    frame for each of them to be while the other one talks.
    """
    character: Optional[str] = None       # a saved character profile
    image: Optional[str] = None           # or an explicit portrait
    text: Optional[str] = None            # what they say
    audio: Optional[str] = None           # or ready-made audio for this line
    voice: Optional[VoiceSpec] = None     # this character's voice
    # Where this character is while the scene runs: "left"/"right" (a column
    # of a side-by-side frame, held for the whole scene) or "full" (the frame
    # cuts to whoever is speaking). Mixing the two in one scene is refused:
    # a character cannot both hold a column and not be on screen.
    layout: Optional[str] = None
    # Silence after this line, so a reply does not tread on the question.
    gap: float = 0.25
    model_config = ConfigDict(extra="allow")


class Scene(BaseModel):
    text: Optional[str] = None
    audio: Optional[str] = None
    duration: Optional[float] = None        # required when there is no narration
    min_duration: float = 0.0
    padding: float = 0.15
    visuals: List[Visual] = Field(default_factory=list)
    voice: Optional[VoiceSpec] = None       # per-scene override (a second speaker)
    # Two or more characters talking, in order. Replaces `text`/`voice`/
    # `presenter` for this scene: the lines ARE the narration, each in its own
    # character's voice, and each character is lip-synced in its own region of
    # the frame.
    speakers: List[SpeakerLine] = Field(default_factory=list)
    # What this scene SOUNDS like:
    #   "voice" (default) — the narration only, as before.
    #   "clip"            — the clip's own sound only, no narration. The scene
    #                       then lasts as long as the clip, because the clip is
    #                       the content and cutting it mid-sentence is never
    #                       what was meant. An explicit `duration` still wins.
    #   "both"            — narration over the clip, the clip ducked under it.
    # Only "clip"/"both" need `visual.keep_audio`; "voice" ignores it.
    audio_mix: str = "voice"
    # The clip's level under the voice in "both", 0-1. A per-visual `volume`
    # overrides it. Not used by "clip", where the clip is the whole mix and is
    # loudness-normalised like narration.
    clip_volume: float = 0.35
    # A presenter block for this scene, or false to opt out of the request's one.
    presenter: Optional[Any] = None
    model_config = ConfigDict(extra="allow")


class PresenterSpec(BaseModel):
    """A character talking to camera, lip-synced to the scene's narration."""
    character: Optional[str] = None          # a saved character profile
    image: Optional[str] = None              # or an explicit portrait
    engine: str = "auto"                     # auto | sadtalker | wav2lip | <installed name>
    motion: str = "generate"                 # generate | still | src
    base_src: Optional[str] = None           # clip to drive (motion=src)
    prompt: Optional[str] = None             # for motion=generate
    video_model: Optional[str] = None        # model for motion=generate (blank = default)
    layout: str = "full"                     # full | pip | split
    position: str = "bottom-right"           # pip
    size: float = 0.38                       # pip: width as a fraction of the canvas
    shape: str = "circle"                    # pip: circle | rounded | rect
    split: str = "bottom"                    # split: which half the presenter takes
    remove_background: bool = False
    border: Optional[Dict[str, Any]] = None  # pip: {"color": …, "width": …}
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
    presenter: Optional[PresenterSpec] = None
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
    """Tell every waiting job where it is in the line (clients show this)."""
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
        # A dialogue's lines ARE the scene's narration and its presenter, so a
        # dialogue scene needs neither `text` nor visuals. Validated here, or a
        # perfectly renderable scene would be refused for having neither.
        if sc.speakers:
            try:
                _speaker_layout(sc)
            except ValueError as exc:
                raise HTTPException(status_code=400,
                                    detail=f"scenes[{i}].speakers: {exc}")
            if len(sc.speakers) > _MAX_SPEAKERS_PER_SCENE:
                raise HTTPException(status_code=400, detail=(
                    f"scenes[{i}].speakers: at most {_MAX_SPEAKERS_PER_SCENE} "
                    f"lines per scene"))
            for k, sp in enumerate(sc.speakers):
                if not ((sp.text or "").strip() or sp.audio):
                    raise HTTPException(status_code=400, detail=(
                        f"scenes[{i}].speakers[{k}]: a line needs `text` or "
                        f"`audio`"))
                if not (sp.character or sp.image
                        or (req.presenter and (req.presenter.character
                                               or req.presenter.image))):
                    raise HTTPException(status_code=400, detail=(
                        f"scenes[{i}].speakers[{k}]: a speaker needs a "
                        f"`character` or an `image` to be lip-synced (or a "
                        f"request-level presenter to inherit one from)"))
                # The same check the presenter gets: a profile name that does
                # not exist is a typo, and finding out mid-render costs the
                # caller the narration they have already paid to synthesise.
                if sp.character:
                    _validate_character_name(sp.character,
                                             f"scenes[{i}].speakers[{k}]")
        has_narration = (bool((sc.text or "").strip()) or bool(sc.audio)
                         or bool(sc.speakers))
        if not has_narration and not (sc.duration and sc.duration > 0):
            raise HTTPException(status_code=400, detail=(
                f"scenes[{i}]: a scene with no text and no audio needs an explicit "
                f"\"duration\""))
        pres = presenter_for(sc, req)
        if pres is not None:
            _validate_presenter(pres, f"scenes[{i}].presenter" if sc.presenter else "presenter")
        if (not sc.visuals and not sc.speakers
                and not (pres and (pres.get("layout") or "full").lower() == "full")):
            raise HTTPException(status_code=400, detail=(
                f"scenes[{i}].visuals: at least one visual is required "
                f"(a scene with no visuals needs a full-screen presenter)"))
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


def presenter_for(scene: Scene, req: ComposeRequest) -> Optional[dict]:
    """This scene's presenter block, or None.

    A scene may override the request's presenter, or opt out with
    ``"presenter": false`` — the sign-off is full screen, the middle is B-roll."""
    own = scene.presenter
    if own is False or (isinstance(own, str) and own.strip().lower() in ("false", "none", "off")):
        return None
    base = req.presenter.model_dump() if req.presenter else None
    if own is None or own is True:
        return dict(base) if base else None
    if isinstance(own, dict):
        merged = dict(base or {})
        merged.update({k: v for k, v in own.items() if v is not None})
        # A scene-level block with neither a character nor an image inherits the
        # request's face rather than failing.
        if not (merged.get("character") or merged.get("image")):
            return None
        return merged
    return dict(base) if base else None


def _validate_character_name(char: str, where: str) -> None:
    """A named profile must exist, or the render fails after paying for TTS.

    Its own function because a dialogue's speakers need exactly this check and
    nothing else from presenter validation -- calling the whole validator for a
    speaker rejected it for having no `layout`, which a speaker has no business
    carrying.
    """
    char = (char or "").strip()
    if not char:
        return
    try:
        from codai.api.characters import _load_character_meta
        known = bool(_load_character_meta(char))
    except Exception:
        known = True          # cannot check → let the render warn instead
    if not known:
        raise HTTPException(
            status_code=400,
            detail=f"{where}.character: no character profile named {char!r}")


def _validate_presenter(spec: dict, where: str) -> None:
    layout = (spec.get("layout") or "full").lower()
    if layout not in ("full", "pip", "split"):
        raise HTTPException(status_code=400, detail=(
            f"{where}.layout: expected full, pip or split (got {spec.get('layout')!r})"))
    if layout == "pip":
        pos = (spec.get("position") or "bottom-right").lower().replace("_", "-")
        if pos not in ("top-left", "top-right", "bottom-left", "bottom-right",
                       "bottom-center", "bottom-centre", "top-center", "top-centre",
                       "center", "centre"):
            raise HTTPException(status_code=400, detail=(
                f"{where}.position: expected one of top-left, top-right, bottom-left, "
                f"bottom-right, bottom-center (got {spec.get('position')!r})"))
        try:
            size = float(spec.get("size") if spec.get("size") is not None else 0.38)
        except (TypeError, ValueError):
            size = -1.0
        if not (0.1 <= size <= 0.95):
            raise HTTPException(status_code=400,
                                detail=f"{where}.size: expected 0.1..0.95 (got {spec.get('size')!r})")
        if (spec.get("shape") or "circle").lower() not in ("circle", "rounded", "round", "rect"):
            raise HTTPException(status_code=400, detail=(
                f"{where}.shape: expected circle, rounded or rect (got {spec.get('shape')!r})"))
    if layout == "split" and (spec.get("split") or "bottom").lower() not in ("top", "bottom"):
        raise HTTPException(status_code=400,
                            detail=f"{where}.split: expected top or bottom")
    if (spec.get("motion") or "generate").lower() not in ("generate", "still", "src"):
        raise HTTPException(status_code=400, detail=(
            f"{where}.motion: expected generate, still or src (got {spec.get('motion')!r})"))
    if (spec.get("motion") or "generate").lower() == "src" and not spec.get("base_src"):
        raise HTTPException(status_code=400,
                            detail=f'{where}.base_src: required when motion is "src"')
    char = (spec.get("character") or "").strip()
    if not char and not (spec.get("image") or "").strip():
        raise HTTPException(status_code=400,
                            detail=f"{where}: needs a character or an image")
    if char:
        _validate_character_name(char, where)
    from codai.compose import presenter as _pres
    want = (spec.get("engine") or "auto").strip()
    if not _pres.engine_installed(want):
        have = _pres.available_engines()
        raise HTTPException(status_code=501, detail=(
            f"{where}: lip-sync engine "
            + (f"{want!r} is not installed" if want.lower() != "auto"
               else "is not installed")
            + (f" (installed: {', '.join(have)})" if have else
               "; install the wav2lip or sadtalker shim (the published images ship both), "
               "or drop the presenter block")))


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


def _voice_specs(req: ComposeRequest) -> list:
    """Every voice this render will actually use, resolved.

    A dialogue's lines each carry their own voice, so reading only
    ``scene.voice`` misses them -- which meant a dialogue-only request passed
    the "is TTS configured" check and then failed at the first line, after the
    job had been accepted.
    """
    out = []
    for sc in req.scenes:
        if getattr(sc, "speakers", None):
            for sp in sc.speakers:
                if (sp.text or "").strip() and not sp.audio:
                    out.append(sp.voice or sc.voice or req.voice)
        if (sc.text or "").strip() and not sc.audio:
            out.append(sc.voice or req.voice)
    return out


def _voice_needed(req: ComposeRequest) -> bool:
    return bool(_voice_specs(req))


def _voice_available(req: ComposeRequest) -> bool:
    specs = _voice_specs(req) or [req.voice]
    engines = {(v.engine or "tts") for v in specs}
    if "clone" in engines:
        try:
            from codai.api.voice_clone import _load_voice
            names = {v.voice_name for v in specs if (v.engine or "tts") == "clone"}
            if not all(n and _load_voice(n) for n in names):
                return False
        except Exception:
            return False
    if engines - {"clone"}:
        # Every distinct TTS model a line asks for has to be installed, not
        # just the request-level default: a dialogue may give each character a
        # different one.
        models = {(v.model or req.voice.model or "") for v in specs
                  if (v.engine or "tts") != "clone"}
        return all(_has_model("tts", m) for m in (models or {""}))
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


class _Lease:
    """A scheduler lease held across a blocking GPU step, released on close().

    Lip-sync is the only GPU work in a composition; everything else is ffmpeg.
    Taking a lease from the central scheduler means it queues with ordinary model
    requests instead of fighting a generation for the card. A scheduler that
    refuses (or is not there) is not a reason to fail: the step just runs."""

    def __init__(self, loop, lease):
        self._loop, self._lease = loop, lease

    def close(self) -> None:
        if self._lease is None:
            return
        try:
            from codai.queue.manager import queue_manager
            _await(self._loop, queue_manager.release(self._lease), timeout=60)
        except Exception:
            pass
        self._lease = None


def _gpu_lease(loop, model_key: str) -> _Lease:
    try:
        from codai.queue.manager import queue_manager
        lease = _await(loop, queue_manager.acquire(
            f"compose-{model_key}-{uuid.uuid4().hex[:8]}", model_key), timeout=3600)
    except Exception:
        lease = None
    return _Lease(loop, lease)


def _matte_presenter(pres: dict, *, index: int, work: str, fps: int, warn, stage,
                     progress: float, scene_no: str, cancel, log, gpu_lease) -> None:
    """Cut this scene's presenter out of their background, or say why not.

    `remove_background` is documented as optional, so nothing here is fatal: a
    scene whose background cannot be removed still gets its presenter, filmed
    room and all, plus a warning naming the scene. The alpha it leaves behind is
    what the compositor uses in place of the bubble's shape mask."""
    layout = (pres.get("layout") or "full").lower()
    if layout not in ("pip", "split"):
        warn(f"scene {index}: presenter.remove_background needs the pip or split "
             f"layout — a full-screen presenter has no B-roll to stand in front of")
        return
    try:
        from codai.compose import matting
    except Exception as exc:
        warn(f"scene {index}: background removal is unavailable here ({exc}); "
             f"the presenter is composited as filmed")
        return
    if not matting.available():
        warn(f"scene {index}: no matting engine (install torchvision, or point "
             f"CODERAI_MATTING_MODEL at a TorchScript matting model); the "
             f"presenter is composited as filmed")
        return
    stage("presenter", progress, f"scene {scene_no}: removing the background")
    try:
        pres["alpha"] = matting.matte(
            pres["path"], os.path.join(work, f"pres{index}-alpha.mp4"),
            fps=fps, cancel=cancel, log=log, gpu_lease=gpu_lease)
        pres["matted"] = matting.engine_name()
    except rnd.Cancelled:
        raise
    except Exception as exc:
        warn(f"scene {index}: the background could not be removed ({exc}); "
             f"the presenter is composited as filmed")


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
    # audio_generate answers with an AudioGenerationResponse -- a pydantic
    # model, not a dict -- so reading it with .get() raised
    # "'AudioGenerationResponse' object has no attribute 'get'" and every
    # compose that asked for AI music rendered silently without any (measured in
    # production 2026-10-28; the warning said "rendered without music" and the
    # render succeeded, so nothing failed loudly enough to notice). Both shapes
    # are accepted because the same call is reachable through paths that answer
    # a plain dict.
    data = getattr(res, "data", None)
    if data is None and isinstance(res, dict):
        data = res.get("data")
    data = data or []
    if not data:
        raise rnd.RenderError("music generation returned nothing")
    item = data[0]
    if not isinstance(item, dict) and hasattr(item, "model_dump"):
        item = item.model_dump()
    blob = item.get("b64_wav") or item.get("b64_json")
    if blob:
        return _b64_to_file(blob, out_path)
    url = item.get("url")
    if not url:
        raise rnd.RenderError("music generation returned no audio")
    with open(out_path, "wb") as f:
        f.write(media.fetch(url))
    return out_path


def _speaker_key(sp, idx: int) -> str:
    """What identifies a character across the lines they speak.

    Their profile name where they have one, else their portrait, else their
    position -- so a dialogue written with inline images still groups the same
    character's lines together instead of putting them in two columns.
    """
    return (str(getattr(sp, "character", None) or "").strip().lower()
            or str(getattr(sp, "image", None) or "").strip()
            or f"_speaker{idx}")


def _speaker_layout(sc) -> str:
    """``row`` or ``cut`` for a dialogue scene, validated.

    ``row`` keeps every character on screen in their own column for the whole
    scene -- the default, because a listener's reaction is half of a dialogue.
    ``cut`` gives the frame to whoever is speaking.
    """
    wants = {str(getattr(sp, "layout", None) or "").strip().lower()
             for sp in (getattr(sc, "speakers", None) or [])}
    wants.discard("")
    if not wants:
        return "row"
    if wants == {"full"}:
        return "cut"
    if wants <= {"left", "right", "centre", "center"}:
        return "row"
    raise ValueError(
        "a scene's speakers must all use a column layout (left/right) or all "
        f"use full; got {sorted(wants)}. A character cannot both hold a column "
        "and be off screen while the other speaks.")


def _speaker_order(sc) -> List[tuple]:
    """``[(key, speaker), ...]`` — the distinct characters, in first-speaking
    order. That order is the left-to-right order of the columns."""
    out, seen = [], set()
    for i, sp in enumerate(getattr(sc, "speakers", None) or []):
        k = _speaker_key(sp, i)
        if k not in seen:
            seen.add(k)
            out.append((k, sp))
    return out


def _scene_mix(sc) -> str:
    """The scene's audio_mix, validated down to one of the three it can be."""
    m = str(getattr(sc, "audio_mix", None) or "voice").strip().lower()
    return m if m in ("voice", "clip", "both") else "voice"


def _scene_clip_visual(sc):
    """The visual whose own sound this scene carries, or ``None``.

    The FIRST video visual that asks to keep its audio. A scene may show
    several clips in sequence, and summing all their soundtracks over one
    narration is a mess nobody asked for; the caller who wants a second clip's
    sound gives it its own scene, where it also gets its own duration. With
    ``audio_mix`` set but no visual marked, the first video visual is taken --
    asking for the clip's sound and not saying which clip can only mean the
    one clip.
    """
    vids = [v for v in (getattr(sc, "visuals", None) or [])
            if str(getattr(v, "type", "") or "").lower() == "video"]
    for v in vids:
        if bool(getattr(v, "keep_audio", False)):
            return v
    if _scene_mix(sc) != "voice" and len(vids) == 1:
        return vids[0]
    return None


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


def _stt_entry(model_id: str) -> dict:
    """An audio model's own models.json entry, without loading the model.

    Read from models.json rather than from the manager's per-model config map:
    that map holds RUNTIME kwargs, keyed by registration id, and a
    whisper-server model has no entry in it at all -- it lives in
    ``whisper_servers``. Asking it therefore answered "no backend" for exactly
    the models whose backend matters, which picked `whisper2` for a render that
    needed word timings (seen on a real render, 2026-10-26).

    Matched on id, alias, path or basename, because an audio model is referred
    to by any of them and the same alias may appear twice -- one entry naming
    ``path``, another ``model_path``. An entry that names a backend wins over
    one that does not.
    """
    want = str(model_id or "").strip().lower()
    if not want:
        return {}
    tail = want.rstrip("/").split("/")[-1]
    entries = []
    try:
        from codai.admin.routes import config_manager
        md = getattr(config_manager, "models_data", {}) or {}
        entries = [e for e in (md.get("audio_models") or []) if isinstance(e, dict)]
    except Exception:
        entries = []
    found = {}
    for e in entries:
        names = set()
        for key in ("id", "alias", "name", "path", "model_path"):
            val = str(e.get(key) or "").strip().lower()
            if val:
                names.add(val)
                names.add(val.rstrip("/").split("/")[-1])
        if want in names or tail in names:
            if e.get("backend") or not found:
                found = e
            if e.get("backend"):
                break
    if found:
        return found
    # No models.json in reach (a pod seeds its catalogue differently): fall back
    # to the manager's map, which carries the entry under `_raw_cfg` when it has
    # one.
    try:
        from codai.models.manager import multi_model_manager
        cfg = multi_model_manager._config_for_model(model_id) or {}
        raw = cfg.get("_raw_cfg") if isinstance(cfg, dict) else None
        return raw if isinstance(raw, dict) else (cfg if isinstance(cfg, dict) else {})
    except Exception:
        return {}


def _stt_backend(model_id: str) -> str:
    """The configured ``backend`` of an audio model, without loading it."""
    return str(_stt_entry(model_id).get("backend") or "").strip().lower()


def _stt_speaks(model_id: str, languages) -> bool:
    """Whether this model accepts every language the render narrates in.

    A model that declares no ``languages`` accepts anything. One that declares
    them REFUSES the rest with a 400 (see transcriptions._validate_language),
    and a per-language model is easy to pick by accident: an install with
    ``wav2vec2-it`` and ``wav2vec2-en`` has two models that each report word
    timings and each reject the other's language.
    """
    from codai.api.lang_codes import aliases
    declared = set()
    for x in (_stt_entry(model_id).get("languages") or []):
        declared |= aliases(x)
    if not declared:
        return True
    # By meaning, like the validator: a model declaring "english" does speak
    # the "en-gb" a British voice narrates in, and comparing the strings
    # literally disqualified every usable model as soon as a render asked for
    # an accent.
    return all(aliases(l) & declared for l in languages if l)


def _request_languages(req) -> list:
    """The languages the narration is in, in order of first appearance."""
    out = []
    for sc in getattr(req, "scenes", None) or []:
        lang = (getattr(sc, "voice", None) or req.voice).language
        lang = str(lang or "").strip().lower()
        if lang and lang not in out:
            out.append(lang)
    return out


def _pick_stt_model(caps: CaptionSpec, languages=()) -> Optional[str]:
    """An audio model to time the words with: one that CAN time words, first.

    The previous order preferred anything named "whisper", then canary. Both are
    the wrong way round here: a whisper-server (whisper.cpp) model returns text
    with no word array, and the NeMo worker behind canary ignores the request --
    so the obvious pick returned no timings at all and captions fell back to
    estimating from character counts, which drifts against the voice. The
    backends that do report words are the HF pipelines (wav2vec2, whisper_hf,
    CrisperWhisper), vosk, and plain faster-whisper.
    """
    if caps.stt_model and _has_model("audio", caps.stt_model):
        return caps.stt_model
    ids = _model_ids("audio")
    if not ids:
        return None
    import codai.api.stt_backends as stt_backends
    able = [i for i in ids if stt_backends.reports_word_timings(_stt_backend(i))]
    speaks = [i for i in able if _stt_speaks(i, languages)] or able
    # Among those, a whisper-family model reads running speech best; wav2vec2 is
    # per-language and vosk is a small CPU model, so they come after.
    for want in ("whisper", "wav2vec", "vosk"):
        for i in speaks:
            if want in i.lower():
                return i
    if speaks:
        return speaks[0]
    # Nothing configured can report words. Still return something: the timings
    # will be estimated, the renderer says so, and the transcript is unused.
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
        scene_mixes: List[str] = []
        # Per scene, for a dialogue: the layout and every line with its
        # character, offset and length. Read by the presenter pass (who is
        # lip-synced where) and by the captions pass (who is speaking).
        scene_lines: Dict[int, dict] = {}
        # Per scene: the visual whose own sound it keeps, already resolved and
        # probed. Resolved HERE, in the narration pass, because with
        # audio_mix="clip" the clip's length IS the scene's length and that has
        # to be known before anything is timed. Stage 3 reuses what this
        # resolved rather than fetching the same file twice.
        clip_media: Dict[int, dict] = {}
        for i, sc in enumerate(req.scenes):
            stage("narration", 2.0 + 26.0 * i / max(1, n_scenes),
                  f"scene {i + 1}/{n_scenes}: narration")
            mix = _scene_mix(sc)
            cv = _scene_clip_visual(sc)
            if cv is not None:
                try:
                    _cp = media.resolve(getattr(cv, "src", None) or "", work,
                                        f"s{i}-clipaudio")
                    if not rnd.has_audio_stream(_cp):
                        raise media.MediaError("the clip has no audio stream")
                    clip_media[i] = {"path": _cp, "probe": media.probe(_cp),
                                     "src": getattr(cv, "src", None) or "",
                                     "trim_start": max(0.0, float(
                                         getattr(cv, "trim_start", 0.0) or 0.0)),
                                     "volume": getattr(cv, "volume", None)}
                except (media.MediaError, OSError) as exc:
                    # Not fatal: the scene still has its narration, or silence.
                    # Said out loud, because a caller who asked to keep a
                    # clip's sound and silently got none would ship it.
                    warn(f"scene {i}: keep_audio requested but {exc}; "
                         f"rendered with {'narration only' if mix == 'both' else 'no sound'}")
                    mix = "voice" if mix == "both" else mix
            elif mix != "voice":
                warn(f"scene {i}: audio_mix={mix!r} but no video visual keeps its "
                     f"audio; narration used instead")
                mix = "voice"
            scene_mixes.append(mix)

            apath = None
            # ---- a dialogue: every line in its own character's voice -------
            if sc.speakers:
                try:
                    _lay = _speaker_layout(sc)
                except ValueError as exc:
                    raise rnd.RenderError(f"scene {i}: {exc}")
                _lines, _t = [], 0.0
                for k, sp in enumerate(sc.speakers):
                    _lp = None
                    if sp.audio:
                        try:
                            _lp = media.resolve(sp.audio, work, f"s{i}l{k}-audio")
                        except media.MediaError as exc:
                            warn(f"scene {i} line {k}: audio could not be fetched "
                                 f"({exc}); synthesised from text instead")
                    if _lp is None and (sp.text or "").strip():
                        _lp = _synth_scene(
                            loop, sp.text, sp.voice or sc.voice or req.voice,
                            os.path.join(work, f"s{i}l{k}-voice.wav"))
                    if not _lp:
                        warn(f"scene {i} line {k}: no text and no audio; skipped")
                        continue
                    try:
                        _ld = media.duration_of(_lp)
                    except media.MediaError as exc:
                        warn(f"scene {i} line {k}: could not be probed ({exc}); skipped")
                        continue
                    _lines.append({"path": _lp, "start": round(_t, 3),
                                   "dur": round(_ld, 3),
                                   "key": _speaker_key(sp, k),
                                   "character": sp.character,
                                   "text": (sp.text or "").strip(),
                                   "index": k})
                    _t += _ld + max(0.0, float(sp.gap or 0.0))
                if _lines:
                    # The lines ARE the narration: one track, each line where
                    # it falls, so captions and music ducking see a normal
                    # scene and need to know nothing about dialogue.
                    apath = rnd.place_at(
                        [(ln["path"], ln["start"]) for ln in _lines],
                        os.path.join(work, f"scene{i}-dialogue.wav"),
                        duration=_t, cancel=cancel, log=log)
                    scene_lines[i] = {"layout": _lay, "lines": _lines,
                                      "span": round(_t, 3)}
                else:
                    warn(f"scene {i}: speakers were given but none could be "
                         f"spoken; the scene falls back to its own text")
            if apath is None and sc.audio:
                try:
                    apath = media.resolve(sc.audio, work, f"scene{i}-audio")
                except media.MediaError as exc:
                    warn(f"scene {i}: audio could not be fetched ({exc}); "
                         f"synthesising from text instead")
            # audio_mix="clip" is "the clip's sound INSTEAD of narration", so
            # there is nothing to synthesise -- and synthesising it anyway would
            # bill a TTS call per scene for audio no one ever hears.
            if apath is None and mix != "clip" and not sc.speakers and (sc.text or "").strip():
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
            # With the clip as the whole soundtrack, the clip's remaining length
            # is the scene's -- cutting it at an arbitrary 3 s default, or at a
            # narration's length, truncates the content itself. An explicit
            # `duration` still wins: a caller who names one means it.
            if mix == "clip" and not sc.duration:
                _cd = float((clip_media.get(i) or {}).get("probe", {}).get("duration") or 0.0)
                _cd -= float((clip_media.get(i) or {}).get("trim_start") or 0.0)
                if _cd > 0:
                    dur = round(_cd, 3)
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

        # Clips are built one transition longer so the overlap is absorbed.
        extra_t = tdur if ttype != "none" else 0.0

        # ---------------------------------------------------- 2. captions
        events: List[dict] = []
        if caps and caps.enabled:
            stage("captions", 30.0, "timing the words")
            timing = (caps.timing or "auto").lower()
            model = (_pick_stt_model(caps, _request_languages(req))
                     if timing in ("auto", "stt") else None)
            if timing == "stt" and not model:
                raise rnd.RenderError("captions.timing=stt but no speech model is configured")
            if timing == "auto" and not model:
                warn("no speech model configured; caption timings are estimated")
            words: List[dict] = []
            _no_words_warned: set = set()
            for i, sc in enumerate(req.scenes):
                # ---- a dialogue: caption each line in its own window -------
                # Better than captioning the scene as one block, not merely
                # different: each line's text, speaker and time window are
                # known exactly, so there is nothing to infer and no chance of
                # a question and its answer landing in one caption.
                info = scene_lines.get(i)
                if info:
                    for ln in info["lines"]:
                        if not ln["text"]:
                            continue
                        _lstt: List[dict] = []
                        if model:
                            try:
                                _lstt = _stt_words(
                                    loop, ln["path"], model,
                                    ((sc.speakers[ln["index"]].voice
                                      or sc.voice or req.voice).language))
                            except Exception as exc:
                                warn(f"scene {i} line {ln['index']}: word "
                                     f"timestamps failed ({exc}); estimated instead")
                        _lstart = starts[i] + float(ln["start"])
                        _lspan = max(0.2, float(ln["dur"]))
                        _got = (cap.align_words(ln["text"], _lstt, _lstart, _lspan)
                                if _lstt else
                                cap.estimate_words(ln["text"], _lstart, _lspan))
                        _who = ln.get("character") or ln["key"]
                        for w in _got:
                            w["speaker"] = _who
                        words += _got
                    continue
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
                    else:
                        # A backend that cannot time words answers 200 with none
                        # of them, so without this the downgrade is invisible and
                        # the captions just drift. Named once, for the model, and
                        # only when the call itself succeeded -- a failure has
                        # already been reported above.
                        if not stt and model not in _no_words_warned:
                            _no_words_warned.add(model)
                            warn(f"{model!r} returned no word timings; caption "
                                 f"timings are estimated. Set captions.stt_model "
                                 f"to a model whose backend reports them "
                                 f"(wav2vec2, whisper_hf, CrisperWhisper, vosk "
                                 f"or faster-whisper).")
                span = max(0.2, scene_dur[i] - max(0.0, float(sc.padding or 0.0)))
                got = (cap.align_words(text, stt, starts[i], span) if stt
                       else cap.estimate_words(text, starts[i], span))
                # A presenter bubble at the bottom would sit on the captions:
                # push this scene's lines above it (per-event ASS margin).
                bump = _caption_margin_bump(presenter_for(sc, req), caps, canvas)
                if bump:
                    for w in got:
                        w["margin_v"] = bump
                words += got
            events = cap.break_lines(words, caps.max_words_per_line, caps.max_lines)
            if caps.font and not media.font_installed(caps.font):
                warn(f"font {caps.font!r} is not installed; captions use the "
                     f"closest available family")

        # ------------------------------------------- 3. presenter (lip-sync)
        # Built before the visuals because a full-screen presenter replaces them,
        # and because it is the GPU part: it should not wait behind ffmpeg.
        presenters: List[Optional[dict]] = [None] * n_scenes
        want_pres = [presenter_for(sc, req) for sc in req.scenes]

        # ---- dialogue scenes: one presenter PER CHARACTER ------------------
        # Built before the single-presenter pass and marked so that pass skips
        # the scene: a scene is a monologue or a dialogue, not both.
        for i, sc in enumerate(req.scenes):
            info = scene_lines.get(i)
            if not info:
                continue
            want_pres[i] = None          # the dialogue owns this scene
            from codai.compose import presenter as _pres
            order = _speaker_order(sc)
            stage("presenter", 30.0 + 9.0 * i / max(1, n_scenes),
                  f"scene {i + 1}/{n_scenes}: {len(order)} speakers")
            base_spec = presenter_for(sc, req) or {}
            per: List[dict] = []
            try:
                for col, (key, sp) in enumerate(order):
                    mine = [ln for ln in info["lines"] if ln["key"] == key]
                    if not mine:
                        continue
                    # This character's own track: their lines where they fall,
                    # silence while the other one talks. One sync run for the
                    # whole scene then gives a mouth that moves on their words
                    # and a closed mouth while they listen.
                    track = rnd.place_at(
                        [(ln["path"], ln["start"]) for ln in mine],
                        os.path.join(work, f"s{i}sp{col}-track.wav"),
                        duration=scene_dur[i] + extra_t, cancel=cancel, log=log)
                    spec = dict(base_spec)
                    spec.pop("layout", None)
                    if sp.character:
                        spec["character"] = sp.character
                    if sp.image:
                        spec["image"] = sp.image
                    spec.setdefault("motion", "still")
                    built = _pres.build_scene_presenter(
                        spec, narration=track, duration=scene_dur[i] + extra_t,
                        width=(canvas.width // max(1, len(order))
                               if info["layout"] == "row" else canvas.width),
                        height=canvas.height, fps=canvas.fps, workdir=work,
                        index=f"{i}s{col}", loop=loop, cancel=cancel, log=log,
                        gpu_lease=lambda: _gpu_lease(loop, "lipsync"))
                    built["key"] = key
                    built["character"] = sp.character
                    per.append(built)
                if not per:
                    raise rnd.RenderError("no speaker could be lip-synced")
                if info["layout"] == "row" and len(per) > 1:
                    path = rnd.composite_row(
                        [b["path"] for b in per],
                        os.path.join(work, f"s{i}-speakers.mp4"),
                        width=canvas.width, height=canvas.height,
                        fps=canvas.fps, cancel=cancel, log=log)
                elif info["layout"] == "cut" and len(per) > 1:
                    # The frame cuts to whoever is speaking: each line is that
                    # character's own clip over that line's window, joined in
                    # order. Taken from the already-synced full-scene clip, so
                    # a character is still mid-expression when we cut to them
                    # rather than starting from a cold still.
                    by_key = {b["key"]: b["path"] for b in per}
                    segs, segdurs = [], []
                    _lines = info["lines"]
                    for k, ln in enumerate(_lines):
                        src = by_key.get(ln["key"])
                        if not src:
                            continue
                        _next = (_lines[k + 1]["start"] if k + 1 < len(_lines)
                                 else scene_dur[i] + extra_t)
                        _len = max(0.08, float(_next) - float(ln["start"]))
                        segs.append(rnd.cut_segment(
                            src, os.path.join(work, f"s{i}cut{k}.mp4"),
                            start=ln["start"], duration=_len, fps=canvas.fps,
                            width=canvas.width, height=canvas.height,
                            cancel=cancel, log=log))
                        segdurs.append(_len)
                    if not segs:
                        raise rnd.RenderError("no dialogue line could be cut")
                    path = os.path.join(work, f"s{i}-speakers.mp4")
                    rnd.concat_clips(segs, path, work, cancel=cancel, log=log)
                else:
                    # A single character: this is just a presenter.
                    path = per[0]["path"]
                presenters[i] = {"path": path, "engine": per[0].get("engine"),
                                 "layout": "full", "spec": base_spec,
                                 "speakers": [b.get("character") or b["key"]
                                              for b in per]}
            except rnd.Cancelled:
                raise
            except Exception as exc:
                warn(f"scene {i}: the dialogue could not be lip-synced ({exc}); "
                     f"the scene's visuals are used and the lines are still heard")
                if not sc.visuals:
                    sc.visuals = [Visual(type="gradient")]

        if any(p is not None for p in want_pres):
            from codai.compose import presenter as _pres
            for i, sc in enumerate(req.scenes):
                spec = want_pres[i]
                if spec is None:
                    continue
                stage("presenter", 30.0 + 9.0 * i / max(1, n_scenes),
                      f"scene {i + 1}/{n_scenes}: lip-sync")
                try:
                    presenters[i] = _pres.build_scene_presenter(
                        spec, narration=scene_audio[i], duration=scene_dur[i] + extra_t,
                        width=canvas.width, height=canvas.height, fps=canvas.fps,
                        workdir=work, index=i, loop=loop, cancel=cancel, log=log,
                        gpu_lease=lambda: _gpu_lease(loop, "lipsync"))
                    presenters[i]["layout"] = (spec.get("layout") or "full").lower()
                    presenters[i]["spec"] = spec
                    if spec.get("remove_background"):
                        _matte_presenter(presenters[i], index=i, work=work,
                                         fps=canvas.fps, warn=warn, stage=stage,
                                         progress=30.0 + 9.0 * i / max(1, n_scenes),
                                         scene_no=f"{i + 1}/{n_scenes}",
                                         cancel=cancel, log=log,
                                         gpu_lease=lambda: _gpu_lease(loop, "lipsync"))
                except rnd.Cancelled:
                    raise
                except Exception as exc:
                    warn(f"scene {i}: presenter skipped ({exc}); the scene's visuals "
                         f"are used instead")
                    if not sc.visuals:
                        # Nothing left to show: a gradient is the documented floor.
                        sc.visuals = [Visual(type="gradient")]

        # ---------------------------------------------------- 4. visuals
        clips: List[str] = []
        clip_durs: List[float] = []
        made = 0
        total_visuals = sum(len(sc.visuals) for sc in req.scenes)
        for i, sc in enumerate(req.scenes):
            pres = presenters[i]
            # A full-screen presenter IS the scene: no visuals are rendered.
            if pres and pres["layout"] == "full":
                stage("visuals", 40.0 + 28.0 * made / max(1, total_visuals or 1),
                      f"scene {i + 1}/{n_scenes}: presenter (full screen)")
                clips.append(pres["path"])
                clip_durs.append(scene_dur[i] + extra_t)
                continue
            scene_clips: List[str] = []
            n = max(1, len(sc.visuals))
            # With a presenter the scene is composited as ONE clip, so its visuals
            # share the scene without a transition between them; without one they
            # stay separate clips and transitions apply between every visual, as
            # they always have.
            slot = scene_dur[i] / n if pres else scene_dur[i] / n
            for j, v in enumerate(sc.visuals):
                stage("visuals", 40.0 + 28.0 * made / max(1, total_visuals),
                      f"scene {i + 1}/{n_scenes}: visual {j + 1}/{n}")
                spec = v.model_dump()
                vt = (spec.get("type") or "image").lower()
                if vt in ("video", "image"):
                    try:
                        # The narration pass already fetched and probed this
                        # exact source when the scene keeps its audio; fetching
                        # it again would download the same clip twice.
                        _cached = clip_media.get(i) or {}
                        if (_cached.get("src")
                                and _cached["src"] == (spec.get("src") or "")):
                            spec["_path"] = _cached["path"]
                            spec["_probe"] = _cached["probe"]
                        else:
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
                                     fps=canvas.fps, duration=slot + extra_t, fit=fit,
                                     index=made, workdir=work, cancel=cancel, log=log)
                except rnd.Cancelled:
                    raise
                except Exception as exc:
                    warn(f"scene {i} visual {j}: render failed ({exc}); used gradient")
                    rnd.build_visual({"type": "gradient"}, out, width=canvas.width,
                                     height=canvas.height, fps=canvas.fps,
                                     duration=slot + extra_t, fit=fit, index=made,
                                     workdir=work, cancel=cancel, log=log)
                scene_clips.append(out)
                made += 1
            if pres:
                # One clip for the scene, then the bubble or the split over it.
                joined = os.path.join(work, f"scene-{i}.mp4")
                if len(scene_clips) == 1:
                    joined = scene_clips[0]
                else:
                    rnd.concat_clips(scene_clips, joined, work, cancel=cancel, log=log)
                comp = os.path.join(work, f"scene-{i}-pres.mp4")
                spec = pres["spec"]
                try:
                    if pres["layout"] == "pip":
                        rnd.composite_pip(joined, pres["path"], comp,
                                          width=canvas.width, height=canvas.height,
                                          fps=canvas.fps, size=spec.get("size", 0.38),
                                          position=spec.get("position") or "bottom-right",
                                          shape=spec.get("shape") or "circle",
                                          border=spec.get("border"),
                                          alpha_video=pres.get("alpha"), workdir=work,
                                          cancel=cancel, log=log)
                    else:
                        rnd.composite_split(joined, pres["path"], comp,
                                            width=canvas.width, height=canvas.height,
                                            fps=canvas.fps,
                                            presenter_half=spec.get("split") or "bottom",
                                            alpha_video=pres.get("alpha"),
                                            cancel=cancel, log=log)
                    clips.append(comp)
                except rnd.Cancelled:
                    raise
                except Exception as exc:
                    warn(f"scene {i}: presenter could not be composited ({exc}); "
                         f"the scene's visuals are used as they are")
                    presenters[i] = None
                    clips.append(joined)
                clip_durs.append(scene_dur[i] + extra_t)
            else:
                clips.extend(scene_clips)
                clip_durs.extend([slot + extra_t] * len(scene_clips))

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
        clip_used = False
        for i, sc in enumerate(req.scenes):
            p = os.path.join(work, f"narr-{i}.wav")
            mix = scene_mixes[i] if i < len(scene_mixes) else "voice"
            cm = clip_media.get(i) if mix in ("clip", "both") else None
            ctrack = None
            if cm:
                # Matched to what build_visual did with the picture: a clip
                # shorter than its slot was looped there, so its sound is
                # looped too or it would fall silent halfway through the shot
                # it belongs to.
                _cdur = float((cm.get("probe") or {}).get("duration") or 0.0)
                _trim = float(cm.get("trim_start") or 0.0)
                _loop = bool(_cdur and (_cdur - _trim) < scene_dur[i] - 0.02)
                _vol = cm.get("volume")
                if _vol is None:
                    _vol = 1.0 if mix == "clip" else float(
                        getattr(sc, "clip_volume", 0.35) or 0.35)
                try:
                    ctrack = rnd.clip_audio(
                        cm["path"], os.path.join(work, f"clipaud-{i}.wav"),
                        duration=scene_dur[i], trim_start=_trim,
                        volume=float(_vol), loop=_loop, cancel=cancel, log=log)
                    clip_used = True
                except rnd.Cancelled:
                    raise
                except Exception as exc:
                    warn(f"scene {i}: the clip's audio could not be extracted "
                         f"({exc}); rendered without it")
                    ctrack = None
            _voice = scene_audio[i] if mix != "clip" else None
            if ctrack:
                rnd.mix_voice_clip(_voice, ctrack, p, duration=scene_dur[i],
                                   duck=(mix == "both"), cancel=cancel, log=log)
            elif _voice:
                rnd.pad_to(_voice, p, scene_dur[i], cancel=cancel, log=log)
            else:
                rnd.silence(p, scene_dur[i], cancel=cancel, log=log)
            parts.append(p)
        narration = rnd.concat_audio(parts, os.path.join(work, "narration.wav"), work,
                                    cancel=cancel, log=log)
        # Levelled whenever the bed carries anything real -- a clip's own sound
        # needs it at least as much as a synthesised voice, since it arrives at
        # whatever level it was filmed at.
        if any(scene_audio) or clip_used:
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
        result["scenes"] = [
            {"index": i, "start": starts[i],
             "end": round(starts[i] + scene_dur[i], 3),
             "presenter": ({"engine": presenters[i]["engine"],
                            "layout": presenters[i]["layout"],
                            # Which matting engine cut them out, or None when
                            # the scene was composited as filmed -- a client
                            # that asked for remove_background can tell the two
                            # apart without reading the warnings.
                            "matted": presenters[i].get("matted")}
                           if presenters[i] else None)}
            for i in range(n_scenes)]
        result["duration"] = total

        _update(job_id, status="done", stage="thumbnail", progress=100.0,
                message="done", result=result)
        print(f"[compose {job_id}] done: {total:.1f}s, {n_scenes} scenes, "
              f"{len(clips)} visuals", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _caption_margin_bump(pres: Optional[dict], caps, canvas) -> int:
    """How far up this scene's captions move, in canvas pixels.

    Only a bottom bubble collides with bottom captions; a full-screen presenter or
    a split layout does not, and centred captions are already above a bubble."""
    if not pres or (pres.get("layout") or "full").lower() != "pip":
        return 0
    if (caps.position or "center").lower() != "bottom":
        return 0
    pos = (pres.get("position") or "bottom-right").lower()
    if not pos.startswith("bottom"):
        return 0
    _, bh, _, y = rnd.bubble_geometry(canvas.width, canvas.height,
                                      pres.get("size", 0.38), pos,
                                      pres.get("shape") or "circle")
    # Clear the bubble's top edge plus a little air.
    return max(0, int(canvas.height - y + canvas.height * 0.02))


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


# =============================================================================
# Talking head — the presenter on its own
# =============================================================================

class TalkingHeadRequest(BaseModel):
    """A character saying one thing, for a preview on a character page."""
    character: Optional[str] = None
    image: Optional[str] = None
    text: Optional[str] = None
    audio: Optional[str] = None               # pre-rendered instead of text
    voice: VoiceSpec = Field(default_factory=VoiceSpec)
    engine: str = "auto"
    motion: str = "still"                     # a preview does not need a generated shot
    base_src: Optional[str] = None
    prompt: Optional[str] = None
    video_model: Optional[str] = None
    canvas: Canvas = Field(default_factory=lambda: Canvas(width=720, height=720, fps=25))
    output: OutputSpec = Field(default_factory=OutputSpec)
    is_async: bool = Field(default=True, alias="async")
    model_config = ConfigDict(extra="allow", populate_by_name=True)


#: A preview is a preview: a quarter-minute is plenty and keeps the GPU free.
TALKING_HEAD_MAX_SECONDS = 15.0


def _talking_head_spec(req: TalkingHeadRequest) -> ComposeRequest:
    """A talking-head request IS a one-scene composition with a full-screen
    presenter, no captions and no music — so it runs on exactly the same code."""
    pres = {"character": req.character, "image": req.image, "engine": req.engine,
            "motion": req.motion, "base_src": req.base_src, "prompt": req.prompt,
            "video_model": req.video_model, "layout": "full"}
    scene = {"text": req.text or "", "audio": req.audio,
             "visuals": [], "padding": 0.05}
    return ComposeRequest(canvas=req.canvas, voice=req.voice,
                          presenter=PresenterSpec(**pres), scenes=[Scene(**scene)],
                          captions=CaptionSpec(enabled=False), output=req.output,
                          outputs=["video"], **{"async": req.is_async})


@router.post("/v1/video/talking-head", summary="A character saying one line",
             tags=["Video"])
async def talking_head(req: TalkingHeadRequest, http_request: Request,
                       _auth=Depends(_require_api_auth)):
    """Lip-sync one line onto a character's face — the preview a character page
    shows before anyone pays for a render.

    Same job semantics as ``/v1/video/compose`` (and the same job store, so
    ``GET /v1/video/compose/{id}`` works too): ``202 {id, status}`` then poll, or
    ``"async": false`` to block. Text is synthesised with ``voice``; supply
    ``audio`` instead to skip synthesis. Capped at 15 s of audio."""
    if not (req.text or "").strip() and not req.audio:
        raise HTTPException(status_code=400, detail="talking-head needs text or audio")
    spec = _talking_head_spec(req)
    _validate(spec)
    if req.audio:
        # A caller's clip is the one thing we can measure before rendering.
        try:
            import tempfile as _tf
            with _tf.TemporaryDirectory() as td:
                p = media.resolve(req.audio, td, "th-audio")
                dur = media.duration_of(p)
            if dur > TALKING_HEAD_MAX_SECONDS + 0.5:
                raise HTTPException(status_code=400, detail=(
                    f"audio is {dur:.1f}s; talking-head is capped at "
                    f"{TALKING_HEAD_MAX_SECONDS:.0f}s (use /v1/video/compose for a full render)"))
        except media.MediaError as exc:
            raise HTTPException(status_code=400, detail=f"audio: {exc}")
    try:
        from codai.api.urlutils import get_base_url
        base_url = get_base_url(http_request)
    except Exception:
        base_url = ""
    rec = submit(spec, base_url)
    if req.is_async:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=202,
                            content={"id": rec["id"], "status": "queued"})
    job_id = rec["id"]
    while True:
        await asyncio.sleep(0.5)
        cur = _get(job_id)
        if cur is None:
            raise HTTPException(status_code=500, detail="job disappeared")
        if cur["status"] in ("done", "failed", "cancelled"):
            if cur["status"] == "failed":
                raise HTTPException(status_code=500,
                                    detail=cur.get("error") or "talking-head failed")
            return _public(cur)


@router.get("/v1/video/talking-head/{job_id}", summary="Talking-head job status",
            tags=["Video"])
async def talking_head_status(job_id: str, _auth=Depends(_require_api_auth)):
    """The same document as ``GET /v1/video/compose/{id}`` — one job store."""
    return await compose_status(job_id, _auth)


@router.post("/v1/video/talking-head/{job_id}/cancel",
             summary="Cancel a talking-head job", tags=["Video"])
async def talking_head_cancel(job_id: str, _auth=Depends(_require_api_auth)):
    return await compose_cancel(job_id, _auth)


@router.get("/v1/video/presenter/engines", summary="Installed lip-sync engines",
            tags=["Video"])
async def presenter_engines(_auth=Depends(_require_api_auth)):
    """What a presenter can be driven with here — so a client can grey out the
    feature instead of discovering a 501 at submission."""
    from codai.compose import presenter as _pres
    have = _pres.available_engines()
    try:
        from codai.compose import matting
        matte = matting.engine_name()
    except Exception:
        matte = None
    # An engine coderai can install but has not is neither "installed" nor
    # permanently absent, and a client that cannot tell the two apart greys the
    # feature out for good. Say which ones an operator can still bring up, and
    # what each is short of.
    installable = {}
    try:
        from codai.api import lipsync_engines as _le
        for _name in _le.MANAGED:
            if _name not in have:
                installable[_name] = _le.missing_pieces(_name)
    except Exception:
        pass
    return {"engines": have, "available": bool(have),
            "video_engines": [e for e in have if e in _pres.VIDEO_ENGINES],
            "portrait_engines": [e for e in have if e in _pres.PORTRAIT_ENGINES],
            "installable": installable,
            "layouts": ["full", "pip", "split"],
            # remove_background is a separate model from the lip-sync engine, so
            # a client greys out that toggle on its own.
            "matting_engine": matte, "remove_background": bool(matte),
            "talking_head_max_seconds": TALKING_HEAD_MAX_SECONDS}


@router.get("/v1/video/compose", summary="List composition jobs", tags=["Video"])
async def compose_list(_auth=Depends(_require_api_auth)):
    """Every job this front still remembers, newest first — for an admin page or
    a client that lost a job id."""
    with _jobs_lock:
        recs = sorted(_jobs.values(), key=lambda r: r.get("created_at") or 0, reverse=True)
    return {"jobs": [_public(r) for r in recs]}
