# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The talking presenter: a character whose mouth moves with the narration.

The most common reel format is a person talking to camera — full screen, or as a
bubble over B-roll. Only this server can build it: composition synthesises the
narration, so it alone holds the exact audio of each scene, and the lip-sync
engines and identity-locked video generation are here too.

Per scene the work is: get a face (a saved character profile's reference, or a
portrait the caller supplied), get something for it to move in (a still, a clip
the caller gave, or a short identity-locked clip generated for this scene), then
drive an engine with **that scene's narration** and hold the result to the
scene's exact length. The layouts (`full`, `pip`, `split`) are compositing and
live in :mod:`codai.compose.render`.

Engines are whatever the install has. ``wav2lip`` and ``sadtalker`` are known by
name; anything else on PATH under its own name can be selected by that name, so a
newer engine (LatentSync, MuseTalk, Hallo…) needs no API change. ``auto`` picks a
video-driven engine when there is a clip to drive and a portrait engine
otherwise.

Nothing here ever fails a render: a missing engine, a face the engine cannot
find, a crash — each falls back to the scene's ordinary visuals and leaves a line
in the job's warnings. That is the same rule a broken stock clip follows.
"""

import glob
import os
import shutil
import subprocess
import threading
from typing import List, Optional

from codai.compose import media
from codai.compose import render as rnd

#: Engines that drive a VIDEO (they need a base clip; the mouth is replaced).
VIDEO_ENGINES = ("wav2lip",)
#: Engines that animate a PORTRAIT (they make their own motion from a still).
PORTRAIT_ENGINES = ("sadtalker",)
#: Preference order for ``engine: "auto"`` — best mouth first for a clip, best
#: motion first for a still.
_AUTO_WITH_BASE = ("wav2lip", "sadtalker")
_AUTO_STILL = ("sadtalker", "wav2lip")


class PresenterError(Exception):
    """This scene gets no presenter; the caller falls back and warns."""


# ------------------------------------------------------------------ engines
def available_engines() -> List[str]:
    """Which lip-sync engines this install can actually run, in a stable order.

    ``wav2lip`` counts when the packaged shim is on PATH *or* the managed install
    already has its code and weights — not when they would merely be
    downloadable, because a compose job is the wrong place to fetch 500 MB."""
    out = []
    if shutil.which("wav2lip") or shutil.which("Wav2Lip"):
        out.append("wav2lip")
    else:
        try:
            from codai.api import lipsync
            if lipsync.is_available():
                out.append("wav2lip")
        except Exception:
            pass
    if shutil.which("sadtalker"):
        out.append("sadtalker")
    return out


def engine_installed(name: str) -> bool:
    n = (name or "").strip().lower()
    if not n or n == "auto":
        return bool(available_engines())
    if n in available_engines():
        return True
    # An engine this module has never heard of, installed under its own name.
    return bool(shutil.which(n))


def pick_engine(requested: str, has_base: bool) -> str:
    """The engine to use, or raise when there is none."""
    req = (requested or "auto").strip().lower()
    have = available_engines()
    if req and req != "auto":
        if req in have or shutil.which(req):
            return req
        raise PresenterError(
            f"lip-sync engine {req!r} is not installed"
            + (f" (installed: {', '.join(have)})" if have else " and neither is any other"))
    for cand in (_AUTO_WITH_BASE if has_base else _AUTO_STILL):
        if cand in have:
            return cand
    if have:
        return have[0]
    raise PresenterError("no lip-sync engine is installed (install the wav2lip or "
                         "sadtalker shim, or the managed wav2lip assets)")


def run_engine(engine: str, face_path: str, audio_path: str, out_path: str,
               cancel: Optional[threading.Event] = None, timeout: int = 3600,
               log=None) -> str:
    """Drive one engine: ``face`` (a clip or a portrait) + audio → a talking clip.

    Kept as one function so a test can replace it, and so a new engine is a
    couple of lines rather than a new code path."""
    if cancel is not None and cancel.is_set():
        raise rnd.Cancelled()
    eng = (engine or "").lower()
    workdir = os.path.dirname(out_path) or "."

    if eng == "wav2lip":
        binary = shutil.which("wav2lip") or shutil.which("Wav2Lip")
        if binary:
            cmd = [binary, "--face", face_path, "--audio", audio_path,
                   "--outfile", out_path]
            _run(cmd, cancel=cancel, timeout=timeout, log=log)
            if os.path.isfile(out_path):
                return out_path
            raise PresenterError("wav2lip produced no output")
        from codai.api.lipsync import run_wav2lip
        return run_wav2lip(face_path, audio_path, out_path, timeout=timeout)

    if eng == "sadtalker":
        binary = shutil.which("sadtalker")
        if not binary:
            raise PresenterError("sadtalker is not on PATH")
        res_dir = os.path.join(workdir, "sadtalker-out")
        os.makedirs(res_dir, exist_ok=True)
        still = media.kind_of(media.sniff_mime(open(face_path, "rb").read(32))) == "image"
        cmd = [binary, "--driven_audio", audio_path,
               "--source_image" if still else "--source_video", face_path,
               "--result_dir", res_dir]
        _run(cmd, cancel=cancel, timeout=timeout, log=log)
        made = sorted(glob.glob(os.path.join(res_dir, "**", "*.mp4"), recursive=True),
                      key=os.path.getmtime)
        if not made:
            raise PresenterError("sadtalker produced no video")
        shutil.move(made[-1], out_path)
        return out_path

    # An engine installed under its own name: the wav2lip-style CLI is the de
    # facto shape (face + audio + outfile). If it wants something else, the
    # operator can wrap it in a shim of that name.
    binary = shutil.which(eng)
    if not binary:
        raise PresenterError(f"lip-sync engine {engine!r} is not installed")
    _run([binary, "--face", face_path, "--audio", audio_path, "--outfile", out_path],
         cancel=cancel, timeout=timeout, log=log)
    if not os.path.isfile(out_path):
        raise PresenterError(f"{engine}: produced no output")
    return out_path


def _run(cmd: List[str], cancel=None, timeout: int = 3600, log=None) -> None:
    if log:
        log("presenter: " + " ".join(cmd[:8]) + (" …" if len(cmd) > 8 else ""))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    waited = 0.0
    while True:
        try:
            _, err = proc.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            waited += 0.5
            if cancel is not None and cancel.is_set():
                proc.kill()
                proc.communicate()
                raise rnd.Cancelled()
            if waited > timeout:
                proc.kill()
                proc.communicate()
                raise PresenterError(f"{os.path.basename(cmd[0])} timed out")
    if proc.returncode != 0:
        tail = (err or b"").decode(errors="replace").strip().splitlines()[-6:]
        raise PresenterError(f"{os.path.basename(cmd[0])} failed "
                             f"(rc={proc.returncode}): {' | '.join(tail)}")


# -------------------------------------------------------------------- face
def portrait_path(spec: dict, workdir: str, name_hint: str = "presenter") -> str:
    """The face to use: an explicit portrait, or a character profile's best
    reference (a front view when one is labelled, else the first)."""
    ref = (spec.get("image") or "").strip() if isinstance(spec.get("image"), str) else ""
    if ref:
        try:
            return media.resolve(ref, workdir, name_hint)
        except media.MediaError as exc:
            raise PresenterError(f"presenter image: {exc}")
    char = (spec.get("character") or "").strip()
    if not char:
        raise PresenterError("presenter needs a character or an image")
    try:
        from codai.api.characters import _load_character_images, _load_character_meta
    except Exception as exc:
        raise PresenterError(f"character profiles unavailable: {exc}")
    if not _load_character_meta(char):
        raise PresenterError(f"no character profile named {char!r}")
    images = _load_character_images(char)
    if not images:
        raise PresenterError(f"character {char!r} has no reference images")
    best = next((i for i in images if "front" in (i.label or "").lower()), images[0])
    try:
        data = media.fetch_bytes(best.data)
    except media.MediaError as exc:
        raise PresenterError(f"character {char!r}: {exc}")
    os.makedirs(workdir, exist_ok=True)
    out = os.path.join(workdir, f"{name_hint}-face.png")
    with open(out, "wb") as f:
        f.write(data)
    return out


# ------------------------------------------------------------------- clips
def base_clip(spec: dict, portrait: str, out_path: str, *, duration: float, width: int,
              height: int, fps: int, workdir: str, loop=None, cancel=None,
              log=None) -> Optional[str]:
    """What the engine drives, or None when the engine animates a portrait.

    ``motion``: ``still`` holds the portrait, ``src`` uses the caller's clip,
    ``generate`` asks the video model for a short identity-locked "talking to
    camera" shot of this character."""
    motion = (spec.get("motion") or "generate").strip().lower()
    if motion == "src":
        ref = (spec.get("base_src") or "").strip()
        if not ref:
            raise PresenterError('motion="src" needs base_src')
        src = media.resolve(ref, workdir, "presenter-base")
        rnd.build_visual({"type": "video", "_path": src, "_probe": media.probe(src)},
                         out_path, width=width, height=height, fps=fps,
                         duration=duration, fit="cover", workdir=workdir,
                         cancel=cancel, log=log)
        return out_path
    if motion == "generate":
        gen = _generate_base(spec, portrait, out_path, duration=duration, width=width,
                             height=height, fps=fps, workdir=workdir, loop=loop,
                             cancel=cancel, log=log)
        if gen:
            return gen
        # Falling back to a still is better than losing the presenter; the caller
        # turns the reason into a warning.
        if log:
            log("presenter: video generation unavailable; using a still")
    rnd.still_video(portrait, out_path, duration=duration, width=width, height=height,
                    fps=fps, cancel=cancel, log=log)
    return out_path


def _generate_base(spec: dict, portrait: str, out_path: str, *, duration: float,
                   width: int, height: int, fps: int, workdir: str, loop=None,
                   cancel=None, log=None) -> Optional[str]:
    """One short identity-locked clip of the character speaking to camera, made
    with the ordinary video endpoint (so its model config, acceleration, LoRAs and
    keyframe identity bridge all apply). None when it cannot be done."""
    from codai.api.compose import _await          # the module's own loop
    prompt = (spec.get("prompt")
              or "talking to the camera, natural head movement, soft studio light")
    model = (spec.get("video_model") or spec.get("model") or "").strip()
    char = (spec.get("character") or "").strip()
    try:
        from codai.api.video import VideoGenerationRequest, generate_video
    except Exception as exc:
        if log:
            log(f"presenter: video endpoint unavailable ({exc})")
        return None
    body = {"prompt": prompt, "mode": "i2v", "image": _data_uri(portrait),
            "width": width, "height": height, "num_frames": max(9, int(duration * fps)),
            "fps": fps, "response_format": "b64_json"}
    if model:
        body["model"] = model
    if char:
        body["character_profiles"] = [char]
    try:
        req = VideoGenerationRequest(**body)
    except Exception as exc:
        if log:
            log(f"presenter: video request rejected ({exc})")
        return None
    try:
        res = _await(loop, generate_video(req, None), timeout=3600)
    except Exception as exc:
        if log:
            log(f"presenter: video generation failed ({exc})")
        return None
    item = ((res or {}).get("data") or [{}])[0] if isinstance(res, dict) else {}
    blob = item.get("b64_json") or item.get("b64_mp4")
    raw = None
    if blob:
        import base64
        raw = base64.b64decode(blob)
    elif item.get("url"):
        try:
            raw = media.fetch_bytes(item["url"])
        except media.MediaError:
            raw = None
    if not raw:
        return None
    gen = os.path.join(workdir, "presenter-gen.mp4")
    with open(gen, "wb") as f:
        f.write(raw)
    # Normalise to the canvas and hold to the clip's length: a generated shot is
    # usually a couple of seconds, the scene may be longer.
    rnd.build_visual({"type": "video", "_path": gen, "_probe": media.probe(gen)},
                     out_path, width=width, height=height, fps=fps, duration=duration,
                     fit="cover", workdir=workdir, cancel=cancel, log=log)
    return out_path


def _data_uri(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        raw = f.read()
    return f"data:{media.sniff_mime(raw)};base64," + base64.b64encode(raw).decode()


# --------------------------------------------------------------- the scene
def build_scene_presenter(spec: dict, *, narration: Optional[str], duration: float,
                          width: int, height: int, fps: int, workdir: str,
                          index: int = 0, loop=None, cancel=None, log=None,
                          gpu_lease=None) -> dict:
    """The presenter clip for one scene.

    Returns ``{"path": …, "engine": …}``; raises :class:`PresenterError` when this
    scene cannot have one (the caller warns and falls back). ``narration`` is the
    scene's audio — without it there is nothing to sync to, and a presenter that
    just stands there is what the scene's visuals already are."""
    if not narration or not os.path.isfile(narration):
        raise PresenterError("the scene has no narration to lip-sync to")
    face = portrait_path(spec, workdir, f"pres{index}")
    motion = (spec.get("motion") or "generate").strip().lower()
    if motion not in ("generate", "still", "src"):
        raise PresenterError(f'motion must be generate, still or src (got {motion!r})')

    # The engine drives the mouth for exactly as long as the narration sounds;
    # the scene's padding is held afterwards.
    speech = media.duration_of(narration)
    engine = pick_engine(spec.get("engine") or "auto",
                         has_base=(motion in ("generate", "src")))
    # Portrait engines make their own motion — a base clip would throw it away.
    base = None
    if engine in VIDEO_ENGINES or motion == "src":
        base = base_clip(spec, face, os.path.join(workdir, f"pres{index}-base.mp4"),
                         duration=speech + 0.35, width=width, height=height, fps=fps,
                         workdir=workdir, loop=loop, cancel=cancel, log=log)
    raw = os.path.join(workdir, f"pres{index}-sync.mp4")
    lease = gpu_lease() if callable(gpu_lease) else None
    try:
        run_engine(engine, base or face, narration, raw, cancel=cancel, log=log)
    finally:
        if lease is not None:
            try:
                lease.close()
            except Exception:
                pass
    if not os.path.isfile(raw):
        raise PresenterError(f"{engine}: produced no clip")
    # Exactly the scene's length, at the canvas fps, whatever the engine returned.
    out = os.path.join(workdir, f"pres{index}.mp4")
    rnd.hold_to(raw, out, duration=duration, fps=fps, width=width, height=height,
                cancel=cancel, log=log)
    return {"path": out, "engine": engine}
