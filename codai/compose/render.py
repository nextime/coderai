# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""The ffmpeg pipeline: pieces in, one uploadable file out.

Shape of the render, and why it is shaped this way:

1. **one normalised clip per visual** — same size, same fps, same pixel format,
   exact duration. Building intermediates rather than one enormous filter graph
   costs a re-encode but buys three things that matter more: a visual that fails
   to decode can be replaced by a gradient without losing the render, progress
   is reportable per visual, and the ffmpeg command stays a length a human can
   read in a log.
2. **join** — concat demuxer when there is no transition (stream copy, free), an
   ``xfade`` chain otherwise. Clips for a transition are built one transition
   longer so the overlap is absorbed and the timeline still matches the audio.
3. **audio** — narration per scene padded to the scene's length and concatenated,
   loudness-normalised; music looped, trimmed, faded and ducked under the
   narration with ``sidechaincompress``; the two mixed.
4. **burn** — captions (libass) and overlays in one pass with the final encode:
   H.264 High + AAC + ``+faststart``, which is what TikTok, Reels, Shorts and
   Facebook accept without re-processing.

Every ffmpeg invocation goes through :func:`run` so a cancelled job kills the
child it is waiting on instead of finishing a five-minute encode nobody wants.
"""

import os
import shlex
import subprocess
import threading
from typing import Callable, List, Optional

from codai.compose import media

# A cheap, visually lossless intermediate: the final encode is the one that
# matters, and veryfast/crf 18 keeps a 20-visual reel from spending its time in
# x264 twice.
_INTER_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


class Cancelled(Exception):
    """The job was cancelled while a child process was running."""


class RenderError(Exception):
    """ffmpeg refused to do something we asked for."""


def _thread_cap() -> str:
    """Leave the machine usable: never more than half its cores per child."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 4
    return str(max(1, min(8, n // 2 or 1)))


def _die_with_parent() -> None:
    """Ask the kernel to kill this child when its parent dies (Linux only).

    Nothing else does. ``run`` kills the child on cancel and on timeout, but
    both of those need this process to still be alive to act: if it is killed
    (SIGKILL, an OOM, a container stop, a test run interrupted), no finally
    block runs and ffmpeg is simply reparented to init and left going. An
    abandoned ffmpeg is not idle either -- a filter graph that never reaches
    EOF spins at 100% CPU for ever. Measured 2026-10-28: two orphans from
    interrupted test runs the day before had burned 15 h and 11 h of CPU
    between them, on a host whose owner had no idea what they were.

    PR_SET_PDEATHSIG makes that unrepresentable rather than merely unlikely.
    Best effort: it is a no-op where prctl is unavailable.
    """
    try:
        import ctypes
        import signal as _signal
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            PR_SET_PDEATHSIG, _signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass


def run(args: List[str], *, cancel: Optional[threading.Event] = None,
        timeout: float = 7200.0, log: Optional[Callable[[str], None]] = None) -> None:
    """Run one ffmpeg command, honouring cancellation while it runs.

    The default ``timeout`` is a backstop for a long video encode, not a
    sensible bound for every call -- an audio mix of a few seconds that is
    still running after a minute is wedged, not slow. Callers that know the
    shape of their work pass their own.
    """
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    cmd = [media.ffmpeg_bin(), "-hide_banner", "-nostdin", "-y",
           "-threads", _thread_cap()] + list(args)
    if log:
        log("ffmpeg " + " ".join(shlex.quote(a) for a in args[:40])
            + (" …" if len(args) > 40 else ""))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            preexec_fn=_die_with_parent)
    waited = 0.0
    while True:
        try:
            # A short poll so a cancel lands in a quarter of a second rather
            # than at the end of a long encode.
            _, err = proc.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            waited += 0.25
            if cancel is not None and cancel.is_set():
                proc.kill()
                proc.communicate()
                raise Cancelled()
            if waited > timeout:
                proc.kill()
                proc.communicate()
                raise RenderError(f"ffmpeg timed out after {timeout:.0f}s")
    if proc.returncode != 0:
        tail = (err or b"").decode(errors="replace").strip().splitlines()[-12:]
        raise RenderError("ffmpeg failed (rc=%d): %s" % (proc.returncode, " | ".join(tail)))


# ------------------------------------------------------------------- visuals
def fit_filter(width: int, height: int, fit: str) -> str:
    """Scale a source of any aspect to the canvas.

    ``cover`` crops to fill (what a reel wants), ``contain`` letterboxes onto
    black, ``blur_pad`` letterboxes onto a blurred blow-up of the same frame —
    the look every vertical repost of a horizontal clip uses."""
    w, h = int(width), int(height)
    fit = (fit or "cover").lower()
    if fit == "contain":
        return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
    if fit == "blur_pad":
        return (f"split=2[bg][fg];"
                f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},gblur=sigma=28[bgb];"
                f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgs];"
                f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1")
    return (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},setsar=1")


def ken_burns_filter(width: int, height: int, fps: int, duration: float,
                     zoom_from: float = 1.0, zoom_to: float = 1.12,
                     pan: str = "auto", index: int = 0) -> str:
    """A slow zoom/pan over a still (``zoompan``).

    The still is first blown up and cropped to the canvas aspect so the move
    never reveals an edge, then zoompan walks the zoom from ``zoom_from`` to
    ``zoom_to`` across the clip. ``pan: auto`` alternates direction per image so
    consecutive stills do not drift the same way."""
    w, h = int(width), int(height)
    frames = max(2, int(round(max(0.04, duration) * max(1, fps))))
    z0 = max(1.0, float(zoom_from or 1.0))
    z1 = max(1.0, float(zoom_to if zoom_to is not None else 1.12))
    if abs(z1 - z0) < 1e-3:
        z1 = z0 + 0.0001                      # zoompan needs a direction
    step = (z1 - z0) / frames
    zexpr = (f"min(max(zoom,{z0:.5f})+{step:.7f},{max(z0, z1):.5f})" if step > 0
             else f"max(max(zoom,{z0:.5f}){step:.7f},{min(z0, z1):.5f})")
    p = (pan or "auto").lower()
    if p == "auto":
        p = ("right", "left", "up", "down")[index % 4]
    # Pan expressions move the crop window across the zoomed frame.
    if p == "left":
        x, y = "(iw-iw/zoom)*(1-on/%d)" % frames, "(ih-ih/zoom)/2"
    elif p == "right":
        x, y = "(iw-iw/zoom)*(on/%d)" % frames, "(ih-ih/zoom)/2"
    elif p == "up":
        x, y = "(iw-iw/zoom)/2", "(ih-ih/zoom)*(1-on/%d)" % frames
    elif p == "down":
        x, y = "(iw-iw/zoom)/2", "(ih-ih/zoom)*(on/%d)" % frames
    else:
        x, y = "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    # Oversample before zoompan: it samples the INPUT, so a canvas-sized input
    # visibly softens as it zooms in.
    pre = (f"scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,"
           f"crop={w * 2}:{h * 2}")
    return (f"{pre},zoompan=z='{zexpr}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={fps},"
            f"setsar=1,format=yuv420p")


def gradient_png(path: str, width: int, height: int, colours: List[str]) -> str:
    """A vertical gradient as a PNG — the fallback for a visual that fails, and
    a visual type of its own. Drawn with PIL rather than ffmpeg's ``gradients``
    filter: one less filter whose options changed between ffmpeg versions."""
    from PIL import Image
    cols = [c for c in (colours or []) if isinstance(c, str) and c.strip()] or \
        ["#6D28D9", "#DB2777"]

    def _rgb(c: str):
        h = c.strip().lstrip("#")
        if len(h) == 3:
            h = "".join(x * 2 for x in h)
        if len(h) != 6:
            h = "6D28D9"
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    stops = [_rgb(c) for c in cols]
    if len(stops) == 1:
        stops = stops * 2
    img = Image.new("RGB", (max(2, int(width)), max(2, int(height))))
    px = img.load()
    h = img.height
    segs = len(stops) - 1
    for y in range(h):
        pos = (y / max(1, h - 1)) * segs
        i = min(segs - 1, int(pos))
        f = pos - i
        a, b = stops[i], stops[i + 1]
        row = (int(a[0] + (b[0] - a[0]) * f), int(a[1] + (b[1] - a[1]) * f),
               int(a[2] + (b[2] - a[2]) * f))
        for x in range(img.width):
            px[x, y] = row
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)
    return path


def build_visual(visual: dict, out_path: str, *, width: int, height: int, fps: int,
                 duration: float, fit: str = "cover", index: int = 0,
                 workdir: str = "", cancel: Optional[threading.Event] = None,
                 log: Optional[Callable[[str], None]] = None) -> None:
    """Render one visual to a clip of exactly ``duration`` seconds."""
    vtype = (visual.get("type") or "").lower()
    dur = max(0.08, float(duration))
    common_out = ["-t", f"{dur:.3f}", "-r", str(fps), "-an"] + _INTER_ARGS + [out_path]

    if vtype == "color":
        colour = visual.get("color") or "#111827"
        args = ["-f", "lavfi", "-i", f"color=c={colour}:s={width}x{height}:r={fps}",
                "-vf", "format=yuv420p,setsar=1"] + common_out
        run(args, cancel=cancel, log=log)
        return

    if vtype == "gradient":
        png = gradient_png(os.path.join(workdir or ".", f"grad-{index}.png"),
                           width, height, visual.get("colors") or visual.get("colours"))
        args = ["-loop", "1", "-i", png,
                "-vf", ken_burns_filter(width, height, fps, dur, 1.0, 1.06, "auto", index)
                ] + common_out
        run(args, cancel=cancel, log=log)
        return

    src = visual.get("_path")
    if not src:
        raise RenderError("visual has no resolved source")

    if vtype == "image":
        kb = visual.get("ken_burns")
        if kb is False:
            vf = fit_filter(width, height, fit) + ",format=yuv420p"
        else:
            kb = kb if isinstance(kb, dict) else {}
            vf = ken_burns_filter(width, height, fps, dur,
                                  kb.get("zoom_from", 1.0), kb.get("zoom_to", 1.12),
                                  kb.get("pan", "auto"), index)
        run(["-loop", "1", "-i", src, "-vf", vf] + common_out, cancel=cancel, log=log)
        return

    # video: seek, loop if it is shorter than its slot, normalise.
    trim = max(0.0, float(visual.get("trim_start") or 0.0))
    info = visual.get("_probe") or {}
    src_dur = float(info.get("duration") or 0.0)
    pre = []
    if src_dur and (src_dur - trim) < dur - 0.02:
        pre += ["-stream_loop", "-1"]          # shorter than its slot → loop
    if trim > 0:
        pre += ["-ss", f"{trim:.3f}"]
    vf = fit_filter(width, height, fit) + f",fps={fps},format=yuv420p"
    run(pre + ["-i", src, "-vf", vf] + common_out, cancel=cancel, log=log)


def still_video(image: str, out_path: str, *, duration: float, width: int, height: int,
                fps: int, cancel=None, log=None) -> str:
    """A motionless clip of a portrait — what a video-driven lip-sync engine needs
    when there is no footage to drive (and what a generated shot falls back to).
    No Ken Burns: a presenter's head must not drift while it talks."""
    run(["-loop", "1", "-i", image, "-t", f"{max(0.08, duration):.3f}",
         "-vf", fit_filter(width, height, "cover") + f",fps={fps},format=yuv420p",
         "-r", str(fps), "-an"] + _INTER_ARGS + [out_path], cancel=cancel, log=log)
    return out_path


def hold_to(src: str, out_path: str, *, duration: float, fps: int,
            width: Optional[int] = None, height: Optional[int] = None,
            fit: str = "cover", cancel=None, log=None) -> str:
    """Make a clip exactly ``duration`` long, and the canvas's size when asked.

    A lip-sync engine returns the length of the audio it was given and whatever
    frame size it likes (SadTalker squares its output), while the scene is the
    narration plus its padding at the canvas size — so: hold the final frame (what
    a presenter does when they stop talking), trim if longer, and scale/crop to
    the canvas."""
    chain = [f"fps={fps}"]
    if width and height:
        chain.append(fit_filter(int(width), int(height), fit))
    chain += [f"tpad=stop_mode=clone:stop_duration={max(0.05, duration):.3f}",
              f"trim=0:{max(0.08, duration):.3f}", "setpts=PTS-STARTPTS",
              "format=yuv420p"]
    run(["-i", src, "-vf", ",".join(chain), "-an", "-r", str(fps)]
        + _INTER_ARGS + [out_path], cancel=cancel, log=log)
    return out_path


# ------------------------------------------------------------ presenter layout
#: The presenter bubble keeps the captions' safe area: clear of the platform UI.
SAFE_X = 0.06
SAFE_Y = 0.10


def bubble_mask(path: str, w: int, h: int, shape: str = "circle",
                border_width: int = 0, border_colour: str = "#FFFFFF") -> tuple:
    """(mask, ring) PNGs for a picture-in-picture bubble.

    The mask is greyscale — white where the presenter shows — and becomes the
    clip's alpha; the ring is an RGBA outline drawn on top afterwards, so the
    border is not eaten by the same mask that rounds the corners."""
    from PIL import Image, ImageDraw
    w, h = max(2, int(w)), max(2, int(h))
    shape = (shape or "circle").lower()
    radius = int(min(w, h) * 0.18)
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    if shape == "circle":
        md.ellipse((0, 0, w - 1, h - 1), fill=255)
    elif shape in ("rounded", "round"):
        md.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    else:
        md.rectangle((0, 0, w - 1, h - 1), fill=255)
    mask_path = path + ".mask.png"
    mask.save(mask_path)

    ring_path = None
    if border_width and border_width > 0:
        col = (border_colour or "#FFFFFF").lstrip("#")
        if len(col) == 3:
            col = "".join(c * 2 for c in col)
        rgb = tuple(int(col[i:i + 2], 16) for i in (0, 2, 4)) if len(col) == 6 else (255, 255, 255)
        ring = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        rd = ImageDraw.Draw(ring)
        bw = max(1, int(border_width))
        box = (bw // 2, bw // 2, w - 1 - bw // 2, h - 1 - bw // 2)
        if shape == "circle":
            rd.ellipse(box, outline=rgb + (255,), width=bw)
        elif shape in ("rounded", "round"):
            rd.rounded_rectangle(box, radius=radius, outline=rgb + (255,), width=bw)
        else:
            rd.rectangle(box, outline=rgb + (255,), width=bw)
        ring_path = path + ".ring.png"
        ring.save(ring_path)
    return mask_path, ring_path


def bubble_geometry(width: int, height: int, size: float, position: str,
                    shape: str = "circle") -> tuple:
    """(w, h, x, y) of the bubble inside the safe area."""
    size = max(0.1, min(0.95, float(size or 0.38)))
    bw = int(round(width * size / 2) * 2)
    bh = bw if (shape or "circle").lower() in ("circle", "rounded", "round")         else int(round(bw * height / max(1, width) / 2) * 2)
    bh = min(bh, int(height * 0.8))
    mx, my = int(width * SAFE_X), int(height * SAFE_Y)
    pos = (position or "bottom-right").lower().replace("_", "-")
    vert, _, horiz = pos.partition("-") if "-" in pos else (pos, "", "center")
    x = {"left": mx, "center": (width - bw) // 2, "centre": (width - bw) // 2,
         "right": width - bw - mx}.get(horiz, width - bw - mx)
    y = {"top": my, "center": (height - bh) // 2, "centre": (height - bh) // 2,
         "middle": (height - bh) // 2,
         "bottom": height - bh - my}.get(vert, height - bh - my)
    return bw, bh, max(0, x), max(0, y)


def composite_pip(base: str, presenter: str, out_path: str, *, width: int, height: int,
                  fps: int, size: float = 0.38, position: str = "bottom-right",
                  shape: str = "circle", border: Optional[dict] = None,
                  alpha_video: Optional[str] = None,
                  workdir: str = "", cancel=None, log=None) -> str:
    """The presenter as a bubble over the scene's visuals.

    ``alpha_video`` is a per-frame mask (see ``codai/compose/matting.py``) and
    takes the shape mask's place: the person is cut out instead of a circle of
    the room they were filmed in. ``size`` and ``position`` still place them,
    but a border would outline nothing, so it is not drawn."""
    bw, bh, x, y = bubble_geometry(width, height, size, position, shape)
    border = border or {}
    if alpha_video:
        mask, ring = alpha_video, None
        mask_filter = (f"[2:v]scale={bw}:{bh}:force_original_aspect_ratio=increase,"
                       f"crop={bw}:{bh},format=gray[pm]")
    else:
        mask, ring = bubble_mask(
            os.path.join(workdir or ".", f"bubble-{bw}x{bh}"), bw, bh,
            shape, int(border.get("width") or 0),
            border.get("color") or border.get("colour") or "#FFFFFF")
        mask_filter = "[2:v]format=gray[pm]"
    inputs = ["-i", base, "-i", presenter, "-i", mask]
    graph = [f"[1:v]scale={bw}:{bh}:force_original_aspect_ratio=increase,"
             f"crop={bw}:{bh},format=rgba[pv]",
             mask_filter,
             "[pv][pm]alphamerge[pa]"]
    top = "pa"
    if ring:
        inputs += ["-i", ring]
        graph.append("[pa][3:v]overlay=0:0[pb]")
        top = "pb"
    graph.append(f"[0:v][{top}]overlay={x}:{y}:format=auto,format=yuv420p,fps={fps}[vout]")
    run(inputs + ["-filter_complex", ";".join(graph), "-map", "[vout]", "-an"]
        + _INTER_ARGS + [out_path], cancel=cancel, log=log)
    return out_path


def composite_split(base: str, presenter: str, out_path: str, *, width: int, height: int,
                    fps: int, presenter_half: str = "bottom",
                    alpha_video: Optional[str] = None, cancel=None,
                    log=None) -> str:
    """Presenter and B-roll stacked, each filling its half.

    With ``alpha_video`` the presenter's half is the person cut out **over** the
    B-roll rather than over their own background, so both halves show the
    scene's visuals and only one of them has someone standing in it."""
    half = int(height / 2 / 2) * 2
    fill = (f"scale={width}:{half}:force_original_aspect_ratio=increase,"
            f"crop={width}:{half},setsar=1")
    presenter_top = (presenter_half or "bottom").lower() == "top"
    if alpha_video:
        graph = [f"[0:v]{fill},split=2[bplain][bunder]",
                 f"[1:v]{fill},format=rgba[pv]",
                 f"[2:v]{fill},format=gray[pm]",
                 "[pv][pm]alphamerge[pa]",
                 "[bunder][pa]overlay=0:0[ph]"]
        top, bottom = ("ph", "bplain") if presenter_top else ("bplain", "ph")
        inputs = ["-i", base, "-i", presenter, "-i", alpha_video]
    else:
        top_in, bottom_in = (1, 0) if presenter_top else (0, 1)
        graph = [f"[{top_in}:v]{fill}[t]", f"[{bottom_in}:v]{fill}[b]"]
        top, bottom = "t", "b"
        inputs = ["-i", base, "-i", presenter]
    graph.append(f"[{top}][{bottom}]vstack=inputs=2,pad={width}:{height}:0:0,"
                 f"format=yuv420p,fps={fps}[vout]")
    run(inputs + ["-filter_complex", ";".join(graph),
                  "-map", "[vout]", "-an"] + _INTER_ARGS + [out_path],
        cancel=cancel, log=log)
    return out_path


# --------------------------------------------------------------------- join
def concat_clips(clips: List[str], out_path: str, workdir: str,
                 cancel: Optional[threading.Event] = None,
                 log: Optional[Callable[[str], None]] = None) -> None:
    """Join identically-encoded clips without re-encoding."""
    lst = os.path.join(workdir, "concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for c in clips:
            f.write("file '%s'\n" % os.path.abspath(c).replace("'", "'\\''"))
    run(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out_path],
        cancel=cancel, log=log)


_XFADE = {"fade": "fadeblack", "crossfade": "fade", "slide": "slideleft",
          "wipe": "wipeleft", "dissolve": "dissolve"}


def xfade_clips(clips: List[str], durations: List[float], out_path: str, *,
                kind: str, dur: float, fps: int,
                cancel: Optional[threading.Event] = None,
                log: Optional[Callable[[str], None]] = None) -> None:
    """Join clips with a transition, keeping the total length intact.

    Each clip was built ``dur`` seconds longer than its slot, so an ``xfade``
    that consumes ``dur`` of overlap per join lands the timeline back where the
    audio expects it."""
    trans = _XFADE.get((kind or "crossfade").lower(), "fade")
    args = []
    for c in clips:
        args += ["-i", c]
    graph = []
    prev = "0:v"
    offset = 0.0
    for i in range(1, len(clips)):
        offset += max(0.05, durations[i - 1] - dur)
        out = f"x{i}"
        graph.append(f"[{prev}][{i}:v]xfade=transition={trans}:duration={dur:.3f}:"
                     f"offset={offset:.3f}[{out}]")
        prev = out
    graph.append(f"[{prev}]fps={fps},format=yuv420p[vout]")
    run(args + ["-filter_complex", ";".join(graph), "-map", "[vout]"]
        + _INTER_ARGS + [out_path], cancel=cancel, log=log)


# -------------------------------------------------------------------- audio
def silence(path: str, duration: float, cancel=None, log=None) -> str:
    run(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
         "-t", f"{max(0.02, duration):.3f}", "-c:a", "pcm_s16le", path],
        cancel=cancel, log=log)
    return path


def pad_to(src: str, out_path: str, duration: float, cancel=None, log=None) -> str:
    """One scene's narration, padded (or trimmed) to the scene's exact length.

    The pad is bounded with ``whole_dur`` rather than left to ``atrim`` to stop.
    A bare ``apad`` generates silence for ever and relies on something
    downstream ending the stream; with more than one input feeding the pad
    through ``amix`` that termination is a race, and when it is lost ffmpeg
    never reaches EOF and spins at 100% CPU with the output half-written.
    Measured on this exact graph (two 1 s inputs, ``amix``, 4 s trim): 4 hangs
    in 10 runs, and two orphans on the host had burned 14 h and 10 h of CPU
    before anyone noticed. ``-threads 1`` and ``-filter_threads 1`` do NOT fix
    it (3/12 and 7/12); bounding the pad does (0/12).
    """
    run(["-i", src, "-af",
         f"aresample=48000,apad=whole_dur={max(0.02, duration):.3f},"
         f"atrim=0:{max(0.02, duration):.3f},asetpts=N/SR/TB",
         "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", out_path],
        cancel=cancel, log=log)
    return out_path


def concat_audio(parts: List[str], out_path: str, workdir: str,
                 cancel=None, log=None) -> str:
    if len(parts) == 1:
        run(["-i", parts[0], "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", out_path],
            cancel=cancel, log=log)
        return out_path
    lst = os.path.join(workdir, "concat-audio.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '%s'\n" % os.path.abspath(p).replace("'", "'\\''"))
    run(["-f", "concat", "-safe", "0", "-i", lst,
         "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", out_path], cancel=cancel, log=log)
    return out_path


def loudnorm(src: str, out_path: str, lufs: float = -16.0, cancel=None, log=None) -> str:
    """Bring narration to about ``lufs`` so every reel is as loud as the last."""
    run(["-i", src, "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11",
         "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", out_path], cancel=cancel, log=log)
    return out_path


def build_music_bed(src: str, out_path: str, *, total: float, volume: float = 0.14,
                    loop: bool = True, fade_in: float = 1.0, fade_out: float = 2.0,
                    cancel=None, log=None) -> str:
    """Music looped/trimmed to the video's length, faded at both ends."""
    pre = ["-stream_loop", "-1"] if loop else []
    fo_start = max(0.0, total - max(0.0, fade_out))
    af = [f"aresample=48000", f"atrim=0:{max(0.1, total):.3f}", "asetpts=N/SR/TB",
          f"volume={max(0.0, float(volume)):.4f}"]
    if fade_in and fade_in > 0:
        af.append(f"afade=t=in:st=0:d={float(fade_in):.3f}")
    if fade_out and fade_out > 0:
        af.append(f"afade=t=out:st={fo_start:.3f}:d={float(fade_out):.3f}")
    run(pre + ["-i", src, "-af", ",".join(af), "-ac", "2", "-ar", "48000",
               "-c:a", "pcm_s16le", out_path], cancel=cancel, log=log)
    return out_path


def place_at(placements: List[tuple], out_path: str, *, duration: float,
             cancel=None, log=None) -> str:
    """One track holding each clip at its own offset, silence in between.

    ``placements`` is ``[(path, start_seconds), ...]``. Built for one speaker in
    a dialogue: their lines where they fall in the scene, nothing where
    somebody else is talking.

    That shape is what lets a multi-speaker scene cost one lip-sync run per
    SPEAKER rather than one per line. A sync engine drives the mouth from the
    audio it is given, so a track that is silent while the other character
    speaks produces a closed mouth there -- which is what a person listening
    looks like, and what a per-line approach has to stitch together by hand.
    """
    if not placements:
        return silence(out_path, duration, cancel=cancel, log=log)
    dur = max(0.02, float(duration))
    # The track's length comes from a silent base input of exactly `dur`, mixed
    # under the placements, and `amix=duration=first` ends the mix with it.
    #
    # It is NOT padded up to length with `apad`. A pad has to be ended by
    # something downstream, and when it is fed through `amix` that handoff is a
    # race: a bare `apad` loses it by generating silence for ever, so ffmpeg
    # never reaches EOF and spins at 100% CPU with the output half-written (two
    # orphans on a dev host had burned 14 h and 10 h of CPU before anyone
    # noticed; 4 hangs in 10 runs of this graph). `apad=whole_dur` stops the
    # hang but loses the same race the other way, silently emitting a track one
    # clip short -- 3 s where 4 s was asked for, in 3 of 14 runs, with no error
    # anywhere. Neither `-threads 1` nor `-filter_threads 1` helps (3/12, 7/12).
    # A base input that is already the right length has no handoff to lose.
    inputs: List[str] = ["-f", "lavfi", "-t", f"{dur:.3f}",
                         "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
    graph: List[str] = []
    labels: List[str] = ["[0:a]"]
    for i, (path, start) in enumerate(placements, start=1):
        inputs += ["-i", path]
        ms = max(0, int(round(float(start) * 1000)))
        graph.append(f"[{i}:a]aresample=48000,adelay={ms}|{ms}[d{i}]")
        labels.append(f"[d{i}]")
    graph.append("".join(labels)
                 + f"amix=inputs={len(labels)}:duration=first:"
                   f"dropout_transition=0:normalize=0,"
                   f"atrim=0:{dur:.3f},asetpts=N/SR/TB[aout]")
    # Mixing a few seconds of PCM runs in milliseconds, so a minute of it means
    # wedged, not slow -- no reason to sit under the two-hour video backstop.
    # Generous even so: 20x the track plus a two-minute floor.
    run(inputs + ["-filter_complex", ";".join(graph), "-map", "[aout]",
                  "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", out_path],
        cancel=cancel, log=log, timeout=max(120.0, 20.0 * dur))
    return out_path


def cut_segment(src: str, out_path: str, *, start: float, duration: float,
                fps: int, width: Optional[int] = None,
                height: Optional[int] = None, cancel=None, log=None) -> str:
    """``duration`` seconds of ``src`` from ``start``, at the canvas size.

    Re-encoded rather than stream-copied: the cut has to land on the exact
    frame, and a copy would snap to the nearest keyframe and drift the dialogue
    against its own audio.
    """
    chain = [f"fps={fps}"]
    if width and height:
        chain.append(fit_filter(int(width), int(height), "cover"))
    chain += [f"tpad=stop_mode=clone:stop_duration={max(0.05, duration):.3f}",
              f"trim=0:{max(0.08, duration):.3f}", "setpts=PTS-STARTPTS",
              "format=yuv420p"]
    run(["-ss", f"{max(0.0, float(start)):.3f}", "-i", src,
         "-vf", ",".join(chain), "-an", "-r", str(fps)]
        + _INTER_ARGS + [out_path], cancel=cancel, log=log)
    return out_path


def composite_row(clips: List[str], out_path: str, *, width: int, height: int,
                  fps: int, cancel=None, log=None) -> str:
    """Several speakers side by side, each filling its own column.

    Each column is cropped to fill rather than letterboxed, because these are
    portraits: a face with black bars beside it reads as a mistake, and a face
    cropped at the shoulders reads as framing.
    """
    n = max(1, len(clips))
    if n == 1:
        return hold_to(clips[0], out_path, duration=0.0, fps=fps,
                       width=width, height=height, cancel=cancel, log=log)
    col = int(width / n / 2) * 2
    fill = (f"scale={col}:{height}:force_original_aspect_ratio=increase,"
            f"crop={col}:{height},setsar=1")
    inputs: List[str] = []
    graph: List[str] = []
    labels: List[str] = []
    for i, c in enumerate(clips):
        inputs += ["-i", c]
        graph.append(f"[{i}:v]{fill}[c{i}]")
        labels.append(f"[c{i}]")
    graph.append("".join(labels) + f"hstack=inputs={n},"
                 f"pad={width}:{height}:0:0,format=yuv420p,fps={fps}[vout]")
    run(inputs + ["-filter_complex", ";".join(graph), "-map", "[vout]", "-an"]
        + _INTER_ARGS + [out_path], cancel=cancel, log=log)
    return out_path


def clip_audio(src: str, out_path: str, *, duration: float, trim_start: float = 0.0,
               volume: float = 1.0, loop: bool = False, cancel=None,
               log=None) -> str:
    """One visual's OWN sound, cut to its slot in the scene.

    The video track of a visual is already trimmed, looped and padded to the
    scene's length by :func:`build_visual`; this does the same to its audio so
    the two stay in step. ``loop`` must match what the video did, or the sound
    drifts out of sync with the picture it belongs to.

    Always 48 kHz stereo PCM, like every other audio part, so the scene parts
    can be concatenated without a resample surprise in the middle.
    """
    pre = ["-stream_loop", "-1"] if loop else []
    if trim_start > 0:
        pre += ["-ss", f"{float(trim_start):.3f}"]
    af = ["aresample=48000",
          f"apad=whole_dur={max(0.02, float(duration)):.3f}",
          f"atrim=0:{max(0.02, float(duration)):.3f}", "asetpts=N/SR/TB"]
    v = max(0.0, float(volume))
    if abs(v - 1.0) > 1e-6:
        af.append(f"volume={v:.4f}")
    run(pre + ["-i", src, "-vn", "-af", ",".join(af),
               "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", out_path],
        cancel=cancel, log=log)
    return out_path


def has_audio_stream(src: str) -> bool:
    """Whether a file carries an audio stream at all.

    Asked before extracting, because ``keep_audio`` on silent footage would
    otherwise fail the render over something the caller cannot see from the
    file name. A probe failure answers "no": the extraction would fail anyway.
    """
    try:
        out = subprocess.run(
            [media._ffprobe_bin(), "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", src],
            capture_output=True, text=True, timeout=60)
    except Exception:
        return False
    return bool((out.stdout or "").strip())


def mix_voice_clip(voice: Optional[str], clip: Optional[str], out_path: str, *,
                   duration: float, duck: bool = True, cancel=None,
                   log=None) -> str:
    """One scene's narration and its clip's sound, as a single track.

    Mixed HERE, per scene, rather than as a third input to the final mix, for
    two reasons: the clip's sound belongs to one scene and must not bleed into
    the next, and folding it into the narration bed means the music ducking
    downstream keys on it too -- which is what "treat clip sound like
    narration" asks for, with no second sidechain.

    With ``duck`` the clip is pushed down while the voice is speaking and comes
    back up between lines, so narration stays intelligible over a noisy shot.
    Without it the two are simply summed at their given levels.
    """
    if not clip:
        if voice:
            return pad_to(voice, out_path, duration, cancel=cancel, log=log)
        return silence(out_path, duration, cancel=cancel, log=log)
    if not voice:
        run(["-i", clip, "-af",
             f"aresample=48000,apad=whole_dur={max(0.02, duration):.3f},"
             f"atrim=0:{max(0.02, duration):.3f},asetpts=N/SR/TB",
             "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", out_path],
            cancel=cancel, log=log)
        return out_path
    # The scene's exact length is held INSIDE the graph. It cannot be an `-af`
    # on the output: ffmpeg refuses simple and complex filtering on the same
    # stream ("Filtergraph ... was specified for a stream fed from a complex
    # filtergraph"), and the mix would never be written at all.
    fit = (f"apad=whole_dur={max(0.02, float(duration)):.3f},"
           f"atrim=0:{max(0.02, float(duration)):.3f},asetpts=N/SR/TB")
    if duck:
        graph = ("[1:a]asplit=2[c1][c2];"
                 "[0:a]asplit=2[v1][v2];"
                 # v2 is only the sidechain key; c1 is what we hear.
                 "[c1][v2]sidechaincompress=threshold=0.04:ratio=6:attack=20:"
                 "release=350:makeup=1[cduck];"
                 "[v1][cduck]amix=inputs=2:duration=longest:dropout_transition=0,"
                 f"alimiter=limit=0.97,{fit}[aout];"
                 "[c2]anullsink")
    else:
        graph = ("[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=0,"
                 f"alimiter=limit=0.97,{fit}[aout]")
    run(["-i", voice, "-i", clip, "-filter_complex", graph, "-map", "[aout]",
         "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", out_path],
        cancel=cancel, log=log)
    return out_path


def mix_narration_music(narration: str, music: Optional[str], out_path: str, *,
                        duck: bool = True, cancel=None, log=None) -> str:
    """Narration over music, with the music ducked under speech when asked."""
    if not music:
        run(["-i", narration, "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", out_path],
            cancel=cancel, log=log)
        return out_path
    if duck:
        graph = ("[1:a]asplit=2[m1][m2];"
                 "[0:a]asplit=2[n1][n2];"
                 # n2 is only the sidechain key; m1 is what we hear.
                 "[m1][n2]sidechaincompress=threshold=0.05:ratio=8:attack=20:"
                 "release=400:makeup=1[mduck];"
                 "[n1][mduck]amix=inputs=2:duration=first:dropout_transition=0,"
                 "alimiter=limit=0.97[aout];"
                 "[m2]anullsink")
    else:
        graph = ("[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0,"
                 "alimiter=limit=0.97[aout]")
    run(["-i", narration, "-i", music, "-filter_complex", graph, "-map", "[aout]",
         "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", out_path], cancel=cancel, log=log)
    return out_path


# ------------------------------------------------------- overlays + captions
def _pos_expr(position: str, pad_x: str, pad_y: str) -> tuple:
    """Nine-box position → (x, y) expressions for overlay/drawtext."""
    p = (position or "bottom-right").lower().replace("_", "-")
    horiz = {"left": pad_x, "center": "(W-w)/2", "centre": "(W-w)/2",
             "right": f"W-w-{pad_x}"}
    vert = {"top": pad_y, "center": "(H-h)/2", "centre": "(H-h)/2",
            "middle": "(H-h)/2", "bottom": f"H-h-{pad_y}"}
    if "-" in p:
        v, _, hh = p.partition("-")
    else:
        v, hh = ("center", p) if p in horiz else (p, "center")
    return horiz.get(hh, horiz["right"]), vert.get(v, vert["bottom"])


def _esc_text(text: str) -> str:
    """Escape for drawtext's expression parser."""
    return ((text or "")
            .replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")
            .replace("%", "\\%").replace(",", "\\,").replace("[", "\\[")
            .replace("]", "\\]").replace(";", "\\;"))


def final_render(video: str, audio: str, out_path: str, *, width: int, height: int,
                 fps: int, total: float, ass_file: Optional[str] = None,
                 overlays: Optional[List[dict]] = None, crf: int = 20,
                 audio_bitrate: str = "192k", video_codec: str = "h264",
                 fonts_dir: Optional[str] = None, cancel=None, log=None) -> None:
    """Burn captions and overlays and encode the deliverable."""
    inputs = ["-i", video, "-i", audio]
    img_overlays = [o for o in (overlays or [])
                    if (o.get("type") or "image").lower() == "image" and o.get("_path")]
    for o in img_overlays:
        inputs += ["-i", o["_path"]]

    graph = [f"[0:v]fps={fps},format=yuv420p[base]"]
    cur = "base"
    if ass_file:
        opts = f"filename={ass_file}"
        if fonts_dir:
            opts += f":fontsdir={fonts_dir}"
        graph.append(f"[{cur}]ass={opts}[capd]")
        cur = "capd"
    # Images first (a logo belongs under text), then text.
    for n, o in enumerate(img_overlays):
        idx = 2 + n
        w = max(0.01, min(1.0, float(o.get("width") or 0.15)))
        alpha = max(0.0, min(1.0, float(o.get("opacity") if o.get("opacity") is not None else 1.0)))
        x, y = _pos_expr(o.get("position"), f"{int(width * 0.05)}", f"{int(height * 0.04)}")
        graph.append(f"[{idx}:v]scale={int(width * w)}:-1,format=rgba,"
                     f"colorchannelmixer=aa={alpha:.3f}[ov{n}]")
        graph.append(f"[{cur}][ov{n}]overlay={x}:{y}[ovd{n}]")
        cur = f"ovd{n}"
    for n, o in enumerate(overlays or []):
        if (o.get("type") or "image").lower() != "text":
            continue
        text = _esc_text(o.get("text") or "")
        if not text:
            continue
        size = max(8, int(height * max(0.008, min(0.2, float(o.get("size") or 0.025)))))
        alpha = max(0.0, min(1.0, float(o.get("opacity") if o.get("opacity") is not None else 1.0)))
        colour = (o.get("color") or "#FFFFFF").lstrip("#")
        x, y = _pos_expr(o.get("position"), f"{int(width * 0.05)}", f"{int(height * 0.04)}")
        ff = media.font_file(o.get("font") or "", bold=True)
        parts = [f"text='{text}'", f"fontsize={size}",
                 f"fontcolor=0x{colour}@{alpha:.3f}", f"x={x}", f"y={y}",
                 "borderw=%d" % max(1, size // 14), "bordercolor=black@0.6"]
        if ff:
            parts.insert(0, f"fontfile={ff}")
        graph.append(f"[{cur}]drawtext={':'.join(parts)}[txt{n}]")
        cur = f"txt{n}"
    graph.append(f"[{cur}]null[vout]")

    codec = {"h264": "libx264", "avc": "libx264", "h265": "libx265",
             "hevc": "libx265"}.get((video_codec or "h264").lower(), "libx264")
    enc = ["-c:v", codec, "-preset", "medium", "-crf", str(int(crf)),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if codec == "libx264":
        enc += ["-profile:v", "high", "-level", "4.1"]
    run(inputs + ["-filter_complex", ";".join(graph), "-map", "[vout]", "-map", "1:a"]
        + enc + ["-c:a", "aac", "-b:a", str(audio_bitrate or "192k"),
                 "-ar", "48000", "-ac", "2", "-shortest",
                 "-t", f"{max(0.1, total):.3f}", out_path],
        cancel=cancel, log=log)


def thumbnail(video: str, out_path: str, *, at: float, width: int, height: int,
              text: str = "", text_color: str = "#FFFFFF",
              highlight_color: str = "#FFE14D", font: str = "",
              cancel=None, log=None) -> None:
    """A frame from the finished video, optionally with the hook text on it."""
    vf = ["format=yuv420p"]
    if text:
        ff = media.font_file(font or "Montserrat", bold=True)
        size = int(height * 0.085)
        # Break the hook into at most three lines of ~16 characters.
        words, lines, cur = (text or "").split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 16 and cur:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
            if len(lines) == 3:
                break
        if cur and len(lines) < 3:
            lines.append(cur)
        block = len(lines)
        for i, line in enumerate(lines):
            y = f"(h-{block}*{int(size * 1.25)})/2+{i * int(size * 1.25)}"
            parts = [f"text='{_esc_text(line)}'", f"fontsize={size}",
                     f"fontcolor=0x{(highlight_color if i == 0 else text_color).lstrip('#')}",
                     "x=(w-text_w)/2", f"y={y}",
                     f"borderw={max(2, size // 12)}", "bordercolor=black@0.85"]
            if ff:
                parts.insert(0, f"fontfile={ff}")
            vf.append("drawtext=" + ":".join(parts))
    run(["-ss", f"{max(0.0, at):.3f}", "-i", video, "-frames:v", "1",
         "-vf", ",".join(vf), "-q:v", "3", out_path], cancel=cancel, log=log)
