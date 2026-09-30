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


def run(args: List[str], *, cancel: Optional[threading.Event] = None,
        timeout: float = 7200.0, log: Optional[Callable[[str], None]] = None) -> None:
    """Run one ffmpeg command, honouring cancellation while it runs."""
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    cmd = [media.ffmpeg_bin(), "-hide_banner", "-nostdin", "-y",
           "-threads", _thread_cap()] + list(args)
    if log:
        log("ffmpeg " + " ".join(shlex.quote(a) for a in args[:40])
            + (" …" if len(args) > 40 else ""))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
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
    """One scene's narration, padded (or trimmed) to the scene's exact length."""
    run(["-i", src, "-af",
         f"aresample=48000,apad,atrim=0:{max(0.02, duration):.3f},asetpts=N/SR/TB",
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
