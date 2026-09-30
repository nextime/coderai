# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Media in, facts out: resolving references, probing files, finding fonts.

A compose request names its media the same four ways every other CoderAI
endpoint accepts — an ``http(s)`` URL, a ``/v1/files/<name>`` path served by
this install, a ``data:`` URI, or raw base64 — and the renderer needs a local
path for each. It also needs two facts ffmpeg can tell us (how long a file is,
whether it has audio) and one fontconfig can (which file a font name means).
"""

import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from typing import Optional

_DATA_URI = re.compile(r"^data:([^;,]*)(;base64)?,", re.I)

# What an upload may be. Kept in step with the compose spec's visual/music types.
ALLOWED_MIME_EXT = {
    "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
    "video/x-matroska": ".mkv", "video/mpeg": ".mpg",
    "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav",
    "audio/x-wav": ".wav", "audio/mp4": ".m4a", "audio/m4a": ".m4a",
    "audio/aac": ".m4a", "audio/ogg": ".ogg", "audio/flac": ".flac",
    "audio/x-flac": ".flac",
}
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"RIFF", None),            # WAV or WEBP — decided below
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
    (b"\x1a\x45\xdf\xa3", "video/x-matroska"),   # also webm
    (b"ID3", "audio/mpeg"),
)


class MediaError(Exception):
    """A reference that cannot be turned into a local file."""


# ----------------------------------------------------------------- sniffing
def sniff_mime(data: bytes, hint: str = "") -> str:
    """The media type of these bytes: content first, caller's hint second.

    Uploads arrive with a browser-supplied content type that is often wrong or
    ``application/octet-stream``; the first bytes are not."""
    head = data[:16]
    for magic, mime in _MAGIC:
        if not head.startswith(magic):
            continue
        if magic == b"RIFF":
            if data[8:12] == b"WEBP":
                return "image/webp"
            if data[8:12] == b"WAVE":
                return "audio/wav"
            continue
        if magic == b"\x1a\x45\xdf\xa3":
            return "video/webm" if b"webm" in data[:256].lower() else "video/x-matroska"
        return mime
    if data[4:8] == b"ftyp":
        brand = data[8:12].lower()
        if brand.startswith(b"qt"):
            return "video/quicktime"
        if brand in (b"m4a ", b"m4b "):
            return "audio/m4a"
        return "video/mp4"
    # MP3 without an ID3 header starts with a frame sync.
    if head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    hint = (hint or "").split(";")[0].strip().lower()
    return hint or "application/octet-stream"


def ext_for(mime: str, filename: str = "") -> str:
    """The extension to store a blob under."""
    if mime in ALLOWED_MIME_EXT:
        return ALLOWED_MIME_EXT[mime]
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in set(ALLOWED_MIME_EXT.values()):
        return ext
    return mimetypes.guess_extension(mime or "") or ".bin"


def kind_of(mime: str) -> str:
    """``video`` | ``image`` | ``audio`` | ``other``."""
    m = (mime or "").lower()
    for k in ("video", "image", "audio"):
        if m.startswith(k + "/"):
            return k
    return "other"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------- resolving
def _decode_data_uri(ref: str) -> bytes:
    m = _DATA_URI.match(ref)
    payload = ref[m.end():]
    if m.group(2):
        return base64.b64decode(payload)
    return urllib.parse.unquote_to_bytes(payload)


#: Set by the server (codai/main.py) so composition can write where
#: ``/v1/files`` reads. Kept here rather than in each module so uploads,
#: composition and reference resolution always agree on one directory.
global_file_path: Optional[str] = None


def set_global_file_path(path: str) -> None:
    global global_file_path
    global_file_path = path or None


def files_dir() -> Optional[str]:
    """Where this install writes generated files (``/v1/files`` serves it).

    Whoever set it first wins: the explicit setter, then the already-imported
    api modules (never importing them — that would drag the whole server in)."""
    if global_file_path:
        return global_file_path
    import sys
    for mod in ("codai.api.app", "codai.api.video", "codai.api.images"):
        m = sys.modules.get(mod)
        d = getattr(m, "global_file_path", None) if m is not None else None
        if d:
            return d
    return None


def _files_dir() -> Optional[str]:
    return files_dir()


def local_path_for_files_url(ref: str) -> Optional[str]:
    """If ``ref`` points at a file this install already holds, its path.

    Covers ``/v1/files/x.mp4`` and any absolute URL whose path ends that way —
    a client that got a full URL back from an earlier call sends it verbatim,
    and downloading our own file over the loopback would be silly (and fails
    when the front is bound elsewhere)."""
    if not ref:
        return None
    path = ref
    if "://" in ref:
        try:
            path = urllib.parse.urlsplit(ref).path
        except ValueError:
            return None
    if "/v1/files/" not in path:
        return None
    name = path.split("/v1/files/", 1)[1].split("/")[-1]
    name = urllib.parse.unquote(name)
    base = _files_dir()
    if not (base and name):
        return None
    safe_base = os.path.realpath(base)
    cand = os.path.realpath(os.path.join(base, name))
    if not (cand == safe_base or cand.startswith(safe_base + os.sep)):
        return None
    return cand if os.path.isfile(cand) else None


def fetch(ref: str, timeout: float = 120.0, max_bytes: int = 2 << 30) -> bytes:
    """The bytes behind a media reference (URL / data URI / base64)."""
    if not ref:
        raise MediaError("empty media reference")
    if _DATA_URI.match(ref):
        return _decode_data_uri(ref)
    if ref.startswith(("http://", "https://")):
        req = urllib.request.Request(ref, headers={"User-Agent": "coderai/compose"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise MediaError(f"{ref}: larger than {max_bytes} bytes")
        return data
    if ref.startswith(("/", "./")) or os.path.isabs(ref):
        # A bare path is only ever ours: /v1/files/... (handled by the caller)
        # or a local file, which we do NOT read — a client cannot name paths on
        # this machine.
        raise MediaError(f"{ref}: not a URL, a /v1/files path, or base64 data")
    try:
        return base64.b64decode(ref, validate=True)
    except Exception:
        raise MediaError("media reference is not a URL, a /v1/files path, a data: URI or base64")


def resolve(ref: str, workdir: str, name_hint: str = "media",
            timeout: float = 120.0) -> str:
    """A media reference → a local file path, downloading into ``workdir`` when
    the reference is remote or inline. Files this install already holds are used
    where they are."""
    local = local_path_for_files_url(ref)
    if local:
        return local
    data = fetch(ref, timeout=timeout)
    if not data:
        raise MediaError(f"{name_hint}: empty media")
    mime = sniff_mime(data)
    os.makedirs(workdir, exist_ok=True)
    out = os.path.join(workdir, f"{name_hint}-{sha256_hex(data)[:12]}{ext_for(mime)}")
    if not os.path.isfile(out):
        tmp = out + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, out)
    return out


# ------------------------------------------------------------------ probing
def _ffprobe_bin() -> str:
    return shutil.which("ffprobe") or "ffprobe"


def ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def probe(path: str) -> dict:
    """``{duration, width, height, fps, has_audio, has_video}`` of a media file."""
    cmd = [_ffprobe_bin(), "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", path]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        doc = json.loads(r.stdout or b"{}")
    except Exception as exc:
        raise MediaError(f"ffprobe failed on {os.path.basename(path)}: {exc}")
    if not doc:
        raise MediaError(f"ffprobe returned nothing for {os.path.basename(path)}"
                         f"{': ' + r.stderr.decode(errors='replace')[:200] if r.stderr else ''}")
    out = {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0,
           "has_audio": False, "has_video": False}
    try:
        out["duration"] = float((doc.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        pass
    for st in doc.get("streams") or []:
        ctype = st.get("codec_type")
        if ctype == "video" and not out["has_video"]:
            out["has_video"] = True
            out["width"] = int(st.get("width") or 0)
            out["height"] = int(st.get("height") or 0)
            fr = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1"
            try:
                num, _, den = fr.partition("/")
                out["fps"] = (float(num) / float(den)) if float(den or 0) else 0.0
            except (TypeError, ValueError, ZeroDivisionError):
                out["fps"] = 0.0
            if not out["duration"]:
                try:
                    out["duration"] = float(st.get("duration") or 0.0)
                except (TypeError, ValueError):
                    pass
        elif ctype == "audio":
            out["has_audio"] = True
            if not out["duration"]:
                try:
                    out["duration"] = float(st.get("duration") or 0.0)
                except (TypeError, ValueError):
                    pass
    # An animated GIF/PNG reports a video stream but no usable duration; treat a
    # zero-duration video as a still (the renderer gives stills a Ken Burns move).
    return out


def duration_of(path: str) -> float:
    return float(probe(path).get("duration") or 0.0)


# -------------------------------------------------------------------- fonts
_FONT_CACHE: dict = {}
# Tried in order when a named font is not installed. DejaVu ships with the
# image's base system, so there is always a last resort.
_FALLBACKS = ("Montserrat", "Inter", "DejaVu Sans", "Noto Sans", "Liberation Sans", "Arial")


def font_file(name: str = "", bold: bool = True) -> Optional[str]:
    """The path of a font file for ``name`` (fontconfig), or a fallback.

    Captions are burned by libass, which resolves families through fontconfig
    itself, but ``drawtext`` (overlays, the thumbnail's hook text) needs a file,
    and we want to know whether the asked-for family exists at all so a missing
    font becomes a warning instead of a silently different look."""
    key = (name or "", bool(bold))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    fcmatch = shutil.which("fc-match")
    out = None
    names = [n for n in ([name] if name else []) + list(_FALLBACKS) if n]
    if fcmatch:
        for cand in names:
            pattern = f"{cand}:style={'Bold' if bold else 'Regular'}"
            try:
                r = subprocess.run([fcmatch, "-f", "%{file}|%{family}", pattern],
                                   capture_output=True, timeout=20)
                got = (r.stdout or b"").decode(errors="replace").strip()
            except Exception:
                got = ""
            if not got or "|" not in got:
                continue
            path, family = got.split("|", 1)
            # fontconfig always answers *something*; only accept the answer when
            # it is the family we asked for (substring, case-insensitive).
            asked = cand.replace(" ", "").lower()
            if path and os.path.isfile(path) and asked in family.replace(" ", "").lower():
                out = path
                break
    if out is None:
        for guess in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                      if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if os.path.isfile(guess):
                out = guess
                break
    _FONT_CACHE[key] = out
    return out


def font_installed(name: str) -> bool:
    """Is this exact family available? (A caption font that is not gets a warning.)"""
    if not name:
        return True
    return bool(font_file(name, bold=True)) and _FONT_CACHE.get((name, True)) is not None \
        and _family_matches(name)


def _family_matches(name: str) -> bool:
    fcmatch = shutil.which("fc-match")
    if not fcmatch:
        return False
    try:
        r = subprocess.run([fcmatch, "-f", "%{family}", name], capture_output=True, timeout=20)
        fam = (r.stdout or b"").decode(errors="replace")
    except Exception:
        return False
    return name.replace(" ", "").lower() in fam.replace(" ", "").lower()
