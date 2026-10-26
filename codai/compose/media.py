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


def uploads_dir() -> Optional[str]:
    """Where ``POST /v1/files/upload`` keeps its content-addressed store.

    The output directory when there is one, so an upload is served back by
    ``/v1/files/<name>`` beside the files this install generates. An install
    with no output directory configured still has a cache directory, and
    uploads land there instead of being refused: every media field accepts the
    ``sha256:`` id the upload returns, and ``/v1/files/<name>`` serves these
    too (codai/api/app.py). Returns None only when even the cache directory
    cannot be determined."""
    return files_dir() or _cache_upload_dir()


def _cache_upload_dir() -> Optional[str]:
    try:
        from codai.platform_paths import user_cache_dir
        return str(user_cache_dir() / "uploads")
    except Exception:
        return None


def _upload_dirs() -> list:
    """Every directory an upload may have landed in, nearest first. The cache
    store stays in the list even once an output directory is configured, so an
    install that gains one does not lose track of what it already holds."""
    out = []
    for d in (files_dir(), _cache_upload_dir()):
        if d and d not in out:
            out.append(d)
    return out


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
    if not name:
        return None
    # An uploaded file may live in the cache store instead, on an install with
    # no output directory; a generated one only ever lives in the output dir.
    bases = _upload_dirs() if name.startswith(UPLOAD_PREFIX) else [_files_dir()]
    for base in bases:
        if not base:
            continue
        safe_base = os.path.realpath(base)
        cand = os.path.realpath(os.path.join(base, name))
        if not (cand == safe_base or cand.startswith(safe_base + os.sep)):
            continue
        if os.path.isfile(cand):
            return cand
    return None


_SHA256 = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$", re.I)
#: Uploads are stored in the output directory under this prefix (codai/api/uploads.py).
UPLOAD_PREFIX = "up-"


def blob_path(ref: str) -> Optional[str]:
    """The stored path for a ``sha256:<hex>`` upload id, whatever extension it
    landed under, or None when this install does not hold it."""
    m = _SHA256.match((ref or "").strip())
    if not m:
        return None
    h = m.group(1).lower()
    for base in _upload_dirs():
        try:
            for name in os.listdir(base):
                if name.startswith(UPLOAD_PREFIX + h):
                    p = os.path.join(base, name)
                    if os.path.isfile(p):
                        return p
        except OSError:
            continue
    return None


def local_path(ref: str) -> Optional[str]:
    """The local file a reference names, when this install already holds it: a
    ``/v1/files/...`` path (or an absolute URL ending in one), or a ``sha256:``
    id from ``/v1/files/upload``. None for anything that must be fetched."""
    return local_path_for_files_url(ref) or blob_path(ref)


def fetch_bytes(ref: str, timeout: float = 120.0, max_bytes: int = 2 << 30) -> bytes:
    """The bytes behind ANY accepted media reference, for the endpoints that want
    bytes rather than a path: a file this install holds (read from disk, never
    over HTTP — a loopback fetch would need a token and would fail when the front
    is bound elsewhere), a ``data:`` URI, base64, or a remote URL."""
    local = local_path(ref)
    if local:
        with open(local, "rb") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise MediaError(f"{os.path.basename(local)}: larger than {max_bytes} bytes")
        return data
    return fetch(ref, timeout=timeout, max_bytes=max_bytes)


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
        # Whitespace is not part of base64 but is everywhere in practice: MIME
        # wrapping at 76 columns, `base64` from a shell, Python's encodebytes.
        # Strict validation is still what rejects a reference that is not base64
        # at all -- it just must not reject a correctly encoded one for its line
        # breaks, which is what every wrapped payload has.
        return base64.b64decode("".join(ref.split()), validate=True)
    except Exception:
        raise MediaError("media reference is not a URL, a /v1/files path, a data: URI or base64")


def resolve(ref: str, workdir: str, name_hint: str = "media",
            timeout: float = 120.0) -> str:
    """A media reference → a local file path, downloading into ``workdir`` when
    the reference is remote or inline. Files this install already holds are used
    where they are."""
    local = local_path(ref)
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


_FONT_EXTS = (".ttf", ".otf", ".ttc", ".pfb", ".woff", ".woff2")
_FONTS_REGISTERED = False


def operator_font_dirs() -> list:
    """Directories an operator may drop caption fonts into, highest priority first.

    ``CODERAI_FONT_DIR`` lets a deployment point somewhere else entirely (a
    shared asset mount, say); otherwise it is ``<config>/fonts``, which rides
    the config volume and so survives both a restart and a new image.
    """
    out = []
    env = os.environ.get("CODERAI_FONT_DIR")
    if env:
        out.extend(d for d in env.split(os.pathsep) if d.strip())
    try:
        from codai.platform_paths import default_fonts_dir
        out.append(str(default_fonts_dir()))
    except Exception:
        pass
    seen, uniq = set(), []
    for d in out:
        d = os.path.abspath(os.path.expanduser(d))
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def register_operator_fonts(force: bool = False) -> list:
    """Make operator-supplied fonts visible to fontconfig, and so to libass.

    Captions are burned by ffmpeg's libass, a SUBPROCESS that does its own
    fontconfig lookup -- so it is not enough for this process to find the file.
    The directory has to be in fontconfig's own configuration, and the one
    place a non-root process can put it that a child also reads is the user
    config: ``$XDG_CONFIG_HOME/fontconfig/conf.d``, inherited through the
    environment. (Writing to /usr/local/share/fonts would be cleaner and is
    not available: the container runs unprivileged.)

    Returns the directories registered. Idempotent, and cheap enough to call at
    every startup: it rewrites the snippet only when the content changes, since
    fc-cache on a large font tree is not free.
    """
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED and not force:
        return []
    dirs = [d for d in operator_font_dirs() if os.path.isdir(d)]
    have = [d for d in dirs
            if any(f.lower().endswith(_FONT_EXTS) for f in os.listdir(d))]
    _FONTS_REGISTERED = True
    if not have:
        return []
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    conf_d = os.path.join(xdg, "fontconfig", "conf.d")
    body = ("<?xml version=\"1.0\"?>\n"
            "<!DOCTYPE fontconfig SYSTEM \"urn:fontconfig:fonts.dtd\">\n"
            "<!-- written by coderai; operator caption fonts -->\n"
            "<fontconfig>\n"
            + "".join(f"  <dir>{d}</dir>\n" for d in have)
            + "</fontconfig>\n")
    conf = os.path.join(conf_d, "99-coderai-fonts.conf")
    try:
        os.makedirs(conf_d, exist_ok=True)
        try:
            unchanged = open(conf, encoding="utf-8").read() == body
        except Exception:
            unchanged = False
        if not unchanged:
            with open(conf, "w", encoding="utf-8") as fh:
                fh.write(body)
        fccache = shutil.which("fc-cache")
        if fccache and not unchanged:
            subprocess.run([fccache, "-f"] + have,
                           capture_output=True, timeout=180)
        _FONT_CACHE.clear()          # a new font changes what was resolved
        try:                         # ...and what the caption measurer loaded
            from codai.compose.captions import _FACE_CACHE
            _FACE_CACHE.clear()
        except Exception:
            pass
    except Exception as exc:
        print(f"[fonts] could not register {have}: {exc}")
        return []
    return have


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
