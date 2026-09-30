# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Generic media upload — ``POST /v1/files/upload``.

A client that is not on this machine has media of its own: the user's footage,
a logo, a music bed. It could inline all of it as base64 in every request, and
a 40 MB clip sent again for each of ten renders is exactly the kind of waste
that makes an API unusable. So: upload once, get a ``/v1/files/...`` URL, name
that URL in as many compose requests as you like.

Storage is content-addressed (sha256 of the bytes), which makes re-uploading
the same file free and makes ``GET /v1/files/blob/{hash}`` a useful question —
a client can ask before sending. This mirrors ``/v1/loras/upload`` for LoRA
weights; the difference is that these files are served back by
``/v1/files/{name}``, so they live in the ordinary output directory under their
hash, not in a private blob store.
"""

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from codai.api.loras import _require_api_auth
from codai.compose import media

router = APIRouter()

# The name a hash is stored under, e.g. up-3f9c…-<12 hex>.mp4. The prefix keeps
# uploads recognisable in the archive listing and in /v1/files URLs.
_PREFIX = "up-"
_MAX_BYTES = 2 << 30          # 2 GiB: a minute of 4K is ~400 MB


def _files_dir() -> str:
    d = media.files_dir()
    if not d:
        raise HTTPException(status_code=501, detail=(
            "this install has no output directory configured, so uploads cannot be "
            "stored (start the server with --file-path / files.path)"))
    os.makedirs(d, exist_ok=True)
    return d


def _stored_name(hexhash: str, ext: str) -> str:
    return f"{_PREFIX}{hexhash}{ext}"


def find_blob(hexhash: str) -> Optional[str]:
    """The stored path for a hash, whatever extension it landed under."""
    h = (hexhash or "").strip().lower()
    if h.startswith("sha256:"):
        h = h[7:]
    if not h or not all(c in "0123456789abcdef" for c in h) or len(h) != 64:
        return None
    try:
        d = _files_dir()
    except HTTPException:
        return None
    try:
        for name in os.listdir(d):
            if name.startswith(_PREFIX + h):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    return p
    except OSError:
        return None
    return None


def store(data: bytes, filename: str = "", content_type: str = "") -> dict:
    """Put bytes in the content-addressed store; returns the response document."""
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > _MAX_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"upload larger than {_MAX_BYTES // (1 << 20)} MiB")
    mime = media.sniff_mime(data, content_type)
    kind = media.kind_of(mime)
    if kind not in ("video", "image", "audio"):
        raise HTTPException(status_code=415, detail=(
            f"unsupported media type {mime!r}: uploads must be video "
            f"(mp4/mov/webm/mkv), image (png/jpg/webp) or audio (mp3/wav/m4a/ogg/flac)"))
    h = media.sha256_hex(data)
    existing = find_blob(h)
    if existing:
        return {"id": f"sha256:{h}", "name": os.path.basename(existing),
                "bytes": os.path.getsize(existing), "mime": mime, "kind": kind,
                "existed": True}
    name = _stored_name(h, media.ext_for(mime, filename))
    path = os.path.join(_files_dir(), name)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return {"id": f"sha256:{h}", "name": name, "bytes": len(data), "mime": mime,
            "kind": kind, "existed": False}


def _with_urls(doc: dict, http_request: Optional[Request]) -> dict:
    from codai.api.urlutils import build_file_url
    name = doc.pop("name")
    doc["path"] = f"/v1/files/{name}"
    doc["url"] = build_file_url(name, http_request)
    return doc


@router.post("/v1/files/upload", summary="Upload media (content-addressed)", tags=["Files"])
async def upload_file(request: Request, _auth=Depends(_require_api_auth)):
    """Store a media file and return the URL to reference it by.

    Accepts the file as ``multipart/form-data`` (field ``file``), as JSON
    ``{"file": "<base64 or data: URI>"}``, or as a raw request body — the three
    forms ``/v1/loras/upload`` takes. Returns ``{id, url, path, bytes, mime,
    kind, existed}``; ``existed: true`` means these exact bytes were already
    here and nothing was written. Video, image and audio only."""
    ctype = request.headers.get("content-type", "")
    data = b""
    filename = ""
    part_type = ""
    if "multipart/form-data" in ctype:
        form = await request.form()
        up = form.get("file") or form.get("upload") or form.get("media")
        if up is None:
            raise HTTPException(status_code=400,
                                detail="multipart upload missing 'file' field")
        if hasattr(up, "read"):
            data = await up.read()
            filename = getattr(up, "filename", "") or ""
            part_type = getattr(up, "content_type", "") or ""
        else:
            data = bytes(up)
    elif "application/json" in ctype:
        body = await request.json()
        blob = body.get("file") or body.get("data")
        if not blob:
            raise HTTPException(status_code=400,
                                detail="JSON upload missing 'file'/'data' (base64)")
        filename = str(body.get("filename") or "")
        try:
            data = media.fetch(str(blob))
        except media.MediaError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    else:
        data = await request.body()
        part_type = ctype
    return _with_urls(store(data, filename, part_type), request)


@router.get("/v1/files/blob/{hash}", summary="Check an uploaded file exists", tags=["Files"])
async def file_blob_info(hash: str, request: Request, _auth=Depends(_require_api_auth)):
    """Does this install already hold these bytes? 200 with the URL, else 404.

    Lets a client skip re-uploading: hash the file locally, ask, and only send
    it when the answer is 404. ``hash`` is a hex sha256, with or without the
    ``sha256:`` prefix."""
    p = find_blob(hash)
    if not p:
        raise HTTPException(status_code=404, detail="blob not found")
    name = os.path.basename(p)
    h = name[len(_PREFIX):].split(".")[0]
    doc = {"id": f"sha256:{h}", "name": name, "bytes": os.path.getsize(p),
           "mime": media.sniff_mime(open(p, "rb").read(32)), "exists": True}
    return _with_urls(doc, request)
