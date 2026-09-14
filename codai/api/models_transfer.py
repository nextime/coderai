# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Push a model's weights to a coderai that cannot fetch them itself.

A rented pod normally downloads its own weights — from HuggingFace, or from a
URL — which costs nothing here and runs at datacenter speed. But some models
have neither: a merge you made locally, a file from a host that is gone, weights
you are not going to publish. For those, the only way to get them onto a pod is
to send them.

That is what this is. It is deliberately a plain streaming upload rather than
anything clever: the receiving side writes the bytes to its model directory and
registers the entry, so the model becomes loadable exactly as if it had been
downloaded there.

  POST /v1/models/upload?name=<file>&model_type=<section>[&alias=][&tar=1]
       body: the raw file (or a tar stream when tar=1), streamed to disk
  GET  /v1/models/uploaded?name=<file>[&bytes=<n>]
       200 when the file is already there (and the right size), 404 otherwise

Sending 30 GB is a real decision, not a default: coderai resolves a HuggingFace
repo id or a URL first and only uploads when told to (`source: "upload"`). The
existence check makes the second pod cheap — but a pod is disposable storage, so
every NEW pod pays the upload again. A RunPod network volume is the better answer
for weights you will use repeatedly.
"""

import os
import shutil
import tempfile

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from codai.api.loras import _require_api_auth

router = APIRouter()

#: models.json sections an upload may target.
_SECTIONS = {
    "text_models", "image_models", "audio_models", "gguf_models", "tts_models",
    "vision_models", "video_models", "audio_gen_models", "embedding_models",
    "spatial_models",
}


def _models_dir() -> str:
    """Where uploaded weights land — the same cache the downloader writes to."""
    try:
        from codai.models.cache import get_model_cache_dir
        return get_model_cache_dir()
    except Exception:
        return os.path.expanduser(os.environ.get("CODERAI_MODELS_DIR", "~/.coderai/models"))


def _safe_target(name: str) -> str:
    """Resolve an upload name to a path inside the model directory.

    The name comes from another machine, so it is reduced to a bare basename and
    the result is re-checked against the directory: a crafted name must not be
    able to write anywhere else.
    """
    base = os.path.basename((name or "").strip())
    if not base or base in (".", "..") or "/" in base or "\\" in base:
        raise HTTPException(status_code=400, detail="invalid model name")
    root = os.path.realpath(_models_dir())
    os.makedirs(root, exist_ok=True)
    target = os.path.realpath(os.path.join(root, base))
    if os.path.commonpath([target, root]) != root:
        raise HTTPException(status_code=400, detail="invalid model name")
    return target


@router.get("/v1/models/uploaded", summary="Check an uploaded model is present")
async def model_uploaded(name: str = Query(...), bytes: int = Query(0),
                         _auth=Depends(_require_api_auth)):
    """200 when the named model is already here, 404 otherwise.

    Lets the sender skip re-uploading to a pod that already has it — the check
    that makes a warm pod free and a cold one pay once.
    """
    target = _safe_target(name)
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="not uploaded")
    size = (os.path.getsize(target) if os.path.isfile(target)
            else sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(target) for f in fs))
    if bytes and size != bytes:
        # A partial or stale copy is worse than none: report it missing so the
        # sender replaces it rather than the pod loading half a model.
        raise HTTPException(status_code=404, detail="size mismatch")
    return {"name": os.path.basename(target), "path": target, "bytes": size}


@router.post("/v1/models/register", summary="Register a model at runtime")
async def register_models(request: Request, _auth=Depends(_require_api_auth)):
    """Teach a running instance about models it did not start with.

    A pod is seeded at boot with what its owner knew then. But a pod is meant to
    be an EXTENSION of the local system: a request can name any model at any
    time, and the pod must be able to learn it, fetch it and serve it rather than
    answer "not available" for the rest of its life.

    Body: one model entry, or a list of them — the same shape models.json uses.
    Each needs a ``path`` the receiver can actually fetch (a HuggingFace repo id
    or a URL); a local path from another machine means nothing here. Registration
    is in memory only, like the boot seed: a pod is disposable.

    Idempotent: a model already known is left exactly as it is, so re-registering
    never disturbs a loaded model.
    """
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}")
    entries = body if isinstance(body, list) else [body]
    added, known, refused = [], [], []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path") or "").strip()
        section = str(entry.get("model_type") or "text_models")
        if not path:
            refused.append("(no path)")
            continue
        if section not in _SECTIONS:
            refused.append(f"{path} (unknown model_type {section!r})")
            continue
        if path.startswith("/") and not os.path.exists(path):
            # A path from the sender's filesystem. Say so plainly rather than
            # register something that will fail at load time.
            refused.append(f"{path} (local path that does not exist here — send a "
                           "HuggingFace id, a URL, or upload the weights)")
            continue
        state = _register_entry(entry, section)
        (added if state == "added" else known).append(path)
    return {"added": added, "already_known": known, "refused": refused}


def _register_entry(entry: dict, section: str) -> str:
    """Add one entry to the in-memory catalogue. Returns 'added' or 'known'."""
    try:
        from codai.admin.routes import config_manager
        md = getattr(config_manager, "models_data", None)
        if not isinstance(md, dict):
            return "known"
        lst = md.setdefault(section, [])
        if not isinstance(lst, list):
            return "known"
        path = entry.get("path")
        if any(isinstance(m, dict) and m.get("path") == path for m in lst):
            return "known"
        lst.append(dict(entry))
        print(f"[register] {path} added to {section}", flush=True)
        return "added"
    except Exception as exc:
        print(f"[register] failed: {exc}", flush=True)
        return "known"


@router.post("/v1/models/upload", summary="Upload model weights")
async def upload_model(request: Request, name: str = Query(...),
                       model_type: str = Query("text_models"),
                       alias: str = Query(""), tar: int = Query(0),
                       _auth=Depends(_require_api_auth)):
    """Stream model weights to this instance and register them.

    Streamed straight to disk in chunks — these are multi-GB files and must never
    be held in memory. A tar stream (``tar=1``) is extracted, which is how a
    multi-file model directory travels.
    """
    if model_type not in _SECTIONS:
        raise HTTPException(status_code=400, detail=f"unknown model_type {model_type!r}")
    target = _safe_target(name)

    tmp = target + ".part"
    total = 0
    try:
        with open(tmp, "wb") as fh:
            async for chunk in request.stream():
                if chunk:
                    fh.write(chunk)
                    total += len(chunk)
        if not total:
            raise HTTPException(status_code=400, detail="empty upload")
        if tar:
            extract_to = target + ".incoming"
            shutil.rmtree(extract_to, ignore_errors=True)
            os.makedirs(extract_to, exist_ok=True)
            import tarfile
            with tarfile.open(tmp, "r:*") as tf:
                _safe_extract(tf, extract_to)
            shutil.rmtree(target, ignore_errors=True)
            os.replace(extract_to, target)
            os.remove(tmp)
        else:
            os.replace(tmp, target)
    except HTTPException:
        _cleanup(tmp)
        raise
    except Exception as exc:
        _cleanup(tmp)
        raise HTTPException(status_code=500, detail=f"upload failed: {exc}")

    _register(target, model_type, alias)
    return {"path": target, "bytes": total, "model_type": model_type}


def _cleanup(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _safe_extract(tf, dest: str) -> None:
    """Extract a tar, refusing any member that would escape ``dest``.

    The archive comes from another machine; a member named ../../etc/x must not
    be able to write outside the destination.
    """
    root = os.path.realpath(dest)
    for member in tf.getmembers():
        out = os.path.realpath(os.path.join(dest, member.name))
        if os.path.commonpath([out, root]) != root:
            raise HTTPException(status_code=400,
                                detail=f"tar member escapes the destination: {member.name}")
        if member.issym() or member.islnk():
            link = os.path.realpath(os.path.join(os.path.dirname(out), member.linkname))
            if os.path.commonpath([link, root]) != root:
                raise HTTPException(status_code=400,
                                    detail=f"tar link escapes the destination: {member.name}")
    tf.extractall(dest)


def _register(path: str, model_type: str, alias: str = "") -> None:
    """Add the uploaded model to the in-memory catalogue so it can be served.

    In memory only, like CODERAI_SEED_MODELS: a pod is disposable, and writing
    models.json here would edit a real installation's catalogue behind its back.
    """
    try:
        from codai.admin.routes import config_manager
        md = getattr(config_manager, "models_data", None)
        if not isinstance(md, dict):
            return
        lst = md.setdefault(model_type, [])
        if not isinstance(lst, list):
            return
        if any(isinstance(m, dict) and m.get("path") == path for m in lst):
            return
        entry = {"path": path, "model_type": model_type, "model_types": [model_type]}
        if alias:
            entry["alias"] = alias
        lst.append(entry)
        print(f"[upload] registered {path} in {model_type}", flush=True)
    except Exception as exc:
        print(f"[upload] could not register {path}: {exc}", flush=True)
