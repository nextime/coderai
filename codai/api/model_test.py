# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Test run — prove a model actually works, where it is configured to run.

Placement has a lot of moving parts now: a model may be served here, from a URL,
or on a RunPod pod whose image, weights and adapters each have to resolve. Most
of that fails minutes into a pod boot, in a log nobody is watching. This runs the
smallest real request for the model's kind and reports what happened.

It goes through the front like any client request, so it exercises the SAME
routing, auth, pod provisioning and staging a real request would — a test that
simulated the path would prove nothing about the path.

  POST /v1/models/test {"model": "...", "where": "auto"|"local"|"runpod"}

``where`` forces placement for this one request: "local" ignores any remote
configuration, "runpod" refuses to fall back to local, so a pod that is not
actually reachable fails the test instead of quietly passing here.

A few kinds still have no honest synthetic input — a face swap needs real faces,
stem separation needs audio worth separating — and those report a reachability
check instead of a generation, saying so rather than implying more than was
proven. Everything else is probed for real: espeak speaks a known sentence for
transcription and for voice cloning, PIL draws known words for OCR, and video is
asked for the smallest clip it can make.
"""

from codai.api import pod_http
import io
import json
import os
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from codai.api.loras import _require_api_auth

router = APIRouter()


class ModelTestRequest(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    model: str = Field(..., description="Model id, alias or path to test.")
    where: str = Field("auto", description='auto | local | runpod')
    timeout: float = Field(1800.0, description="Give up after this many seconds.")


@router.get("/v1/models/test/state", summary="What this engine currently believes")
async def test_state(model: str = "", _auth=Depends(_require_api_auth)):
    """The engine's LIVE view of a model's placement and the remote config.

    Editing config.json or models.json is not enough to change where a request
    goes: the engine re-reads them only when the front pushes a reload, and that
    push is deferred while the engine is busy. So a caller that writes a config
    and immediately tests is testing the OLD one — which is exactly what
    happened: a whole test pass reported "nothing configures this to run
    remotely" while the file on disk said otherwise, and an earlier pass sent
    production embeddings to a pod because the engine still held a stale remote.

    Poll this until it reflects what you wrote, then test. It answers for the
    process that actually serves the request, so "the engine has it" is a fact
    here rather than an assumption.
    """
    try:
        from codai.admin.routes import config_manager
        remotes = getattr(getattr(config_manager, "config", None), "remotes", None)
    except Exception:
        remotes = None
    out = {
        "pid": os.getpid(),
        "remotes_enabled": bool(getattr(remotes, "enabled", False)),
        "capability_endpoints": sorted((getattr(remotes, "endpoints", None) or {}).keys()),
        "capability_pods": sorted((getattr(remotes, "pods", None) or {}).keys()),
    }
    if model:
        entry, capability = _describe(model)
        where, _ = ("", None)
        try:
            from codai.api.remote_gateway import model_placement
            where, _ = model_placement(model)
        except Exception:
            pass
        if isinstance((entry or {}).get("runpod"), dict):
            # What a pod would be rented with, and why — decided before any
            # money is spent. `weights_gb` on the entry overrides the estimate.
            try:
                from codai.api.runpod_worker import (disk_for, estimate_weights_gb,
                                                     parse_model_runpod)
                mcfg = parse_model_runpod(entry["runpod"])
                gb, note = disk_for(entry, mcfg)
                stated = (entry.get("weights_gb")
                          or (entry.get("runpod") or {}).get("weights_gb"))
                out["pod_disk"] = {
                    "container_disk_gb": gb,
                    "weights_gb": round(estimate_weights_gb(entry, mcfg), 1),
                    "weights_source": "stated (weights_gb)" if stated else "estimated from HuggingFace",
                    "note": note or "configured size is enough",
                }
            except Exception as exc:
                out["pod_disk"] = {"error": str(exc)}
        out.update({
            "model": model,
            "known": bool(entry),
            "capability": capability or "text",
            "placement": where or "(none — falls through to the capability map)",
            "backend": str((entry or {}).get("backend") or ""),
            "has_runpod_block": isinstance((entry or {}).get("runpod"), dict)
                                and bool(entry["runpod"]),
        })
    return out


#: capability -> (path, body builder). The body is the smallest thing that still
#: exercises a real generation.
def _probe_for(capability: str, model: str):
    if capability in ("", "text"):
        # 64, not 8: a reasoning model spends its budget thinking and answers
        # with nothing at all if the cap is tight — which looks like a pass
        # (HTTP 200) while proving the model produced no output.
        return "/v1/chat/completions", {
            "model": model, "max_tokens": 64, "temperature": 0.0,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}]}
    if capability == "images":
        return "/v1/images/generations", {
            "model": model, "prompt": "a red square on white", "n": 1,
            "size": "256x256", "response_format": "b64_json"}
    if capability == "embeddings":
        return "/v1/embeddings", {"model": model, "input": "test"}
    if capability == "rerank":
        return "/v1/rerank", {"model": model, "query": "cat",
                              "documents": ["a cat", "a car"]}
    if capability == "tts":
        # No `voice`: TTSRequest's own default is a voice the shipped models
        # actually have. Passing "default" failed on Kokoro with "Voice default
        # not found in available voices" — the probe inventing a voice name, not
        # the model being broken.
        return "/v1/audio/speech", {"model": model, "input": "test"}
    if capability == "voice":
        # Voice cloning needs a reference sample AND its transcript, which is
        # exactly what espeak produces: we choose the sentence, so the transcript
        # is known rather than guessed. This was listed as un-probeable, which
        # was true only for as long as nobody tried.
        import base64
        return "/v1/audio/clone", {
            "model": model, "text": "cloned voice test",
            "ref_audio": base64.b64encode(_spoken_wav()).decode(),
            "ref_text": _SPOKEN, "response_format": "b64_wav"}
    if capability == "audio_gen":
        # 2 seconds, not the 10s default: this proves the model renders audio,
        # and the remaining 8 seconds would only prove it more expensively.
        return "/v1/audio/generate", {
            "model": model, "prompt": "a short beep", "duration": 2.0,
            "response_format": "b64_wav"}
    if capability == "video":
        # The cheapest thing a video model can be asked for: smallest frame
        # count, smallest resolution, fewest steps. Still the most expensive
        # probe here, which is why it stays behind an explicit request.
        return "/v1/video/generations", {
            "model": model, "prompt": "a red square on white",
            "width": 256, "height": 256, "num_frames": 9,
            "num_inference_steps": 4, "mode": "t2v"}
    return "", None


#: Kinds whose smallest real request is still expensive (a video is minutes of
#: GPU and real money) or needs input we cannot synthesise honestly.
#:
#: 'stt' and 'ocr' used to be here. They are not any more: an audio clip and a
#: document image CAN be synthesised — espeak says a known sentence, PIL draws
#: known words — and the round trip is checked against what went in. A
#: reachability check on those two reported ok:true while proving only that a
#: URL was configured, which is the kind of pass that hides a broken pod.
_REACHABILITY_ONLY = {
    # 'speaker' is not here: diarization takes audio, espeak makes audio, and a
    # pod with no diarization stack cannot return a labelled turn — which is the
    # failure this catches.
    "faceswap": "a face swap needs a source and target face, and a drawn one "
                "does not survive face detection — give it real images instead",
    "spatial": "3D generation needs an image or a mesh",
    "stems": "stem separation needs an audio file to separate",
    "audio_clean": "cleanup needs an audio file to clean",
}

#: What the synthesised probes say. Checked against the result, so the test
#: fails when a model returns confident nonsense as well as when it errors.
_SPOKEN = "the quick brown fox jumps over the lazy dog"
_PRINTED = "CODERAI OCR TEST"


def _multipart(fields: dict, filename: str, content: bytes,
               file_field: str = "file",
               content_type: str = "application/octet-stream") -> tuple:
    """Build a multipart/form-data body by hand: (content_type, body).

    The probe goes through the ASGI bridge as raw bytes, so there is no client
    library here to do it — and the upload has to look exactly like a real one,
    because the multipart path is itself part of what is being tested.
    """
    boundary = "----coderai-probe-" + uuid.uuid4().hex
    out = bytearray()
    for key, value in fields.items():
        if value is None:
            continue
        out += (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                f"{value}\r\n").encode()
    out += (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n").encode()
    out += content + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", bytes(out)


def _spoken_wav() -> bytes:
    """A few seconds of synthetic speech saying _SPOKEN, as a 16 kHz mono WAV.

    espeak is not a pleasant voice, but it is intelligible to every STT model
    worth shipping — and it needs no network, no model and no GPU to produce.
    """
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        path = fh.name
    try:
        subprocess.run(["espeak", "-w", path, "-s", "130", _SPOKEN],
                       check=True, capture_output=True, timeout=30)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _printed_png() -> bytes:
    """A clean white image with _PRINTED written on it, as PNG."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (900, 220), "white")
    draw = ImageDraw.Draw(img)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
    except Exception:
        font = None      # the bitmap default is small but still legible
    draw.text((40, 70), _PRINTED, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _upload_probe(capability: str, model: str):
    """(path, content_type, body) for a capability whose input is a file.

    Returns (None, '', b'') when the input cannot be produced here — a missing
    espeak, no fonts — so the caller falls back to a reachability check and says
    why, rather than failing a model for a gap on this side.
    """
    try:
        if capability == "stt":
            ct, body = _multipart({"model": model, "response_format": "json"},
                                  "probe.wav", _spoken_wav(),
                                  content_type="audio/wav")
            return "/v1/audio/transcriptions", ct, body
        if capability == "ocr":
            ct, body = _multipart({"engine": model}, "probe.png", _printed_png(),
                                  content_type="image/png")
            return "/v1/ocr", ct, body
        if capability == "speaker":
            # Diarization needs audio, and espeak provides it. One speaker
            # saying one sentence is a thin test musically but an honest one
            # here: the model has to return at least one labelled turn, which is
            # exactly what a pod with no diarization stack cannot do.
            ct, body = _multipart({"model": model}, "probe.wav", _spoken_wav(),
                                  content_type="audio/wav")
            return "/v1/audio/diarization", ct, body
    except Exception as exc:
        print(f"[test] could not synthesise a {capability} probe: {exc}", flush=True)
    return None, "", b""


@router.post("/v1/models/test", summary="Test run a model where it is configured")
async def test_model(req: ModelTestRequest, request: Request,
                     _auth=Depends(_require_api_auth)):
    """Run the smallest real request for this model and report what happened."""
    where = (req.where or "auto").strip().lower()
    if where not in ("auto", "local", "runpod", "remote"):
        raise HTTPException(status_code=400, detail=f"unknown where={where!r}")
    force = {"auto": "", "local": "local", "runpod": "remote", "remote": "remote"}[where]

    entry, capability = _describe(req.model)
    placement = _placement_summary(req.model, entry, force)

    # Forcing remote has to be verified HERE, not left to the gateway. Chat and
    # completions are deliberately outside the gateway (RemoteOpenAIBackend owns
    # them), so a forced header is ignored on that path and the request would be
    # served locally while this reported "remote" — a test passing in the wrong
    # place, which is worse than no test at all.
    if force == "remote":
        why = _no_remote_reason(req.model, entry, capability)
        if why:
            return {"model": req.model, "capability": capability or "text",
                    "where": "remote", "target": "(not configured)",
                    "ran": "", "ok": False, "seconds": 0.0, "error": why}

    path, body = _probe_for(capability, req.model)
    content_type = "application/json"
    raw_body = json.dumps(body).encode() if path else b""
    if not path:
        # An upload-shaped probe: synthesised speech, a rendered document.
        path, content_type, raw_body = _upload_probe(capability, req.model)
    if not path:
        reason = _REACHABILITY_ONLY.get(capability, "no cheap probe for this kind")
        ok, detail = _reachable(placement)
        return {"model": req.model, "capability": capability or "text",
                "where": placement["where"], "target": placement["target"],
                "ran": "reachability", "ok": ok, "detail": detail,
                "note": f"no generation was run: {reason}"}

    headers = {"Content-Type": content_type}
    if force:
        from codai.api.remote_gateway import PLACEMENT_HEADER
        headers[PLACEMENT_HEADER] = force
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth

    started = time.time()
    try:
        from codai.broker.asgi_bridge import execute_api_request
        resp = await execute_api_request(request, method="POST", path=path,
                                         headers=headers, body=raw_body)
    except Exception as exc:
        return {"model": req.model, "capability": capability or "text",
                "where": placement["where"], "target": placement["target"],
                "ran": path, "ok": False, "seconds": round(time.time() - started, 1),
                "error": f"{type(exc).__name__}: {exc}"}

    seconds = round(time.time() - started, 1)
    status = int(resp.get("status_code", 500))
    raw = resp.get("body") or b""
    out = {"model": req.model, "capability": capability or "text",
           "where": placement["where"], "target": placement["target"],
           "ran": path, "status": status, "ok": 200 <= status < 300,
           "seconds": seconds}
    # What the remote actually is, and what it thinks it can serve. Without this
    # a failure means reading engine logs to find the pod URL and asking it by
    # hand — which is how the last three failures were diagnosed.
    out["pods"] = _pod_diagnostics()
    warn = _weight_warning(req.model, entry)
    if warn:
        out["warning"] = warn
    if out["ok"]:
        out["sample"] = _sample(raw, capability)
        empty = _empty_result(raw, capability)
        if empty:
            # A 200 that carried no output is not a working model. Reporting it
            # as a pass is worse than reporting nothing.
            out["ok"] = False
            out["error"] = empty
    else:
        out["error"] = _error_text(raw)
    return out


def _empty_result(raw: bytes, capability: str) -> str:
    """'' when the response carries real output, else why it does not."""
    try:
        obj = json.loads(raw)
    except Exception:
        return "" if raw else "the response was empty"
    if capability in ("", "text"):
        choice = (obj.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content") or "").strip()
        if content:
            return ""
        produced = (obj.get("usage") or {}).get("completion_tokens", 0)
        return (f"the model returned no text (completion_tokens={produced}, "
                f"finish_reason={choice.get('finish_reason')!r}) — it answered the "
                "request with nothing")
    if capability == "embeddings":
        try:
            return "" if obj["data"][0]["embedding"] else "the embedding was empty"
        except Exception:
            return "no embedding in the response"
    if capability == "images":
        try:
            d = (obj.get("data") or [{}])[0]
            return "" if (d.get("b64_json") or d.get("url")) else "no image in the response"
        except Exception:
            return "no image in the response"
    if capability == "rerank":
        return "" if (obj.get("results") or obj.get("data")) else "no ranking returned"
    if capability in ("voice", "audio_gen", "video"):
        # No content check is possible — there is no ground truth for generated
        # audio or video — so this asks only whether anything was actually
        # produced. A 200 carrying no media is still a failure.
        for key in ("b64_wav", "b64_mp3", "audio", "url", "b64_video", "video",
                    "data"):
            if isinstance(obj, dict) and obj.get(key):
                return ""
        return f"the {capability} model returned a 200 with no media in it"
    if capability == "speaker":
        # No ground truth for who is speaking — espeak is one voice — so this
        # asks only whether the model returned labelled turns at all.
        turns = obj.get("segments") or obj.get("turns") or obj.get("speakers") \
            if isinstance(obj, dict) else None
        if not turns:
            return "no speaker turns in the response"
        return ""
    if capability in ("stt", "ocr"):
        # The probe put known words in, so check they come back. A model that
        # returns fluent nonsense is broken in a way an "is it non-empty?" check
        # waves straight through — and on a rented pod the usual cause is a
        # half-loaded model, not a bad transcription.
        want = _SPOKEN if capability == "stt" else _PRINTED
        got = _text_of(obj).lower()
        if not got.strip():
            return f"the {capability} model returned no text"
        hits = sum(1 for w in want.lower().split() if w in got)
        need = max(1, len(want.split()) // 3)
        if hits < need:
            return (f"the {capability} result does not match what was sent: "
                    f"expected words from {want!r}, got {got[:120]!r}")
    return ""


def _text_of(obj) -> str:
    """Whatever text a transcription or OCR response carries, flattened.

    The two endpoints do not share a response shape, and the OCR engines differ
    among themselves (a flat string, pages, blocks), so this looks for text
    rather than insisting on one layout.
    """
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return " ".join(_text_of(x) for x in obj)
    if isinstance(obj, dict):
        for key in ("text", "content", "full_text", "transcript"):
            if isinstance(obj.get(key), str) and obj[key].strip():
                return obj[key]
        return " ".join(_text_of(obj[k]) for k in ("pages", "blocks", "lines",
                                                   "segments", "results", "data")
                        if k in obj)
    return ""


def _no_remote_reason(model: str, entry: dict, capability: str) -> str:
    """'' when this model really is configured to run remotely, else why not."""
    entry = entry or {}
    if str(entry.get("service_url") or "").strip():
        return ""
    if str(entry.get("backend") or "").strip().lower() == "runpod":
        return ""
    if isinstance(entry.get("runpod"), dict) and entry["runpod"]:
        return ""
    if str(entry.get("backend") or "").lower() == "host" and (
            str((entry.get("host") or {}).get("url") or "").strip()
            or (entry.get("host") or {}).get("hosts")):
        return ""
    try:
        from codai.api.remote_gateway import capability_endpoints, capability_pods
        if capability and (capability in capability_endpoints()
                           or capability in capability_pods()):
            return ""
    except Exception:
        pass
    if not entry:
        return (f"{model!r} is not in the model list, so nothing can say where it "
                "should run")
    if not capability:
        # A text/LLM model. Do NOT offer a 'text' capability remote as the fix:
        # /v1/chat/completions is deliberately outside the gateway (the backend
        # owns it), so configuring one would route nothing and the test would
        # then report a remote pass for a request served locally.
        return (f"nothing configures {model!r} to run remotely — set a service_url "
                "on the model, or backend 'runpod' with a runpod block on it. A "
                "capability remote does not apply to chat: that path is served by "
                "the backend, not the gateway")
    return (f"nothing configures {model!r} to run remotely — set a service_url, or "
            "backend 'runpod' with a runpod block on the model, or a remote for "
            f"its {capability} capability")


def _weight_warning(model: str, entry: dict) -> str:
    """The 'this downloads 150 GB every cold boot' warning, if it applies.

    Said here as well as at provision time: by the time a pod is booting the
    money is already being spent, and a test run is where someone looks BEFORE
    committing to a configuration.
    """
    try:
        from codai.api.runpod_worker import (weight_transfer_warning,
                                             parse_model_runpod)
        block = (entry or {}).get("runpod")
        if not isinstance(block, dict):
            return ""
        return weight_transfer_warning(entry, parse_model_runpod(block))
    except Exception:
        return ""


def _pod_diagnostics() -> list:
    """Live pods and what each is serving, for the answer to stand on its own."""
    try:
        from codai.api.runpod_worker import pods_status, _pools, _pools_lock
    except Exception:
        return []
    out = []
    try:
        for pod in pods_status():
            row = {"pod": pod.get("pod_id"), "for": pod.get("model"),
                   "gpu": pod.get("gpu"), "state": pod.get("state"),
                   "uptime_s": pod.get("uptime_s"),
                   "usd_so_far": pod.get("live_cost_usd")}
            out.append(row)
        # Add the URL and catalogue for the pods this process holds.
        with _pools_lock:
            pools = list(_pools.values())
        urls = {}
        for pool in pools:
            with pool._cv:
                for p in pool.pods:
                    urls[p.pod_id] = (p.url, getattr(pool, "api_key", ""))
        for row in out:
            url, key = urls.get(row["pod"], ("", ""))
            if not url:
                continue
            row["url"] = url
            row["serves"] = _remote_catalogue(url, key)
    except Exception as exc:
        return [{"error": f"pod diagnostics unavailable: {exc}"}]
    return out


def _remote_catalogue(url: str, api_key: str) -> list:
    """The model ids a remote lists, or a short reason it cannot be asked."""
    import requests
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = pod_http.get(url.rstrip("/") + "/v1/models", headers=headers, timeout=15)
        if r.status_code != 200:
            return [f"(HTTP {r.status_code})"]
        return [str(m.get("id")) for m in (r.json().get("data") or [])] or ["(none)"]
    except Exception as exc:
        return [f"({exc})"]


#: OCR engines are selected by name, not registered in models.json like a model.
#: Without this an OCR test answered "'surya' is not in the model list, so
#: nothing can say where it should run" — true of the catalogue, and useless.
_OCR_ENGINES = ("paddle", "doctr", "surya")


def _describe(model: str):
    """(entry, capability) for a model — capability '' means a text/LLM model."""
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(model) or {}
    except Exception:
        entry = {}
    if not entry and (model or "").strip().lower() in _OCR_ENGINES:
        return {"path": model, "model_type": "ocr"}, "ocr"
    try:
        from codai.api.runpod_worker import model_capability
        return entry, model_capability(entry)
    except Exception:
        return entry, ""


def _placement_summary(model: str, entry: dict, force: str) -> dict:
    """Where this test will actually run, decided by the real routing code."""
    if force == "local":
        return {"where": "local", "target": "this instance (forced)"}
    try:
        from codai.api.remote_gateway import model_placement
        kind, detail = model_placement(model)
    except Exception:
        kind, detail = "", None
    if kind == "local":
        return {"where": "local", "target": "pinned local"}
    if kind == "url":
        return {"where": "remote", "target": str(detail)}
    if kind == "pod":
        return {"where": "runpod", "target": "its own pod"}
    if kind == "host":
        _entry, hblock = detail
        how = "started on demand" if str(hblock.get("start_cmd") or "").strip() else "always on"
        try:
            from codai.api.host_worker import parse_hosts
            urls = [h.url for h in parse_hosts(hblock)]
        except Exception:
            urls = [hblock.get("url")]
        return {"where": "remote", "target": f"host {', '.join(str(u) for u in urls)} ({how})"}
    backend = str((entry or {}).get("backend") or "").lower()
    if backend == "runpod":
        return {"where": "runpod", "target": "RunPod backend"}
    return {"where": "local" if force != "remote" else "remote",
            "target": "this instance" if force != "remote" else "capability remote"}


def _reachable(placement: dict):
    """A health check, for kinds whose real request is too costly to run."""
    if placement["where"] == "local":
        return True, "served by this instance"
    return True, (f"configured to run on {placement['target']} — not verified by a "
                  "request; run a real one to prove the remote works")


def _sample(raw: bytes, capability: str) -> str:
    """A short, human-readable sign of life from the response."""
    try:
        obj = json.loads(raw)
    except Exception:
        return f"{len(raw)} bytes"
    if capability in ("", "text"):
        try:
            return (obj["choices"][0]["message"]["content"] or "")[:120]
        except Exception:
            return str(obj)[:120]
    if capability == "embeddings":
        try:
            return f"{len(obj['data'][0]['embedding'])} dimensions"
        except Exception:
            return str(obj)[:120]
    if capability == "images":
        try:
            b64 = obj["data"][0].get("b64_json") or ""
            return f"image returned ({len(b64) * 3 // 4} bytes)"
        except Exception:
            return str(obj)[:120]
    if capability == "speaker":
        turns = (obj.get("segments") or obj.get("turns") or
                 obj.get("speakers") or []) if isinstance(obj, dict) else []
        return f"{len(turns)} speaker turn(s)"
    if capability in ("stt", "ocr"):
        return (_text_of(obj) or str(obj))[:120]
    if capability in ("voice", "audio_gen", "tts"):
        for key in ("b64_wav", "b64_mp3", "audio", "url"):
            val = obj.get(key) if isinstance(obj, dict) else None
            if val:
                return (f"{key}: {len(val) * 3 // 4} bytes" if key != "url"
                        else f"url: {val}")[:120]
        return str(obj)[:120]
    if capability == "video":
        for key in ("url", "b64_video", "video"):
            val = obj.get(key) if isinstance(obj, dict) else None
            if val:
                return (f"{key}: {val}" if key == "url"
                        else f"{key}: {len(val) * 3 // 4} bytes")[:120]
        data = obj.get("data") if isinstance(obj, dict) else None
        if isinstance(data, list) and data:
            return f"video returned ({str(data[0])[:80]})"
        return str(obj)[:120]
    return str(obj)[:120]


def _error_text(raw: bytes) -> str:
    try:
        obj = json.loads(raw)
    except Exception:
        return (raw or b"").decode("utf-8", "replace")[:400]
    for key in ("detail", "error", "message"):
        val = obj.get(key)
        if isinstance(val, dict):
            val = val.get("message") or json.dumps(val)
        if val:
            return str(val)[:400]
    return json.dumps(obj)[:400]
