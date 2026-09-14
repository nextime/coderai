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

Some kinds have no cheap synthetic input (faceswap needs faces, video costs real
money and minutes). Those report a reachability check instead of a generation,
and say so rather than implying more than was proven.
"""

import io
import json
import time
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
        return "/v1/audio/speech", {"model": model, "input": "test", "voice": "default"}
    return "", None


#: Kinds whose smallest real request is still expensive (a video is minutes of
#: GPU and real money) or needs input we cannot synthesise honestly.
_REACHABILITY_ONLY = {
    "video": "a video generation costs minutes of GPU time and real money",
    "voice": "voice cloning needs a reference sample",
    "faceswap": "a face swap needs a source and target face",
    "spatial": "3D generation needs an image or a mesh",
    "audio_gen": "audio generation is minutes of GPU time",
    "stems": "stem separation needs an audio file",
    "audio_clean": "cleanup needs an audio file",
    "stt": "transcription needs an audio file",
    "ocr": "OCR needs a document image",
}


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
    if not path:
        reason = _REACHABILITY_ONLY.get(capability, "no cheap probe for this kind")
        ok, detail = _reachable(placement)
        return {"model": req.model, "capability": capability or "text",
                "where": placement["where"], "target": placement["target"],
                "ran": "reachability", "ok": ok, "detail": detail,
                "note": f"no generation was run: {reason}"}

    headers = {"Content-Type": "application/json"}
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
                                         headers=headers,
                                         body=json.dumps(body).encode())
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
    return (f"nothing configures {model!r} to run remotely — set a service_url, or "
            "backend 'runpod' with a runpod block on the model, or a remote for "
            f"its {capability or 'text'} capability")


def _describe(model: str):
    """(entry, capability) for a model — capability '' means a text/LLM model."""
    try:
        from codai.models.manager import _model_entry_for
        entry = _model_entry_for(model) or {}
    except Exception:
        entry = {}
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
