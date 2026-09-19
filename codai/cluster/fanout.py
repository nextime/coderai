# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""One request, several machines: split the WORK, not the model.

A generation request often carries more than one unit of work — ``n``
images or videos, a list of texts to embed, documents to rerank, a long text
to speak, a long recording to transcribe, a batch of documents to OCR.
Every engine that has the model (local, cluster node, host) can take a
share. This module knows, per endpoint, how to cut a request into parts,
send the parts concurrently through the front's own engine clients, and
put the answers back together in the original order — so the caller sees
one answer, and the Tasks page sees one request fanned out over N engines.

What it deliberately does NOT do: split one generation across machines
(one image's denoising loop, one clip's frames). That is the model's
business (docs/cluster.md — RPC, vLLM on Ray, component placement).

Per model: ``distribute`` in models.json —
``{"enabled": true, "nodes": "auto" | "box2, box3", "max_parts": 0,
   "min_items": 2, "chunk_seconds": 120}``.
"""

import asyncio
import base64
import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple

#: Endpoints this module can split, and by what.
SPLITTABLE = {
    "/v1/images/generations": "count",        # n
    "/v1/video/generations": "count",         # n
    "/v1/embeddings": "list:input",
    "/v1/rerank": "list:documents",
    "/v1/audio/speech": "text:input",
    "/v1/audio/transcriptions": "audio",      # multipart, chunked by time
    "/v1/ocr/batch": "files",                 # multipart, one file per part
}

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n{2,}")


def parse_distribute(raw) -> dict:
    """The per-model block, normalized. Disabled unless enabled explicitly."""
    d = raw if isinstance(raw, dict) else {}
    nodes = d.get("nodes")
    if isinstance(nodes, str):
        nodes = [x.strip() for x in nodes.split(",") if x.strip()]
        if nodes == ["auto"]:
            nodes = []
    out = {
        "enabled": bool(d.get("enabled", False)),
        "nodes": [str(x) for x in (nodes or [])],
    }
    for k, default in (("max_parts", 0), ("min_items", 2), ("chunk_seconds", 120)):
        try:
            out[k] = int(d.get(k) or default)
        except (TypeError, ValueError):
            out[k] = default
    return out


def kind_for(path: str) -> Optional[str]:
    return SPLITTABLE.get(path.split("?", 1)[0].rstrip("/"))


# ---------------------------------------------------------------- splitting
def _even(total: int, parts: int) -> List[int]:
    """Sizes that sum to ``total`` over ``parts`` slots, largest first."""
    parts = max(1, min(parts, total))
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def split_json(kind: str, body: dict, parts: int, min_items: int) -> Optional[List[dict]]:
    """Cut a JSON request into up to ``parts`` requests, or None when there
    is nothing to split (one image, a single string, too few items)."""
    if kind == "count":
        n = int(body.get("n") or 1)
        if n < max(2, min_items) or parts < 2:
            return None
        sizes = _even(n, parts)
        out, offset = [], 0
        for i, sz in enumerate(sizes):
            b = copy.deepcopy(body)
            b["n"] = sz
            # A fixed seed must not make every part generate the same image:
            # each part gets its own offset, reproducible for the same request.
            if body.get("seed") is not None:
                b["seed"] = int(body["seed"]) + offset
            offset += sz
            out.append(b)
        return out
    if kind.startswith("list:"):
        field = kind.split(":", 1)[1]
        items = body.get(field)
        if not isinstance(items, list) or len(items) < max(2, min_items) or parts < 2:
            return None
        sizes = _even(len(items), parts)
        out, i = [], 0
        for sz in sizes:
            b = copy.deepcopy(body)
            b[field] = items[i:i + sz]
            i += sz
            out.append(b)
        return out
    if kind.startswith("text:"):
        field = kind.split(":", 1)[1]
        text = body.get(field)
        if not isinstance(text, str) or parts < 2:
            return None
        sentences = [s for s in _SENTENCE_END.split(text) if s and s.strip()]
        if len(sentences) < max(2, min_items):
            return None
        sizes = _even(len(sentences), parts)
        out, i = [], 0
        for sz in sizes:
            b = copy.deepcopy(body)
            b[field] = " ".join(s.strip() for s in sentences[i:i + sz])
            i += sz
            out.append(b)
        return out
    return None


# ------------------------------------------------------------------ merging
def merge_json(kind: str, path: str, body: dict, results: List[dict]) -> dict:
    """Put the parts' answers back into one, in the original order."""
    if kind == "count":
        out = dict(results[0])
        data = []
        for r in results:
            data.extend(r.get("data") or [])
        out["data"] = data
        return out
    if kind == "list:input":                     # embeddings
        out = dict(results[0])
        data, usage_p, usage_t, idx = [], 0, 0, 0
        for r in results:
            for e in (r.get("data") or []):
                e = dict(e)
                e["index"] = idx
                idx += 1
                data.append(e)
            u = r.get("usage") or {}
            usage_p += int(u.get("prompt_tokens") or 0)
            usage_t += int(u.get("total_tokens") or 0)
        out["data"] = data
        if any(r.get("usage") for r in results):
            out["usage"] = {"prompt_tokens": usage_p, "total_tokens": usage_t}
        return out
    if kind == "list:documents":                 # rerank
        out = dict(results[0])
        merged, offset = [], 0
        for r, part in zip(results, _part_sizes(body.get("documents") or [], len(results))):
            for item in (r.get("results") or r.get("data") or []):
                item = dict(item)
                item["index"] = int(item.get("index") or 0) + offset
                merged.append(item)
            offset += part
        merged.sort(key=lambda x: -float(x.get("relevance_score", x.get("score", 0)) or 0))
        top_n = body.get("top_n")
        if top_n:
            merged = merged[: int(top_n)]
        key = "results" if "results" in results[0] else "data"
        out[key] = merged
        return out
    return results[0]


def _part_sizes(items, parts: int) -> List[int]:
    return _even(len(items), parts) if items else [0] * parts


# ------------------------------------------------------------------- audio
def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _duration(path: str) -> float:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", path],
                           capture_output=True, text=True, timeout=60)
        return float((r.stdout or "0").strip() or 0)
    except Exception:
        return 0.0


def _silences(path: str, min_len: float = 0.4) -> List[float]:
    """Midpoints of silences (seconds) — the places a cut costs nothing."""
    try:
        r = subprocess.run(["ffmpeg", "-nostats", "-i", path, "-af",
                            f"silencedetect=noise=-35dB:d={min_len}", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=600)
    except Exception:
        return []
    starts, out = [], []
    for line in (r.stderr or "").splitlines():
        m = re.search(r"silence_start: ([0-9.]+)", line)
        if m:
            starts.append(float(m.group(1)))
        m = re.search(r"silence_end: ([0-9.]+)", line)
        if m and starts:
            s = starts.pop(0)
            out.append((s + float(m.group(1))) / 2.0)
    return out


def audio_cut_points(path: str, chunk_seconds: float, parts_max: int) -> List[Tuple[float, float]]:
    """(start, end) windows of about ``chunk_seconds``, each boundary moved
    to the nearest silence within a quarter chunk so words are not cut."""
    dur = _duration(path)
    if dur <= 0 or chunk_seconds <= 0 or dur < chunk_seconds * 1.5:
        return []
    n = int(dur // chunk_seconds) + (1 if dur % chunk_seconds > chunk_seconds * 0.5 else 0)
    n = max(2, min(n, parts_max if parts_max > 0 else n))
    target = dur / n
    sil = _silences(path)
    cuts = [0.0]
    for i in range(1, n):
        want = i * target
        near = [s for s in sil if abs(s - want) <= target / 4]
        cuts.append(min(near, key=lambda s: abs(s - want)) if near else want)
    cuts.append(dur)
    return [(cuts[i], cuts[i + 1]) for i in range(n) if cuts[i + 1] > cuts[i]]


def cut_audio(path: str, windows: List[Tuple[float, float]], workdir: str) -> List[str]:
    out = []
    for i, (s, e) in enumerate(windows):
        dest = os.path.join(workdir, f"part{i:03d}.wav")
        subprocess.run(["ffmpeg", "-nostats", "-loglevel", "error", "-y", "-i", path,
                        "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-ac", "1", "-ar", "16000",
                        dest], check=True, timeout=600)
        out.append(dest)
    return out


def merge_transcriptions(fmt: str, parts: List[dict], windows: List[Tuple[float, float]]):
    """Shift each part's timestamps by its window start and concatenate."""
    fmt = (fmt or "json").lower()
    text = " ".join((p.get("text") or "").strip() for p in parts if isinstance(p, dict)).strip()
    segments, words = [], []
    for p, (start, _end) in zip(parts, windows):
        if not isinstance(p, dict):
            continue
        for s in p.get("segments") or []:
            s = dict(s)
            s["start"] = float(s.get("start") or 0) + start
            s["end"] = float(s.get("end") or 0) + start
            segments.append(s)
        for w in p.get("words") or []:
            w = dict(w)
            w["start"] = float(w.get("start") or 0) + start
            w["end"] = float(w.get("end") or 0) + start
            words.append(w)
    for i, s in enumerate(segments):
        s["id"] = i
    if fmt == "verbose_json":
        out = dict(parts[0]) if parts and isinstance(parts[0], dict) else {}
        out.update({"text": text, "segments": segments,
                    "duration": windows[-1][1] if windows else 0})
        if words:
            out["words"] = words
        return out
    return {"text": text}


def concat_audio(files: List[str], fmt: str, workdir: str) -> bytes:
    """Join spoken parts into one file of the requested format."""
    lst = os.path.join(workdir, "list.txt")
    with open(lst, "w") as f:
        for p in files:
            f.write(f"file '{p}'\n")
    ext = {"mp3": "mp3", "wav": "wav", "flac": "flac", "opus": "ogg", "aac": "aac",
           "pcm": "wav"}.get((fmt or "mp3").lower(), "mp3")
    dest = os.path.join(workdir, f"joined.{ext}")
    subprocess.run(["ffmpeg", "-nostats", "-loglevel", "error", "-y", "-f", "concat",
                    "-safe", "0", "-i", lst, dest], check=True, timeout=600)
    with open(dest, "rb") as f:
        return f.read()


# --------------------------------------------------------------- planning
def choose_engines(registry, model: str, required_cap: Optional[str],
                   wanted: List[str], max_parts: int) -> list:
    """Engines to fan out over: the named ones, else every healthy engine
    that can serve the capability, the ones already holding the model first."""
    cands = []
    for e in registry.all():
        if getattr(e, "role", "engine") == "system" or not e.healthy:
            continue
        if not e.can_serve(required_cap):
            continue
        if wanted:
            names = [w.lower() for w in wanted]
            is_local = not getattr(e, "remote", False)
            if e.name.lower() not in names and not (is_local and "local" in names):
                continue
        cands.append(e)
    holder = registry.engine_for_model(model, required_cap) if model else None
    cands.sort(key=lambda e: (0 if e is holder else 1, int(getattr(e, "inflight", 0) or 0)))
    if max_parts > 0:
        cands = cands[:max_parts]
    return cands


class FanoutResult:
    def __init__(self, status: int, content: bytes, media_type: str):
        self.status = status
        self.content = content
        self.media_type = media_type


async def run_json(front, request, path: str, kind: str, body: dict, engines: list,
                   model: str, dcfg: dict) -> Optional["FanoutResult"]:
    """Split a JSON request over ``engines``; None when it cannot be split."""
    parts = split_json(kind, body, len(engines), dcfg["min_items"])
    if not parts:
        return None
    engines = engines[: len(parts)]
    headers = {"content-type": "application/json"}
    fwd_auth = request.headers.get("authorization")
    if fwd_auth:
        headers["authorization"] = fwd_auth

    async def _one(engine, part):
        rid = engine.enter_request({"model": model, "kind": front._task_kind(path),
                                    "path": path, "fanout": f"{len(parts)} parts"})
        try:
            r = await front._lc(engine).request(
                "POST", engine.url + path, headers=front._eh(engine, headers),
                content=json.dumps(part).encode())
            return (engine, r.status_code, front._rewrite_node_files(engine, r.content, request),
                    r.headers.get("content-type", ""))
        except Exception as exc:
            return engine, 502, json.dumps({"error": f"{engine.name} unreachable: {exc}"}).encode(), "application/json"
        finally:
            engine.exit_request(rid)

    t0 = time.time()
    results = await asyncio.gather(*[_one(e, p) for e, p in zip(engines, parts)])
    # A failed part is retried once on another engine before the whole
    # request fails — a node that just went away should not cost the answer.
    fixed = []
    for i, (eng, status, content, ctype) in enumerate(results):
        if 200 <= status < 300:
            fixed.append((status, content, ctype))
            continue
        others = [e for e in engines if e is not eng and e.healthy]
        if others:
            print(f"[fanout] part {i + 1}/{len(parts)} failed on {eng.name} ({status}); "
                  f"retrying on {others[0].name}", flush=True)
            _e, status, content, ctype = await _one(others[0], parts[i])
        if not (200 <= status < 300):
            return FanoutResult(status, content, ctype or "application/json")
        fixed.append((status, content, ctype))
    try:
        parsed = [json.loads(c) for _s, c, _t in fixed]
    except Exception as exc:
        return FanoutResult(502, json.dumps({"error": f"fan-out part answered non-JSON: {exc}"}).encode(),
                            "application/json")
    merged = merge_json(kind, path, body, parsed)
    print(f"[fanout] {path} {model}: {len(parts)} parts over "
          f"{', '.join(e.name for e in engines)} in {time.time() - t0:.1f}s", flush=True)
    return FanoutResult(200, json.dumps(merged).encode(), "application/json")


async def run_speech(front, request, path: str, body: dict, engines: list,
                     model: str, dcfg: dict) -> Optional["FanoutResult"]:
    """TTS: sentences over engines, spoken parts joined with ffmpeg."""
    if not ffmpeg_available():
        return None
    parts = split_json("text:input", body, len(engines), dcfg["min_items"])
    if not parts:
        return None
    engines = engines[: len(parts)]
    want_fmt = str(body.get("response_format") or "mp3")
    headers = {"content-type": "application/json"}
    if request.headers.get("authorization"):
        headers["authorization"] = request.headers["authorization"]

    async def _one(engine, part):
        part = dict(part)
        part["response_format"] = "wav"        # lossless parts, one final encode
        rid = engine.enter_request({"model": model, "kind": "tts", "path": path,
                                    "fanout": f"{len(parts)} parts"})
        try:
            r = await front._lc(engine).request(
                "POST", engine.url + path, headers=front._eh(engine, headers),
                content=json.dumps(part).encode())
            return r.status_code, r.content
        except Exception as exc:
            return 502, json.dumps({"error": f"{engine.name} unreachable: {exc}"}).encode()
        finally:
            engine.exit_request(rid)

    results = await asyncio.gather(*[_one(e, p) for e, p in zip(engines, parts)])
    for status, content in results:
        if not (200 <= status < 300):
            return FanoutResult(status, content, "application/json")
    workdir = tempfile.mkdtemp(prefix="coderai-fanout-")
    try:
        files = []
        for i, (_s, content) in enumerate(results):
            p = os.path.join(workdir, f"part{i:03d}.wav")
            with open(p, "wb") as f:
                f.write(content)
            files.append(p)
        joined = await asyncio.to_thread(concat_audio, files, want_fmt, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    media = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac",
             "opus": "audio/ogg", "aac": "audio/aac"}.get(want_fmt.lower(), "audio/mpeg")
    return FanoutResult(200, joined, media)


async def run_transcription(front, request, path: str, engines: list, model: str,
                            dcfg: dict) -> Optional["FanoutResult"]:
    """A long recording cut at silences into windows, one per engine."""
    if not ffmpeg_available() or len(engines) < 2:
        return None
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return None
    fmt = str(form.get("response_format") or "json")
    if fmt not in ("json", "verbose_json", "text"):
        return None                       # srt/vtt: let one engine number the cues
    if str(form.get("diarize") or "").lower() in ("1", "true", "yes"):
        return None                       # speaker labels must be consistent across the file
    workdir = tempfile.mkdtemp(prefix="coderai-fanout-")
    try:
        src = os.path.join(workdir, "input" + os.path.splitext(upload.filename or "a.wav")[1])
        with open(src, "wb") as f:
            f.write(await upload.read())
        windows = await asyncio.to_thread(audio_cut_points, src, float(dcfg["chunk_seconds"]),
                                          len(engines))
        if len(windows) < 2:
            return None
        files = await asyncio.to_thread(cut_audio, src, windows, workdir)
        fields = {k: v for k, v in form.multi_items()
                  if k != "file" and isinstance(v, str)}
        fields["response_format"] = "verbose_json"

        async def _one(engine, fpath):
            rid = engine.enter_request({"model": model, "kind": "transcription", "path": path,
                                        "fanout": f"{len(files)} chunks"})
            try:
                with open(fpath, "rb") as fh:
                    r = await front._lc(engine).request(
                        "POST", engine.url + path,
                        headers=front._eh(engine, {k: v for k, v in request.headers.items()
                                                   if k.lower() == "authorization"}),
                        data=fields, files={"file": (os.path.basename(fpath), fh, "audio/wav")})
                return r.status_code, r.content
            except Exception as exc:
                return 502, json.dumps({"error": f"{engine.name} unreachable: {exc}"}).encode()
            finally:
                engine.exit_request(rid)

        engines = [engines[i % len(engines)] for i in range(len(files))]
        results = await asyncio.gather(*[_one(e, f) for e, f in zip(engines, files)])
        parts = []
        for status, content in results:
            if not (200 <= status < 300):
                return FanoutResult(status, content, "application/json")
            try:
                parts.append(json.loads(content))
            except Exception:
                parts.append({"text": content.decode("utf-8", "replace")})
        merged = merge_transcriptions(fmt, parts, windows)
        if fmt == "text":
            return FanoutResult(200, (merged.get("text") or "").encode(), "text/plain")
        return FanoutResult(200, json.dumps(merged).encode(), "application/json")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def run_ocr_batch(front, request, path: str, engines: list, model: str,
                        dcfg: dict) -> Optional["FanoutResult"]:
    """A batch of documents, one document per part, results in order."""
    form = await request.form()
    files = [v for k, v in form.multi_items() if k == "files" and hasattr(v, "read")]
    if len(files) < max(2, dcfg["min_items"]) or len(engines) < 2:
        return None
    fields = {k: v for k, v in form.multi_items() if k != "files" and isinstance(v, str)}
    auth = {k: v for k, v in request.headers.items() if k.lower() == "authorization"}

    async def _one(engine, group):
        rid = engine.enter_request({"model": model or "ocr", "kind": "ocr", "path": path,
                                    "fanout": f"{len(files)} files"})
        try:
            payload = []
            for up in group:
                payload.append(("files", (up.filename, await up.read(), up.content_type or "application/octet-stream")))
            r = await front._lc(engine).request(
                "POST", engine.url + path, headers=front._eh(engine, auth),
                data=fields, files=payload)
            return r.status_code, r.content
        except Exception as exc:
            return 502, json.dumps({"error": f"{engine.name} unreachable: {exc}"}).encode()
        finally:
            engine.exit_request(rid)

    sizes = _even(len(files), len(engines))
    groups, i = [], 0
    for sz in sizes:
        groups.append(files[i:i + sz])
        i += sz
    results = await asyncio.gather(*[_one(e, g) for e, g in zip(engines, groups) if g])
    out, key = None, "results"
    merged = []
    for status, content in results:
        if not (200 <= status < 300):
            return FanoutResult(status, content, "application/json")
        d = json.loads(content)
        if out is None:
            out = dict(d)
            key = "results" if "results" in d else ("data" if "data" in d else "results")
        merged.extend(d.get(key) or [])
    out[key] = merged
    return FanoutResult(200, json.dumps(out).encode(), "application/json")
