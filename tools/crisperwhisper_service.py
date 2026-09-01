#!/usr/bin/env python3
"""Standalone CrisperWhisper ASR HTTP microservice — run in its OWN venv.

CrisperWhisper does verbatim transcription with precise word timestamps, but its
generation config breaks on transformers 5.x. So it runs here in a venv pinned to
a compatible transformers, and coderai talks to it as a remote STT backend
(codai.api.crisperwhisper_worker).

Endpoints:
    GET  /health      -> {"ok": true, "model": ...}
    POST /transcribe  -> {"text", "words":[{word,start,end}], "segments", "language"}
                         body: raw audio bytes; query: ?language=&task=transcribe|translate
"""

import argparse
import json
import os
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


def _to_wav_16k_mono(raw: bytes) -> str:
    fd, src = tempfile.mkstemp(suffix=".input"); os.write(fd, raw); os.close(fd)
    fd, out = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    try:
        p = subprocess.run(["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1",
                            "-c:a", "pcm_s16le", "-f", "wav", out],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            raise RuntimeError("ffmpeg conversion failed: " +
                               p.stderr.decode("utf-8", "replace")[-300:])
        return out
    finally:
        try:
            os.unlink(src)
        except OSError:
            pass


def _adjust_pauses(chunks, split_threshold=0.12):
    """CrisperWhisper's pause redistribution for accurate verbatim word bounds."""
    out = [dict(c) for c in chunks]
    for i in range(len(out) - 1):
        cs, ce = out[i].get("timestamp") or (None, None)
        ns, ne = out[i + 1].get("timestamp") or (None, None)
        if None in (ce, ns):
            continue
        pause = ns - ce
        if pause > 0:
            d = (split_threshold / 2) if pause > split_threshold else (pause / 2)
            out[i]["timestamp"] = (cs, ce + d)
            out[i + 1]["timestamp"] = (ns - d, ne)
    return out


class _Engine:
    def __init__(self, model_name: str):
        from transformers import pipeline
        import torch
        self.model_name = model_name
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev == "cuda" else torch.float32
        # NOTE: no chunk_length_s — the pipeline's internal long-form word-timestamp
        # stitching crashes on CrisperWhisper (IndexError at the ~30s chunk
        # boundary). We window manually below so every forward pass is ≤ WINDOW_S
        # (one Whisper window), which is the path that works.
        self._pipe = pipeline(
            "automatic-speech-recognition", model=model_name,
            torch_dtype=dtype, device=dev)

    # Whisper's receptive field is 30s; stay a touch under so each window is a
    # single, non-chunked pass.
    WINDOW_S = 28.0

    def _one(self, arr, sr, gen, offset):
        """Transcribe one ≤30s window; return (text, [word dicts]) with timestamps
        shifted by `offset` seconds."""
        out = self._pipe({"raw": arr, "sampling_rate": sr},
                         return_timestamps="word", generate_kwargs=gen)
        text = (out.get("text") if isinstance(out, dict) else str(out)) or ""
        words = []
        for ch in _adjust_pauses(out.get("chunks") or []) if isinstance(out, dict) else []:
            ts = ch.get("timestamp") or (None, None)
            words.append({"word": (ch.get("text") or "").strip(),
                          "start": (ts[0] or 0.0) + offset,
                          "end": (ts[1] or 0.0) + offset})
        return text.strip(), words

    def transcribe(self, wav_path: str, language: str = None, task: str = None) -> dict:
        import soundfile as sf
        gen = {"task": task or "transcribe"}
        if language:
            gen["language"] = language
        data, sr = sf.read(wav_path, dtype="float32")
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        dur = len(data) / sr
        texts, words = [], []
        if dur <= 30.0:
            t, w = self._one(data, sr, gen, 0.0)
            texts.append(t); words.extend(w)
        else:
            # Manual long-form windowing (avoids the pipeline's buggy stitching).
            pos = 0.0
            while pos < dur:
                a = int(pos * sr)
                b = int(min(pos + self.WINDOW_S, dur) * sr)
                t, w = self._one(data[a:b], sr, gen, pos)
                if t:
                    texts.append(t)
                words.extend(w)
                pos += self.WINDOW_S
        text = " ".join(x for x in texts if x).strip()
        segments = []
        if words:
            segments = [{"start": words[0]["start"], "end": words[-1]["end"],
                         "text": text}]
        return {"text": text, "words": words, "segments": segments,
                "language": language}


ENGINE = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._send(200, json.dumps({"ok": True, "model": ENGINE.model_name}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/transcribe":
            self._send(404, b'{"error":"not found"}')
            return
        wav = None
        try:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b""
            if not raw:
                self._send(400, b'{"error":"empty audio"}')
                return
            q = parse_qs(parsed.query)
            def _q(k):
                v = q.get(k)
                return v[0] if v else None
            wav = _to_wav_16k_mono(raw)
            result = ENGINE.transcribe(wav, language=_q("language"), task=_q("task"))
            self._send(200, json.dumps(result).encode())
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, json.dumps({"error": str(e)}).encode())
        finally:
            if wav:
                try:
                    os.unlink(wav)
                except OSError:
                    pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Standalone CrisperWhisper HTTP service")
    ap.add_argument("--model", default="nyrahealth/CrisperWhisper")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8126)
    args = ap.parse_args(argv)
    global ENGINE
    print(f"Loading CrisperWhisper {args.model} …")
    ENGINE = _Engine(args.model)
    print(f"Ready: {args.model} — serving on http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
