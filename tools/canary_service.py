#!/usr/bin/env python3
"""Standalone NVIDIA NeMo ASR HTTP microservice — run in its OWN venv.

NeMo (Canary / Parakeet) pins a stack that conflicts with coderai's transformers
5.x, so it runs here behind a tiny stdlib HTTP shim and coderai talks to it as a
remote STT backend (``codai.api.stt_backends._RemoteNemoBackend``), managed by
``codai.api.canary_worker``.

Endpoints:
    GET  /health      -> {"ok": true, "model": ..., "multitask": bool}
    POST /transcribe  -> {"text", "segments":[{start,end,text}], "language"}
                         body: raw audio bytes;
                         query: ?language=<src>&target_language=<tgt>

The uploaded audio is transcoded to 16 kHz mono WAV via ffmpeg before inference.
"""

import argparse
import json
import os
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


def _to_wav_16k_mono(raw: bytes) -> str:
    fd, src = tempfile.mkstemp(suffix=".input")
    os.write(fd, raw)
    os.close(fd)
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        p = subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1",
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


class _Engine:
    """Loads a NeMo ASR model once and transcribes audio files.

    Handles both Canary-style multitask models (source/target language +
    translation, ``EncDecMultiTaskModel``) and plain ASR models (Parakeet, etc.)."""

    def __init__(self, model_name: str, model_path: str = None):
        from nemo.collections.asr.models import ASRModel
        self.model_name = model_name
        if model_path and os.path.isfile(model_path):
            self._model = ASRModel.restore_from(model_path)
        else:
            self._model = ASRModel.from_pretrained(model_name)
        try:
            import torch
            if torch.cuda.is_available():
                self._model = self._model.cuda()
        except Exception:
            pass
        self._model.eval()
        # Multitask (Canary) models accept source_lang/target_lang.
        self.multitask = self._model.__class__.__name__ == "EncDecMultiTaskModel"

    def transcribe(self, wav_path: str, language: str = None,
                   target_language: str = None) -> dict:
        kwargs = {"timestamps": True}
        if self.multitask:
            src = language or "en"
            kwargs["source_lang"] = src
            kwargs["target_lang"] = target_language or src
            kwargs["pnc"] = "yes"
        try:
            out = self._model.transcribe([wav_path], **kwargs)
        except TypeError:
            # Older NeMo without timestamps kwarg.
            kwargs.pop("timestamps", None)
            out = self._model.transcribe([wav_path], **kwargs)
        hyp = out[0] if isinstance(out, (list, tuple)) and out else out
        text = getattr(hyp, "text", None)
        if text is None:
            text = hyp if isinstance(hyp, str) else str(hyp)
        segments, words = [], []
        ts = getattr(hyp, "timestamp", None)
        if isinstance(ts, dict):
            for seg in (ts.get("segment") or []):
                segments.append({
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": (seg.get("segment") or seg.get("text") or "").strip(),
                })
            for w in (ts.get("word") or []):
                words.append({
                    "word": (w.get("word") or w.get("text") or "").strip(),
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                })
        return {"text": (text or "").strip(), "segments": segments, "words": words,
                "language": target_language or language}


ENGINE = None  # set in main()


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
            self._send(200, json.dumps(
                {"ok": True, "model": ENGINE.model_name,
                 "multitask": ENGINE.multitask}).encode())
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
            language = (q.get("language") or [None])[0]
            target = (q.get("target_language") or [None])[0]
            wav = _to_wav_16k_mono(raw)
            result = ENGINE.transcribe(wav, language=language, target_language=target)
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
    ap = argparse.ArgumentParser(description="Standalone NVIDIA NeMo ASR HTTP service")
    ap.add_argument("--model", default="nvidia/canary-1b-flash")
    ap.add_argument("--model-path", default=None,
                    help="Local .nemo checkpoint (overrides --model)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8124)
    args = ap.parse_args(argv)

    global ENGINE
    print(f"Loading NeMo model {args.model_path or args.model} …")
    ENGINE = _Engine(args.model, args.model_path)
    print(f"Ready: {args.model} (multitask={ENGINE.multitask}) — "
          f"serving on http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
