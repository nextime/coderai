#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""MeloTTS speech service (runs inside the isolated MeloTTS venv).

MeloTTS is what OpenVoice V2 synthesises with before its tone-colour converter
re-paints the timbre. coderai uses it the same way — as the base voice of the
`chain` cloning engine — and as a plain multilingual TTS engine.

Same contract as tools/parler_tts_service.py so the shared remote backend in
codai/api/tts_backends.py can drive either:

  GET  /health  -> {"ok": true, "model": …, "languages": […]}
  POST /speak   -> audio/wav   (body: {text, voice, speed, language})

`voice` selects a speaker within the language pack (MeloTTS ships several per
language, e.g. EN-US / EN-BR / EN-Default); unknown names fall back to the first.
"""

import argparse
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

ENGINE = None

#: MeloTTS ships one model per language; the request's `language` picks the pack.
LANGUAGES = ("EN", "ES", "FR", "ZH", "JP", "KR")

#: Two-letter codes the rest of coderai speaks -> MeloTTS language packs.
LANG_MAP = {"en": "EN", "es": "ES", "fr": "FR", "zh": "ZH", "ja": "JP",
            "jp": "JP", "ko": "KR", "kr": "KR"}


class _Engine:
    """Lazily loads one TTS model per language and keeps them around."""

    def __init__(self, device: str = "auto"):
        self._device = device
        self._models = {}
        self.sr = 44100

    def _model_for(self, language: str):
        lang = LANG_MAP.get((language or "en").lower()[:2], "EN")
        if lang not in self._models:
            from melo.api import TTS
            print(f"Loading MeloTTS language pack {lang} …", flush=True)
            model = TTS(language=lang, device=self._device)
            self._models[lang] = model
            self.sr = int(getattr(model, "hps", None).data.sampling_rate)
        return self._models[lang], lang

    def speakers(self, language: str = "EN") -> list:
        model, _ = self._model_for(language)
        return list(getattr(model, "hps").data.spk2id.keys())

    def speak(self, text: str, voice: str, speed: float, language: str) -> np.ndarray:
        model, lang = self._model_for(language)
        spk2id = getattr(model, "hps").data.spk2id
        names = list(spk2id.keys())
        name = voice if voice in spk2id else names[0]
        buf = io.BytesIO()
        # MeloTTS writes through soundfile; ask for a buffer rather than a path.
        model.tts_to_file(text, spk2id[name], buf, speed=float(speed or 1.0),
                          format="WAV")
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32")
        self.sr = int(sr)
        return data


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, fmt, *args):   # quieter logs
        return

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            self._send(200, json.dumps({
                "ok": True, "model": "melotts", "languages": list(LANGUAGES),
                "speakers": ENGINE.speakers("EN") if ENGINE else [],
            }).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path.split("?")[0] != "/speak":
            self._send(404, b'{"error":"not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            text = (body.get("text") or "").strip()
            if not text:
                self._send(400, b'{"error":"text is required"}')
                return
            wav = ENGINE.speak(text, body.get("voice") or "",
                               body.get("speed") or 1.0,
                               body.get("language") or "en")
            buf = io.BytesIO()
            sf.write(buf, np.asarray(wav, dtype="float32"), ENGINE.sr, format="WAV")
            self._send(200, buf.getvalue(), "audio/wav")
        except Exception as exc:
            self._send(500, json.dumps({"error": str(exc)}).encode())


def main(argv=None):
    global ENGINE
    ap = argparse.ArgumentParser(description="MeloTTS speech service")
    ap.add_argument("--model", default="melotts", help="ignored; language picks the pack")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = ap.parse_args(argv)

    ENGINE = _Engine(args.device)
    print(f"MeloTTS service on http://{args.host}:{args.port} "
          f"(languages: {', '.join(LANGUAGES)})", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
