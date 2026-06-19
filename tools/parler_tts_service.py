#!/usr/bin/env python3
"""Standalone Parler-TTS HTTP microservice — run in its OWN venv.

parler-tts hard-pins an old transformers/tokenizers/huggingface-hub that conflict
with the coderai server's stack (transformers 5.x). So instead of polluting that
environment, Parler runs here behind a tiny stdlib HTTP shim, and coderai talks to
it as a remote TTS backend (``_RemoteParlerBackend``, selected when a model's
config carries a ``service_url``).

Setup (separate venv!):

    python3 -m venv ~/.venvs/parler
    source ~/.venvs/parler/bin/activate
    pip install "git+https://github.com/huggingface/parler-tts.git" soundfile
    python tools/parler_tts_service.py \
        --model parler-tts/parler-tts-mini-multilingual --port 8123

Then point a coderai TTS model's config at it, e.g. in models.json:

    "tts:parler-tts/parler-tts-mini-multilingual": {"service_url": "http://127.0.0.1:8123"}

Endpoints:
    GET  /health  -> {"ok": true, "model": ..., "sampling_rate": N}
    POST /speak   -> audio/wav   (body: {text, voice, speed, emotion, style, description?})
"""

import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf


# These mirror the editor's gated controls. coderai surfaces the same lists via
# codai.api.tts_backends._FAMILY_{EMOTIONS,STYLES}["parler"].
EMOTIONS = ["neutral", "happy", "sad", "angry", "excited", "calm", "fearful"]
STYLES = ["normal", "whispering", "shouting", "monotone", "expressive"]


def build_description(voice: str, speed, emotion: str, style: str, speaker: str = "") -> str:
    """Map the UI controls into a Parler natural-language delivery description."""
    spk = (voice or "").strip()
    if spk and ("/" in spk or spk.lower().startswith(("af_", "am_", "bf_", "bm_"))):
        spk = ""  # a path or a Kokoro id is not a Parler speaker name
    who = spk or speaker or "A speaker"
    bits = [f"{who} speaks"]
    if emotion and emotion != "neutral":
        bits.append(f"in a {emotion} tone")
    smap = {"whispering": "whispering softly", "shouting": "shouting loudly",
            "monotone": "in a flat monotone", "expressive": "in a very expressive, animated way"}
    if style and style not in ("", "normal"):
        bits.append(smap.get(style, style))
    try:
        sp = float(speed or 1.0)
    except (TypeError, ValueError):
        sp = 1.0
    bits.append(f"at a {'slow' if sp < 0.9 else 'fast' if sp > 1.15 else 'moderate'} pace")
    return (" ".join(bits) +
            ". The recording is very high quality, the voice clear and close up "
            "with no background noise.")


class _Engine:
    """Loads the Parler model once and synthesizes to a float waveform."""

    def __init__(self, model_name: str):
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        import torch
        self.model_name = model_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self._device)
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self.sr = int(self._model.config.sampling_rate)

    def speak(self, text: str, description: str) -> np.ndarray:
        ids = self._tok(description, return_tensors="pt").input_ids.to(self._device)
        prompt = self._tok(text, return_tensors="pt").input_ids.to(self._device)
        gen = self._model.generate(input_ids=ids, prompt_input_ids=prompt)
        return np.asarray(gen.cpu().numpy().squeeze(), dtype=np.float32)


ENGINE: _Engine = None  # set in main()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt, *args):  # quieter logs
        pass

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            self._send(200, json.dumps(
                {"ok": True, "model": ENGINE.model_name, "sampling_rate": ENGINE.sr}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path.split("?")[0] != "/speak":
            self._send(404, b'{"error":"not found"}')
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            text = (req.get("text") or "").strip()
            if not text:
                self._send(400, b'{"error":"empty text"}')
                return
            desc = req.get("description") or build_description(
                req.get("voice", ""), req.get("speed", 1.0),
                req.get("emotion", ""), req.get("style", ""))
            audio = ENGINE.speak(text, desc)
            buf = io.BytesIO()
            sf.write(buf, audio, ENGINE.sr, format="WAV")
            self._send(200, buf.getvalue(), ctype="audio/wav")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, json.dumps({"error": str(e)}).encode())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Standalone Parler-TTS HTTP service")
    ap.add_argument("--model", default="parler-tts/parler-tts-mini-multilingual")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args(argv)

    global ENGINE
    print(f"Loading {args.model} …")
    ENGINE = _Engine(args.model)
    print(f"Ready: {args.model} @ {ENGINE.sr} Hz — serving on http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
