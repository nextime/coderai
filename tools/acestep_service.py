#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""ACE-Step text-to-music, run in its own interpreter.

    POST /generate  -> audio/wav bytes
                       body: {"prompt", "lyrics", "duration", "steps",
                              "guidance_scale", "seed", "scheduler_type"}
    GET  /health    -> {"ok": true}

This file runs inside ``~/.coderai/acestep_venv`` (or the image's baked
/opt/coderai/acestep_venv), NEVER in the server's venv, and it must not import
anything from ``codai``. Two reasons it cannot share:

* ACE-Step pins ``transformers==4.50.0`` and ``spacy==3.8.4``; the server runs
  transformers 5.x. Installing ACE-Step into the server venv would downgrade
  it, exactly as parler-tts does.
* ``spacy==3.8.4`` has no cp313 wheel, so this venv is PYTHON 3.12 while the
  server is 3.13. They cannot be the same interpreter, let alone the same venv.

The pipeline writes .wav files and returns their paths; this reads the first
one back and returns the bytes, because that is the shape coderai's audio_gen
expects from every other backend.
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

ENGINE = None
_LOCK = threading.Lock()


class Engine:
    """Holds the loaded ACEStepPipeline. Loaded once, on first use."""

    def __init__(self, checkpoint_dir=None, device_id=0, dtype="bfloat16",
                 cpu_offload=False, quantized=False, overlapped_decode=False):
        self.kw = dict(checkpoint_dir=checkpoint_dir or None,
                       device_id=int(device_id), dtype=dtype,
                       cpu_offload=bool(cpu_offload),
                       quantized=bool(quantized),
                       overlapped_decode=bool(overlapped_decode))
        self.pipe = None

    def _ensure(self):
        if self.pipe is not None:
            return
        # `quantized` does not quantize anything: it loads a DIFFERENT set of
        # weights (ACE-Step's int4wo export) from the same directory layout. If
        # those files are not there the failure is a bare FileNotFoundError
        # minutes in, so say what is missing first -- before the import, which
        # pulls in torch and several seconds of it.
        ckpt = self.kw.get("checkpoint_dir")
        if self.kw.get("quantized") and ckpt:
            need = os.path.join(ckpt, "ace_step_transformer",
                                "diffusion_pytorch_model_int4wo.bin")
            if not os.path.isfile(need):
                raise RuntimeError(
                    "this model is configured quantized, but ACE-Step's "
                    f"int4 weights are not in {ckpt} (expected {need}). "
                    "Quantized ACE-Step is a separate upstream checkpoint, "
                    "not a conversion of this one: point the model's path at "
                    "it, or turn the quantization switch off.")
        from acestep.pipeline_ace_step import ACEStepPipeline
        sys.stderr.write(f"acestep: loading pipeline {self.kw}\n")
        sys.stderr.flush()
        self.pipe = ACEStepPipeline(**self.kw)
        # Constructing the pipeline loads NOTHING -- it sets `loaded = False`
        # and reads the checkpoints on the first __call__. Do it here, so the
        # VRAM this model needs is allocated while the parent is still
        # measuring the load (see main()).
        if self.kw.get("quantized"):
            self.pipe.load_quantized_checkpoint(self.pipe.checkpoint_dir)
        else:
            self.pipe.load_checkpoint(self.pipe.checkpoint_dir)
        sys.stderr.write("acestep: checkpoints loaded\n")
        sys.stderr.flush()

    def generate(self, body: dict) -> bytes:
        self._ensure()
        # manual_seeds is a LIST in this API; a bare int is silently ignored,
        # which would make `seed` look honoured while every call differed.
        seed = body.get("seed")
        seeds = None
        if seed is not None:
            seeds = [int(seed)] if not isinstance(seed, (list, tuple)) else list(seed)

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.wav")
            call = dict(
                format="wav",
                prompt=body.get("prompt") or "",
                # Empty string, not None: the pipeline treats lyrics as text and
                # None reaches the tokenizer.
                lyrics=body.get("lyrics") or "",
                audio_duration=float(body.get("duration") or 60.0),
                infer_step=int(body.get("steps") or 60),
                guidance_scale=float(body.get("guidance_scale") or 15.0),
                scheduler_type=str(body.get("scheduler_type") or "euler"),
                manual_seeds=seeds,
                save_path=out,
            )
            paths = self.pipe(**call)
            wavs = [p for p in (paths or [])
                    if isinstance(p, str) and p.lower().endswith(".wav")
                    and os.path.isfile(p)]
            if not wavs:
                raise RuntimeError(
                    f"ACE-Step produced no .wav (returned {paths!r})")
            with open(wavs[0], "rb") as f:
                return f.read()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt, *args):      # quieter: the parent pumps our stderr
        sys.stderr.write("acestep: " + (fmt % args) + "\n")

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._send(200, json.dumps({"ok": True}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path.rstrip("/") != "/generate":
            self._send(404, b'{"error":"not found"}')
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, json.dumps({"error": f"bad request: {e}"}).encode())
            return
        # One generation at a time: a single pipeline on one GPU, and two
        # concurrent calls would interleave on the same modules.
        try:
            with _LOCK:
                wav = ENGINE.generate(body)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())
            return
        self._send(200, wav, ctype="audio/wav")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--cpu-offload", action="store_true")
    ap.add_argument("--quantized", action="store_true")
    ap.add_argument("--overlapped-decode", action="store_true")
    a = ap.parse_args()

    # Honour the configured precision EXACTLY. The pipeline's own `dtype`
    # argument is a two-way switch -- `torch.bfloat16 if dtype == "bfloat16"
    # else torch.float32` -- so asking it for float16 quietly gets float32,
    # twice the VRAM on a 3.5B model. ACE_PIPELINE_DTYPE is read after that
    # switch and takes any torch dtype name, so it is the one that actually
    # works.
    import torch
    if not isinstance(getattr(torch, a.dtype, None), torch.dtype):
        sys.stderr.write(f"acestep: unknown dtype {a.dtype!r}\n")
        return 2
    os.environ["ACE_PIPELINE_DTYPE"] = a.dtype

    global ENGINE
    ENGINE = Engine(checkpoint_dir=a.checkpoint_dir, device_id=a.device_id,
                    dtype=a.dtype, cpu_offload=a.cpu_offload,
                    quantized=a.quantized,
                    overlapped_decode=a.overlapped_decode)
    # Load the weights BEFORE answering /health, as canary_service does. Not a
    # detail: coderai measures a model's VRAM as the device-wide free-memory
    # delta across the load, and that window closes when the parent sees this
    # process go healthy. Loading lazily on the first /generate would measure
    # ~0 GB, and eviction would then treat a resident 3.5B model as free and
    # over-commit the GPU. The parent watches for the process dying, so taking
    # minutes here is safe.
    ENGINE._ensure()
    srv = HTTPServer((a.host, a.port), Handler)
    sys.stderr.write(f"acestep: ready, listening on {a.host}:{a.port}\n")
    sys.stderr.flush()
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main() or 0)
