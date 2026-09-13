#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Mux-engine service — hosts a colibri or kimi-k3-in-c engine over HTTP.

colibri and k3 are the two engines coderai drives DIRECTLY over a stdin/stdout
mux protocol rather than through an HTTP server, which is exactly what stopped
them from being pointed at a remote box the way every other worker can be. This
service is the missing HTTP boundary: it runs a real ``MuxEngine`` locally and
exposes it, so a remote host (a rented pod, another machine with the container
on fast storage) can serve the engine while coderai stays a thin client.

The client is :class:`codai.api.mux_remote.RemoteMuxEngine`, which implements
the same surface the backends use (``run``/``is_alive``/``pause``/``resume``/
``close``/``family``), so nothing downstream can tell local from remote.

  GET  /health   -> {"ok": true, "family": "glm", "alive": true}
  POST /run      -> NDJSON stream: {"t": "…"} chunks, then {"stats": {…}}
                    (body: {prompt, max_tokens, temperature, top_p})
  POST /pause    -> {"ok": true}     release the engine's VRAM
  POST /resume   -> {"ok": true}

Cancellation is the dropped connection: when the client stops reading, the
write fails and generation is cancelled — the same semantics a local engine
gets from its ``cancelled`` callback.
"""

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENGINE = None
KIND = "colibri"
_LOCK = threading.Lock()


def _build_engine(kind: str, cfg: dict, model_dir: str, ctx: int):
    """Start the real engine through the normal worker path."""
    ns = SimpleNamespace(**(cfg or {}))
    if kind == "k3":
        from codai.api import k3_worker as worker
    else:
        from codai.api import colibri_worker as worker
    return worker.ensure_engine(ns, model_dir=model_dir or None, ctx=ctx or None)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):   # quieter than the stdlib default
        pass

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") != "/health":
            return self._json(404, {"error": "not found"})
        alive = bool(ENGINE and ENGINE.is_alive())
        self._json(200, {"ok": alive, "kind": KIND, "alive": alive,
                         "family": getattr(ENGINE, "family", "")})

    def do_POST(self):
        path = self.path.rstrip("/")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except Exception as exc:
            return self._json(400, {"error": f"bad request body: {exc}"})

        if path in ("/pause", "/resume"):
            try:
                getattr(ENGINE, path[1:])()
            except Exception as exc:
                return self._json(500, {"error": str(exc)})
            return self._json(200, {"ok": True})
        if path != "/run":
            return self._json(404, {"error": "not found"})
        if ENGINE is None or not ENGINE.is_alive():
            return self._json(503, {"error": "engine is not running: "
                                             + (ENGINE.log_tail() if ENGINE else "not started")})

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        dead = threading.Event()

        def _emit(obj):
            line = (json.dumps(obj) + "\n").encode()
            try:
                self.wfile.write(b"%x\r\n" % len(line) + line + b"\r\n")
                self.wfile.flush()
            except Exception:
                # Client hung up — that is the cancel signal.
                dead.set()

        # One generation at a time: a mux engine's slots are its own business, but
        # this service exists to front a single engine for a single coderai.
        with _LOCK:
            try:
                stats = ENGINE.run(
                    req.get("prompt") or "",
                    int(req.get("max_tokens") or 1024),
                    float(req.get("temperature") or 0.0),
                    float(req.get("top_p") or 1.0),
                    on_text=lambda t: _emit({"t": t}),
                    cancelled=dead.is_set,
                )
                _emit({"stats": stats or {}})
            except Exception as exc:
                _emit({"error": f"{type(exc).__name__}: {exc}"})
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass


def main():
    global ENGINE, KIND
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=("colibri", "k3"),
                    default=os.environ.get("CODERAI_MUX_KIND", "colibri"))
    ap.add_argument("--host", default=os.environ.get("CODERAI_MUX_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CODERAI_MUX_PORT", "8092")))
    ap.add_argument("--model-dir", default=os.environ.get("CODERAI_MUX_MODEL_DIR", ""),
                    help="container/checkpoint directory the engine serves")
    ap.add_argument("--ctx", type=int, default=int(os.environ.get("CODERAI_MUX_CTX", "0") or 0))
    ap.add_argument("--config-json", default=os.environ.get("CODERAI_MUX_CONFIG", "{}"),
                    help="engine config fields (the ds4/colibri/k3 config block) as JSON")
    args = ap.parse_args()

    KIND = args.kind
    try:
        cfg = json.loads(args.config_json or "{}")
    except Exception as exc:
        print(f"--config-json is not valid JSON: {exc}", file=sys.stderr)
        return 2
    print(f"[mux-service] starting {args.kind} engine on {args.model_dir or '(config)'} …",
          flush=True)
    ENGINE = _build_engine(args.kind, cfg, args.model_dir, args.ctx)
    print(f"[mux-service] {args.kind} ready, serving on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    sys.exit(main() or 0)
