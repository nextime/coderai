#!/usr/bin/env python3
# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# Licensed under the GNU General Public License v3 or later.
"""What a rented training pod runs: a small server the orchestrator drives.

Baked into the training image, so the image tag pins the runner and a pod has
no boot-time dependency on anything but its own filesystem. Fetching it would
have meant a second pinning mechanism and an outage whenever the place it was
fetched from had one.

The orchestrator PULLS. coderai already addresses pods -- it proxies inference
to them -- and ``wait_for_url(..., direct_tcp=True)`` hands back a usable
``ip:port`` that bypasses the RunPod proxy, so there is no Cloudflare idle
limit or body cap in the way of a 160 MB adapter. That removes the need for the
pod to know any address at all: nothing here is configured with a URL, and a
wrong one cannot strand a fleet.

    POST /dataset          the training set, once
    POST /round            {round, indices, settings, base} -> start training
    PUT  /adapter/start    the previous round's average, to resume from
    GET  /status           {state, round, step, examples_trained, error}
    GET  /adapter          the finished round's adapter

Training runs in a worker thread so ``/status`` keeps answering while the GPU
is busy -- which is the point: the orchestrator can tell "step 40 of 300" from
"gone", where silence cannot.
"""

import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("CODERAI_TRAIN_PORT") or "8000")
SECRET = os.environ.get("CODERAI_TRAIN_SECRET") or ""
WORK = os.environ.get("CODERAI_TRAIN_WORKDIR") or "/workspace/train"
DATASET = os.path.join(WORK, "dataset.jsonl")

_lock = threading.Lock()
_state = {"state": "idle", "round": -1, "step": 0, "examples_trained": 0,
          "error": "", "adapter": ""}
_worker = None
#: The encoded dataset, kept between rounds: encoding 7000 examples takes
#: minutes and the file never changes, so paying it once matters when a round
#: is twenty.
_cache = {"examples": None, "tokenizer": None, "base": ""}


def _set(**kw):
    with _lock:
        _state.update(kw)


def _snapshot():
    with _lock:
        return dict(_state)


def _train(payload):
    """One round, in a thread. Every exit path writes a terminal state."""
    import sys
    sys.path.insert(0, "/opt/trainer")
    from codai.api import text_lora_dataset as tld
    from codai.api import text_lora_trainer as tlt
    try:
        rnd = int(payload["round"])
        st = payload.get("settings") or {}
        settings = tlt.QLoraSettings(
            rank=int(st.get("rank", 16)), alpha=int(st.get("alpha", 32)),
            dropout=float(st.get("dropout", 0.05)),
            learning_rate=float(st.get("learning_rate", 1e-4)),
            max_len=int(st.get("max_len", 8192)),
            accumulation_steps=int(st.get("accumulation_steps", 8)),
            enable_thinking=bool(st.get("enable_thinking", False)),
            seed=int(st.get("seed", 42)))
        base = payload.get("base") or _cache["base"]
        if not base:
            raise RuntimeError("no base model was given")

        if _cache["tokenizer"] is None or _cache["base"] != base:
            from transformers import AutoTokenizer
            _set(state="loading-tokenizer")
            _cache["tokenizer"] = AutoTokenizer.from_pretrained(
                base, trust_remote_code=True)
            _cache["base"] = base
            _cache["examples"] = None

        if _cache["examples"] is None:
            if not os.path.isfile(DATASET):
                raise RuntimeError("no dataset has been uploaded")
            _set(state="encoding")
            rows = list(tld.load_jsonl(DATASET,
                                       field_map=payload.get("field_map") or None))
            enc, stats = tld.prepare(_cache["tokenizer"], rows,
                                     max_len=settings.max_len,
                                     enable_thinking=settings.enable_thinking)
            _cache["examples"] = enc
            print(f"[pod] encoded: {stats.describe()}", flush=True)

        mine = [_cache["examples"][i] for i in payload.get("indices") or []
                if i < len(_cache["examples"])]
        if not mine:
            raise RuntimeError("no examples in this round's slice")

        out_dir = os.path.join(WORK, f"round{rnd}")
        start = os.path.join(WORK, "start.safetensors")
        _set(state="training", round=rnd, step=0, examples_trained=0,
             adapter="", error="")
        result = tlt.run_training(
            base=base, encoded=mine, settings=settings, out_dir=out_dir,
            resume=False,
            init_adapter=(start if os.path.isfile(start) else ""),
            on_progress=lambda **kw: _set(step=int(kw.get("step") or 0)))
        adapter = os.path.join(out_dir, "adapter_model.safetensors")
        if not os.path.isfile(adapter):
            raise RuntimeError(f"training produced no adapter in {out_dir}")
        _set(state="done", adapter=adapter,
             examples_trained=int(result.get("examples_trained") or 0))
        print(f"[pod] round {rnd} finished: {result}", flush=True)
    except Exception as exc:
        # Reported rather than only logged: the orchestrator is polling, and a
        # pod that dies quietly is indistinguishable from one that is slow.
        _set(state="error", error=f"{type(exc).__name__}: {exc}"[:400])
        traceback.print_exc()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):      # one line, not two
        print(f"[pod] {self.address_string()} {fmt % args}", flush=True)

    # -- plumbing ------------------------------------------------------- #

    def _authed(self) -> bool:
        if not SECRET:
            return True
        sent = (self.headers.get("Authorization") or "").strip()
        if sent.lower().startswith("bearer "):
            sent = sent[7:].strip()
        import hmac
        ok = hmac.compare_digest(sent or "", SECRET)
        if not ok:
            self._json(401, {"error": "bad credential"})
        return ok

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    # -- the API -------------------------------------------------------- #

    def do_GET(self):
        if self.path == "/healthz":
            return self._json(200, {"ok": True})     # no credential: liveness
        if not self._authed():
            return
        if self.path == "/status":
            return self._json(200, _snapshot())
        if self.path == "/adapter":
            snap = _snapshot()
            path = snap.get("adapter") or ""
            if snap.get("state") != "done" or not os.path.isfile(path):
                return self._json(404, {"error": "no finished adapter",
                                        "state": snap.get("state")})
            size = os.path.getsize(path)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(path, "rb") as fh:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            return
        self._json(404, {"error": "no such path"})

    def do_PUT(self):
        if not self._authed():
            return
        if self.path == "/adapter/start":
            os.makedirs(WORK, exist_ok=True)
            data = self._body()
            with open(os.path.join(WORK, "start.safetensors"), "wb") as fh:
                fh.write(data)
            return self._json(200, {"ok": True, "bytes": len(data)})
        self._json(404, {"error": "no such path"})

    def do_POST(self):
        global _worker
        if not self._authed():
            return
        if self.path == "/dataset":
            os.makedirs(WORK, exist_ok=True)
            data = self._body()
            if not data:
                return self._json(400, {"error": "empty dataset"})
            tmp = DATASET + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, DATASET)        # never a half file to read back
            _cache["examples"] = None       # re-encode against the new bytes
            return self._json(200, {"ok": True, "bytes": len(data)})
        if self.path == "/round":
            snap = _snapshot()
            if snap.get("state") in ("training", "encoding", "loading-tokenizer"):
                # Refused, not queued: two rounds in flight would hand back an
                # adapter from the wrong one.
                return self._json(409, {"error": "already working",
                                        "state": snap.get("state"),
                                        "round": snap.get("round")})
            try:
                payload = json.loads(self._body() or b"{}")
            except Exception as exc:
                return self._json(400, {"error": f"bad JSON: {exc}"})
            _worker = threading.Thread(target=_train, args=(payload,), daemon=True)
            _worker.start()
            return self._json(202, {"ok": True, "round": payload.get("round")})
        self._json(404, {"error": "no such path"})


def main() -> int:
    os.makedirs(WORK, exist_ok=True)
    if not SECRET:
        print("[pod] WARNING: CODERAI_TRAIN_SECRET is unset; the API is open",
              flush=True)
    print(f"[pod] listening on 0.0.0.0:{PORT}, work dir {WORK}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
