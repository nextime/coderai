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
    GET  /status           {state, round, step, examples_trained,
                            asked, skipped_oom, base, error}
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
          "skipped_oom": 0, "error": "", "adapter": "", "base": {},
          #: How many examples THIS POD encoded, and how many it was asked
          #: for. The orchestrator cannot otherwise tell whether a slice
          #: landed: it sends indices into its own ordering and had no way to
          #: check that the pod's ordering is the same length.
          "encoded": 0, "asked": 0}
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
            seed=int(st.get("seed", 42)),
            objective=str(st.get("objective", "sft")),
            decision_letters=tuple(st.get("decision_letters") or ()),
            decision_loss_scope=str(st.get("decision_loss_scope", "letters")),
            target_suffixes=tuple(st.get("target_suffixes") or ()),
            quantization=str(st.get("quantization", "nf4") or "nf4"),
            lr_scheduler_type=str(st.get("lr_scheduler_type", "constant")),
            warmup_ratio=float(st.get("warmup_ratio", 0.0) or 0.0),
            lr_total_steps=int(st.get("lr_total_steps", 0) or 0),
            lr_step_offset=int(st.get("lr_step_offset", 0) or 0),
            weight_decay=float(st.get("weight_decay", 0.01)))
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
                                     enable_thinking=settings.enable_thinking,
                                     objective=settings.objective)
            _cache["examples"] = enc
            _set(encoded=len(enc))
            print(f"[pod] encoded: {stats.describe()}", flush=True)

        # REFUSED, NOT FILTERED. This was a comprehension with
        # `if i < len(_cache["examples"])`, which dropped every index past
        # the end of this pod's own encoding SILENTLY: the pod then trained
        # whatever survived and reported that as `examples_trained`, so a pod
        # whose encoded set disagreed with the orchestrator's looked like a
        # pod that had simply done less work. Measured on a live run: an
        # exchange averaged 297 examples where ~2000 had been handed out, and
        # nothing anywhere said why.
        #
        # The two sides must agree on the ordering or the indices mean
        # nothing, so a disagreement is fatal and says both counts.
        _want = list(payload.get("indices") or [])
        _have = len(_cache["examples"])
        _out = [i for i in _want if i >= _have]
        if _out:
            raise RuntimeError(
                f"this round asked for {len(_want)} example(s) but "
                f"{len(_out)} of them are past the end of this pod's "
                f"encoded set ({_have} examples, highest index asked for "
                f"{max(_want)}): the orchestrator and this pod disagree "
                f"about the dataset, so the indices do not mean the same "
                f"thing on both sides")
        mine = [_cache["examples"][i] for i in _want]
        if not mine:
            raise RuntimeError("no examples in this round's slice")
        # SAID OUT LOUD, every round. Pod stdout is not collected, so this is
        # also reported through /status below: the pod's own example count is
        # the only way to tell "trained its slice" from "trained a tenth of
        # it", and it was the one number that never reached the orchestrator.
        print(f"[pod] round {rnd}: {len(mine)} example(s) of this pod's "
              f"{_have} encoded", flush=True)

        out_dir = os.path.join(WORK, f"round{rnd}")
        start = os.path.join(WORK, "start.safetensors")
        _set(state="training", round=rnd, step=0, examples_trained=0,
             skipped_oom=0, adapter="", error="", encoded=_have,
             asked=len(mine))
        def _on_progress(**kw):
            # PASSES THE FOOTPRINT THROUGH. This used to be
            # `lambda **kw: _set(step=...)`, which dropped everything else --
            # so when the trainer began reporting what the card was actually
            # holding, the report died here instead of reaching `/status`.
            # Pod stdout is collected nowhere, so anything this drops is lost
            # for good.
            fields = {"step": int(kw.get("step") or 0)}
            if kw.get("base"):
                fields["base"] = kw["base"]
            if kw.get("status"):
                fields["phase"] = str(kw["status"])
            _set(**fields)

        result = tlt.run_training(
            base=base, encoded=mine, settings=settings, out_dir=out_dir,
            resume=False,
            init_adapter=(start if os.path.isfile(start) else ""),
            on_progress=_on_progress)
        adapter = os.path.join(out_dir, "adapter_model.safetensors")
        if not os.path.isfile(adapter):
            raise RuntimeError(f"training produced no adapter in {out_dir}")
        # `skipped_oom` IS REPORTED. `run_training` has always returned it and
        # the pod kept it to itself, so the one path that drops an example
        # without failing the round was invisible to the orchestrator: a fleet
        # training 4 of every 29 examples it was asked for said only "done",
        # and the cause had to be inferred from the averaging weights.
        _set(state="done", adapter=adapter,
             examples_trained=int(result.get("examples_trained") or 0),
             skipped_oom=int(result.get("skipped_oom") or 0),
             base=result.get("base") or {})
        print(f"[pod] round {rnd} finished: {result}", flush=True)
    except Exception as exc:
        # Reported rather than only logged: the orchestrator is polling, and a
        # pod that dies quietly is indistinguishable from one that is slow.
        _set(state="error", error=f"{type(exc).__name__}: {exc}"[:400])
        traceback.print_exc()


def _engine_state() -> dict:
    """What `/internal/engine-state` means for a TRAINING pod.

    The shape is the collector's, not a new one: `vram.devices[]` with FREE
    and total per card (free, because that is what a context-free query
    returns and the page derives used from the difference), plus a `load`
    block for host RAM and CPU. Fields the collector does not find are simply
    absent -- it draws a dash for what it was not told rather than inventing a
    zero -- so a partial answer is better than none and an exception here must
    never cost the training run.

    `requests_done` and the queue counters have no meaning on a trainer, so
    they are left out rather than reported as zero: zero served requests would
    read as an idle serving pod, which is the opposite of the truth.
    """
    out: dict = {}
    try:
        import torch
        if torch.cuda.is_available():
            devs = []
            for i in range(torch.cuda.device_count()):
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    # GIGABYTES, rounded, exactly as codai.api.app builds this
                    # block. mem_get_info returns BYTES, and the collector's
                    # fields are named *_gb and are not converted anywhere --
                    # so reporting bytes put a number 1e9 too large on the
                    # page under a GB label. Caught by feeding this payload
                    # through the real collector rather than by reading it.
                    d = {"index": i,
                         "name": torch.cuda.get_device_name(i),
                         "free": round(free / 1e9, 2),
                         "total": round(total / 1e9, 2)}
                    try:
                        # Utilisation and temperature are best-effort: present
                        # on a pod whose driver exposes them, absent otherwise,
                        # which the page already handles.
                        import pynvml
                        pynvml.nvmlInit()
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        d["util"] = int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                        d["temp"] = int(pynvml.nvmlDeviceGetTemperature(
                            h, pynvml.NVML_TEMPERATURE_GPU))
                    except Exception:
                        pass
                    devs.append(d)
                except Exception:
                    continue
            if devs:
                # The totals beside the devices, as the canonical block has
                # them: a reader that looks at `vram.total` rather than
                # walking `devices` gets the same answer.
                out["vram"] = {
                    "devices": devs, "device_count": len(devs),
                    "free": round(sum(d["free"] for d in devs), 2),
                    "total": round(sum(d["total"] for d in devs), 2),
                    "used": round(sum(d["total"] - d["free"] for d in devs), 2),
                    "gpu": (devs[0].get("name") if len(devs) == 1
                            else f"{len(devs)}x GPU")}
    except Exception:
        pass
    load: dict = {}
    try:
        with open("/proc/meminfo") as fh:
            mem = {}
            for line in fh:
                k, _, rest = line.partition(":")
                mem[k.strip()] = float(rest.strip().split()[0]) * 1024.0
        total = mem.get("MemTotal") or 0.0
        avail = mem.get("MemAvailable") or 0.0
        if total:
            load["ram_total_gb"] = round(total / 1e9, 2)
            load["ram_free_gb"] = round(avail / 1e9, 2)
            load["ram_used_gb"] = round((total - avail) / 1e9, 2)
    except Exception:
        pass
    try:
        load["cpu_cores"] = int(os.cpu_count() or 0)
        # From /proc/stat deltas against the last call, because a single read
        # gives totals since boot and would report the same number forever.
        global _CPU_LAST
        with open("/proc/stat") as fh:
            parts = fh.readline().split()[1:]
        vals = [float(x) for x in parts[:8]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)
        busy = sum(vals) - idle
        prev = _CPU_LAST
        _CPU_LAST = (busy, idle)
        if prev:
            db, di = busy - prev[0], idle - prev[1]
            if (db + di) > 0:
                load["cpu_util"] = round(100.0 * db / (db + di), 1)
    except Exception:
        pass
    if load:
        out["load"] = load
    return out


#: Last (busy, idle) jiffies, for the CPU delta above.
_CPU_LAST = None


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
        if self.path == "/internal/engine-state":
            # The SAME endpoint every coderai pod serves, because the admin
            # page's telemetry is pulled from the pods themselves and asks for
            # exactly this path. Without it a training pod showed dashes for
            # VRAM, RAM, GPU and CPU for its whole paid life -- the operator
            # could see a card costing money and not what it was doing.
            return self._json(200, _engine_state())
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
            if snap.get("state") in ("starting", "training", "encoding",
                                     "loading-tokenizer"):
                # Refused, not queued: two rounds in flight would hand back an
                # adapter from the wrong one. `starting` counts as working for
                # the same reason the others do -- it is set below, before the
                # thread exists, so two overlapping POSTs cannot both start.
                return self._json(409, {"error": "already working",
                                        "state": snap.get("state"),
                                        "round": snap.get("round")})
            try:
                payload = json.loads(self._body() or b"{}")
            except Exception as exc:
                return self._json(400, {"error": f"bad JSON: {exc}"})
            # SYNCHRONOUSLY, BEFORE THE THREAD. This used to answer 202 having
            # touched nothing, so until the worker got as far as its first
            # `_set` the status still read `state=done, round=<this round>`
            # with the previous block's example count -- and that is exactly
            # what the orchestrator collects on. It therefore fetched the
            # already-collected adapter a second time and credited it to the
            # block just handed out, which no gradient step had touched.
            # Measured on the run that found it: blocks 90 and 110 passed
            # between four shards for an hour, and 139 blocks were reported
            # covered on 384 of 7,308 examples trained.
            _set(state="starting", round=int(payload.get("round") or 0),
                 step=0, examples_trained=0, skipped_oom=0,
                 adapter="", error="")
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
