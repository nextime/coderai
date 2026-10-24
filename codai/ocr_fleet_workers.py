# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""coderai-ocr-workers: N extra processes that forward /v1/ocr pages to pods.

WHY. One Python process forwards ~16-23 pages/s at most, however many pods it
has: measured on a 35-pod fleet, front 99% CPU, engine 101%, 14 of 16 cores
idle and the pod GPUs at 0-7% KV cache. 128 pods need ~85 pages/s.

WHAT. With ``ocr.workers`` (or ``CODERAI_OCR_WORKERS``) = N > 0 this starts N
``--engine-only`` processes on 127.0.0.1:18800+i, each with
``CODERAI_OCR_WORKER=i/N`` (see codai/api/runpod_fleet_share.py: a worker
follows the fleet the main engine rents and never rents, scales or kills), and
writes the nginx include files that send exact ``/v1/ocr`` and
``/v1/ocr/batch`` to them, least-connections, with the front as the backup.
Everything else keeps going through the front.

N = 0 (the default) writes empty includes: nginx routes exactly as before.

It is a supervisor in its own right: a worker that dies is taken out of the
upstream, restarted with a backoff, and put back once /healthz answers. The
config is re-read every CONFIG_EVERY_S so a changed N takes effect without a
container restart.

Stdlib only, on purpose: this runs beside the server and must not cost what it
is there to save.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PORT_BASE = int(os.environ.get("CODERAI_OCR_WORKER_PORT_BASE", "18800"))
FRONT = os.environ.get("CODERAI_OCR_FRONT_UPSTREAM", "127.0.0.1:18776")
INCLUDE_DIR = os.environ.get("CODERAI_NGINX_INCLUDE_DIR", "/tmp/coderai-nginx")
NGINX = os.environ.get("CODERAI_NGINX_BIN", "/usr/sbin/nginx")
NGINX_CONF = os.environ.get("CODERAI_NGINX_CONF", "/etc/nginx/nginx.conf")
PYTHON = os.environ.get("CODERAI_PYTHON", "/opt/coderai/python/bin/python3")
APP = os.environ.get("CODERAI_APP", "/opt/coderai/app/coderai")

CONFIG_EVERY_S = 15.0
POLL_S = 2.0
#: How long a worker may take to answer /healthz after starting. It loads the
#: whole app (no models), which on the orchestrator takes ~20-40 s.
START_GRACE_S = 180.0
#: How long a worker being retired may finish its in-flight pages. A page is
#: ~60 s on a pod; a page that has not finished in this is not going to.
DRAIN_S = 600.0
RESTART_BACKOFF_MAX_S = 60.0
#: Hard upper bound. Each worker is a full app process (~1-2 GB).
MAX_WORKERS = 32

UPSTREAM_FILE = "ocr-upstream.conf"
LOCATION_FILE = "ocr-locations.conf"


def log(msg: str) -> None:
    print(f"[ocr-workers] {msg}", flush=True)


def config_dir() -> str:
    return os.path.join(os.environ.get("CODERAI_CONFIG_DIR", "/config"), "coderai")


def configured_hf_cache_dir() -> str:
    """``models.hf_cache_dir`` from config.json, or "" when it is not set. The
    same value codai/hf_cache.py reads, read the same way."""
    try:
        with open(os.path.join(config_dir(), "config.json")) as fh:
            raw = (json.load(fh).get("models") or {}).get("hf_cache_dir")
    except Exception:
        return ""
    return os.path.expanduser(str(raw).strip()) if raw and str(raw).strip() else ""


def wanted_workers() -> int:
    """N from the env (wins) or config.json ``ocr.workers``. 0 on anything odd."""
    raw = str(os.environ.get("CODERAI_OCR_WORKERS", "") or "").strip()
    if not raw:
        try:
            with open(os.path.join(config_dir(), "config.json")) as fh:
                raw = str((json.load(fh).get("ocr") or {}).get("workers") or 0)
        except Exception:
            raw = "0"
    try:
        n = int(raw)
    except ValueError:
        log(f"ignoring workers={raw!r}: not a number")
        return 0
    return max(0, min(MAX_WORKERS, n))


def worker_env(i: int, n: int) -> dict:
    """The env one worker runs with: the launcher's (same libraries, same
    config, cache and model dirs as the main server), plus who it is."""
    env = dict(os.environ)
    cfg = os.environ.get("CODERAI_CONFIG_DIR", "/config")
    env.setdefault("XDG_CONFIG_HOME", cfg)
    env.setdefault("XDG_DATA_HOME", os.environ.get("CODERAI_MODELS_DIR", "/models"))
    cache = os.environ.get("CODERAI_CACHE_DIR", "/cache")
    env.setdefault("XDG_CACHE_HOME", cache)
    # The HuggingFace cache comes from config (models.hf_cache_dir), never from
    # the cache dir: deriving it here pointed every worker at a second cache and
    # had them re-download models the install already held. Read inline rather
    # than through codai.hf_cache -- importing the package into this supervisor
    # would cost exactly what it exists to save (see the module docstring).
    hf = configured_hf_cache_dir()
    if hf:
        env["HF_HOME"] = hf
        env["HF_HUB_CACHE"] = os.path.join(hf, "hub")
        env["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf, "hub")
        env["TRANSFORMERS_CACHE"] = os.path.join(hf, "transformers")
    env["PYTHONUNBUFFERED"] = "1"
    env["CODERAI_OCR_WORKER"] = f"{i}/{n}"
    env["CODERAI_ENGINE_NAME"] = f"ocrw{i}"
    # No GPU work happens here: pages go to pods. Pinning the backend keeps a
    # worker from probing (and holding a context on) a card the engines use.
    env["CODERAI_ENGINE_BACKEND"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    # Register no models: a worker serves /v1/ocr, whose engines come from the
    # ocr config, and must not pre-load anything from models.json.
    env["CODERAI_ENGINE_MODELS"] = "[]"
    # It is reached by nginx directly, not by the front, so it authenticates the
    # caller's own bearer -- and must not demand the front's internal token.
    for k in ("CODERAI_INTERNAL_TOKEN", "CODERAI_ENGINE_PRIMARY", "CODERAI_FRONT_PID"):
        env.pop(k, None)
    return env


def worker_cmd(i: int) -> list:
    return [PYTHON, APP, "--config", config_dir(),
            "--engine-only", "--internal-port", str(PORT_BASE + i)]


def _die_with_parent() -> None:
    """Run in the child: SIGKILL it when the launcher dies, however it dies.

    Workers get their own session (so a signal to the launcher's group cannot
    half-kill one mid-page), which also puts them out of reach of supervisord's
    killasgroup; without this a SIGKILLed launcher would leave N workers holding
    their ports, and the next launcher's workers could not bind.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def reap_stale() -> None:
    """Kill workers a previous launcher left behind (named coderai-ocrw<i>)."""
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                name = fh.read().split(b"\0", 1)[0]
        except OSError:
            continue
        if name.startswith(b"coderai-ocrw"):
            log(f"killing stale worker {pid} ({name.decode(errors='replace')})")
            try:
                os.kill(int(pid), signal.SIGKILL)
            except OSError:
                pass


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ------------------------------------------------------------------- nginx #

def render(ports) -> tuple:
    """(upstream, locations) file bodies for the healthy worker ports."""
    if not ports:
        return "", ""
    servers = "\n".join(f"    server 127.0.0.1:{p} max_fails=1 fail_timeout=5s;"
                        for p in sorted(ports))
    upstream = (
        "# Written by coderai-ocr-workers; do not edit.\n"
        "upstream coderai_ocr {\n"
        "    least_conn;\n"
        # Shared memory, so all nginx worker processes balance on ONE set of
        # counts; without it each balances only what it alone has sent.
        "    zone coderai_ocr 256k;\n"
        f"{servers}\n"
        # Every worker down: the front still serves OCR the old way. nginx
        # reaches a backup only once the primaries are MARKED down, not within
        # the request that found them dead, so max_fails=1 keeps that to one
        # request per worker; this launcher then routes to the front itself
        # within POLL_S.
        f"    server {FRONT} backup;\n"
        "    keepalive 512;\n"
        "}\n")
    # Retry another worker ONLY when the request never reached one (`error`
    # covers connect failures). No try limit: a cap of N tries is spent on N dead
    # workers and never reaches the front, which is exactly the case the backup
    # is there for (measured: a SIGKILLed launcher -> 502). Never on `timeout`: a page a worker is still
    # reading would be read twice, on two pods, and billed twice.
    loc_body = (
        "    proxy_pass http://coderai_ocr;\n"
        "    proxy_set_header Connection \"\";\n"
        "    proxy_next_upstream error;\n")
    locations = (
        "# Written by coderai-ocr-workers; do not edit.\n"
        "location = /v1/ocr {\n" + loc_body + "}\n"
        "location = /v1/ocr/batch {\n" + loc_body + "}\n")
    return upstream, locations


def _read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def _write(path: str, body: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(body)
    os.replace(tmp, path)


def apply_nginx(ports) -> bool:
    """Point nginx at ``ports``. True when the running config now matches.

    A config nginx rejects is rolled back before reloading, so a bad write can
    never take the public port down.
    """
    os.makedirs(INCLUDE_DIR, exist_ok=True)
    up_p = os.path.join(INCLUDE_DIR, UPSTREAM_FILE)
    loc_p = os.path.join(INCLUDE_DIR, LOCATION_FILE)
    up, loc = render(ports)
    old_up, old_loc = _read(up_p), _read(loc_p)
    if (old_up, old_loc) == (up, loc) and os.path.exists(up_p) and os.path.exists(loc_p):
        return True
    _write(up_p, up)
    _write(loc_p, loc)
    test = subprocess.run([NGINX, "-t", "-q", "-c", NGINX_CONF],
                          capture_output=True, text=True)
    if test.returncode != 0:
        log(f"nginx rejected the routing for {sorted(ports)}; keeping the old one: "
            f"{(test.stderr or test.stdout).strip()[-400:]}")
        _write(up_p, old_up)
        _write(loc_p, old_loc)
        return False
    rel = subprocess.run([NGINX, "-s", "reload", "-c", NGINX_CONF],
                         capture_output=True, text=True)
    if rel.returncode != 0:
        # nginx not up yet: it reads these files when it starts.
        log(f"nginx reload failed ({(rel.stderr or '').strip()[-200:]}); "
            f"it will pick the routing up when it starts")
        return False
    log(f"/v1/ocr -> {('workers ' + ','.join(map(str, sorted(ports)))) if ports else 'the front'}")
    return True


# ---------------------------------------------------------------- workers #

class Worker:
    def __init__(self, i: int, n: int):
        self.i, self.n = i, n
        self.port = PORT_BASE + i
        self.proc = None
        self.started = 0.0
        self.ready = False
        self.backoff = 2.0
        self.next_start = 0.0

    def start(self) -> None:
        log(f"starting worker {self.i}/{self.n} on 127.0.0.1:{self.port}")
        self.proc = subprocess.Popen(worker_cmd(self.i), env=worker_env(self.i, self.n),
                                     start_new_session=True,
                                     preexec_fn=_die_with_parent)
        self.started = time.monotonic()
        self.ready = False

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def signal(self, sig) -> None:
        if self.alive():
            try:
                os.killpg(self.proc.pid, sig)
            except OSError:
                pass


class Launcher:
    def __init__(self):
        self.n = 0
        self.workers = []
        self.routed = None          # ports nginx currently sends /v1/ocr to
        self.stopping = False

    def ready_ports(self):
        return [w.port for w in self.workers if w.ready and w.alive()]

    def route(self) -> None:
        ports = self.ready_ports()
        if ports != self.routed and apply_nginx(ports):
            self.routed = ports

    def retire_all(self) -> None:
        """Take every worker out of nginx, let in-flight pages finish, stop them."""
        if not self.workers:
            return
        if apply_nginx([]):
            self.routed = []
        for w in self.workers:
            w.signal(signal.SIGTERM)
        deadline = time.monotonic() + DRAIN_S
        while time.monotonic() < deadline and any(w.alive() for w in self.workers):
            time.sleep(1.0)
        for w in self.workers:
            w.signal(signal.SIGKILL)
        self.workers = []

    def resize(self, n: int) -> None:
        # Every worker's slot names N (i/N), and a pod's assignment depends on
        # it, so a new N is a new set of workers, not one more or one fewer.
        log(f"workers: {self.n} -> {n}")
        self.retire_all()
        self.n = n
        self.workers = [Worker(i, n) for i in range(n)]
        for w in self.workers:
            w.start()

    def tick(self) -> None:
        now = time.monotonic()
        for w in self.workers:
            if w.alive():
                if not w.ready:
                    if healthy(w.port):
                        w.ready = True
                        w.backoff = 2.0
                        log(f"worker {w.i} ready on :{w.port}")
                    elif now - w.started > START_GRACE_S:
                        log(f"worker {w.i} not healthy after {START_GRACE_S:.0f}s; restarting")
                        w.signal(signal.SIGKILL)
                continue
            if w.proc is not None:
                log(f"worker {w.i} exited ({w.proc.returncode}); restarting in {w.backoff:.0f}s")
                w.proc = None
                w.ready = False
                w.next_start = now + w.backoff
                w.backoff = min(RESTART_BACKOFF_MAX_S, w.backoff * 2)
            elif now >= w.next_start:
                w.start()
        self.route()

    def run(self) -> None:
        def _stop(*_):
            self.stopping = True
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        reap_stale()
        # Start from "the front serves OCR": stale includes from a previous run
        # may name workers that do not exist yet.
        if apply_nginx([]):
            self.routed = []
        next_cfg = 0.0
        while not self.stopping:
            now = time.monotonic()
            if now >= next_cfg:
                next_cfg = now + CONFIG_EVERY_S
                n = wanted_workers()
                if n != self.n:
                    self.resize(n)
            self.tick()
            time.sleep(POLL_S)
        log("stopping")
        self.retire_all()


def main() -> int:
    Launcher().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
