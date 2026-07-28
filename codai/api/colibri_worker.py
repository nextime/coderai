# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed colibri (GLM-5.2) worker — the C engine, driven directly.

colibri (https://github.com/JustVugg/colibri) is a pure-C MoE inference engine for
GLM-5.2 that streams experts from disk. Its Python side (``coli`` / ``openai_server.py``)
is only a thin OpenAI gateway around the C engine; there is no long-lived server we
would proxy to. So — unlike :mod:`codai.api.ds4_worker`, which proxies HTTP to a
managed ``ds4-server`` — coderai here drives the **C engine binary directly** over its
stdin/stdout "mux" wire protocol (``docs/serve_protocol.md``): we own the build, the
subprocess, and the protocol client. The GLM-5.2 chat template lives in
:mod:`codai.backends.colibri` (the server owns the template); this module owns the
process + wire protocol.

Lifecycle, mirroring the other managed workers:

* :func:`ensure_built` clones the repo and runs ``make`` (CUDA when available) so the
  ``colibri`` engine binary exists (idempotent).
* :func:`ensure_engine` launches the engine on the configured GLM-5.2 container in
  serve-mux mode, completes the ``READY`` handshake, and returns a live
  :class:`MuxEngine` the backend generates through.

The matching ``ColibriBackend.cleanup()`` calls :func:`stop_service`, so the model
manager's normal eviction tears the engine process down.
"""

import codecs
import collections
import os
import platform
import queue
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

# The engine → server "mux" startup sentinel (docs/serve_protocol.md): the engine
# writes this once, then a STAT line, before it will accept SUBMIT frames.
READY = b"\x01\x01READY\x01\x01\n"

_lock = threading.RLock()
# Live engines keyed by service key (the GLM-5.2 container dir) so a config change to
# a different container restarts cleanly and two models never share one process.
_services: dict[str, "MuxEngine"] = {}
_built = False


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def default_install_dir() -> Path:
    return Path(os.environ.get("CODERAI_COLIBRI_DIR")
                or os.path.expanduser("~/.coderai/colibri"))


def _install_dir(cfg) -> Path:
    return Path(cfg.install_dir).expanduser() if getattr(cfg, "install_dir", None) \
        else default_install_dir()


def _engine_bin(install_dir: Path) -> Path:
    """The C engine binary. colibri builds it as ``colibri`` inside the ``c/`` dir
    (``glm`` is the pre-#391 name, kept as a fallback for old trees)."""
    cdir = install_dir / "c"
    for name in ("colibri", "colibri.exe", "glm", "glm.exe"):
        cand = cdir / name
        if cand.exists():
            return cand
    return cdir / "colibri"


def _detect_build_target() -> str:
    """Pick a build flavour from the host: CUDA when the toolkit is present."""
    if platform.system() == "Darwin":
        return "metal"
    if shutil.which("nvcc") or os.path.isdir("/usr/local/cuda"):
        return "cuda"
    if shutil.which("hipcc") or os.path.isdir("/opt/rocm"):
        return "hip"
    return "cpu"


def _make_args(cfg) -> list:
    """``make`` arguments for the resolved build target."""
    target = (getattr(cfg, "build_target", "auto") or "auto").strip().lower()
    if target in ("", "auto"):
        target = _detect_build_target()
    if target == "cuda":
        # native SASS is fine — coderai builds on the same GPU host it runs on. For a
        # portable image build pass CUDA_ARCH=portable via extra make env if needed.
        return ["colibri", "CUDA=1", f"CUDA_ARCH={os.environ.get('COLI_CUDA_ARCH', 'native')}"]
    if target == "hip":
        return ["colibri", "HIP=1", f"HIP_ARCH={os.environ.get('COLI_HIP_ARCH', 'native')}"]
    if target == "metal":
        return ["colibri", "METAL=1"]
    return ["colibri"]


def _run_logged(cmd, cwd, label, tail, **kw):
    """Run a subprocess, streaming its output with a ``[colibri]`` prefix into ``tail``."""
    print(f"[colibri] $ {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, **kw)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            print(f"[colibri] {line}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        joined = " | ".join(list(tail)[-5:])
        raise RuntimeError(f"{label} failed (exit {proc.returncode}). {joined}")


def ensure_built(cfg) -> Path:
    """Clone + build colibri if the engine binary is missing. Returns its path."""
    global _built
    install_dir = _install_dir(cfg)
    binary = _engine_bin(install_dir)
    if binary.exists():
        _built = True
        return binary
    if not getattr(cfg, "auto_build", True):
        raise RuntimeError(
            f"colibri engine not found at {binary} and auto_build is disabled. Build it "
            f"manually (git clone {cfg.repo_url}; cd c; make colibri [CUDA=1]) or enable "
            "auto_build.")

    tail = collections.deque(maxlen=40)
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (install_dir / ".git").exists() and not (install_dir / "c" / "Makefile").exists():
        print(f"[colibri] cloning {cfg.repo_url} → {install_dir} …", flush=True)
        _run_logged(["git", "clone", "--depth", "1", cfg.repo_url, str(install_dir)],
                    cwd=install_dir.parent, label="git clone", tail=tail)

    cdir = install_dir / "c"
    make_args = _make_args(cfg)
    print(f"[colibri] building engine (make {' '.join(make_args)}) — this can take a while …",
          flush=True)
    _run_logged(["make", "-s"] + make_args, cwd=cdir, label="make", tail=tail)

    binary = _engine_bin(install_dir)
    if not binary.exists():
        raise RuntimeError(
            f"colibri build completed but {binary} is missing. Last output: "
            + " | ".join(list(tail)[-5:]))
    _built = True
    print(f"[colibri] built {binary}", flush=True)
    return binary


# --------------------------------------------------------------------------- #
# mux protocol client (ported from colibri's openai_server.py `Engine`)
# --------------------------------------------------------------------------- #
def _read_engine_turn(stream, sentinel: bytes) -> dict:
    """Consume bytes up to ``sentinel`` (the READY handshake), then the STAT line."""
    pending = b""
    while True:
        byte = stream.read(1)
        if byte == b"":
            raise RuntimeError("colibri engine exited before READY")
        pending += byte
        if pending.endswith(sentinel):
            break
    fields = stream.readline().decode("utf-8", "replace").strip().split()
    if len(fields) < 5 or fields[0] != "STAT":
        raise RuntimeError(f"invalid engine status after READY: {' '.join(fields)}")
    return _parse_stat(fields)


def _parse_stat(fields) -> dict:
    return {
        "completion_tokens": int(fields[1]),
        "tokens_per_second": float(fields[2]),
        "cache_hit_percent": float(fields[3]),
        "rss_gb": float(fields[4]),
        "prompt_tokens": int(fields[5]) if len(fields) > 5 else 0,
        "length_limited": bool(int(fields[6])) if len(fields) > 6 else False,
    }


class MuxEngine:
    """A running colibri engine in serve-mux mode, spoken to over stdin/stdout.

    One process serves one GLM-5.2 container with up to ``KV_SLOTS`` cached
    conversations (continuous batching). :meth:`run` renders nothing — it takes an
    already-rendered prompt (the backend owns the GLM-5.2 chat template), submits it,
    streams decoded text to ``on_text`` and returns the turn stats. A small slot pool
    keeps concurrent requests off each other's KV slot (avoids ``SLOT_BUSY``).
    """

    def __init__(self, binary: Path, model_dir: str, *, cap: int = 8,
                 max_tokens: int = 1024, kv_slots: int = 1, env: Optional[dict] = None):
        kv_slots = max(1, min(16, int(kv_slots or 1)))
        child_env = dict(env or os.environ, SNAP=str(model_dir), SERVE="1",
                         SERVE_BATCH="1", NGEN=str(max_tokens), KV_SLOTS=str(kv_slots))
        self.model_dir = str(model_dir)
        self.kv_slots = kv_slots
        self.process = subprocess.Popen(
            [str(binary), str(cap)], env=child_env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        self.write_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending: dict[str, queue.Queue] = {}
        self.next_request_id = 1
        self.closed = False
        self.dispatcher_error: Optional[Exception] = None
        self.hwinfo = None
        self.tiers = None
        self.emap = None
        self.hits = None
        # Free KV slots, handed out per request and returned on completion.
        self._slots: queue.Queue = queue.Queue()
        for i in range(kv_slots):
            self._slots.put(i)
        # Engine logs go to stderr; keep stdout pure protocol. Pump stderr with a prefix.
        self._log_tail = collections.deque(maxlen=30)
        threading.Thread(target=self._pump_stderr, daemon=True,
                         name="colibri-stderr").start()
        # READY handshake must complete before the dispatcher starts consuming lines.
        _read_engine_turn(self.process.stdout, READY)
        self.dispatcher = threading.Thread(target=self._dispatch_stdout,
                                           name="colibri-stdout", daemon=True)
        self.dispatcher.start()

    # ---- logs / health ---------------------------------------------------- #
    def _pump_stderr(self):
        try:
            for line in iter(self.process.stderr.readline, b""):
                text = line.decode("utf-8", "replace").rstrip()
                if text:
                    self._log_tail.append(text)
                    print(f"[colibri] {text}", flush=True)
        except Exception:
            pass

    def log_tail(self) -> str:
        return " | ".join(list(self._log_tail)[-5:]).strip()

    def is_alive(self) -> bool:
        return (not self.closed and self.dispatcher_error is None
                and self.process.poll() is None)

    def _read_exact(self, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = self.process.stdout.read(remaining)
            if chunk == b"":
                raise RuntimeError("truncated engine DATA payload")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _fail_pending(self, error: Exception):
        with self.pending_lock:
            requests = list(self.pending.values())
            self.pending.clear()
        for events in requests:
            events.put(("error", error))

    def _dispatch_stdout(self):
        try:
            while True:
                line = self.process.stdout.readline()
                if line == b"":
                    raise RuntimeError("colibri engine exited unexpectedly")
                fields = line.decode("utf-8", "replace").strip().split()
                if not fields:
                    continue
                kind = fields[0]
                if kind == "DATA" and len(fields) == 3:
                    request_id = fields[1]
                    size = int(fields[2])
                    if not 0 <= size <= 65536:
                        raise RuntimeError("invalid engine DATA size")
                    data = self._read_exact(size)
                    if self._read_exact(1) != b"\n":
                        raise RuntimeError("invalid engine DATA terminator")
                    with self.pending_lock:
                        events = self.pending.get(request_id)
                    if events is not None:
                        events.put(("data", data))
                elif kind == "DONE" and len(fields) >= 7:
                    request_id = fields[1]
                    stats = _parse_stat(fields[2:])
                    with self.pending_lock:
                        events = self.pending.pop(request_id, None)
                    if events is not None:
                        events.put(("done", stats))
                elif kind == "ERROR" and len(fields) >= 2:
                    request_id = fields[1]
                    message = " ".join(fields[2:]) or "engine request failed"
                    with self.pending_lock:
                        events = self.pending.pop(request_id, None)
                    if events is not None:
                        events.put(("error", RuntimeError(message)))
                elif kind == "HWINFO" and len(fields) >= 7:
                    parts = " ".join(fields[6:]).split("|")
                    self.hwinfo = {"cores": int(fields[1]), "ram_total_gb": float(fields[2]),
                                   "ram_avail_gb": float(fields[3]), "gpus": int(fields[4]),
                                   "vram_total_gb": float(fields[5]),
                                   "cpu": parts[0].strip() if len(parts) > 0 else "",
                                   "gpu": parts[1].strip() if len(parts) > 1 else ""}
                elif kind == "TIERS" and len(fields) >= 6:
                    self.tiers = {"vram": int(fields[1]), "ram": int(fields[2]),
                                  "disk": int(fields[3]), "vram_gb": float(fields[4]),
                                  "ram_gb": float(fields[5])}
                elif kind == "EMAP" and len(fields) == 4:
                    self.emap = {"rows": int(fields[1]), "cols": int(fields[2]), "map": fields[3]}
                elif kind == "HITS" and len(fields) == 4:
                    self.hits = fields[3]
                else:
                    # Forward-compatibility: ignore telemetry line kinds we don't know
                    # (PROF/ENTROPY/GPUS/TOPK/REPIN/…) rather than erroring.
                    continue
        except Exception as error:  # noqa: BLE001
            if not self.closed:
                self.dispatcher_error = error
                self._fail_pending(error)

    def run(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
            on_text: Callable[[str], None], cancelled: Optional[Callable[[], bool]] = None
            ) -> dict:
        """Submit one rendered prompt; stream decoded text to ``on_text``; return stats."""
        if self.dispatcher_error is not None:
            raise RuntimeError("colibri engine dispatcher stopped: "
                               + (self.log_tail() or str(self.dispatcher_error)))
        if self.process.poll() is not None:
            raise RuntimeError("colibri engine is not running. " + self.log_tail())
        payload = prompt.encode("utf-8")
        if b"\0" in payload:
            raise ValueError("NUL bytes are not supported in prompts.")
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        # A KV slot for this turn (blocks briefly if all slots are busy).
        slot = self._slots.get()
        events: queue.Queue = queue.Queue()
        try:
            with self.pending_lock:
                if self.closed:
                    raise RuntimeError("colibri engine is shutting down")
                request_id = str(self.next_request_id)
                self.next_request_id += 1
                self.pending[request_id] = events
            header = (f"SUBMIT {request_id} {slot} {len(payload)} {max_tokens} "
                      f"{temperature:.8g} {top_p:.8g}\n").encode()
            try:
                with self.write_lock:
                    if self.process.poll() is not None:
                        raise RuntimeError("colibri engine is not running")
                    self.process.stdin.write(header + payload + b"\n")
                    self.process.stdin.flush()
            except Exception:
                with self.pending_lock:
                    self.pending.pop(request_id, None)
                raise

            cancel_sent = False
            while True:
                kind, value = events.get()
                if kind == "data":
                    text = decoder.decode(value)
                    if text and not cancel_sent:
                        on_text(text)
                    if cancelled and cancelled() and not cancel_sent:
                        cancel_sent = True
                        with self.write_lock:
                            self.process.stdin.write(f"CANCEL {request_id}\n".encode())
                            self.process.stdin.flush()
                elif kind == "done":
                    tail = decoder.decode(b"", final=True)
                    if tail and not cancel_sent:
                        on_text(tail)
                    return value
                else:  # error
                    raise value if isinstance(value, Exception) else RuntimeError(str(value))
        finally:
            self._slots.put(slot)

    def close(self):
        with self.pending_lock:
            if self.closed:
                return
            self.closed = True
        self._fail_pending(RuntimeError("colibri engine is shutting down"))
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# registry / lifecycle
# --------------------------------------------------------------------------- #
def _coderai_offload_dir() -> str:
    """coderai's configured disk-offload directory (config.offload.directory), or ''."""
    try:
        from codai.admin.routes import config_manager
        if config_manager is not None and config_manager.config is not None:
            d = (getattr(config_manager.config.offload, "directory", "") or "").strip()
            return os.path.expanduser(d) if d else ""
    except Exception:
        pass
    return ""


def resolve_service_key(cfg, model_dir: Optional[str] = None):
    """Decide which GLM-5.2 container the engine serves and the key to cache it under.

    Preference: the requested model's own container dir → an explicit
    ``cfg.model_path`` override → '' (nothing to serve). Returns
    ``(resolved_dir_or_'', svc_key)``; the key is the dir when we have one (so two
    containers get their own engine), else ``model_id``.
    """
    resolved = ""
    for cand in (model_dir, getattr(cfg, "model_path", "") or ""):
        cand = os.path.expanduser((cand or "").strip())
        if cand and os.path.isdir(cand):
            resolved = os.path.abspath(cand)
            break
    svc_key = resolved or (getattr(cfg, "model_id", "glm-5.2-colibri") or "glm-5.2-colibri")
    return resolved, svc_key


def _build_env(cfg) -> tuple:
    """Engine environment: CUDA_EXPERT_GB + free-form extra_env KEY=VALUE pairs."""
    env = os.environ.copy()
    applied = {}
    ceg = (getattr(cfg, "cuda_expert_gb", "") or "").strip()
    if ceg and "CUDA_EXPERT_GB" not in env:
        env["CUDA_EXPERT_GB"] = ceg
        applied["CUDA_EXPERT_GB"] = ceg
    extra_env = (getattr(cfg, "extra_env", "") or "").strip()
    if extra_env:
        for tok in shlex.split(extra_env):
            if "=" in tok:
                k, v = tok.split("=", 1)
                k = k.strip()
                if k:
                    env[k] = v
                    applied[k] = v
    return env, applied


def ensure_engine(cfg, model_dir: Optional[str] = None, ctx: Optional[int] = None,
                  ready_timeout: float = 3600.0) -> MuxEngine:
    """Build (as needed), then start (or reuse) the colibri engine for a container.

    ``model_dir`` is the requested model's container path; when it resolves to a
    directory (or ``cfg.model_path`` is set) the engine loads THAT. ``ctx`` sizes the
    per-turn generation budget (NGEN). Returns a live :class:`MuxEngine`.
    """
    resolved, svc_key = resolve_service_key(cfg, model_dir)
    with _lock:
        eng = _services.get(svc_key)
        if eng and eng.is_alive():
            return eng
        if eng and not eng.is_alive():
            eng.close()
            _services.pop(svc_key, None)

        binary = ensure_built(cfg)
        if not resolved:
            raise RuntimeError(
                "colibri: no GLM-5.2 container resolved for this request. Point the "
                "model at the int4 container directory (or set colibri.model_path). "
                "There is no auto-download of the ~372 GB container.")

        try:
            ngen = int(ctx) if ctx else 0
        except (TypeError, ValueError):
            ngen = 0
        if ngen <= 0:
            ngen = int(getattr(cfg, "ctx", 100000) or 100000)

        env, applied = _build_env(cfg)
        env_note = ("  (" + " ".join(f"{k}={v}" for k, v in applied.items()) + ")"
                    if applied else "")
        kv_slots = int(getattr(cfg, "kv_slots", 1) or 1)
        cap = int(getattr(cfg, "cap", 8) or 8)
        print(f"[colibri] launching engine {binary} on {resolved} "
              f"(cap={cap}, kv_slots={kv_slots}, ngen={ngen}){env_note}", flush=True)
        eng = MuxEngine(binary, resolved, cap=cap, max_tokens=ngen,
                        kv_slots=kv_slots, env=env)
        _services[svc_key] = eng

    # READY already completed inside MuxEngine.__init__; a quick liveness gate here
    # surfaces an engine that died immediately (bad container, OOM) as a clean error.
    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if not eng.is_alive():
            tail = eng.log_tail()
            stop_service(svc_key)
            raise RuntimeError("colibri engine exited before serving"
                               + (f". Last output: {tail}" if tail else ""))
        # Alive and past READY — ready to serve.
        print(f"[colibri] engine ready for {svc_key}", flush=True)
        return eng
    stop_service(svc_key)
    raise RuntimeError(f"colibri engine for {svc_key} did not become ready in time")


def stop_service(model_id: str) -> None:
    with _lock:
        eng = _services.pop(model_id, None)
    if not eng:
        return
    try:
        eng.close()
    except Exception:
        pass
    print(f"[colibri] engine for {model_id} stopped", flush=True)


def stop_all() -> None:
    for mid in list(_services.keys()):
        stop_service(mid)


import atexit as _atexit
_atexit.register(stop_all)
