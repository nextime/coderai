# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fully-managed CrisperWhisper worker (isolated venv).

CrisperWhisper needs its verbatim word-timestamp generation, which breaks on the
server's transformers 5.x. So it runs in its own venv (pinned transformers 4.x)
behind a small HTTP service (tools/crisperwhisper_service.py); coderai talks to it
as a remote STT backend. Mirrors codai.api.canary_worker / pyannote_worker.
"""

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE_SCRIPT = _REPO_ROOT / "tools" / "crisperwhisper_service.py"
_REQUIREMENTS = _REPO_ROOT / "requirements-crisperwhisper.txt"
_BAKED_VENV = Path("/opt/coderai/crisperwhisper_venv")

_lock = threading.RLock()
_services: dict = {}   # model_name -> {"proc","port","url"}
_bootstrapped = False

#: What one resident worker costs when nothing has been measured yet. Whisper-large
#: at fp16 weighs ~3.1 GB, but a verbatim word-timestamp pass holds the encoder's
#: attention for a full 30s window on top of that, so the figure that matters for
#: eviction is the inference peak, not the checkpoint. Measured per model on the
#: first transcription (see `record_measurement`); this is only the opening bid.
DEFAULT_VRAM_GB = 8.0

#: Per-model in-flight transcription counts, and an event that is set while every
#: worker is idle. A worker must never be torn down mid-transcription: the request
#: would be lost, and the manager's own busy signal cannot see this work because
#: acquire_stt_backend pops the model pool, leaving `_is_key_busy` always False.
_inflight: dict = {}
_idle = threading.Event()
_idle.set()
_last_used: dict = {}      # model_name -> time.monotonic() of last use
_releaser_registered = False


def resolve_venv_dir() -> Path:
    explicit = os.environ.get("CODERAI_CRISPERWHISPER_VENV")
    if explicit:
        return Path(explicit)
    if (_BAKED_VENV / ("Scripts" if os.name == "nt" else "bin")).exists():
        return _BAKED_VENV
    return Path(os.path.expanduser("~/.coderai/crisperwhisper_venv"))


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")


def _pip_ok(py: Path) -> bool:
    try:
        return subprocess.run(
            [str(py), "-c", "import transformers, torch, soundfile"],
            capture_output=True).returncode == 0
    except Exception:
        return False


def _bootstrap_venv() -> Path:
    global _bootstrapped
    venv = resolve_venv_dir()
    py = _venv_python(venv)
    if _bootstrapped and py.exists():
        return py
    if py.exists() and _pip_ok(py):
        _bootstrapped = True
        return py
    if not py.exists():
        print(f"[crisperwhisper] creating isolated venv at {venv} …", flush=True)
        venv.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    py = _venv_python(venv)
    if not _pip_ok(py):
        if not _REQUIREMENTS.exists():
            raise RuntimeError(f"CrisperWhisper requirements missing: {_REQUIREMENTS}")
        print("[crisperwhisper] installing pinned transformers+torch into the isolated "
              "venv (first run downloads several GB) …", flush=True)
        subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(_REQUIREMENTS)],
                       check=True)
        if not _pip_ok(py):
            raise RuntimeError("CrisperWhisper venv install did not yield importable deps")
    _bootstrapped = True
    return py


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _pump_logs(proc: subprocess.Popen, tail):
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            print(f"[crisperwhisper] {line}", flush=True)


def _health_ok(url: str) -> bool:
    import requests
    try:
        r = requests.get(url + "/health", timeout=3)
        return r.ok and bool(r.json().get("ok"))
    except Exception:
        return False


def ensure_service(model_name: str, config: dict = None,
                   ready_timeout: float = 1800.0) -> str:
    config = config or {}
    # An explicit `service_url` points at a CrisperWhisper worker running somewhere else — another
    # host, a container, a rented pod — so nothing is built, spawned or downloaded
    # here and coderai just proxies to it. This function already returns a URL, so
    # its callers cannot tell the difference.
    _remote = ((config.get("service_url") or "") or os.environ.get("CODERAI_CRISPERWHISPER_SERVICE_URL") or "").strip()
    if _remote:
        _remote = str(_remote).rstrip("/")
        if not _health_ok(_remote):
            raise RuntimeError(
                f"configured service_url {_remote} is not answering its health check")
        print(f"[crisperwhisper] using the configured remote service at {_remote}", flush=True)
        return _remote

    # Both directions of eviction, from the first use: this worker can take the
    # card from a tracked model (acquire_stt_backend, upstream) and the manager can
    # take it back from this worker (the releaser, below).
    register_releaser()

    with _lock:
        svc = _services.get(model_name)
        if svc and svc["proc"].poll() is None and _health_ok(svc["url"]):
            _last_used[model_name] = time.monotonic()
            return svc["url"]
        if svc and svc["proc"].poll() is not None:
            _services.pop(model_name, None)

    # A second resident worker is another whole Whisper-large on the same card, and
    # the sibling's own reservation does not cover it: the model page can configure
    # several CrisperWhisper models, and before this they stacked until one OOMed
    # mid-transcription. Give the card to the model being asked for.
    evict_idle_siblings(keep=model_name)

    with _lock:
        py = _bootstrap_venv()
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        try:
            # All four HF variables, not just the hub pair: this worker runs in
            # its own venv and never imports codai, so HF_HOME and
            # TRANSFORMERS_CACHE would stay as the image set them and the model
            # would be fetched a second time into that directory.
            from codai.hf_cache import child_env as _hf_child_env
            env = _hf_child_env(env)
        except Exception:
            pass
        gpu = config.get("gpu_device")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        proc = subprocess.Popen(
            [str(py), str(_SERVICE_SCRIPT), "--model", model_name,
             "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, env=env, cwd=str(_REPO_ROOT))
        import collections
        tail = collections.deque(maxlen=15)
        threading.Thread(target=_pump_logs, args=(proc, tail), daemon=True).start()
        _services[model_name] = {"proc": proc, "port": port, "url": url}
        _last_used[model_name] = time.monotonic()

    def _tail_msg():
        joined = " | ".join(l.strip()[:300] for l in list(tail) if l.strip()).strip()
        return f". Last output: {joined}" if joined else ""

    # Hold the room open until the worker is actually holding it. The card looks
    # free for the whole of a cold start — shards downloading, weights loading —
    # because free VRAM is read from the driver and this worker has not allocated
    # anything yet. Live traffic reads that as room, loads into it, and the
    # transcription OOMs on a card nobody over-committed on purpose. The
    # reservation makes the load window visible to every other load.
    _res_key = model_key(model_name)
    _note_loading(_res_key, reserved_vram_gb(model_name, config))
    try:
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"CrisperWhisper worker exited (code {proc.returncode}) before ready"
                    + _tail_msg())
            if _health_ok(url):
                print(f"[crisperwhisper] service ready for {model_name} at {url}", flush=True)
                return url
            time.sleep(2)
    finally:
        _clear_loading(_res_key)
    stop_service(model_name)
    raise RuntimeError(f"CrisperWhisper worker for {model_name} did not become ready"
                       + _tail_msg())


def stop_service(model_name: str) -> None:
    with _lock:
        svc = _services.pop(model_name, None)
        _last_used.pop(model_name, None)
        # A worker that is gone is not busy. Drop its count before the terminate so
        # a releaser waiting on `_idle` is not left waiting for a dead process.
        _inflight.pop(model_name, None)
        if not any(_inflight.values()):
            _idle.set()
    if not svc:
        return
    proc = svc["proc"]
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
    print(f"[crisperwhisper] service for {model_name} stopped", flush=True)


def stop_all() -> None:
    for name in list(_services.keys()):
        stop_service(name)


# ── VRAM: taking the card, and giving it back ────────────────────────────────

def model_key(model_name: str) -> str:
    """The key the model manager tracks this worker's residency under."""
    return f"audio:{model_name}"


def reserved_vram_gb(model_name: str, config: dict = None) -> float:
    """What to reserve before a transcription on ``model_name``.

    Order of trust: an operator's ``used_vram_gb`` from the model page, then the
    peak this very model measured on an earlier run, then DEFAULT_VRAM_GB. The
    measurement matters because the weights-only guess that used to be hard-coded
    here (3 GB) is about a third of what a verbatim pass actually peaks at, so
    eviction freed a third of the room needed and the pass OOMed with the card
    full of models nobody had asked it to keep."""
    cfg = config or {}
    explicit = float(cfg.get("used_vram_gb") or 0.0)
    if explicit > 0:
        return explicit
    try:
        from codai.models.manager import multi_model_manager
        measured = float(multi_model_manager._measured_vram_gb.get(
            model_key(model_name)) or 0.0)
    except Exception:
        measured = 0.0
    return measured if measured > 0 else DEFAULT_VRAM_GB


def _note_loading(key: str, gb: float) -> None:
    """Reserve ``gb`` against this key until _clear_loading. Never raises."""
    if not gb or gb <= 0:
        return
    try:
        from codai.models.manager import multi_model_manager
        multi_model_manager.note_loading(key, float(gb))
    except Exception:
        pass


def _clear_loading(key: str) -> None:
    try:
        from codai.models.manager import multi_model_manager
        multi_model_manager.clear_loading(key)
    except Exception:
        pass


def record_measurement(model_name: str, vram_gb: float) -> None:
    """Write a worker's reported peak back as this model's footprint.

    High-water mark, never an average: a one-word clip touches a fraction of what
    a full 30s window does, and reserving the smaller figure would under-evict for
    the longer audio that follows."""
    if not vram_gb or vram_gb <= 0:
        return
    key = model_key(model_name)
    try:
        from codai.models.manager import multi_model_manager
        prev = float(multi_model_manager._measured_vram_gb.get(key) or 0.0)
        if vram_gb > prev + 0.05:
            multi_model_manager._measured_vram_gb[key] = float(vram_gb)
            print(f"[crisperwhisper] measured {vram_gb:.2f} GB for {model_name}"
                  + (f" (was {prev:.2f})" if prev else ""), flush=True)
    except Exception as e:
        print(f"[crisperwhisper] could not record the measurement: {e}", flush=True)


class inflight:
    """Mark a model busy for the length of one transcription.

    Context manager; nests safely. While any worker is in flight the releaser
    waits for it rather than killing it, so eviction costs a pause, not a lost
    request."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def __enter__(self):
        with _lock:
            _inflight[self.model_name] = _inflight.get(self.model_name, 0) + 1
            _idle.clear()
            _last_used[self.model_name] = time.monotonic()
        return self

    def __exit__(self, *exc):
        with _lock:
            n = max(0, _inflight.get(self.model_name, 1) - 1)
            _inflight[self.model_name] = n
            _last_used[self.model_name] = time.monotonic()
            if not any(_inflight.values()):
                _idle.set()
        return False


def _drain_timeout_s() -> float:
    try:
        from codai.config import config as _cfg
        sec = getattr(_cfg, "models", None) or _cfg
        return max(0.0, float(getattr(sec, "evict_drain_timeout_s", 0) or 0)) or 300.0
    except Exception:
        return 300.0


def _footprint_gb(model_name: str) -> float:
    try:
        from codai.models.manager import multi_model_manager
        return float(multi_model_manager._measured_vram_gb.get(
            model_key(model_name)) or 0.0) or DEFAULT_VRAM_GB
    except Exception:
        return DEFAULT_VRAM_GB


def evict_idle_siblings(keep: str = None, needed_gb: float = 0.0,
                        force: bool = False) -> float:
    """Stop every idle worker except ``keep``. Returns the GB released.

    Each worker is a resident Whisper-large of its own, so two of them on a 24 GB
    card shared with text, video and embeddings is already too many. An idle
    sibling is pure cost: it restarts from cache on its next request.

    ``force`` takes a busy worker too. Only the releaser uses it, and only once the
    drain budget has run out — the alternative there is not a slower transcription
    but a wedged card."""
    with _lock:
        names = [n for n in _services
                 if n != keep and (force or not _inflight.get(n))]
    freed = 0.0
    for name in names:
        freed += _footprint_gb(name)
        stop_service(name)
    if names:
        print(f"[crisperwhisper] released ~{freed:.1f} GB: stopped idle "
              f"worker(s) {', '.join(names)}", flush=True)
    return freed


def release_vram(needed_gb: float = 999.0) -> float:
    """External VRAM releaser: stop the workers so another model can have the card.

    The manager's contract is ``fn(needed_gb) -> float``: safe to call with nothing
    to free, and it must not raise. Idle workers go immediately; a worker with a
    transcription in flight is asked for by waiting out the drain budget, because a
    30s window is seconds of work and losing the request to save that wait is the
    worse trade."""
    freed = evict_idle_siblings()
    with _lock:
        busy = [n for n in _services if _inflight.get(n)]
    if not busy:
        return freed
    budget = _drain_timeout_s()
    print(f"[crisperwhisper] {len(busy)} transcription(s) in flight — waiting up "
          f"to {budget:.0f}s for the card", flush=True)
    drained = _idle.wait(budget)
    if not drained:
        print(f"[crisperwhisper] still busy after {budget:.0f}s — releasing anyway; "
              f"the in-flight transcription will fail", flush=True)
    return freed + evict_idle_siblings(force=not drained)


def register_releaser() -> None:
    """Make these workers' VRAM reclaimable by the model manager. Idempotent."""
    global _releaser_registered
    if _releaser_registered:
        return
    try:
        from codai.models.manager import multi_model_manager
        multi_model_manager.register_external_vram_releaser(release_vram)
        _releaser_registered = True
    except Exception as e:
        print(f"[crisperwhisper] could not register the VRAM releaser: {e}", flush=True)


import atexit as _atexit
_atexit.register(stop_all)
