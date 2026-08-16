# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
# GPLv3 - see the project LICENSE.

"""Background builder for the isolated OCR engine venvs (Paddle / Surya).

Building a venv (python -m venv + pip install torch/paddle/surya) takes minutes, so the
web UI triggers it asynchronously and polls status. Also usable headless. The engine
classes own the venv dir / requirements / pip command, so this just drives them.
"""

import os
import subprocess
import sys
import threading
from collections import deque
from typing import Optional

# engine -> {"state": idle|running|done|error, "log": [..], "returncode": int|None}
_STATE = {}
_LOCK = threading.Lock()
_LOG_MAX = 400


def _engine_for(cfg, engine: str):
    engine = (engine or "").strip().lower()
    if engine == "paddle":
        from codai.ocr.paddle import PaddleEngine
        return PaddleEngine(cfg)
    if engine == "surya":
        from codai.ocr.surya import SuryaEngine
        return SuryaEngine(cfg)
    raise ValueError(f"no isolated venv for engine '{engine}' (only paddle/surya)")


def status(engine: str) -> dict:
    with _LOCK:
        st = _STATE.get(engine)
        if not st:
            return {"engine": engine, "state": "idle", "log": [], "returncode": None}
        return {"engine": engine, "state": st["state"],
                "log": list(st["log"])[-40:], "returncode": st["returncode"]}


def is_running(engine: str) -> bool:
    with _LOCK:
        st = _STATE.get(engine)
        return bool(st and st["state"] == "running")


def build_async(cfg, engine: str) -> dict:
    """Kick off a background build for the engine's isolated venv. Idempotent while running."""
    engine = (engine or "").strip().lower()
    eng = _engine_for(cfg, engine)          # validates engine name early
    if is_running(engine):
        return {"engine": engine, "state": "running", "already": True}

    venv_dir = os.path.expanduser(eng._venv_dir())
    req = eng._requirements()
    py = os.path.join(venv_dir, "bin", "python")

    with _LOCK:
        _STATE[engine] = {"state": "running", "log": deque(maxlen=_LOG_MAX), "returncode": None}

    def _log(line):
        with _LOCK:
            _STATE[engine]["log"].append(line.rstrip("\n"))

    def _run():
        try:
            if not req or not os.path.isfile(req):
                raise RuntimeError(f"requirements file not found: {req}")
            _log(f"[venv] target {venv_dir}")
            if not os.path.isfile(py):
                os.makedirs(os.path.dirname(venv_dir) or ".", exist_ok=True)
                _log("[venv] creating virtualenv…")
                _stream([sys.executable, "-m", "venv", venv_dir], _log)
            _log("[pip] upgrading pip…")
            _stream([py, "-m", "pip", "install", "-U", "pip"], _log)
            _log(f"[pip] installing from {os.path.basename(req)} (this can take several minutes)…")
            _stream(eng._pip_install_cmd(py, req), _log)
            # sanity: the worker can import its engine
            _log("[check] importing engine in the new venv…")
            mod = "paddleocr" if engine == "paddle" else "surya"
            _stream([py, "-c", f"import {mod}; print('{mod} import OK')"], _log)
            with _LOCK:
                _STATE[engine]["state"] = "done"; _STATE[engine]["returncode"] = 0
            _log("[done] venv ready")
        except subprocess.CalledProcessError as e:
            _log(f"[error] command failed (rc={e.returncode})")
            with _LOCK:
                _STATE[engine]["state"] = "error"; _STATE[engine]["returncode"] = e.returncode
        except Exception as e:
            _log(f"[error] {e}")
            with _LOCK:
                _STATE[engine]["state"] = "error"; _STATE[engine]["returncode"] = -1

    threading.Thread(target=_run, name=f"ocr-venv-build-{engine}", daemon=True).start()
    return {"engine": engine, "state": "running", "venv_dir": venv_dir}


def _stream(cmd, log_fn):
    """Run cmd, streaming combined output to log_fn; raise CalledProcessError on failure."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        log_fn(line)
    rc = p.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
