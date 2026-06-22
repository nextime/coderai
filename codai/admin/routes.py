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

"""Admin dashboard routes."""
from pathlib import Path
import asyncio
import re
import shutil
from typing import Optional

from fastapi import APIRouter, Request, Response, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from codai.admin.auth import SessionManager
from codai.platform_paths import default_whisper_server_path
import queue as _q
import threading as _t
import uuid as _uuid
import json as _j


router = APIRouter()

# Templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Session manager (will be initialized in main.py)
session_manager: Optional[SessionManager] = None
SESSION_COOKIE_NAME: str = "session"   # overridden at startup to be port-specific
config_manager = None  # set via set_config_manager()
_download_sessions: dict = {}
_download_status: dict = {}   # session_id → latest progress state (survives SSE disconnect)
_download_cancelled: set = set()  # session_ids the user has requested to cancel
_download_procs: dict = {}    # session_id → multiprocessing.Process running the download


def _worker_preexec():
    """Child preexec: die with the parent (PR_SET_PDEATHSIG=SIGKILL).

    Download workers are spawned as plain subprocesses; without this they survive a
    server/engine restart as orphans, keep holding huggingface_hub's per-blob file
    lock, and make the next re-download deadlock at 0%. Tying their lifetime to the
    parent means a restart cleans them up, and the re-download resumes from the
    ``.incomplete`` blob cleanly."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass


def get_active_download_model_ids() -> set:
    """Return the set of model IDs whose download is currently in progress."""
    return {
        s["model_id"]
        for s in _download_status.values()
        if s.get("status") in ("starting", "downloading")
    }


def _active_download_session(model_id: str, file_pattern: str):
    """Return the session_id of a live download for this exact (model_id, pattern),
    or None. Used to dedup re-download clicks: a second worker for a model already
    downloading would only block on huggingface_hub's per-blob file lock and sit at
    0% forever, so we attach the new client to the running download instead."""
    for sid, s in _download_status.items():
        if (s.get("model_id") == model_id
                and (s.get("file_pattern") or "") == (file_pattern or "")
                and s.get("status") in ("starting", "downloading")
                and sid in _download_sessions):
            return sid
    return None


def _url(request: Request, path: str) -> str:
    """Return a proxy-aware absolute path (root_path prefix + path)."""
    from codai.api.urlutils import get_public_prefix
    return get_public_prefix(request) + path


def _tmpl(request: Request, name: str, ctx: dict = None):
    """Render a template with root_path injected into the context.

    Admin pages are served with no-cache headers so template/UI changes are
    picked up on a normal reload instead of being masked by the browser cache.
    """
    from codai.api.urlutils import get_public_prefix
    from codai import __version__ as _coderai_version
    c = ctx or {}
    c.setdefault("root_path", get_public_prefix(request))
    c.setdefault("coderai_version", _coderai_version)
    resp = templates.TemplateResponse(request, name, c)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def init_session_manager(config_dir: Path, port: int = 0):
    """Initialize the session manager.

    port is used to derive a port-specific cookie name so that two instances
    running on different ports on the same host don't share (and overwrite)
    each other's browser cookie.
    """
    global session_manager, SESSION_COOKIE_NAME
    session_manager = SessionManager(config_dir)
    if port:
        SESSION_COOKIE_NAME = f"session_{port}"


def set_config_manager(mgr):
    """Set the shared ConfigManager instance."""
    global config_manager
    config_manager = mgr
    from codai.models.capabilities import init_capability_cache
    init_capability_cache(str(mgr.config_dir))


def _broker_notify_models_updated(request: Request) -> None:
    """Fire-and-forget: tell AISBF broker to refresh its model cache if connected."""
    try:
        broker_service = getattr(request.app.state, "broker_service", None)
        if broker_service is None:
            return
        client = getattr(broker_service, "client", None)
        if client is None:
            return
        import asyncio
        asyncio.create_task(client.notify_models_updated())
    except Exception:
        pass


def _next_whisper_server_model_id(audio_models) -> str:
    used_suffixes = set()
    for model in audio_models or []:
        if not isinstance(model, dict):
            continue
        if model.get("backend") != "whisper-server":
            continue
        match = re.fullmatch(r"whisper(\d+)", str(model.get("id") or "").strip())
        if match:
            used_suffixes.add(int(match.group(1)))

    suffix = 0
    while suffix in used_suffixes:
        suffix += 1
    return f"whisper{suffix}"


def _default_whisper_server_path() -> str:
    return shutil.which("whisper-server") or default_whisper_server_path()


def _next_free_whisper_port(audio_models, default: int = 8744) -> int:
    """Lowest 87xx-ish port not already claimed by a whisper-server runner."""
    used = set()
    for m in audio_models or []:
        if isinstance(m, dict) and m.get("backend") == "whisper-server":
            try:
                used.add(int(m.get("port")))
            except (TypeError, ValueError):
                pass
    port = default
    while port in used:
        port += 1
    return port


def _is_whisper_runner(m) -> bool:
    """A whisper-server RUNNER (has a model_path) vs a whisper MODEL config (a
    .gguf entry with backend=whisper-server but no model_path). Runners are the
    actual subprocess instances; model configs only enable the model."""
    return (isinstance(m, dict) and m.get("backend") == "whisper-server"
            and bool(m.get("model_path")))


def _sync_whisper_runner(model_path: str, model_entry: dict) -> bool:
    """Ensure exactly ONE whisper-server runner per whisper gguf MODEL config
    (strict 1:1). The runner's id (the whisper-server model_id) is inherited from
    the config's `alias`, which is the link between the model config and its
    runner. Returns True if models.json changed."""
    audio_list = config_manager.models_data.setdefault("audio_models", [])
    rid = (model_entry.get("alias") or "").strip()
    if not rid:
        # No alias given: mint the runner id and stamp it as the config's alias so
        # the config↔runner link (config.alias == runner.id) always exists.
        rid = _next_whisper_server_model_id(audio_list)
        model_entry["alias"] = rid
    # Runner for this config (id == config alias) already exists? Nothing to do.
    if any(_is_whisper_runner(m) and m.get("id") == rid for m in audio_list):
        return False
    runner = {
        "id": rid,
        "backend": "whisper-server",
        "server_path": _default_whisper_server_path(),
        "model_path": model_path,
        "port": _next_free_whisper_port(audio_list),
        "gpu_device": int(model_entry.get("gpu_device", 0) or 0),
        "load_mode": model_entry.get("load_mode", "on-request"),
        "model_type": "audio_models",
        "model_types": ["audio_models"],
        "alias": rid,
    }
    if model_entry.get("engine"):
        runner["engine"] = model_entry["engine"]
    audio_list.append(runner)
    return True


def get_current_user(request: Request) -> Optional[str]:
    """Get the current logged-in user from the session cookie.

    Validates whichever ``session`` / ``session_<port>`` cookie is present by HMAC
    signature, so the exact (port-derived) cookie name doesn't matter — the front,
    an engine and the coderai-system worker bind different ports yet must all accept
    the same browser cookie."""
    if session_manager is None:
        return None
    for k, v in request.cookies.items():
        if k != "session" and not k.startswith("session_"):
            continue
        if v.endswith(".MUST_CHANGE"):
            v = v[:-12]
        user = session_manager.validate_session(v)
        if user:
            return user
    return None


def require_auth(request: Request) -> str:
    """Dependency that requires authentication."""
    username = get_current_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return username


def require_admin(request: Request) -> str:
    """Dependency that requires admin role."""
    username = require_auth(request)
    if not session_manager.is_admin(username):
        raise HTTPException(status_code=403, detail="Admin access required")
    return username


@router.post("/login", summary="Authenticate admin login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    """Handle login form submission."""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session manager not initialized")

    session_cookie = session_manager.authenticate(username, password)

    if not session_cookie:
        return _tmpl(request, "login.html", {"error": "Invalid username or password"})

    # Check if must change password
    must_change = session_cookie.endswith(".MUST_CHANGE")
    if must_change:
        session_cookie = session_cookie[:-12]

    redirect_path = "/admin/change-password" if must_change else "/admin"
    response = RedirectResponse(url=_url(request, redirect_path), status_code=302)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_cookie,
        httponly=True,
        secure=False,  # Set to True if using HTTPS
        samesite="strict",
        max_age=30 * 24 * 60 * 60
    )
    return response


@router.get("/logout", summary="Log out")
async def logout(request: Request):
    """Handle logout."""
    if session_manager:
        cookie = request.cookies.get(SESSION_COOKIE_NAME)
        session_manager.destroy_session(cookie)

    response = RedirectResponse(url=_url(request, "/login"), status_code=302)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


@router.post("/admin/change-password", summary="Change admin password")
async def change_password(
    request: Request,
    old_password: Optional[str] = Form(None),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    username: str = Depends(require_auth)
):
    user = session_manager.get_user(username)
    is_admin = session_manager.is_admin(username)
    must_change = user.get("must_change_password", False) if user else False

    def render_error(msg: str):
        return _tmpl(request, "change_password.html", {
            "username": username, "must_change": must_change,
            "is_admin": is_admin, "error": msg,
        })

    if new_password != confirm_password:
        return render_error("Passwords do not match")
    if len(new_password) < 8:
        return render_error("Password must be at least 8 characters")

    if must_change:
        success = session_manager.force_password_change(username, new_password)
    else:
        if not old_password:
            return render_error("Current password is required")
        success = session_manager.change_password(username, old_password, new_password)

    if not success:
        return render_error("Current password is incorrect")

    return RedirectResponse(url=_url(request, "/admin"), status_code=302)


@router.get("/admin/api/models", summary="List configured models")
async def api_list_models(username: str = Depends(require_admin)):
    """List all configured models with details."""
    models_data = session_manager._load_auth_data()  # TODO: move to ModelManager
    from codai.models.manager import multi_model_manager
    try:
        # all_engines=True: the admin config view must show every configured model,
        # including ones pinned to a secondary engine (e.g. whisper-servers on the
        # radeon engine) which the per-engine assignment filter would otherwise hide.
        return multi_model_manager.list_models(all_engines=True)
    except Exception:
        return []


_DISK_MIN_FREE_BYTES = 256 * 1024 * 1024  # 256 MB safety margin


def _check_disk_space(path: str, needed_bytes: int = 0) -> None:
    """Raise RuntimeError if `path`'s filesystem lacks enough free space."""
    import os as _os, shutil
    # Walk up to an existing ancestor — the target dir may not exist yet on first download
    check_path = path
    while check_path and not _os.path.exists(check_path):
        parent = _os.path.dirname(check_path)
        if parent == check_path:
            break
        check_path = parent
    try:
        free = shutil.disk_usage(check_path).free
    except OSError:
        return  # can't stat — proceed anyway
    required = needed_bytes + _DISK_MIN_FREE_BYTES
    if free < required:
        free_gb = free / 1e9
        needed_gb = needed_bytes / 1e9
        msg = (
            f"Not enough disk space: {free_gb:.1f} GB free"
            + (f", ~{needed_gb:.1f} GB needed" if needed_bytes else "")
            + ". Free up space and try again."
        )
        raise RuntimeError(msg)


def _get_hf_expected_size(model_id: str, file_pattern: str) -> int:
    """Return expected download size in bytes for a HF model (best-effort, 0 on failure)."""
    try:
        import fnmatch
        from huggingface_hub import model_info as _hf_model_info
        info = _hf_model_info(model_id, files_metadata=True)
        siblings = info.siblings or []
        if file_pattern:
            if file_pattern.startswith('.'):
                pats = [f"*{file_pattern}"]
            elif '/' in file_pattern:
                pats = [file_pattern]
            else:
                pats = [f"*{file_pattern}"]
            siblings = [s for s in siblings if any(fnmatch.fnmatch(s.rfilename, p) for p in pats)]
        return sum(getattr(s, 'size', 0) or 0 for s in siblings)
    except Exception:
        return 0


def _make_tqdm_class(pq, status=None, session_id=None, cache_dir=None):
    """Return a tqdm-compatible class that forwards progress events to pq and optionally updates a status dict."""
    import time as _time

    class _PQTqdm:
        def __init__(self, iterable=None, desc=None, total=None, initial=0, **kwargs):
            self.iterable = iterable
            self.desc = str(desc or 'downloading')
            self.total = int(total) if total else 0
            self.n = int(initial) if initial else 0
            self._start = _time.time()
            self._update_count = 0
            if self.total:
                pq.put({"type": "start", "filename": self.desc, "total": self.total})
                if status is not None:
                    status.update({"status": "downloading", "filename": self.desc,
                                   "total": self.total, "downloaded": self.n, "percent": 0})

        def update(self, n=1):
            if session_id and session_id in _download_cancelled:
                raise RuntimeError("Download cancelled by user")
            self.n += n
            self._update_count += 1
            # Check disk space every 64 progress ticks
            if cache_dir and self._update_count % 64 == 0:
                _check_disk_space(cache_dir)
            elapsed = (_time.time() - self._start) or 0.001
            rate = self.n / elapsed
            eta = (self.total - self.n) / rate if rate and self.total else None
            pct = round(self.n / self.total * 100, 1) if self.total else 0
            evt = {
                "type": "progress",
                "filename": self.desc,
                "downloaded": self.n,
                "total": self.total,
                "percent": pct,
                "rate": round(rate),
                "eta": round(eta) if eta is not None else None,
            }
            pq.put(evt)
            if status is not None:
                status.update({"status": "downloading", "filename": self.desc,
                               "percent": pct, "rate": round(rate), "eta": evt["eta"],
                               "downloaded": self.n, "total": self.total})

        def close(self): pass
        def refresh(self, nolock=False, lock_args=None): pass
        def clear(self, nolock=False): pass
        def display(self, msg=None, pos=None): pass
        def unpause(self): pass
        def moveto(self, n): pass
        def set_postfix(self, *a, **kw): pass
        def set_description(self, desc=None, **kw):
            if desc: self.desc = str(desc)
        def set_postfix_str(self, *a, **kw): pass
        def reset(self, total=None):
            self.n = 0
            self._start = _time.time()
            if total is not None: self.total = int(total)
        def __enter__(self): return self
        def __exit__(self, *a): self.close()
        def __iter__(self):
            for obj in (self.iterable or []):
                yield obj
        def write(self, s, **kw):
            pq.put({"type": "info", "message": str(s)})

        monitor_interval = 0
        monitor = None
        _lock = None

        @classmethod
        def get_lock(cls):
            import threading
            if cls._lock is None:
                cls._lock = threading.RLock()
            return cls._lock

        @classmethod
        def set_lock(cls, lock):
            cls._lock = lock

    return _PQTqdm


def _run_download_thread(session_id: str, model_id: str, file_pattern: str, pq):
    """Supervisor thread: spawn a child process that performs the download and
    relay its progress events onto the SSE queue `pq`. Running the download out of
    process is what makes it cancellable — see the inline note below."""
    import time
    import os

    status = {"session_id": session_id, "model_id": model_id, "file_pattern": file_pattern,
              "status": "starting", "started_at": time.time(),
              "percent": 0, "filename": "", "rate": 0, "eta": None}
    _download_status[session_id] = status

    def push(evt):
        pq.put(evt)
        t = evt.get("type")
        if t == "start":
            status.update({"status": "downloading", "filename": evt.get("filename", ""),
                           "total": evt.get("total", 0), "downloaded": 0, "percent": 0})
        elif t == "progress":
            status.update({"status": "downloading",
                           "filename": evt.get("filename", status.get("filename", "")),
                           "percent": evt.get("percent", 0), "rate": evt.get("rate", 0),
                           "eta": evt.get("eta"), "downloaded": evt.get("downloaded", 0),
                           "total": evt.get("total", 0)})
        elif t == "done":
            status.update({"status": "done", "percent": 100, "path": evt.get("path", "")})
        elif t == "error":
            status.update({"status": "error", "error": evt.get("message", "")})
        elif t == "info":
            status["last_info"] = evt.get("message", "")

    # Run the actual download in a SEPARATE SUBPROCESS so it can be cancelled
    # reliably. huggingface_hub fetches each file over several parallel chunk
    # connections (and, with Xet, a separate transfer path) that ignore any
    # in-thread stop flag, so a daemon thread can't be interrupted — but
    # terminating a process tears every connection down at once. We launch a clean
    # `python -m codai.admin.download_worker` (NOT multiprocessing: the spawn start
    # method re-imports the parent's __main__, i.e. the server launcher, which
    # hangs re-initialising the whole server). The child streams progress events as
    # JSON lines on stdout, which we relay onto this session's SSE queue.
    import subprocess as _sp
    import sys as _sys
    import collections as _collections
    import pathlib as _pathlib

    # The worker runs as `python -m codai.admin.download_worker`; when coderai is
    # run from source (not pip-installed) the child won't find the `codai`
    # package unless the repo root is on its path. routes.py lives at
    # <repo>/codai/admin/routes.py, so parents[2] is the repo root.
    _repo_root = str(_pathlib.Path(__file__).resolve().parents[2])

    # huggingface_hub raises this when a Xet-only blob is too large for the plain
    # HTTPS path and Xet is disabled — we re-enable Xet and retry in that case.
    _XET_REQUIRED_RE = re.compile(
        r"too large to be downloaded using the regular download|"
        r"install\s+`?hf_xet`?|xet-powered", re.IGNORECASE)

    def _hf_xet_available() -> bool:
        try:
            import hf_xet  # noqa: F401
            return True
        except Exception:
            return False

    def _attempt(disable_xet: bool, force_xet: bool = False, hold_error: bool = False):
        """Spawn the worker once; relay its events.

        Returns ``(terminal, rc, tail, error_msg)``. When ``hold_error`` is set, an
        ``error`` event is captured into ``error_msg`` instead of being pushed to
        the client (so the caller can decide to retry — e.g. enable Xet — without
        the user seeing a spurious error first)."""
        env = dict(os.environ)
        env["PYTHONPATH"] = _repo_root + (os.pathsep + env["PYTHONPATH"]
                                          if env.get("PYTHONPATH") else "")
        # hf_xet (the accelerated transfer) bypasses our tqdm progress hook — so
        # the bar freezes near 100% while a big file silently downloads — and can
        # hard-crash the worker (segfault / signal kill) with no traceback. The
        # plain HTTPS path reports byte-accurate progress and is reliable, so we
        # default to it unless the operator explicitly opted in (set
        # HF_HUB_DISABLE_XET=0). A crash retry always disables it. BUT some blobs
        # are Xet-only and the plain path refuses them ("file too large … install
        # hf_xet") — for those we re-enable Xet (force_xet) since it IS installed.
        if force_xet:
            env["HF_HUB_DISABLE_XET"] = "0"
        elif disable_xet or os.environ.get("HF_HUB_DISABLE_XET") is None:
            env["HF_HUB_DISABLE_XET"] = "1"
        proc = _sp.Popen(
            [_sys.executable, "-m", "codai.admin.download_worker", model_id, file_pattern or ""],
            stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, bufsize=1, env=env, cwd=_repo_root,
            preexec_fn=_worker_preexec if os.name == "posix" else None,
        )
        _download_procs[session_id] = proc
        terminal = None
        held_error_msg = ""
        recent = _collections.deque(maxlen=12)
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = _j.loads(line)
                except Exception:
                    # Non-JSON output (warnings / tracebacks) → surface as info
                    # and keep a tail so a hard crash can report what it printed.
                    recent.append(line)
                    push({"type": "info", "message": line})
                    continue
                etype = evt.get("type")
                if etype == "error" and hold_error:
                    held_error_msg = evt.get("message", "") or ""
                    terminal = "error"
                    continue
                push(evt)
                if etype in ("done", "error"):
                    terminal = etype
        except Exception as exc:
            push({"type": "error", "message": str(exc)})
            terminal = "error"
        finally:
            if proc.poll() is None:
                try:
                    proc.terminate(); proc.wait(timeout=10)
                except Exception:
                    pass
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            _download_procs.pop(session_id, None)
        return terminal, proc.poll(), " | ".join(list(recent)[-4:]).strip(), held_error_msg

    try:
        # First pass with our reliable Xet-disabled default, but hold any error so
        # we can transparently retry Xet-only blobs instead of failing the user.
        terminal, rc, tail, errmsg = _attempt(disable_xet=False, hold_error=True)
        # Xet-only large file refused by the plain HTTPS path → re-enable Xet
        # (hf_xet is bundled) and try again.
        xet_required = bool(terminal == "error" and errmsg
                            and _XET_REQUIRED_RE.search(errmsg))
        if xet_required and _hf_xet_available():
            push({"type": "info",
                  "message": "Large Xet-backed file detected; retrying with the Xet "
                             "accelerator (progress may not update during transfer)…"})
            _download_status.get(session_id, {}).update({"status": "downloading", "percent": 0})
            terminal, rc, tail, errmsg = _attempt(disable_xet=False, force_xet=True)
        elif terminal == "error" and errmsg:
            # Held a non-Xet error on the first pass → surface it now.
            push({"type": "error", "message": errmsg})
        # A hard crash (no done/error event, not a user cancel) is the classic
        # hf_xet failure — retry once with Xet disabled before giving up.
        crashed = (terminal is None and session_id not in _download_cancelled)
        if crashed and "HF_HUB_DISABLE_XET" not in os.environ:
            push({"type": "info",
                  "message": "Transfer crashed; retrying without the Xet accelerator…"})
            _download_status.get(session_id, {}).update({"status": "downloading", "percent": 0})
            terminal, rc, tail, errmsg = _attempt(disable_xet=True)

        if terminal is None:
            # Still no final event → cancelled or died for good.
            if session_id in _download_cancelled:
                pq.put({"type": "cancelled", "message": "Download cancelled by user"})
                _download_status.get(session_id, {}).update({"status": "cancelled"})
            else:
                detail = f"Download process exited unexpectedly (exit code {rc})"
                if rc is not None and rc < 0:
                    detail += f" — killed by signal {-rc} (often out-of-memory)"
                if tail:
                    detail += f". Last output: {tail}"
                push({"type": "error", "message": detail})
    finally:
        _download_cancelled.discard(session_id)

        def _gc():
            time.sleep(300)
            _download_sessions.pop(session_id, None)
            _download_status.pop(session_id, None)
        _t.Thread(target=_gc, daemon=True).start()


@router.post("/admin/api/model-download", summary="Download a model")
async def api_download_model(
    request: Request,
    username: str = Depends(require_admin)
):
    """Start a background download; returns session_id for SSE progress streaming."""
    data = await request.json()
    model_id = data.get("model_id")
    file_pattern = (data.get("file_pattern") or "").strip()

    if not model_id:
        raise HTTPException(status_code=400, detail="Model ID required")

    # Dedup: if this exact model is already downloading (e.g. the previous attempt
    # survived a page reload, or the user clicked "download" again), attach to the
    # live session instead of spawning a second worker. A duplicate worker would
    # only deadlock on huggingface_hub's per-blob file lock and show 0% forever
    # while the first worker quietly finishes.
    existing = _active_download_session(model_id, file_pattern)
    if existing:
        return {"session_id": existing, "attached": True}

    # A download supersedes any "to download" wishlist entry for this model.
    if config_manager is not None:
        changed = _prune_to_download(model_id)
        if file_pattern:
            changed = _prune_to_download(file_pattern) or changed
        if changed:
            config_manager.save_models()

    session_id = str(_uuid.uuid4())
    pq = _q.Queue()
    _download_sessions[session_id] = pq

    _t.Thread(
        target=_run_download_thread,
        args=(session_id, model_id, file_pattern, pq),
        daemon=True,
    ).start()

    return {"session_id": session_id}


@router.get("/admin/api/download-stream/{session_id}", summary="Stream model download progress")
async def api_download_stream(
    session_id: str,
    request: Request,
    username: str = Depends(require_admin),
):
    """Server-Sent Events stream for download progress."""
    import asyncio

    pq = _download_sessions.get(session_id)
    if pq is None:
        raise HTTPException(status_code=404, detail="Download session not found")

    async def _generate():
        loop = asyncio.get_event_loop()
        while True:
            try:
                evt = await loop.run_in_executor(None, lambda: pq.get(timeout=2))
                yield f"data: {_j.dumps(evt)}\n\n"
                if evt.get("type") in ("done", "error", "cancelled"):
                    break
            except _q.Empty:
                yield 'data: {"type":"keepalive"}\n\n'

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.delete("/admin/api/models/{model_identifier}", summary="Remove a configured model")
async def api_delete_model(
    model_identifier: str,
    username: str = Depends(require_admin)
):
    """Remove a model from local cache."""
    from codai.models.cache import remove_cached_model
    
    try:
        removed = remove_cached_model(model_identifier)
        if not removed:
            raise HTTPException(status_code=404, detail="Model not found")
        return {"success": True, "removed_count": len(removed)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Download status / cache management ---

@router.get("/admin/api/hf-files", summary="List files in a Hugging Face repo")
async def api_hf_repo_files(repo_id: str, username: str = Depends(require_admin)):
    """Return the file list for a HuggingFace repo with name and size metadata."""
    import asyncio

    def _fetch():
        try:
            from huggingface_hub import model_info as _hf_model_info
            info = _hf_model_info(repo_id, files_metadata=True)
            return {
                "repo_id": repo_id,
                "files": [
                    {"name": f.rfilename, "size": getattr(f, "size", None)}
                    for f in (info.siblings or [])
                ],
            }
        except Exception as exc:
            return {"repo_id": repo_id, "files": [], "error": str(exc)}

    return await asyncio.to_thread(_fetch)


@router.get("/admin/api/downloads", summary="List active downloads")
async def api_list_downloads(username: str = Depends(require_admin)):
    """Return status of all active and recently completed download sessions."""
    return list(_download_status.values())


# Catalog of the official DeepSeek V4 GGUF weights ds4 ships (matches antirez's
# download_model.sh). Picking one and downloading it reuses the normal model
# downloader (/admin/api/model-download with file_pattern = the exact .gguf), which
# flattens the file into the gguf cache and surfaces it in the model list.
_DS4_DEFAULT_REPO = "antirez/deepseek-v4-gguf"
_DS4_DEFAULT_MODELS = [
    {"key": "q2-imatrix", "label": "Flash q2-imatrix (~81 GB) — recommended for 96/128 GB RAM",
     "file": "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
     "size_gb": 81},
    {"key": "q2-q4-imatrix", "label": "Flash q2-q4-imatrix (~98 GB) — higher quality, last layers Q4",
     "file": "DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf",
     "size_gb": 98},
    {"key": "q4-imatrix", "label": "Flash q4-imatrix (~153 GB) — best quality, 256 GB+ RAM",
     "file": "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf",
     "size_gb": 153},
    {"key": "mtp", "label": "MTP speculative-decoding component (~3.5 GB) — optional, run ds4 with --mtp",
     "file": "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf",
     "size_gb": 4},
]


@router.get("/admin/api/ds4/default-models", summary="List downloadable ds4 default models")
async def api_ds4_default_models(username: str = Depends(require_admin)):
    """Catalog of official DeepSeek V4 GGUF variants ds4 can serve. The UI offers
    these in a select; downloading one goes through /admin/api/model-download."""
    gguf_dir = ""
    try:
        if config_manager is not None and config_manager.config is not None:
            gguf_dir = (config_manager.config.models.gguf_cache_dir or "").strip()
    except Exception:
        gguf_dir = ""
    out = []
    for m in _DS4_DEFAULT_MODELS:
        present = False
        if gguf_dir:
            try:
                present = os.path.exists(os.path.join(os.path.expanduser(gguf_dir), m["file"]))
            except Exception:
                present = False
        out.append({**m, "repo": _DS4_DEFAULT_REPO, "present": present})
    return {"repo": _DS4_DEFAULT_REPO, "gguf_cache_dir": gguf_dir, "models": out}


def _cancel_download_session(session_id: str) -> bool:
    """Cancel an active download by flagging the session and terminating its worker
    process. Returns False if there is no such download session.

    Flagging the session (so the supervisor classifies it as cancelled, not
    failed) and killing the child process tears down every HF chunk connection at
    once — the supervisor's relay loop then exits cleanly. Shared by the dedicated
    download-cancel endpoint and the unified Tasks-page cancel path."""
    if session_id not in _download_sessions and session_id not in _download_status:
        return False
    _download_cancelled.add(session_id)
    proc = _download_procs.get(session_id)
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
    return True


@router.post("/admin/api/download-cancel/{session_id}", summary="Cancel a download")
async def api_cancel_download(session_id: str, username: str = Depends(require_admin)):
    """Cancel an active download by terminating its worker process immediately."""
    if not _cancel_download_session(session_id):
        raise HTTPException(status_code=404, detail="Download session not found")
    return {"success": True}


@router.post("/admin/api/download-cancel-all", summary="Cancel all active downloads")
async def api_cancel_all_downloads(username: str = Depends(require_admin)):
    """Cancel every active download at once (terminates all worker processes)."""
    sessions = list(_download_procs.keys())
    for sid in sessions:
        _download_cancelled.add(sid)
        proc = _download_procs.get(sid)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
    return {"success": True, "cancelled": len(sessions)}


@router.post("/admin/api/model-upload", summary="Upload a model file")
async def api_model_upload(request: Request, username: str = Depends(require_admin)):
    """Upload a GGUF model file in chunks."""
    from codai.models.cache import get_model_cache_dir
    import tempfile
    
    form = await request.form()
    chunk = form.get("chunk")
    filename = form.get("filename", "model.gguf")
    chunk_index = int(form.get("chunk_index", 0))
    total_chunks = int(form.get("total_chunks", 1))

    if not chunk or not hasattr(chunk, "read"):
        raise HTTPException(status_code=400, detail="No file chunk provided")

    # Reject path-traversal in the client-supplied filename: it must be a bare
    # name that stays inside the model cache directory.
    filename = os.path.basename(str(filename))
    if not filename or filename in (".", "..") or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    cache_dir = get_model_cache_dir()
    temp_dir = tempfile.gettempdir()
    # upload_id only ever names a temp scratch file; derive a safe slug from it
    # so it can't escape temp_dir or collide via traversal either.
    upload_id = os.path.basename(str(form.get("upload_id", filename))) or filename
    upload_id = re.sub(r"[^A-Za-z0-9._-]", "_", upload_id)
    temp_path = os.path.join(temp_dir, f"upload_{upload_id}.part")
    
    # Append chunk
    chunk_data = await chunk.read()
    with open(temp_path, "ab") as f:
        f.write(chunk_data)
    
    # If last chunk, move to final location
    if chunk_index == total_chunks - 1:
        final_path = os.path.join(cache_dir, filename)
        # Belt-and-suspenders: ensure the resolved destination is still inside
        # the cache directory before committing the upload.
        if os.path.commonpath([os.path.realpath(final_path),
                                os.path.realpath(cache_dir)]) != os.path.realpath(cache_dir):
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise HTTPException(status_code=400, detail="Invalid destination path")
        os.replace(temp_path, final_path)
        return {"success": True, "complete": True, "path": final_path}
    
    return {"success": True, "complete": False, "chunk_index": chunk_index}


# ── cache scan helpers (run in thread pool) ──────────────────────────────────

def _hf_repo_id_from_path(path: str) -> str:
    """Extract a HuggingFace repo ID from an HF hub cache path.

    HF hub cache paths look like:
      .../hub/models--OWNER--REPO-NAME/snapshots/HASH/filename.gguf
    The first '--' inside the 'models--...' component separates owner from repo.
    """
    for part in path.replace('\\', '/').split('/'):
        if part.startswith('models--'):
            repo_part = part[len('models--'):]
            sep = repo_part.find('--')
            if sep != -1:
                return repo_part[:sep] + '/' + repo_part[sep + 2:]
    return ''


# Categories that hold real (configured) models in models.json.
_VALID_MODEL_CATS = {
    "text_models", "image_models", "audio_models", "gguf_models", "tts_models",
    "vision_models", "video_models", "audio_gen_models", "embedding_models",
    "spatial_models",
}


def _entry_key(entry) -> str:
    """The identifying path/id of a models.json entry (str or dict)."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("path") or entry.get("id") or ""
    return ""


def _basename_key(key: str) -> str:
    import os as _os
    return _os.path.basename(key) if ("/" in key or _os.sep in key) else key


def _is_model_configured(model_id: str) -> bool:
    """True if model_id is already a configured model (matched by id or basename)."""
    if config_manager is None:
        return False
    fname = _basename_key(model_id)
    for cat in _VALID_MODEL_CATS:
        for m in config_manager.models_data.get(cat, []):
            key = _entry_key(m)
            if key == model_id or (fname and _basename_key(key) == fname):
                return True
    return False


def _prune_to_download(model_id: str) -> bool:
    """Drop any 'to download' wishlist entry matching model_id. Returns True if changed."""
    if config_manager is None:
        return False
    lst = config_manager.models_data.get("to_download")
    if not lst:
        return False
    fname = _basename_key(model_id)
    kept = [e for e in lst
            if not (_entry_key(e) == model_id
                    or (fname and _basename_key(_entry_key(e)) == fname))]
    if len(kept) != len(lst):
        config_manager.models_data["to_download"] = kept
        return True
    return False


def _scan_caches() -> dict:
    import os
    result: dict = {"hf": [], "gguf": []}

    from codai.models.cache import get_all_cache_dirs, get_model_cache_dir
    from codai.models.capabilities import (
        detect_model_capabilities, lookup_capability_cache,
    )
    caches = get_all_cache_dirs()

    # Collect configured models.
    # configured_settings: path → primary (first) config entry  (backward compat)
    # all_configs:         path → list of all config entries     (for multi-config support)
    configured_settings: dict = {}
    all_configs: dict = {}
    if config_manager:
        md = config_manager.models_data
        for cat in ("text_models", "image_models", "audio_models",
                    "gguf_models", "tts_models", "vision_models", "video_models",
                    "audio_gen_models", "embedding_models", "spatial_models"):
            for m in md.get(cat, []):
                if isinstance(m, str):
                    p, s = m, {}
                else:
                    # A whisper-server RUNNER (has model_path: port/gpu/which-model)
                    # is managed in its own card, not as a config of the backing GGUF
                    # file — exclude it. The whisper MODEL config (a .gguf entry with
                    # backend=whisper-server but no model_path) IS shown on the GGUF
                    # row as that file's config.
                    if _is_whisper_runner(m):
                        continue
                    p = m.get("path") or m.get("id") or ""
                    s = m if isinstance(m, dict) else {}
                if not p:
                    continue
                if p not in configured_settings:
                    configured_settings[p] = (s, cat)
                # A single logical config can be registered under multiple
                # categories via model_types (for example text+vision). It is
                # stored once per category in models.json with the same
                # config_id, but the UI should show it as one editable config,
                # not duplicate pills that appear to delete each other.
                _cfg_list = all_configs.setdefault(p, [])
                _cid = s.get("config_id") if isinstance(s, dict) else None
                _existing = None
                if _cid:
                    for _cfg in _cfg_list:
                        _settings = _cfg.get("settings") or {}
                        if isinstance(_settings, dict) and _settings.get("config_id") == _cid:
                            _existing = _cfg
                            break
                if _existing is not None:
                    _cats = _existing.setdefault("cats", [])
                    if cat not in _cats:
                        _cats.append(cat)
                    if not _existing.get("cat"):
                        _existing["cat"] = cat
                else:
                    _cfg_list.append({"settings": s, "cat": cat, "cats": [cat]})

    # Secondary index: basename → (settings_tuple, original_path)
    # Used to reconnect a config to a re-downloaded file that landed at a different path.
    # Only populated for .gguf entries whose basename is unique (avoids ambiguous matches).
    _cfg_by_fname: dict = {}
    for _p, _val in configured_settings.items():
        _bn = os.path.basename(_p) if ('/' in _p or os.sep in _p) else _p
        if _bn and _bn.endswith('.gguf'):
            if _bn in _cfg_by_fname:
                _cfg_by_fname[_bn] = None   # mark as ambiguous — don't use
            else:
                _cfg_by_fname[_bn] = (_val, _p)

    # HuggingFace cache
    hf_dir = caches.get("huggingface")
    if hf_dir:
        try:
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir(hf_dir)

            # Build set of repo IDs that have incomplete/corrupted cache entries.
            # huggingface_hub reports these via info.warnings (CorruptedCacheException).
            incomplete_repos: set = set()
            for w in getattr(info, 'warnings', []):
                rid = getattr(w, 'repo_id', None)
                if rid:
                    incomplete_repos.add(str(rid))
            # Also scan each repo's blobs directory for .incomplete marker files
            # (used by some huggingface_hub versions for in-progress downloads).
            try:
                for _repo_entry in os.scandir(hf_dir):
                    if not _repo_entry.is_dir() or not _repo_entry.name.startswith('models--'):
                        continue
                    _blobs = os.path.join(_repo_entry.path, 'blobs')
                    if os.path.isdir(_blobs) and any(
                        n.endswith('.incomplete') or n.endswith('.lock')
                        for n in os.listdir(_blobs)
                    ):
                        _rid = _repo_entry.name[len('models--'):].replace('--', '/', 1)
                        incomplete_repos.add(_rid)
            except Exception:
                pass

            for repo in sorted(info.repos, key=lambda r: r.repo_id):
                revs = sorted(repo.revisions, key=lambda r: r.commit_hash)
                size_bytes = sum(r.size_on_disk for r in repo.revisions)
                files = sorted(f.file_name for f in revs[-1].files) if revs else []

                # If ALL model files are .gguf, treat as GGUF entries not HF entries
                model_files = [f for f in files if not f.endswith(('.json', '.txt', '.md', '.py', '.gitattributes'))]
                if model_files and all(f.endswith('.gguf') for f in model_files):
                    for rev in revs[-1:]:
                        for hf_file in rev.files:
                            if not hf_file.file_name.endswith('.gguf'):
                                continue
                            fpath = str(hf_file.file_path)
                            fname = hf_file.file_name
                            fsize = hf_file.size_on_disk
                            cfg = (configured_settings.get(fpath)
                                   or configured_settings.get(fname))
                            _fname_match = None if cfg else _cfg_by_fname.get(fname)
                            cfg = cfg or (_fname_match[0] if _fname_match else None) or ({}, None)
                            _configured_path = _fname_match[1] if _fname_match else None
                            cfg_s = cfg[0] if isinstance(cfg[0], dict) else {}
                            saved_caps = cfg_s.get("capabilities") or []
                            caps_list = saved_caps if saved_caps else detect_model_capabilities(fname).to_list()
                            _direct_match = fpath in configured_settings or fname in configured_settings
                            _gguf_key = fpath if fpath in all_configs else (fname if fname in all_configs else (_configured_path or fpath))
                            _entry = {
                                "filename": fname,
                                "path": fpath,
                                "size_gb": round(fsize / 1e9, 2),
                                "size_bytes": fsize,
                                "in_config": _direct_match or bool(_fname_match),
                                "model_type": cfg[1] if cfg[1] and cfg[1] != "gguf_models" else "text_models",
                                "settings": cfg_s,
                                "capabilities": caps_list,
                                "source_repo": repo.repo_id,
                                "incomplete": repo.repo_id in incomplete_repos,
                                "configs": all_configs.get(_gguf_key, []),
                            }
                            if _configured_path:
                                _entry["configured_path"] = _configured_path
                            result["gguf"].append(_entry)
                    continue  # skip adding to hf list

                cfg = configured_settings.get(repo.repo_id, ({}, None))
                cfg_settings = cfg[0] if isinstance(cfg[0], dict) else {}
                saved_caps = cfg_settings.get("capabilities") or []
                if saved_caps:
                    caps_list = saved_caps
                else:
                    caps = (lookup_capability_cache(repo.repo_id)
                            or detect_model_capabilities(repo.repo_id))
                    caps_list = caps.to_list()
                result["hf"].append({
                    "id": repo.repo_id,
                    "size_gb": round(size_bytes / 1e9, 2),
                    "size_bytes": size_bytes,
                    "revision_count": len(list(repo.revisions)),
                    "files": files[:30],
                    "file_count": len(files),
                    "in_config": repo.repo_id in configured_settings,
                    "model_type": cfg[1] if cfg[1] and cfg[1] != "gguf_models" else "text_models",
                    "settings": cfg_settings,
                    "capabilities": caps_list,
                    "incomplete": repo.repo_id in incomplete_repos,
                    "configs": all_configs.get(repo.repo_id, []),
                })
        except Exception as e:
            result["hf_error"] = str(e)

    # GGUF cache (coderai-specific)
    gguf_dir = caches.get("coderai") or get_model_cache_dir()
    if gguf_dir and os.path.exists(gguf_dir):
        # Files with these suffixes are known-incomplete downloads
        _incomplete_gguf_stems = {
            os.path.splitext(n)[0]
            for n in os.listdir(gguf_dir)
            if n.endswith(('.part', '.tmp', '.download', '.incomplete'))
        }
        for fname in sorted(os.listdir(gguf_dir)):
            fpath = os.path.join(gguf_dir, fname)
            if not os.path.isfile(fpath):
                continue
            # Skip the partial-download sentinel files themselves
            if any(fname.endswith(s) for s in ('.part', '.tmp', '.download', '.incomplete')):
                continue
            size = os.path.getsize(fpath)
            cfg = (configured_settings.get(fpath)
                   or configured_settings.get(fname))
            _fname_match = None if cfg else _cfg_by_fname.get(fname)
            cfg = cfg or (_fname_match[0] if _fname_match else None) or ({}, None)
            _configured_path = _fname_match[1] if _fname_match else None
            cfg_s = cfg[0] if isinstance(cfg[0], dict) else {}
            saved_caps = cfg_s.get("capabilities") or []
            caps_list = saved_caps if saved_caps else detect_model_capabilities(fname).to_list()
            # A file is incomplete if there is a same-stem partial file alongside it,
            # or if an active download session targets this exact path/filename.
            _stem = os.path.splitext(fname)[0]
            _dl_active = any(
                s.get("model_id") in (fname, fpath) and s.get("status") not in ("done", "error", "cancelled")
                for s in _download_status.values()
            )
            _direct_match = fpath in configured_settings or fname in configured_settings
            _key = fpath if fpath in all_configs else (fname if fname in all_configs else (_configured_path or fpath))
            _entry = {
                "filename": fname,
                "path": fpath,
                "size_gb": round(size / 1e9, 2),
                "size_bytes": size,
                "in_config": _direct_match or bool(_fname_match),
                "model_type": cfg[1] if cfg[1] and cfg[1] != "gguf_models" else "text_models",
                "settings": cfg_s,
                "capabilities": caps_list,
                "incomplete": _stem in _incomplete_gguf_stems or _dl_active,
                "configs": all_configs.get(_key, []),
            }
            if _configured_path:
                _entry["configured_path"] = _configured_path
            result["gguf"].append(_entry)

    # Add configured GGUF models not yet in the list (e.g., HF repo IDs or external paths)
    existing_paths = {m["path"] for m in result["gguf"]}
    existing_fnames = {m["filename"] for m in result["gguf"]}
    for path, (settings, mtype) in configured_settings.items():
        if path in existing_paths or path in existing_fnames:
            continue
        fname = os.path.basename(path) if '/' in path else path
        if fname in existing_fnames:
            continue
        # Check if it's a GGUF model (ends with .gguf or is in a GGUF repo)
        is_gguf = path.endswith('.gguf') or 'gguf' in path.lower() or mtype == "gguf_models"
        if is_gguf:
            file_exists = os.path.isfile(path)
            size_bytes = os.path.getsize(path) if file_exists else 0
            caps = detect_model_capabilities(path)
            s = settings if isinstance(settings, dict) else {}
            # Derive HF repo ID from path when not explicitly stored in settings
            source_repo = s.get("source_repo") or _hf_repo_id_from_path(path)
            result["gguf"].append({
                "filename": os.path.basename(path) if '/' in path else path,
                "path": path,
                "size_gb": round(size_bytes / 1e9, 2) if size_bytes else 0,
                "size_bytes": size_bytes,
                "in_config": True,
                "missing": not file_exists,
                "source_repo": source_repo,
                "model_type": mtype if mtype and mtype != "gguf_models" else "text_models",
                "settings": s,
                "capabilities": caps.to_list(),
                "configs": all_configs.get(path, []),
            })

    # Add configured non-GGUF HF models whose files have been evicted from disk
    # (e.g. via "Free disk"). They are absent from the HF cache scan above, so
    # surface them here as missing so they keep a Re-download button.
    from codai.models.cache import is_huggingface_model_id
    existing_hf_ids = {m["id"] for m in result["hf"]}
    for path, (settings, mtype) in configured_settings.items():
        if path in existing_hf_ids:
            continue
        s = settings if isinstance(settings, dict) else {}
        if s.get("backend") == "whisper-server":
            continue
        # Only HF-style repo IDs (owner/repo) — skip local paths and GGUF files
        if os.path.isabs(path) or path.endswith('.gguf') or not is_huggingface_model_id(path):
            continue
        # A real local relative path that still exists isn't an evicted model
        if os.path.exists(path):
            continue
        caps = s.get("capabilities") or detect_model_capabilities(path).to_list()
        result["hf"].append({
            "id": path,
            "size_gb": 0, "size_bytes": 0, "revision_count": 0,
            "files": [], "file_count": 0,
            "in_config": True, "missing": True,
            "source_repo": path,
            "model_type": mtype if mtype and mtype != "gguf_models" else "text_models",
            "settings": s,
            "capabilities": caps,
            "incomplete": False,
            "configs": all_configs.get(path, []),
        })

    # Surface "to download" wishlist entries: models the user wants listed for
    # later download but has NOT configured and are NOT on disk. They appear as
    # non-configured rows with a download button (in_config=False, missing=True).
    seen_gguf = {m["path"] for m in result["gguf"]} | {m["filename"] for m in result["gguf"]}
    seen_hf = {m["id"] for m in result["hf"]}
    if config_manager:
        for entry in config_manager.models_data.get("to_download", []):
            e = entry if isinstance(entry, dict) else {"path": entry}
            mid = (e.get("path") or e.get("id") or "").strip()
            if not mid or _is_model_configured(mid):
                continue
            repo = e.get("source_repo") or mid
            mtype = e.get("model_type") or "text_models"
            is_gguf = (bool(e.get("is_gguf")) or mid.lower().endswith(".gguf")
                       or "gguf" in mid.lower() or mtype == "gguf_models")
            fname = os.path.basename(mid) if ("/" in mid or os.sep in mid) else mid
            caps = e.get("capabilities") or detect_model_capabilities(mid).to_list()
            if is_gguf:
                if mid in seen_gguf or fname in seen_gguf:
                    continue
                result["gguf"].append({
                    "filename": fname, "path": mid,
                    "size_gb": 0, "size_bytes": 0,
                    "in_config": False, "missing": True, "to_download": True,
                    "source_repo": repo,
                    "model_type": mtype if mtype != "gguf_models" else "text_models",
                    "settings": {}, "capabilities": caps,
                    "incomplete": False, "configs": [],
                })
                seen_gguf.add(mid); seen_gguf.add(fname)
            else:
                if mid in seen_hf:
                    continue
                result["hf"].append({
                    "id": mid, "size_gb": 0, "size_bytes": 0, "revision_count": 0,
                    "files": [], "file_count": 0,
                    "in_config": False, "missing": True, "to_download": True,
                    "source_repo": repo, "model_type": mtype,
                    "settings": {}, "capabilities": caps,
                    "incomplete": False, "configs": [],
                })
                seen_hf.add(mid)

    return result


def _get_cache_stats() -> dict:
    import os
    stats = {"hf_bytes": 0, "hf_models": 0, "gguf_bytes": 0, "gguf_files": 0,
             "hf_disk_free_bytes": None, "hf_disk_total_bytes": None,
             "gguf_disk_free_bytes": None, "gguf_disk_total_bytes": None}
    from codai.models.cache import get_all_cache_dirs, get_model_cache_dir
    caches = get_all_cache_dirs()

    hf_dir = caches.get("huggingface")
    if hf_dir:
        try:
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir(hf_dir)
            # Only count non-GGUF repos
            for repo in info.repos:
                revs = list(repo.revisions)
                if not revs:
                    continue
                files = [f.file_name for f in revs[-1].files]
                model_files = [f for f in files if not f.endswith(('.json', '.txt', '.md', '.py', '.gitattributes'))]
                # Skip if all model files are GGUF
                if model_files and all(f.endswith('.gguf') for f in model_files):
                    continue
                stats["hf_bytes"] += sum(r.size_on_disk for r in repo.revisions)
                stats["hf_models"] += 1
        except Exception:
            pass
        # HF disk space
        try:
            sv = os.statvfs(hf_dir)
            stats["hf_disk_free_bytes"] = sv.f_bavail * sv.f_frsize
            stats["hf_disk_total_bytes"] = sv.f_blocks * sv.f_frsize
        except Exception:
            pass

    gguf_dir = caches.get("coderai") or get_model_cache_dir()
    if gguf_dir and os.path.exists(gguf_dir):
        files = [f for f in os.listdir(gguf_dir)
                 if os.path.isfile(os.path.join(gguf_dir, f))]
        stats["gguf_files"] = len(files)
        stats["gguf_bytes"] = sum(os.path.getsize(os.path.join(gguf_dir, f)) for f in files)
        # GGUF disk space
        try:
            sv = os.statvfs(gguf_dir)
            stats["gguf_disk_free_bytes"] = sv.f_bavail * sv.f_frsize
            stats["gguf_disk_total_bytes"] = sv.f_blocks * sv.f_frsize
        except Exception:
            pass

    # Also count GGUF files in HF cache
    if hf_dir:
        try:
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir(hf_dir)
            for repo in info.repos:
                revs = list(repo.revisions)
                if not revs:
                    continue
                files = [f.file_name for f in revs[-1].files]
                model_files = [f for f in files if not f.endswith(('.json', '.txt', '.md', '.py', '.gitattributes'))]
                # If all model files are GGUF, count them in gguf_bytes
                if model_files and all(f.endswith('.gguf') for f in model_files):
                    for rev in repo.revisions:
                        for hf_file in rev.files:
                            if hf_file.file_name.endswith('.gguf'):
                                stats["gguf_bytes"] += hf_file.size_on_disk
                                stats["gguf_files"] += 1
        except Exception:
            pass

    return stats


def _do_clear_cache(cache_type: str) -> dict:
    import os, shutil
    from codai.models.cache import get_all_cache_dirs, get_model_cache_dir
    caches = get_all_cache_dirs()
    freed = 0

    if cache_type in ("all", "hf"):
        hf_dir = caches.get("huggingface")
        if hf_dir and os.path.exists(hf_dir):
            try:
                from huggingface_hub import scan_cache_dir
                info = scan_cache_dir(hf_dir)
                hashes = [r.commit_hash for repo in info.repos for r in repo.revisions]
                if hashes:
                    strategy = info.delete_revisions(*hashes)
                    freed += strategy.expected_freed_size
                    strategy.execute()
            except Exception:
                for item in os.listdir(hf_dir):
                    p = os.path.join(hf_dir, item)
                    try:
                        if os.path.isdir(p):
                            shutil.rmtree(p)
                        else:
                            freed += os.path.getsize(p)
                            os.remove(p)
                    except Exception:
                        pass

    if cache_type in ("all", "gguf"):
        gguf_dir = caches.get("coderai") or get_model_cache_dir()
        if gguf_dir and os.path.exists(gguf_dir):
            for f in os.listdir(gguf_dir):
                fp = os.path.join(gguf_dir, f)
                if os.path.isfile(fp):
                    try:
                        freed += os.path.getsize(fp)
                        os.remove(fp)
                    except Exception:
                        pass

    return {"success": True, "freed_bytes": freed}


def _do_delete_model(model_id: str, cache_type: str) -> dict:
    import os, shutil
    from codai.models.cache import get_all_cache_dirs, get_model_cache_dir
    caches = get_all_cache_dirs()

    if cache_type == "hf":
        hf_dir = caches.get("huggingface")
        if hf_dir:
            # huggingface_hub logs a WARNING + full traceback when a repo dir has
            # already vanished (e.g. a GGUF model whose HF repo was never really
            # cached). Quiet it during the delete — "repo gone" is exactly the
            # end state Free disk wants, so it's not an error.
            import logging as _logging
            _hf_log = _logging.getLogger("huggingface_hub.utils._cache_manager")
            _prev_lvl = _hf_log.level
            _hf_log.setLevel(_logging.ERROR)
            try:
                from huggingface_hub import scan_cache_dir
                info = scan_cache_dir(hf_dir)
                repo = next((r for r in info.repos if r.repo_id == model_id), None)
                if repo:
                    hashes = [r.commit_hash for r in repo.revisions]
                    info.delete_revisions(*hashes).execute()
                    return {"success": True}
            except Exception:
                pass
            finally:
                _hf_log.setLevel(_prev_lvl)
            # Fallback: remove the repo dir directly if it's still there.
            safe = model_id.replace("/", "--")
            d = os.path.join(hf_dir, f"models--{safe}")
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
            # Whether or not anything was on disk, the files are gone now — Free
            # disk is idempotent, so report success instead of a scary error.
            return {"success": True}
        return {"success": False, "detail": "HF cache directory not configured"}

    if cache_type == "gguf":
        gguf_dir = get_model_cache_dir()
        # Support full absolute path (e.g. HF-cached GGUF) or bare filename
        if os.path.isabs(model_id):
            fp = model_id
        else:
            fp = os.path.join(gguf_dir, model_id)
        if os.path.isfile(fp):
            os.remove(fp)
            return {"success": True}
        return {"success": False, "detail": "File not found"}

    return {"success": False, "detail": "Unknown cache_type"}


# Cache scans walk the whole HF/GGUF cache on disk (huggingface_hub.scan_cache_dir
# + per-file size sums), which takes many seconds on a large cache. They change only
# when a model is added/removed/downloaded, so we gate the expensive scan on a CHEAP
# change-signature: the names + mtimes of the top-level model dirs/files. When the
# signature is unchanged we serve the cached result instantly and never re-walk;
# when it changes we refresh in the background (returning the last result meanwhile).
# A long backstop TTL re-scans occasionally even if mtimes somehow miss a change.
import threading as _cache_thr
_SCAN_TTL = 600.0
_scan_state = {"data": None, "sig": None, "at": 0.0, "busy": False}
_stats_state = {"data": None, "sig": None, "at": 0.0, "busy": False}


def _scan_signature():
    """Cheap fingerprint of the model caches: (path, mtime) of each top-level entry
    in the HF and GGUF cache dirs. A handful of stat() calls — orders of magnitude
    cheaper than the full scan it gates."""
    import os
    try:
        from codai.models.cache import get_all_cache_dirs, get_model_cache_dir
        caches = get_all_cache_dirs()
        dirs = [caches.get("huggingface"),
                caches.get("coderai") or get_model_cache_dir()]
    except Exception:
        return ()
    sig = []
    for d in dirs:
        if not d:
            continue
        try:
            for e in os.scandir(d):
                try:
                    sig.append((e.path, round(e.stat().st_mtime, 3)))
                except OSError:
                    pass
        except OSError:
            pass
    return tuple(sorted(sig))


def _cached(state, fn):
    import time as _t
    now = _t.time()
    sig = _scan_signature()
    fresh = (state["sig"] == sig and (now - state["at"]) < _SCAN_TTL)
    if state["data"] is not None and fresh:
        return state["data"]                      # nothing changed → instant
    if state["data"] is None:
        state["data"] = fn()                      # first call: compute synchronously
        state["sig"] = sig
        state["at"] = _t.time()
        return state["data"]
    if not state["busy"]:                          # changed: refresh in background
        state["busy"] = True

        def _bg():
            try:
                d = fn()
                state["data"] = d
                state["sig"] = _scan_signature()
                state["at"] = _t.time()
            finally:
                state["busy"] = False
        _cache_thr.Thread(target=_bg, daemon=True).start()
    return state["data"]


def _invalidate_cache_scan():
    """Force the next cached-models/cache-stats read to refresh (after a model is
    deleted/downloaded/freed)."""
    _scan_state["sig"] = None
    _stats_state["sig"] = None


@router.get("/admin/api/cached-models", summary="List cached models")
async def api_cached_models(username: str = Depends(require_admin)):
    """Scan both caches and return all locally stored models (stale-while-revalidate)."""
    import asyncio
    return await asyncio.to_thread(_cached, _scan_state, _scan_caches)


@router.get("/admin/api/cache-stats", summary="Model cache statistics")
async def api_cache_stats(username: str = Depends(require_admin)):
    """Return disk-usage statistics for each cache (stale-while-revalidate)."""
    import asyncio
    return await asyncio.to_thread(_cached, _stats_state, _get_cache_stats)


@router.delete("/admin/api/cache", summary="Clear the model cache")
async def api_clear_cache(cache_type: str = "all", username: str = Depends(require_admin)):
    """Bulk-delete cache. cache_type: all | hf | gguf"""
    import asyncio
    r = await asyncio.to_thread(_do_clear_cache, cache_type)
    _invalidate_cache_scan()
    return r


@router.delete("/admin/api/cached-models/{model_id:path}", summary="Evict a cached model")
async def api_delete_cached_model(
    model_id: str,
    cache_type: str = "hf",
    username: str = Depends(require_admin),
):
    """Delete a specific cached model (HF repo ID or GGUF filename)."""
    import asyncio
    r = await asyncio.to_thread(_do_delete_model, model_id, cache_type)
    _invalidate_cache_scan()
    return r


@router.post("/admin/api/model-free-disk", summary="Delete a model's files but keep its config")
async def api_model_free_disk(request: Request, username: str = Depends(require_admin)):
    """Reclaim disk space by deleting a model's files while keeping its
    models.json entry, so it can be re-downloaded on demand. The source repo is
    persisted onto the config entry first so the Re-download button has a target
    once the file is gone."""
    if config_manager is None:
        raise HTTPException(status_code=503, detail="Config manager not initialized")
    import os as _os, asyncio
    data = await request.json()
    path = (data.get("path") or data.get("model_id") or "").strip()
    cache_type = data.get("cache_type", "gguf")
    source_repo = (data.get("source_repo") or "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="path is required")

    # Persist source_repo onto the matching config entries so re-download works
    # after the file is deleted (flat GGUF files retain no HF repo info on disk).
    # Skip when the entry key already IS the repo id (HF models re-download by id).
    if source_repo and source_repo != path:
        fname = _os.path.basename(path) if ("/" in path or _os.sep in path) else ""
        changed = False
        for cat in ("text_models", "image_models", "audio_models",
                    "gguf_models", "tts_models", "vision_models", "video_models",
                    "audio_gen_models", "embedding_models", "spatial_models"):
            lst = config_manager.models_data.get(cat, [])
            for i, m in enumerate(lst):
                key = m if isinstance(m, str) else (m.get("path") or m.get("id") or "")
                if key == path or (fname and _os.path.basename(key) == fname):
                    if isinstance(m, str):
                        lst[i] = {"path": m, "source_repo": source_repo}
                        changed = True
                    elif not m.get("source_repo"):
                        m["source_repo"] = source_repo
                        changed = True
        if changed:
            config_manager.save_models()

    result = await asyncio.to_thread(_do_delete_model, path, cache_type)
    _invalidate_cache_scan()
    _broker_notify_models_updated(request)
    return result


@router.post("/admin/api/model-add-known", summary="Register a model in config without downloading")
async def api_model_add_known(request: Request, username: str = Depends(require_admin)):
    """Add a model to models.json as a known-but-not-downloaded reference.

    The model then appears in the model list as "missing" with a working
    Re-download button, without fetching any files now — the same end state as
    "Free disk", but reached without ever having the files locally."""
    if config_manager is None:
        raise HTTPException(status_code=503, detail="Config manager not initialized")
    import os as _os
    data = await request.json()
    model_id = (data.get("model_id") or data.get("path") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    source_repo = (data.get("source_repo") or model_id).strip()
    model_type = (data.get("model_type") or "").strip()
    is_gguf = (bool(data.get("is_gguf")) or model_type == "gguf_models"
               or "gguf" in model_id.lower())
    valid = {"text_models", "image_models", "audio_models", "gguf_models", "tts_models",
             "vision_models", "video_models", "audio_gen_models", "embedding_models", "spatial_models"}
    if is_gguf:
        model_type = "gguf_models"
    if model_type not in valid:
        model_type = "text_models"

    # GGUF entries must persist source_repo so Re-download has a target (flat GGUF
    # files keep no repo info on disk). Plain HF repos re-download by id, so a bare
    # path string is enough and surfaces as a missing HF model.
    if is_gguf:
        entry = {"path": model_id, "source_repo": source_repo}
    else:
        entry = model_id

    # Dedupe across all categories by path / basename so we don't double-add.
    fname = _os.path.basename(model_id) if ("/" in model_id or _os.sep in model_id) else model_id
    for cat in valid:
        for m in config_manager.models_data.get(cat, []):
            key = m if isinstance(m, str) else (m.get("path") or m.get("id") or "")
            if key == model_id or (fname and _os.path.basename(key) == fname):
                return {"success": True, "already": True}

    config_manager.models_data.setdefault(model_type, []).append(entry)
    _prune_to_download(model_id)
    config_manager.save_models()
    _broker_notify_models_updated(request)
    return {"success": True}


@router.post("/admin/api/model-mark-download", summary="List a model for later download")
async def api_model_mark_download(request: Request, username: str = Depends(require_admin)):
    """Record a model in the 'to download' wishlist: it appears in the model list
    as a non-configured, to-be-downloaded entry (no files fetched, no serving
    config created). Used by 'Free disk' on unconfigured models, 'Remove' on a
    model with no files left, and 'Add to list' in the download window."""
    if config_manager is None:
        raise HTTPException(status_code=503, detail="Config manager not initialized")
    data = await request.json()
    model_id = (data.get("model_id") or data.get("path") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    source_repo = (data.get("source_repo") or model_id).strip()
    model_type = (data.get("model_type") or "").strip()
    is_gguf = (bool(data.get("is_gguf")) or model_type == "gguf_models"
               or model_id.lower().endswith(".gguf") or "gguf" in model_id.lower())
    if is_gguf:
        model_type = "gguf_models"
    if model_type not in _VALID_MODEL_CATS:
        model_type = "text_models"
    # Already a real (configured) model — nothing to add.
    if _is_model_configured(model_id):
        return {"success": True, "already_configured": True}
    import os as _os
    lst = config_manager.models_data.setdefault("to_download", [])
    fname = _basename_key(model_id)
    for e in lst:
        k = _entry_key(e)
        if k == model_id or (fname and _basename_key(k) == fname):
            return {"success": True, "already": True}
    lst.append({"path": model_id, "source_repo": source_repo,
                "model_type": model_type, "is_gguf": is_gguf})
    config_manager.save_models()
    _broker_notify_models_updated(request)
    return {"success": True}


@router.post("/admin/api/model-unmark-download", summary="Remove a model from the download list")
async def api_model_unmark_download(request: Request, username: str = Depends(require_admin)):
    """Drop a model from the 'to download' wishlist (the user no longer wants it
    listed). Has no effect on configured models or files on disk."""
    if config_manager is None:
        raise HTTPException(status_code=503, detail="Config manager not initialized")
    data = await request.json()
    model_id = (data.get("model_id") or data.get("path") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    if _prune_to_download(model_id):
        config_manager.save_models()
        _broker_notify_models_updated(request)
    return {"success": True}


@router.post("/admin/api/model-enable", summary="Enable a model")
async def api_model_enable(request: Request, username: str = Depends(require_admin)):
    """Register a cached model in models.json so CoderAI can use it."""
    if config_manager is None:
        raise HTTPException(status_code=503, detail="Config manager not initialized")
    data = await request.json()
    path = data.get("path") or data.get("model_id", "")
    model_type = data.get("model_type", "text_models")
    valid = {"text_models", "image_models", "audio_models", "gguf_models", "tts_models", "vision_models",
             "video_models", "audio_gen_models", "embedding_models", "spatial_models"}
    if model_type not in valid:
        raise HTTPException(status_code=400, detail=f"model_type must be one of {valid}")
    lst = config_manager.models_data.setdefault(model_type, [])
    changed = False
    if path not in lst:
        lst.append(path)
        changed = True
    if _prune_to_download(path):
        changed = True
    if changed:
        config_manager.save_models()
    _broker_notify_models_updated(request)
    return {"success": True}


@router.post("/admin/api/model-disable", summary="Disable a model")
async def api_model_disable(request: Request, username: str = Depends(require_admin)):
    """Remove a model from models.json (keeps it cached locally)."""
    if config_manager is None:
        raise HTTPException(status_code=503, detail="Config manager not initialized")
    import os as _os
    data = await request.json()
    path = data.get("path") or data.get("model_id", "")
    config_id = (data.get("config_id") or "").strip()
    # Also match by bare filename so entries stored without full path are caught
    fname = _os.path.basename(path) if (_os.sep in path or "/" in path) else ""

    def _matches(m_entry) -> bool:
        if isinstance(m_entry, str):
            return m_entry == path or (fname and _os.path.basename(m_entry) == fname)
        if config_id:
            # Targeted removal: only remove the entry with this config_id
            return m_entry.get("config_id", "") == config_id
        # A whisper-server entry has path=None (its file is under model_path); guard
        # against None so basename() doesn't blow up the whole disable request.
        key = m_entry.get("path") or m_entry.get("id") or ""
        return key == path or (fname and bool(key) and _os.path.basename(key) == fname)

    # Strict 1:1: a whisper gguf MODEL config owns the runner whose id == the
    # config's alias. Pre-collect the alias(es) of the model config(s) being
    # removed so the matching runner(s) come down with them.
    removed_runner_ids = set()
    for _cat in ("audio_models",):
        for _m in config_manager.models_data.get(_cat, []):
            if not isinstance(_m, dict) or _is_whisper_runner(_m):
                continue
            if _m.get("backend") != "whisper-server":
                continue
            if _matches(_m):
                _a = (_m.get("alias") or "").strip()
                if _a:
                    removed_runner_ids.add(_a)

    def _is_runner_of_removed_model(m_entry) -> bool:
        # Single-config removal (config_id) or whole-model removal (by path):
        # drop the runner(s) whose id matches a removed config's alias.
        if not _is_whisper_runner(m_entry):
            return False
        if m_entry.get("id") in removed_runner_ids:
            return True
        if config_id:
            return False  # targeted: only the alias-linked runner
        mp = m_entry.get("model_path") or ""
        return mp == path or (fname and bool(mp) and _os.path.basename(mp) == fname)

    removed = []
    changed = False
    for cat in ("text_models", "image_models", "audio_models",
                "gguf_models", "tts_models", "vision_models", "video_models",
                "audio_gen_models", "embedding_models", "spatial_models"):
        lst = config_manager.models_data.get(cat, [])
        keep = []
        for m in lst:
            if _matches(m) or _is_runner_of_removed_model(m):
                removed.append(m)
            else:
                keep.append(m)
        if len(keep) != len(lst):
            config_manager.models_data[cat] = keep
            changed = True
    if changed:
        config_manager.save_models()

    # Kill the subprocess + drop the registry entries for every whisper-server
    # runner we just removed, so the server doesn't linger until a restart.
    try:
        from codai.models.manager import multi_model_manager as _mmm
        for m in removed:
            if isinstance(m, dict) and m.get("backend") == "whisper-server":
                mid = m.get("id")
                if not mid:
                    continue
                # Stop the subprocess + clear VRAM accounting, then forget the runner
                # entirely (its config is gone, unlike a plain unload).
                _mmm.stop_whisper_server(mid)
                _mmm.whisper_servers.pop(mid, None)
    except Exception as e:
        print(f"  [admin] whisper runner teardown failed: {e}")

    _broker_notify_models_updated(request)
    return {"success": True}


@router.get("/admin/api/quantize-capabilities", summary="GPTQ/AWQ quantization availability")
async def api_quantize_capabilities(username: str = Depends(require_admin)):
    """Report whether fast-kernel (GPTQ/AWQ) quantization is available + any jobs."""
    from codai.models import quant
    return {
        "capabilities": quant.capabilities(),
        "available": quant.is_available(),
        "jobs": quant.all_jobs(),
    }


@router.post("/admin/api/model-quantize", summary="Quantize a model to fast-kernel 4-bit")
async def api_model_quantize(request: Request, username: str = Depends(require_admin)):
    """Start (or report) an on-demand background GPTQ/AWQ quantization.

    Body: {path|model_id, method?(gptq|awq), bits?(4), group_size?(128)}.
    Quantization is heavy and slow; it runs in the background and the produced
    checkpoint is picked up automatically on the model's next load. Falls back to
    bitsandbytes if the fast kernels are unavailable or the arch is unsupported.
    """
    from codai.models import quant
    data = await request.json()
    model_id = (data.get("path") or data.get("model_id") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="path/model_id is required")
    method = (data.get("method") or "gptq").lower()
    if method not in ("gptq", "awq"):
        raise HTTPException(status_code=400, detail="method must be 'gptq' or 'awq'")
    try:
        bits = int(data.get("bits", 4))
        group_size = int(data.get("group_size", 128))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="bits/group_size must be integers")
    job = quant.start_quantization(model_id, method=method, bits=bits, group_size=group_size)
    return {"success": job.get("status") != "unavailable", "job": job}


@router.get("/admin/api/quantize-status", summary="Quantization job status")
async def api_quantize_status(model_id: str = "", username: str = Depends(require_admin)):
    """Status for one model's quant job (?model_id=...), or all jobs."""
    from codai.models import quant
    if model_id:
        return {"job": quant.get_job(model_id.strip())}
    return {"jobs": quant.all_jobs()}


@router.get("/admin/api/model-loaded-status", summary="Model load status")
async def api_model_loaded_status(username: str = Depends(require_admin)):
    """Return loaded model keys with per-model instance pool info."""
    from codai.models.manager import multi_model_manager
    loaded = list(multi_model_manager.models.keys())

    # Whisper-server models run as their own subprocess (not in .models). Surface
    # each running server under both its id and its `audio:` alias so the models
    # page (which checks `audio:<id>` and `<id>`) shows it as loaded.
    for mid, wsm in multi_model_manager.whisper_servers.items():
        try:
            running = wsm.is_running()
        except Exception:
            running = False
        if running:
            loaded.append(mid)
            loaded.append(f"audio:{mid}")
            mp = getattr(wsm, "_model_path", None)
            if mp:
                loaded.append(mp)

    instance_pools = {}
    for key, pool in multi_model_manager.model_pools.items():
        instance_pools[key] = {"loaded": pool.count, "max": pool.max_instances}

    configured_max = {}
    if config_manager:
        for cat in ("text_models", "image_models", "audio_models", "vision_models",
                    "tts_models", "gguf_models", "video_models", "audio_gen_models",
                    "embedding_models", "spatial_models"):
            for m in config_manager.models_data.get(cat, []):
                if isinstance(m, dict):
                    path = m.get("path") or m.get("id") or ""
                    max_inst = m.get("max_instances", 1)
                    if path and max_inst and int(max_inst) > 1:
                        configured_max[path] = int(max_inst)

    return {"loaded": loaded, "instances": instance_pools, "configured_max": configured_max}


@router.post("/admin/api/model-load", summary="Load a model into memory")
async def api_model_load(request: Request, username: str = Depends(require_admin)):
    """Load a configured model into VRAM (same VRAM checks as a real request)."""
    from codai.models.manager import multi_model_manager
    data = await request.json()
    path = data.get("path", "")
    if not path:
        raise HTTPException(status_code=400, detail="path required")

    # A whisper-server runner: starting it IS the model load (subprocess onto the
    # GPU). Route through the accounted start so it evicts for VRAM and registers
    # as a loaded model (and "Unload" later frees it).
    for _mid in (path, path.split("audio:")[-1]):
        if _mid in multi_model_manager.whisper_servers:
            ok = await asyncio.to_thread(multi_model_manager.start_whisper_server, _mid)
            if not ok:
                raise HTTPException(status_code=500, detail="whisper-server failed to start")
            return {"success": True, "model_key": f"audio:{_mid}"}

    # Find the model config entry to determine its type. A model may be
    # registered in several categories (e.g. a vision LLM advertises image_to_text
    # → also listed under vision_models). The category-bucket loop below would pick
    # whichever non-text bucket it hits first, sending the model to the diffusers /
    # transformers loader — which calls from_pretrained() and fails on a GGUF file.
    # So the entry's DECLARED primary model_type wins, and a GGUF/llama.cpp text
    # model always loads via the text path regardless of its other buckets.
    model_type = "text"
    model_cfg: dict = {}
    if config_manager:
        md = config_manager.models_data
        for cat, mtype in (("image_models", "image"), ("audio_models", "audio"),
                           ("vision_models", "vision"), ("tts_models", "tts"),
                           ("video_models", "video"),
                           ("audio_gen_models", "audio_gen"),
                           ("embedding_models", "embedding"),
                           ("spatial_models", "spatial")):
            for m in md.get(cat, []):
                mid = m if isinstance(m, str) else m.get("path") or m.get("id") or ""
                if mid == path:
                    model_type = mtype
                    model_cfg = m if isinstance(m, dict) else {}
                    break
        # Respect the entry's declared primary type; a text/gguf model (or any
        # .gguf path) is a llama.cpp model and must use the text loader even when
        # it's also bucketed under image/vision for capability routing.
        _primary = (model_cfg.get("model_type") if isinstance(model_cfg, dict) else "") or ""
        if _primary in ("text_models", "gguf_models") or str(path).lower().endswith(".gguf"):
            model_type = "text"

    # Offload to a thread: request_model may block (thermal wait / busy model /
    # actual load) and would otherwise freeze the whole admin web UI event loop.
    result = await asyncio.to_thread(
        multi_model_manager.request_model,
        path, model_type if model_type != "text" else None)
    if result.get("already_loaded"):
        return {"success": True, "already_loaded": True}

    # Proactive VRAM check: evict loaded models if free VRAM is insufficient.
    # request_model handles this for "on-request" load mode; cover remaining modes here.
    # HuggingFace repo IDs return needed_gb==0 (size unknown); handle that case too.
    model_key = result.get("model_key") or path
    needed_gb = multi_model_manager._get_model_used_vram_gb(model_key, path)
    free_gb = multi_model_manager._get_free_vram_gb()
    if needed_gb > 0 and free_gb < needed_gb:
        print(f"Admin model-load: need {needed_gb:.1f} GB VRAM, have {free_gb:.1f} GB free — evicting models")
        await asyncio.to_thread(multi_model_manager._evict_models_for_vram, needed_gb)
    elif needed_gb == 0 and multi_model_manager.models and free_gb < 4.0:
        # Unknown model size but VRAM nearly full — evict everything to avoid OOM on first attempt
        print(f"Admin model-load: unknown model size, only {free_gb:.1f} GB free — evicting models proactively")
        await asyncio.to_thread(multi_model_manager.unload_all_models)

    # Not loaded yet — trigger actual load
    try:
        if model_type == "text":
            # In a thread: the GGUF/llama load is heavy and would block the admin
            # event loop (freezing the whole web UI) if run inline.
            # _load_model_by_name already records the VRAM delta internally.
            mm = await asyncio.to_thread(
                multi_model_manager._load_model_by_name, result["model_name"] or path)
            if mm is None:
                raise RuntimeError("Model failed to load")
            multi_model_manager.models[result["model_key"] or path] = mm
            multi_model_manager.active_in_vram = result["model_key"] or path
        elif model_type == "audio":
            wsm = multi_model_manager.whisper_servers.get(path)
            if wsm is not None:
                _snap = multi_model_manager.vram_before_load()
                started = wsm.start(getattr(wsm, "_model_path", None), gpu_device=getattr(wsm, "_gpu_device", 0))
                if not wsm.is_running():
                    raise RuntimeError("whisper-server failed to start")
                model_key = f"audio:{path}"
                multi_model_manager.models[model_key] = wsm
                multi_model_manager.active_in_vram = model_key
                multi_model_manager.models_in_vram.add(model_key)
                multi_model_manager.record_vram_delta(model_key, _snap)
                return {"success": True, "already_loaded": False, "started_model": started}
        elif model_type == "image":
            from codai.api.images import _load_diffusers_pipeline, _is_gguf_model, _load_sdcpp_model
            from codai.api.state import get_global_args
            global_args = get_global_args()
            model_key = f"image:{path}"
            _snap = multi_model_manager.vram_before_load()
            if _is_gguf_model(path):
                resolved = await asyncio.to_thread(multi_model_manager.load_model, path)
                import os as _os
                if resolved and _os.path.isfile(resolved):
                    sd_model = await asyncio.to_thread(_load_sdcpp_model, resolved, global_args)
                    if sd_model:
                        multi_model_manager.add_model(model_key, sd_model)
                        multi_model_manager.record_vram_delta(model_key, _snap)
            else:
                pipeline = await asyncio.to_thread(_load_diffusers_pipeline, path, global_args)
                if pipeline:
                    multi_model_manager.add_model(model_key, pipeline)
                    multi_model_manager.record_vram_delta(model_key, _snap)
        elif model_type == "video":
            from codai.api.video import _load_video_pipeline, _derive_device
            model_key = f"video:{path}"
            device = _derive_device()
            _snap = multi_model_manager.vram_before_load()
            _offload = model_cfg.get("offload_strategy") or None
            pipe = await asyncio.to_thread(_load_video_pipeline, path, device, "t2v", _offload, model_cfg)
            if pipe is None:
                raise RuntimeError("Video model failed to load")
            multi_model_manager.models[model_key] = pipe
            multi_model_manager.current_model_key = model_key
            multi_model_manager.active_in_vram = model_key
            multi_model_manager.models_in_vram.add(model_key)
            multi_model_manager.record_vram_delta(model_key, _snap)
        elif model_type == "audio_gen":
            from codai.api.audio_gen import _load_musicgen, _load_audioldm, _detect_audio_gen_type, _derive_device
            model_key = f"audio_gen:{path}"
            device = _derive_device()
            _snap = multi_model_manager.vram_before_load()
            gen_type = _detect_audio_gen_type(path)
            if gen_type in ("musicgen", "audiogen"):
                pipe = await asyncio.to_thread(_load_musicgen, path, device)
            else:
                pipe = await asyncio.to_thread(_load_audioldm, path, device)
            if pipe is None:
                raise RuntimeError("Audio gen model failed to load")
            multi_model_manager.models[model_key] = pipe
            multi_model_manager.current_model_key = model_key
            multi_model_manager.active_in_vram = model_key
            multi_model_manager.models_in_vram.add(model_key)
            multi_model_manager.record_vram_delta(model_key, _snap)
        elif model_type == "tts":
            model_key = f"tts:{path}"
            _snap = multi_model_manager.vram_before_load()
            # Use the same backend factory as a real request so every engine is
            # handled identically — in particular a Parler model boots its managed
            # worker here, so "loading" it from the interface starts the service.
            cfg = (multi_model_manager.config.get(model_key)
                   or multi_model_manager.config.get(f"tts:{path}")
                   or model_cfg or {})
            def _load_tts():
                from codai.api import tts_backends
                return tts_backends.load_backend(path, path, cfg)
            tts_obj = await asyncio.to_thread(_load_tts)
            if tts_obj is None:
                raise RuntimeError("TTS model failed to load")
            multi_model_manager.models[model_key] = tts_obj
            multi_model_manager.current_model_key = model_key
            multi_model_manager.active_in_vram = model_key
            multi_model_manager.models_in_vram.add(model_key)
            multi_model_manager.record_vram_delta(model_key, _snap)
        elif model_type in ("embedding", "spatial", "vision"):
            from codai.api.images import _load_diffusers_pipeline
            from codai.api.state import get_global_args
            model_key = f"{model_type}:{path}"
            _snap = multi_model_manager.vram_before_load()
            pipeline = await asyncio.to_thread(_load_diffusers_pipeline, path, get_global_args())
            if pipeline is None:
                raise RuntimeError(f"{model_type} model failed to load")
            multi_model_manager.add_model(model_key, pipeline)
            multi_model_manager.active_in_vram = model_key
            multi_model_manager.models_in_vram.add(model_key)
            multi_model_manager.record_vram_delta(model_key, _snap)
        return {"success": True, "already_loaded": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/api/model-unload", summary="Unload a model")
async def api_model_unload(request: Request, username: str = Depends(require_admin)):
    """Unload a model from VRAM (keeps it available for on-request reload)."""
    from codai.models.manager import multi_model_manager
    data = await request.json()
    path = data.get("path", "")
    if not path:
        raise HTTPException(status_code=400, detail="path required")

    def _matches(k: str) -> bool:
        return bool(k) and (k == path or k.endswith(f":{path}")
                            or k.endswith(path.split("/")[-1]))

    # A whisper-server model runs as its own subprocess (tracked in whisper_servers),
    # but it's ALSO registered in the generic .models/.model_pools registry under
    # `audio:<id>`. Stopping only the subprocess leaves that registry entry behind,
    # so the UI keeps showing it as loaded. Match by id, `audio:<id>` or the gguf
    # model_path (so unloading the file stops every server using it), stop the
    # server, and drop the registry entries so the loaded-state flips to "Load".
    stopped_whisper = False
    for mid in list(multi_model_manager.whisper_servers.keys()):
        wsm = multi_model_manager.whisper_servers.get(mid)
        if wsm is None:
            continue
        mp = getattr(wsm, "_model_path", None) or ""
        if _matches(mid) or _matches(f"audio:{mid}") or _matches(mp):
            # Stops the subprocess AND clears its VRAM accounting (frees VRAM).
            multi_model_manager.stop_whisper_server(mid)
            stopped_whisper = True
    if stopped_whisper:
        return {"success": True, "was_loaded": True}

    # Find the key across BOTH the single-model cache and the instance pools — a
    # model served with max_instances>1 lives only in model_pools, so searching
    # .models alone would miss it and leave the pooled instances in VRAM.
    key = None
    for k in list(multi_model_manager.models.keys()) + list(multi_model_manager.model_pools.keys()):
        if _matches(k):
            key = k
            break
    if key is None:
        return {"success": True, "was_loaded": False}

    # unload_model() drops the cache entry AND cleans up the whole instance pool,
    # then runs gc + torch.cuda.empty_cache() — the manual pop above froze pooled
    # instances' VRAM. Offload to a thread: it may briefly wait for an in-flight
    # request to finish and would otherwise block the admin event loop.
    was = await asyncio.to_thread(multi_model_manager.unload_model, key)
    return {"success": True, "was_loaded": bool(was)}


def _sanitize_engine_int_overrides(raw) -> dict:
    """Clean a {engine_name: int} override map: keep positive ints, drop the rest."""
    out = {}
    if isinstance(raw, dict):
        for name, val in raw.items():
            if val in (None, ""):
                continue
            try:
                iv = int(val)
            except (TypeError, ValueError):
                continue
            if iv >= 1:
                out[str(name)] = iv
    return out


def _resolve_engine_spec(engine_name: str, engine_specs):
    """Find the declared engine matching ``engine_name`` (by name or backend)."""
    for s in (engine_specs or []):
        if not isinstance(s, dict):
            continue
        if (s.get("name") or "").lower() == engine_name.lower() \
                or (s.get("backend") or "").lower() == engine_name.lower():
            return s
    return None


def validate_engine_pin(engine_name: str, model_path: str, engine_specs,
                        model_backend: str = None, ds4_cfg=None) -> list:
    """Return human-readable warnings if pinning ``model_path`` to ``engine_name``
    is wrong (unknown engine, or an engine that can't run this model's format).

    Empty list = the pin is fine. Used to *notify* the admin instead of silently
    ignoring a bad pin (the router would otherwise just fall back)."""
    engine_name = (engine_name or "").strip()
    if not engine_name:
        return []
    from codai.frontproxy.registry import _DEFAULT_CAPS
    from codai.frontproxy.router import required_capability
    specs = engine_specs or []
    if specs:
        spec = _resolve_engine_spec(engine_name, specs)
        if spec is None:
            names = [s.get("name") for s in specs if isinstance(s, dict) and s.get("name")]
            return [f"Engine '{engine_name}' is not declared. Known engines: "
                    f"{', '.join(names) or '(none)'}."]
        backend = (spec.get("backend") or "auto").lower()
        caps = set(spec.get("capabilities")
                   or _DEFAULT_CAPS.get(backend, {"transformers", "gguf"}))
    else:
        # Auto-detection: no engine_specs to resolve against — infer the engine's
        # capabilities from its vendor/backend name so we can still catch an
        # impossible pin (e.g. a transformers model pinned to the Radeon engine).
        key = engine_name.lower()
        backend = {"radeon": "vulkan", "amd": "vulkan", "intel": "vulkan",
                   "cuda": "nvidia"}.get(key, key)
        caps = _DEFAULT_CAPS.get(backend)
        if caps is None:
            return []   # unknown name, nothing to validate against — accept silently
        caps = set(caps)
    req = required_capability(
        model_path, backend=model_backend,
        ds4_model_id=getattr(ds4_cfg, "model_id", None) if ds4_cfg else None,
        ds4_enabled=bool(getattr(ds4_cfg, "enabled", False)) if ds4_cfg else False)
    if req and req not in caps:
        return [f"Engine '{engine_name}' (backend '{backend}') can't run this model: "
                f"it needs '{req}' capability but the engine only provides "
                f"{sorted(caps)}. The request would fall back to a compatible engine — "
                f"pick a different engine or adjust the engine's capabilities."]
    return []


@router.post("/admin/api/model-configure", summary="Update a model's configuration")
async def api_model_configure(request: Request, username: str = Depends(require_admin)):
    """Save per-model configuration and register/update in models.json."""
    if config_manager is None:
        raise HTTPException(status_code=503, detail="Config manager not initialized")
    data = await request.json()
    if data.get("backend") == "whisper-server":
        model_id = (data.get("model_id") or "").strip()
        if not model_id:
            model_id = _next_whisper_server_model_id(config_manager.models_data.get("audio_models", []))
        server_path = (data.get("server_path") or "").strip()
        if not server_path:
            server_path = _default_whisper_server_path()
        model_source = (data.get("model_source") or "cached-gguf").strip() or "cached-gguf"
        if model_source not in {"cached-gguf", "manual-path"}:
            raise HTTPException(status_code=400, detail="model_source must be one of: cached-gguf, manual-path")
        model_path = (data.get("model_path") or "").strip()
        if not model_path:
            raise HTTPException(status_code=400, detail=f"model_path is required for {model_source}")
        port = int(data.get("port", 8744))
        if port < 1 or port > 65535:
            raise HTTPException(status_code=400, detail="port must be between 1 and 65535")
        gpu_device = int(data.get("gpu_device", 0))
        if gpu_device < 0:
            raise HTTPException(status_code=400, detail="gpu_device must be >= 0")
        alias = (data.get("alias") or "").strip() or None
        # Fields the whisper form manages. Anything NOT here (e.g. `engine`,
        # `config_id`) is preserved when editing an existing entry.
        fields = {
            "id": model_id,
            "backend": "whisper-server",
            "server_path": server_path,
            "model_path": model_path,
            "port": port,
            "gpu_device": gpu_device,
            "load_mode": data.get("load_mode", "on-request"),
            "model_type": "audio_models",
            "model_types": ["audio_models"],
        }
        if alias:
            fields["alias"] = alias
        if data.get("used_vram_gb") is not None:
            fields["used_vram_gb"] = data["used_vram_gb"]
        audio_list = config_manager.models_data.setdefault("audio_models", [])
        # Update in place when the id already exists (this is an edit); only append
        # for a genuinely new id. Otherwise editing an existing whisper-server model
        # would either 409 or silently create a duplicate config.
        existing = next(
            (m for m in audio_list
             if isinstance(m, dict) and m.get("backend") == "whisper-server"
             and m.get("id") == model_id),
            None,
        )
        if existing is not None:
            if not alias:
                existing.pop("alias", None)
            existing.update(fields)
        else:
            audio_list.append(fields)
        config_manager.save_models()
        result = {"success": True, "model_id": model_id, "model_path": model_path, "server_path": server_path}
        if alias:
            result["alias"] = alias
        return result
    path = data.get("path") or data.get("model_id", "")
    valid = {"text_models", "image_models", "audio_models", "tts_models", "vision_models", "video_models",
             "audio_gen_models", "embedding_models", "spatial_models"}
    if not path:
        raise HTTPException(status_code=400, detail="path is required")

    # Accept model_types (list) or fall back to single model_type
    raw_types = data.get("model_types") or []
    if not raw_types:
        raw_types = [data.get("model_type", "text_models")]
    # Normalize: gguf_models → text_models, deduplicate, filter valid
    model_types = list(dict.fromkeys(
        ("text_models" if t == "gguf_models" else t)
        for t in raw_types if t
    ))
    model_types = [t for t in model_types if t in valid]
    if not model_types:
        model_types = ["text_models"]

    # config_id: when provided, identifies a specific config entry to update.
    # A new UUID is assigned if none is given (new entry).
    config_id = (data.get("config_id") or "").strip()
    is_new_config = not config_id
    if is_new_config:
        config_id = str(_uuid.uuid4())

    # Remove from all categories.
    # When config_id matches an existing entry, remove only that entry so that
    # sibling configs (same path, different config_id) are preserved.
    # Fall back to path-based removal for entries that predate config_id support.
    import os as _os
    paths_to_remove = {path}
    orig_path = (data.get("orig_path") or "").strip()
    if orig_path and orig_path != path:
        paths_to_remove.add(orig_path)
    fnames_to_remove = {_os.path.basename(p) for p in paths_to_remove if _os.sep in p or "/" in p}

    def _should_remove(m_entry) -> bool:
        if not isinstance(m_entry, dict):
            # Legacy string entry — fall back to path matching
            return m_entry in paths_to_remove
        existing_cid = m_entry.get("config_id", "")
        if existing_cid and not is_new_config:
            # Targeted removal: only the entry that shares this config_id
            return existing_cid == config_id
        if existing_cid and is_new_config:
            # Adding a new configuration for the same model must preserve modern
            # sibling configs. Only legacy entries without config_id fall through
            # to path-based replacement because they cannot be targeted safely.
            return False
        # Path-based removal (no config_id on either side, or new entry replacing old)
        key = m_entry.get("path", m_entry.get("id", ""))
        return key in paths_to_remove or (fnames_to_remove and _os.path.basename(key) in fnames_to_remove)

    for cat in valid | {"gguf_models"}:
        lst = config_manager.models_data.get(cat, [])
        config_manager.models_data[cat] = [m for m in lst if not _should_remove(m)]

    # Auto-estimate used_vram_gb from file size if not provided
    used_vram_gb = data.get("used_vram_gb")
    if used_vram_gb is None:
        import os
        from codai.models.cache import is_huggingface_model_id
        if os.path.isfile(path):
            size_bytes = os.path.getsize(path)
            multiplier = 1.1 if path.endswith(".gguf") else 1.2
            used_vram_gb = round(size_bytes / 1e9 * multiplier, 2)
        elif is_huggingface_model_id(path):
            from codai.models.manager import MultiModelManager
            size_bytes = MultiModelManager._hf_cached_model_size_bytes(path)
            if size_bytes > 0:
                used_vram_gb = round(size_bytes / 1e9 * 1.2, 2)

    # Build settings entry
    entry: dict = {"path": path, "model_type": model_types[0], "model_types": model_types, "config_id": config_id}
    if used_vram_gb is not None:
        entry["used_vram_gb"] = used_vram_gb
    # Store video sub-types (t2v / i2v / v2v) when present
    if data.get("video_subtypes"):
        entry["video_subtypes"] = data["video_subtypes"]
    for key in ("alias", "config_name", "backend", "load_mode", "n_gpu_layers", "n_ctx",
                "max_gpu_percent", "manual_ram_gb", "load_in_4bit", "load_in_8bit",
                "flash_attention", "no_ram", "offload_strategy", "offload_dir",
                "system_prompt", "parser", "tools_closer_prompt", "grammar_guided",
                "max_instances", "preload_all_instances", "capabilities",
                "model_template", "vae_path", "t5xxl_path", "clip_l_path",
                "clip_g_path", "clip_vision_path", "lora_path", "lora_model_dir",
                "lora_train_base_model",
                "max_vram", "sdcpp_flash_attn", "sdcpp_diffusion_flash_attn", "vae_tiling",
                "component_quantization", "output_crf", "force_vram_update",
                "balanced_gpu_percent", "acceleration",
                "cache_type_k", "cache_type_v", "kv_offload", "n_batch", "n_ubatch", "n_seq_max",
                "gpu_split", "tensor_split", "split_strategy", "split_secondary_cap_gb",
                "turboquant", "engine", "engine_fallback",
                "quant_backend", "kv_cache_budget_mb", "kv_cache_slots", "mmproj",
                "auto_compact", "auto_compact_pct", "auto_compact_strategy",
                "auto_compact_model", "suppress_reasoning"):
        if key in data:
            entry[key] = data[key]

    # Per-model ds4 launch overrides (only meaningful for a deepseek4 model served
    # via the ds4 engine). Normalize to a small dict and drop it when empty, so the
    # entry stays clean and inherits the global ds4 config as the default.
    if "ds4" in data:
        src = data.get("ds4") if isinstance(data.get("ds4"), dict) else {}
        ds4o = {}
        sv = src.get("ssd_streaming")
        if sv is not None and sv != "":
            ds4o["ssd_streaming"] = (sv.lower() in ("1", "true", "on", "yes")
                                     if isinstance(sv, str) else bool(sv))
        rg = src.get("expert_cache_reserve_gb")
        if rg not in (None, "", 0, "0"):
            try:
                ds4o["expert_cache_reserve_gb"] = max(0, int(rg))
            except (TypeError, ValueError):
                pass
        for k in ("extra_args", "extra_env"):
            v = src.get(k)
            if isinstance(v, str) and v.strip():
                ds4o[k] = v.strip()
        if ds4o:
            entry["ds4"] = ds4o
        else:
            entry.pop("ds4", None)

    # A GGUF LLM is served by llama.cpp. Its multimodal projector (mmproj) gives
    # it VISION INPUT, which is the `image_to_text` capability served through
    # llama.cpp — NOT the diffusers `vision_models`/`image_models` categories
    # (those route the .gguf to from_pretrained() and fail). So for a GGUF text
    # model: auto-tag image_to_text when an mmproj is set, and keep it out of the
    # diffusers buckets (it stays a llama.cpp model that advertises vision).
    _path_l = str(path).lower()
    _is_gguf_llm = ((_path_l.endswith(".gguf") or "gguf" in _path_l
                     or entry.get("model_type") == "gguf_models")
                    and entry.get("model_type") in ("text_models", "gguf_models"))
    if _is_gguf_llm:
        _caps = list(entry.get("capabilities") or [])
        if entry.get("mmproj") and "image_to_text" not in _caps:
            _caps.append("image_to_text")
            entry["capabilities"] = _caps
        _DIFFUSERS_CATS = {"image_models", "vision_models", "video_models", "spatial_models"}
        _kept = [t for t in model_types if t not in _DIFFUSERS_CATS] or ["text_models"]
        if _kept != model_types:
            model_types = _kept
            entry["model_types"] = model_types

    # A .gguf configured for speech-to-text is a whisper model — it runs via a
    # whisper-server runner subprocess. Mark the model entry's backend so the
    # engine doesn't try to load the gguf as a transformers audio model, and so it
    # stays the MODEL config on the GGUF row (its runner lives in the whisper card).
    _is_whisper_model = (str(path).lower().endswith(".gguf")
                         and "audio_models" in model_types
                         and "speech_to_text" in set(entry.get("capabilities") or []))
    if _is_whisper_model:
        entry["backend"] = "whisper-server"

    # Add entry to each selected category
    for mtype in model_types:
        config_manager.models_data.setdefault(mtype, []).append(entry)
    config_manager.save_models()

    # Keep exactly one whisper-server runner per whisper gguf model (1:1):
    # auto-create it on enable. (Disabling the model tears the runner down — see
    # api_model_disable.)
    if _is_whisper_model:
        try:
            if _sync_whisper_runner(path, entry):
                config_manager.save_models()
        except Exception as e:
            print(f"  [admin] whisper runner auto-create failed: {e}")

    # Apply to the running server immediately so config changes (e.g.
    # lora_train_base_model, vae_path, quant flags) take effect without a
    # restart. Only the live config dict is updated — loaded weights are left
    # alone; weight-level changes (quantization) still apply on next (re)load.
    applied = 0
    try:
        from codai.main import apply_model_entry_live, _CATEGORY_TYPE_PREFIX
        # Drop stale live-config keys for any path we removed (e.g. a rename),
        # so the running server doesn't keep serving the old entry's config.
        from codai.models.manager import multi_model_manager as _mmm
        stale = {p for p in paths_to_remove if p and p != path}
        if stale:
            prefixes = {pref for (_t, pref) in _CATEGORY_TYPE_PREFIX.values()}
            for sp in stale:
                for pref in prefixes:
                    _mmm.config.pop(f"{pref}{sp}", None)
        applied = apply_model_entry_live(entry, model_types)
    except Exception as e:
        print(f"  [admin] live config apply failed (restart to apply): {e}")
    warnings = []
    if entry.get("engine"):
        warnings = validate_engine_pin(
            entry["engine"], path, config_manager.config.server.engine_specs,
            model_backend=entry.get("backend"),
            ds4_cfg=getattr(config_manager.config, "ds4", None))
        for w in warnings:
            print(f"  [admin] engine-pin warning: {w}")
    return {"success": True, "applied_live": applied, "warnings": warnings}


@router.get("/admin/api/accel-presets", summary="List acceleration / distillation presets")
async def api_accel_presets(username: str = Depends(require_admin)):
    """Return the acceleration/distillation preset catalog (Lightning / Turbo /
    LCM / Hyper-SD) so the model-config UI dropdown stays in sync with the Python
    source of truth in codai/models/acceleration.py."""
    try:
        from codai.models.acceleration import ACCEL_PRESETS
        return {"presets": ACCEL_PRESETS}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/api/accel-loras", summary="List cached distill-LoRA files for acceleration")
async def api_accel_loras(username: str = Depends(require_admin)):
    """Scan the HF cache for distill / step-distillation LoRA repos and return each
    repo's ``.safetensors`` files, pre-classified into high-noise / low-noise (for
    Wan2.2's two experts) so the model-config UI can offer cascading dropdowns:
    pick the distill model first, then the high/low (or single) LoRA from it.

    A repo qualifies if its id matches a distill keyword (lightning/lightx2v/lcm/
    hyper/turbo/distill/dmd) or is referenced by an ACCEL_PRESETS entry."""
    import os
    import re as _re
    out: list = []
    try:
        from codai.models.acceleration import ACCEL_PRESETS
        from codai.models.cache import get_all_cache_dirs
        from huggingface_hub import scan_cache_dir

        # Repos named by presets are always relevant (the `repo:file` head before ':').
        preset_repos = set()
        for p in (ACCEL_PRESETS or {}).values():
            for k in ("lora", "lora_high", "lora_low"):
                ref = p.get(k)
                if ref and ":" in str(ref):
                    preset_repos.add(str(ref).split(":", 1)[0])
                elif ref:
                    preset_repos.add(str(ref))

        kw = _re.compile(r"light(ning|x2v)|lcm|hyper[-_ ]?sd|turbo|distill|dmd|seko",
                         _re.IGNORECASE)
        _hi = _re.compile(r"high[-_ ]?noise", _re.IGNORECASE)
        _lo = _re.compile(r"low[-_ ]?noise", _re.IGNORECASE)
        # A file is a distill LoRA (vs a full model's component weights like
        # vae/transformer/text_encoder) if its path names a LoRA or noise level.
        _loraname = _re.compile(r"lora|noise|light(ning|x2v)|lcm|hyper|distill|dmd",
                                _re.IGNORECASE)

        hf_dir = (get_all_cache_dirs() or {}).get("huggingface")
        info = scan_cache_dir(hf_dir) if hf_dir else scan_cache_dir()
        for repo in info.repos:
            rid = repo.repo_id
            in_preset = rid in preset_repos
            if not (in_preset or kw.search(rid)):
                continue
            # Newest revision's snapshot → relative .safetensors paths. Keep only
            # LoRA-looking files, unless the repo is a curated preset repo (then
            # trust all its safetensors). This drops full-model component weights
            # from repos that merely match a keyword (e.g. "*-Turbo" base models).
            rev = max(repo.revisions, key=lambda r: (r.last_modified or 0), default=None)
            if rev is None:
                continue
            snap = str(rev.snapshot_path)
            files = []
            for f in rev.files:
                fp = str(f.file_path)
                if not fp.endswith(".safetensors"):
                    continue
                rel = os.path.relpath(fp, snap).replace(os.sep, "/")
                if in_preset or _loraname.search(rel):
                    files.append(rel)
            if not files:
                continue
            files.sort()
            out.append({
                "repo": rid,
                "files": files,
                "high": [f for f in files if _hi.search(f)],
                "low":  [f for f in files if _lo.search(f)],
            })
        out.sort(key=lambda m: m["repo"].lower())
    except Exception as e:
        # Cache scan is best-effort; the UI falls back to the free-text fields.
        return {"models": [], "error": str(e)}
    return {"models": out}


@router.get("/admin/api/turboquant-info", summary="TurboQuant backend availability")
async def api_turboquant_info(username: str = Depends(require_admin)):
    """Report which TurboQuant embedding-quantization backends are available so
    the model-config UI can offer 'builtin' (always) and 'library' (turboquant-py
    when installed)."""
    try:
        from codai.models import turboquant as _tq
        return {
            "builtin": True,
            "library": _tq.have_library(),
            "library_package": "turboquant-py",
            "bit_widths": [8, 6, 4, 2],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Task / queue management ---

def _human_bytes(n: float) -> str:
    """Compact human-readable byte size (e.g. 45.2 MB) for download readouts."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _human_duration(seconds: float) -> str:
    """Compact h/m/s duration (e.g. 3m 12s) for download ETAs."""
    try:
        s = int(seconds)
    except (TypeError, ValueError):
        return ""
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"


@router.get("/admin/api/tasks", summary="List active and recent tasks")
def api_tasks(username: str = Depends(require_admin)):
    """Unified live view of long-running work: in-flight / recent generations
    (image, video, audio, text) from the task registry, durable LoRA training
    jobs, and queued requests waiting for a slot. The Tasks page polls this.

    SYNC handler on purpose (runs in FastAPI's threadpool, not the event loop):
    it reads disk job records, queue/registry state and thermal sensors. Keeping
    it off the event loop means the Tasks page stays responsive while a model is
    loading (the load releases the GIL during its C call)."""
    from codai.tasks import task_registry
    from codai.api.loras import list_jobs
    from codai.queue.manager import queue_manager

    tasks = []
    seen = set()

    # Training jobs are authoritative (persisted, survive restarts).
    for j in list_jobs():
        jid = j.get("job_id")
        if not jid:
            continue
        seen.add(jid)
        status = j.get("status") or "unknown"
        norm = "running" if status in ("preparing", "training", "saving") else status
        active = norm in ("queued", "running")
        tasks.append({
            "id": jid,
            "kind": "training",
            "title": j.get("name") or "",
            "model": j.get("base_model") or "",
            "status": norm,
            "step": j.get("step") or 0,
            "total": j.get("total") or 0,
            "message": j.get("message") or "",
            "started_at": j.get("started_at"),
            "active": active,
            "cancellable": active,
            "pausable": norm == "running",
            "paused": bool(task_registry.is_paused(jid)),
            "restartable": status in ("cancelled", "error", "interrupted", "done"),
        })

    # Generations + anything else from the live registry (skip training dupes).
    for t in task_registry.list():
        if t["id"] in seen or t.get("kind") == "training":
            continue
        seen.add(t["id"])
        t = dict(t)
        t["cancellable"] = bool(t.get("cancellable", True) and t.get("active", False))
        t["pausable"] = bool(t.get("pausable", True) and t.get("status") == "running")
        t["restartable"] = False
        tasks.append(t)

    # Queued requests waiting for a free slot/model (e.g. text) not shown yet.
    for w in queue_manager.list_waiting():
        rid = w.get("request_id")
        if not rid or rid in seen or rid.startswith("lora-train-"):
            continue
        seen.add(rid)
        tasks.append({
            "id": rid,
            "kind": "text" if rid.startswith("req-") else "request",
            "title": "",
            "model": w.get("model_key") or "",
            "status": "queued",
            "step": 0, "total": 0,
            "message": "waiting for a free slot",
            "started_at": w.get("enqueued_at"),
            "active": True,
            "cancellable": False,
            "restartable": False,
        })

    # Model downloads run in out-of-process workers tracked in `_download_status`,
    # not the task registry. Surface them here so the Tasks page shows downloads
    # in progress alongside generations, training and queued requests.
    for sid, d in list(_download_status.items()):
        if sid in seen:
            continue
        seen.add(sid)
        dstatus = d.get("status") or "starting"
        active = dstatus in ("starting", "downloading")
        pct = int(round(d.get("percent") or 0))
        bits = []
        fn = d.get("filename") or ""
        if fn:
            bits.append(fn)
        rate = d.get("rate") or 0
        if active and rate:
            bits.append(f"{_human_bytes(rate)}/s")
        eta = d.get("eta")
        if active and eta:
            bits.append(f"ETA {_human_duration(eta)}")
        if d.get("error"):
            bits.append(str(d["error"]))
        elif not fn and d.get("last_info"):
            bits.append(str(d["last_info"]))
        tasks.append({
            "id": sid,
            "kind": "download",
            "title": d.get("model_id") or "",
            "model": d.get("file_pattern") or "",
            "status": "running" if active else dstatus,
            "step": pct, "total": 100,
            "percent": pct,
            "message": " · ".join(bits),
            "started_at": d.get("started_at"),
            "active": active,
            "cancellable": active,
            "pausable": False,
            "restartable": False,
        })

    # GPTQ/AWQ quantization jobs run in in-process daemon threads (status persisted
    # to disk so it survives a restart). Surface them alongside downloads/training.
    try:
        from codai.models import quant as _quant
        for _name, _qj in _quant.all_jobs().items():
            if _name in seen:
                continue
            seen.add(_name)
            _qs = _qj.get("status") or "running"
            _active = _qs == "running"
            _pct = int(round((_qj.get("progress") or 0) * 100))
            _msg = _qj.get("message") or ""
            if _qj.get("error"):
                _msg = str(_qj["error"])
            tasks.append({
                "id": f"quantize:{_name}",
                "kind": "quantize",
                "title": _name,
                "model": (_qj.get("method") or "gptq").upper(),
                "status": _qs,
                "step": _pct, "total": 100, "percent": _pct,
                "message": _msg,
                "started_at": _qj.get("started"),
                "active": _active,
                "cancellable": False,
                "pausable": False,
                "restartable": _qs in ("failed", "interrupted"),
            })
    except Exception:
        pass

    # Successfully-finished work is dropped from the live list — a "done" job is
    # no longer actionable, so it shouldn't clutter the view. Terminal-but-notable
    # states (cancelled / error / interrupted) stay, so they can be inspected,
    # restarted, or removed manually.
    tasks = [t for t in tasks if t.get("status") != "done"]

    # A thermal pause is global hardware state: while cooling, every running
    # worker is blocked at its next checkpoint. Surface it on the running tasks
    # and as a top-level banner so the Tasks page shows "cooling down".
    cooling = {"active": False}
    try:
        from codai.models import thermal
        cs = thermal.get_cooldown_state()
        if cs.get("active"):
            parts = []
            if cs.get("gpu") is not None:
                parts.append(f"GPU {cs['gpu']:.0f}°C")
            if cs.get("cpu") is not None:
                parts.append(f"CPU {cs['cpu']:.0f}°C")
            waited = int(cs.get("waited") or 0)
            detail = ", ".join(parts)
            label = "Cooling down" + (f" — {detail}" if detail else "")
            if waited:
                label += f" ({waited}s)"
            cooling = {"active": True, "message": label,
                       "gpu": cs.get("gpu"), "cpu": cs.get("cpu"), "waited": waited}
            for t in tasks:
                if t.get("active") and t.get("status") == "running":
                    t["cooling"] = True
                    t["cooling_message"] = label
    except Exception:
        pass

    # Proactive CPU soft-throttle (distinct from the hard cooldown): generations
    # are being gently slowed because the CPU is in its warm band. Computed live
    # from the CPU temp, and only shown when something is actually running (idle
    # warmth isn't being throttled). Suppressed during a hard cooldown.
    soft = {"active": False}
    try:
        from codai.models import thermal as _therm
        _any_running = any(t.get("status") == "running" for t in tasks)
        ss = _therm.soft_throttle_status()
        if ss.get("active") and _any_running and not cooling.get("active"):
            _cpu = ss.get("cpu")
            _slp = ss.get("sleep") or 0
            label = "CPU soft-throttle"
            if _cpu is not None:
                label += f" — CPU {_cpu:.0f}°C"
            if _slp:
                label += f" (+{_slp:.1f}s/step)"
            soft = {"active": True, "message": label, "cpu": _cpu, "sleep": _slp}
            for t in tasks:
                if (t.get("active") and t.get("status") == "running"
                        and not t.get("cooling")):
                    t["throttling"] = True
                    t["throttle_message"] = label
    except Exception:
        pass

    # The queue-summary header must reflect ALL model activity, not just requests
    # that flow through queue_manager (text/pipelines/training). Image/video/audio
    # generations run their own paths and live only in the task registry, so derive
    # active/waiting from the unified `tasks` list; keep max_parallel from the
    # queue manager.
    queue = dict(queue_manager.get_metrics())
    queue["active"] = sum(1 for t in tasks if t.get("status") == "running")
    queue["waiting"] = sum(1 for t in tasks if t.get("status") == "queued")
    return {"tasks": tasks, "queue": queue, "thermal": cooling, "soft_throttle": soft}


def _read_vram_info() -> Optional[dict]:
    """Best-effort {used, total, gpu} in GB. CUDA via torch, else AMD/Intel sysfs."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {"used": (total - free) / 1e9, "total": total / 1e9,
                    "gpu": torch.cuda.get_device_name(0)}
    except Exception:
        pass
    try:
        import glob as _glob
        for card in sorted(_glob.glob("/sys/class/drm/card[0-9]")):
            dev = card + "/device"
            tot = dev + "/mem_info_vram_total"
            if not os.path.exists(tot):
                continue
            total_b = int(open(tot).read())
            used_b = int(open(dev + "/mem_info_vram_used").read())
            return {"used": used_b / 1e9, "total": total_b / 1e9, "gpu": ""}
    except Exception:
        pass
    return None


def _do_task_cancel(task_id: str) -> bool:
    """Cancel a task by id. Training ids route through loras.cancel_job (handles
    queued vs running + the durable job record); everything else goes through the
    in-memory task registry."""
    from codai.tasks import task_registry
    from codai.api.loras import cancel_job
    # Download workers live in their own session registry, not the task registry.
    if _cancel_download_session(task_id):
        return True
    if cancel_job(task_id):
        return True
    return task_registry.cancel(task_id)


@router.post("/admin/api/tasks/{task_id}/cancel", summary="Cancel a task")
async def api_task_cancel(task_id: str, username: str = Depends(require_admin)):
    """Cancel a queued or running task. Running generations/training stop at the
    next step boundary; queued items are dropped before they start."""
    if not _do_task_cancel(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    return {"ok": True, "task_id": task_id, "action": "cancel"}


@router.post("/admin/api/tasks/{task_id}/interrupt", summary="Interrupt a running task")
async def api_task_interrupt(task_id: str, username: str = Depends(require_admin)):
    """Alias of cancel for a running task."""
    if not _do_task_cancel(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    return {"ok": True, "task_id": task_id, "action": "interrupt"}


@router.delete("/admin/api/tasks/{task_id}", summary="Remove a finished task")
async def api_task_remove(task_id: str, username: str = Depends(require_admin)):
    """Dismiss a finished/cancelled/errored task from the Tasks view. Refuses to
    remove a task that is still active — cancel it first."""
    from codai.tasks import task_registry
    from codai.api.loras import remove_job
    # Finished/failed download session — drop it from the live status map.
    d = _download_status.get(task_id)
    if d is not None:
        if d.get("status") in ("starting", "downloading"):
            raise HTTPException(status_code=409, detail="Download is still active — cancel it first")
        _download_status.pop(task_id, None)
        _download_sessions.pop(task_id, None)
        return {"ok": True, "task_id": task_id, "removed": True}
    # Training job (durable record) first.
    if remove_job(task_id):
        return {"ok": True, "task_id": task_id, "removed": True}
    # Otherwise a live registry (generation) task — only when it's not active.
    t = task_registry.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if t.get("active"):
        raise HTTPException(status_code=409, detail="Task is still active — cancel it first")
    task_registry.remove(task_id)
    return {"ok": True, "task_id": task_id, "removed": True}


@router.post("/admin/api/tasks/{task_id}/pause", summary="Pause a running task")
async def api_task_pause(task_id: str, username: str = Depends(require_admin)):
    """Pause a running task. It suspends at the next step boundary (holding the
    model/GPU) until resumed. Works for generations and LoRA training."""
    from codai.tasks import task_registry
    if not task_registry.pause(task_id):
        raise HTTPException(status_code=404,
                            detail="Task not found or not running")
    return {"ok": True, "task_id": task_id, "action": "pause"}


@router.post("/admin/api/tasks/{task_id}/resume", summary="Resume a paused task")
async def api_task_resume(task_id: str, username: str = Depends(require_admin)):
    """Resume a task previously paused from the Tasks page."""
    from codai.tasks import task_registry
    if not task_registry.resume(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    return {"ok": True, "task_id": task_id, "action": "resume"}


@router.post("/admin/api/tasks/{task_id}/restart", summary="Restart a training task")
async def api_task_restart(task_id: str, username: str = Depends(require_admin)):
    """Restart a finished/cancelled/interrupted LoRA training job, resuming from
    its last on-disk checkpoint (only training tasks are restartable)."""
    from codai.api.loras import restart_job
    jid = restart_job(task_id)
    if not jid:
        raise HTTPException(status_code=400,
                            detail="Task is not restartable (training jobs only, and the saved request must exist)")
    return {"ok": True, "job_id": jid, "status": "queued"}


# --- System endpoints ---

@router.post("/admin/api/system/reload", summary="Reload server configuration")
async def api_reload_config(username: str = Depends(require_admin)):
    """Reload configuration from disk."""
    try:
        from fastapi import Request
        # config_mgr is stored in app state
        request = Request({})
        config = request.app.state.config_mgr.reload()
        return {
            "success": True,
            "message": "Configuration reloaded",
            "config": {
                "loaded": config.models.loaded,
                "preload": config.models.preload,
                "load_mode": config.models.default_load_mode
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from datetime import datetime


# --- Settings page ---

def _detect_gpu_cards() -> list:
    """Every physical GPU on the machine ([{key, vendor, name}]) for the per-card
    VRAM-cap settings UI. Best-effort: returns [] if detection is unavailable."""
    try:
        from codai.frontproxy.gpu_detect import gpu_cards
        return gpu_cards()
    except Exception:
        return []


def build_settings_dict(c, gpu_cards):
    """Pure ``Config`` → settings dict. Shared by the engine handler and the front
    proxy (which holds the same Config) so both serve an identical
    /admin/api/settings without the front having to round-trip to the engine."""
    return {
        "server": {
            "host": c.server.host,
            "port": c.server.port,
            "https": c.server.https,
            "https_key_path": c.server.https_key_path,
            "https_cert_path": c.server.https_cert_path,
            "queue_max_size": c.server.queue_max_size,
            "max_parallel_requests": c.server.max_parallel_requests,
            "max_parallel_requests_overrides": c.server.max_parallel_requests_overrides,
            "internal_port_base": c.server.internal_port_base,
            "default_engine": c.server.default_engine,
            # Engine names available to pick as the default (for the settings UI).
            "engine_names": [s.get("name") for s in (c.server.engine_specs or [])
                             if isinstance(s, dict) and s.get("name")],
        },
        "backend": {
            "type": c.backend.type,
            "image_backend": c.backend.image_backend,
            "audio_backend": c.backend.audio_backend,
            "tts_backend": c.backend.tts_backend,
        },
        "models": {
            "default_load_mode": c.models.default_load_mode,
            "hf_cache_dir": c.models.hf_cache_dir,
            "gguf_cache_dir": c.models.gguf_cache_dir,
            "max_model_instances": c.models.max_model_instances,
            "max_model_instances_overrides": c.models.max_model_instances_overrides,
        },
        "offload": {
            "directory": c.offload.directory,
            "strategy": c.offload.strategy,
            "max_gpu_percent": c.offload.max_gpu_percent,
            "no_ram": c.offload.no_ram,
            "load_in_4bit": c.offload.load_in_4bit,
            "load_in_8bit": c.offload.load_in_8bit,
            "manual_ram_gb": c.offload.manual_ram_gb,
            "flash_attention": c.offload.flash_attention,
            "max_ram_gb": c.offload.max_ram_gb,
            "evict_idle_on_ram": c.offload.evict_idle_on_ram,
            "ram_leak_watch": c.offload.ram_leak_watch,
            "ram_watch_poll_seconds": c.offload.ram_watch_poll_seconds,
            "ram_watch_soft_fraction": c.offload.ram_watch_soft_fraction,
            "ram_watch_cuda": c.offload.ram_watch_cuda,
            "gpu_split": c.offload.gpu_split,
            "tensor_split": c.offload.tensor_split,
            "split_strategy": c.offload.split_strategy,
            "split_secondary_cap_gb": c.offload.split_secondary_cap_gb,
            "split_card_caps_gb": c.offload.split_card_caps_gb,
            # Every physical card on the machine, so the UI can render a per-card cap
            # row (key + name) and the user can cap each independently.
            "gpu_cards": gpu_cards,
        },
        "vulkan": {
            "n_gpu_layers": c.vulkan.n_gpu_layers,
            "n_ctx": c.vulkan.n_ctx,
            "device_id": c.vulkan.device_id,
            "single_gpu": c.vulkan.single_gpu,
        },
        "archive": {
            "enabled": c.archive.enabled,
            "directory": c.archive.directory,
            "retention": c.archive.retention,
        },
        "thermal": {
            "cpu_enabled": c.thermal.cpu_enabled,
            "gpu_enabled": c.thermal.gpu_enabled,
            "cpu_high": c.thermal.cpu_high,
            "cpu_resume": c.thermal.cpu_resume,
            "gpu_high": c.thermal.gpu_high,
            "gpu_resume": c.thermal.gpu_resume,
            "gpu_overrides": c.thermal.gpu_overrides,
            "poll_seconds": c.thermal.poll_seconds,
            "soft_throttle_enabled": c.thermal.soft_throttle_enabled,
            "soft_throttle_temp": c.thermal.soft_throttle_temp,
            "soft_throttle_max_sleep": c.thermal.soft_throttle_max_sleep,
        },
        "jobs": {
            "resume_on_restart": c.jobs.resume_on_restart,
        },
        "enhance": {
            "allow_ffmpeg": c.enhance.allow_ffmpeg,
            "allow_rife_ncnn": c.enhance.allow_rife_ncnn,
        },
        "ds4": {
            "enabled": c.ds4.enabled,
            "repo_url": c.ds4.repo_url,
            "install_dir": c.ds4.install_dir,
            "build_target": c.ds4.build_target,
            "model_path": c.ds4.model_path,
            "auto_download": c.ds4.auto_download,
            "model_variant": c.ds4.model_variant,
            "model_id": c.ds4.model_id,
            "host": c.ds4.host,
            "port": c.ds4.port,
            "ctx": c.ds4.ctx,
            "ssd_streaming": c.ds4.ssd_streaming,
            "extra_args": c.ds4.extra_args,
            "expert_cache_reserve_gb": c.ds4.expert_cache_reserve_gb,
            "extra_env": c.ds4.extra_env,
            "auto_build": c.ds4.auto_build,
            "kv_cache_cleanup_enabled": c.ds4.kv_cache_cleanup_enabled,
            "kv_cache_max_age_hours": c.ds4.kv_cache_max_age_hours,
            "kv_cache_cleanup_interval_minutes": c.ds4.kv_cache_cleanup_interval_minutes,
        },
        "compaction": {
            "enabled": c.compaction.enabled,
            "pct": c.compaction.pct,
            "strategy": c.compaction.strategy,
            "model": c.compaction.model,
        },
        "broker": {
            "enabled": c.broker.enabled,
            "base_url": c.broker.base_url,
            "scope": c.broker.scope,
            "username": c.broker.username,
            "provider_id": c.broker.provider_id,
            "client_id": c.broker.client_id,
            "registration_token": c.broker.registration_token,
            "advertised_endpoint": c.broker.advertised_endpoint,
            "websocket_path": c.broker.websocket_path,
            "transport": c.broker.transport,
            "heartbeat_interval_seconds": c.broker.heartbeat_interval_seconds,
            "connect_timeout_seconds": c.broker.connect_timeout_seconds,
            "request_timeout_seconds": c.broker.request_timeout_seconds,
            "reconnect_initial_delay_seconds": c.broker.reconnect_initial_delay_seconds,
            "reconnect_max_delay_seconds": c.broker.reconnect_max_delay_seconds,
            "websocket_ping_interval": c.broker.websocket_ping_interval,
        },
        "system_prompt": c.system_prompt,
        "tools_closer_prompt": c.tools_closer_prompt,
        "grammar_guided": c.grammar_guided,
        "parser": c.parser,
        "tmp_dir": c.tmp_dir,
    }


@router.post("/admin/api/settings", summary="Update server settings")
async def api_save_settings(request: Request, username: str = Depends(require_admin)):
    """Update and persist config.json from submitted JSON. Only sections present in the payload are updated."""
    if config_manager is None or config_manager.config is None:
        raise HTTPException(status_code=503, detail="Config manager not initialized")

    data = await request.json()
    c = config_manager.config
    _settings_warnings: list = []

    if "server" in data:
        srv = data["server"]
        c.server.host = srv.get("host", c.server.host)
        c.server.port = int(srv.get("port", c.server.port))
        c.server.https = bool(srv.get("https", c.server.https))
        c.server.https_key_path = srv.get("https_key_path") or None
        c.server.https_cert_path = srv.get("https_cert_path") or None
        if "queue_max_size" in srv:
            c.server.queue_max_size = max(1, int(srv["queue_max_size"]))
            from codai.queue.manager import queue_manager
            queue_manager.max_size = c.server.queue_max_size
        if "max_parallel_requests" in srv:
            c.server.max_parallel_requests = int(srv["max_parallel_requests"])
            from codai.queue.manager import queue_manager
            queue_manager.max_parallel_requests = c.server.max_parallel_requests
        if "max_parallel_requests_overrides" in srv:
            c.server.max_parallel_requests_overrides = _sanitize_engine_int_overrides(
                srv["max_parallel_requests_overrides"])
        if "internal_port_base" in srv:
            try:
                c.server.internal_port_base = max(1, min(65535, int(srv["internal_port_base"])))
            except (TypeError, ValueError):
                pass
        if "default_engine" in srv:
            c.server.default_engine = (srv.get("default_engine") or "").strip() or None
            # Only validate against engine_specs when they're explicitly declared.
            # With auto-detection engine_specs is empty and the engines (nvidia/
            # radeon/…) are only known to the front, so don't false-warn there — the
            # front validates the name at routing time and logs if it can't honour it.
            if (c.server.default_engine and c.server.engine_specs
                    and _resolve_engine_spec(c.server.default_engine,
                                             c.server.engine_specs) is None):
                names = [s.get("name") for s in (c.server.engine_specs or [])
                         if isinstance(s, dict) and s.get("name")]
                _settings_warnings.append(
                    f"Default engine '{c.server.default_engine}' is not declared. "
                    f"Known engines: {', '.join(names) or '(none)'}.")

    if "backend" in data:
        bk = data["backend"]
        c.backend.type = bk.get("type", c.backend.type)
        c.backend.image_backend = bk.get("image_backend", c.backend.image_backend)
        c.backend.audio_backend = bk.get("audio_backend", c.backend.audio_backend)
        c.backend.tts_backend = bk.get("tts_backend", c.backend.tts_backend)

    if "models" in data:
        mdl = data["models"]
        c.models.default_load_mode = mdl.get("default_load_mode", c.models.default_load_mode)
        if "hf_cache_dir" in mdl:
            c.models.hf_cache_dir = mdl["hf_cache_dir"] or None
        if "gguf_cache_dir" in mdl:
            c.models.gguf_cache_dir = mdl["gguf_cache_dir"] or None
        if "max_model_instances" in mdl:
            try:
                c.models.max_model_instances = max(1, int(mdl["max_model_instances"]))
            except (TypeError, ValueError):
                pass
        if "max_model_instances_overrides" in mdl:
            c.models.max_model_instances_overrides = _sanitize_engine_int_overrides(
                mdl["max_model_instances_overrides"])

    if "offload" in data:
        off = data["offload"]
        c.offload.directory = off.get("directory", c.offload.directory)
        c.offload.strategy = off.get("strategy", c.offload.strategy)
        if "max_gpu_percent" in off:
            c.offload.max_gpu_percent = off["max_gpu_percent"] or None
        c.offload.no_ram = bool(off.get("no_ram", c.offload.no_ram))
        c.offload.load_in_4bit = bool(off.get("load_in_4bit", c.offload.load_in_4bit))
        c.offload.load_in_8bit = bool(off.get("load_in_8bit", c.offload.load_in_8bit))
        if "manual_ram_gb" in off:
            c.offload.manual_ram_gb = off["manual_ram_gb"] or None
        c.offload.flash_attention = bool(off.get("flash_attention", c.offload.flash_attention))
        if "max_ram_gb" in off:
            c.offload.max_ram_gb = off["max_ram_gb"] or None
        c.offload.evict_idle_on_ram = bool(off.get("evict_idle_on_ram", c.offload.evict_idle_on_ram))
        c.offload.ram_leak_watch = bool(off.get("ram_leak_watch", c.offload.ram_leak_watch))
        if "ram_watch_poll_seconds" in off:
            c.offload.ram_watch_poll_seconds = float(off["ram_watch_poll_seconds"] or c.offload.ram_watch_poll_seconds)
        if "ram_watch_soft_fraction" in off:
            c.offload.ram_watch_soft_fraction = float(off["ram_watch_soft_fraction"] or c.offload.ram_watch_soft_fraction)
        c.offload.ram_watch_cuda = bool(off.get("ram_watch_cuda", c.offload.ram_watch_cuda))
        c.offload.gpu_split = bool(off.get("gpu_split", c.offload.gpu_split))
        if "tensor_split" in off:
            c.offload.tensor_split = off["tensor_split"] or None
        if "split_strategy" in off:
            c.offload.split_strategy = off["split_strategy"] or "vram"
        if "split_secondary_cap_gb" in off:
            _v = off["split_secondary_cap_gb"]
            try:
                c.offload.split_secondary_cap_gb = float(_v) if _v not in (None, "", 0, "0") else None
            except (TypeError, ValueError):
                c.offload.split_secondary_cap_gb = None
        if "split_card_caps_gb" in off:
            # {card_key: gb}. Drop blank/zero entries so an unset card means uncapped.
            _raw = off["split_card_caps_gb"] or {}
            _caps = {}
            if isinstance(_raw, dict):
                for _k, _v in _raw.items():
                    try:
                        if _v not in (None, "", 0, "0"):
                            _caps[str(_k)] = float(_v)
                    except (TypeError, ValueError):
                        pass
            c.offload.split_card_caps_gb = _caps
        # Push the RAM-cap settings to live global_args so the watcher, per-load
        # budget clamp and eviction honour them without a restart.
        try:
            from codai.api.state import get_global_args
            ga = get_global_args()
            if ga is not None:
                ga.max_ram_gb = c.offload.max_ram_gb
                ga.evict_idle_on_ram = c.offload.evict_idle_on_ram
                ga.ram_leak_watch = c.offload.ram_leak_watch
                ga.ram_watch_poll_seconds = c.offload.ram_watch_poll_seconds
                ga.ram_watch_soft_fraction = c.offload.ram_watch_soft_fraction
                ga.ram_watch_cuda = c.offload.ram_watch_cuda
                ga.gpu_split = c.offload.gpu_split
                ga.tensor_split = c.offload.tensor_split
                ga.split_strategy = c.offload.split_strategy
                ga.split_secondary_cap_gb = c.offload.split_secondary_cap_gb
                ga.split_card_caps_gb = c.offload.split_card_caps_gb
        except Exception:
            pass

    if "vulkan" in data:
        vk = data["vulkan"]
        c.vulkan.n_gpu_layers = int(vk.get("n_gpu_layers", c.vulkan.n_gpu_layers))
        c.vulkan.n_ctx = int(vk.get("n_ctx", c.vulkan.n_ctx))
        c.vulkan.device_id = int(vk.get("device_id", c.vulkan.device_id))
        c.vulkan.single_gpu = bool(vk.get("single_gpu", c.vulkan.single_gpu))

    if "system_prompt" in data:
        c.system_prompt = data["system_prompt"] or None
    if "tools_closer_prompt" in data:
        c.tools_closer_prompt = bool(data["tools_closer_prompt"])
    if "grammar_guided" in data:
        c.grammar_guided = bool(data["grammar_guided"])
    if "parser" in data:
        c.parser = data["parser"]
    if "tmp_dir" in data:
        # Persisted now and applied live so it takes effect without a restart.
        c.tmp_dir = (data["tmp_dir"] or "").strip() or None
        if c.tmp_dir:
            try:
                import tempfile as _tf, os as _os
                _td = _os.path.abspath(_os.path.expanduser(c.tmp_dir))
                _os.makedirs(_td, exist_ok=True)
                _tf.tempdir = _td
                _os.environ["TMPDIR"] = _td
                _os.environ["TMP"] = _td
                _os.environ["TEMP"] = _td
            except Exception:
                pass

    if "archive" in data:
        import os as _os
        from codai.api.archive import archive_manager, RETENTION_OPTIONS
        arc = data["archive"]
        c.archive.enabled = bool(arc.get("enabled", c.archive.enabled))
        raw_dir = (arc.get("directory") or "").strip()
        c.archive.directory = raw_dir  # store as-is (empty = default)
        ret = arc.get("retention", c.archive.retention)
        c.archive.retention = ret if ret in RETENTION_OPTIONS else "never"
        # Resolve for live reconfiguration
        cfg_dir = str(config_manager.config_dir)
        resolved = raw_dir if raw_dir and _os.path.isabs(raw_dir) else _os.path.join(cfg_dir, raw_dir or "archive")
        archive_manager.configure(c.archive.enabled, resolved, c.archive.retention)

    if "thermal" in data:
        th = data["thermal"]
        c.thermal.cpu_enabled = bool(th.get("cpu_enabled", c.thermal.cpu_enabled))
        c.thermal.gpu_enabled = bool(th.get("gpu_enabled", c.thermal.gpu_enabled))
        c.thermal.cpu_high = float(th.get("cpu_high", c.thermal.cpu_high))
        c.thermal.cpu_resume = float(th.get("cpu_resume", c.thermal.cpu_resume))
        c.thermal.gpu_high = float(th.get("gpu_high", c.thermal.gpu_high))
        c.thermal.gpu_resume = float(th.get("gpu_resume", c.thermal.gpu_resume))
        if "gpu_overrides" in th and isinstance(th["gpu_overrides"], dict):
            # Sanitize: {vendor: {high, resume}} with numeric values only.
            clean = {}
            for vendor, ov in th["gpu_overrides"].items():
                if not isinstance(ov, dict):
                    continue
                entry = {}
                for k in ("high", "resume"):
                    if ov.get(k) not in (None, ""):
                        try:
                            entry[k] = float(ov[k])
                        except (TypeError, ValueError):
                            pass
                if entry:
                    clean[str(vendor).lower()] = entry
            c.thermal.gpu_overrides = clean
        c.thermal.poll_seconds = max(1.0, float(th.get("poll_seconds", c.thermal.poll_seconds)))
        c.thermal.soft_throttle_enabled = bool(th.get("soft_throttle_enabled", c.thermal.soft_throttle_enabled))
        c.thermal.soft_throttle_temp = float(th.get("soft_throttle_temp", c.thermal.soft_throttle_temp))
        c.thermal.soft_throttle_max_sleep = max(0.0, float(th.get("soft_throttle_max_sleep", c.thermal.soft_throttle_max_sleep)))
        # Push to the live global_args so changes apply without a restart.
        try:
            from codai.api.state import get_global_args
            ga = get_global_args()
            if ga is not None:
                ga.thermal_cpu_enabled = c.thermal.cpu_enabled
                ga.thermal_gpu_enabled = c.thermal.gpu_enabled
                ga.thermal_cpu_high = c.thermal.cpu_high
                ga.thermal_cpu_resume = c.thermal.cpu_resume
                ga.thermal_gpu_high = c.thermal.gpu_high
                ga.thermal_gpu_resume = c.thermal.gpu_resume
                ga.thermal_gpu_overrides = c.thermal.gpu_overrides
                ga.thermal_poll_seconds = c.thermal.poll_seconds
                ga.thermal_soft_throttle_enabled = c.thermal.soft_throttle_enabled
                ga.thermal_soft_throttle_temp = c.thermal.soft_throttle_temp
                ga.thermal_soft_throttle_max_sleep = c.thermal.soft_throttle_max_sleep
        except Exception:
            pass

    if "jobs" in data:
        jb = data["jobs"]
        c.jobs.resume_on_restart = bool(jb.get("resume_on_restart", c.jobs.resume_on_restart))
        # Apply live so the change takes effect on the next restart-recovery pass
        # without needing a server restart to re-read config.
        try:
            from codai.api.loras import set_resume_enabled
            set_resume_enabled(c.jobs.resume_on_restart)
        except Exception:
            pass

    if "enhance" in data:
        en = data["enhance"]
        if "allow_ffmpeg" in en:
            c.enhance.allow_ffmpeg = bool(en["allow_ffmpeg"])
        if "allow_rife_ncnn" in en:
            c.enhance.allow_rife_ncnn = bool(en["allow_rife_ncnn"])
        # Apply live to global_args so the video pipeline honours it immediately.
        try:
            from codai.api.state import get_global_args
            ga = get_global_args()
            if ga is not None:
                ga.enhance_allow_ffmpeg = c.enhance.allow_ffmpeg
                ga.enhance_allow_rife_ncnn = c.enhance.allow_rife_ncnn
        except Exception:
            pass

    if "ds4" in data:
        d = data["ds4"]
        c.ds4.enabled = bool(d.get("enabled", c.ds4.enabled))
        if "repo_url" in d:
            c.ds4.repo_url = (d.get("repo_url") or c.ds4.repo_url or "").strip()
        if "install_dir" in d:
            c.ds4.install_dir = (d.get("install_dir") or "").strip() or None
        if "build_target" in d:
            c.ds4.build_target = (d.get("build_target") or "auto").strip()
        if "model_path" in d:
            c.ds4.model_path = (d.get("model_path") or "").strip()
        if "auto_download" in d:
            c.ds4.auto_download = bool(d["auto_download"])
        if "ssd_streaming" in d:
            c.ds4.ssd_streaming = bool(d["ssd_streaming"])
        if "model_variant" in d:
            c.ds4.model_variant = (d.get("model_variant") or c.ds4.model_variant).strip()
        if "model_id" in d:
            c.ds4.model_id = (d.get("model_id") or c.ds4.model_id or "deepseek-v4").strip()
        if "host" in d:
            c.ds4.host = (d.get("host") or "127.0.0.1").strip()
        if "port" in d:
            c.ds4.port = int(d.get("port") or 0)
        if "ctx" in d:
            c.ds4.ctx = max(1024, int(d.get("ctx") or c.ds4.ctx))
        if "extra_args" in d:
            c.ds4.extra_args = (d.get("extra_args") or "").strip()
        if "expert_cache_reserve_gb" in d:
            try:
                c.ds4.expert_cache_reserve_gb = max(0, int(d.get("expert_cache_reserve_gb") or 0))
            except (TypeError, ValueError):
                c.ds4.expert_cache_reserve_gb = 0
        if "extra_env" in d:
            c.ds4.extra_env = (d.get("extra_env") or "").strip()
        if "auto_build" in d:
            c.ds4.auto_build = bool(d["auto_build"])
        if "kv_cache_cleanup_enabled" in d:
            c.ds4.kv_cache_cleanup_enabled = bool(d["kv_cache_cleanup_enabled"])
        if "kv_cache_max_age_hours" in d:
            try:
                c.ds4.kv_cache_max_age_hours = max(0.0, float(d["kv_cache_max_age_hours"]))
            except (TypeError, ValueError):
                pass
        if "kv_cache_cleanup_interval_minutes" in d:
            try:
                c.ds4.kv_cache_cleanup_interval_minutes = max(1.0, float(d["kv_cache_cleanup_interval_minutes"]))
            except (TypeError, ValueError):
                pass

    if "compaction" in data:
        cp = data["compaction"] or {}
        if "enabled" in cp:
            c.compaction.enabled = bool(cp["enabled"])
        if "pct" in cp:
            try:
                c.compaction.pct = max(50, min(99, int(cp["pct"])))
            except (TypeError, ValueError):
                pass
        if "strategy" in cp:
            _st = (cp.get("strategy") or "drop_oldest").strip()
            if _st in ("drop_oldest", "keep_head_tail", "summarize"):
                c.compaction.strategy = _st
        if "model" in cp:
            c.compaction.model = (cp.get("model") or "").strip()

    if "broker" in data:
        bro = data["broker"]
        c.broker.enabled = bool(bro.get("enabled", c.broker.enabled))
        c.broker.base_url = (bro.get("base_url") or "").strip()
        c.broker.scope = (bro.get("scope") or c.broker.scope or "user").strip()
        broker_username = (bro.get("username") or "").strip()
        if c.broker.scope == "global":
            c.broker.username = "global"
        else:
            c.broker.username = broker_username
        c.broker.provider_id = (bro.get("provider_id") or "").strip()
        c.broker.client_id = (bro.get("client_id") or "").strip()
        c.broker.registration_token = (bro.get("registration_token") or "").strip()
        c.broker.advertised_endpoint = (bro.get("advertised_endpoint") or "").strip()
        c.broker.websocket_path = (bro.get("websocket_path") or "").strip()
        c.broker.transport = (bro.get("transport") or c.broker.transport or "websocket").strip()
        c.broker.heartbeat_interval_seconds = max(1, int(bro.get("heartbeat_interval_seconds", c.broker.heartbeat_interval_seconds)))
        c.broker.connect_timeout_seconds = max(1, int(bro.get("connect_timeout_seconds", c.broker.connect_timeout_seconds)))
        c.broker.request_timeout_seconds = max(1, int(bro.get("request_timeout_seconds", c.broker.request_timeout_seconds)))
        c.broker.reconnect_initial_delay_seconds = max(1, int(bro.get("reconnect_initial_delay_seconds", c.broker.reconnect_initial_delay_seconds)))
        c.broker.reconnect_max_delay_seconds = max(
            c.broker.reconnect_initial_delay_seconds,
            int(bro.get("reconnect_max_delay_seconds", c.broker.reconnect_max_delay_seconds)),
        )
        c.broker.websocket_ping_interval = max(5, int(bro.get("websocket_ping_interval", c.broker.websocket_ping_interval)))
        from codai.broker.config import BrokerConfigError, build_broker_runtime_config
        try:
            request.app.state.broker_runtime = build_broker_runtime_config(c.broker)
        except BrokerConfigError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    config_manager.save_config()
    return {"success": True, "warnings": _settings_warnings}


# =============================================================================
# Archive management
# =============================================================================

def _hf_file_size(sibling: dict) -> int:
    """Return actual byte size from an HF siblings entry (prefers LFS size)."""
    lfs = sibling.get("lfs") or {}
    return lfs.get("size") or sibling.get("size") or 0


@router.get("/admin/api/hf-search", summary="Search Hugging Face models")
async def api_hf_search(
    q: str = "",
    gguf_mode: str = "gguf",   # "gguf" | "all" | "no-gguf"
    pipeline_tag: str = "",
    sort: str = "downloads",
    sizes: str = "",            # comma-separated e.g. "7b,70b"
    arch: str = "",
    capabilities: str = "",     # comma-separated e.g. "function-calling,vision"
    component_type: str = "",   # "vae" | "t5xxl" | "clip_l" | "clip_g" | "clip_vision" | "lora" | "encoder" | "controlnet" | "unet"
    username: str = Depends(require_admin),
):
    """Proxy HuggingFace model search; supports multiple sizes via parallel requests."""
    import asyncio
    import urllib.request
    import urllib.parse
    import json as _json
    from codai.models.capabilities import detect_capabilities_from_pipeline_tag

    if sort not in ("downloads", "likes", "lastModified", "createdAt"):
        sort = "downloads"

    # Component type → search keywords + HF tags
    # Most components are safetensors, so override gguf_mode → "all" unless caller forced it
    _COMP_SEARCH: dict = {
        "vae":          {"kw": "vae",              "tags": ["vae"]},
        "t5xxl":        {"kw": "t5xxl OR t5-xxl",  "tags": []},
        "clip_l":       {"kw": "clip-l encoder",   "tags": []},
        "clip_g":       {"kw": "clip-g encoder",   "tags": []},
        "clip_vision":  {"kw": "clip vision encoder","tags": []},
        "lora":         {"kw": "lora",             "tags": ["lora"]},
        "encoder":      {"kw": "text encoder",     "tags": []},
        "controlnet":   {"kw": "controlnet",       "tags": ["controlnet"]},
        "unet":         {"kw": "unet",             "tags": []},
    }
    comp_kw: str = ""
    comp_tags: list = []
    if component_type and component_type in _COMP_SEARCH:
        spec = _COMP_SEARCH[component_type]
        comp_kw = spec["kw"]
        comp_tags = spec["tags"]
        if gguf_mode == "gguf":
            gguf_mode = "all"  # components are usually safetensors; respect explicit "no-gguf" only

    # Filter tags shared across all requests
    filter_pairs: list = []
    if gguf_mode == "gguf":
        filter_pairs.append(("filter", "gguf"))
    if pipeline_tag:
        filter_pairs.append(("filter", pipeline_tag))
    if arch == "lora":
        filter_pairs.append(("filter", "lora"))
    for tag in comp_tags:
        filter_pairs.append(("filter", tag))

    # Capability filters
    cap_list = [c.strip() for c in capabilities.split(",") if c.strip()]
    for cap in cap_list:
        filter_pairs.append(("filter", cap))

    # Base search keywords
    base_parts = [q.strip()] if q.strip() else []
    if comp_kw:
        base_parts.append(comp_kw)
    if arch == "moe":
        base_parts.append("moe")

    size_list = [s.strip() for s in sizes.split(",") if s.strip()][:6]

    async def _one(extra_kw: str = "") -> list:
        parts = base_parts + ([extra_kw] if extra_kw else [])
        effective_q = " ".join(parts)
        limit = "12" if size_list else "20"
        pairs = []
        if effective_q:
            pairs.append(("search", effective_q))
        pairs.extend(filter_pairs)
        pairs += [("sort", sort), ("direction", "-1"), ("limit", limit), ("full", "true")]
        url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(pairs)
        rq = urllib.request.Request(url, headers={"User-Agent": "coderai-admin/1.0"})
        def _fetch():
            with urllib.request.urlopen(rq, timeout=15) as resp:
                return _json.loads(resp.read())
        return await asyncio.to_thread(_fetch)

    try:
        if size_list:
            batches = await asyncio.gather(*[_one(sz) for sz in size_list], return_exceptions=True)
        else:
            batches = [await _one()]

        seen: set = set()
        merged: list = []
        for batch in batches:
            if isinstance(batch, Exception):
                continue
            for m in batch:
                mid = m.get("modelId") or m.get("id", "")
                if mid and mid not in seen:
                    seen.add(mid)
                    merged.append(m)

        if sort == "downloads":
            merged.sort(key=lambda m: m.get("downloads", 0), reverse=True)
        elif sort == "likes":
            merged.sort(key=lambda m: m.get("likes", 0), reverse=True)

        if gguf_mode == "no-gguf":
            merged = [m for m in merged if "gguf" not in (m.get("modelId") or m.get("id", "")).lower()]

        # Get VRAM info
        vram_total_gb = None
        vram_free_gb = None
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                vram_total_gb = round(total / 1e9, 2)
                vram_free_gb = round(free / 1e9, 2)
        except Exception:
            pass

        from codai.models.capabilities import update_capability_cache
        results = []
        for m in merged[:20]:
            mid = m.get("modelId") or m.get("id", "")
            caps = detect_capabilities_from_pipeline_tag(
                m.get("pipeline_tag", ""), mid,
            )
            # Only cache when pipeline_tag gave us authoritative information
            if m.get("pipeline_tag"):
                update_capability_cache(mid, caps)
            # Estimate size from safetensors metadata when available
            safetensors_size_gb = None
            sf = m.get("safetensors") or {}
            total_params = sf.get("total", 0)
            if total_params:
                params_by_dtype = sf.get("parameters") or {}
                dominant = max(params_by_dtype, key=params_by_dtype.get) if params_by_dtype else "BF16"
                bpp = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "I4": 0.5, "U8": 1}.get(dominant, 2)
                safetensors_size_gb = round(total_params * bpp / 1e9, 2)

            results.append({
                "id": mid,
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "pipeline_tag": m.get("pipeline_tag", ""),
                "vram_total": vram_total_gb,
                "vram_free": vram_free_gb,
                "safetensors_size_gb": safetensors_size_gb,
                "capabilities": caps.to_list(),
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace API error: {e}")


@router.get("/admin/api/hf-model-files", summary="List Hugging Face model files")
async def api_hf_model_files(model_id: str, username: str = Depends(require_admin)):
    """Return GGUF files (name, size, VRAM estimate, quant type) for an HF model repo."""
    import urllib.request
    import urllib.parse
    import json as _json

    safe_id = urllib.parse.quote(model_id, safe="/")
    url = f"https://huggingface.co/api/models/{safe_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "coderai-admin/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace API error: {e}")

    files = []
    for sib in data.get("siblings", []):
        name = sib.get("rfilename", "")
        if not name.lower().endswith(".gguf"):
            continue
        size_bytes = _hf_file_size(sib)
        size_gb = round(size_bytes / 1024 ** 3, 2) if size_bytes else None
        vram_gb = round(size_gb * 1.1, 1) if size_gb else None
        m = _QUANT_RE.search(name)
        quant = m.group(1).upper() if m else None
        files.append({
            "name": name,
            "size_gb": size_gb,
            "vram_gb": vram_gb,
            "quant": quant,
        })

    files.sort(key=lambda f: f.get("size_gb") or 0)
    return files


# =============================================================================
# Character profile management proxy (admin UI)
# =============================================================================

@router.get("/admin/api/characters", summary="List characters")
async def api_list_characters(username: str = Depends(require_auth)):
    from codai.api.characters import _list_characters
    return {"characters": _list_characters()}


@router.get("/admin/api/characters/{name}", summary="Get a character")
async def api_get_character(name: str, username: str = Depends(require_auth)):
    from codai.api.characters import _load_character_meta, _load_character_images
    meta = _load_character_meta(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")
    images = _load_character_images(name)
    return {
        "name": meta['name'],
        "description": meta.get('description', ''),
        "image_count": meta['image_count'],
        "created_at": meta['created_at'],
        "images": [img.model_dump() for img in images],
    }


@router.get("/admin/api/characters/{name}/thumbnail", summary="Character thumbnail")
async def api_character_thumbnail(name: str, username: str = Depends(require_auth)):
    import os as _os
    from codai.api.characters import _char_dir, _load_character_meta
    from fastapi.responses import FileResponse
    meta = _load_character_meta(name)
    if not meta:
        raise HTTPException(status_code=404)
    cdir = _char_dir(name)
    for img_info in meta.get('images', []):
        fpath = _os.path.join(cdir, img_info['file'])
        if _os.path.exists(fpath):
            ext = img_info['file'].rsplit('.', 1)[-1].lower()
            media_type = 'image/png' if ext == 'png' else 'image/jpeg'
            return FileResponse(fpath, media_type=media_type)
    raise HTTPException(status_code=404)


@router.delete("/admin/api/characters/{name}", summary="Delete a character")
async def api_delete_character(name: str, username: str = Depends(require_auth)):
    import os as _os, shutil
    from codai.api.characters import _char_dir
    cdir = _char_dir(name)
    if not _os.path.isdir(cdir):
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")
    shutil.rmtree(cdir)
    return {"ok": True, "name": name}


# =============================================================================
# Environment profile management proxy (admin UI)
# =============================================================================

@router.get("/admin/api/environments", summary="List environments")
async def api_list_environments(username: str = Depends(require_auth)):
    from codai.api.environments import _list_environments
    return {"environments": _list_environments()}


@router.get("/admin/api/environments/{name}", summary="Get an environment")
async def api_get_environment(name: str, username: str = Depends(require_auth)):
    from codai.api.environments import _load_environment_meta, _load_environment_images
    meta = _load_environment_meta(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Environment '{name}' not found")
    images = _load_environment_images(name)
    return {
        "name": meta['name'],
        "description": meta.get('description', ''),
        "image_count": meta['image_count'],
        "created_at": meta['created_at'],
        "images": [img.model_dump() for img in images],
    }


@router.get("/admin/api/environments/{name}/thumbnail", summary="Environment thumbnail")
async def api_environment_thumbnail(name: str, username: str = Depends(require_auth)):
    import os as _os
    from codai.api.environments import _env_dir, _load_environment_meta
    from fastapi.responses import FileResponse
    meta = _load_environment_meta(name)
    if not meta:
        raise HTTPException(status_code=404)
    edir = _env_dir(name)
    for img_info in meta.get('images', []):
        fpath = _os.path.join(edir, img_info['file'])
        if _os.path.exists(fpath):
            ext = img_info['file'].rsplit('.', 1)[-1].lower()
            media_type = 'image/png' if ext == 'png' else 'image/jpeg'
            return FileResponse(fpath, media_type=media_type)
    raise HTTPException(status_code=404)


@router.delete("/admin/api/environments/{name}", summary="Delete an environment")
async def api_delete_environment(name: str, username: str = Depends(require_auth)):
    import os as _os, shutil
    from codai.api.environments import _env_dir
    edir = _env_dir(name)
    if not _os.path.isdir(edir):
        raise HTTPException(status_code=404, detail=f"Environment '{name}' not found")
    shutil.rmtree(edir)
    return {"ok": True, "name": name}


# =============================================================================
# Voice profile management proxy (admin UI)
# =============================================================================

@router.get("/admin/api/voices", summary="List voices")
async def api_list_voices(username: str = Depends(require_auth)):
    from codai.api.voice_clone import _list_voices
    return {"voices": _list_voices()}


@router.get("/admin/api/voices/{name}", summary="Get a voice")
async def api_get_voice(name: str, username: str = Depends(require_auth)):
    from codai.api.voice_clone import _load_voice
    meta = _load_voice(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    return {"voice": meta}


@router.delete("/admin/api/voices/{name}", summary="Delete a voice")
async def api_delete_voice(name: str, username: str = Depends(require_auth)):
    import os as _os, shutil
    from codai.api.voice_clone import _voice_path
    vdir = _voice_path(name)
    if not _os.path.exists(vdir):
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    shutil.rmtree(vdir)
    return {"deleted": True, "name": name}


@router.get("/admin/api/hf-model-info", summary="Get Hugging Face model info")
async def api_hf_model_info(model_id: str, username: str = Depends(require_admin)):
    """Full metadata for a single HuggingFace model repo."""
    import urllib.request
    import urllib.parse
    import json as _json

    safe_id = urllib.parse.quote(model_id, safe="/")
    url = f"https://huggingface.co/api/models/{safe_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "coderai-admin/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace API error: {e}")

    card = data.get("cardData") or {}

    # Parameter count from safetensors metadata
    params_label = None
    sf = data.get("safetensors") or {}
    total = sf.get("total")
    if total:
        if total >= 1e12:
            params_label = f"{total/1e12:.1f}T"
        elif total >= 1e9:
            params_label = f"{total/1e9:.1f}B"
        elif total >= 1e6:
            params_label = f"{total/1e6:.0f}M"
        else:
            params_label = str(total)

    # GGUF files with quant/size info
    gguf_files = []
    for sib in data.get("siblings", []):
        name = sib.get("rfilename", "")
        if not name.lower().endswith(".gguf"):
            continue
        size_bytes = _hf_file_size(sib)
        size_gb = round(size_bytes / 1024 ** 3, 2) if size_bytes else None
        vram_gb = round(size_gb * 1.1, 1) if size_gb else None
        m = _QUANT_RE.search(name)
        gguf_files.append({
            "name": name,
            "size_gb": size_gb,
            "vram_gb": vram_gb,
            "quant": m.group(1).upper() if m else None,
        })
    gguf_files.sort(key=lambda f: f.get("size_gb") or 0)

    # All repo files (for total count)
    all_files = [sib.get("rfilename", "") for sib in data.get("siblings", [])]

    # Relevant tags (strip common noisy ones)
    _noise = {"transformers", "safetensors", "gguf", "endpoints_compatible",
              "has_space", "region:us", "license:other"}
    tags = [t for t in data.get("tags", []) if t not in _noise]

    base_model = card.get("base_model") or ""
    if isinstance(base_model, list):
        base_model = ", ".join(base_model)

    return {
        "id": data.get("modelId") or data.get("id", ""),
        "author": data.get("author", ""),
        "pipeline_tag": data.get("pipeline_tag", ""),
        "downloads": data.get("downloads", 0),
        "likes": data.get("likes", 0),
        "last_modified": data.get("lastModified", ""),
        "private": data.get("private", False),
        "gated": data.get("gated", False),
        "tags": tags,
        "license": card.get("license", ""),
        "language": card.get("language") or [],
        "base_model": base_model,
        "params_label": params_label,
        "gguf_files": gguf_files,
        "file_count": len(all_files),
    }
