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
import re
import shutil
from typing import Optional

from fastapi import APIRouter, Request, Response, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from codai.admin.auth import SessionManager
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
config_manager = None  # set via set_config_manager()
_download_sessions: dict = {}
_download_status: dict = {}   # session_id → latest progress state (survives SSE disconnect)


def init_session_manager(config_dir: Path):
    """Initialize the session manager."""
    global session_manager
    session_manager = SessionManager(config_dir)


def set_config_manager(mgr):
    """Set the shared ConfigManager instance."""
    global config_manager
    config_manager = mgr
    from codai.models.capabilities import init_capability_cache
    init_capability_cache(str(mgr.config_dir))


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
    return shutil.which("whisper-server") or "/usr/local/bin/whisper-server"


def get_current_user(request: Request) -> Optional[str]:
    """Get the current logged-in user from session cookie."""
    if session_manager is None:
        return None
    
    cookie = request.cookies.get("session")
    if not cookie:
        return None
    
    # Handle MUST_CHANGE flag
    if cookie.endswith(".MUST_CHANGE"):
        cookie = cookie[:-12]  # Remove .MUST_CHANGE suffix
    
    return session_manager.validate_session(cookie)


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


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Display login page."""
    # If already logged in, redirect to dashboard
    username = get_current_user(request)
    if username:
        return RedirectResponse(url="/admin", status_code=302)
    
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
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
        return templates.TemplateResponse(request, "login.html", {"error": "Invalid username or password"})
    
    # Check if must change password
    must_change = session_cookie.endswith(".MUST_CHANGE")
    if must_change:
        session_cookie = session_cookie[:-12]
    
    response = RedirectResponse(
        url="/admin/change-password" if must_change else "/admin",
        status_code=302
    )
    response.set_cookie(
        key="session",
        value=session_cookie,
        httponly=True,
        secure=False,  # Set to True if using HTTPS
        samesite="strict",
        max_age=30 * 24 * 60 * 60
    )
    return response


@router.get("/logout")
async def logout(request: Request):
    """Handle logout."""
    if session_manager:
        cookie = request.cookies.get("session")
        session_manager.destroy_session(cookie)
    
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("session")
    return response


@router.get("/admin/change-password", response_class=HTMLResponse)
async def change_password_page(request: Request, username: str = Depends(require_auth)):
    user = session_manager.get_user(username)
    must_change = user.get("must_change_password", False) if user else False
    return templates.TemplateResponse(request, "change_password.html", {
        "username": username,
        "must_change": must_change,
        "is_admin": session_manager.is_admin(username),
        "error": None,
    })


@router.post("/admin/change-password")
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
        return templates.TemplateResponse(request, "change_password.html", {
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

    return RedirectResponse(url="/admin", status_code=302)


@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, username: str = Depends(require_auth)):
    is_admin = session_manager.is_admin(username)
    return templates.TemplateResponse(request, "dashboard.html", {
        "username": username, "is_admin": is_admin,
    })


@router.get("/admin/models", response_class=HTMLResponse)
async def models_page(request: Request, username: str = Depends(require_admin)):
    return templates.TemplateResponse(request, "models.html", {"username": username, "is_admin": True})


@router.get("/admin/tokens", response_class=HTMLResponse)
async def tokens_page(request: Request, username: str = Depends(require_admin)):
    return templates.TemplateResponse(request, "tokens.html", {"username": username, "is_admin": True})


@router.get("/admin/users", response_class=HTMLResponse)
async def users_page(request: Request, username: str = Depends(require_admin)):
    users = session_manager.list_users()
    return templates.TemplateResponse(request, "users.html", {
        "username": username, "is_admin": True, "users": users,
    })


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, username: str = Depends(require_auth)):
    return templates.TemplateResponse(request, "chat.html", {
        "username": username, "is_admin": session_manager.is_admin(username),
    })


# API endpoints for admin operations
@router.get("/admin/api/status")
async def api_status(username: str = Depends(require_auth)):
    """Get system status."""
    from codai.models.manager import multi_model_manager
    from codai.api.state import get_load_mode

    loaded_keys = list(multi_model_manager.models.keys())

    # VRAM info
    vram = None
    is_cuda = False
    try:
        import torch
        if torch.cuda.is_available():
            is_cuda = True
            free, total = torch.cuda.mem_get_info()
            used = total - free
            vram = {"used": round(used / 1e9, 2), "free": round(free / 1e9, 2), "total": round(total / 1e9, 2),
                    "gpu": torch.cuda.get_device_name(0)}
    except Exception:
        pass

    # Non-CUDA: read from sysfs (AMD amdgpu / Intel i915 / Arc)
    if not is_cuda:
        import os, glob as _glob
        for card in sorted(_glob.glob("/sys/class/drm/card[0-9]")):
            dev = card + "/device"
            vram_total_path = dev + "/mem_info_vram_total"
            if not os.path.exists(vram_total_path):
                continue
            try:
                total_b = int(open(vram_total_path).read())
                used_b  = int(open(dev + "/mem_info_vram_used").read())
                free_b  = total_b - used_b
                # GPU name from lspci
                gpu_name = ""
                try:
                    pci_addr = os.path.basename(os.path.realpath(dev))
                    import subprocess
                    r = subprocess.run(["lspci", "-s", pci_addr], capture_output=True, text=True, timeout=3)
                    if r.returncode == 0 and r.stdout:
                        # "05:00.0 VGA compatible controller: AMD Radeon RX 580"
                        gpu_name = r.stdout.split(":", 2)[-1].strip().rstrip()
                except Exception:
                    pass
                vram = {
                    "gpu": gpu_name,
                    "used": round(used_b / 1e9, 2),
                    "free": round(free_b / 1e9, 2),
                    "total": round(total_b / 1e9, 2),
                }
                break
            except Exception:
                continue

    # Request stats from queue manager
    req_total = 0
    req_active = 0
    req_waiting = 0
    req_metrics = {
        "max_parallel_requests": 0,
        "queue_max_size": 0,
        "active_by_model": {},
        "waiting_by_model": {},
    }
    try:
        from codai.queue.manager import queue_manager
        metrics = queue_manager.get_metrics()
        req_active = int(metrics.get("active", 0))
        req_waiting = int(metrics.get("waiting", 0))
        req_total = req_active + req_waiting
        req_metrics = {
            "max_parallel_requests": metrics.get("max_parallel_requests", 0),
            "queue_max_size": metrics.get("queue_max_size", 0),
            "active_by_model": metrics.get("active_by_model", {}),
            "waiting_by_model": metrics.get("waiting_by_model", {}),
        }
    except Exception:
        pass

    # Backend info
    backend = "—"
    try:
        from codai.models.manager import model_manager
        if model_manager.backend_type:
            backend = model_manager.backend_type
    except Exception:
        pass

    # Enabled (configured) models + aliases
    enabled_models = []
    enabled_aliases: dict = {}  # alias -> [model_id, ...]
    try:
        if config_manager:
            md = config_manager.models_data
            for cat in ("text_models", "image_models", "audio_models", "vision_models", "tts_models",
                        "video_models", "audio_gen_models", "embedding_models"):
                for m in md.get(cat, []):
                    if isinstance(m, dict):
                        mid = m.get("path") or m.get("id") or ""
                        alias = (m.get("alias") or "").strip()
                    else:
                        mid = m
                        alias = ""
                    if mid and mid not in enabled_models:
                        enabled_models.append(mid)
                    if alias:
                        enabled_aliases.setdefault(alias, [])
                        if mid and mid not in enabled_aliases[alias]:
                            enabled_aliases[alias].append(mid)
    except Exception:
        pass

    # Recent activity
    recent_activity = []
    try:
        from codai.api.log import get_recent_activity
        for row in get_recent_activity():
            if not isinstance(row, dict):
                continue
            try:
                ts = int(row.get("time", 0) or 0)
            except (TypeError, ValueError):
                ts = 0
            try:
                status = int(row.get("status", 0) or 0)
            except (TypeError, ValueError):
                status = 0
            try:
                duration = round(float(row.get("duration", 0) or 0), 2)
            except (TypeError, ValueError):
                duration = 0.0
            recent_activity.append({
                "time": ts,
                "model": str(row.get("model") or "—"),
                "type": str(row.get("type") or "unknown"),
                "status": status,
                "duration": duration,
            })
    except Exception:
        pass

    # Whisper-server status
    whisper_status = None
    try:
        from codai.models.manager import multi_model_manager as _mmm
        if _mmm.whisper_servers:
            whisper_status = {mid: wsm.get_status() for mid, wsm in _mmm.whisper_servers.items()}
        elif _mmm.whisper_server:
            whisper_status = {"whisper-server": _mmm.whisper_server.get_status()}
    except Exception:
        pass

    return {
        "status": "ok",
        "backend": backend,
        "load_mode": get_load_mode(),
        "models_loaded": len(loaded_keys),
        "loaded_models": loaded_keys,
        "enabled_models": enabled_models,
        "enabled_aliases": enabled_aliases,
        "vram": vram,
        "cuda": is_cuda,
        "requests": {
            "total": req_total,
            "active": req_active,
            "waiting": req_waiting,
            "max_parallel_requests": req_metrics["max_parallel_requests"],
            "queue_max_size": req_metrics["queue_max_size"],
            "active_by_model": req_metrics["active_by_model"],
            "waiting_by_model": req_metrics["waiting_by_model"],
        },
        "recent_activity": recent_activity,
        "whisper_server": whisper_status,
    }


@router.post("/admin/api/users")
async def api_create_user(
    request: Request,
    username: str = Depends(require_admin)
):
    """Create a new user."""
    data = await request.json()
    new_username = data.get("username")
    password = data.get("password")
    role = data.get("role", "user")
    
    if not new_username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")
    
    success = session_manager.create_user(new_username, password, role)
    if not success:
        raise HTTPException(status_code=400, detail="User already exists")
    
    return {"success": True}


@router.delete("/admin/api/users/{user_id}")
async def api_delete_user(
    user_id: int,
    username: str = Depends(require_admin)
):
    """Delete a user."""
    users = session_manager._load_auth_data().get("users", [])
    user = next((u for u in users if u["id"] == user_id), None)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    success = session_manager.delete_user(user["username"])
    if not success:
        raise HTTPException(status_code=400, detail="Cannot delete user")
    
    return {"success": True}


# --- Token management endpoints ---

@router.get("/admin/api/tokens", response_model=list)
async def api_list_tokens(username: str = Depends(require_admin)):
    """List all API tokens."""
    auth_data = session_manager._load_auth_data()
    tokens = []
    for token in auth_data.get("tokens", []):
        tokens.append({
            "id": token["id"],
            "name": token["name"],
            "token": token["token"],
            "provider": token["provider"],
            "created_at": token["created_at"],
            "last_used": token.get("last_used")
        })
    return tokens


@router.post("/admin/api/tokens")
async def api_create_token(request: Request, username: str = Depends(require_admin)):
    """Create a new API token."""
    data = await request.json()
    name = data.get("name")
    provider = data.get("provider", "openai")
    
    if not name:
        raise HTTPException(status_code=400, detail="Token name is required")
    
    auth_data = session_manager._load_auth_data()
    
    # Generate token
    token_id = len(auth_data.get("tokens", [])) + 1
    import secrets
    new_token = {
        "id": token_id,
        "name": name,
        "token": f"sk-coderai-{secrets.token_hex(32)}",
        "provider": provider,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "last_used": None
    }
    
    auth_data.setdefault("tokens", []).append(new_token)
    session_manager._save_auth_data(auth_data)
    
    return {
        "token": new_token["token"],
        "id": new_token["id"],
        "name": new_token["name"],
        "provider": new_token["provider"]
    }


@router.delete("/admin/api/tokens/{token_id}")
async def api_delete_token(token_id: int, username: str = Depends(require_admin)):
    """Delete an API token."""
    auth_data = session_manager._load_auth_data()
    tokens = auth_data.get("tokens", [])
    
    new_tokens = [t for t in tokens if t["id"] != token_id]
    if len(new_tokens) == len(tokens):
        raise HTTPException(status_code=404, detail="Token not found")
    
    auth_data["tokens"] = new_tokens
    session_manager._save_auth_data(auth_data)
    
    return {"success": True}


# --- Models management endpoints ---

@router.get("/admin/api/models")
async def api_list_models(username: str = Depends(require_admin)):
    """List all configured models with details."""
    models_data = session_manager._load_auth_data()  # TODO: move to ModelManager
    from codai.models.manager import multi_model_manager
    try:
        return multi_model_manager.list_models()
    except Exception:
        return []


def _make_tqdm_class(pq, status=None):
    """Return a tqdm-compatible class that forwards progress events to pq and optionally updates a status dict."""
    import time as _time

    class _PQTqdm:
        def __init__(self, iterable=None, desc=None, total=None, initial=0, **kwargs):
            self.iterable = iterable
            self.desc = str(desc or 'downloading')
            self.total = int(total) if total else 0
            self.n = int(initial) if initial else 0
            self._start = _time.time()
            if self.total:
                pq.put({"type": "start", "filename": self.desc, "total": self.total})
                if status is not None:
                    status.update({"status": "downloading", "filename": self.desc,
                                   "total": self.total, "downloaded": self.n, "percent": 0})

        def update(self, n=1):
            self.n += n
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
    """Background thread: download model via HF snapshot_download and stream progress events."""
    import time
    import os

    status = {"session_id": session_id, "model_id": model_id, "status": "starting",
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

    try:
        from codai.models.cache import is_huggingface_model_id
        from huggingface_hub import snapshot_download

        tqdm_cls = _make_tqdm_class(pq, status=status)

        if is_huggingface_model_id(model_id):
            if file_pattern:
                # Convert suffix/quant pattern to fnmatch glob for allow_patterns
                if file_pattern.startswith('.'):
                    allow = [f"*{file_pattern}"]          # ".gguf" → "*.gguf"
                elif '/' in file_pattern:
                    allow = [file_pattern]                  # exact subpath
                else:
                    allow = [f"*{file_pattern}"]          # "Q4_K_M.gguf" → "*Q4_K_M.gguf"
                push({"type": "info", "message": f"Downloading {allow[0]} from {model_id}…"})
                path = snapshot_download(model_id, allow_patterns=allow, tqdm_class=tqdm_cls)
            else:
                push({"type": "info", "message": f"Downloading full repository {model_id}…"})
                path = snapshot_download(model_id, tqdm_class=tqdm_cls)

        else:
            # Direct URL download (non-HF source)
            import requests as _req
            import hashlib
            from codai.models.cache import get_model_cache_dir

            cache_dir = get_model_cache_dir()
            url_path = model_id.split('?')[0]
            filename = os.path.basename(url_path) or "model.bin"
            url_hash = hashlib.sha256(model_id.encode()).hexdigest()
            dest = os.path.join(cache_dir, f"{url_hash}_{filename}")

            if os.path.exists(dest):
                push({"type": "done", "path": dest})
                return

            resp = _req.get(model_id, stream=True, timeout=60, allow_redirects=True)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            push({"type": "start", "filename": filename, "total": total})

            downloaded = 0
            start_t = time.time()
            last_evt = 0.0
            with open(dest, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=524288):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_evt >= 0.25:
                            last_evt = now
                            elapsed = (now - start_t) or 0.001
                            rate = downloaded / elapsed
                            eta = (total - downloaded) / rate if rate and total else None
                            push({
                                "type": "progress", "filename": filename,
                                "downloaded": downloaded, "total": total,
                                "percent": round(downloaded / total * 100, 1) if total else 0,
                                "rate": round(rate),
                                "eta": round(eta) if eta is not None else None,
                            })
            path = dest

        push({"type": "done", "path": str(path)})

    except Exception as exc:
        push({"type": "error", "message": str(exc)})
    finally:
        def _gc():
            time.sleep(300)
            _download_sessions.pop(session_id, None)
            _download_status.pop(session_id, None)
        _t.Thread(target=_gc, daemon=True).start()


@router.post("/admin/api/model-download")
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

    session_id = str(_uuid.uuid4())
    pq = _q.Queue()
    _download_sessions[session_id] = pq

    _t.Thread(
        target=_run_download_thread,
        args=(session_id, model_id, file_pattern, pq),
        daemon=True,
    ).start()

    return {"session_id": session_id}


@router.get("/admin/api/download-stream/{session_id}")
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
                if evt.get("type") in ("done", "error"):
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


@router.delete("/admin/api/models/{model_identifier}")
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

@router.get("/admin/api/downloads")
async def api_list_downloads(username: str = Depends(require_admin)):
    """Return status of all active and recently completed download sessions."""
    return list(_download_status.values())


@router.post("/admin/api/model-upload")
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
    
    cache_dir = get_model_cache_dir()
    temp_dir = tempfile.gettempdir()
    upload_id = form.get("upload_id", filename)
    temp_path = os.path.join(temp_dir, f"upload_{upload_id}.part")
    
    # Append chunk
    chunk_data = await chunk.read()
    with open(temp_path, "ab") as f:
        f.write(chunk_data)
    
    # If last chunk, move to final location
    if chunk_index == total_chunks - 1:
        final_path = os.path.join(cache_dir, filename)
        os.replace(temp_path, final_path)
        return {"success": True, "complete": True, "path": final_path}
    
    return {"success": True, "complete": False, "chunk_index": chunk_index}


# ── cache scan helpers (run in thread pool) ──────────────────────────────────

def _scan_caches() -> dict:
    import os
    result: dict = {"hf": [], "gguf": []}

    from codai.models.cache import get_all_cache_dirs, get_model_cache_dir
    from codai.models.capabilities import (
        detect_model_capabilities, lookup_capability_cache,
    )
    caches = get_all_cache_dirs()

    # Collect configured models: key (path/id) → (settings_dict, model_type)
    configured_settings: dict = {}
    if config_manager:
        md = config_manager.models_data
        for cat in ("text_models", "image_models", "audio_models",
                    "gguf_models", "tts_models", "vision_models", "video_models",
                    "audio_gen_models", "embedding_models"):
            for m in md.get(cat, []):
                if isinstance(m, str):
                    p = m
                    configured_settings[p] = ({}, cat)
                else:
                    p = m.get("path") or m.get("id") or ""
                    if p:
                        configured_settings[p] = (m, cat)

    # HuggingFace cache
    hf_dir = caches.get("huggingface")
    if hf_dir:
        try:
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir(hf_dir)
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
                                   or configured_settings.get(fname)
                                   or ({}, None))
                            caps = detect_model_capabilities(fname)
                            result["gguf"].append({
                                "filename": fname,
                                "path": fpath,
                                "size_gb": round(fsize / 1e9, 2),
                                "size_bytes": fsize,
                                "in_config": fpath in configured_settings or fname in configured_settings,
                                "model_type": cfg[1] if cfg[1] and cfg[1] != "gguf_models" else "text_models",
                                "settings": cfg[0] if isinstance(cfg[0], dict) else {},
                                "capabilities": caps.to_list(),
                                "source_repo": repo.repo_id,
                            })
                    continue  # skip adding to hf list

                cfg = configured_settings.get(repo.repo_id, ({}, None))
                caps = (lookup_capability_cache(repo.repo_id)
                        or detect_model_capabilities(repo.repo_id))
                result["hf"].append({
                    "id": repo.repo_id,
                    "size_gb": round(size_bytes / 1e9, 2),
                    "size_bytes": size_bytes,
                    "revision_count": len(list(repo.revisions)),
                    "files": files[:30],
                    "file_count": len(files),
                    "in_config": repo.repo_id in configured_settings,
                    "model_type": cfg[1] if cfg[1] and cfg[1] != "gguf_models" else "text_models",
                    "settings": cfg[0] if isinstance(cfg[0], dict) else {},
                    "capabilities": caps.to_list(),
                })
        except Exception as e:
            result["hf_error"] = str(e)

    # GGUF cache (coderai-specific)
    gguf_dir = caches.get("coderai") or get_model_cache_dir()
    if gguf_dir and os.path.exists(gguf_dir):
        for fname in sorted(os.listdir(gguf_dir)):
            fpath = os.path.join(gguf_dir, fname)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                cfg = (configured_settings.get(fpath)
                       or configured_settings.get(fname)
                       or ({}, None))
                caps = detect_model_capabilities(fname)
                result["gguf"].append({
                    "filename": fname,
                    "path": fpath,
                    "size_gb": round(size / 1e9, 2),
                    "size_bytes": size,
                    "in_config": fpath in configured_settings or fname in configured_settings,
                    "model_type": cfg[1] if cfg[1] and cfg[1] != "gguf_models" else "text_models",
                    "settings": cfg[0] if isinstance(cfg[0], dict) else {},
                    "capabilities": caps.to_list(),
                })

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
            # Try to get size if it's a local file
            size_bytes = 0
            if os.path.isfile(path):
                size_bytes = os.path.getsize(path)
            caps = detect_model_capabilities(path)
            result["gguf"].append({
                "filename": os.path.basename(path) if '/' in path else path,
                "path": path,
                "size_gb": round(size_bytes / 1e9, 2) if size_bytes else 0,
                "size_bytes": size_bytes,
                "in_config": True,
                "model_type": mtype if mtype and mtype != "gguf_models" else "text_models",
                "settings": settings if isinstance(settings, dict) else {},
                "capabilities": caps.to_list(),
            })

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
            # Fallback: remove directory directly
            safe = model_id.replace("/", "--")
            d = os.path.join(hf_dir, f"models--{safe}")
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
                return {"success": True}
        return {"success": False, "detail": "Model not found in HF cache"}

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


@router.get("/admin/api/cached-models")
async def api_cached_models(username: str = Depends(require_admin)):
    """Scan both caches and return all locally stored models."""
    import asyncio
    return await asyncio.to_thread(_scan_caches)


@router.get("/admin/api/cache-stats")
async def api_cache_stats(username: str = Depends(require_admin)):
    """Return disk-usage statistics for each cache."""
    import asyncio
    return await asyncio.to_thread(_get_cache_stats)


@router.delete("/admin/api/cache")
async def api_clear_cache(cache_type: str = "all", username: str = Depends(require_admin)):
    """Bulk-delete cache. cache_type: all | hf | gguf"""
    import asyncio
    return await asyncio.to_thread(_do_clear_cache, cache_type)


@router.delete("/admin/api/cached-models/{model_id:path}")
async def api_delete_cached_model(
    model_id: str,
    cache_type: str = "hf",
    username: str = Depends(require_admin),
):
    """Delete a specific cached model (HF repo ID or GGUF filename)."""
    import asyncio
    return await asyncio.to_thread(_do_delete_model, model_id, cache_type)


@router.post("/admin/api/model-enable")
async def api_model_enable(request: Request, username: str = Depends(require_admin)):
    """Register a cached model in models.json so CoderAI can use it."""
    if config_manager is None:
        raise HTTPException(status_code=503, detail="Config manager not initialized")
    data = await request.json()
    path = data.get("path") or data.get("model_id", "")
    model_type = data.get("model_type", "text_models")
    valid = {"text_models", "image_models", "audio_models", "gguf_models", "tts_models", "vision_models",
             "video_models", "audio_gen_models", "embedding_models"}
    if model_type not in valid:
        raise HTTPException(status_code=400, detail=f"model_type must be one of {valid}")
    lst = config_manager.models_data.setdefault(model_type, [])
    if path not in lst:
        lst.append(path)
        config_manager.save_models()
    return {"success": True}


@router.post("/admin/api/model-disable")
async def api_model_disable(request: Request, username: str = Depends(require_admin)):
    """Remove a model from models.json (keeps it cached locally)."""
    if config_manager is None:
        raise HTTPException(status_code=503, detail="Config manager not initialized")
    import os as _os
    data = await request.json()
    path = data.get("path") or data.get("model_id", "")
    # Also match by bare filename so entries stored without full path are caught
    fname = _os.path.basename(path) if (_os.sep in path or "/" in path) else ""

    def _matches(m_entry) -> bool:
        key = m_entry if isinstance(m_entry, str) else m_entry.get("path", m_entry.get("id", ""))
        return key == path or (fname and _os.path.basename(key) == fname)

    changed = False
    for cat in ("text_models", "image_models", "audio_models",
                "gguf_models", "tts_models", "vision_models", "video_models",
                "audio_gen_models", "embedding_models"):
        lst = config_manager.models_data.get(cat, [])
        new_lst = [m for m in lst if not _matches(m)]
        if len(new_lst) != len(lst):
            config_manager.models_data[cat] = new_lst
            changed = True
    if changed:
        config_manager.save_models()
    return {"success": True}


@router.get("/admin/api/model-loaded-status")
async def api_model_loaded_status(username: str = Depends(require_admin)):
    """Return loaded model keys with per-model instance pool info."""
    from codai.models.manager import multi_model_manager
    loaded = list(multi_model_manager.models.keys())

    instance_pools = {}
    for key, pool in multi_model_manager.model_pools.items():
        instance_pools[key] = {"loaded": pool.count, "max": pool.max_instances}

    configured_max = {}
    if config_manager:
        for cat in ("text_models", "image_models", "audio_models", "vision_models",
                    "tts_models", "gguf_models", "video_models", "audio_gen_models", "embedding_models"):
            for m in config_manager.models_data.get(cat, []):
                if isinstance(m, dict):
                    path = m.get("path") or m.get("id") or ""
                    max_inst = m.get("max_instances", 1)
                    if path and max_inst and int(max_inst) > 1:
                        configured_max[path] = int(max_inst)

    return {"loaded": loaded, "instances": instance_pools, "configured_max": configured_max}


@router.post("/admin/api/model-load")
async def api_model_load(request: Request, username: str = Depends(require_admin)):
    """Load a configured model into VRAM (same VRAM checks as a real request)."""
    from codai.models.manager import multi_model_manager
    data = await request.json()
    path = data.get("path", "")
    if not path:
        raise HTTPException(status_code=400, detail="path required")

    # Find the model config entry to determine its type
    model_type = "text"
    if config_manager:
        md = config_manager.models_data
        for cat, mtype in (("image_models", "image"), ("audio_models", "audio"),
                           ("vision_models", "vision"), ("tts_models", "tts"),
                           ("video_models", "video"),
                           ("audio_gen_models", "audio_gen"),
                           ("embedding_models", "embedding")):
            for m in md.get(cat, []):
                mid = m if isinstance(m, str) else m.get("path") or m.get("id") or ""
                if mid == path:
                    model_type = mtype
                    break

    result = multi_model_manager.request_model(path, model_type if model_type != "text" else None)
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
        multi_model_manager._evict_models_for_vram(needed_gb)
    elif needed_gb == 0 and multi_model_manager.models and free_gb < 4.0:
        # Unknown model size but VRAM nearly full — evict everything to avoid OOM on first attempt
        print(f"Admin model-load: unknown model size, only {free_gb:.1f} GB free — evicting models proactively")
        multi_model_manager.unload_all_models()

    # Not loaded yet — trigger actual load
    try:
        if model_type == "text":
            # _load_model_by_name already records the VRAM delta internally
            mm = multi_model_manager._load_model_by_name(result["model_name"] or path)
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
                resolved = multi_model_manager.load_model(path)
                import os as _os
                if resolved and _os.path.isfile(resolved):
                    sd_model = _load_sdcpp_model(resolved, global_args)
                    if sd_model:
                        multi_model_manager.add_model(model_key, sd_model)
                        multi_model_manager.record_vram_delta(model_key, _snap)
            else:
                pipeline = _load_diffusers_pipeline(path, global_args)
                if pipeline:
                    multi_model_manager.add_model(model_key, pipeline)
                    multi_model_manager.record_vram_delta(model_key, _snap)
        return {"success": True, "already_loaded": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/api/model-unload")
async def api_model_unload(request: Request, username: str = Depends(require_admin)):
    """Unload a model from VRAM (keeps it available for on-request reload)."""
    import gc
    from codai.models.manager import multi_model_manager
    data = await request.json()
    path = data.get("path", "")
    if not path:
        raise HTTPException(status_code=400, detail="path required")

    # Find the key in loaded models (exact or prefixed)
    key = None
    for k in list(multi_model_manager.models.keys()):
        if k == path or k.endswith(f":{path}") or k.endswith(path.split("/")[-1]):
            key = k
            break
    if key is None:
        return {"success": True, "was_loaded": False}

    model_obj = multi_model_manager.models.pop(key, None)
    if model_obj is not None:
        try:
            if hasattr(model_obj, "cleanup"):
                model_obj.cleanup()
            elif hasattr(model_obj, "to"):
                model_obj.to("cpu")
        except Exception:
            pass
    if multi_model_manager.active_in_vram == key:
        multi_model_manager.active_in_vram = None
    if multi_model_manager.current_model_key == key:
        multi_model_manager.current_model_key = None

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {"success": True, "was_loaded": True}


@router.post("/admin/api/model-configure")
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
        # Remove existing entry with same id (update semantics)
        audio_list = config_manager.models_data.get("audio_models", [])
        config_manager.models_data["audio_models"] = [
            m for m in audio_list
            if not (isinstance(m, dict) and m.get("id") == model_id)
        ]
        alias = (data.get("alias") or "").strip() or None
        entry = {
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
            entry["alias"] = alias
        if data.get("used_vram_gb") is not None:
            entry["used_vram_gb"] = data["used_vram_gb"]
        config_manager.models_data.setdefault("audio_models", []).append(entry)
        config_manager.save_models()
        result = {"success": True, "model_id": model_id, "model_path": model_path, "server_path": server_path}
        if alias:
            result["alias"] = alias
        return result
    path = data.get("path") or data.get("model_id", "")
    valid = {"text_models", "image_models", "audio_models", "tts_models", "vision_models", "video_models",
             "audio_gen_models", "embedding_models"}
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

    # Remove from all categories (handles type changes and quant switches)
    # paths_to_remove: the new path + the original path before a quant change
    import os as _os
    paths_to_remove = {path}
    orig_path = (data.get("orig_path") or "").strip()
    if orig_path and orig_path != path:
        paths_to_remove.add(orig_path)
    # Also match by bare filename so entries stored without full path are caught
    fnames_to_remove = {_os.path.basename(p) for p in paths_to_remove if _os.sep in p or "/" in p}

    def _should_remove(m_entry) -> bool:
        key = m_entry if isinstance(m_entry, str) else m_entry.get("path", m_entry.get("id", ""))
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
    entry: dict = {"path": path, "model_type": model_types[0], "model_types": model_types}
    if used_vram_gb is not None:
        entry["used_vram_gb"] = used_vram_gb
    # Store video sub-types (t2v / i2v / v2v) when present
    if data.get("video_subtypes"):
        entry["video_subtypes"] = data["video_subtypes"]
    for key in ("alias", "backend", "load_mode", "n_gpu_layers", "n_ctx",
                "max_gpu_percent", "manual_ram_gb", "load_in_4bit", "load_in_8bit",
                "flash_attention", "no_ram", "offload_strategy", "offload_dir",
                "system_prompt", "parser", "tools_closer_prompt", "grammar_guided",
                "max_instances", "preload_all_instances"):
        if key in data:
            entry[key] = data[key]

    # Add entry to each selected category
    for mtype in model_types:
        config_manager.models_data.setdefault(mtype, []).append(entry)
    config_manager.save_models()
    return {"success": True}


# --- System endpoints ---

@router.post("/admin/api/system/reload")
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

@router.get("/admin/settings", response_class=HTMLResponse)
async def settings_page(request: Request, username: str = Depends(require_admin)):
    return templates.TemplateResponse(request, "settings.html", {"username": username, "is_admin": True})


@router.get("/admin/api/settings")
async def api_get_settings(username: str = Depends(require_admin)):
    """Return current config.json as JSON."""
    if config_manager is None or config_manager.config is None:
        raise HTTPException(status_code=503, detail="Config manager not initialized")
    c = config_manager.config
    return {
        "server": {
            "host": c.server.host,
            "port": c.server.port,
            "https": c.server.https,
            "https_key_path": c.server.https_key_path,
            "https_cert_path": c.server.https_cert_path,
            "queue_max_size": c.server.queue_max_size,
            "max_parallel_requests": c.server.max_parallel_requests,
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
        "system_prompt": c.system_prompt,
        "tools_closer_prompt": c.tools_closer_prompt,
        "grammar_guided": c.grammar_guided,
        "parser": c.parser,
    }


@router.post("/admin/api/settings")
async def api_save_settings(request: Request, username: str = Depends(require_admin)):
    """Update and persist config.json from submitted JSON. Only sections present in the payload are updated."""
    if config_manager is None or config_manager.config is None:
        raise HTTPException(status_code=503, detail="Config manager not initialized")

    data = await request.json()
    c = config_manager.config

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

    config_manager.save_config()
    return {"success": True}


# =============================================================================
# Archive management
# =============================================================================

@router.get("/admin/archive", response_class=HTMLResponse)
async def archive_page(request: Request, username: str = Depends(require_admin)):
    return templates.TemplateResponse(request, "archive.html", {"username": username, "is_admin": True})


@router.get("/admin/api/archive")
async def api_archive_list(
    limit: int = 50,
    offset: int = 0,
    username: str = Depends(require_admin),
):
    from codai.api.archive import archive_manager
    entries = archive_manager.list_entries(limit=limit, offset=offset)
    total = archive_manager.count_entries()
    return {"entries": entries, "total": total}


@router.get("/admin/api/archive/{gen_id}")
async def api_archive_get(gen_id: str, username: str = Depends(require_admin)):
    from codai.api.archive import archive_manager
    entry = archive_manager.get_entry(gen_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry


@router.delete("/admin/api/archive/{gen_id}")
async def api_archive_delete(gen_id: str, username: str = Depends(require_admin)):
    from codai.api.archive import archive_manager
    if not archive_manager.delete_entry(gen_id):
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"success": True}


@router.get("/admin/api/archive/{gen_id}/files/{filename}")
async def api_archive_file(
    gen_id: str,
    filename: str,
    username: str = Depends(require_auth),
):
    from fastapi.responses import FileResponse
    from codai.api.archive import archive_manager
    path = archive_manager.get_file_path(gen_id, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    import mimetypes
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@router.get("/admin/api/archive-settings")
async def api_archive_settings_get(username: str = Depends(require_admin)):
    if config_manager is None or config_manager.config is None:
        raise HTTPException(status_code=503, detail="Config not ready")
    import os as _os
    c = config_manager.config.archive
    from codai.api.archive import RETENTION_OPTIONS
    default_dir = _os.path.join(str(config_manager.config_dir), "archive")
    return {
        "enabled": c.enabled,
        "directory": c.directory,
        "default_directory": default_dir,
        "retention": c.retention,
        "retention_options": RETENTION_OPTIONS,
    }


# --- HuggingFace model search proxy ---

import re as _re
_QUANT_RE = _re.compile(
    r'\b(IQ[1-4]_XX[SML]?|Q[2-8]_K_[MSLX]|Q[2-8]_K|Q[2-8]_[0-9]|F16|F32|BF16)\b',
    _re.IGNORECASE,
)


def _hf_file_size(sibling: dict) -> int:
    """Return actual byte size from an HF siblings entry (prefers LFS size)."""
    lfs = sibling.get("lfs") or {}
    return lfs.get("size") or sibling.get("size") or 0


@router.get("/admin/api/hf-search")
async def api_hf_search(
    q: str = "",
    gguf_mode: str = "gguf",   # "gguf" | "all" | "no-gguf"
    pipeline_tag: str = "",
    sort: str = "downloads",
    sizes: str = "",            # comma-separated e.g. "7b,70b"
    arch: str = "",
    capabilities: str = "",     # comma-separated e.g. "function-calling,vision"
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

    # Filter tags shared across all requests
    filter_pairs: list = []
    if gguf_mode == "gguf":
        filter_pairs.append(("filter", "gguf"))
    if pipeline_tag:
        filter_pairs.append(("filter", pipeline_tag))
    if arch == "lora":
        filter_pairs.append(("filter", "lora"))
    
    # Capability filters
    cap_list = [c.strip() for c in capabilities.split(",") if c.strip()]
    for cap in cap_list:
        filter_pairs.append(("filter", cap))

    # Base search keywords
    base_parts = [q.strip()] if q.strip() else []
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


@router.get("/admin/api/hf-model-files")
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

@router.get("/admin/api/characters")
async def api_list_characters(username: str = Depends(require_auth)):
    from codai.api.characters import _list_characters
    return {"characters": _list_characters()}


@router.get("/admin/api/characters/{name}")
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


@router.get("/admin/api/characters/{name}/thumbnail")
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


@router.delete("/admin/api/characters/{name}")
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

@router.get("/admin/api/environments")
async def api_list_environments(username: str = Depends(require_auth)):
    from codai.api.environments import _list_environments
    return {"environments": _list_environments()}


@router.get("/admin/api/environments/{name}")
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


@router.get("/admin/api/environments/{name}/thumbnail")
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


@router.delete("/admin/api/environments/{name}")
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

@router.get("/admin/api/voices")
async def api_list_voices(username: str = Depends(require_auth)):
    from codai.api.voice_clone import _list_voices
    return {"voices": _list_voices()}


@router.get("/admin/api/voices/{name}")
async def api_get_voice(name: str, username: str = Depends(require_auth)):
    from codai.api.voice_clone import _load_voice
    meta = _load_voice(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    return {"voice": meta}


@router.delete("/admin/api/voices/{name}")
async def api_delete_voice(name: str, username: str = Depends(require_auth)):
    import os as _os, shutil
    from codai.api.voice_clone import _voice_path
    vdir = _voice_path(name)
    if not _os.path.exists(vdir):
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    shutil.rmtree(vdir)
    return {"deleted": True, "name": name}


@router.get("/admin/api/hf-model-info")
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
