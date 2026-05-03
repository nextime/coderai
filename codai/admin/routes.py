"""Admin dashboard routes."""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Response, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from codai.admin.auth import SessionManager


router = APIRouter()

# Templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Session manager (will be initialized in main.py)
session_manager: Optional[SessionManager] = None


def init_session_manager(config_dir: Path):
    """Initialize the session manager."""
    global session_manager
    session_manager = SessionManager(config_dir)


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
    
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": None
    })


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
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid username or password"
        })
    
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
        max_age=7200  # 2 hours
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
    """Display password change page."""
    user = session_manager.get_user(username)
    must_change = user.get("must_change_password", False) if user else False
    
    return templates.TemplateResponse("change_password.html", {
        "request": request,
        "username": username,
        "must_change": must_change,
        "error": None
    })


@router.post("/admin/change-password")
async def change_password(
    request: Request,
    old_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    username: str = Depends(require_auth)
):
    """Handle password change."""
    if new_password != confirm_password:
        return templates.TemplateResponse("change_password.html", {
            "request": request,
            "username": username,
            "must_change": False,
            "error": "Passwords do not match"
        })
    
    if len(new_password) < 8:
        return templates.TemplateResponse("change_password.html", {
            "request": request,
            "username": username,
            "must_change": False,
            "error": "Password must be at least 8 characters"
        })
    
    # Check if this is a forced change (first login)
    user = session_manager.get_user(username)
    if user and user.get("must_change_password"):
        # Force change without verifying old password
        success = session_manager.force_password_change(username, new_password)
    else:
        success = session_manager.change_password(username, old_password, new_password)
    
    if not success:
        return templates.TemplateResponse("change_password.html", {
            "request": request,
            "username": username,
            "must_change": False,
            "error": "Current password is incorrect"
        })
    
    return RedirectResponse(url="/admin", status_code=302)


@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, username: str = Depends(require_auth)):
    """Display admin dashboard."""
    is_admin = session_manager.is_admin(username)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "username": username,
        "is_admin": is_admin
    })


@router.get("/admin/models", response_class=HTMLResponse)
async def models_page(request: Request, username: str = Depends(require_admin)):
    """Display models management page."""
    return templates.TemplateResponse("models.html", {
        "request": request,
        "username": username
    })


@router.get("/admin/tokens", response_class=HTMLResponse)
async def tokens_page(request: Request, username: str = Depends(require_admin)):
    """Display API tokens management page."""
    return templates.TemplateResponse("tokens.html", {
        "request": request,
        "username": username
    })


@router.get("/admin/users", response_class=HTMLResponse)
async def users_page(request: Request, username: str = Depends(require_admin)):
    """Display users management page."""
    users = session_manager.list_users()
    
    return templates.TemplateResponse("users.html", {
        "request": request,
        "username": username,
        "users": users
    })


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, username: str = Depends(require_auth)):
    """Display chat interface."""
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "username": username
    })


# API endpoints for admin operations
@router.get("/admin/api/status")
async def api_status(username: str = Depends(require_auth)):
    """Get system status."""
    # TODO: Implement actual status gathering
    return {
        "status": "ok",
        "backend": "auto",
        "models_loaded": 0,
        "uptime": "0h 0m"
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
    # Find user by ID
    users = session_manager._load_auth_data().get("users", [])
    user = next((u for u in users if u["id"] == user_id), None)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    success = session_manager.delete_user(user["username"])
    if not success:
        raise HTTPException(status_code=400, detail="Cannot delete user")
    
    return {"success": True}
