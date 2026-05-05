"""Authentication and session management for admin dashboard."""
import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta


SECRET_KEY_FILE = "secret_key"


def get_or_create_secret(config_dir: Path) -> bytes:
    """Get or create a secret key for session signing."""
    secret_path = config_dir / SECRET_KEY_FILE
    if secret_path.exists():
        with open(secret_path, 'rb') as f:
            return f.read()
    else:
        secret = secrets.token_bytes(32)
        with open(secret_path, 'wb') as f:
            f.write(secret)
        secret_path.chmod(0o600)
        return secret


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt.
    
    In production, use argon2 or bcrypt. This is a minimal implementation
    for environments where those libraries aren't available.
    """
    # Use SHA-256 with a pepper-like secret for basic hashing
    # Real implementation should use argon2 from main.py
    salt = b'static_salt_'  # In production, use per-user random salt
    return hashlib.sha256(salt + password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    # Try argon2 first
    try:
        from argon2 import PasswordHasher
        from argon2.exceptions import VerifyMismatchError
        ph = PasswordHasher()
        try:
            return ph.verify(password_hash, password)
        except VerifyMismatchError:
            return False
        except Exception:
            pass
    except ImportError:
        pass
    
    # Fallback to simple hash
    return hash_password(password) == password_hash


class SessionManager:
    """Manages user sessions."""
    
    def __init__(self, config_dir: Path, session_timeout_minutes: int = 120):
        self.config_dir = config_dir
        self.secret = get_or_create_secret(config_dir)
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self._lock = __import__('threading').Lock()
    
    def _load_auth_data(self) -> Dict[str, Any]:
        """Load auth.json data."""
        auth_path = self.config_dir / "auth.json"
        if auth_path.exists():
            try:
                with open(auth_path, 'r') as f:
                    content = f.read()
                if content.strip():
                    return json.loads(content)
            except (json.JSONDecodeError, OSError):
                pass
        return {"users": [], "tokens": [], "sessions": {}}
    
    def _save_auth_data(self, auth_data: Dict[str, Any]):
        """Save auth.json data atomically."""
        auth_path = self.config_dir / "auth.json"
        with self._lock:
            import os, tempfile
            fd, tmp = tempfile.mkstemp(dir=str(self.config_dir), suffix='.tmp')
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(auth_data, f, indent=2)
                os.replace(tmp, str(auth_path))
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
    
    def create_session(self, username: str) -> str:
        """Create a new session for a user.
        
        Returns:
            Session ID cookie value
        """
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + self.session_timeout
        
        auth_data = self._load_auth_data()
        
        # Update sessions dict
        sessions = auth_data.get("sessions", {})
        sessions[session_id] = {
            "username": username,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat()
        }
        auth_data["sessions"] = sessions
        self._save_auth_data(auth_data)
        
        # Create signed cookie value: session_id.signature
        message = session_id.encode()
        signature = hmac.new(self.secret, message, hashlib.sha256).hexdigest()
        return f"{session_id}.{signature}"
    
    def validate_session(self, cookie_value: Optional[str]) -> Optional[str]:
        """Validate a session cookie.
        
        Returns:
            Username if valid, None otherwise
        """
        if not cookie_value:
            return None
        
        try:
            session_id, signature = cookie_value.rsplit('.', 1)
        except ValueError:
            return None
        
        # Verify signature
        message = session_id.encode()
        expected_sig = hmac.new(self.secret, message, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None
        
        # Check session in storage
        auth_data = self._load_auth_data()
        sessions = auth_data.get("sessions", {})
        session = sessions.get(session_id)
        
        if not session:
            return None
        
        # Check expiration
        expires_at = datetime.fromisoformat(session["expires_at"].replace('Z', '+00:00'))
        if datetime.utcnow() > expires_at:
            # Clean up expired session
            del sessions[session_id]
            auth_data["sessions"] = sessions
            self._save_auth_data(auth_data)
            return None
        
        # Extend session (sliding expiration)
        new_expires = datetime.utcnow() + self.session_timeout
        session["expires_at"] = new_expires.isoformat()
        auth_data["sessions"] = sessions
        self._save_auth_data(auth_data)
        
        return session["username"]
    
    def destroy_session(self, cookie_value: Optional[str]):
        """Destroy a session."""
        if not cookie_value:
            return
        
        try:
            session_id, _ = cookie_value.rsplit('.', 1)
        except ValueError:
            return
        
        auth_data = self._load_auth_data()
        sessions = auth_data.get("sessions", {})
        if session_id in sessions:
            del sessions[session_id]
            auth_data["sessions"] = sessions
            self._save_auth_data(auth_data)
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user and create a session.
        
        Returns:
            Session cookie value if successful, None otherwise
        """
        auth_data = self._load_auth_data()
        
        for user in auth_data.get("users", []):
            if user["username"] == username:
                if verify_password(password, user["password_hash"]):
                    # Password is correct - check if must change
                    if user.get("must_change_password"):
                        # Set a special flag so we know to redirect
                        # We'll create a temp session that marks this
                        return self.create_session(username) + ".MUST_CHANGE"
                    return self.create_session(username)
        
        return None
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change a user's password.
        
        Returns:
            True if successful, False otherwise
        """
        auth_data = self._load_auth_data()
        
        for user in auth_data.get("users", []):
            if user["username"] == username:
                if not verify_password(old_password, user["password_hash"]):
                    return False
                
                user["password_hash"] = hash_password(new_password)
                user["must_change_password"] = False
                self._save_auth_data(auth_data)
                return True
        
        return False
    
    def force_password_change(self, username: str, new_password: str) -> bool:
        """Force a user to change password (e.g., on first login).
        
        Returns:
            True if successful, False otherwise
        """
        auth_data = self._load_auth_data()
        
        for user in auth_data.get("users", []):
            if user["username"] == username:
                user["password_hash"] = hash_password(new_password)
                user["must_change_password"] = False
                user["last_changed_at"] = datetime.utcnow().isoformat()
                self._save_auth_data(auth_data)
                return True
        
        return False
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user details."""
        auth_data = self._load_auth_data()
        for user in auth_data.get("users", []):
            if user["username"] == username:
                return user
        return None
    
    def is_admin(self, username: str) -> bool:
        """Check if a user is an admin."""
        user = self.get_user(username)
        return user is not None and user.get("role") == "admin"
    
    def list_users(self) -> list:
        """List all users (excluding password hashes)."""
        auth_data = self._load_auth_data()
        users = []
        for user in auth_data.get("users", []):
            users.append({
                "id": user["id"],
                "username": user["username"],
                "role": user["role"],
                "created_at": user["created_at"]
            })
        return users
    
    def create_user(self, username: str, password: str, role: str = "user") -> bool:
        """Create a new user.
        
        Returns:
            True if successful, False if user already exists
        """
        auth_data = self._load_auth_data()
        
        # Check if user already exists
        for user in auth_data.get("users", []):
            if user["username"] == username:
                return False
        
        # Find next available ID
        max_id = max([u["id"] for u in auth_data.get("users", [])], default=0)
        
        new_user = {
            "id": max_id + 1,
            "username": username,
            "password_hash": hash_password(password),
            "role": role,
            "created_at": datetime.utcnow().isoformat(),
            "must_change_password": False
        }
        
        auth_data["users"].append(new_user)
        self._save_auth_data(auth_data)
        return True
    
    def delete_user(self, username: str) -> bool:
        """Delete a user.
        
        Returns:
            True if successful, False if user doesn't exist or is last admin
        """
        auth_data = self._load_auth_data()
        users = auth_data.get("users", [])
        
        # Check if last admin
        admin_count = sum(1 for u in users if u.get("role") == "admin")
        user = next((u for u in users if u["username"] == username), None)
        
        if not user:
            return False
        
        if user.get("role") == "admin" and admin_count <= 1:
            return False  # Can't delete last admin
        
        # Remove user
        auth_data["users"] = [u for u in users if u["username"] != username]
        
        # Remove user's sessions
        sessions = auth_data.get("sessions", {})
        auth_data["sessions"] = {
            k: v for k, v in sessions.items()
            if v["username"] != username
        }
        
        self._save_auth_data(auth_data)
        return True
