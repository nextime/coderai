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

"""Authentication and session management for admin dashboard."""
import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import UTC, datetime, timedelta


SECRET_KEY_FILE = "secret_key"


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(UTC)


def parse_session_timestamp(timestamp: str) -> datetime:
    """Parse persisted session timestamps, normalizing legacy naive values to UTC."""
    parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


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
        try:
            secret_path.chmod(0o600)
        except OSError:
            pass
        return secret


def hash_password(password: str) -> str:
    """Hash a password using argon2 (preferred) or scrypt as fallback.

    New hashes are always produced with a proper key-derivation function and
    a per-password random salt.  The legacy SHA-256/static-salt format is
    only retained for *verification* of pre-existing hashes.
    """
    try:
        from argon2 import PasswordHasher
        ph = PasswordHasher()
        return ph.hash(password)
    except ImportError:
        pass
    # scrypt fallback: encode as "scrypt:<b64salt>:<b64key>"
    salt = os.urandom(16)
    key = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return "scrypt:" + base64.b64encode(salt).decode() + ":" + base64.b64encode(key).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash.

    Supports argon2, scrypt (new format), and the legacy SHA-256/static-salt
    format so that old stored hashes continue to work.
    """
    # --- argon2 ---
    try:
        from argon2 import PasswordHasher
        from argon2.exceptions import VerifyMismatchError, InvalidHashError
        ph = PasswordHasher()
        try:
            return ph.verify(password_hash, password)
        except VerifyMismatchError:
            return False
        except InvalidHashError:
            pass  # not an argon2 hash; fall through
        except Exception:
            pass
    except ImportError:
        pass

    # --- scrypt ---
    if password_hash.startswith("scrypt:"):
        try:
            parts = password_hash.split(":")
            if len(parts) == 3:
                salt = base64.b64decode(parts[1])
                stored_key = base64.b64decode(parts[2])
                new_key = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
                return hmac.compare_digest(new_key, stored_key)
        except Exception:
            pass
        return False

    # --- legacy SHA-256 with static salt (read-only; never written for new passwords) ---
    legacy = hashlib.sha256(b'static_salt_' + password.encode()).hexdigest()
    return hmac.compare_digest(legacy, password_hash)


class SessionManager:
    """Manages user sessions."""
    
    def __init__(self, config_dir: Path, session_timeout_days: int = 30):
        self.config_dir = config_dir
        self.secret = get_or_create_secret(config_dir)
        self.session_timeout = timedelta(days=session_timeout_days)
        self._lock = threading.Lock()
    
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
    
    def _write_auth_data(self, auth_data: Dict[str, Any]):
        """Write auth.json to disk atomically. Caller MUST hold ``self._lock``."""
        auth_path = self.config_dir / "auth.json"
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

    def _save_auth_data(self, auth_data: Dict[str, Any]):
        """Atomically persist a full auth_data snapshot (acquires the lock).

        Prefer :meth:`update_auth_data` for read-modify-write sequences: a bare
        load + mutate + save is not atomic and concurrent writers can clobber
        each other's changes.
        """
        with self._lock:
            self._write_auth_data(auth_data)

    def update_auth_data(self, mutator):
        """Atomic read-modify-write of auth.json under the instance lock.

        ``mutator(auth_data)`` receives the freshly loaded dict, mutates it in
        place, and returns ``(should_save, result)``. The file is rewritten only
        when ``should_save`` is truthy; ``result`` is returned to the caller.
        Holding the lock across the whole load → mutate → write cycle serializes
        concurrent writers so updates can't lose each other's changes.
        """
        with self._lock:
            auth_data = self._load_auth_data()
            should_save, result = mutator(auth_data)
            if should_save:
                self._write_auth_data(auth_data)
            return result
    
    def create_session(self, username: str) -> str:
        """Create a new session for a user.
        
        Returns:
            Session ID cookie value
        """
        session_id = secrets.token_urlsafe(32)
        expires_at = utc_now() + self.session_timeout

        def _mut(auth_data):
            sessions = auth_data.setdefault("sessions", {})
            sessions[session_id] = {
                "username": username,
                "created_at": utc_now().isoformat(),
                "expires_at": expires_at.isoformat(),
            }
            return True, None
        self.update_auth_data(_mut)

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

        # Look up, expire-check, and extend the session atomically so a
        # concurrent login or token write can't clobber the sessions map.
        def _mut(auth_data):
            sessions = auth_data.get("sessions", {})
            session = sessions.get(session_id)
            if not session:
                return False, None
            expires_at = parse_session_timestamp(session["expires_at"])
            if utc_now() > expires_at:
                del sessions[session_id]          # clean up expired session
                return True, None
            # Extend session (sliding expiration)
            session["expires_at"] = (utc_now() + self.session_timeout).isoformat()
            return True, session["username"]
        return self.update_auth_data(_mut)
    
    def destroy_session(self, cookie_value: Optional[str]):
        """Destroy a session."""
        if not cookie_value:
            return
        
        try:
            session_id, _ = cookie_value.rsplit('.', 1)
        except ValueError:
            return

        def _mut(auth_data):
            sessions = auth_data.get("sessions", {})
            if session_id in sessions:
                del sessions[session_id]
                return True, None
            return False, None
        self.update_auth_data(_mut)
    
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
        def _mut(auth_data):
            for user in auth_data.get("users", []):
                if user["username"] == username:
                    if not verify_password(old_password, user["password_hash"]):
                        return False, False
                    user["password_hash"] = hash_password(new_password)
                    user["must_change_password"] = False
                    return True, True
            return False, False
        return self.update_auth_data(_mut)

    def force_password_change(self, username: str, new_password: str) -> bool:
        """Force a user to change password (e.g., on first login).

        Returns:
            True if successful, False otherwise
        """
        def _mut(auth_data):
            for user in auth_data.get("users", []):
                if user["username"] == username:
                    user["password_hash"] = hash_password(new_password)
                    user["must_change_password"] = False
                    user["last_changed_at"] = utc_now().isoformat()
                    return True, True
            return False, False
        return self.update_auth_data(_mut)
    
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
        def _mut(auth_data):
            # Check if user already exists
            for user in auth_data.get("users", []):
                if user["username"] == username:
                    return False, False
            # Find next available ID
            max_id = max([u["id"] for u in auth_data.get("users", [])], default=0)
            new_user = {
                "id": max_id + 1,
                "username": username,
                "password_hash": hash_password(password),
                "role": role,
                "created_at": utc_now().isoformat(),
                "must_change_password": False
            }
            auth_data.setdefault("users", []).append(new_user)
            return True, True
        return self.update_auth_data(_mut)
    
    def verify_token(self, token: str) -> bool:
        """Verify an API bearer token."""
        if not token:
            return False
        auth_data = self._load_auth_data()
        for t in auth_data.get("tokens", []):
            stored = t.get("token", "")
            if stored and hmac.compare_digest(stored, token):
                return True
        return False

    def delete_user(self, username: str) -> bool:
        """Delete a user.
        
        Returns:
            True if successful, False if user doesn't exist or is last admin
        """
        def _mut(auth_data):
            users = auth_data.get("users", [])
            # Check if last admin
            admin_count = sum(1 for u in users if u.get("role") == "admin")
            user = next((u for u in users if u["username"] == username), None)
            if not user:
                return False, False
            if user.get("role") == "admin" and admin_count <= 1:
                return False, False  # Can't delete last admin
            # Remove user
            auth_data["users"] = [u for u in users if u["username"] != username]
            # Remove user's sessions
            sessions = auth_data.get("sessions", {})
            auth_data["sessions"] = {
                k: v for k, v in sessions.items()
                if v["username"] != username
            }
            return True, True
        return self.update_auth_data(_mut)
