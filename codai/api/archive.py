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

"""Generation archive — persists all generated artifacts with metadata on disk."""

import io
import json
import logging
import os
import shutil
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

RETENTION_OPTIONS = ["1h", "1d", "2d", "1w", "1m", "3m", "6m", "1y", "never"]

_RETENTION_SECONDS: Dict[str, Optional[int]] = {
    "1h":       3_600,
    "1d":      86_400,
    "2d":     172_800,
    "1w":     604_800,
    "1m":   2_592_000,   # 30 days
    "3m":   7_776_000,   # 90 days
    "6m":  15_552_000,   # 180 days
    "1y":  31_536_000,   # 365 days
    "never":      None,
}


def _safe_id(s: str) -> bool:
    return bool(s) and "/" not in s and ".." not in s and "\\" not in s


def _safe_filename(s: str) -> bool:
    return bool(s) and "/" not in s and ".." not in s and "\\" not in s


class ArchiveManager:
    """Manages the on-disk generation archive."""

    def __init__(self):
        self.enabled: bool = True
        self.directory: str = os.path.abspath("./archive")
        self.retention: str = "never"
        self._lock = threading.Lock()

    def configure(self, enabled: bool, directory: str, retention: str):
        self.enabled = enabled
        self.directory = os.path.abspath(directory or "./archive")
        self.retention = retention if retention in _RETENTION_SECONDS else "never"

    def _gen_dir(self, gen_id: str) -> str:
        return os.path.join(self.directory, gen_id)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_generation(
        self,
        gen_type: str,
        endpoint: str,
        model: Optional[str],
        prompt: Optional[str],
        params: Dict[str, Any],
        artifacts: List[Tuple[bytes, str]],  # [(raw_bytes, file_ext), ...]
    ) -> Optional[str]:
        """Save a generation to archive.  Returns gen_id on success, None otherwise."""
        if not self.enabled:
            return None
        try:
            gen_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
            gen_dir = self._gen_dir(gen_id)
            os.makedirs(gen_dir, exist_ok=True)

            saved_files: List[str] = []
            preview_file: Optional[str] = None

            for i, (data, ext) in enumerate(artifacts):
                if not data:
                    continue
                fname = f"output_{i}.{ext}"
                with open(os.path.join(gen_dir, fname), "wb") as fh:
                    fh.write(data)
                saved_files.append(fname)

                if i == 0 and preview_file is None:
                    try:
                        preview_file = self._make_preview(data, ext, gen_dir)
                    except Exception as e:
                        _log.debug("Archive preview failed: %s", e)

            meta: Dict[str, Any] = {
                "id": gen_id,
                "type": gen_type,
                "endpoint": endpoint,
                "model": model or "",
                "prompt": prompt or "",
                "parameters": params,
                "files": saved_files,
                "preview": preview_file or (saved_files[0] if saved_files else None),
                "created_at": int(time.time()),
                "created_at_iso": datetime.utcnow().isoformat() + "Z",
            }
            with open(os.path.join(gen_dir, "metadata.json"), "w") as fh:
                json.dump(meta, fh, indent=2)

            return gen_id
        except Exception as e:
            _log.warning("Archive save failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_entries(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Return metadata dicts, newest first."""
        if not os.path.isdir(self.directory):
            return []
        entries: List[Dict] = []
        try:
            for name in os.listdir(self.directory):
                meta_path = os.path.join(self.directory, name, "metadata.json")
                if not os.path.isfile(meta_path):
                    continue
                try:
                    with open(meta_path) as fh:
                        entries.append(json.load(fh))
                except Exception:
                    pass
        except Exception:
            pass
        entries.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return entries[offset: offset + limit]

    def count_entries(self) -> int:
        if not os.path.isdir(self.directory):
            return 0
        count = 0
        try:
            for name in os.listdir(self.directory):
                if os.path.isfile(os.path.join(self.directory, name, "metadata.json")):
                    count += 1
        except Exception:
            pass
        return count

    def get_entry(self, gen_id: str) -> Optional[Dict]:
        if not _safe_id(gen_id):
            return None
        meta_path = os.path.join(self._gen_dir(gen_id), "metadata.json")
        if not os.path.isfile(meta_path):
            return None
        try:
            with open(meta_path) as fh:
                return json.load(fh)
        except Exception:
            return None

    def delete_entry(self, gen_id: str) -> bool:
        if not _safe_id(gen_id):
            return False
        gen_dir = self._gen_dir(gen_id)
        if not os.path.isdir(gen_dir):
            return False
        try:
            shutil.rmtree(gen_dir)
            return True
        except Exception:
            return False

    def get_file_path(self, gen_id: str, filename: str) -> Optional[str]:
        if not _safe_id(gen_id) or not _safe_filename(filename):
            return None
        path = os.path.join(self._gen_dir(gen_id), filename)
        return path if os.path.isfile(path) else None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def run_cleanup(self):
        """Remove entries older than the configured retention window."""
        max_age = _RETENTION_SECONDS.get(self.retention)
        if max_age is None:
            return
        if not os.path.isdir(self.directory):
            return
        cutoff = time.time() - max_age
        deleted = 0
        try:
            for name in list(os.listdir(self.directory)):
                meta_path = os.path.join(self.directory, name, "metadata.json")
                try:
                    with open(meta_path) as fh:
                        meta = json.load(fh)
                    if meta.get("created_at", time.time()) < cutoff:
                        shutil.rmtree(os.path.join(self.directory, name), ignore_errors=True)
                        deleted += 1
                except Exception:
                    pass
        except Exception:
            pass
        if deleted:
            _log.info("Archive cleanup: removed %d entry/entries", deleted)

    # ------------------------------------------------------------------
    # Preview generation
    # ------------------------------------------------------------------

    def _make_preview(self, data: bytes, ext: str, gen_dir: str) -> Optional[str]:
        ext = ext.lower()
        if ext in ("png", "jpg", "jpeg", "webp", "bmp", "gif"):
            from PIL import Image
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.thumbnail((512, 512))
            dst = os.path.join(gen_dir, "preview.webp")
            img.save(dst, "WEBP", quality=75)
            return "preview.webp"

        if ext in ("mp4", "webm", "avi", "mov"):
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            frame_path = os.path.join(gen_dir, "_frame_tmp.jpg")
            try:
                r = subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_path, "-vframes", "1", "-q:v", "3", frame_path],
                    capture_output=True, timeout=15,
                )
                if r.returncode == 0 and os.path.exists(frame_path):
                    from PIL import Image
                    img = Image.open(frame_path).convert("RGB")
                    img.thumbnail((512, 512))
                    dst = os.path.join(gen_dir, "preview.webp")
                    img.save(dst, "WEBP", quality=75)
                    return "preview.webp"
            except Exception:
                pass
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                try:
                    os.unlink(frame_path)
                except Exception:
                    pass

        # Audio or unrecognised — no visual preview
        return None


# Module-level singleton
archive_manager = ArchiveManager()
