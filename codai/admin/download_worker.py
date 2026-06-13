# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Out-of-process model downloader.

Model downloads (``snapshot_download`` / ``hf_hub_download``) cannot be cancelled
from the calling thread: huggingface_hub fetches each file with several parallel
chunk connections (and, with Xet, an entirely separate transfer path), none of
which honour an in-thread "please stop" flag — so a daemon thread running the
download is effectively un-stoppable. Running the download in a *child process*
makes cancellation reliable: the supervisor simply ``terminate()``s the process,
which tears down every chunk connection at once.

This module is deliberately self-contained (stdlib + huggingface_hub +
codai.models.cache only, NO FastAPI / torch imports) so it loads fast and safely
under the ``spawn`` start method.
"""

import os
import time

_DISK_MIN_FREE_BYTES = 256 * 1024 * 1024  # 256 MB safety margin


def _check_disk_space(path: str, needed_bytes: int = 0) -> None:
    """Raise RuntimeError if `path`'s filesystem lacks enough free space."""
    import os as _os
    import shutil
    check_path = path
    while check_path and not _os.path.exists(check_path):
        parent = _os.path.dirname(check_path)
        if parent == check_path:
            break
        check_path = parent
    try:
        free = shutil.disk_usage(check_path).free
    except OSError:
        return
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


def _make_tqdm_class(q, cache_dir=None):
    """tqdm-compatible class that forwards progress events to the multiprocessing
    queue `q`. Cancellation is handled by the parent terminating this process, so
    no in-band cancel flag is needed here."""
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
                q.put({"type": "start", "filename": self.desc, "total": self.total})

        def update(self, n=1):
            self.n += n
            self._update_count += 1
            if cache_dir and self._update_count % 64 == 0:
                _check_disk_space(cache_dir)
            elapsed = (_time.time() - self._start) or 0.001
            rate = self.n / elapsed
            eta = (self.total - self.n) / rate if rate and self.total else None
            pct = round(self.n / self.total * 100, 1) if self.total else 0
            q.put({
                "type": "progress",
                "filename": self.desc,
                "downloaded": self.n,
                "total": self.total,
                "percent": pct,
                "rate": round(rate),
                "eta": round(eta) if eta is not None else None,
            })

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
            q.put({"type": "info", "message": str(s)})

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


def run_download(model_id: str, file_pattern: str, q) -> None:
    """Child-process entry point: download `model_id` (optionally filtered by
    `file_pattern`) and stream progress events into the multiprocessing queue `q`.
    Terminating this process cancels the download cleanly."""
    try:
        from codai.models.cache import (
            is_huggingface_model_id, get_model_cache_dir, get_hf_hub_cache_dir,
        )
        from huggingface_hub import snapshot_download

        if is_huggingface_model_id(model_id):
            is_gguf_download = file_pattern and '.gguf' in file_pattern.lower()
            if is_gguf_download:
                gguf_cache = get_model_cache_dir()
                dl_cache_dir = gguf_cache
            else:
                dl_cache_dir = get_hf_hub_cache_dir()

            expected_bytes = _get_hf_expected_size(model_id, file_pattern)
            _check_disk_space(dl_cache_dir, expected_bytes)

            tqdm_cls = _make_tqdm_class(q, cache_dir=dl_cache_dir)

            if is_gguf_download:
                import fnmatch as _fnmatch
                import shutil as _shutil
                from huggingface_hub import list_repo_files, hf_hub_download

                _is_exact = ('*' not in file_pattern and '?' not in file_pattern
                             and file_pattern.lower().endswith('.gguf'))
                if _is_exact:
                    matching = [file_pattern]
                else:
                    if file_pattern.startswith('.'):
                        pat = f"*{file_pattern}"
                    elif '/' in file_pattern:
                        pat = file_pattern
                    else:
                        pat = f"*{file_pattern}"
                    all_repo_files = list(list_repo_files(model_id))
                    matching = [
                        f for f in all_repo_files
                        if _fnmatch.fnmatch(f, pat) or _fnmatch.fnmatch(os.path.basename(f), pat)
                    ]
                    if not matching:
                        q.put({"type": "error", "message": f"No files matching {file_pattern!r} found in {model_id}"})
                        return

                last_dest = gguf_cache
                for hf_filename in matching:
                    basename = os.path.basename(hf_filename)
                    q.put({"type": "info", "message": f"Downloading {basename} from {model_id}…"})
                    dl_path = hf_hub_download(
                        repo_id=model_id,
                        filename=hf_filename,
                        local_dir=gguf_cache,
                        tqdm_class=tqdm_cls,
                    )
                    flat_dest = os.path.join(gguf_cache, basename)
                    if os.path.abspath(dl_path) != os.path.abspath(flat_dest) and os.path.isfile(dl_path):
                        _shutil.move(dl_path, flat_dest)
                    last_dest = flat_dest
                path = last_dest

            elif file_pattern:
                if file_pattern.startswith('.'):
                    allow = [f"*{file_pattern}"]
                elif '/' in file_pattern:
                    allow = [file_pattern]
                else:
                    allow = [f"*{file_pattern}"]
                q.put({"type": "info", "message": f"Downloading {allow[0]} from {model_id}…"})
                path = snapshot_download(model_id, cache_dir=dl_cache_dir, allow_patterns=allow, tqdm_class=tqdm_cls)
            else:
                q.put({"type": "info", "message": f"Downloading full repository {model_id}…"})
                path = snapshot_download(model_id, cache_dir=dl_cache_dir, tqdm_class=tqdm_cls)

        else:
            # Direct URL download (non-HF source)
            import requests as _req
            import hashlib

            dl_cache_dir = get_model_cache_dir()
            _check_disk_space(dl_cache_dir)

            url_path = model_id.split('?')[0]
            filename = os.path.basename(url_path) or "model.bin"
            url_hash = hashlib.sha256(model_id.encode()).hexdigest()
            dest = os.path.join(dl_cache_dir, f"{url_hash}_{filename}")

            if os.path.exists(dest):
                q.put({"type": "done", "path": dest})
                return

            resp = _req.get(model_id, stream=True, timeout=60, allow_redirects=True)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            if total:
                _check_disk_space(dl_cache_dir, total)
            q.put({"type": "start", "filename": filename, "total": total})

            downloaded = 0
            start_t = time.time()
            last_evt = 0.0
            with open(dest, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=524288):
                    if chunk:
                        if downloaded % (64 * 1024 * 1024) < len(chunk):
                            _check_disk_space(dl_cache_dir)
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_evt >= 0.25:
                            last_evt = now
                            elapsed = (now - start_t) or 0.001
                            rate = downloaded / elapsed
                            eta = (total - downloaded) / rate if rate and total else None
                            q.put({
                                "type": "progress", "filename": filename,
                                "downloaded": downloaded, "total": total,
                                "percent": round(downloaded / total * 100, 1) if total else 0,
                                "rate": round(rate),
                                "eta": round(eta) if eta is not None else None,
                            })
            path = dest

        q.put({"type": "done", "path": str(path)})

    except Exception as exc:
        q.put({"type": "error", "message": str(exc)})


class _StdoutQueue:
    """Queue-compatible shim that emits each event as one JSON line on stdout.

    Lets ``run_download`` (written against a ``.put(evt)`` queue) drive the CLI
    entry point unchanged: the parent reads these lines back and relays them onto
    the SSE stream."""

    def put(self, evt):
        import sys as _sys
        import json as _json
        try:
            _sys.stdout.write(_json.dumps(evt) + "\n")
            _sys.stdout.flush()
        except Exception:
            pass


def main(argv=None):
    """CLI entry point: ``python -m codai.admin.download_worker <model_id> [pattern]``.

    Run as a *subprocess* (not multiprocessing) so the child's ``__main__`` is this
    module — not the server's launcher — avoiding a costly/hanging re-import of the
    whole server under the spawn start method. Cancellation = terminating this
    process, which tears down every HF chunk connection at once."""
    import argparse
    ap = argparse.ArgumentParser(prog="codai.admin.download_worker")
    ap.add_argument("model_id")
    ap.add_argument("file_pattern", nargs="?", default="")
    args = ap.parse_args(argv)
    run_download(args.model_id, args.file_pattern or "", _StdoutQueue())


if __name__ == "__main__":
    main()
