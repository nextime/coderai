# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Client for a colibri/k3 engine hosted by tools/mux_service.py.

colibri and k3 are driven over a stdin/stdout mux protocol, so unlike every
other worker they had no URL to redirect. :class:`RemoteMuxEngine` supplies the
missing half: it speaks HTTP to a remote ``mux_service`` while presenting the
exact surface the backends use on a local ``MuxEngine`` — ``run``, ``is_alive``,
``pause``, ``resume``, ``close``, ``log_tail`` and ``family`` — so
``colibri_worker.ensure_engine()`` can hand one back and neither backend nor
model manager can tell the difference.

Cancellation travels as a dropped connection: when ``cancelled()`` goes true we
stop reading and close the response, and the service sees its write fail and
aborts generation. That is the same contract the local engine has.
"""

import json
from typing import Callable, Optional


class RemoteMuxEngine:
    """A colibri/k3 engine running behind tools/mux_service.py."""

    def __init__(self, url: str, kind: str = "colibri", family: str = "",
                 timeout: float = 3600.0):
        self.url = (url or "").rstrip("/")
        self.kind = kind
        self.family = family or "glm"
        self.timeout = float(timeout)
        self._last_error = ""

    # ---- lifecycle -------------------------------------------------------
    def is_alive(self) -> bool:
        info = self.health()
        return bool(info and info.get("alive"))

    def health(self) -> Optional[dict]:
        import requests
        try:
            r = requests.get(self.url + "/health", timeout=10)
            return r.json() if r.ok else None
        except Exception as exc:
            self._last_error = f"health check failed: {exc}"
            return None

    def log_tail(self) -> str:
        return self._last_error

    def close(self) -> None:
        """A remote engine outlives this process — never kill what we don't own."""
        return None

    def pause(self):
        return self._post_simple("/pause")

    def resume(self):
        return self._post_simple("/resume")

    def _post_simple(self, path: str):
        import requests
        try:
            requests.post(self.url + path, timeout=120)
        except Exception as exc:
            self._last_error = f"{path} failed: {exc}"

    # ---- generation ------------------------------------------------------
    def run(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
            on_text: Callable[[str], None],
            cancelled: Optional[Callable[[], bool]] = None) -> dict:
        """Submit one rendered prompt; stream text to ``on_text``; return stats."""
        import requests
        payload = {"prompt": prompt, "max_tokens": int(max_tokens),
                   "temperature": float(temperature), "top_p": float(top_p)}
        stats: dict = {}
        emitted = 0
        try:
            resp = requests.post(self.url + "/run", json=payload,
                                 stream=True, timeout=(30, self.timeout))
        except Exception as exc:
            self._last_error = str(exc)
            raise RuntimeError(f"remote {self.kind} engine at {self.url} is unreachable: {exc}")
        if not resp.ok:
            detail = ""
            try:
                detail = resp.json().get("error", "")
            except Exception:
                detail = (resp.text or "")[:400]
            self._last_error = detail
            raise RuntimeError(f"remote {self.kind} engine returned HTTP {resp.status_code}"
                               + (f": {detail}" if detail else ""))
        try:
            for line in resp.iter_lines(decode_unicode=False):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if "t" in obj:
                    on_text(obj["t"])
                    emitted += 1
                    # Closing the response is how the far side learns to stop.
                    if cancelled is not None and cancelled():
                        break
                elif "stats" in obj:
                    stats = obj["stats"] or {}
                elif "error" in obj:
                    self._last_error = str(obj["error"])
                    raise RuntimeError(f"remote {self.kind} engine: {obj['error']}")
        finally:
            try:
                resp.close()
            except Exception:
                pass
        if not stats:
            stats = {"completion_tokens": emitted, "prompt_tokens": 0,
                     "total_tokens": emitted, "tokens_per_second": 0.0,
                     "cache_hit_percent": 0.0, "rss_gb": 0.0, "length_limited": False}
        return stats


def remote_engine_for(kind: str, url: str, family: str = "") -> RemoteMuxEngine:
    """Build a client and prove the far side is answering before returning it."""
    eng = RemoteMuxEngine(url, kind=kind, family=family)
    info = eng.health()
    if not info:
        raise RuntimeError(
            f"configured service_url {eng.url} is not answering its health check")
    if not info.get("alive"):
        raise RuntimeError(
            f"remote {kind} service at {eng.url} is up but its engine is not running")
    eng.family = family or info.get("family") or eng.family
    return eng
