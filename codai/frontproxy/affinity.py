# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Prefix-cache-aware routing across engines and nodes.

When a model is resident in more than one place — two local engines, a
local engine and a cluster node, replicas after a fan-out — the follow-up
turns of a conversation are worth sending where its earlier turns went:
that engine's KV / prefix cache (the multi-slot cache of the GGUF and HF
backends, vLLM's prefix caching) already holds the shared opening, and the
prompt is processed from the first new token instead of from scratch.

The key is the same one the engines use for their own slot affinity
(codai/api/text.py ``_conversation_session_key``): an explicit
``X-Session-Id`` header or OpenAI ``user`` field, else a hash of the
conversation's stable opening (system prompt + first user turn, or the
prompt head). The map is key → engine name with a TTL; entries are cheap
and bounded. A preference is only that: the router still requires the
engine to be alive, capable and to hold the model.
"""

import hashlib
import json
import threading
import time
from typing import Optional

_AFFINE_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/responses")


def conversation_key(path: str, body: Optional[bytes], headers=None) -> Optional[str]:
    """The affinity key of a request, or None when there is nothing stable to
    hang it on (non-text paths, empty bodies)."""
    p = (path or "").split("?", 1)[0].rstrip("/")
    if p not in _AFFINE_PATHS or not body:
        return None
    try:
        sid = ""
        if headers is not None:
            try:
                sid = headers.get("x-session-id") or ""
            except Exception:
                sid = ""
        if sid:
            return f"sid:{sid}"
        d = json.loads(body)
        if not isinstance(d, dict):
            return None
        uid = d.get("user")
        if uid:
            return f"user:{uid}"
        parts = []
        msgs = d.get("messages")
        if isinstance(msgs, list) and msgs:
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role, content = m.get("role"), m.get("content")
                if not isinstance(content, str):
                    content = json.dumps(content, sort_keys=True) if content is not None else ""
                if role == "system":
                    parts.append("system:" + content)
                elif role == "user":
                    parts.append("user:" + content)
                    break
        else:
            prompt = d.get("prompt") if "prompt" in d else d.get("input")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            if prompt:
                parts.append(str(prompt)[:1024])
        if not parts:
            return None
        return "hash:" + hashlib.sha256("\n".join(parts).encode("utf-8", "ignore")).hexdigest()[:16]
    except Exception:
        return None


class PrefixAffinity:
    """key → engine name, remembered for ``ttl_s`` after its last use."""

    def __init__(self, ttl_s: float = 1800.0, max_entries: int = 20000):
        self.ttl_s = float(ttl_s)
        self.max_entries = int(max_entries)
        self._m: dict = {}
        self._lock = threading.Lock()

    def get(self, key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        now = time.time()
        with self._lock:
            ent = self._m.get(key)
            if ent is None:
                return None
            name, ts = ent
            if self.ttl_s > 0 and now - ts > self.ttl_s:
                self._m.pop(key, None)
                return None
            return name

    def remember(self, key: Optional[str], engine_name: Optional[str]) -> None:
        if not key or not engine_name:
            return
        now = time.time()
        with self._lock:
            self._m[key] = (engine_name, now)
            if len(self._m) > self.max_entries:
                # Drop the oldest tenth; a scan this rare is cheaper than an LRU list.
                for k, _ in sorted(self._m.items(), key=lambda kv: kv[1][1])[: self.max_entries // 10]:
                    self._m.pop(k, None)

    def forget_engine(self, engine_name: str) -> None:
        with self._lock:
            for k in [k for k, (n, _) in self._m.items() if n == engine_name]:
                self._m.pop(k, None)

    def __len__(self) -> int:
        return len(self._m)
