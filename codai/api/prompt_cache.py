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

"""
Prompt prefix cache manager.

Provides two features:
1. Prefix key computation for same-prefix request scheduling (prompt aggregation).
2. Per-model last-prompt tracking so callers can report accurate cached_tokens.

llama.cpp's KV cache naturally reuses computation when consecutive requests
share a prompt prefix.  This manager helps exploit that by:
  - Giving the scheduler a stable key to group requests by shared prefix.
  - Letting text.py read back how many tokens were cached from timings.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional


@dataclass
class _CacheEntry:
    messages_hash: str
    prefix_hash: str
    token_count: int
    timestamp: float = field(default_factory=time.time)


class PromptCacheManager:
    """
    Tracks recently-processed prompt prefixes per model instance.

    Usage
    -----
    # Before dispatching to the model:
    prefix_key = manager.get_prefix_key(messages)   # for QueueManager scheduling

    # After the model call completes:
    manager.store(messages, model_key, prompt_tokens)

    # In the API response usage block:
    cached = manager.get_cached_tokens(model_key)    # from last store
    """

    def __init__(self, max_entries: int = 256, ttl_seconds: float = 600.0):
        self._entries: Dict[str, _CacheEntry] = {}
        self._by_model: Dict[str, str] = {}   # model_key -> last messages_hash
        self._cached_tokens: Dict[str, int] = {}  # model_key -> cached tokens from last call
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Hashing helpers
    # ------------------------------------------------------------------

    def _hash_messages(self, messages: List[Dict]) -> str:
        """Stable SHA-256 hash (truncated) of a message list."""
        canonical = json.dumps(
            [{"role": m.get("role"), "content": m.get("content")} for m in messages],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:20]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prefix_key(self, messages: List[Dict]) -> str:
        """
        Stable key for the *cacheable* portion of a request.

        The cacheable prefix is everything except the final user turn, since
        system prompts and prior assistant turns stay constant across related
        requests and benefit most from KV cache reuse.

        Returns an empty string when there is no cacheable prefix.
        """
        if not messages:
            return ""
        prefix = messages[:-1] if messages[-1].get("role") == "user" else messages
        return self._hash_messages(prefix) if prefix else ""

    def store(self, messages: List[Dict], model_key: str, prompt_tokens: int,
              cached_tokens: int = 0) -> None:
        """Record a completed prompt so future requests can match against it."""
        with self._lock:
            msg_hash = self._hash_messages(messages)
            prefix_hash = self.get_prefix_key(messages)
            self._entries[msg_hash] = _CacheEntry(
                messages_hash=msg_hash,
                prefix_hash=prefix_hash,
                token_count=prompt_tokens,
            )
            self._by_model[model_key] = msg_hash
            self._cached_tokens[model_key] = cached_tokens
            self._evict_locked()

    def get_cached_tokens(self, model_key: str) -> int:
        """Return the cached_tokens count stored by the last store() call for this model."""
        with self._lock:
            return self._cached_tokens.get(model_key, 0)

    def has_warm_prefix(self, messages: List[Dict], model_key: str) -> bool:
        """
        Return True if the current request shares a prefix with the last
        request processed by this model (i.e., the KV cache is likely warm).
        """
        with self._lock:
            last_hash = self._by_model.get(model_key)
            if not last_hash:
                return False
            entry = self._entries.get(last_hash)
            if not entry or time.time() - entry.timestamp > self._ttl:
                return False
            current_prefix = self.get_prefix_key(messages)
            return bool(current_prefix and current_prefix == entry.prefix_hash)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_locked(self) -> None:
        now = time.time()
        expired = [k for k, v in self._entries.items() if now - v.timestamp > self._ttl]
        for k in expired:
            del self._entries[k]
        while len(self._entries) > self._max_entries:
            oldest = min(self._entries, key=lambda k: self._entries[k].timestamp)
            del self._entries[oldest]


prompt_cache_manager = PromptCacheManager()
