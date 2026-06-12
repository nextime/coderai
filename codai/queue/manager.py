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

"""Central scheduler for model-backed request admission and queue reporting."""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Set
import asyncio
import time


@dataclass
class SchedulerLease:
    request_id: str
    model_key: str
    started_at: float = field(default_factory=time.time)
    waited: bool = False
    wait_time_seconds: float = 0.0


@dataclass
class WaitingRequest:
    request_id: str
    model_key: str
    enqueued_at: float
    sequence: int
    event: asyncio.Event = field(default_factory=asyncio.Event)
    bypassed_by: int = 0
    prefix_key: str = ""  # stable hash of the cacheable prompt prefix


class QueueManager:
    """Central in-process request scheduler."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.max_size: int = 6
        self.max_parallel_requests: int = 2
        self.waiting: Deque[WaitingRequest] = deque()
        self.waiting_by_id: Dict[str, WaitingRequest] = {}
        self.active_leases: Dict[str, SchedulerLease] = {}
        self.active_by_model: Dict[str, int] = {}
        self.loaded_models: Set[str] = set()
        self.sequence: int = 0
        self.fairness_bypass_limit: int = 2
        self.current_request_id: Optional[str] = None
        self.model_loading: bool = False
        self.model_name: Optional[str] = None
        self._processing: bool = False
        self._ready_request_ids: Set[str] = set()
        self._last_prefix_key: str = ""  # prefix key of the last completed request

    def set_loaded_models(self, model_keys: Set[str]) -> None:
        self.loaded_models = set(model_keys)

    def mark_model_loaded(self, model_key: str) -> None:
        self.loaded_models.add(model_key)

    def mark_model_unloaded(self, model_key: str) -> None:
        self.loaded_models.discard(model_key)

    def reset_for_tests(self) -> None:
        self.waiting.clear()
        self.waiting_by_id.clear()
        self.active_leases.clear()
        self.active_by_model.clear()
        self.loaded_models.clear()
        self.sequence = 0
        self.current_request_id = None
        self.model_loading = False
        self.model_name = None
        self._processing = False
        self._ready_request_ids.clear()
        self._last_prefix_key = ""

    async def is_full(self) -> bool:
        async with self.lock:
            return len(self.waiting) >= self.max_size

    async def acquire(self, request_id: str, model_key: str,
                      prefix_key: str = "") -> SchedulerLease:
        waiter = None
        async with self.lock:
            if self._can_start_now(model_key):
                return self._grant_lease(request_id, model_key)
            waiter = self._enqueue_waiter(request_id, model_key, prefix_key)

        await waiter.event.wait()
        async with self.lock:
            self._ready_request_ids.discard(request_id)
            lease = self._grant_lease(request_id, model_key)
            lease.waited = True
            lease.wait_time_seconds = max(0.0, time.time() - waiter.enqueued_at)
            return lease

    async def release(self, lease: SchedulerLease,
                      prefix_key: str = "") -> None:
        async with self.lock:
            self.active_leases.pop(lease.request_id, None)
            current = self.active_by_model.get(lease.model_key, 0)
            if current <= 1:
                self.active_by_model.pop(lease.model_key, None)
            else:
                self.active_by_model[lease.model_key] = current - 1
            if self.current_request_id == lease.request_id:
                self.current_request_id = None
            if prefix_key:
                self._last_prefix_key = prefix_key
            self._processing = bool(self.active_leases)
            self._wake_waiters_locked()

    async def add_waiting(self, request_id: str, model_key: str = "",
                          prefix_key: str = "") -> None:
        async with self.lock:
            if request_id in self.waiting_by_id:
                return
            self._enqueue_waiter(request_id, model_key or request_id, prefix_key)

    async def remove_waiting(self, request_id: str) -> None:
        async with self.lock:
            waiter = self.waiting_by_id.pop(request_id, None)
            if waiter and waiter in self.waiting:
                self.waiting.remove(waiter)
            self._ready_request_ids.discard(request_id)

    async def start_processing(self, request_id: str, model_name: str = None) -> None:
        async with self.lock:
            waiter = self.waiting_by_id.pop(request_id, None)
            if waiter and waiter in self.waiting:
                self.waiting.remove(waiter)
            self.current_request_id = request_id
            self.model_name = model_name
            self._processing = True

    async def finish_processing(self) -> None:
        async with self.lock:
            self.current_request_id = None
            self._processing = bool(self.active_leases)

    async def is_waiting(self, request_id: str) -> bool:
        async with self.lock:
            return request_id in self.waiting_by_id

    async def get_wait_time(self, request_id: str) -> float:
        async with self.lock:
            waiter = self.waiting_by_id.get(request_id)
            if waiter:
                return time.time() - waiter.enqueued_at
            return 0.0

    async def get_queue_position(self, request_id: str) -> int:
        async with self.lock:
            for index, waiter in enumerate(self.waiting, start=1):
                if waiter.request_id == request_id:
                    return index
            return 0

    def list_waiting(self) -> list:
        """Best-effort snapshot of queued (waiting) requests for the Tasks view.
        Read without the async lock — fine for a read-only UI snapshot."""
        out = []
        for w in list(self.waiting):
            out.append({
                "request_id": w.request_id,
                "model_key": w.model_key,
                "enqueued_at": w.enqueued_at,
            })
        return out

    def list_active(self) -> list:
        """Best-effort snapshot of in-flight leases for the Tasks view."""
        return [{"request_id": rid, "model_key": lease.model_key}
                for rid, lease in list(self.active_leases.items())]

    def get_metrics(self) -> Dict[str, object]:
        return {
            "active": len(self.active_leases),
            "waiting": len(self.waiting),
            "max_parallel_requests": self.max_parallel_requests,
            "queue_max_size": self.max_size,
            "active_by_model": dict(self.active_by_model),
            "waiting_by_model": self._waiting_counts_locked(),
            "loaded_models": sorted(self.loaded_models),
        }

    def _enqueue_waiter(self, request_id: str, model_key: str,
                        prefix_key: str = "") -> WaitingRequest:
        self.sequence += 1
        waiter = WaitingRequest(
            request_id=request_id,
            model_key=model_key,
            enqueued_at=time.time(),
            sequence=self.sequence,
            prefix_key=prefix_key,
        )
        self.waiting.append(waiter)
        self.waiting_by_id[request_id] = waiter
        return waiter

    def _grant_lease(self, request_id: str, model_key: str) -> SchedulerLease:
        lease = SchedulerLease(request_id=request_id, model_key=model_key)
        self.active_leases[request_id] = lease
        self.active_by_model[model_key] = self.active_by_model.get(model_key, 0) + 1
        self.current_request_id = request_id
        self.model_name = model_key
        self._processing = True
        return lease

    def _can_start_now(self, model_key: str) -> bool:
        if self.max_parallel_requests > 0:
            in_flight = len(self.active_leases) + len(self._ready_request_ids)
            if in_flight >= self.max_parallel_requests:
                return False
        if self.active_by_model.get(model_key, 0) > 0:
            return False
        return self._is_loaded_model(model_key) or self._can_schedule_model_switch(model_key)

    def _waiter_can_start_locked(self, waiter: WaitingRequest) -> bool:
        return self._can_start_now(waiter.model_key)

    def _is_loaded_model(self, model_key: str) -> bool:
        return model_key in self.loaded_models

    def _can_schedule_model_switch(self, model_key: str) -> bool:
        if not self.loaded_models:
            return True
        if self._waiting_counts_locked().get(model_key, 0) > 0 and self._is_loaded_model(model_key):
            return True
        for loaded_key in self.loaded_models:
            if self.active_by_model.get(loaded_key, 0) > 0:
                return False
            if self._waiting_counts_locked().get(loaded_key, 0) > 0:
                return False
        return True

    def _wake_waiters_locked(self) -> None:
        while True:
            candidate = self._pick_next_waiter_locked()
            if candidate is None:
                return
            self.waiting.remove(candidate)
            self.waiting_by_id.pop(candidate.request_id, None)
            self._ready_request_ids.add(candidate.request_id)
            candidate.event.set()
            if self.max_parallel_requests > 0 and len(self.active_leases) + len(self._ready_request_ids) >= self.max_parallel_requests:
                return

    def _pick_next_waiter_locked(self) -> Optional[WaitingRequest]:
        # Collect all candidates that can start now.
        candidates = [w for w in self.waiting if self._waiter_can_start_locked(w)]
        if not candidates:
            return None

        # Fairness: don't bypass an older waiter more than the limit.
        def _is_fair(waiter: WaitingRequest) -> bool:
            older_blocked = [
                other for other in self.waiting
                if other.sequence < waiter.sequence and not self._waiter_can_start_locked(other)
            ]
            if any(other.bypassed_by >= self.fairness_bypass_limit for other in older_blocked):
                return False
            for other in older_blocked:
                other.bypassed_by += 1
            return True

        # Prompt aggregation: prefer candidates whose prefix key matches the
        # last completed request — they will hit a warm KV cache.
        if self._last_prefix_key:
            warm_candidates = [
                w for w in candidates
                if w.prefix_key and w.prefix_key == self._last_prefix_key
            ]
            for waiter in warm_candidates:
                if _is_fair(waiter):
                    return waiter

        # Fall back to FIFO order.
        for waiter in candidates:
            if _is_fair(waiter):
                return waiter

        return None

    def _waiting_counts_locked(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for waiter in self.waiting:
            counts[waiter.model_key] = counts.get(waiter.model_key, 0) + 1
        return counts


queue_manager = QueueManager()
