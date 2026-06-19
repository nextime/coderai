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

"""Central registry of long-running tasks (generation + training).

Lets the admin UI list every in-flight / recent task and cooperatively cancel
one. Cancellation is *cooperative*: the per-step diffusers callbacks
(``_step_cb`` / ``_vid_step_cb`` / ``_aud_step_cb`` / sd.cpp ``progress_callback``)
and the LoRA training loops call ``raise_if_cancelled(task_id)`` between steps,
which raises :class:`TaskCancelled` once the task is cancelled. This mirrors the
``codai.models.thermal.checkpoint()`` pause already invoked in those same hooks,
so the running model aborts at the next step boundary.

The registry is process-local and in-memory: it is the live view. Durable LoRA
training state lives separately in ``codai/api/loras.py`` (``_train_jobs.json``);
a task with a ``job_id`` links the two.
"""

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


class TaskCancelled(Exception):
    """Raised inside a worker when its task has been cancelled by the user."""


ACTIVE_STATES = ("queued", "running")
TERMINAL_STATES = ("done", "error", "cancelled")


@dataclass
class Task:
    id: str
    kind: str                      # training | image | video | audio | text | pipeline
    title: str = ""
    model: str = ""
    status: str = "queued"         # queued | running | done | error | cancelled
    step: int = 0
    total: int = 0
    rate: float = 0.0              # throughput (tokens/s) for text generation
    message: str = ""
    job_id: Optional[str] = None   # link to a durable loras training job, if any
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    cancellable: bool = True
    restartable: bool = False
    pausable: bool = True
    paused: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["active"] = self.status in ACTIVE_STATES
        return d


class TaskRegistry:
    """Thread-safe registry. All public methods take the registry lock briefly;
    the per-task cancel flag is a ``threading.Event`` so ``is_cancelled`` is a
    cheap lock-free read suitable for a hot step loop."""

    def __init__(self, history: int = 50):
        self._lock = threading.Lock()
        self._tasks: Dict[str, Task] = {}
        self._events: Dict[str, threading.Event] = {}
        self._pause_events: Dict[str, threading.Event] = {}
        # Per-task (last_time, last_step) for throughput (it/s) estimation in step().
        self._rate_state: Dict[str, tuple] = {}
        self._history = history

    def register(self, kind: str, *, title: str = "", model: str = "",
                 total: int = 0, job_id: Optional[str] = None,
                 status: str = "queued", cancellable: bool = True,
                 restartable: bool = False, pausable: bool = True,
                 task_id: Optional[str] = None) -> str:
        tid = task_id or f"task-{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._tasks[tid] = Task(
                id=tid, kind=kind, title=title, model=model, total=total,
                job_id=job_id, status=status, cancellable=cancellable,
                restartable=restartable, pausable=pausable,
            )
            self._events[tid] = threading.Event()
            self._pause_events[tid] = threading.Event()
            self._prune_locked()
        return tid

    def start(self, tid: str) -> None:
        with self._lock:
            t = self._tasks.get(tid)
            if t and t.status in ACTIVE_STATES:
                t.status = "running"
                t.started_at = t.started_at or time.time()

    def update(self, tid: str, **fields) -> None:
        with self._lock:
            t = self._tasks.get(tid)
            if not t:
                return
            for k, v in fields.items():
                if hasattr(t, k):
                    setattr(t, k, v)

    def step(self, tid: str, step: int, total: Optional[int] = None) -> None:
        with self._lock:
            t = self._tasks.get(tid)
            if not t:
                return
            new_step = int(step)
            if total is not None:
                t.total = int(total)
            # Estimate throughput (iterations/s) for generation tasks from the
            # rate of step increments, EMA-smoothed. Text generation reports its
            # own tokens/s via update(rate=…), so it doesn't go through here.
            now = time.time()
            prev = self._rate_state.get(tid)
            if prev is not None:
                _pt, _ps = prev
                _dt = now - _pt
                _dn = new_step - _ps
                if _dt >= 0.3 and _dn > 0:
                    _inst = _dn / _dt
                    t.rate = round(_inst if not t.rate else 0.5 * t.rate + 0.5 * _inst, 2)
                    self._rate_state[tid] = (now, new_step)
            else:
                self._rate_state[tid] = (now, new_step)
            t.step = new_step

    def current_loading_task(self) -> Optional[str]:
        """Id of the most-recently-started running ``loading`` task, if any.

        Used by the tqdm progress capture (which runs in the load's executor
        thread, with no handle to the task id) to publish shard/component
        progress onto the live loading entry."""
        with self._lock:
            best = None
            best_t = -1.0
            for tid, t in self._tasks.items():
                if t.kind == "loading" and t.status == "running":
                    st = t.started_at or t.created_at or 0.0
                    if st >= best_t:
                        best_t, best = st, tid
            return best

    def finish(self, tid: str, status: str = "done", message: str = "") -> None:
        with self._lock:
            t = self._tasks.get(tid)
            if not t:
                return
            # A user cancel wins over a late 'done' from the worker unwinding.
            if not (t.status == "cancelled" and status == "done"):
                t.status = status
            if message:
                t.message = message
            t.ended_at = time.time()
            # Replace the live instantaneous rate with the run AVERAGE (it/s) so
            # the history shows a stable, representative throughput.
            if t.step and t.started_at:
                _dur = (t.ended_at or t.started_at) - t.started_at
                if _dur > 0 and not (t.kind == "text" and t.rate):
                    t.rate = round(t.step / _dur, 2)
            self._rate_state.pop(tid, None)
            self._prune_locked()

    def cancel(self, tid: str) -> bool:
        with self._lock:
            t = self._tasks.get(tid)
            if not t:
                return False
            ev = self._events.get(tid)
            if ev:
                ev.set()
            # Release any pause so a paused→cancelled task unblocks immediately.
            pev = self._pause_events.get(tid)
            if pev:
                pev.set()  # wakes wait_if_paused; its is_cancelled check then raises
            t.paused = False
            was = t.status
            if was in ACTIVE_STATES:
                t.status = "cancelled"
                t.message = "cancelled"
                # A queued task never entered a worker, so it's terminal now;
                # a running task finalises when its worker observes the flag.
                if was == "queued":
                    t.ended_at = time.time()
            return True

    def is_cancelled(self, tid: Optional[str]) -> bool:
        if not tid:
            return False
        ev = self._events.get(tid)
        return bool(ev and ev.is_set())

    def raise_if_cancelled(self, tid: Optional[str]) -> None:
        if self.is_cancelled(tid):
            raise TaskCancelled(tid)

    # --- Pause / resume (cooperative, at the next step boundary) -------------
    def pause(self, tid: str) -> bool:
        with self._lock:
            t = self._tasks.get(tid)
            ev = self._pause_events.get(tid)
            if not t or ev is None or t.status not in ACTIVE_STATES:
                return False
            ev.set()
            t.paused = True
            return True

    def resume(self, tid: str) -> bool:
        with self._lock:
            t = self._tasks.get(tid)
            ev = self._pause_events.get(tid)
            if not t or ev is None:
                return False
            ev.clear()
            t.paused = False
            return True

    def is_paused(self, tid: Optional[str]) -> bool:
        if not tid:
            return False
        ev = self._pause_events.get(tid)
        return bool(ev and ev.is_set())

    def wait_if_paused(self, tid: Optional[str], poll: float = 0.2) -> None:
        """Block while the task is paused, returning when it is resumed.

        Stays responsive to cancellation: a paused task that is then cancelled
        raises :class:`TaskCancelled` instead of hanging. Safe to call from a hot
        step loop — a no-op unless the task is actually paused."""
        ev = self._pause_events.get(tid) if tid else None
        if ev is None or not ev.is_set():
            return
        while ev.is_set():
            if self.is_cancelled(tid):
                raise TaskCancelled(tid)
            ev.wait(timeout=poll)

    def remove(self, tid: str) -> bool:
        """Drop a task from the registry entirely (used to dismiss a finished/
        cancelled task from the view). No-op on a missing id."""
        with self._lock:
            self._events.pop(tid, None)
            self._pause_events.pop(tid, None)
            self._rate_state.pop(tid, None)
            return self._tasks.pop(tid, None) is not None

    def get(self, tid: str) -> Optional[dict]:
        with self._lock:
            t = self._tasks.get(tid)
            return t.to_dict() if t else None

    def list(self) -> List[dict]:
        with self._lock:
            return [t.to_dict() for t in sorted(
                self._tasks.values(), key=lambda x: x.created_at, reverse=True)]

    def _prune_locked(self) -> None:
        """Keep all active tasks + the most recent ``history`` terminal ones."""
        terminal = [t for t in self._tasks.values() if t.status in TERMINAL_STATES]
        if len(terminal) <= self._history:
            return
        terminal.sort(key=lambda x: x.ended_at or x.created_at)
        for t in terminal[:-self._history]:
            self._tasks.pop(t.id, None)
            self._events.pop(t.id, None)
            self._pause_events.pop(t.id, None)
            self._rate_state.pop(t.id, None)


# Process-wide singleton.
task_registry = TaskRegistry()


def raise_if_cancelled(task_id: Optional[str]) -> None:
    """Free helper mirroring ``thermal.checkpoint()`` — call from hot step loops.

    Raises :class:`TaskCancelled` if ``task_id`` has been cancelled; a falsy
    ``task_id`` is a no-op (so callers needn't guard)."""
    task_registry.raise_if_cancelled(task_id)


def wait_if_paused(task_id: Optional[str]) -> None:
    """Free helper — block at a step boundary while ``task_id`` is paused.

    Returns immediately when not paused; raises :class:`TaskCancelled` if the
    task is cancelled while paused. A falsy ``task_id`` is a no-op."""
    task_registry.wait_if_paused(task_id)


# --- tqdm progress capture for model loads ---------------------------------
# diffusers / transformers / huggingface_hub emit their load progress through
# tqdm ("Loading checkpoint shards", "Loading pipeline components", "Loading
# weights", download bars). We monkeypatch the base tqdm class for the duration
# of a load so those bars publish step/total/desc onto the live `loading` task —
# turning the Tasks-page "working…" into the same detailed progress the terminal
# shows. Ref-counted so concurrent/nested loads share one patch.
_tqdm_patch_lock = threading.Lock()
_tqdm_patch_depth = 0
_tqdm_orig: Dict[str, object] = {}


def _publish_loading_progress(desc, n, total):
    tid = task_registry.current_loading_task()
    if not tid:
        return
    try:
        n = int(n or 0)
        total = int(total or 0)
    except (TypeError, ValueError):
        return
    # tqdm may store the desc with a trailing ": " (set_description) — normalise.
    desc = (str(desc).strip().rstrip(":").strip() if desc else "") or "Loading"
    # Step only counts up to total; the message carries the human-readable phase.
    task_registry.step(tid, n, total if total > 0 else None)
    msg = f"{desc}: {n}/{total}" if total > 0 else desc
    task_registry.update(tid, message=msg)


def _install_tqdm_capture():
    global _tqdm_patch_depth
    with _tqdm_patch_lock:
        _tqdm_patch_depth += 1
        if _tqdm_patch_depth > 1:
            return
        try:
            from tqdm import std as _tqdm_std
        except Exception:
            return
        cls = _tqdm_std.tqdm
        _tqdm_orig['update'] = cls.update
        _tqdm_orig['close'] = cls.close
        _tqdm_orig['cls'] = cls

        def _patched_update(self, n=1):
            r = _tqdm_orig['update'](self, n)
            try:
                if not getattr(self, 'disable', False):
                    _publish_loading_progress(
                        getattr(self, 'desc', ''), getattr(self, 'n', 0),
                        getattr(self, 'total', 0))
            except Exception:
                pass
            return r

        def _patched_close(self):
            try:
                if not getattr(self, 'disable', False) and getattr(self, 'total', 0):
                    _publish_loading_progress(
                        getattr(self, 'desc', ''), getattr(self, 'total', 0),
                        getattr(self, 'total', 0))
            except Exception:
                pass
            return _tqdm_orig['close'](self)

        cls.update = _patched_update
        cls.close = _patched_close


def _remove_tqdm_capture():
    global _tqdm_patch_depth
    with _tqdm_patch_lock:
        if _tqdm_patch_depth <= 0:
            return
        _tqdm_patch_depth -= 1
        if _tqdm_patch_depth > 0:
            return
        cls = _tqdm_orig.get('cls')
        if cls is not None:
            if 'update' in _tqdm_orig:
                cls.update = _tqdm_orig['update']
            if 'close' in _tqdm_orig:
                cls.close = _tqdm_orig['close']
        _tqdm_orig.clear()


@contextmanager
def loading_task(model: str, *, model_type: str = "model", title: Optional[str] = None):
    """Context manager that shows a model load as a Tasks-page entry.

    Model loading can't be paused or cancelled (it's a single blocking
    ``from_pretrained`` / ``Llama(...)`` call), so the task is registered
    non-cancellable and non-pausable — the Tasks UI shows it with no action
    buttons. While the context is active, tqdm progress bars emitted by
    diffusers/transformers/hf_hub are captured and published onto the task as
    step/total + a phase message ("Loading checkpoint shards: 7/12"), so the UI
    mirrors the terminal instead of a bare "working…". The task finishes ``done``
    on success or ``error`` on exception."""
    label = title or f"Loading {model}"
    tid = task_registry.register(
        "loading", title=label, model=model or "", status="running",
        cancellable=False, restartable=False, pausable=False)
    _install_tqdm_capture()
    try:
        yield tid
        task_registry.finish(tid, "done")
    except BaseException as e:  # noqa: BLE001 — record then re-raise
        task_registry.finish(tid, "error", str(e)[:200] or e.__class__.__name__)
        raise
    finally:
        _remove_tqdm_capture()
