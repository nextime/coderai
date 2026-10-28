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

"""Progress for ONE request, instead of one global slot per media type.

``/v1/images/progress`` and its video/audio siblings each reported a single
module-level dict. With two people generating at once, neither can tell their
own request's steps from the other's, so a client that cares about its own
progress has to stop reading them altogether -- which is what happened: a
caller rendering video footage and AI edits showed no step progress at all
while the server was drawing.

A request can therefore name itself, with ``request_id`` in the body or an
``X-Request-Id`` header, and ask for only its own progress. That name is used
as the TASK id, so this is a view over :mod:`codai.tasks.registry` rather than
a second place where progress is written: the task registry already tracks
step/total/status per task under a lock, prunes finished ones, and is already
updated from inside every step callback. A parallel registry would be a second
thing to keep correct.

Without an id the endpoints answer exactly as before, so nothing that does not
ask for this changes.
"""

from typing import Any, Dict, Optional

#: The header a client may use instead of a body field. Same name HTTP proxies
#: and tracing tools already use, so a caller often has one to hand.
HEADER = "X-Request-Id"

#: How a caller's own id is accepted in a request body.
BODY_FIELD = "request_id"

#: Cap on a client-supplied id. It becomes a task id and is echoed back, so it
#: is bounded and stripped of anything that is not safe to put in a log line.
MAX_LEN = 128


def sanitize(raw: Any) -> Optional[str]:
    """A client id reduced to something safe to use as a task id, or None.

    Client-supplied, so it is treated as untrusted: control characters cannot
    reach a log line, and the length is bounded. An id that survives none of
    its characters is no id at all.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    keep = [c for c in s if c.isalnum() or c in "-_.:"]
    s = "".join(keep)[:MAX_LEN]
    return s or None


def from_request(body: Any = None, http_request: Any = None) -> Optional[str]:
    """The caller's own id for this request: body field, else header, else None.

    The body wins, because a caller that put the id in the payload meant that
    one; a header may have been added by a proxy on the way.
    """
    rid = None
    if body is not None:
        got = getattr(body, BODY_FIELD, None)
        if got is None and isinstance(body, dict):
            got = body.get(BODY_FIELD)
        rid = sanitize(got)
    if rid:
        return rid
    if http_request is not None:
        try:
            rid = sanitize(http_request.headers.get(HEADER))
        except Exception:
            rid = None
    return rid


def _phase(task: Dict[str, Any]) -> str:
    """The task's state in the words a progress bar needs.

    ``queued`` while it waits its turn, ``loading`` while the model is being
    loaded (running, but no step budget yet -- a caller holds the bar there
    rather than animating a total it does not have), ``generating`` once steps
    are being counted, and the terminal state otherwise.
    """
    status = str(task.get("status") or "")
    if status == "queued":
        return "queued"
    if status == "running":
        return "generating" if int(task.get("total") or 0) > 0 else "loading"
    return status or "idle"


def view(request_id: str) -> Optional[Dict[str, Any]]:
    """``{phase, current, total, pct, active, ...}`` for one request, or None.

    None means "nothing known under that id" -- either it was never used or it
    has been pruned -- and the endpoints turn that into a 404, so a client can
    stop polling on a definite answer instead of a stale global one.
    """
    from codai.tasks import task_registry
    task = task_registry.get(request_id)
    if not task:
        return None
    current = int(task.get("step") or 0)
    total = int(task.get("total") or 0)
    return {
        "request_id": request_id,
        "phase": _phase(task),
        "current": current,
        "total": total,
        "pct": int(current / total * 100) if total > 0 else 0,
        "active": bool(task.get("active")),
        "model": task.get("model") or "",
        "message": task.get("message") or "",
        "status": task.get("status") or "",
        "rate": task.get("rate") or 0.0,
        "paused": bool(task.get("paused")),
    }


# --------------------------------------------------------------------------- #
# Carrying the id from the endpoint to the task that does the work
# --------------------------------------------------------------------------- #
#
# The id is resolved at the endpoint, but the task that counts steps is
# registered deeper in, past several call layers and inside a worker thread.
# A ContextVar carries it without threading an argument through all of them:
# `asyncio.to_thread` copies the context, which is how every generation path
# here offloads, so the value is visible in the step callback.
#
# `take` is one-shot ON PURPOSE. One request registers more than one task -- a
# short "loading" task and then the generation itself -- and if both adopted
# the caller's id the second would overwrite the first under the same key and
# the client would be served whichever happened to be last. The generation
# task claims the id; anything else keeps its own generated one.

import contextvars as _contextvars

_current: "_contextvars.ContextVar[Optional[str]]" = _contextvars.ContextVar(
    "coderai_request_progress_id", default=None)


def set_current(request_id: Optional[str]):
    """Make ``request_id`` the id the next task may claim. Returns a token."""
    return _current.set(request_id or None)


def reset_current(token) -> None:
    try:
        _current.reset(token)
    except Exception:
        pass


def take() -> Optional[str]:
    """The caller's id, once. Returns None on every later call."""
    rid = _current.get()
    if rid:
        _current.set(None)
    return rid


def peek() -> Optional[str]:
    """The caller's id without consuming it (for logs and messages)."""
    return _current.get()
