"""Cross-engine GPU exclusivity for long in-engine background ops (LoRA training).

The front swap-gate serializes REQUEST-level GPU use, but a LoRA training runs as
an in-engine BACKGROUND job (its POST returns a job_id immediately), so no front
request is in flight to hold the gate for the training's whole duration. This lock
closes that gap: training reserves the GPU here AND on co-located sibling engines
(POST /internal/gpu-reserve), and every ordinary model-load path calls
``wait_until_free()`` first — so neither the same engine (image/video) nor a
sibling (gguf text) loads a model and contends with training's VRAM (the OOM this
prevents). A waiting request isn't dropped: it blocks until training releases, then
loads and serves — the same queue-behind-the-owner behaviour the swap-gate gives
request-level work.

The training thread is exempt from its OWN reservation, so training loading its
base model never deadlocks on the lock it holds.
"""
import os
import threading
import time

_cv = threading.Condition()
_local_reason = None            # this engine holds it (e.g. "lora-training:<name>")
_local_thread = None            # tid that reserved locally — exempt from waiting
_sibling_reasons = {}           # sibling key -> reason (set via internal endpoint)


def reserved() -> bool:
    return _local_reason is not None or bool(_sibling_reasons)


def current_reason() -> str:
    with _cv:
        if _local_reason:
            return _local_reason
        for r in _sibling_reasons.values():
            return r
        return ""


def _set_local(reason) -> None:
    global _local_reason, _local_thread
    with _cv:
        _local_reason = reason
        _local_thread = threading.get_ident() if reason else None
        _cv.notify_all()


def set_sibling(key: str, reason) -> None:
    """Set (reason truthy) or clear (reason falsy) a co-located sibling's reservation."""
    with _cv:
        if reason:
            _sibling_reasons[str(key)] = reason
        else:
            _sibling_reasons.pop(str(key), None)
        _cv.notify_all()


def wait_until_free(timeout: float = 900.0, context: str = "") -> bool:
    """Block until the GPU isn't reserved (by this engine or a sibling). Returns
    True if free, False on timeout (caller proceeds anyway — bounded so a stuck
    reservation can't hang forever). The thread that holds the local reservation
    (the training thread) never waits on itself."""
    if _local_thread is not None and threading.get_ident() == _local_thread:
        return True
    if not reserved():
        return True
    _r = current_reason()
    print(f"  [gpu-lock] {context or 'model load'} waiting — GPU reserved ({_r})",
          flush=True)
    deadline = time.time() + max(0.0, timeout)
    with _cv:
        while reserved():
            if _local_thread is not None and threading.get_ident() == _local_thread:
                return True
            remaining = deadline - time.time()
            if remaining <= 0:
                print(f"  [gpu-lock] wait timed out after {timeout:.0f}s "
                      f"({_r}) — proceeding", flush=True)
                return False
            _cv.wait(timeout=min(remaining, 5.0))
    print(f"  [gpu-lock] GPU free — {context or 'model load'} proceeding", flush=True)
    return True


def _cosited_urls():
    return [u for u in (os.environ.get("CODERAI_COSITED_URLS") or "").split(",") if u]


def _notify_siblings(reason) -> None:
    urls = _cosited_urls()
    if not urls:
        return
    try:
        import httpx
    except Exception:
        return
    tok = os.environ.get("CODERAI_INTERNAL_TOKEN") or ""
    me = os.environ.get("CODERAI_ENGINE_NAME") or str(os.getpid())
    path = "/internal/gpu-reserve" if reason else "/internal/gpu-release"
    for u in urls:
        try:
            httpx.post(f"{u}{path}", json={"from": me, "reason": reason or ""},
                       headers={"x-coderai-internal": tok}, timeout=15.0)
        except Exception as e:
            print(f"  [gpu-lock] sibling {u}{path} failed: {e}", flush=True)


def reserve(reason: str) -> None:
    """Reserve the GPU for an exclusive local op (training): block this engine's
    ordinary loads AND tell co-located siblings to hold theirs."""
    _set_local(reason or "reserved")
    _notify_siblings(reason or "reserved")


def release() -> None:
    """Release the local reservation and clear it on siblings."""
    _set_local(None)
    _notify_siblings(None)
