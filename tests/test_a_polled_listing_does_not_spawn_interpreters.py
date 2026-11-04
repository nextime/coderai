"""`/v1/video/presenter/engines` answers from memory, and probes in a thread.

The listing looks like a dictionary lookup and is not. `missing_pieces()` calls
`_pip_ok()` for every managed engine that is not installed, and `_pip_ok`
SPAWNS A PYTHON INTERPRETER to try the engine's imports inside its venv. On a
polled endpoint, on the event loop, a client measured 2026-11-03:

    0.9-1.3 s per call while the server was busy, about 5 s at peaks
    (/v1/models, for comparison: 0.04 s)

and their create page waits for it before drawing. Nothing in the answer
changes without an operator installing or removing an engine, so it is cached
and refreshed off the loop -- and an install drops the cache rather than
letting a new engine wait out the TTL.
"""
import asyncio
import inspect
import time

import pytest

from codai.api import compose as C


@pytest.fixture(autouse=True)
def _clear():
    C.invalidate_engine_cache()
    yield
    C.invalidate_engine_cache()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_the_probe_runs_once_for_many_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(C, "_probe_presenter_engines",
                        lambda: calls.append(1) or {"engines": ["wav2lip"]})
    for _ in range(5):
        assert _run(C.presenter_engines(_auth=None))["engines"] == ["wav2lip"]
    assert len(calls) == 1, "a polled endpoint must not re-probe per call"


def test_an_install_drops_the_cache(monkeypatch):
    answers = [{"engines": []}, {"engines": ["musetalk"]}]
    monkeypatch.setattr(C, "_probe_presenter_engines", lambda: answers.pop(0))
    assert _run(C.presenter_engines(_auth=None))["engines"] == []
    C.invalidate_engine_cache()
    assert _run(C.presenter_engines(_auth=None))["engines"] == ["musetalk"]


def test_the_cache_expires(monkeypatch):
    answers = [{"engines": ["a"]}, {"engines": ["b"]}]
    monkeypatch.setattr(C, "_probe_presenter_engines", lambda: answers.pop(0))
    monkeypatch.setattr(C, "_ENGINES_TTL", 0.0)
    assert _run(C.presenter_engines(_auth=None))["engines"] == ["a"]
    assert _run(C.presenter_engines(_auth=None))["engines"] == ["b"]


def test_the_probe_never_runs_on_the_event_loop(monkeypatch):
    """A subprocess spawn on the loop stalls every other request in flight."""
    where = {}

    def _probe():
        try:
            asyncio.get_running_loop()
            where["on_loop"] = True
        except RuntimeError:
            where["on_loop"] = False
        return {"engines": []}

    monkeypatch.setattr(C, "_probe_presenter_engines", _probe)
    _run(C.presenter_engines(_auth=None))
    assert where["on_loop"] is False


def test_the_handler_itself_does_no_probing():
    """Everything expensive belongs in the function the executor calls."""
    src = inspect.getsource(C.presenter_engines)
    assert "run_in_executor" in src
    for forbidden in ("missing_pieces", "available_engines", "shutil.which"):
        assert forbidden not in src, (
            f"{forbidden} in the handler puts a filesystem/subprocess probe "
            f"back on the event loop")


def test_an_install_and_an_uninstall_both_invalidate():
    """Checked as source: running them installs 6 GB of engine."""
    from codai.api import lipsync_engines as LE
    for fn in (LE.start_install, LE.uninstall):
        assert "invalidate_engine_cache" in inspect.getsource(fn), (
            f"{fn.__name__} must drop the presenter listing cache")
