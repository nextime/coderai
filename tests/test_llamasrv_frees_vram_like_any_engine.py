"""A llama-server holds VRAM in a subprocess, and must answer eviction anyway.

Every other engine frees VRAM in-process: the manager drops the model object and
the tensors go. A managed ``llama-server`` holds the whole model in a child
process, so the card is only free once that process is GONE. The manager evicts
it the same way as anything else -- it drops the backend and calls ``cleanup()``
-- which means the entire chain

    _evict_one -> ModelManager.cleanup -> LlamaSrvBackend.cleanup -> stop_service

has to end in a dead process, and every link that quietly does nothing turns an
eviction into a leak the manager believes it already performed. It then measures
no freed VRAM, loads the next model anyway, and OOMs against a server nobody is
tracking.
"""

import inspect

import pytest

import codai.api.llamasrv_worker as worker
from codai.backends.llamasrv import LlamaSrvBackend
from codai.config import LlamaSrvConfig, LlamaSrvForkConfig


class FakeProc:
    """A llama-server that dies after `survives` signals."""

    def __init__(self, survives=0):
        self.survives = survives
        self.signals = []
        self.waits = []
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.signals.append("terminate")
        if self.survives <= 0:
            self._alive = False
        self.survives -= 1

    def kill(self):
        self.signals.append("kill")
        if self.survives <= 0:
            self._alive = False
        self.survives -= 1

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if not self._alive:
            return 0
        raise RuntimeError("still running")


@pytest.fixture(autouse=True)
def _clean_registry():
    worker._services.clear()
    yield
    worker._services.clear()


def _register(key, proc):
    worker._services[key] = {"proc": proc, "port": 1, "url": "http://x", "key": ""}


# ------------------------------------------------------- the eviction chain --

def test_the_manager_evicting_a_model_stops_its_server(monkeypatch):
    """The whole point: the manager's generic eviction has to reach the
    subprocess. It drops the backend and calls cleanup() -- nothing in the
    manager knows this model lives in another process."""
    stopped = []
    monkeypatch.setattr(worker, "stop_service", lambda k: stopped.append(k) or True)
    be = LlamaSrvBackend(LlamaSrvForkConfig())
    be._svc_key = "llamasrv_fork::/AI/model.gguf"

    class Wrapper:          # what multi_model_manager holds and _evict_one pops
        def __init__(self, backend): self.backend = backend
        def cleanup(self):
            if self.backend is not None:
                self.backend.cleanup()
                self.backend = None

    Wrapper(be).cleanup()
    assert stopped == ["llamasrv_fork::/AI/model.gguf"]


def test_cleanup_without_a_service_key_still_names_its_own_engine(monkeypatch):
    """An unprefixed fallback key matches no service: stop_service does nothing,
    and the manager goes on believing it freed the model's whole VRAM."""
    seen = []
    monkeypatch.setattr(worker, "stop_service", lambda k: seen.append(k) or True)
    for cfg, want in ((LlamaSrvConfig(), "llamasrv::"),
                      (LlamaSrvForkConfig(), "llamasrv_fork::")):
        cfg.model_id = "mymodel"
        be = LlamaSrvBackend(cfg)
        be._svc_key = None
        be.cleanup()
    assert seen == ["llamasrv::mymodel", "llamasrv_fork::mymodel"]


def test_evicting_one_engines_model_leaves_the_others_server_alone():
    """Stock and fork can serve the same GGUF at once. Eviction must take only
    the one it was asked for, or measuring a fork against stock would kill the
    comparison halfway through."""
    stock, fork = FakeProc(), FakeProc()
    _register("llamasrv::/AI/m.gguf", stock)
    _register("llamasrv_fork::/AI/m.gguf", fork)
    assert worker.stop_service("llamasrv_fork::/AI/m.gguf") is True
    assert fork.signals and not stock.signals
    assert "llamasrv::/AI/m.gguf" in worker._services


# ------------------------------------------- the card is free when it is free --

def test_a_stubborn_server_is_killed_and_waited_for():
    """Returning once the process has been SIGNALLED is what makes an eviction
    look like it freed nothing: the weights are still mapped."""
    proc = FakeProc(survives=1)          # ignores terminate, dies on kill
    _register("llamasrv::m", proc)
    assert worker.stop_service("llamasrv::m") is True
    assert proc.signals == ["terminate", "kill"]
    # waited after BOTH signals, not just the first
    assert len(proc.waits) == 2


def test_a_server_that_will_not_die_is_reported_not_claimed():
    """Claiming the VRAM is free when it is not sends the next load into an OOM
    against a process nobody is tracking any more."""
    proc = FakeProc(survives=99)
    _register("llamasrv::m", proc)
    assert worker.stop_service("llamasrv::m") is False


def test_an_unknown_key_is_reported_rather_than_silently_fine(capsys):
    _register("llamasrv::/AI/real.gguf", FakeProc())
    assert worker.stop_service("llamasrv::/AI/typo.gguf") is False
    out = capsys.readouterr().out
    assert "matched no running server" in out
    assert "/AI/real.gguf" in out, "the log should name what IS running"


def test_stopping_nothing_at_all_is_quiet(capsys):
    assert worker.stop_service("llamasrv::whatever") is False
    assert capsys.readouterr().out == ""


# -------------------------------------------------- an engine that is killed --

def test_the_server_is_told_to_die_with_its_engine():
    """Eviction works THROUGH the owning engine. An engine that is SIGKILLed
    (OOM killer, crash, pkill) never runs atexit, so without PR_SET_PDEATHSIG
    the server outlives it holding the entire model's VRAM, with nothing left
    that knows how to evict it. That is the 'survivor with no address' failure,
    on a GPU instead of a rented pod."""
    src = inspect.getsource(worker)
    assert "PR_SET_PDEATHSIG" in src
    assert "preexec_fn=_die_with_parent" in src
    # prctl(PR_SET_PDEATHSIG=1, SIGKILL=9)
    assert "prctl(1, 9)" in src


def test_a_clean_exit_still_stops_every_server():
    """atexit covers the ordinary case; both paths must exist, not one."""
    src = inspect.getsource(worker)
    assert "_atexit.register(stop_all)" in src
    a, b = FakeProc(), FakeProc()
    _register("llamasrv::a", a)
    _register("llamasrv_fork::b", b)
    worker.stop_all()
    assert a.signals and b.signals
    assert worker._services == {}
