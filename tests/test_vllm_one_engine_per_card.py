"""One vLLM per model, however many callers ask at once or how busy it is.

Measured on a rented A40 (2026-10-19): the engine stopped answering under 128
pages in flight; the next caller started a second engine beside the live one,
another caller started a third, and for twenty minutes vLLM was relaunched
every few seconds and killed with -9 mid-start -- two EngineCores one second
apart in the log. Two rules:

  * an engine that is STILL STARTING is waited for, never duplicated;
  * an engine whose process is ALIVE but slow to answer is BUSY, not dead,
    and is handed out rather than replaced.
"""
import threading
import time
import types

import pytest

from codai.api import vllm_worker as vw

#: Captured before any fixture swaps time.sleep (vw.time IS the time module).
_REAL_SLEEP = time.sleep


class _Proc:
    def __init__(self):
        self.returncode = None
        self.stdout = iter(())

    def poll(self):
        return self.returncode


@pytest.fixture
def fake(monkeypatch):
    launches = []
    state = {"healthy": set()}

    def _popen(cmd, **kw):
        p = _Proc()
        launches.append(p)
        return p

    monkeypatch.setattr(vw, "resolve_service_key", lambda cfg, mp: ("m", "k"))
    monkeypatch.setattr(vw, "ensure_built", lambda cfg: "py")
    monkeypatch.setattr(vw, "_launch_cmd", lambda *a, **k: ["vllm"])
    monkeypatch.setattr(vw, "_model_config_for", lambda *a, **k: {})
    monkeypatch.setattr(vw, "_sibling_lora_specs", lambda *a, **k: [])
    monkeypatch.setattr(vw, "needs_ray", lambda cfg: False)
    monkeypatch.setattr(vw, "_pump_logs", lambda *a, **k: None)
    monkeypatch.setattr(vw, "_serves_name", lambda *a, **k: True)
    monkeypatch.setattr(vw, "_adapter_is_stale", lambda *a, **k: False)
    ports = iter(range(41000, 42000))
    monkeypatch.setattr(vw, "_free_port", lambda: next(ports))
    monkeypatch.setattr(vw.subprocess, "Popen", _popen)
    monkeypatch.setattr(vw, "_health_ok",
                        lambda url, timeout=3: url in state["healthy"])
    monkeypatch.setattr(vw.time, "sleep", lambda s: None)
    vw._services.clear()
    yield types.SimpleNamespace(launches=launches, state=state)
    vw._services.clear()


CFG = types.SimpleNamespace(host="127.0.0.1", port=0, gpu="", extra_env="",
                            service_url="", model_path="m")


def test_a_second_caller_waits_for_the_first_start(fake, monkeypatch):
    up = threading.Event()
    real_sleep = _REAL_SLEEP
    monkeypatch.setattr(vw.time, "sleep", lambda s: real_sleep(0.01))
    monkeypatch.setattr(vw, "_health_ok", lambda url, timeout=3: up.is_set())
    got = {}
    a = threading.Thread(target=lambda: got.__setitem__(
        "a", vw._ensure_service_once(CFG, "m", "m", ready_timeout=30)))
    a.start()
    for _ in range(300):
        if fake.launches:
            break
        real_sleep(0.01)
    assert len(fake.launches) == 1
    b = threading.Thread(target=lambda: got.__setitem__(
        "b", vw._ensure_service_once(CFG, "m", "m", ready_timeout=30)))
    b.start()
    real_sleep(0.3)                     # B arrives while A is still starting
    assert len(fake.launches) == 1, "a starting engine was duplicated"
    up.set()
    a.join(5)
    b.join(5)
    assert got["a"] == got["b"]
    assert len(fake.launches) == 1


def test_a_busy_engine_is_kept_not_replaced(fake, capsys):
    fake.state["healthy"].add("http://127.0.0.1:41000")
    first = vw._ensure_service_once(CFG, "m", "m", ready_timeout=30)
    assert len(fake.launches) == 1
    fake.state["healthy"].clear()                   # alive, but not answering
    again = vw._ensure_service_once(CFG, "m", "m", ready_timeout=30)
    assert again == first
    assert len(fake.launches) == 1, "a live engine was replaced"
    assert "BUSY" in capsys.readouterr().out


def test_a_dead_engine_is_replaced(fake):
    fake.state["healthy"].add("http://127.0.0.1:41000")
    vw._ensure_service_once(CFG, "m", "m", ready_timeout=30)
    fake.launches[0].returncode = -9                # the process is gone
    fake.state["healthy"].add("http://127.0.0.1:41001")
    vw._ensure_service_once(CFG, "m", "m", ready_timeout=30)
    assert len(fake.launches) == 2
