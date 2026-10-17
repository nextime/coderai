"""The runtime half: how long did the event loop go without running?

The static check (test_no_blocking_calls_on_the_loop.py) only sees calls it has
a name for. This sees any of them, including a block inside a library, because
it measures the loop's own behaviour: a sleep that asks for 50 ms and returns at
300 ms says the loop was busy elsewhere for 250 ms and served nothing.
"""

import asyncio
import time

import pytest

from codai.api import loop_watch


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _clean():
    loop_watch.reset()
    yield
    loop_watch.reset()


@pytest.mark.anyio
async def test_a_quiet_loop_reports_no_stall(capsys):
    task = loop_watch.install(threshold=0.2, interval=0.02)
    assert task is not None
    await asyncio.sleep(0.2)
    task.cancel()
    assert loop_watch.stats()["stalls"] == 0
    assert "blocked for" not in capsys.readouterr().out


@pytest.mark.anyio
async def test_a_blocked_loop_is_reported_with_its_duration(capsys):
    task = loop_watch.install(threshold=0.2, interval=0.02)
    await asyncio.sleep(0.05)
    time.sleep(0.5)                 # exactly what this exists to catch
    await asyncio.sleep(0.05)
    task.cancel()
    st = loop_watch.stats()
    assert st["stalls"] >= 1
    assert st["worst"] >= 0.4, st
    out = capsys.readouterr().out
    assert "the event loop was blocked for" in out
    assert "no request of any kind was served" in out


@pytest.mark.anyio
async def test_the_worst_stall_is_kept_not_the_last():
    task = loop_watch.install(threshold=0.2, interval=0.02)
    await asyncio.sleep(0.05)
    time.sleep(0.6)
    await asyncio.sleep(0.05)
    time.sleep(0.25)
    await asyncio.sleep(0.05)
    task.cancel()
    st = loop_watch.stats()
    assert st["worst"] >= 0.5
    assert st["blocked_seconds"] >= st["worst"]
    assert st["worst_at"] > 0


@pytest.mark.anyio
async def test_a_stall_shorter_than_the_threshold_is_not_reported(capsys):
    """Or an operator learns to ignore the message."""
    task = loop_watch.install(threshold=1.0, interval=0.02)
    await asyncio.sleep(0.05)
    time.sleep(0.2)
    await asyncio.sleep(0.05)
    task.cancel()
    assert loop_watch.stats()["stalls"] == 0
    assert "blocked for" not in capsys.readouterr().out


def test_install_outside_a_loop_is_not_an_error():
    """Measuring must never be able to stop a process from serving."""
    assert loop_watch.install() is None


@pytest.mark.anyio
async def test_a_nonsense_interval_installs_nothing():
    assert loop_watch.install(threshold=0.0, interval=0.0) is None


@pytest.mark.anyio
async def test_stats_is_a_copy():
    loop_watch.stats()["worst"] = 99.0
    assert loop_watch.stats()["worst"] != 99.0
