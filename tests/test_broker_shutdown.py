"""Cancelling the broker client must actually stop it.

The incident this file exists for: a full test run reached 46 GB RSS + 20 GB swap on
a 54 GB machine and the OOM killer took the live coderai server with it. The shape was
``except asyncio.CancelledError: pass`` wrapped around an await whose job was to
collect a CHILD task. asyncio delivers a cancellation as a CancelledError at the next
await point, so that handler caught two different things: the child saying it stopped,
and the caller saying *we* must stop. Swallowing the second one left run_forever
looping — and with the reconnect delay mocked to zero by the test that drove it, the
loop spun at full speed allocating ~78 MB/s, because AsyncMock records every call.

So these tests are about cancellation being observed, not about websockets. They each
bound the loop, because the failure mode under test is precisely "the loop does not
stop" and a test for that must not be the thing that hangs.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.broker.client import BrokerClient, being_cancelled
from codai.broker.config import BrokerRuntimeConfig

pytestmark = pytest.mark.anyio("asyncio")

# Generous next to the two reconnects these tests need, and tiny next to the
# ~400,000 iterations the unfixed loop managed in the 5 seconds I measured.
SPIN_LIMIT = 200


class _Runaway(BaseException):
    """BaseException: anything else is caught by run_forever's reconnect handler."""


def _bounded_sleep(calls):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        calls.append(delay)
        if len(calls) > SPIN_LIMIT:
            raise _Runaway(f"the reconnect loop slept {len(calls)} times")
        await real_sleep(0)

    return fake_sleep


def _client(**kwargs):
    runtime = BrokerRuntimeConfig(enabled=True, reconnect_initial_delay_seconds=1,
                                  reconnect_max_delay_seconds=4, **kwargs)
    return BrokerClient(runtime)


class _DisconnectingWebSocket:
    def __init__(self):
        self.closed = False

    async def recv(self):
        raise RuntimeError("disconnect")

    async def close(self):
        self.closed = True


# ------------------------------------------------------------------ the mechanism
async def test_being_cancelled_is_false_normally():
    assert being_cancelled() is False


async def test_being_cancelled_is_true_once_we_have_been_cancelled():
    """This is the whole distinction the fix rests on."""
    seen = []

    async def victim():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            seen.append(being_cancelled())
            raise

    task = asyncio.create_task(victim())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen == [True]


# ------------------------------------------------------------------- run_forever
async def test_cancelling_run_forever_stops_it():
    """The incident, in miniature: zero reconnect delay plus a cancel that was ignored."""
    client = _client()
    attempts = []
    sleep_calls = []
    reconnected = asyncio.Event()

    async def fake_connect_and_register():
        attempts.append("connect")
        client.websocket = _DisconnectingWebSocket()
        if len(attempts) >= 2:
            reconnected.set()

    client.connect_and_register = AsyncMock(side_effect=fake_connect_and_register)

    with patch("codai.broker.client.asyncio.sleep",
               new=AsyncMock(side_effect=_bounded_sleep(sleep_calls))):
        task = asyncio.create_task(client.run_forever())
        await asyncio.wait_for(reconnected.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

    assert len(attempts) < SPIN_LIMIT, (
        f"{len(attempts)} reconnects: the cancel was swallowed and the loop spun on")
    assert task.cancelled()


async def test_the_cancel_is_honoured_even_while_the_heartbeat_is_being_collected():
    """The exact await the old handler sat on: _stop_heartbeat_task's `await task`.

    A heartbeat that takes a moment to finish is the window in which the cancellation
    arrives, and the window the old code lost it in.
    """
    client = _client()
    in_cleanup = asyncio.Event()

    class _SlowStop(_DisconnectingWebSocket):
        async def close(self):
            in_cleanup.set()
            await asyncio.sleep(0)
            self.closed = True

    socket = _SlowStop()

    async def fake_connect_and_register():
        client.websocket = socket

    client.connect_and_register = AsyncMock(side_effect=fake_connect_and_register)
    task = asyncio.create_task(client.run_forever())
    await asyncio.wait_for(in_cleanup.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)


async def test_shutdown_still_closes_the_websocket():
    """Honouring the cancel must not cost the cleanup.

    run_forever's cancellation handler awaits three things, and every one of those
    awaits is itself a cancellation point — so with the cancel left pending they would
    re-raise at the first of them and strand an open socket. It clears the cancel for
    the duration and re-raises at the end.
    """
    client = _client(heartbeat_interval_seconds=60)
    connected = asyncio.Event()
    socket = _DisconnectingWebSocket()

    async def fake_connect_and_register():
        client.websocket = socket
        connected.set()
        await asyncio.sleep(3600)       # hold the loop open, inside the try

    client.connect_and_register = AsyncMock(side_effect=fake_connect_and_register)
    task = asyncio.create_task(client.run_forever())
    await asyncio.wait_for(connected.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)

    assert socket.closed, "the websocket was left open on shutdown"
    assert client.websocket is None
    assert client._heartbeat_task is None


# ------------------------------------------------------------------- the helpers
async def test_stopping_the_heartbeat_swallows_the_childs_cancellation():
    """The half that was always right: a child reporting that it stopped is expected."""
    client = _client()

    async def heartbeat():
        await asyncio.sleep(3600)

    client._heartbeat_task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    await client._stop_heartbeat_task()         # must not raise
    assert client._heartbeat_task is None


async def test_stopping_the_heartbeat_re_raises_our_own_cancellation():
    client = _client()
    outcome = []

    async def heartbeat():
        await asyncio.sleep(3600)

    async def caller():
        client._heartbeat_task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        try:
            await client._stop_heartbeat_task()
        except asyncio.CancelledError:
            outcome.append("raised")
            raise
        outcome.append("swallowed")

    task = asyncio.create_task(caller())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert outcome == ["raised"], outcome


async def test_cancelling_inflight_work_re_raises_our_own_cancellation():
    """Same defect, same shape, one method along — and the one that holds request work."""
    client = _client()
    outcome = []

    async def inflight():
        await asyncio.sleep(3600)

    async def caller():
        client._inflight_tasks.add(asyncio.create_task(inflight()))
        await asyncio.sleep(0)
        try:
            await client._cancel_inflight_tasks()
        except asyncio.CancelledError:
            outcome.append("raised")
            raise
        outcome.append("swallowed")

    task = asyncio.create_task(caller())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert outcome == ["raised"], outcome


async def test_cancelling_inflight_work_still_ignores_a_failed_request():
    """An inflight request that raised is not a shutdown problem."""
    client = _client()

    async def boom():
        raise RuntimeError("request failed")

    client._inflight_tasks.add(asyncio.create_task(boom()))
    await asyncio.sleep(0)
    await client._cancel_inflight_tasks()      # must not raise
    assert not client._inflight_tasks


async def test_a_cancelled_keepalive_reports_itself_as_cancelled():
    """It swallowed its own cancellation and so looked like a clean completion, which
    hides a keepalive still running against a finished request."""
    client = _client()
    task = asyncio.create_task(client._send_keepalives("req-1", interval=3600))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_stopping_the_service_re_raises_our_own_cancellation():
    """BrokerService.stop() collects the client task the same way, and is itself
    awaited from application shutdown — which may already be cancelled."""
    from codai.broker.service import BrokerService

    client = _client()

    async def never_returns():
        await asyncio.sleep(3600)

    client.connect_and_register = AsyncMock(side_effect=never_returns)
    service = BrokerService(client)
    service.start()
    await asyncio.sleep(0)
    outcome = []

    async def caller():
        try:
            await service.stop()
        except asyncio.CancelledError:
            outcome.append("raised")
            raise
        outcome.append("swallowed")

    task = asyncio.create_task(caller())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert outcome == ["raised"], outcome


async def test_stopping_the_service_normally_does_not_raise():
    from codai.broker.service import BrokerService

    client = _client()

    async def never_returns():
        await asyncio.sleep(3600)

    client.connect_and_register = AsyncMock(side_effect=never_returns)
    service = BrokerService(client)
    service.start()
    await asyncio.sleep(0)
    await service.stop()                   # must not raise
    assert service.task is None


def test_no_cancellation_handler_in_the_broker_swallows_a_cancel_blind():
    """Four sites had the same shape; a fifth will be written one day.

    Every `except asyncio.CancelledError:` that wraps an await of a CHILD task has to
    ask whose cancellation it caught. This reads the source because the point is to
    catch a NEW handler, not the ones already covered above.
    """
    import re

    for name in ("client.py", "service.py"):
        src = (ROOT / "codai" / "broker" / name).read_text()
        for match in re.finditer(r"except asyncio\.CancelledError:\n((?:[ \t]*(?:#[^\n]*)?\n)*)"
                                 r"([ \t]*)([^\n]+)", src):
            body = match.group(3).strip()
            assert body != "pass", (
                f"codai/broker/{name}: a CancelledError handler swallows the cancel "
                f"blind. If it is collecting a child task it must re-raise when "
                f"being_cancelled(); if it is its own task's cancellation it must "
                f"re-raise outright.")


# --------------------------------------------------------------- the host guards
def test_the_test_run_has_a_memory_ceiling_and_a_stall_ceiling():
    """Without these, the next swallowed cancel takes the machine down again."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "coderai_test_guards", ROOT / "tests" / "conftest.py")
    guards = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guards)
    assert guards.RSS_LIMIT_MB > 0
    assert guards.STALL_SECONDS > 0
    # Headroom for the real suite (it peaks well under 2 GB), far below a 54 GB host.
    assert 2048 <= guards.RSS_LIMIT_MB <= 16384
    assert hasattr(guards, "_watch_rss")


def test_the_guards_can_be_turned_off_or_retuned_from_the_environment():
    src = (ROOT / "tests/conftest.py").read_text()
    assert "CODERAI_TEST_RSS_LIMIT_MB" in src
    assert "CODERAI_TEST_STALL_SECONDS" in src
    assert "CODERAI_TEST_ABORT_LOG" in src


def test_the_report_survives_the_process_exiting_hard():
    """pytest captures stderr per test and drops the buffer when the process does not
    unwind, so a watchdog that only printed would die silently."""
    src = (ROOT / "tests/conftest.py").read_text()
    assert "file=_open_log()" in src
    assert "os._exit(97)" in src
