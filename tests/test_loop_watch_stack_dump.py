"""A loop stall names its culprit while it is still happening.

Rented OCR pods froze for 108-784 s under load and said only "blocked for N s".
There is no shell on a pod to attach a profiler, so the process must print the
loop thread's stack itself, from a thread that runs while the loop cannot.
"""
import asyncio
import time

from codai.api import loop_watch


def _block_the_loop_in_a_named_function(seconds):
    time.sleep(seconds)


def test_a_blocking_call_on_the_loop_is_named_in_the_dump(monkeypatch, capsys):
    monkeypatch.setattr(loop_watch, "DUMP_AFTER_SECONDS", 0.6)
    loop_watch._state.update(beat=0.0, dumps=0)

    async def main():
        loop_watch.install(threshold=0.3, interval=0.1)
        await asyncio.sleep(0.3)
        _block_the_loop_in_a_named_function(2.0)
        await asyncio.sleep(0.3)

    asyncio.run(main())
    out = capsys.readouterr().out
    assert "STALL IN PROGRESS" in out
    assert "_block_the_loop_in_a_named_function" in out
    assert "end of stack" in out


def test_a_healthy_loop_dumps_nothing(monkeypatch, capsys):
    monkeypatch.setattr(loop_watch, "DUMP_AFTER_SECONDS", 0.6)
    loop_watch._state.update(beat=0.0, dumps=0)

    async def main():
        loop_watch.install(threshold=0.3, interval=0.1)
        for _ in range(15):
            await asyncio.sleep(0.1)

    asyncio.run(main())
    assert "STALL IN PROGRESS" not in capsys.readouterr().out


def test_the_dumper_stops_with_its_loop(monkeypatch):
    import threading
    monkeypatch.setattr(loop_watch, "DUMP_AFTER_SECONDS", 0.4)

    async def main():
        loop_watch.install(threshold=0.3, interval=0.1)
        await asyncio.sleep(0.2)

    asyncio.run(main())
    time.sleep(0.6)
    assert not [t for t in threading.enumerate() if t.name == "loop-stall-dumper"]
