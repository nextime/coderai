"""A LongCat generation must be cancellable, like every other video path.

Every diffusers-based video path calls task_registry.raise_if_cancelled() between
denoise steps, so the Tasks page can stop it. LongCat did not: it was never
registered at all. Its poll loop even claimed, in its own docstring, to "carry
cancellation across the process boundary" — and only mirrored progress.

That left the LONGEST-running video job in the system as the only one nothing
could stop. A 93-frame bf16 render offloading across PCIe can run for a very long
time, and the only way out was killing the service by hand.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
VIDEO = ROOT / "codai" / "api" / "video.py"


def _longcat_body() -> str:
    src = VIDEO.read_text(encoding="utf-8")
    start = src.index("async def _generate_longcat")
    nxt = re.search(r"\n(async def |def )", src[start + 10:])
    return src[start:start + 10 + nxt.start()] if nxt else src[start:]


def test_the_generation_is_registered_as_a_task():
    body = _longcat_body()
    assert "task_registry.register(" in body, "not on the Tasks page at all"
    assert "task_registry.start(" in body


def test_it_checks_for_cancellation():
    assert "task_registry.raise_if_cancelled(" in _longcat_body()


def test_cancelling_stops_the_service():
    """It runs in another process, so cancelling cannot raise inside it — stopping
    the service is what actually ends the work."""
    body = _longcat_body()
    assert "longcat_worker.stop_service(" in body


def test_progress_reaches_the_task_not_only_the_bar():
    body = _longcat_body()
    assert "task_registry.step(" in body, "the Tasks page would show no progress"


def test_the_task_is_closed_on_every_exit():
    body = _longcat_body()
    for status in ('"cancelled"', '"error"', '"done"'):
        assert f"task_registry.finish(_tid, {status}" in body, status


def test_a_cancelled_generation_does_not_report_success():
    body = _longcat_body()
    assert "499" in body, "a cancelled request should not return a video or a 200"


def test_the_docstring_no_longer_claims_what_the_code_does_not_do():
    body = _longcat_body()
    if "carry cancellation across the process boundary" in body:
        assert "raise_if_cancelled" in body, \
            "the docstring promises cancellation; the code must deliver it"
