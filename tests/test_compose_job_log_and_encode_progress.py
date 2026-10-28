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

"""13b: finer compose progress, and the job's own log.

A client logs every change of `stage`/`message` and moves its bar on
`progress`. Some steps are long with no change at all -- the encode from 76 to
95 %, a presenter being generated, a model loading -- so the bar sat still for
minutes, which reads as a stall. The encode now reports frames written, and the
job's log (already kept, previously unreachable) is served both as a tail in
the status document and incrementally.
"""

import subprocess

import pytest

from codai.api import compose as C
from codai.compose import render as rnd


def _ffmpeg_ok():
    try:
        from codai.compose import media
        subprocess.run([media.ffmpeg_bin(), "-version"],
                       capture_output=True, timeout=30)
        return True
    except Exception:
        return False


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")


# ------------------------------------------------------------------ the log

def test_the_status_document_carries_a_log_tail_and_the_real_length():
    rec = {"id": "j1", "status": "running", "stage": "render", "progress": 80.0,
           "log": [f"line {i}" for i in range(200)]}
    doc = C._public(rec)
    assert doc["log_total"] == 200, "a client cannot tell it is seeing a tail"
    assert len(doc["log"]) == C._LOG_TAIL
    assert doc["log"][-1] == "line 199", "the tail must be the RECENT lines"


def test_a_job_with_no_log_still_answers():
    doc = C._public({"id": "j2", "status": "queued"})
    assert doc["log"] == [] and doc["log_total"] == 0


def test_the_log_endpoint_serves_only_what_is_new():
    """A client passes back the previous `next`, so each poll is incremental."""
    import asyncio
    jid = "test-log-job"
    C._jobs[jid] = {"id": jid, "status": "running",
                    "log": ["a", "b", "c", "d"]}
    try:
        first = asyncio.run(C.compose_log(jid, since=0, _auth=None))
        assert first["log"] == ["a", "b", "c", "d"]
        assert first["next"] == 4 and first["dropped"] is False

        C._jobs[jid]["log"].append("e")
        second = asyncio.run(C.compose_log(jid, since=first["next"], _auth=None))
        assert second["log"] == ["e"], second
        assert second["next"] == 5
    finally:
        C._jobs.pop(jid, None)


def test_polling_past_the_end_is_empty_not_an_error():
    """A client that polls once more after the job finished gets a clean tail."""
    import asyncio
    jid = "test-log-job-2"
    C._jobs[jid] = {"id": jid, "status": "done", "log": ["x"]}
    try:
        got = asyncio.run(C.compose_log(jid, since=99, _auth=None))
        assert got["log"] == []
        assert got["dropped"] is True, "a since past the end should say so"
    finally:
        C._jobs.pop(jid, None)


def test_an_unknown_job_log_is_a_404():
    import asyncio
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        asyncio.run(C.compose_log("no-such-job", since=0, _auth=None))
    assert ei.value.status_code == 404


def test_the_log_is_the_live_list_not_a_snapshot():
    """The record holds the SAME list the job appends to, so a poll sees lines
    as they are written rather than as they were at some earlier moment."""
    import inspect
    src = inspect.getsource(C._run_job)
    assert "_update(job_id, log=log_lines)" in src, (
        "the job's log list is no longer published on the record")


# -------------------------------------------------------- encode progress

@needs_ffmpeg
def test_run_reports_frames_written(tmp_path):
    """The hook that makes the encode band move."""
    out = str(tmp_path / "out.mp4")
    seen = []
    rnd.run(["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2",
             "-pix_fmt", "yuv420p", out],
            on_frame=seen.append, timeout=120)
    assert seen, "ffmpeg wrote frames but none were reported"
    assert seen == sorted(seen), f"the frame count went backwards: {seen}"
    assert seen[-1] >= 10, f"only reached {seen[-1]} frames of ~20"


@needs_ffmpeg
def test_a_command_without_the_hook_is_unchanged(tmp_path):
    """-progress is asked for only when somebody is listening."""
    out = str(tmp_path / "plain.mp4")
    rnd.run(["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
             "-pix_fmt", "yuv420p", out], timeout=120)
    import os
    assert os.path.getsize(out) > 0


@needs_ffmpeg
def test_a_raising_callback_does_not_break_the_encode(tmp_path):
    """Reporting progress must never cost the render."""
    out = str(tmp_path / "boom.mp4")

    def _boom(n):
        raise RuntimeError("callback exploded")

    rnd.run(["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
             "-pix_fmt", "yuv420p", out], on_frame=_boom, timeout=120)
    import os
    assert os.path.getsize(out) > 0, "a broken callback lost the output"


def test_final_render_accepts_the_hook():
    import inspect
    assert "on_frame" in inspect.signature(rnd.final_render).parameters


def test_compose_drives_its_encode_band_from_frames():
    import inspect
    src = inspect.getsource(C._run_job)
    assert "encoding {n}/{_enc_frames} frames" in src or \
           'f"encoding {n}/{_enc_frames} frames"' in src, (
        "the encode no longer reports frames written")
    assert "on_frame=_on_frame" in src, "the hook is not passed to final_render"
