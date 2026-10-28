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

"""13: progress for ONE request, not a single global slot.

`/v1/images/progress` and its video/audio siblings reported one module-level
dict, so with two people generating at once neither could tell their own
request's steps from the other's -- and a client that cares about its own
progress has to stop reading them, which is what happened.

A request may name itself (`request_id` in the body, or an `X-Request-Id`
header) and ask only about itself. The name is used as the TASK id, so this is
a view over the existing task registry rather than a second place progress is
written.
"""

import pytest

from codai.api import request_progress as rp
from codai.tasks import task_registry


# ------------------------------------------------------------------ the id

@pytest.mark.parametrize("raw,expect", [
    ("abc-123", "abc-123"),
    ("  spaced  ", "spaced"),
    ("a/b\\c", "abc"),                      # path separators dropped
    ("tab\there", "tabhere"),               # control characters dropped
    ("req.1:2-3_4", "req.1:2-3_4"),         # the safe punctuation survives
    ("", None),
    ("   ", None),
    (None, None),
    ("///", None),                          # nothing safe left == no id
])
def test_a_client_id_is_reduced_to_something_safe(raw, expect):
    assert rp.sanitize(raw) == expect


def test_a_very_long_id_is_bounded():
    got = rp.sanitize("x" * 5000)
    assert got is not None and len(got) == rp.MAX_LEN


def test_the_body_field_wins_over_the_header():
    """A caller that put the id in the payload meant that one; a header may
    have been added by a proxy on the way."""
    class _Body:
        request_id = "from-body"

    class _Http:
        headers = {"X-Request-Id": "from-header"}

    assert rp.from_request(_Body(), _Http()) == "from-body"


def test_the_header_is_used_when_the_body_has_no_id():
    class _Body:
        request_id = None

    class _Http:
        headers = {"X-Request-Id": "from-header"}

    assert rp.from_request(_Body(), _Http()) == "from-header"


def test_no_id_anywhere_is_none():
    class _Body:
        request_id = None

    class _Http:
        headers = {}

    assert rp.from_request(_Body(), _Http()) is None
    assert rp.from_request(None, None) is None


def test_a_dict_body_works_too():
    assert rp.from_request({"request_id": "d1"}, None) == "d1"


# ------------------------------------------------- carrying it to the task

def test_the_id_is_claimed_exactly_once():
    """One request registers more than one task.

    A short "loading" task is registered before the generation itself. If both
    adopted the caller's id the second would overwrite the first under the same
    key and the client would be served whichever happened to be last.
    """
    rp.set_current("only-once")
    assert rp.take() == "only-once"
    assert rp.take() is None, "a second task also claimed the caller's id"


def test_peek_does_not_consume():
    rp.set_current("peeked")
    assert rp.peek() == "peeked"
    assert rp.peek() == "peeked"
    assert rp.take() == "peeked"


def test_no_id_set_means_tasks_keep_their_own():
    rp.set_current(None)
    assert rp.take() is None


# --------------------------------------------------------------- the view

@pytest.fixture
def a_task():
    tid = "test-req-" + "abc123"
    task_registry.remove(tid) if hasattr(task_registry, "remove") else None
    yield tid
    try:
        task_registry.remove(tid)
    except Exception:
        pass


def test_an_unknown_id_has_no_view(a_task):
    """So the endpoint can 404 and the client stops polling on a definite
    answer rather than reading a stale global slot."""
    assert rp.view("nothing-was-ever-registered-here") is None


def test_a_queued_request_reports_queued(a_task):
    task_registry.register("image", total=0, status="queued", task_id=a_task)
    v = rp.view(a_task)
    assert v["phase"] == "queued"
    assert v["active"] is True
    assert v["pct"] == 0


def test_a_running_request_with_no_step_budget_reports_loading(a_task):
    """A client holds the bar during a load rather than animating a total it
    does not have yet."""
    task_registry.register("image", total=0, task_id=a_task)
    task_registry.start(a_task)
    v = rp.view(a_task)
    assert v["phase"] == "loading", v
    assert v["total"] == 0


def test_a_running_request_with_steps_reports_generating_and_a_pct(a_task):
    task_registry.register("image", total=20, model="m", task_id=a_task)
    task_registry.start(a_task)
    task_registry.step(a_task, 5)
    v = rp.view(a_task)
    assert v["phase"] == "generating"
    assert (v["current"], v["total"], v["pct"]) == (5, 20, 25)
    assert v["model"] == "m"
    assert v["request_id"] == a_task


def test_a_finished_request_is_no_longer_active(a_task):
    task_registry.register("image", total=4, task_id=a_task)
    task_registry.start(a_task)
    task_registry.step(a_task, 4)
    task_registry.finish(a_task, "done")
    v = rp.view(a_task)
    assert v is None or v["active"] is False, v


def test_two_requests_do_not_see_each_others_steps():
    """The whole point of 13."""
    a, b = "test-req-aaa", "test-req-bbb"
    try:
        task_registry.register("image", total=10, task_id=a)
        task_registry.register("image", total=100, task_id=b)
        task_registry.start(a)
        task_registry.start(b)
        task_registry.step(a, 1)
        task_registry.step(b, 50)
        va, vb = rp.view(a), rp.view(b)
        assert (va["current"], va["total"], va["pct"]) == (1, 10, 10)
        assert (vb["current"], vb["total"], vb["pct"]) == (50, 100, 50)
    finally:
        for t in (a, b):
            try:
                task_registry.remove(t)
            except Exception:
                pass


def test_the_generation_models_declare_the_field():
    """So it is discoverable in /openapi.json, not only accepted as an extra."""
    from codai.pydantic.imagerequest import ImageGenerationRequest
    from codai.pydantic.videorequest import VideoGenerationRequest
    from codai.pydantic.audiogenrequest import AudioGenerationRequest
    for model in (ImageGenerationRequest, VideoGenerationRequest,
                  AudioGenerationRequest):
        assert "request_id" in model.model_fields, model.__name__
