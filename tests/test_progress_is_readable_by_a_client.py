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

"""A client polling a distributed run must be able to tell how far along it is.

`step`/`total` carried the ROUND number against the planned round count, and
that stopped being a position the moment a run began ending on dataset
coverage instead: a fleet smaller than planned takes more exchanges than
`rounds`, a larger one fewer, so "round 4 of 16" says nothing about how much
has been trained. The only honest numbers are blocks covered against blocks
wanted, and a client should not have to parse an English message to get them.

They are exposed the same way every other long operation's are --
`GET /v1/loras/progress` -- and mirrored into the shared snapshot so a poll
without a job id sees them too.
"""

import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from codai.api import loras


_FIELDS = ("blocks_done", "blocks_total", "percent",
           "examples_done", "examples_total",
           "exchange", "epochs", "pods", "pods_target", "pods_ceiling")


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(loras, "_require_api_auth", lambda: True,
                        raising=False)
    monkeypatch.setattr(loras, "_loras_dir", lambda: str(tmp_path),
                        raising=False)
    monkeypatch.setattr(loras, "_jobs", {})
    monkeypatch.setattr(loras, "_jobs_lock", threading.Lock())
    app = FastAPI()
    app.include_router(loras.router)
    app.dependency_overrides[loras._require_api_auth] = lambda: True
    return TestClient(app)


def test_the_progress_endpoint_reports_coverage(client):
    loras._new_job("job-p", target="text", name="adapter-p",
                   status="training", session="sess-p")
    loras._update_job("job-p", blocks_done=37, blocks_total=252,
                      percent=14.7, examples_done=1073,
                      examples_total=7308, exchange=3, epochs=1.0,
                      pods=5, pods_target=16, pods_ceiling=16,
                      hourly_usd=2.95, force=True)

    r = client.get("/v1/loras/progress?job=job-p")
    assert r.status_code == 200
    got = r.json()
    for f in _FIELDS:
        assert f in got, f"{f} is not visible to a client"
    assert got["blocks_done"] == 37
    assert got["blocks_total"] == 252
    assert got["percent"] == 14.7
    assert got["examples_done"] == 1073
    assert got["pods"] == 5 and got["pods_target"] == 16
    assert got["hourly_usd"] == 2.95


def test_a_client_that_lost_the_job_id_still_gets_them(client):
    """Recovery by session, which is what their client actually polls."""
    loras._new_job("job-q", target="text", name="adapter-q",
                   status="training", session="digesta-a4v2")
    loras._update_job("job-q", blocks_done=9, blocks_total=252,
                      percent=3.6, exchange=1, pods=3, force=True)
    r = client.get("/v1/loras/progress?session=digesta-a4v2")
    assert r.status_code == 200
    got = r.json()
    assert got["blocks_done"] == 9
    assert got["percent"] == 3.6
    assert got["pods"] == 3


def test_the_fields_are_mirrored_into_the_shared_snapshot():
    """A poll with neither job nor session returns the global blob, and it has
    to carry the same numbers or the no-argument form is useless."""
    for f in _FIELDS:
        assert f in loras._MIRROR_FIELDS, f


def test_percent_is_derived_from_coverage_not_from_the_round_number():
    """The whole point: a round index against a planned total can read 100%
    while a fifth of the data has been trained."""
    import inspect
    src = inspect.getsource(loras._text_train_distributed)
    assert "coord.coverage()" in src
    i = src.index("percent=round(")
    # it comes from the blocks, not from `rnd` or `rounds`
    window = src[max(0, i - 400):i + 120]
    assert "_done_b" in window or "_dn" in window


def test_an_unknown_job_still_answers_a_shape_a_client_can_read(client):
    r = client.get("/v1/loras/progress?job=job-nope")
    assert r.status_code == 200
    got = r.json()
    assert got["active"] is False
    assert got["status"] == "unknown"
