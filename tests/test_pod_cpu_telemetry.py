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

"""A pod's CPU has to be visible, because it is the ceiling nothing else shows.

Measured on a production orchestrator, 2026-11-02: an OCR pod answered 200 to
everything it finished, reported its KV cache a third full, held 45 of 64 pages
in flight on average, and the run still crawled. Every GPU-side number said
"this pod has room" -- which is exactly what a pod whose CORES are pinned looks
like, because the work ahead of the GPU (decoding pages to pixels, moving
bytes) never reaches the card. "Saturated" and "starved" were indistinguishable
on the page an operator opens when something is wrong.

So the pod reports its own CPU, next to the memory it already reported:

* as a share of ITS QUOTA, not of the machine's cores. A pod pinned to 8 cores
  of a 128-core host is finished at 8, and dividing by 128 reports 6% busy
  while every request queues behind the scheduler;
* from cgroup accounting first, for the same reason host memory is: inside a
  container /proc/stat counts the HOST's cores, so a noisy neighbour would be
  billed to this pod;
* as a RATE, which needs two samples -- so the first reading after a process
  starts carries the core count and no utilisation. The alternative is blocking
  a status-page request for a sampling interval, which is worse than one dash.

The GPU figure needs no pod-side change (`util` is already in the `vram` block,
live-checked on both production pods: 0.0% on an idle LLM pod, 100.0% on an OCR
pod mid-run) -- what it needed was a BAR, which is asserted here against the
template, because a number that an operator has to find in a text run-on is a
number nobody reads at 2am.
"""

import inspect
import os
import re
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.api import serving_load as SL

_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "codai", "admin", "templates", "runpod.html")


def _burn(seconds: float) -> None:
    """Keep one core busy. The reading is a rate; it needs time to pass."""
    end = time.monotonic() + seconds
    x = 0
    while time.monotonic() < end:
        x += 1            # noqa: B018 - the point is the cycles, not the value


# ---------------------------------------------------------------------------
# the quota, not the machine
# ---------------------------------------------------------------------------

def test_the_core_count_is_a_positive_number():
    cores = SL._cpu_cores()
    assert cores is not None and cores > 0


def test_a_cpu_quota_is_preferred_over_the_machines_cores(monkeypatch, tmp_path):
    """A pod with a 4-core quota on a 128-core host is saturated at 4. This is
    the whole reason the figure is a percentage of the quota: the same load
    reads 100% against the quota and 3% against the host."""
    quota = tmp_path / "cpu.max"
    quota.write_text("400000 100000\n")           # 4 cores
    real_open = open

    def _open(path, *a, **kw):
        if str(path) == "/sys/fs/cgroup/cpu.max":
            return real_open(quota, *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", _open)
    assert SL._cpu_cores() == 4.0


def test_an_unlimited_quota_falls_back_to_the_affinity_mask(monkeypatch, tmp_path):
    """`max` means no quota -- then the cpuset the process may run on is the
    ceiling, and that is what sched_getaffinity reports."""
    quota = tmp_path / "cpu.max"
    quota.write_text("max 100000\n")
    real_open = open

    def _open(path, *a, **kw):
        if str(path) == "/sys/fs/cgroup/cpu.max":
            return real_open(quota, *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", _open)
    assert SL._cpu_cores() == float(len(os.sched_getaffinity(0)))


# ---------------------------------------------------------------------------
# the rate
# ---------------------------------------------------------------------------

def test_the_first_reading_has_no_utilisation_but_does_have_the_cores():
    """Utilisation is a delta. Reporting one from a single sample would mean
    reporting the process's average since boot as its load right now."""
    SL.forget()
    first = SL.host_cpu()
    assert "cpu_util" not in first
    assert first.get("cpu_cores")


def test_the_second_reading_measures_real_work():
    SL.forget()
    SL.host_cpu()
    _burn(0.7)
    got = SL.host_cpu()
    assert "cpu_util" in got, got
    assert 0.0 < got["cpu_util"] <= 100.0
    assert got.get("cpu_source") in ("cgroup", "procstat")


def test_it_is_never_outside_nought_to_a_hundred():
    """The bar is drawn from this number. A quota smaller than the cores the
    fallback counts (a cpuset-limited pod reading /proc/stat) would otherwise
    produce 400%, and a bar 4x the width of its cell."""
    SL.forget()
    SL.host_cpu()
    _burn(0.6)
    with SL._CPU_LOCK:
        SL._CPU_SAMPLE["busy"] = 0.0           # as if the counter started now
    got = SL.host_cpu()
    assert got.get("cpu_util") is None or 0.0 <= got["cpu_util"] <= 100.0


def test_a_counter_that_went_backwards_reports_nothing(monkeypatch):
    """cgroup accounting restarts with the cgroup, and a pod's container is
    replaced in place on an upgrade. A negative delta is not 0% load; it is no
    measurement, and a bar at 0 would read as an idle pod."""
    SL.forget()
    SL.host_cpu()
    _burn(0.6)
    with SL._CPU_LOCK:
        SL._CPU_SAMPLE["busy"] = float(SL._CPU_SAMPLE["busy"]) + 1e9
    assert "cpu_util" not in SL.host_cpu()


def test_two_readings_in_the_same_instant_reuse_the_last_figure():
    """Several open tabs poll together, and the page's own TTL is shorter than
    a sampling interval. Dividing by ~0ms would report quantisation noise as a
    reading."""
    SL.forget()
    SL.host_cpu()
    _burn(0.7)
    first = SL.host_cpu()
    again = SL.host_cpu()                       # immediately: dt < 0.5s
    assert again.get("cpu_util") == first.get("cpu_util")


def test_forget_clears_the_cpu_sample_too():
    SL.forget()
    SL.host_cpu()
    SL.forget()
    assert SL.host_cpu().get("cpu_util") is None


# ---------------------------------------------------------------------------
# it has to reach the page
# ---------------------------------------------------------------------------

def test_the_snapshot_carries_cpu(monkeypatch):
    """`/internal/engine-state` is the only thing the orchestrator asks a pod.
    A reading the snapshot does not include is a reading nobody will ever see."""
    monkeypatch.setattr(SL, "engine_load", lambda url: {})
    monkeypatch.setattr(SL, "host_memory", lambda: {"ram_used_gb": 1.0})
    SL.forget()
    SL.snapshot("")
    _burn(0.7)
    out = SL.snapshot("")
    assert "cpu_util" in out and "cpu_cores" in out
    assert out["ram_used_gb"] == 1.0            # and it did not displace memory


def test_the_collector_forwards_the_cpu_fields(monkeypatch):
    """pod_telemetry copies a NAMED list of keys out of the pod's `load` block,
    so a field the pod reports and the list omits is dropped in silence."""
    from codai.api import pod_telemetry as PT

    class _FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"vram": {"devices": [{"total": 48.0, "free": 8.0,
                                          "util": 97.0, "temp": 61.0}]},
                    "load": {"cpu_util": 88.5, "cpu_cores": 16,
                             "ram_used_gb": 12.0, "ram_total_gb": 60.0,
                             "kv_cache_pct": 31.0}}

    class _FakePodHttp:
        @staticmethod
        def get(url, **kw):
            return _FakeResp()

    # BOTH lookup routes: the forwards do `from codai.api import pod_http`,
    # which resolves the ATTRIBUTE on the already-imported package and only
    # falls back to sys.modules -- so patching sys.modules alone passes when
    # this file runs alone and tests nothing in the full suite.
    import codai.api
    monkeypatch.setattr(codai.api, "pod_http", _FakePodHttp, raising=False)
    monkeypatch.setitem(sys.modules, "codai.api.pod_http", _FakePodHttp)

    PT.forget()
    got = PT._fetch("pod1", "https://pod/v1", "k")
    assert got["cpu_util"] == 88.5
    assert got["cpu_cores"] == 16
    assert got["gpu_util"] == 97.0              # already worked; must keep working
    assert got["ram_used_gb"] == 12.0


# ---------------------------------------------------------------------------
# and it has to be a BAR
# ---------------------------------------------------------------------------

def _load_cell_source() -> str:
    with open(_TEMPLATE, encoding="utf-8") as fh:
        src = fh.read()
    body = src.split("function _loadCell(p){", 1)
    assert len(body) == 2, "the Load cell is not built by _loadCell any more"
    return body[1].split("\nfunction ", 1)[0]


def test_gpu_and_cpu_are_drawn_as_meters_not_as_text():
    """They were a run-on of text bits next to the KV bar -- `31% KV` with
    `12 run · 97% gpu · 61C` underneath. The request was a visible level, and
    the reason is that a bar is read at a glance and a percentage in a sentence
    is not."""
    cell = _load_cell_source()
    assert re.search(r"_meter\(p\.gpu_util", cell), cell
    assert re.search(r"_meter\(p\.cpu_util", cell), cell


def test_every_bar_in_the_cell_is_labelled():
    """Three bars stacked in one cell cannot be told apart by their order --
    especially when a missing reading removes one of them."""
    cell = _load_cell_source()
    for tag in ("'KV'", "'GPU'", "'CPU'"):
        assert f"tag: {tag}" in cell, tag


def test_the_gpu_bar_is_not_banded_and_the_cpu_bar_is():
    """A saturated GPU is the GOAL of a throughput run; painting it red marks
    every pod that is earning its rent as broken, and an operator learns to
    ignore the colour -- which is worse than no colour. A pinned CPU is a wall,
    so it keeps the green/amber/red bands."""
    cell = _load_cell_source()
    # Up to the end of the `_meter(...)` CALL: a template literal in the label
    # contains braces of its own, so splitting on `}` reads a few characters of
    # the label and then asserts nothing -- which is how this test first passed
    # the wrong code and then failed the right code.
    gpu = cell.split("_meter(p.gpu_util", 1)[1].split("));", 1)[0]
    assert "band: false" in gpu, gpu
    cpu = cell.split("_meter(p.cpu_util", 1)[1].split("));", 1)[0]
    assert "tag: 'CPU'" in cpu, cpu
    assert "band: false" not in cpu, cpu


def test_a_pod_that_reports_nothing_still_draws_no_bars():
    """A booting or unreachable pod has no telemetry, and an empty bar reads as
    "idle" when the truth is "unknown"."""
    cell = _load_cell_source()
    assert "if (p.cpu_util != null)" in cell
    assert "if (p.gpu_util != null)" in cell
    assert "if (!bars.length)" in cell


def test_the_old_text_only_gpu_bit_is_gone():
    """Otherwise the same number appears twice in one cell, once as a bar and
    once in the line under it."""
    cell = _load_cell_source()
    assert "% gpu`" not in cell


def test_the_template_is_still_valid_javascript():
    """These assertions read the template as text, which cannot catch a stray
    comment delimiter -- and the bars were added by splitting an existing
    comment block in two."""
    node = None
    for cand in ("node", "nodejs"):
        from shutil import which
        if which(cand):
            node = cand
            break
    if not node:
        pytest.skip("no node available to parse the template's script")
    with open(_TEMPLATE, encoding="utf-8") as fh:
        src = fh.read()
    script = src.split("{% block scripts %}", 1)[1].split("</script>", 1)[0]
    script = script.split("<script>", 1)[1]
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cell.js")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(script)
        r = subprocess.run([node, "--check", p], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
