"""The pods page draws host RAM and reserved VRAM as two separate bars.

They used to share one cell: a bar for host RAM with VRAM printed underneath as
a line of text. That made the GPU side of a rented pod the one number an
operator had to read differently from everything else on the row, and on a page
whose whole job is "what is this A40 doing", the GPU is not a footnote.

The second half of this is a colour decision, which is why it is tested rather
than left to taste. vLLM reserves its gpu_memory_utilization share at boot, so
VRAM reads ~90% from the first second on every healthy pod. Run that through the
green/amber/red bands used for load and every pod is permanently red -- an alarm
that is always on, which an operator correctly learns to ignore, and then the
real red ones (host RAM near the OOM ceiling, which has actually killed pods)
get ignored with it. So VRAM gets a neutral bar and the word "reserved".
"""
import json
import pathlib
import re
import shutil
import subprocess

import pytest

PAGE = pathlib.Path(__file__).resolve().parents[1] / "codai/admin/templates/runpod.html"


@pytest.fixture(scope="module")
def html():
    return PAGE.read_text(encoding="utf-8")


def _fn(html, name):
    m = re.search(r"^function %s\(.*?^}" % re.escape(name), html, re.S | re.M)
    assert m, f"{name}() is gone from the page"
    return m.group(0)


def _render(html, tmp_path, pod):
    """Run the page's own _memCell() over a pod payload, in a real JS engine."""
    js = tmp_path / "memcell.js"
    js.write_text(
        _fn(html, "_meter") + "\n" + _fn(html, "_memCell") + "\n"
        + "console.log(_memCell(%s));" % json.dumps(pod),
        encoding="utf-8")
    proc = subprocess.run(["node", str(js)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr.strip()[:800]
    return proc.stdout


def _fills(out):
    return re.findall(r"width:([0-9.]+)%", out)


needs_node = pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")


@needs_node
def test_a_pod_with_both_readings_gets_two_labelled_bars(html, tmp_path):
    out = _render(html, tmp_path,
                  {"ram_used_gb": 31.4, "ram_total_gb": 125.0,
                   "vram_used_gb": 43.2, "vram_total_gb": 48.0})
    assert len(_fills(out)) == 2, "expected one bar for RAM and one for VRAM"
    assert ">RAM<" in out and ">VRAM<" in out, "the bars are not named"
    # Proportions are each cell's own, not a shared total.
    assert _fills(out) == ["25.12", "90"]
    assert "31.4/125 GB" in out and "43.2/48 GB" in out


@needs_node
def test_reserved_vram_is_not_painted_as_an_alarm(html, tmp_path):
    """90% reserved is how a healthy pod boots, so it must not read as red."""
    out = _render(html, tmp_path,
                  {"ram_used_gb": 12.0, "ram_total_gb": 125.0,
                   "vram_used_gb": 43.2, "vram_total_gb": 48.0})
    assert "#ef4444" not in out, "a freshly booted pod is drawn as critical"
    assert "reserved" in out, "the VRAM bar must say it is a reservation"


@needs_node
def test_host_ram_keeps_its_warning_bands(html, tmp_path):
    """Host RAM is the ceiling that has really killed pods; it stays banded."""
    hot = _render(html, tmp_path, {"ram_used_gb": 118.0, "ram_total_gb": 125.0})
    assert "#ef4444" in hot, "host RAM at 94% is not flagged"
    cool = _render(html, tmp_path, {"ram_used_gb": 12.0, "ram_total_gb": 125.0})
    assert "#22c55e" in cool and "#ef4444" not in cool


@needs_node
@pytest.mark.parametrize("pod,bars", [
    ({"ram_used_gb": 12.0, "ram_total_gb": 125.0}, 1),
    ({"vram_used_gb": 43.2, "vram_total_gb": 48.0}, 1),
    ({}, 0),
])
def test_missing_telemetry_draws_no_bar_at_all(html, tmp_path, pod, bars):
    """An empty bar reads as "idle"; a booting or unreachable pod is "unknown"."""
    out = _render(html, tmp_path, pod)
    assert len(_fills(out)) == bars
    if not bars:
        assert 'class="dim"' in out, "a pod with no telemetry must show a dash"
