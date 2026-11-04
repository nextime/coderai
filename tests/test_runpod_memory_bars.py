"""The pods page draws host RAM and VRAM as two separate, banded usage bars.

They used to share one cell: a bar for host RAM with VRAM printed underneath as
a line of text. That made the GPU side of a rented pod the one number an
operator had to read differently from everything else on the row, and on a page
whose whole job is "what is this A40 doing", the GPU is not a footnote.

The second half of this is a colour decision, which is why it is tested rather
than left to taste, and it has been taken BOTH ways.

It was first drawn neutral, with the word "reserved", on this reasoning: vLLM
claims its gpu_memory_utilization share at boot, so VRAM reads ~90% from the
first second on every healthy pod, and banding it makes every pod permanently
red -- an alarm that is always on, which an operator correctly learns to ignore,
taking the real red ones (host RAM near the OOM ceiling, which has actually
killed pods) with it.

It is now banded, by decision: how full the card is decides whether the pod can
take another model, a longer context or a second engine, and leaving the one
number that says "this card is full" as the only number on the page without a
band did not make the reservation clearer -- it just made the figure easy to
skip. The caveat moved into the label instead, and only onto the pods it is
true of: a pod that reports a KV cache says "incl. reserve", and the KV bar in
the Load column beside it is the one that moves with load. A pod with no KV
cache -- diffusion, OCR, llama.cpp -- is reporting plain allocation, and
attaching a vLLM caveat to it would be the opposite of informative.

What survives from the first decision is the part that was never about VRAM:
host RAM keeps its bands, and a pod with no telemetry draws no bar at all.
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
def test_vram_is_a_banded_usage_bar(html, tmp_path):
    """A card at 90% is drawn as a card at 90%, in the same colours as the rest.

    Asserted on a pod whose HOST ram is cool, so the only thing that can have
    produced the red is the VRAM bar.
    """
    out = _render(html, tmp_path,
                  {"ram_used_gb": 12.0, "ram_total_gb": 125.0,
                   "vram_used_gb": 43.2, "vram_total_gb": 48.0})
    assert _fills(out) == ["9.6", "90"]
    assert "#ef4444" in out, "a card at 90% is not flagged"
    cool = _render(html, tmp_path, {"vram_used_gb": 4.8, "vram_total_gb": 48.0})
    assert "#22c55e" in cool and "#ef4444" not in cool


@needs_node
def test_the_reservation_caveat_lands_only_on_pods_that_have_one(html, tmp_path):
    """vLLM pre-claims its share; a diffusion or OCR pod does not.

    So the word belongs to the pods whose figure is mostly a reservation, and
    `kv_cache_pct` is what says a pod is one -- it is reported by the engines
    that reserve and absent from the ones that do not.
    """
    vllm = _render(html, tmp_path,
                   {"vram_used_gb": 43.2, "vram_total_gb": 48.0,
                    "kv_cache_pct": 11.0})
    assert "43.2/48 GB incl. reserve" in vllm
    plain = _render(html, tmp_path,
                    {"vram_used_gb": 43.2, "vram_total_gb": 48.0})
    assert "43.2/48 GB" in plain
    assert "reserve" not in plain, (
        "a pod with no KV cache is reporting plain allocation")


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
