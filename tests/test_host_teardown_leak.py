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

"""Host-resident pipeline weights must be released on teardown too.

The existing teardown-leak breaker in video.py is CUDA-only on both sides: it
registers under ``if _tens.is_cuda`` and sweeps under ``if not (... and
_obj.is_cuda): continue``. A pipeline loaded with ``enable_model_cpu_offload``
keeps its weights in HOST RAM, and 17 of 21 video loads in one production day
took that path, so none of those weights were tracked and nothing looked for
them afterwards.

Measured 2026-10-29 with nothing loaded and 482 MiB of VRAM in use:

    coderai-nvidia  33.14 GB PSS_Anon
    30.85 GB anon in 6588 mappings; 6565 of them under 100 MB (20.25 GB)

Thousands of mid-sized allocations is retained tensors, not allocator slack --
so gc + malloc_trim every 15 s never moved it, and the ladder evicted healthy
models instead.
"""

import gc
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codai.models import teardown as T

torch = pytest.importorskip("torch")
nn = torch.nn


def _pipe(**comps):
    p = type("Pipe", (), {})()
    p.components = comps
    return p


def test_host_tensors_are_tracked():
    pipe = _pipe(transformer=nn.Linear(64, 64), scheduler=object(), tokenizer=None)
    refs = T.track_host_tensors(pipe)
    assert refs, "no host tensors tracked for a CPU-resident pipeline"


def test_gpu_tensors_are_left_to_the_cuda_breaker():
    """The two arms must not both chase the same objects."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pipe = _pipe(transformer=nn.Linear(64, 64).cuda())
    assert T.track_host_tensors(pipe) == []


def test_a_clean_teardown_reports_nothing_surviving():
    pipe = _pipe(transformer=nn.Linear(256, 256))
    refs = T.track_host_tensors(pipe)
    del pipe
    gc.collect()
    r = T.break_host_leaks(refs)
    assert r["survived_gb"] == 0.0
    assert r["tensors"] == 0
    assert r["broken"] == 0


def test_a_tensor_pinned_by_a_dict_is_found_and_released():
    """The real shape of the leak: a stray cache entry outlives the pipeline
    and pins the weights, so dropping the pipeline frees nothing."""
    pipe = _pipe(transformer=nn.Linear(512, 512))
    stray_cache = {"weight": pipe.components["transformer"].weight}
    refs = T.track_host_tensors(pipe)
    del pipe
    gc.collect()

    r = T.break_host_leaks(refs)
    assert r["tensors"] >= 1, "the surviving host tensor was not detected"
    assert r["broken"] >= 1, "the reference pinning it was not broken"
    assert stray_cache["weight"] is None, "the cache still pins the weight"


def test_a_tensor_pinned_by_a_list_is_released():
    pipe = _pipe(transformer=nn.Linear(256, 256))
    stray = [pipe.components["transformer"].weight]
    refs = T.track_host_tensors(pipe)
    del pipe
    gc.collect()
    T.break_host_leaks(refs)
    assert stray[0] is None


def test_the_surviving_size_is_reported():
    pipe = _pipe(transformer=nn.Linear(1024, 1024, bias=False).to(torch.float32))
    held = {"w": pipe.components["transformer"].weight}
    refs = T.track_host_tensors(pipe)
    del pipe
    gc.collect()
    r = T.break_host_leaks(refs)
    # reported to 3 dp, i.e. 1 MB granularity
    assert r["survived_gb"] == pytest.approx(1024 * 1024 * 4 / 1e9, abs=0.001)
    assert held["w"] is None


def test_only_tracked_tensors_are_touched():
    """Another model's memory must never be at risk. This is why the host arm
    tracks weakrefs to the objects instead of matching storage addresses: a
    freed host address is readily reused by malloc, and a pointer match would
    then null a live tensor belonging to something else."""
    victim = nn.Linear(128, 128)
    bystander = {"w": victim.weight}
    other = _pipe(transformer=nn.Linear(128, 128))
    refs = T.track_host_tensors(other)
    del other
    gc.collect()
    T.break_host_leaks(refs)
    assert bystander["w"] is victim.weight, "an untracked tensor was nulled"


def test_buffers_are_tracked_not_just_parameters():
    """Pre-quantized formats keep their payload in buffers, so a
    parameters-only sweep would leave a 4-bit model's weights pinned."""
    m = nn.Module()
    m.register_buffer("qweight", torch.empty((256, 64), dtype=torch.int32))
    pipe = _pipe(transformer=m)
    held = {"q": m.qweight}
    refs = T.track_host_tensors(pipe)
    del pipe, m
    gc.collect()
    r = T.break_host_leaks(refs)
    assert r["tensors"] >= 1
    assert held["q"] is None


def test_the_cuda_arm_still_only_tracks_cuda():
    """Guard the division of labour in video.py: if the CUDA registration ever
    starts taking host tensors too, the two arms would double-handle them."""
    src = io.open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "codai/api/video.py"), encoding="utf-8").read()
    assert "_orig_cuda_ptrs.add" in src
    reg = src.split("_orig_cuda_ptrs.add", 1)[0][-400:]
    assert "is_cuda" in reg, (
        "the CUDA breaker no longer gates registration on is_cuda")


def test_trim_is_available_and_safe_to_call():
    assert T.trim_host_allocator() in (True, False)
