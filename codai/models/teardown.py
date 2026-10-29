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

"""Finding a pipeline's weights that outlived its teardown -- on the HOST side.

``codai/api/video.py`` already carries a teardown-leak breaker, and it works:
on a production host it went from reporting "16.9 GB of this pipeline's tensors
survived teardown; nulled 7581 external reference(s)" to 0.0 GB once the
injected components were released in the right order. But both halves of it are
CUDA-only -- it records ``_orig_cuda_ptrs`` under ``if _tens.is_cuda`` and
sweeps under ``if not (isinstance(_obj, Tensor) and _obj.is_cuda): continue``.

A pipeline loaded with ``enable_model_cpu_offload`` keeps its weights in HOST
RAM and streams modules to the GPU for each forward. Seventeen of twenty-one
video loads in one production day took that path. None of those weights were
ever tracked, so nothing looked for them after teardown. Measured on the same
host, 2026-10-29, with nothing loaded and 482 MiB of VRAM in use:

    coderai-nvidia   33.14 GB PSS_Anon
    total anon:      30.85 GB in 6588 mappings
      >= 1 GB:          2
      100 MB - 1 GB:   21   (3.49 GB)
      < 100 MB:     6,565   (20.25 GB)

Thousands of separate mid-sized allocations is retained tensors, not allocator
slack -- which is why the mitigation ladder's gc + ``malloc_trim`` could run
every 15 s for hours without the figure moving, and why it then evicted healthy
models to chase it.

Identity, not addresses
-----------------------
The CUDA breaker matches surviving tensors by storage pointer. That is sound
enough on the GPU, but a freed *host* address is readily handed back out by
malloc, so a pointer that once belonged to this pipeline may by then belong to
another model's tensor -- and nulling its referrers would corrupt a live model.
So this tracks :mod:`weakref` references to the tensor objects themselves. A
referent that is still alive after teardown IS the leaked tensor, with no
address-reuse ambiguity, and a referent that has gone is proof the teardown
worked.
"""

from __future__ import annotations

import gc
import weakref


def _component_tensors(pipe):
    """Every parameter and buffer of every component of a pipeline."""
    comps = getattr(pipe, "components", None)
    if not isinstance(comps, dict):
        comps = {}
    for comp in comps.values():
        if comp is None or not hasattr(comp, "parameters"):
            continue
        try:
            params = list(comp.parameters(recurse=True))
        except Exception:
            params = []
        try:
            bufs = list(comp.buffers(recurse=True))
        except Exception:
            bufs = []
        for t in params + bufs:
            yield t
            # A bnb 4-bit parameter keeps its quantized payload on `.data`,
            # which can be a different tensor object from the parameter.
            d = getattr(t, "data", None)
            if d is not None and d is not t:
                yield d


def track_host_tensors(pipe) -> list:
    """Weak references to this pipeline's HOST-resident tensors.

    Call before teardown. GPU-resident tensors are left to the existing CUDA
    breaker, so the two do not fight over the same objects.
    """
    refs = []
    seen = set()
    for t in _component_tensors(pipe):
        try:
            if getattr(t, "is_cuda", False) or t.device.type != "cpu":
                continue
            if t.numel() == 0:
                continue
        except Exception:
            continue
        if id(t) in seen:
            continue
        seen.add(id(t))
        try:
            refs.append(weakref.ref(t))
        except TypeError:
            continue            # not weak-referenceable; nothing we can do
    return refs


def _null_referrers(obj) -> int:
    """Clear the slots in lists/dicts that still point at ``obj``.

    The same approach the CUDA breaker uses: a weight pinned by a stray entry
    in some cache dict or list cannot be freed by dropping the pipeline, and
    these two container kinds are what actually holds them in practice
    (``_modules`` dicts, hook state, our own caches).
    """
    broken = 0
    for ref in gc.get_referrers(obj):
        try:
            if isinstance(ref, list):
                for i, item in enumerate(ref):
                    if item is obj:
                        ref[i] = None
                        broken += 1
            elif isinstance(ref, dict):
                for k, v in list(ref.items()):
                    if v is obj:
                        ref[k] = None
                        broken += 1
        except Exception:
            continue
    return broken


def break_host_leaks(refs, *, min_report_gb: float = 0.05) -> dict:
    """Null external references to host tensors that survived teardown.

    ``refs`` is what :func:`track_host_tensors` returned. Returns a report with
    the surviving size, how many references were broken, and the biggest
    holders, mirroring what the CUDA breaker prints so the two read alike.

    Only tensors we tracked are touched, and each by identity, so another
    model's memory is never at risk.
    """
    out = {"survived_gb": 0.0, "tensors": 0, "broken": 0, "holders": []}
    if not refs:
        return out
    gc.collect()
    for ref in refs:
        t = ref()
        if t is None:
            continue                     # freed as it should have been
        try:
            gb = t.numel() * t.element_size() / 1e9
        except Exception:
            continue
        out["survived_gb"] += gb
        out["tensors"] += 1
        out["broken"] += _null_referrers(t)
        if gb >= min_report_gb and len(out["holders"]) < 6:
            try:
                out["holders"].append(f"{gb:.2f} GB {tuple(t.shape)} {t.dtype}")
            except Exception:
                pass
        del t
    gc.collect()
    out["survived_gb"] = round(out["survived_gb"], 3)
    return out


def trim_host_allocator() -> bool:
    """Ask glibc to return freed heap to the OS.

    Worth doing only AFTER the references are broken: ``malloc_trim`` can
    return nothing while the memory is still referenced, which is exactly what
    the mitigation ladder was doing every 15 s for hours.
    """
    try:
        import ctypes
        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except Exception:
        return False
