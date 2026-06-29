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

"""On-disk cache of *built* diffusers pipelines.

Building a large quantized video pipeline (e.g. Wan2.2 A14B at 4-bit) is slow:
download + bitsandbytes quantization of ~28B parameters. The weights don't change
between restarts, so once built we ``save_pretrained`` the pipeline to a local
cache keyed by ``(model, quantization, precision)``. A later start with
``--pipeline-cache`` reloads from there with a plain ``from_pretrained`` of the
already-quantized weights — no re-download, no re-quantization.

Scope: only the *base* pipeline is cached. The acceleration/distillation LoRA is
NOT baked into the cache — it is re-fused on every load (a fast operation), so the
cache stays independent of the (cheap to re-apply) ``acceleration`` config and we
avoid the fragile round-trip of serialising a fused + quantized model.

Everything here is best-effort: any failure (save or load) is swallowed and the
caller falls back to a normal build, so the cache can never break generation.
"""

import glob
import hashlib
import json
import os
import shutil
import time
from typing import Optional

# Bump when the cache layout / marker format changes so stale caches are ignored.
_CACHE_VERSION = 1


def _global_args():
    try:
        from codai.api.state import get_global_args
        return get_global_args()
    except Exception:
        return None


def enabled() -> bool:
    """True when --pipeline-cache was passed."""
    ga = _global_args()
    return bool(ga is not None and getattr(ga, "pipeline_cache", False))


def _force_rebuild() -> bool:
    ga = _global_args()
    return bool(ga is not None and getattr(ga, "rebuild_pipeline_cache", False))


def cache_root() -> str:
    """Root dir for cached pipelines. Sits next to the offload dir by default."""
    ga = _global_args()
    offload_dir = getattr(ga, "offload_dir", None) if ga else None
    if offload_dir:
        root = os.path.join(os.path.dirname(os.path.abspath(os.path.expanduser(offload_dir))),
                            "pipeline_cache")
    else:
        root = os.path.join(os.path.expanduser("~"), ".cache", "coderai", "pipeline_cache")
    return root


def _signature(model_name: str, model_cfg: Optional[dict]) -> str:
    """Stable hash of everything that changes the *built* (quantized) weights:
    the model id, the quantization choices, and the precision. NOT acceleration
    (re-applied per load) and NOT offload (a runtime placement decision)."""
    c = model_cfg or {}
    payload = {
        "v": _CACHE_VERSION,
        "model": model_name,
        "precision": c.get("precision") or "bf16",
        "load_in_4bit": bool(c.get("load_in_4bit", False)),
        "load_in_8bit": bool(c.get("load_in_8bit", False)),
        "component_quantization": c.get("component_quantization") or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _safe_name(model_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in model_name)[:80]


def path(model_name: str, model_cfg: Optional[dict]) -> str:
    """Absolute cache directory for this model + quant/precision signature."""
    return os.path.join(cache_root(),
                        f"{_safe_name(model_name)}__{_signature(model_name, model_cfg)}")


def _marker(p: str) -> str:
    return os.path.join(p, ".coderai_pipeline_cache.json")


def valid(p: str) -> bool:
    """True if a complete, current cache exists at ``p`` and rebuild wasn't forced."""
    if not p or _force_rebuild():
        return False
    try:
        if not os.path.isfile(os.path.join(p, "model_index.json")):
            return False
        with open(_marker(p)) as f:
            meta = json.load(f)
        return meta.get("version") == _CACHE_VERSION and meta.get("complete") is True
    except Exception:
        return False


# ── Per-component (incremental) cache ────────────────────────────────────────
# A big quantized video pipeline (Wan2.2 A14B) often won't fit on the GPU and can
# only be loaded with offload — and an offloaded pipeline can't be save_pretrained
# wholesale. But we CAN cache it one component at a time: each component is small
# enough to pass through the GPU alone (where bitsandbytes quantizes it to a
# uniform 4-bit), so we save each as it's built. The cache is then a set of
# per-component subdirs under the same model+signature dir as the monolithic
# cache, each with its OWN completion marker. A load that's evicted before every
# component is cached still leaves the finished ones valid; the next load mixes
# cached components with freshly-built (and now-cached) ones, converging to a full
# cache over a couple of runs. Each component carries the same model+quant+
# precision signature, so a config change invalidates it like the monolithic one.

def mark_monolithic_complete(model_name: str, model_cfg: Optional[dict]) -> bool:
    """Write the monolithic completion marker for a cache dir that was populated
    component-by-component (heavy components via save_component, light components +
    model_index.json by the finalizer). After this, valid() returns True and the
    dir loads as a normal pipeline via from_pretrained(dir, device_map=…) — no
    injection — so device_map can place every component and big-clip generation
    fits. Only call once model_index.json and every component subdir are present."""
    try:
        p = path(model_name, model_cfg)
        if not os.path.isfile(os.path.join(p, "model_index.json")):
            return False
        with open(_marker(p), "w") as f:
            json.dump({
                "version": _CACHE_VERSION, "complete": True,
                "model": model_name, "saved_at": time.time(),
                "signature": _signature(model_name, model_cfg),
                "built": "incremental",
            }, f)
        return True
    except Exception:
        return False


def _component_marker(cdir: str) -> str:
    return os.path.join(cdir, ".coderai_component.json")


def component_dir(model_name: str, model_cfg: Optional[dict], comp: str) -> str:
    """Cache subdir for a single pipeline component (e.g. 'transformer')."""
    return os.path.join(path(model_name, model_cfg), comp)


def component_valid(model_name: str, model_cfg: Optional[dict], comp: str) -> bool:
    """True if a complete, signature-matching cache for this component exists."""
    if _force_rebuild():
        return False
    try:
        cdir = component_dir(model_name, model_cfg, comp)
        if not os.path.isfile(os.path.join(cdir, "config.json")):
            return False
        with open(_component_marker(cdir)) as f:
            meta = json.load(f)
        return (meta.get("version") == _CACHE_VERSION
                and meta.get("complete") is True
                and meta.get("signature") == _signature(model_name, model_cfg))
    except Exception:
        return False


def save_component(comp_obj, model_name: str, model_cfg: Optional[dict], comp: str) -> bool:
    """Atomically save one freshly-built (4-bit) component to its cache subdir.

    Writes to ``<comp>.building`` then renames, with the completion marker written
    last, so an interrupted save never leaves a half-written component that
    component_valid() would accept. Best-effort: any failure returns False and the
    caller keeps the in-memory component regardless."""
    cdir = component_dir(model_name, model_cfg, comp)
    if not cdir:
        return False
    tmp = cdir + ".building"
    try:
        os.makedirs(os.path.dirname(cdir), exist_ok=True)
        sweep_stale()
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"  [pipeline-cache] caching component '{comp}' → {cdir}")
        t0 = time.time()
        comp_obj.save_pretrained(tmp)
        with open(_component_marker(tmp), "w") as f:
            json.dump({
                "version": _CACHE_VERSION, "complete": True,
                "model": model_name, "component": comp, "saved_at": time.time(),
                "signature": _signature(model_name, model_cfg),
                "bytes": _dir_size(tmp),
            }, f)
        if os.path.exists(cdir):
            shutil.rmtree(cdir, ignore_errors=True)
        os.replace(tmp, cdir)
        print(f"  [pipeline-cache] cached '{comp}' in {time.time() - t0:.0f}s")
        return True
    except Exception as e:
        print(f"  [pipeline-cache] component '{comp}' save failed ({e}) — skipping")
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass
        return False


def _dir_size(d: str) -> int:
    total = 0
    try:
        for root, _, files in os.walk(d):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except Exception:
                    pass
    except Exception:
        pass
    return total


def invalidate(model_name: str, model_cfg: Optional[dict]) -> None:
    """Delete a model's cache dir (e.g. after a failed cache load) so the next
    build rewrites it. Best-effort."""
    try:
        p = path(model_name, model_cfg)
        if p and os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
            print(f"  [pipeline-cache] invalidated {p}")
    except Exception:
        pass


def sweep_stale(max_age_s: float = 1800.0) -> None:
    """Delete orphaned ``*.building`` temp dirs left by a save that was killed
    mid-write (the atomic ``os.replace`` never ran). They waste tens of GB and,
    being incomplete, are never valid() anyway. Best-effort; only removes dirs
    older than ``max_age_s`` so it never races a save in progress."""
    try:
        root = cache_root()
        now = time.time()
        for d in glob.glob(os.path.join(root, "*.building")):
            try:
                if now - os.path.getmtime(d) >= max_age_s:
                    shutil.rmtree(d, ignore_errors=True)
                    print(f"  [pipeline-cache] swept stale {os.path.basename(d)}")
            except Exception:
                pass
    except Exception:
        pass


def _unsavable_reason(pipe) -> Optional[str]:
    """Return why ``pipe`` cannot be serialized, or None if it's savable.

    diffusers refuses save_pretrained on a pipeline with CPU-offload hooks
    (enable_model_cpu_offload / enable_sequential_cpu_offload), and a device_map
    pipeline that spilled to disk holds meta tensors that serialize to garbage.
    Detect both so we skip cleanly (no half-written .building) instead of failing
    deep inside save_pretrained — the cache must be built from a savable
    (pre-offload, fully-materialised) pipeline."""
    try:
        if getattr(pipe, "_all_hooks", None) or getattr(pipe, "hf_device_map", None):
            return "accelerate offload active (cpu/sequential/device_map)"
        comps = getattr(pipe, "components", {}) or {}
        for name, comp in comps.items():
            if not hasattr(comp, "parameters"):
                continue
            for prm in comp.parameters():
                if getattr(prm, "is_meta", False) or str(getattr(prm, "device", "")) == "meta":
                    return f"component '{name}' has meta/offloaded tensors"
                break  # one param per component is enough to classify
    except Exception:
        return None  # can't tell — let save_pretrained try
    return None


def save(pipe, p: str, *, model_name: str = "", model_cfg: Optional[dict] = None) -> bool:
    """Serialize ``pipe`` to the cache dir ``p`` (atomic via a temp dir).

    Returns True on success. Any failure is logged and returns False — the caller
    keeps the freshly built in-memory pipeline regardless."""
    if not p:
        return False
    _why = _unsavable_reason(pipe)
    if _why:
        print(f"  [pipeline-cache] not caching {os.path.basename(p)}: {_why} "
              f"— cache must be built from a pre-offload pipeline")
        return False
    tmp = p + ".building"
    try:
        os.makedirs(cache_root(), exist_ok=True)
        sweep_stale()
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"  [pipeline-cache] saving quantized pipeline → {p}")
        t0 = time.time()
        pipe.save_pretrained(tmp)
        with open(_marker(tmp), "w") as f:
            json.dump({
                "version": _CACHE_VERSION, "complete": True,
                "model": model_name, "saved_at": time.time(),
                "signature": _signature(model_name, model_cfg),
            }, f)
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
        os.replace(tmp, p)
        print(f"  [pipeline-cache] saved in {time.time() - t0:.0f}s")
        return True
    except Exception as e:
        print(f"  [pipeline-cache] save failed ({e}) — continuing without a cache")
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass
        return False
