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


def save(pipe, p: str, *, model_name: str = "", model_cfg: Optional[dict] = None) -> bool:
    """Serialize ``pipe`` to the cache dir ``p`` (atomic via a temp dir).

    Returns True on success. Any failure is logged and returns False — the caller
    keeps the freshly built in-memory pipeline regardless."""
    if not p:
        return False
    tmp = p + ".building"
    try:
        os.makedirs(cache_root(), exist_ok=True)
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
