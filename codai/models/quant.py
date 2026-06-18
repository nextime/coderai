"""GPTQ/AWQ fast-kernel quantization support for the HuggingFace (NvidiaBackend) path.

bitsandbytes NF4 is the slowest 4-bit option and especially hurts MoE models. This
module lets coderai (a) detect whether GPTQModel + fast kernels (Marlin/ExLlama) are
available, (b) resolve where a locally-quantized checkpoint for a model lives, and
(c) run an on-demand background job that quantizes a model to 4-bit GPTQ and caches
the result. Loading a produced (or otherwise pre-quantized) checkpoint then goes
through transformers' native quantization path, which picks the fast kernel.

The whole module degrades gracefully: if GPTQModel can't import, capability checks
return False and callers fall back to bitsandbytes.
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from codai.models.cache import get_model_cache_dir


# --------------------------------------------------------------------------------
# Capability detection (cached — import probing is cheap but not free)
# --------------------------------------------------------------------------------

_caps_cache: Optional[Dict[str, Any]] = None
_caps_lock = threading.Lock()


def capabilities(refresh: bool = False) -> Dict[str, Any]:
    """Return a dict describing GPTQ/AWQ availability and usable kernels.

    Keys: ``available`` (bool), ``version`` (str|None), ``backends`` (list[str] of
    fast-kernel names that imported), ``error`` (str|None). Result is memoised.
    """
    global _caps_cache
    if _caps_cache is not None and not refresh:
        return _caps_cache
    with _caps_lock:
        if _caps_cache is not None and not refresh:
            return _caps_cache
        caps: Dict[str, Any] = {"available": False, "version": None,
                                "backends": [], "error": None}
        try:
            import gptqmodel  # noqa: F401
            caps["version"] = getattr(gptqmodel, "__version__", "?")
            # Which accelerated inference kernels are importable on this box.
            try:
                from gptqmodel.utils.backend import BACKEND
                wanted = ["MARLIN", "EXLLAMA_V2", "EXLLAMA_V1", "TRITON",
                          "AWQ_MARLIN", "AWQ_GEMM"]
                caps["backends"] = [b for b in wanted if hasattr(BACKEND, b)]
            except Exception:
                caps["backends"] = []
            caps["available"] = True
        except Exception as e:  # ImportError or a broken transitive dep
            caps["error"] = str(e)
        _caps_cache = caps
        return caps


def is_available() -> bool:
    """True when GPTQModel imports and at least one fast kernel is present."""
    c = capabilities()
    return bool(c["available"] and c["backends"])


# --------------------------------------------------------------------------------
# Quantized-checkpoint locations
# --------------------------------------------------------------------------------

def _safe_slug(model_name: str) -> str:
    """Filesystem-safe slug for a model id/path."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name)).strip("_")


def quantized_checkpoint_dir(model_name: str, method: str = "gptq") -> Path:
    """Cache directory where coderai stores a self-quantized checkpoint."""
    root = Path(get_model_cache_dir()) / "quantized" / method.lower()
    return root / _safe_slug(model_name)


def find_quantized_checkpoint(model_name: str, method: str = "gptq") -> Optional[str]:
    """Return the path to a usable locally-quantized checkpoint, or None.

    A checkpoint counts as ready when its directory holds a config.json plus at
    least one weights shard (so a half-finished/aborted quant isn't picked up).
    """
    d = quantized_checkpoint_dir(model_name, method)
    if not d.is_dir():
        return None
    if not (d / "config.json").is_file():
        return None
    has_weights = any(d.glob("*.safetensors")) or any(d.glob("*.bin"))
    return str(d) if has_weights else None


# --------------------------------------------------------------------------------
# Background quantization job (persisted so status survives a server restart)
# --------------------------------------------------------------------------------

_jobs: Dict[str, Dict[str, Any]] = {}     # model_name -> job status dict
_jobs_lock = threading.Lock()


def _jobs_file() -> Path:
    return Path(get_model_cache_dir()) / "quantized" / "jobs.json"


def _pid_alive(pid) -> bool:
    """True if process ``pid`` is still running. Conservative: unknown → alive."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except Exception:
        return True


def _save_jobs_locked() -> None:
    """Persist the job table to disk. Caller holds ``_jobs_lock``.

    Merges with whatever is on disk so a *different* process's jobs (the front and
    engines each import this module) aren't erased by a last-writer-wins overwrite.
    This process's in-memory entries win for keys it owns.
    """
    try:
        import json
        f = _jobs_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        merged = {}
        try:
            if f.is_file():
                disk = json.loads(f.read_text())
                if isinstance(disk, dict):
                    merged.update(disk)
        except Exception:
            pass
        merged.update(_jobs)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged))
        tmp.replace(f)
    except Exception:
        pass


def _load_jobs() -> None:
    """Load persisted jobs on import. A job left 'running' whose owning process is
    no longer alive can't still be running → mark interrupted. A job still owned by
    a live process (another engine, or this one) is left untouched."""
    try:
        import json
        f = _jobs_file()
        if not f.is_file():
            return
        data = json.loads(f.read_text())
        if not isinstance(data, dict):
            return
        for name, job in data.items():
            if (isinstance(job, dict) and job.get("status") == "running"
                    and not _pid_alive(job.get("pid"))):
                job["status"] = "interrupted"
                job["message"] = "interrupted by server restart"
            _jobs[name] = job
    except Exception:
        pass


def get_job(model_name: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        j = _jobs.get(model_name)
        return dict(j) if j else None


def all_jobs() -> Dict[str, Dict[str, Any]]:
    with _jobs_lock:
        return {k: dict(v) for k, v in _jobs.items()}


def _set_job(model_name: str, **fields) -> None:
    with _jobs_lock:
        j = _jobs.setdefault(model_name, {"model": model_name})
        j.update(fields)
        _save_jobs_locked()


_load_jobs()


def start_quantization(model_name: str, method: str = "gptq", bits: int = 4,
                       group_size: int = 128) -> Dict[str, Any]:
    """Kick off (or report) a background quantization for ``model_name``.

    Returns the job status dict. Idempotent: if a job is already running or a
    checkpoint already exists, it returns that state instead of starting again.
    """
    method = (method or "gptq").lower()
    if not is_available():
        return {"model": model_name, "status": "unavailable",
                "error": "GPTQModel / fast kernels not installed",
                "caps": capabilities()}

    existing = find_quantized_checkpoint(model_name, method)
    if existing:
        _set_job(model_name, status="done", method=method, output=existing,
                 progress=1.0, message="already quantized")
        return get_job(model_name)

    with _jobs_lock:
        cur = _jobs.get(model_name)
        if cur and cur.get("status") == "running":
            return dict(cur)
        _jobs[model_name] = {"model": model_name, "method": method, "bits": bits,
                             "status": "running", "progress": 0.0,
                             "message": "starting", "started": time.time(),
                             "pid": os.getpid(), "error": None, "output": None}
        _save_jobs_locked()

    t = threading.Thread(
        target=_quantize_worker,
        args=(model_name, method, bits, group_size),
        name=f"quantize-{_safe_slug(model_name)[:24]}",
        daemon=True,
    )
    t.start()
    return get_job(model_name)


def _quantize_worker(model_name: str, method: str, bits: int, group_size: int) -> None:
    """Run GPTQModel quantization and write the checkpoint to the cache dir.

    Heavy and slow (loads the source model, runs calibration). Runs in its own
    thread; never raises — failures are recorded on the job and the loader falls
    back to bitsandbytes.
    """
    out_dir = quantized_checkpoint_dir(model_name, method)
    try:
        _set_job(model_name, message="loading quantizer", progress=0.02)
        from gptqmodel import GPTQModel, QuantizeConfig

        # Calibration data: a small generic text sample. Enough to populate the
        # GPTQ Hessian statistics without a domain-specific corpus.
        calib = _calibration_samples()

        _set_job(model_name, message="loading source model (this is slow)", progress=0.05)
        qcfg = QuantizeConfig(bits=bits, group_size=group_size)
        model = GPTQModel.load(model_name, qcfg)

        _set_job(model_name, message="quantizing (calibration passes)", progress=0.15)
        model.quantize(calib)

        out_dir.mkdir(parents=True, exist_ok=True)
        _set_job(model_name, message="saving checkpoint", progress=0.9)
        model.save(str(out_dir))

        # Persist the tokenizer alongside so the checkpoint loads standalone.
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(model_name, trust_remote_code=True).save_pretrained(str(out_dir))
        except Exception:
            pass

        # Verify the quantizer actually compressed the weights. Some architectures
        # (e.g. gemma-4's fused-batched MoE experts) aren't covered by GPTQModel's
        # module map, so quantization silently leaves the bulk of the weights in
        # bf16/fp16 — producing a near-full-size "checkpoint" that wastes disk and
        # would offload at load time. Reject those instead of marking them done.
        frac = _quantized_fraction(out_dir)
        if frac is not None and frac < 0.5:
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
            _set_job(model_name, status="failed", progress=1.0, finished=time.time(),
                     error=(f"only {frac*100:.0f}% of weights were quantized — this "
                            f"architecture's layers (likely fused MoE experts) aren't "
                            f"supported by the quantizer; checkpoint discarded"),
                     message="quantization left most weights unquantized — discarded")
            return

        _set_job(model_name, status="done", progress=1.0,
                 message="quantization complete", output=str(out_dir),
                 finished=time.time())
    except Exception as e:
        # Likely causes: arch not supported by the quantizer, or OOM. Leave any
        # partial output in place but mark the job failed so the loader uses bnb.
        _set_job(model_name, status="failed", error=str(e),
                 message=f"quantization failed: {e}", finished=time.time())


def _quantized_fraction(ckpt_dir: Path) -> Optional[float]:
    """Fraction of large weight bytes that are actually low-bit (int-packed).

    Scans the saved safetensors and compares int8/int16/int32 (GPTQ qweight/qzeros)
    bytes against bf16/fp16 weight bytes. Near 1.0 = properly quantized; near 0.0 =
    the quantizer skipped most layers. Returns None if it can't be determined.
    """
    try:
        from safetensors import safe_open
        _BPE = {"I32": 4, "I16": 2, "I8": 1, "U8": 1, "BF16": 2, "F16": 2, "F32": 4}
        low_bits = 0
        full = 0
        shards = list(ckpt_dir.glob("*.safetensors"))
        if not shards:
            return None
        for f in shards:
            with safe_open(str(f), framework="numpy") as h:
                for k in h.keys():
                    sl = h.get_slice(k)
                    dt = sl.get_dtype()
                    n = 1
                    for s in sl.get_shape():
                        n *= s
                    nbytes = n * _BPE.get(dt, 2)
                    if dt in ("I32", "I16", "I8", "U8"):
                        low_bits += nbytes
                    elif dt in ("BF16", "F16", "F32"):
                        # Ignore small tensors (norms, biases, scales, router gates);
                        # only large 2-D+ weights signal an unquantized layer.
                        if len(sl.get_shape()) >= 2 and n >= 1_000_000:
                            full += nbytes
        total = low_bits + full
        if total <= 0:
            return None
        return low_bits / total
    except Exception:
        return None


def _calibration_samples() -> List[str]:
    """A small, generic calibration set for GPTQ Hessian estimation."""
    base = [
        "The quick brown fox jumps over the lazy dog.",
        "In computer science, a hash table is a data structure that maps keys to values.",
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "The mitochondria is the powerhouse of the cell, generating most of the cell's ATP.",
        "Once upon a time, in a land far away, there lived a wise old programmer.",
        "To be, or not to be, that is the question that has echoed through the centuries.",
        "Machine learning models learn patterns from data to make predictions on unseen inputs.",
        "The capital of France is Paris, a city known for its art, culture, and history.",
    ]
    # GPTQModel wants a few hundred short rows; repeat the seed set.
    return [base[i % len(base)] for i in range(256)]
