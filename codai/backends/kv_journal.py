"""What a KV cache really cost, written down the moment llama.cpp says so.

The KV placement plan in :mod:`codai.backends.vulkan` can only act on a measured
size, because predicting one from GGUF metadata is architecture-specific and was
off by 5x in both directions on the model that prompted it. llama.cpp reports the
real figure in its own log ("CPU KV buffer size = 48875.00 MiB"), and the loader
already read it — but only after a successful load, as a line of advice telling
whoever was watching to copy the number into the model config by hand.

A KV cache big enough to matter is a KV cache big enough to get the engine
SIGKILLed by the OOM killer while it is still loading. In that case there is no
"after the load" and nobody is watching: the engine dies, the front respawns it,
the next request repeats the identical load, and the measurement that would have
stopped it is lost every time. One model on one deployment did that four times in
an evening, taking every other model on the engine down with it.

So the figure is journalled here as soon as llama.cpp prints it, which is before
the pages are faulted in and therefore before the kill. It is a cache, not
configuration: ``measured_kv_gb`` on the model still wins, entries are keyed by
model path and context length, and a missing or unreadable journal only costs the
plan its measurement.
"""

import json
import os
import tempfile
import threading

_LOCK = threading.Lock()
_PATH_OVERRIDE = None


def journal_path() -> str:
    """Next to the config it supplements, so a deployment keeps one home."""
    if _PATH_OVERRIDE:
        return _PATH_OVERRIDE
    try:
        from codai.platform_paths import legacy_style_config_dir
        base = str(legacy_style_config_dir())
    except Exception:
        base = os.path.expanduser('~/.coderai')
    return os.path.join(base, 'cache', 'kv_sizes.json')


def _key(model_path, n_ctx) -> str:
    return f"{model_path}@{int(n_ctx or 0)}"


def _load() -> dict:
    try:
        with open(journal_path(), 'r') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def record(model_path, n_ctx, gb) -> bool:
    """Write down that this model at this context length cost ``gb`` of KV.

    Returns True when the entry was written. Never raises: a journal that cannot
    be written must not fail a load that would otherwise succeed.
    """
    try:
        if not model_path or not n_ctx or not gb or float(gb) <= 0:
            return False
        path = journal_path()
        with _LOCK:
            data = _load()
            data[_key(model_path, n_ctx)] = round(float(gb), 2)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            except Exception:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
                raise
        return True
    except Exception:
        return False


def lookup(model_path, n_ctx) -> float:
    """Journalled KV size in GB for this model at this context, else 0.0."""
    try:
        if not model_path or not n_ctx:
            return 0.0
        val = _load().get(_key(model_path, n_ctx))
        return float(val) if val else 0.0
    except Exception:
        return 0.0
