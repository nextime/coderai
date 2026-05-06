import importlib.util
from functools import lru_cache


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@lru_cache(maxsize=1)
def detect_audio_backends() -> dict:
    demucs_ok = _has_module("demucs")
    deepfilter_ok = _has_module("df") or _has_module("deepfilternet")
    rnnoise_ok = _has_module("rnnoise")
    voicefixer_ok = _has_module("voicefixer")

    restoration_engine = None
    if deepfilter_ok:
        restoration_engine = "deepfilternet"
    elif rnnoise_ok:
        restoration_engine = "rnnoise"
    elif voicefixer_ok:
        restoration_engine = "voicefixer"

    return {
        "separation": {
            "available": demucs_ok,
            "engine": "demucs" if demucs_ok else None,
            "candidates": ["demucs"],
        },
        "restoration": {
            "available": bool(restoration_engine),
            "engine": restoration_engine,
            "candidates": ["deepfilternet", "rnnoise", "voicefixer"],
        },
    }


def reset_audio_backend_cache() -> None:
    detect_audio_backends.cache_clear()
