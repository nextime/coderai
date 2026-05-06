from pathlib import Path
import sys

sys.path.insert(0, '/storage/coderai')

from codai.api.audio_backends import detect_audio_backends, reset_audio_backend_cache


def test_detect_audio_backends_reports_missing_providers(monkeypatch):
    reset_audio_backend_cache()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    backends = detect_audio_backends()

    assert backends["separation"]["available"] is False
    assert backends["separation"]["engine"] is None
    assert backends["restoration"]["available"] is False
    assert backends["restoration"]["engine"] is None


def test_detect_audio_backends_prefers_deepfilter_for_restoration(monkeypatch):
    reset_audio_backend_cache()

    def fake_find_spec(name: str):
        if name in {"demucs", "df"}:
            return object()
        return None

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    backends = detect_audio_backends()

    assert backends["separation"] == {
        "available": True,
        "engine": "demucs",
        "candidates": ["demucs"],
    }
    assert backends["restoration"]["available"] is True
    assert backends["restoration"]["engine"] == "deepfilternet"
    assert "voicefixer" in backends["restoration"]["candidates"]


def test_requirements_include_audio_ml_dependencies():
    req = Path('/storage/coderai/requirements.txt').read_text().lower()
    assert 'demucs' in req
    assert 'deepfilternet' in req or 'df[' in req
