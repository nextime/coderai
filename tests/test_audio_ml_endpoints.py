import base64
import os
import sys
import wave
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def sample_wav_b64():
    buf = BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * 800)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_audio_stems_uses_provider_output(monkeypatch, tmp_path, sample_wav_b64):
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "codai" / "api" / "audio_stems.py"
    spec = importlib.util.spec_from_file_location("test_audio_stems_module", module_path)
    audio_stems = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(audio_stems)

    app = FastAPI()
    app.include_router(audio_stems.router)
    audio_stems.set_global_file_path(str(tmp_path))

    vocals = tmp_path / "vocals.wav"
    inst = tmp_path / "inst.wav"
    vocals.write_bytes(b"wav")
    inst.write_bytes(b"wav")

    monkeypatch.setattr(audio_stems, "separate_with_provider", lambda *args, **kwargs: {
        "engine": "demucs",
        "model": "htdemucs",
        "stem_mode": "vocals-instrumental",
        "artifacts": [
            {"name": "vocals", "path": str(vocals), "role": "vocals"},
            {"name": "instrumental", "path": str(inst), "role": "instrumental"},
        ],
        "limitations": [],
    })
    monkeypatch.setattr(audio_stems, "detect_audio_backends", lambda: {
        "separation": {"available": True, "engine": "demucs", "candidates": ["demucs"]}
    })

    client = TestClient(app)
    response = client.post(
        "/v1/audio/stems",
        json={"audio": sample_wav_b64, "stem_mode": "vocals-instrumental", "response_format": "url"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["backend"]["engine"] == "demucs"
    assert body["backend"]["model"] == "htdemucs"
    assert body["backend"]["quality"] == "ml"
    assert len(body["data"]) == 2
    assert body["data"][0]["url"].endswith(".wav")


def test_audio_cleanup_uses_restore_provider(monkeypatch, tmp_path, sample_wav_b64):
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "codai" / "api" / "audio_clean.py"
    spec = importlib.util.spec_from_file_location("test_audio_clean_module", module_path)
    audio_clean = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(audio_clean)

    app = FastAPI()
    app.include_router(audio_clean.router)
    audio_clean.set_global_file_path(str(tmp_path))

    cleaned = tmp_path / "cleaned.wav"
    cleaned.write_bytes(b"wav")

    monkeypatch.setattr(audio_clean, "restore_with_provider", lambda *args, **kwargs: {
        "engine": "deepfilternet",
        "model": "DeepFilterNet3",
        "path": str(cleaned),
        "applied": ["denoise", "normalize"],
        "limitations": [],
    })
    monkeypatch.setattr(audio_clean, "detect_audio_backends", lambda: {
        "restoration": {"available": True, "engine": "deepfilternet", "candidates": ["deepfilternet", "rnnoise", "voicefixer"]}
    })

    client = TestClient(app)
    response = client.post(
        "/v1/audio/cleanup",
        json={
            "audio": sample_wav_b64,
            "noise_reduction": True,
            "normalize": True,
            "remove_hum": False,
            "repair_clicks": False,
            "response_format": "url",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["backend"]["engine"] == "deepfilternet"
    assert body["backend"]["model"] == "DeepFilterNet3"
    assert body["backend"]["quality"] == "ml"
    assert body["applied"] == ["denoise", "normalize"]
    assert body["data"][0]["url"].endswith(".wav")
