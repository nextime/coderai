import base64
import os
import sys
import types
import wave
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def studio_client(tmp_path):
    from codai.api import audio_clean, audio_stems, custom_pipelines, transcriptions, tts, text, embeddings
    from codai.admin import routes as admin_routes

    audio_stems.set_global_file_path(str(tmp_path))
    audio_clean.set_global_file_path(str(tmp_path))

    app = FastAPI()
    app.include_router(audio_stems.router)
    app.include_router(audio_clean.router)
    app.include_router(custom_pipelines.router)
    app.include_router(transcriptions.router)
    app.include_router(tts.router)
    app.include_router(text.router)
    app.include_router(embeddings.router)
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_auth] = lambda: "tester"
    return TestClient(app)


@pytest.fixture
def sample_wav_b64():
    buf = BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * 800)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_audio_understanding_composes_transcript_and_summary(monkeypatch, studio_client):
    from codai.api import custom_pipelines

    async def fake_run_step(step, context, http_request):
        if step["type"] == "stt":
            return {"output": "meeting transcript text", "text": "meeting transcript text"}
        if step["type"] == "text_gen":
            assert context["step0"]["output"] == "meeting transcript text"
            prompt = step["params"]["prompt"]
            assert "Summarize action items" in prompt
            assert "{{step0.output}}" in prompt
            return {"output": "summary from transcript"}
        raise AssertionError(f"unexpected step type {step['type']}")

    monkeypatch.setattr(custom_pipelines, "_run_step", fake_run_step)

    response = studio_client.post(
        "/v1/pipelines/audio-understand",
        json={
            "input": "Summarize action items",
            "audio_model": "whisper-small",
            "text_model": "qwen-text",
            "audio": "ZmFrZQ==",
            "language": "en",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["pipeline"] == "audio-understand"
    assert body["transcript"] == "meeting transcript text"
    assert body["summary"] == "summary from transcript"
    assert [step["type"] for step in body["steps"]] == ["stt", "text_gen"]


def test_audio_understanding_returns_transcript_only_without_text_model(monkeypatch, studio_client):
    from codai.api import custom_pipelines

    async def fake_run_step(step, context, http_request):
        assert step["type"] == "stt"
        return {"output": "raw transcript", "text": "raw transcript"}

    monkeypatch.setattr(custom_pipelines, "_run_step", fake_run_step)

    response = studio_client.post(
        "/v1/pipelines/audio-understand",
        json={
            "input": "Describe the call",
            "audio_model": "whisper-small",
            "audio": "ZmFrZQ==",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "raw transcript"
    assert body["summary"] is None
    assert len(body["steps"]) == 1


def test_audio_understanding_pipeline_steps_release_scheduler_slots(monkeypatch, studio_client):
    from codai.api import custom_pipelines
    from codai.queue.manager import queue_manager

    observed = []

    async def fake_run_step(step, context, http_request):
        observed.append(queue_manager.get_metrics()["active"])
        return {"output": step["type"], "text": step["type"]}

    monkeypatch.setattr(custom_pipelines, "_run_step", fake_run_step)
    queue_manager.reset_for_tests()
    queue_manager.set_loaded_models({"audio:whisper-small", "qwen-text"})

    response = studio_client.post(
        "/v1/pipelines/audio-understand",
        json={
            "input": "Summarize",
            "audio_model": "whisper-small",
            "text_model": "qwen-text",
            "audio": "ZmFrZQ==",
        },
    )

    assert response.status_code == 200
    assert observed == [1, 1]
    assert queue_manager.get_metrics()["active"] == 0


def test_audio_understanding_requires_audio_source(studio_client):
    response = studio_client.post(
        "/v1/pipelines/audio-understand",
        json={"audio_model": "whisper-small", "input": "Summarize"},
    )

    assert response.status_code == 422
    assert "audio" in response.text.lower()


def test_music_dub_pipeline_returns_full_stage_outputs(monkeypatch, studio_client):
    from codai.api import custom_pipelines

    async def fake_run_full_music_dub(request, http_request):
        return {
            "vocals": {"path": "vocals.wav"},
            "instrumental": {"path": "inst.wav"},
            "transcript": "lyrics",
            "translated_lyrics": "translated lyrics",
            "converted_vocals": {"path": "dub.wav"},
            "final_mix": {"path": "mix.wav"},
            "steps": [
                {"step": 0, "type": "stems"},
                {"step": 1, "type": "stt"},
                {"step": 2, "type": "translate"},
                {"step": 3, "type": "voice_convert"},
                {"step": 4, "type": "remix"},
            ],
        }

    monkeypatch.setattr(custom_pipelines, "run_full_music_dub", fake_run_full_music_dub)

    response = studio_client.post(
        "/v1/pipelines/audio-music-dub",
        json={
            "audio_model": "whisper-small",
            "audio": "ZmFrZQ==",
            "target_lang": "es",
            "notes": "Prefer singability",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["pipeline"] == "audio-music-dub"
    assert body["status"] == "available"
    assert body["transcript"] == "lyrics"
    assert body["translated_lyrics"] == "translated lyrics"
    assert body["final_mix"]["path"] == "mix.wav"
    assert [step["type"] for step in body["steps"]] == ["stems", "stt", "translate", "voice_convert", "remix"]


def test_stem_separation_returns_artifacts_and_limitations(monkeypatch, studio_client, sample_wav_b64, tmp_path):
    from codai.api import audio_stems

    stem_paths = []
    for name in ("vocals.wav", "instrumental.wav"):
        path = tmp_path / name
        path.write_bytes(b"wav")
        stem_paths.append(str(path))

    def fake_split(audio_bytes, mode, workdir):
        assert audio_bytes
        assert mode == "vocals-instrumental"
        return {
            "stem_mode": mode,
            "artifacts": [
                {"name": "vocals", "path": stem_paths[0], "role": "lead-vocal"},
                {"name": "instrumental", "path": stem_paths[1], "role": "backing-mix"},
            ],
            "engine": "ffmpeg-phase-invert",
            "limitations": ["center-panned-only"],
        }

    monkeypatch.setattr(audio_stems, "_split_audio", fake_split)

    response = studio_client.post(
        "/v1/audio/stems",
        json={"audio": sample_wav_b64, "stem_mode": "vocals-instrumental", "response_format": "url", "fallback_mode": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stem_mode"] == "vocals-instrumental"
    assert body["backend"]["engine"] == "ffmpeg-phase-invert"
    assert body["backend"]["quality"] == "best-effort"
    assert len(body["data"]) == 2
    assert "/v1/files/" in body["data"][0]["url"]
    assert "/v1/files/" in body["data"][1]["url"]
    assert "center-panned-only" in body["limitations"]


def test_audio_cleanup_returns_artifact_and_applied_operations(monkeypatch, studio_client, sample_wav_b64, tmp_path):
    from codai.api import audio_clean

    cleaned_path = tmp_path / "cleaned.wav"
    cleaned_path.write_bytes(b"wav")

    def fake_cleanup(audio_bytes, options, workdir):
        assert audio_bytes
        assert options["noise_reduction"] is True
        assert options["normalize"] is True
        return {
            "path": str(cleaned_path),
            "engine": "ffmpeg-afftdn",
            "applied": ["noise_reduction", "normalize"],
            "limitations": ["not-ml-restoration"],
        }

    monkeypatch.setattr(audio_clean, "_cleanup_audio", fake_cleanup)

    response = studio_client.post(
        "/v1/audio/cleanup",
        json={
            "audio": sample_wav_b64,
            "noise_reduction": True,
            "normalize": True,
            "remove_hum": False,
            "repair_clicks": False,
            "response_format": "url",
            "fallback_mode": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["backend"]["engine"] == "ffmpeg-afftdn"
    assert body["backend"]["quality"] == "best-effort"
    assert body["applied"] == ["noise_reduction", "normalize"]
    assert body["limitations"] == ["not-ml-restoration"]


def test_admin_status_includes_recent_activity(studio_client, monkeypatch):
    from codai.admin import routes as admin_routes
    from codai.api import log as api_log

    api_log._activity.clear()
    api_log._activity.appendleft({
        "time": 1715000000,
        "model": "demo-model",
        "type": "chat",
        "status": 200,
        "duration": 1.23,
    })
    api_log._activity.appendleft({
        "time": "bad-time",
        "model": None,
        "type": None,
        "status": "500",
        "duration": "2.5",
    })

    monkeypatch.setattr(admin_routes, "config_manager", types.SimpleNamespace(models_data={}, pipelines_data=[]), raising=False)

    response = studio_client.get("/admin/api/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["recent_activity"][0] == {
        "time": 0,
        "model": "—",
        "type": "unknown",
        "status": 500,
        "duration": 2.5,
    }
    assert payload["recent_activity"][1] == {
        "time": 1715000000,
        "model": "demo-model",
        "type": "chat",
        "status": 200,
        "duration": 1.23,
    }


def test_chat_template_wires_preview_shells_for_new_runnable_panels():
    template_path = "/storage/coderai/.worktrees/web-admin-polish/codai/admin/templates/chat.html"
    text = open(template_path, "r", encoding="utf-8").read()

    assert "id=\"at-preview\"" in text
    assert "id=\"as-preview\"" in text
    assert "id=\"ig-preview\"" in text
    assert "id=\"em-preview\"" in text
    assert "id=\"ast-preview\"" in text
    assert "id=\"ac-preview\"" in text
    assert "'aud-tts':" in text
    assert "'aud-stt':" in text
    assert "'aud-stems':" in text
    assert "'aud-clean':" in text
    assert "buildAudioUnderstandPreviewData" in text
    assert "buildMusicDubPreviewData" in text
    assert "buildStemPreviewData" in text
    assert "buildCleanupPreviewData" in text


def test_chat_template_marks_full_quality_audio_panels_with_runtime_backend_metadata():
    template_path = "/storage/coderai/.worktrees/web-admin-polish/codai/admin/templates/chat.html"
    text = open(template_path, "r", encoding="utf-8").read()

    assert "audioBackendHealth" in text
    assert "renderAudioBackendHealth" in text
    assert "aud-music-dub" in text
    assert "aud-stems" in text
    assert "aud-clean" in text


def test_chat_template_exposes_ml_preview_and_artifact_markers():
    template_path = "/storage/coderai/.worktrees/web-admin-polish/codai/admin/templates/chat.html"
    text = open(template_path, "r", encoding="utf-8").read()

    assert "buildStemPreviewData" in text
    assert "buildCleanupPreviewData" in text
    assert "buildMusicDubPreviewData" in text
    assert "pushArtifactHistory({" in text
    assert "backend?.model" in text
    assert "translated_lyrics" in text


def test_studio_generation_panel_uses_wider_control_column():
    template_path = "/storage/coderai/.worktrees/web-admin-polish/codai/admin/templates/chat.html"
    text = open(template_path, "r", encoding="utf-8").read()

    assert ".gen-ctrl { width:min(380px,36vw); min-width:340px; max-width:420px;" in text
    assert "@media (max-width: 720px) {" in text
