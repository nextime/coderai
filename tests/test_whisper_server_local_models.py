from types import SimpleNamespace

from fastapi.testclient import TestClient


def test_model_configure_persists_whisper_server_audio_model(monkeypatch, tmp_path):
    from codai.admin import routes
    from codai.api.app import app
    from codai.config import (
        BackendConfig,
        Config,
        ConfigManager,
        ImageConfig,
        ModelsConfig,
        OffloadConfig,
        ServerConfig,
        VulkanConfig,
        WhisperConfig,
    )

    cfg = ConfigManager(str(tmp_path))
    cfg.models_data = {
        "text_models": [],
        "image_models": [],
        "audio_models": [],
        "vision_models": [],
        "tts_models": [],
        "gguf_models": [],
        "video_models": [],
        "audio_gen_models": [],
        "embedding_models": [],
        "aliases": {},
    }
    cfg.config = Config(
        version="1.0",
        server=ServerConfig(),
        backend=BackendConfig(),
        models=ModelsConfig(),
        offload=OffloadConfig(),
        vulkan=VulkanConfig(),
        image=ImageConfig(),
        whisper=WhisperConfig(),
    )
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app.dependency_overrides[routes.require_admin] = lambda: "admin"

    client = TestClient(app)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "whisper-vulkan-base",
            "model_type": "audio_models",
            "backend": "whisper-server",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/ggml-base.bin",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
            "used_vram_gb": 1.8,
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"] == [
        {
            "id": "whisper-vulkan-base",
            "backend": "whisper-server",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/ggml-base.bin",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
            "used_vram_gb": 1.8,
            "model_type": "audio_models",
            "model_types": ["audio_models"],
        }
    ]

    app.dependency_overrides.clear()


def test_model_configure_rejects_duplicate_whisper_server_model_id(monkeypatch, tmp_path):
    from codai.admin import routes
    from codai.api.app import app
    from codai.config import (
        BackendConfig,
        Config,
        ConfigManager,
        ImageConfig,
        ModelsConfig,
        OffloadConfig,
        ServerConfig,
        VulkanConfig,
        WhisperConfig,
    )

    cfg = ConfigManager(str(tmp_path))
    cfg.models_data = {
        "text_models": [],
        "image_models": [],
        "audio_models": [
            {
                "id": "whisper-vulkan-base",
                "backend": "whisper-server",
                "server_path": "/usr/local/bin/whisper-server",
                "model_path": "/models/ggml-base.bin",
                "port": 8744,
                "gpu_device": 0,
                "load_mode": "on-request",
            }
        ],
        "vision_models": [],
        "tts_models": [],
        "gguf_models": [],
        "video_models": [],
        "audio_gen_models": [],
        "embedding_models": [],
        "aliases": {},
    }
    cfg.config = Config(
        version="1.0",
        server=ServerConfig(),
        backend=BackendConfig(),
        models=ModelsConfig(),
        offload=OffloadConfig(),
        vulkan=VulkanConfig(),
        image=ImageConfig(),
        whisper=WhisperConfig(),
    )
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app.dependency_overrides[routes.require_admin] = lambda: "admin"

    client = TestClient(app)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "whisper-vulkan-base",
            "model_type": "audio_models",
            "backend": "whisper-server",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/ggml-small.bin",
            "port": 8745,
            "gpu_device": 1,
            "load_mode": "load",
        },
    )

    assert response.status_code in {400, 409}
    assert "duplicate" in response.text.lower() or "already" in response.text.lower()

    app.dependency_overrides.clear()


def test_model_load_and_unload_manage_whisper_server_runtime(monkeypatch):
    from codai.admin import routes
    from codai.api.app import app
    from codai.models.manager import multi_model_manager

    runtime = SimpleNamespace(
        started=[],
        stopped=False,
        is_running=lambda: True,
        start=lambda model_path=None, gpu_device=0: runtime.started.append((model_path, gpu_device)) or model_path,
        cleanup=lambda: setattr(runtime, "stopped", True),
        _model_path="/models/ggml-base.bin",
        _gpu_device=0,
    )

    monkeypatch.setattr(
        routes,
        "config_manager",
        SimpleNamespace(
            models_data={
                "audio_models": [
                    {
                        "id": "whisper-vulkan-base",
                        "backend": "whisper-server",
                        "server_path": "/usr/local/bin/whisper-server",
                        "model_path": "/models/ggml-base.bin",
                        "port": 8744,
                        "gpu_device": 0,
                        "load_mode": "on-request",
                    }
                ]
            }
        ),
        raising=False,
    )
    monkeypatch.setitem(multi_model_manager.whisper_servers, "whisper-vulkan-base", runtime)
    multi_model_manager.models.clear()
    app.dependency_overrides[routes.require_admin] = lambda: "admin"

    client = TestClient(app)
    load_response = client.post("/admin/api/model-load", json={"path": "whisper-vulkan-base"})
    assert load_response.status_code == 200
    assert runtime.started == [("/models/ggml-base.bin", 0)]
    assert "audio:whisper-vulkan-base" in multi_model_manager.models

    unload_response = client.post("/admin/api/model-unload", json={"path": "whisper-vulkan-base"})
    assert unload_response.status_code == 200
    assert runtime.stopped is True
    assert "audio:whisper-vulkan-base" not in multi_model_manager.models

    app.dependency_overrides.clear()
    multi_model_manager.models.clear()
    multi_model_manager.whisper_servers.clear()


def test_transcription_requires_configured_whisper_server_model_id():
    import asyncio

    import pytest
    from fastapi import HTTPException

    from codai.api import transcriptions
    from codai.models.manager import multi_model_manager

    multi_model_manager.whisper_servers.clear()
    multi_model_manager.models.clear()
    multi_model_manager.audio_models[:] = []

    class DummyUpload:
        filename = "sample.wav"

        async def read(self):
            return b"audio"

    async def run_call():
        return await transcriptions.create_transcription(
            model="whisper-server",
            file=DummyUpload(),
            language=None,
            prompt=None,
            response_format="json",
            temperature=0.0,
        )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(run_call())

    assert exc.value.status_code in {400, 404}
    assert "not configured" in str(exc.value.detail).lower() or "not available" in str(exc.value.detail).lower()
