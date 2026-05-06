from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient


def _base_models_data():
    return {
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


def _build_config(tmp_path):
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
    cfg.models_data = _base_models_data()
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
    return cfg


def _build_admin_test_client(routes):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.require_admin] = lambda: "admin"
    return app, TestClient(app)


def test_model_configure_persists_whisper_server_audio_model(monkeypatch, tmp_path):
    from codai.admin import routes

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app, client = _build_admin_test_client(routes)
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

    cfg = _build_config(tmp_path)
    cfg.models_data["audio_models"] = [
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
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app, client = _build_admin_test_client(routes)
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


def test_model_configure_defaults_missing_whisper_server_model_id_to_whisper0(monkeypatch, tmp_path):
    from codai.admin import routes

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app, client = _build_admin_test_client(routes)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "",
            "model_type": "audio_models",
            "backend": "whisper-server",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/ggml-base.bin",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"][0]["id"] == "whisper0"

    app.dependency_overrides.clear()


def test_model_configure_defaults_missing_whisper_server_model_id_to_smallest_free_suffix(monkeypatch, tmp_path):
    from codai.admin import routes

    cfg = _build_config(tmp_path)
    cfg.models_data["audio_models"] = [
        {"id": "whisper0", "backend": "whisper-server"},
        {"id": "whisper1", "backend": "whisper-server"},
        {"id": "whisper3", "backend": "whisper-server"},
    ]
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app, client = _build_admin_test_client(routes)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_type": "audio_models",
            "backend": "whisper-server",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/ggml-small.bin",
            "port": 8745,
            "gpu_device": 1,
            "load_mode": "load",
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"][-1]["id"] == "whisper2"

    app.dependency_overrides.clear()


def test_model_configure_defaults_missing_whisper_server_model_id_skips_non_whisper_server_collision(monkeypatch, tmp_path):
    from codai.admin import routes

    cfg = _build_config(tmp_path)
    cfg.models_data["audio_models"] = [
        {"id": "whisper0", "backend": "faster-whisper"},
    ]
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app, client = _build_admin_test_client(routes)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "backend": "whisper-server",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/ggml-base.bin",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"][-1]["id"] == "whisper0"

    app.dependency_overrides.clear()


def test_model_configure_allows_whisper_server_id_coexistence_with_other_audio_backends(monkeypatch, tmp_path):
    from codai.admin import routes

    cfg = _build_config(tmp_path)
    cfg.models_data["audio_models"] = [
        {"id": "whisper0", "backend": "faster-whisper"},
    ]
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app, client = _build_admin_test_client(routes)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "whisper0",
            "backend": "whisper-server",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/ggml-base.bin",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
        },
    )

    assert response.status_code == 200
    assert [m["backend"] for m in cfg.models_data["audio_models"]] == ["faster-whisper", "whisper-server"]
    assert cfg.models_data["audio_models"][-1]["id"] == "whisper0"

    app.dependency_overrides.clear()


def test_model_configure_defaults_missing_whisper_server_path_to_usr_local_bin(monkeypatch, tmp_path):
    from codai.admin import routes

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    monkeypatch.setattr(routes.shutil, "which", lambda _: None)
    app, client = _build_admin_test_client(routes)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "whisper-vulkan-base",
            "model_type": "audio_models",
            "backend": "whisper-server",
            "server_path": "",
            "model_path": "/models/ggml-base.bin",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"][0]["server_path"] == "/usr/local/bin/whisper-server"

    app.dependency_overrides.clear()


def test_model_configure_defaults_missing_whisper_server_path_uses_shutil_which_when_present(monkeypatch, tmp_path):
    from codai.admin import routes

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    monkeypatch.setattr(routes.shutil, "which", lambda _: "/opt/bin/whisper-server")
    app, client = _build_admin_test_client(routes)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "whisper-vulkan-base",
            "model_type": "audio_models",
            "backend": "whisper-server",
            "server_path": "",
            "model_path": "/models/ggml-base.bin",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"][0]["server_path"] == "/opt/bin/whisper-server"

    app.dependency_overrides.clear()


def test_model_configure_preserves_explicit_whisper_server_model_id_and_server_path_overrides(monkeypatch, tmp_path):
    from codai.admin import routes

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    monkeypatch.setattr(routes.shutil, "which", lambda _: "/opt/bin/whisper-server")
    app, client = _build_admin_test_client(routes)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "custom-whisper-id",
            "model_type": "audio_models",
            "backend": "whisper-server",
            "server_path": "/custom/bin/whisper-server",
            "model_path": "/models/custom.gguf",
            "port": 9123,
            "gpu_device": 2,
            "load_mode": "load",
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"][0]["id"] == "custom-whisper-id"
    assert cfg.models_data["audio_models"][0]["server_path"] == "/custom/bin/whisper-server"

    app.dependency_overrides.clear()


def test_model_configure_accepts_cached_gguf_whisper_server_model(monkeypatch, tmp_path):
    from codai.admin import routes

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app, client = _build_admin_test_client(routes)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "whisper-vulkan-base",
            "model_type": "audio_models",
            "backend": "whisper-server",
            "model_source": "cached-gguf",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/base.en.gguf",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"] == [
        {
            "id": "whisper-vulkan-base",
            "backend": "whisper-server",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/base.en.gguf",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
            "model_type": "audio_models",
            "model_types": ["audio_models"],
        }
    ]

    app.dependency_overrides.clear()


def test_model_configure_defaults_missing_whisper_server_model_source_to_cached_gguf(monkeypatch, tmp_path):
    from codai.admin import routes

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app, client = _build_admin_test_client(routes)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "whisper-vulkan-base",
            "model_type": "audio_models",
            "backend": "whisper-server",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/base.en.gguf",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"] == [
        {
            "id": "whisper-vulkan-base",
            "backend": "whisper-server",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/base.en.gguf",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
            "model_type": "audio_models",
            "model_types": ["audio_models"],
        }
    ]

    app.dependency_overrides.clear()


def test_model_configure_rejects_cached_gguf_whisper_server_without_model_path(monkeypatch, tmp_path):
    from codai.admin import routes

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app, client = _build_admin_test_client(routes)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "whisper-vulkan-base",
            "model_type": "audio_models",
            "backend": "whisper-server",
            "model_source": "cached-gguf",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "model_path is required for cached-gguf"

    app.dependency_overrides.clear()


def test_model_configure_rejects_manual_path_whisper_server_without_model_path(monkeypatch, tmp_path):
    from codai.admin import routes

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app, client = _build_admin_test_client(routes)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "whisper-vulkan-base",
            "model_type": "audio_models",
            "backend": "whisper-server",
            "model_source": "manual-path",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "model_path is required for manual-path"

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


def test_get_all_allowed_identifiers_includes_configured_whisper_server_id_without_legacy_alias(monkeypatch):
    from codai.admin import routes
    from codai.models.manager import MultiModelManager

    manager = MultiModelManager()
    manager.audio_models[:] = ["whisper-vulkan-base"]
    monkeypatch.setattr(
        routes,
        "config_manager",
        SimpleNamespace(
            models_data={
                "text_models": [],
                "image_models": [],
                "audio_models": [{"id": "whisper-vulkan-base", "backend": "whisper-server"}],
                "vision_models": [],
                "tts_models": [],
                "gguf_models": [],
                "video_models": [],
                "audio_gen_models": [],
                "embedding_models": [],
                "aliases": {},
            }
        ),
        raising=False,
    )

    allowed = manager.get_all_allowed_identifiers()

    assert "whisper-vulkan-base" in allowed
    assert "audio:whisper-vulkan-base" in allowed
    assert "whisper-server" not in allowed


def test_main_startup_registration_code_uses_entry_local_whisper_server_settings():
    content = Path("codai/main.py").read_text()

    assert 'm.get("backend") == "whisper-server"' in content
    assert 'server_path=m.get("server_path", "")' in content
    assert 'port=int(m.get("port", 8744))' in content
    assert 'gpu_device=int(m.get("gpu_device", 0))' in content
    assert 'config.whisper.server_path' not in content[content.index('audio_models = models_config.get("audio_models", [])'):content.index('# Image models')]


def test_settings_api_does_not_return_whisper_fields(monkeypatch):
    from codai.admin import routes
    from codai.api.app import app
    from codai.config import Config, ServerConfig, BackendConfig, ModelsConfig, OffloadConfig, VulkanConfig, ImageConfig, WhisperConfig

    cfg = SimpleNamespace(
        config=Config(
            version="1.0",
            server=ServerConfig(),
            backend=BackendConfig(),
            models=ModelsConfig(),
            offload=OffloadConfig(),
            vulkan=VulkanConfig(),
            image=ImageConfig(),
            whisper=WhisperConfig(server_path="/usr/local/bin/whisper-server", server_port=8744),
        )
    )
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app.dependency_overrides[routes.require_admin] = lambda: "admin"

    client = TestClient(app)
    response = client.get("/admin/api/settings")

    assert response.status_code == 200
    assert "whisper" not in response.json()

    app.dependency_overrides.clear()


def test_models_template_contains_whisper_server_add_model_form():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "Whisper-server simulated models" in template
    assert "Add model" in template
    assert "ws-model-id" in template
    assert "ws-server-path" in template
    assert "Downloaded GGUF" in template
    assert "Manual path" in template
    assert "ws-model-source" in template
    assert "ws-gguf-select" in template


def test_models_template_preserves_whisper_server_manual_path_controls():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "ws-model-path" in template
    assert 'placeholder="Manual path: /models/ggml-base.bin"' in template
    assert "modelPath.style.display = useCached ? 'none' : '';" in template


def test_models_template_defines_whisper_server_source_toggle_behavior():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "function toggleWhisperModelSource()" in template
    assert "const sourceSelect = document.getElementById('ws-model-source');" in template
    assert "const source = sourceSelect ? sourceSelect.value : 'cached-gguf';" in template
    assert "const ggufSelect = document.getElementById('ws-gguf-select');" in template
    assert "const modelPath = document.getElementById('ws-model-path');" in template
    assert "const useCached = source === 'cached-gguf';" in template
    assert "ggufSelect.style.display = useCached ? '' : 'none';" in template
    assert "modelPath.style.display = useCached ? 'none' : '';" in template


def test_models_template_defines_whisper_server_gguf_prefill_helpers():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "function refreshWhisperGgufOptions()" in template
    assert "function prefillWhisperServerFromGguf(path)" in template
    assert "prefillWhisperServerFromGguf(path)" in template


def test_models_template_defines_next_whisper_server_model_id_helper():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "function nextWhisperServerModelId()" in template
    assert "whisperModels.map(m => m.id || '')" in template
    assert "while(existingIds.has(`whisper${suffix}`)) suffix += 1;" in template
    assert "return `whisper${suffix}`;" in template


def test_models_template_defines_whisper_server_builder_default_helpers():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "function defaultWhisperServerPath()" in template
    assert "return '/usr/local/bin/whisper-server';" in template
    assert "function resetWhisperServerBuilderDefaults()" in template
    assert "const modelIdInput = document.getElementById('ws-model-id');" in template
    assert "const serverPathInput = document.getElementById('ws-server-path');" in template
    assert "if(modelIdInput && !modelIdInput.value.trim())" in template
    assert "if(serverPathInput && !serverPathInput.value.trim())" in template
    assert "modelIdInput.value = nextWhisperServerModelId();" in template
    assert "serverPathInput.value = defaultWhisperServerPath();" in template


def test_models_template_resets_whisper_server_builder_defaults_after_refresh_and_add():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "resetWhisperServerBuilderDefaults();" in template
    assert "document.getElementById('ws-model-id').value = nextWhisperServerModelId();" in template
    assert "document.getElementById('ws-server-path').value = defaultWhisperServerPath();" in template
    assert "document.getElementById('ws-model-path').value = '';" in template
    assert "document.getElementById('ws-model-source').value = 'cached-gguf';" in template
    assert "document.getElementById('ws-gguf-select').value = d.model_path || '';" in template
    assert "toggleWhisperModelSource();\n    refreshLocal();" in template
    assert "refreshLocal();" in template


def test_models_template_prefill_uses_reset_defaults_before_selecting_gguf():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "resetWhisperServerBuilderDefaults();" in template
    assert "sourceSelect.value = 'cached-gguf';" in template
    assert "ggufSelect.value = path;" in template


def test_models_template_submits_whisper_server_source_and_resolved_gguf_path():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "function getWhisperServerModelPath()" in template
    assert "const modelSource = document.getElementById('ws-model-source').value;" in template
    assert "? document.getElementById('ws-gguf-select').value" in template
    assert ": document.getElementById('ws-model-path').value.trim()) || null;" in template
    assert "model_source: modelSource," in template
    assert "model_path: getWhisperServerModelPath()," in template


def test_models_template_uses_fixed_cached_gguf_default_without_route_context():
    template = Path("codai/admin/templates/models.html").read_text()
    routes_content = Path("codai/admin/routes.py").read_text()

    assert '<option value="cached-gguf" selected>Downloaded GGUF</option>' in template
    assert "whisper_server_default_source" not in template
    assert "whisper_server_default_source" not in routes_content


def test_models_template_adds_use_with_whisper_server_gguf_action():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "Use with whisper-server" in template
    assert "onclick='prefillWhisperServerFromGguf(${JSON.stringify(f.path)})'" in template
    assert "prefillWhisperServerFromGguf('${esc(f.path)}')" not in template


def test_models_template_truncates_configured_whisper_server_model_paths():
    template = Path("codai/admin/templates/models.html").read_text()

    assert '<th style="text-align:left;padding:.3rem .25rem;font-weight:700">Model path</th>' in template
    assert 'max-width:160px;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;line-height:1.25;max-height:2.5em' in template
    assert 'title="${esc(m.model_path || \"—\")}"' in template
    assert '>${esc(m.model_path || "—")}</td>' in template


def test_settings_template_no_longer_contains_whisper_server_section():
    template = Path("codai/admin/templates/settings.html").read_text()

    assert "Whisper Server" not in template
    assert "wsStart" not in template
    assert "wsStop" not in template


def test_model_info_supports_whisper_server_metadata_fields():
    content = Path("codai/pydantic/textrequest.py").read_text()

    assert "backend: Optional[str] = None" in content
    assert "model_path: Optional[str] = None" in content
    assert "port: Optional[int] = None" in content
    assert "gpu_device: Optional[int] = None" in content
    assert "load_mode: Optional[str] = None" in content


def test_removed_whisper_server_admin_routes_return_not_found(monkeypatch):
    from codai.admin import routes
    from codai.api.app import app

    app.dependency_overrides[routes.require_admin] = lambda: "admin"
    client = TestClient(app)

    assert client.get("/admin/api/whisper-server/status").status_code == 404
    assert client.post("/admin/api/whisper-server/start", json={}).status_code == 404
    assert client.post("/admin/api/whisper-server/stop", json={}).status_code == 404

    app.dependency_overrides.clear()


def test_settings_template_keeps_queue_size_control():
    template = Path("codai/admin/templates/settings.html").read_text()

    assert "Request queue max size" in template
    assert "s-queue-max" in template
