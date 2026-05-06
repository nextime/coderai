# Whisper-Server Local Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add whisper-server simulated models to the Local Models page as first-class persisted audio models, remove the old Settings-based whisper-server UI, and unify whisper-server lifecycle with generic model load/unload and on-request behavior.

**Architecture:** Persist whisper-server definitions as `audio_models` entries in `models.json` keyed by `id`, register them at startup through the existing audio-model bootstrap, and route both admin load/unload and transcription on-request startup through the same `MultiModelManager` model lifecycle. Update the Models page to create and display these entries directly, and remove the separate Settings-only whisper-server workflow and endpoints.

**Tech Stack:** FastAPI, Jinja2 templates, vanilla JavaScript, Python config persistence, existing `WhisperServerManager` / `MultiModelManager` model runtime.

---

## File Structure

- Modify: `codai/admin/templates/models.html`
  - Add the whisper-server simulated models form to the Local Models tab.
  - Render configured whisper-server entries as local models with standard actions.
  - Remove the old whisper-server status polling card and switch to generic model status refresh.
- Modify: `codai/admin/templates/settings.html`
  - Remove the dedicated whisper-server configuration and start/stop section.
  - Remove page JS that reads or posts whisper-server settings fields.
- Modify: `codai/admin/routes.py`
  - Remove dedicated whisper-server admin endpoints.
  - Extend model configuration, model listing, model load, and model unload behavior for whisper-server entries.
  - Stop surfacing whisper-server settings fields in Settings API responses.
- Modify: `codai/models/manager.py`
  - Remove legacy single-instance whisper-server fallback state.
  - Ensure runtime registration, allowed-model detection, load/unload behavior, and model listing metadata work for whisper-server entries keyed by `id`.
- Modify: `codai/api/transcriptions.py`
  - Remove single-instance fallback logic and rely only on configured whisper-server model ids.
- Modify: `codai/main.py`
  - Register whisper-server entries using entry-local settings only, without legacy config fallback.
- Modify: `codai/config.py`
  - Stop persisting/surfacing whisper-server settings in `config.json` if no longer used by the UI.
- Create: `tests/test_whisper_server_local_models.py`
  - Add focused regression tests for persistence, load/unload, and transcription routing.

## Task 1: Add backend tests for whisper-server model persistence and runtime lifecycle

**Files:**
- Create: `tests/test_whisper_server_local_models.py`
- Modify: `codai/admin/routes.py`
- Modify: `codai/models/manager.py`
- Modify: `codai/api/transcriptions.py`

- [ ] **Step 1: Write the failing persistence test for whisper-server model creation**

```python
from types import SimpleNamespace

from fastapi.testclient import TestClient


def test_model_configure_persists_whisper_server_audio_model(monkeypatch, tmp_path):
    from codai.admin import routes
    from codai.config import ConfigManager, Config, ServerConfig, BackendConfig, ModelsConfig, OffloadConfig, VulkanConfig, ImageConfig, WhisperConfig
    from codai.main import app

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
```

- [ ] **Step 2: Run the persistence test to verify it fails**

Run: `pytest tests/test_whisper_server_local_models.py::test_model_configure_persists_whisper_server_audio_model -v`
Expected: FAIL because the current configure endpoint writes `path`-oriented entries and does not build a whisper-server `id` entry.

- [ ] **Step 3: Write the failing duplicate-id validation test**

```python
def test_model_configure_rejects_duplicate_whisper_server_model_id(monkeypatch, tmp_path):
    from codai.admin import routes
    from codai.config import ConfigManager, Config, ServerConfig, BackendConfig, ModelsConfig, OffloadConfig, VulkanConfig, ImageConfig, WhisperConfig
    from codai.main import app

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
```

- [ ] **Step 4: Run the duplicate-id test to verify it fails**

Run: `pytest tests/test_whisper_server_local_models.py::test_model_configure_rejects_duplicate_whisper_server_model_id -v`
Expected: FAIL because duplicate whisper-server ids are not validated.

- [ ] **Step 5: Write the failing admin load/unload lifecycle test**

```python
from types import SimpleNamespace


def test_model_load_and_unload_manage_whisper_server_runtime(monkeypatch):
    from codai.admin import routes
    from codai.main import app
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

    monkeypatch.setattr(routes, "config_manager", SimpleNamespace(models_data={
        "audio_models": [{
            "id": "whisper-vulkan-base",
            "backend": "whisper-server",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/ggml-base.bin",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
        }]
    }), raising=False)
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
```

- [ ] **Step 6: Run the lifecycle test to verify it fails**

Run: `pytest tests/test_whisper_server_local_models.py::test_model_load_and_unload_manage_whisper_server_runtime -v`
Expected: FAIL because generic load/unload does not yet manage whisper-server lifecycle directly.

- [ ] **Step 7: Write the failing transcription-routing test without legacy fallback**

```python
def test_transcription_requires_configured_whisper_server_model_id(monkeypatch):
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

    import pytest
    with pytest.raises(Exception) as exc:
        import asyncio
        asyncio.run(run_call())

    assert "not configured" in str(exc.value).lower() or "not available" in str(exc.value).lower()
```

- [ ] **Step 8: Run the transcription-routing test to verify it fails**

Run: `pytest tests/test_whisper_server_local_models.py::test_transcription_requires_configured_whisper_server_model_id -v`
Expected: FAIL because legacy fallback behavior still accepts the old single-instance path.

- [ ] **Step 9: Implement the minimal backend changes to make the four tests pass**

```python
# codai/admin/routes.py (inside api_model_configure)
if data.get("backend") == "whisper-server":
    model_id = (data.get("model_id") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    if not data.get("server_path"):
        raise HTTPException(status_code=400, detail="server_path is required")
    port = int(data.get("port", 8744))
    gpu_device = int(data.get("gpu_device", 0))
    for existing in config_manager.models_data.get("audio_models", []):
        if isinstance(existing, dict) and existing.get("id") == model_id:
            raise HTTPException(status_code=409, detail=f"whisper-server model '{model_id}' already exists")
    entry = {
        "id": model_id,
        "backend": "whisper-server",
        "server_path": data["server_path"],
        "model_path": data.get("model_path") or None,
        "port": port,
        "gpu_device": gpu_device,
        "load_mode": data.get("load_mode", "on-request"),
        "model_type": "audio_models",
        "model_types": ["audio_models"],
    }
    if data.get("used_vram_gb") is not None:
        entry["used_vram_gb"] = data["used_vram_gb"]
    config_manager.models_data.setdefault("audio_models", []).append(entry)
    config_manager.save_models()
    return {"success": True}

# codai/admin/routes.py (inside api_model_load)
if model_type == "audio":
    wsm = multi_model_manager.whisper_servers.get(path)
    if wsm is not None:
        started = wsm.start(getattr(wsm, "_model_path", None), gpu_device=getattr(wsm, "_gpu_device", 0))
        if not wsm.is_running():
            raise RuntimeError("whisper-server failed to start")
        key = f"audio:{path}"
        multi_model_manager.models[key] = wsm
        multi_model_manager.active_in_vram = key
        multi_model_manager.models_in_vram.add(key)
        return {"success": True, "already_loaded": False, "started_model": started}

# codai/api/transcriptions.py
wsm = multi_model_manager.whisper_servers.get(model)
```

- [ ] **Step 10: Run the focused backend tests to verify they pass**

Run: `pytest tests/test_whisper_server_local_models.py -v`
Expected: PASS for the new persistence, duplicate, lifecycle, and transcription tests.

- [ ] **Step 11: Commit the backend test-first lifecycle foundation**

```bash
git add tests/test_whisper_server_local_models.py codai/admin/routes.py codai/models/manager.py codai/api/transcriptions.py
git commit -m "feat: persist and load whisper-server audio models"
```

## Task 2: Remove legacy whisper-server fallback and settings exposure

**Files:**
- Modify: `codai/models/manager.py`
- Modify: `codai/main.py`
- Modify: `codai/admin/routes.py`
- Modify: `codai/config.py`
- Modify: `tests/test_whisper_server_local_models.py`

- [ ] **Step 1: Write the failing manager test for configured whisper-server identifiers only**

```python
def test_get_all_allowed_identifiers_includes_configured_whisper_server_id_without_legacy_alias(monkeypatch):
    from types import SimpleNamespace
    from codai.admin import routes
    from codai.models.manager import MultiModelManager

    manager = MultiModelManager()
    manager.audio_models[:] = ["whisper-vulkan-base"]
    monkeypatch.setattr(routes, "config_manager", SimpleNamespace(models_data={
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
    }), raising=False)

    allowed = manager.get_all_allowed_identifiers()

    assert "whisper-vulkan-base" in allowed
    assert "audio:whisper-vulkan-base" in allowed
    assert "whisper-server" not in allowed
```

- [ ] **Step 2: Run the identifier test to verify it fails**

Run: `pytest tests/test_whisper_server_local_models.py::test_get_all_allowed_identifiers_includes_configured_whisper_server_id_without_legacy_alias -v`
Expected: FAIL because legacy identifiers and fallback behavior are still present.

- [ ] **Step 3: Write the failing startup-registration test for entry-local server settings**

```python
def test_startup_registration_uses_whisper_server_entry_settings(monkeypatch):
    from codai.main import setup_models
    from codai.models.manager import multi_model_manager

    calls = []
    monkeypatch.setattr(multi_model_manager, "register_whisper_server", lambda **kwargs: calls.append(kwargs))

    class DummyConfig:
        whisper = type("Whisper", (), {"server_path": "/legacy/path", "server_port": 9999})()
        vulkan = type("Vulkan", (), {"device_id": 7})()

    setup_models(
        DummyConfig(),
        {
            "audio_models": [{
                "id": "whisper-vulkan-base",
                "backend": "whisper-server",
                "server_path": "/usr/local/bin/whisper-server",
                "model_path": "/models/ggml-base.bin",
                "port": 8744,
                "gpu_device": 0,
            }]
        },
        backend="auto",
    )

    assert calls == [{
        "model_id": "whisper-vulkan-base",
        "server_path": "/usr/local/bin/whisper-server",
        "model_path": "/models/ggml-base.bin",
        "port": 8744,
        "gpu_device": 0,
        "config": {"backend": "whisper-server", "server_path": "/usr/local/bin/whisper-server", "model_path": "/models/ggml-base.bin", "port": 8744, "gpu_device": 0},
    }]
```

- [ ] **Step 4: Run the startup-registration test to verify it fails**

Run: `pytest tests/test_whisper_server_local_models.py::test_startup_registration_uses_whisper_server_entry_settings -v`
Expected: FAIL because startup still falls back to `config.whisper` values.

- [ ] **Step 5: Write the failing settings-API test proving whisper UI fields are removed**

```python
def test_settings_api_does_not_return_whisper_fields(monkeypatch):
    from codai.admin import routes
    from codai.main import app
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
```

- [ ] **Step 6: Run the settings-API test to verify it fails**

Run: `pytest tests/test_whisper_server_local_models.py::test_settings_api_does_not_return_whisper_fields -v`
Expected: FAIL because the Settings API still returns whisper settings.

- [ ] **Step 7: Implement the minimal legacy-removal changes to make the tests pass**

```python
# codai/models/manager.py
self.whisper_servers: Dict[str, WhisperServerManager] = {}
# remove self.whisper_server from __init__

# codai/main.py
multi_model_manager.register_whisper_server(
    model_id=mid,
    server_path=m.get("server_path", ""),
    model_path=m.get("model_path") or None,
    port=int(m.get("port", 8744)),
    gpu_device=int(m.get("gpu_device", 0)),
    config=cfg,
)

# codai/admin/routes.py
return {
    "server": {...},
    "backend": {...},
    "models": {...},
    "offload": {...},
    "vulkan": {...},
    "system_prompt": c.system_prompt,
    "tools_closer_prompt": c.tools_closer_prompt,
    "grammar_guided": c.grammar_guided,
    "parser": c.parser,
}

# codai/config.py
config_dict = {
    "version": self.config.version,
    "server": {...},
    "backend": {...},
    "models": {...},
    "offload": {...},
    "vulkan": {...},
    "image": {...},
    "system_prompt": self.config.system_prompt,
    "tools_closer_prompt": self.config.tools_closer_prompt,
    "grammar_guided": self.config.grammar_guided,
    "file_path": self.config.file_path,
    "hf_chat_templates": self.config.hf_chat_templates,
    "reasoning_options": self.config.reasoning_options,
    "parser": self.config.parser,
}
```

- [ ] **Step 8: Run the legacy-removal tests to verify they pass**

Run: `pytest tests/test_whisper_server_local_models.py -k "allowed_identifiers or startup_registration or settings_api" -v`
Expected: PASS for the new manager, startup, and settings API tests.

- [ ] **Step 9: Commit the legacy whisper-server removal**

```bash
git add tests/test_whisper_server_local_models.py codai/models/manager.py codai/main.py codai/admin/routes.py codai/config.py
git commit -m "refactor: remove legacy whisper-server settings flow"
```

## Task 3: Surface whisper-server entries in the Local Models UI

**Files:**
- Modify: `codai/admin/templates/models.html`
- Modify: `codai/admin/routes.py`
- Modify: `codai/models/manager.py`
- Modify: `tests/test_whisper_server_local_models.py`

- [ ] **Step 1: Write the failing template smoke test for the Local Models whisper-server form**

```python
def test_models_template_contains_whisper_server_add_model_form():
    from pathlib import Path

    template = Path("codai/admin/templates/models.html").read_text()

    assert "Whisper-server simulated models" in template
    assert "Add model" in template
    assert "ws-model-id" in template
    assert "ws-server-path" in template
```

- [ ] **Step 2: Run the template smoke test to verify it fails**

Run: `pytest tests/test_whisper_server_local_models.py::test_models_template_contains_whisper_server_add_model_form -v`
Expected: FAIL because the Models page still has only the old whisper status card.

- [ ] **Step 3: Write the failing template smoke test for settings cleanup**

```python
def test_settings_template_no_longer_contains_whisper_server_section():
    from pathlib import Path

    template = Path("codai/admin/templates/settings.html").read_text()

    assert "Whisper Server" not in template
    assert "wsStart" not in template
    assert "wsStop" not in template
```

- [ ] **Step 4: Run the settings-template smoke test to verify it fails**

Run: `pytest tests/test_whisper_server_local_models.py::test_settings_template_no_longer_contains_whisper_server_section -v`
Expected: FAIL because the old settings block still exists.

- [ ] **Step 5: Write the failing listing-metadata test for whisper-server models**

```python
def test_list_models_includes_whisper_server_metadata(monkeypatch):
    from types import SimpleNamespace
    from codai.admin import routes
    from codai.models.manager import MultiModelManager

    manager = MultiModelManager()
    monkeypatch.setattr(routes, "config_manager", SimpleNamespace(models_data={
        "text_models": [],
        "image_models": [],
        "audio_models": [{
            "id": "whisper-vulkan-base",
            "backend": "whisper-server",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/ggml-base.bin",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
        }],
        "vision_models": [],
        "tts_models": [],
        "gguf_models": [],
        "video_models": [],
        "audio_gen_models": [],
        "embedding_models": [],
        "aliases": {},
    }), raising=False)

    models = manager.list_models()
    row = next(m for m in models if m.id == "whisper-vulkan-base")

    assert row.type == "audio"
    assert getattr(row, "backend", None) == "whisper-server"
    assert getattr(row, "model_path", None) == "/models/ggml-base.bin"
```

- [ ] **Step 6: Run the listing-metadata test to verify it fails**

Run: `pytest tests/test_whisper_server_local_models.py::test_list_models_includes_whisper_server_metadata -v`
Expected: FAIL because list-model payloads do not expose whisper-server metadata.

- [ ] **Step 7: Implement the minimal UI and metadata changes to make the tests pass**

```html
<!-- codai/admin/templates/models.html -->
<div class="card" id="ws-model-builder">
  <div class="card-title">Whisper-server simulated models</div>
  <p class="muted small">Create local audio models backed by dedicated whisper-server subprocess configurations.</p>
  <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem">
    <input id="ws-model-id" class="form-input" placeholder="whisper-vulkan-base">
    <input id="ws-server-path" class="form-input" placeholder="/usr/local/bin/whisper-server">
    <input id="ws-model-path" class="form-input" placeholder="/models/ggml-base.bin">
    <input id="ws-port" class="form-input" type="number" value="8744">
    <input id="ws-gpu-device" class="form-input" type="number" value="0">
    <select id="ws-load-mode" class="form-input">
      <option value="on-request">On request</option>
      <option value="load">Load</option>
    </select>
  </div>
  <div style="margin-top:.75rem">
    <button class="btn btn-primary" onclick="addWhisperServerModel()">Add model</button>
  </div>
</div>
```

```javascript
async function addWhisperServerModel(){
  const payload = {
    model_id: document.getElementById('ws-model-id').value.trim(),
    model_type: 'audio_models',
    backend: 'whisper-server',
    server_path: document.getElementById('ws-server-path').value.trim(),
    model_path: document.getElementById('ws-model-path').value.trim() || null,
    port: parseInt(document.getElementById('ws-port').value, 10) || 8744,
    gpu_device: parseInt(document.getElementById('ws-gpu-device').value, 10) || 0,
    load_mode: document.getElementById('ws-load-mode').value,
  };
  const r = await fetch('/admin/api/model-configure', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  const d = await r.json();
  if(!r.ok) throw new Error(d.detail || 'Failed to add whisper-server model');
  refreshLocal();
}
```

```python
# codai/models/manager.py
models.append(ModelInfo(
    id=model_id,
    type=resolved_type,
    capabilities=caps.to_list(),
    backend=meta.get("backend"),
    model_path=meta.get("model_path"),
    port=meta.get("port"),
    gpu_device=meta.get("gpu_device"),
    load_mode=meta.get("load_mode"),
))
```

- [ ] **Step 8: Run the UI and metadata tests to verify they pass**

Run: `pytest tests/test_whisper_server_local_models.py -k "template_contains_whisper_server or settings_template_no_longer or list_models_includes_whisper_server_metadata" -v`
Expected: PASS for the Local Models form, Settings cleanup, and listing metadata tests.

- [ ] **Step 9: Commit the Models-page whisper-server UI**

```bash
git add tests/test_whisper_server_local_models.py codai/admin/templates/models.html codai/admin/templates/settings.html codai/admin/routes.py codai/models/manager.py
git commit -m "feat: manage whisper-server models from local models page"
```

## Task 4: Finish unified verification and cleanup

**Files:**
- Modify: `codai/admin/templates/models.html`
- Modify: `codai/admin/routes.py`
- Modify: `codai/models/manager.py`
- Modify: `tests/test_whisper_server_local_models.py`

- [ ] **Step 1: Write the failing endpoint-absence test for removed whisper-server admin routes**

```python
def test_removed_whisper_server_admin_routes_return_not_found(monkeypatch):
    from codai.admin import routes
    from codai.main import app

    app.dependency_overrides[routes.require_admin] = lambda: "admin"
    client = TestClient(app)

    assert client.get("/admin/api/whisper-server/status").status_code == 404
    assert client.post("/admin/api/whisper-server/start", json={}).status_code == 404
    assert client.post("/admin/api/whisper-server/stop", json={}).status_code == 404

    app.dependency_overrides.clear()
```

- [ ] **Step 2: Run the endpoint-absence test to verify it fails**

Run: `pytest tests/test_whisper_server_local_models.py::test_removed_whisper_server_admin_routes_return_not_found -v`
Expected: FAIL because the old whisper-server endpoints still exist.

- [ ] **Step 3: Remove the old whisper-server admin endpoints and dead JS references**

```python
# codai/admin/routes.py
# delete:
# @router.get("/admin/api/whisper-server/status")
# @router.post("/admin/api/whisper-server/start")
# @router.post("/admin/api/whisper-server/stop")
```

```javascript
// codai/admin/templates/models.html
// delete loadWsStatus(), ws polling, and refresh hooks tied to /admin/api/whisper-server/status
refreshLocal();
```

- [ ] **Step 4: Run the endpoint-absence test to verify it passes**

Run: `pytest tests/test_whisper_server_local_models.py::test_removed_whisper_server_admin_routes_return_not_found -v`
Expected: PASS.

- [ ] **Step 5: Run the full targeted test file**

Run: `pytest tests/test_whisper_server_local_models.py -v`
Expected: PASS for all whisper-server local-model tests.

- [ ] **Step 6: Run a broader admin/transcription regression slice**

Run: `pytest -k "admin or transcription" -v`
Expected: PASS, or only unrelated pre-existing failures.

- [ ] **Step 7: Inspect git diff and verify no leftover settings-era whisper-server strings remain**

Run: `git diff -- codai/admin/templates/settings.html codai/admin/templates/models.html codai/admin/routes.py codai/models/manager.py codai/api/transcriptions.py codai/main.py codai/config.py tests/test_whisper_server_local_models.py`
Expected: only the planned whisper-server local-model integration changes.

- [ ] **Step 8: Commit the final cleanup and verification**

```bash
git add codai/admin/templates/models.html codai/admin/templates/settings.html codai/admin/routes.py codai/models/manager.py codai/api/transcriptions.py codai/main.py codai/config.py tests/test_whisper_server_local_models.py
git commit -m "refactor: unify whisper-server with model lifecycle"
```
