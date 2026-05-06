# Whisper-Server GGUF Association Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let whisper-server simulated models on the Local Models page use either a downloaded local GGUF file or a manual path, while keeping the persisted whisper-server model schema unchanged.

**Architecture:** Extend the existing whisper-server builder in `codai/admin/templates/models.html` with a model-source selector and reuse the already-loaded GGUF cache inventory to populate a dropdown of downloaded files. Keep `POST /admin/api/model-configure` as the only persistence path, but tighten whisper-server validation around the final resolved `model_path` so both source modes converge to the same saved config shape.

**Tech Stack:** FastAPI, Jinja2 templates, vanilla JavaScript, Python config persistence, pytest.

---

## File Structure

- Modify: `codai/admin/templates/models.html`
  - Add `Downloaded GGUF` and `Manual path` controls to the existing whisper-server builder.
  - Reuse the local GGUF cache list in page JS to feed the whisper-server selector.
  - Optionally add a GGUF-row shortcut that pre-fills the builder.
- Modify: `codai/admin/routes.py`
  - Tighten whisper-server `model_path` validation for the two source modes while keeping the persisted schema unchanged.
- Modify: `tests/test_whisper_server_local_models.py`
  - Add regression tests for dual source mode rendering and whisper-server config validation.

## Task 1: Add backend validation for downloaded GGUF and manual-path whisper-server configs

**Files:**
- Modify: `tests/test_whisper_server_local_models.py`
- Modify: `codai/admin/routes.py`

- [ ] **Step 1: Write the failing test for a downloaded-GGUF whisper-server configuration**

```python
def test_model_configure_accepts_whisper_server_cached_gguf_path(monkeypatch, tmp_path):
    from codai.admin import routes
    from codai.api.app import app

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app.dependency_overrides[routes.require_admin] = lambda: "admin"

    client = TestClient(app)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "whisper-vulkan-small",
            "backend": "whisper-server",
            "model_type": "audio_models",
            "model_source": "cached-gguf",
            "model_path": "/storage/coderai/.cache/models/ggml-small-q5.gguf",
            "server_path": "/usr/local/bin/whisper-server",
            "port": 8744,
            "gpu_device": 0,
            "load_mode": "on-request",
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"][0]["model_path"] == "/storage/coderai/.cache/models/ggml-small-q5.gguf"
    assert cfg.models_data["audio_models"][0]["backend"] == "whisper-server"

    app.dependency_overrides.clear()
```

- [ ] **Step 2: Run the downloaded-GGUF config test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_model_configure_accepts_whisper_server_cached_gguf_path -v`
Expected: FAIL because the endpoint does not yet understand the explicit `model_source` workflow.

- [ ] **Step 3: Write the failing test for missing selected downloaded GGUF**

```python
def test_model_configure_rejects_whisper_server_missing_cached_gguf_selection(monkeypatch, tmp_path):
    from codai.admin import routes
    from codai.api.app import app

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app.dependency_overrides[routes.require_admin] = lambda: "admin"

    client = TestClient(app)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "whisper-vulkan-small",
            "backend": "whisper-server",
            "model_type": "audio_models",
            "model_source": "cached-gguf",
            "model_path": "",
            "server_path": "/usr/local/bin/whisper-server",
            "port": 8744,
            "gpu_device": 0,
        },
    )

    assert response.status_code == 400
    assert "model_path" in response.text.lower() or "gguf" in response.text.lower()

    app.dependency_overrides.clear()
```

- [ ] **Step 4: Run the missing-selection test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_model_configure_rejects_whisper_server_missing_cached_gguf_selection -v`
Expected: FAIL because the endpoint does not yet distinguish this source-specific validation case.

- [ ] **Step 5: Write the failing test for missing manual path**

```python
def test_model_configure_rejects_whisper_server_missing_manual_path(monkeypatch, tmp_path):
    from codai.admin import routes
    from codai.api.app import app

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app.dependency_overrides[routes.require_admin] = lambda: "admin"

    client = TestClient(app)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "whisper-vulkan-small",
            "backend": "whisper-server",
            "model_type": "audio_models",
            "model_source": "manual-path",
            "model_path": "",
            "server_path": "/usr/local/bin/whisper-server",
            "port": 8744,
            "gpu_device": 0,
        },
    )

    assert response.status_code == 400
    assert "model_path" in response.text.lower() or "manual" in response.text.lower()

    app.dependency_overrides.clear()
```

- [ ] **Step 6: Run the missing-manual-path test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_model_configure_rejects_whisper_server_missing_manual_path -v`
Expected: FAIL because the endpoint currently just stores the raw path without source-specific messaging.

- [ ] **Step 7: Implement the minimal whisper-server validation changes**

```python
# codai/admin/routes.py inside api_model_configure whisper-server branch
model_source = (data.get("model_source") or "manual-path").strip()
model_path = (data.get("model_path") or "").strip()
if model_source not in {"cached-gguf", "manual-path"}:
    raise HTTPException(status_code=400, detail="model_source must be 'cached-gguf' or 'manual-path'")
if not model_path:
    if model_source == "cached-gguf":
        raise HTTPException(status_code=400, detail="model_path is required for selected downloaded GGUF")
    raise HTTPException(status_code=400, detail="model_path is required for manual whisper-server path")

entry = {
    "id": model_id,
    "backend": "whisper-server",
    "server_path": server_path,
    "model_path": model_path,
    "port": port,
    "gpu_device": gpu_device,
    "load_mode": data.get("load_mode", "on-request"),
    "model_type": "audio_models",
    "model_types": ["audio_models"],
}
```

- [ ] **Step 8: Run the focused backend whisper-server validation tests**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py -k "cached_gguf_path or missing_cached_gguf_selection or missing_manual_path" -v`
Expected: PASS for the new whisper-server source validation tests.

- [ ] **Step 9: Commit the whisper-server source validation work**

```bash
git add tests/test_whisper_server_local_models.py codai/admin/routes.py
git commit -m "feat: validate whisper-server gguf model sources"
```

## Task 2: Add downloaded-GGUF and manual-path controls to the whisper-server builder

**Files:**
- Modify: `tests/test_whisper_server_local_models.py`
- Modify: `codai/admin/templates/models.html`

- [ ] **Step 1: Write the failing template test for both whisper-server source controls**

```python
def test_models_template_contains_whisper_server_source_controls():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "Downloaded GGUF" in template
    assert "Manual path" in template
    assert "ws-model-source" in template
    assert "ws-gguf-select" in template
```

- [ ] **Step 2: Run the source-controls template test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_models_template_contains_whisper_server_source_controls -v`
Expected: FAIL because the builder currently only exposes the plain manual-path input.

- [ ] **Step 3: Write the failing template test that manual path remains available**

```python
def test_models_template_keeps_whisper_server_manual_path_input():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "ws-model-path" in template
    assert "Manual path" in template
```

- [ ] **Step 4: Run the manual-path preservation template test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_models_template_keeps_whisper_server_manual_path_input -v`
Expected: FAIL until the builder is refactored to expose both modes explicitly.

- [ ] **Step 5: Implement the minimal dual-source builder UI**

```html
<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem;margin-bottom:.75rem">
  <select id="ws-model-source" class="form-input" onchange="toggleWhisperModelSource()">
    <option value="cached-gguf">Downloaded GGUF</option>
    <option value="manual-path">Manual path</option>
  </select>
  <select id="ws-gguf-select" class="form-input"></select>
  <input id="ws-model-path" class="form-input" placeholder="/models/ggml-base.bin">
</div>
```

```javascript
function toggleWhisperModelSource(){
  const source = document.getElementById('ws-model-source').value;
  const gguf = document.getElementById('ws-gguf-select');
  const manual = document.getElementById('ws-model-path');
  gguf.style.display = source === 'cached-gguf' ? '' : 'none';
  manual.style.display = source === 'manual-path' ? '' : 'none';
}
```

- [ ] **Step 6: Run the builder template tests to verify they pass**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py -k "source_controls or keeps_whisper_server_manual_path_input" -v`
Expected: PASS for the new dual-source builder tests.

- [ ] **Step 7: Commit the dual-source whisper-server builder UI**

```bash
git add tests/test_whisper_server_local_models.py codai/admin/templates/models.html
git commit -m "feat: add whisper-server gguf source selector"
```

## Task 3: Feed cached GGUF files into the whisper-server builder and support prefill shortcuts

**Files:**
- Modify: `tests/test_whisper_server_local_models.py`
- Modify: `codai/admin/templates/models.html`

- [ ] **Step 1: Write the failing template test for whisper-server GGUF selector hydration support**

```python
def test_models_template_contains_whisper_server_gguf_prefill_helpers():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "refreshWhisperGgufOptions" in template
    assert "prefillWhisperServerFromGguf" in template
```

- [ ] **Step 2: Run the GGUF prefill-helper test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_models_template_contains_whisper_server_gguf_prefill_helpers -v`
Expected: FAIL because the current page does not expose those helper flows yet.

- [ ] **Step 3: Write the failing template test for a GGUF row whisper-server shortcut**

```python
def test_models_template_contains_gguf_use_with_whisper_server_action():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "Use with whisper-server" in template
    assert "prefillWhisperServerFromGguf" in template
```

- [ ] **Step 4: Run the GGUF-row shortcut test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_models_template_contains_gguf_use_with_whisper_server_action -v`
Expected: FAIL because GGUF rows currently have no whisper-server shortcut action.

- [ ] **Step 5: Implement the minimal GGUF selector hydration and row shortcut**

```javascript
let _ggufFiles = [];

function refreshWhisperGgufOptions(){
  const select = document.getElementById('ws-gguf-select');
  if(!select) return;
  const options = _ggufFiles.map(f => `<option value="${esc(f.path)}">${esc(f.filename)}${f.size_gb ? ` (${f.size_gb} GB)` : ''}</option>`);
  select.innerHTML = options.length ? options.join('') : '<option value="">No GGUF files downloaded</option>';
}

function prefillWhisperServerFromGguf(path){
  document.getElementById('ws-model-source').value = 'cached-gguf';
  toggleWhisperModelSource();
  document.getElementById('ws-gguf-select').value = path;
  document.getElementById('ws-model-builder').scrollIntoView({behavior:'smooth', block:'start'});
}
```

```javascript
// inside loadCachedModels() after const gguf = d.gguf||[];
_ggufFiles = gguf;
refreshWhisperGgufOptions();
```

```javascript
// inside GGUF row actions
<button class="btn btn-secondary btn-sm" onclick="prefillWhisperServerFromGguf(${JSON.stringify(f.path)})">Use with whisper-server</button>
```

- [ ] **Step 6: Run the GGUF helper and shortcut tests to verify they pass**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py -k "gguf_prefill_helpers or use_with_whisper_server_action" -v`
Expected: PASS for the new helper and shortcut tests.

- [ ] **Step 7: Commit the GGUF-to-whisper prefill workflow**

```bash
git add tests/test_whisper_server_local_models.py codai/admin/templates/models.html
git commit -m "feat: prefill whisper-server models from gguf cache"
```

## Task 4: Wire the builder submit flow to the resolved model path and run final verification

**Files:**
- Modify: `tests/test_whisper_server_local_models.py`
- Modify: `codai/admin/templates/models.html`
- Modify: `codai/admin/routes.py`

- [ ] **Step 1: Write the failing template test for whisper-server submit payload source fields**

```python
def test_models_template_posts_whisper_server_model_source_fields():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "model_source:" in template
    assert "ws-model-source" in template
    assert "ws-gguf-select" in template
```

- [ ] **Step 2: Run the submit-payload template test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_models_template_posts_whisper_server_model_source_fields -v`
Expected: FAIL because `addWhisperServerModel()` currently always reads the manual text input only.

- [ ] **Step 3: Implement the minimal submit-path resolution logic**

```javascript
function getWhisperServerModelPath(){
  const source = document.getElementById('ws-model-source').value;
  if(source === 'cached-gguf'){
    return document.getElementById('ws-gguf-select').value.trim();
  }
  return document.getElementById('ws-model-path').value.trim();
}

async function addWhisperServerModel(){
  const usedVram = parseFloat(document.getElementById('ws-used-vram').value);
  const payload = {
    model_id: document.getElementById('ws-model-id').value.trim(),
    model_type: 'audio_models',
    backend: 'whisper-server',
    model_source: document.getElementById('ws-model-source').value,
    server_path: document.getElementById('ws-server-path').value.trim(),
    model_path: getWhisperServerModelPath(),
    port: parseInt(document.getElementById('ws-port').value, 10) || 8744,
    gpu_device: parseInt(document.getElementById('ws-gpu-device').value, 10) || 0,
    load_mode: document.getElementById('ws-load-mode').value,
    used_vram_gb: Number.isNaN(usedVram) ? null : usedVram,
  };
  // existing fetch logic unchanged
}
```

- [ ] **Step 4: Run the submit-payload template test to verify it passes**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_models_template_posts_whisper_server_model_source_fields -v`
Expected: PASS.

- [ ] **Step 5: Run the full whisper-server regression file**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py -v`
Expected: PASS for all existing and new whisper-server local-model tests.

- [ ] **Step 6: Inspect the final implementation diff**

Run: `git diff -- codai/admin/templates/models.html codai/admin/routes.py tests/test_whisper_server_local_models.py`
Expected: only the planned GGUF association changes.

- [ ] **Step 7: Commit the completed GGUF association workflow**

```bash
git add tests/test_whisper_server_local_models.py codai/admin/templates/models.html codai/admin/routes.py
git commit -m "feat: link downloaded gguf files to whisper-server models"
```
