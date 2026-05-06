# Whisper-Server Builder Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add smart whisper-server builder defaults for `model_id` and `server_path`, and render configured whisper-server `model_path` values with truncation plus full hover disclosure.

**Architecture:** Apply smallest-missing `whisperN` id generation and `whisper-server` binary path detection in both the UI and backend so interactive and API-driven configuration stay consistent. Update the whisper-server list rendering only at presentation time, using a constrained path cell with ellipsis and `title` hover text while keeping the stored `model_path` unchanged.

**Tech Stack:** FastAPI, Jinja2 templates, vanilla JavaScript, Python stdlib (`shutil`, `re`), pytest.

---

## File Structure

- Modify: `codai/admin/templates/models.html`
  - Prefill whisper-server builder defaults on load and after successful add.
  - Add small JS helpers for next-free `whisperN` generation and binary-path fallback.
  - Truncate configured whisper-server `model_path` display with hover tooltip.
- Modify: `codai/admin/routes.py`
  - Apply backend defaults for omitted `model_id` and `server_path`.
  - Reuse configured whisper-server entries to compute the next free `whisperN`.
- Modify: `tests/test_whisper_server_local_models.py`
  - Add backend and template regression tests for default id generation, default server path fallback, and truncated path rendering.

## Task 1: Add backend defaults for whisper-server model id and server path

**Files:**
- Modify: `tests/test_whisper_server_local_models.py`
- Modify: `codai/admin/routes.py`

- [ ] **Step 1: Write the failing backend test for omitted whisper-server model id defaulting to `whisper0`**

```python
def test_model_configure_defaults_missing_whisper_server_model_id_to_whisper0(monkeypatch, tmp_path):
    from codai.admin import routes
    from codai.api.app import app

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app.dependency_overrides[routes.require_admin] = lambda: "admin"

    client = TestClient(app)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "backend": "whisper-server",
            "model_type": "audio_models",
            "model_source": "manual-path",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/base.en.gguf",
            "port": 8744,
            "gpu_device": 0,
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"][0]["id"] == "whisper0"

    app.dependency_overrides.clear()
```

- [ ] **Step 2: Run the missing-model-id test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_model_configure_defaults_missing_whisper_server_model_id_to_whisper0 -v`
Expected: FAIL because `api_model_configure` currently requires `model_id`.

- [ ] **Step 3: Write the failing backend test for smallest-missing-number allocation**

```python
def test_model_configure_defaults_missing_whisper_server_model_id_to_smallest_free_suffix(monkeypatch, tmp_path):
    from codai.admin import routes
    from codai.api.app import app

    cfg = _build_config(tmp_path)
    cfg.models_data["audio_models"] = [
        {"id": "whisper0", "backend": "whisper-server", "server_path": "/usr/local/bin/whisper-server", "model_path": "/models/a.gguf", "port": 8744, "gpu_device": 0},
        {"id": "whisper1", "backend": "whisper-server", "server_path": "/usr/local/bin/whisper-server", "model_path": "/models/b.gguf", "port": 8745, "gpu_device": 0},
        {"id": "whisper3", "backend": "whisper-server", "server_path": "/usr/local/bin/whisper-server", "model_path": "/models/c.gguf", "port": 8746, "gpu_device": 0},
    ]
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app.dependency_overrides[routes.require_admin] = lambda: "admin"

    client = TestClient(app)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "backend": "whisper-server",
            "model_type": "audio_models",
            "model_source": "manual-path",
            "server_path": "/usr/local/bin/whisper-server",
            "model_path": "/models/d.gguf",
            "port": 8747,
            "gpu_device": 0,
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"][-1]["id"] == "whisper2"

    app.dependency_overrides.clear()
```

- [ ] **Step 4: Run the smallest-missing-number test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_model_configure_defaults_missing_whisper_server_model_id_to_smallest_free_suffix -v`
Expected: FAIL because missing `model_id` is not yet auto-generated.

- [ ] **Step 5: Write the failing backend test for omitted server path fallback**

```python
def test_model_configure_defaults_missing_whisper_server_server_path(monkeypatch, tmp_path):
    from codai.admin import routes
    from codai.api.app import app

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    monkeypatch.setattr(routes.shutil, "which", lambda name: None)
    app.dependency_overrides[routes.require_admin] = lambda: "admin"

    client = TestClient(app)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "backend": "whisper-server",
            "model_type": "audio_models",
            "model_source": "manual-path",
            "model_path": "/models/base.en.gguf",
            "port": 8744,
            "gpu_device": 0,
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"][0]["server_path"] == "/usr/local/bin/whisper-server"

    app.dependency_overrides.clear()
```

- [ ] **Step 6: Run the missing-server-path test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_model_configure_defaults_missing_whisper_server_server_path -v`
Expected: FAIL because `api_model_configure` currently requires `server_path`.

- [ ] **Step 7: Implement minimal backend helper logic for defaults**

```python
# codai/admin/routes.py imports
import re
import shutil

_WHISPER_ID_RE = re.compile(r"^whisper(\d+)$")


def _next_whisper_server_id(audio_models):
    used = set()
    for entry in audio_models or []:
        if isinstance(entry, dict) and entry.get("backend") == "whisper-server":
            match = _WHISPER_ID_RE.match(entry.get("id", ""))
            if match:
                used.add(int(match.group(1)))
    n = 0
    while n in used:
        n += 1
    return f"whisper{n}"


def _default_whisper_server_path():
    return shutil.which("whisper-server") or "/usr/local/bin/whisper-server"

# inside api_model_configure whisper-server branch
model_id = (data.get("model_id") or "").strip() or _next_whisper_server_id(config_manager.models_data.get("audio_models", []))
server_path = (data.get("server_path") or "").strip() or _default_whisper_server_path()
```

- [ ] **Step 8: Run the focused backend default tests**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py -k "missing_whisper_server_model_id_to_whisper0 or smallest_free_suffix or missing_whisper_server_server_path" -v`
Expected: PASS for all three new backend default tests.

- [ ] **Step 9: Commit the backend whisper-server default generation**

```bash
git add tests/test_whisper_server_local_models.py codai/admin/routes.py
git commit -m "feat: default whisper-server builder fields"
```

## Task 2: Prefill whisper-server builder defaults in the Models page UI

**Files:**
- Modify: `tests/test_whisper_server_local_models.py`
- Modify: `codai/admin/templates/models.html`

- [ ] **Step 1: Write the failing template test for next-free `whisperN` helper presence**

```python
def test_models_template_defines_next_whisper_server_id_helper():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "function nextWhisperServerModelId()" in template
    assert "const used = new Set();" in template
    assert "return `whisper${n}`;" in template
```

- [ ] **Step 2: Run the next-id helper test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_models_template_defines_next_whisper_server_id_helper -v`
Expected: FAIL because the helper does not yet exist.

- [ ] **Step 3: Write the failing template test for whisper-server path detection/fallback logic**

```python
def test_models_template_defines_whisper_server_path_default_logic():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "function defaultWhisperServerPath()" in template
    assert "'/usr/local/bin/whisper-server'" in template
    assert "function resetWhisperServerBuilderDefaults()" in template
```

- [ ] **Step 4: Run the path-default helper test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_models_template_defines_whisper_server_path_default_logic -v`
Expected: FAIL because the helper/reset logic does not yet exist.

- [ ] **Step 5: Implement minimal UI default helpers**

```javascript
function nextWhisperServerModelId(){
  const used = new Set();
  _localModels.forEach(m => {
    if(m.cacheType !== 'whisper-server') return;
    const match = /^whisper(\d+)$/.exec(m.label || '');
    if(match) used.add(parseInt(match[1], 10));
  });
  let n = 0;
  while(used.has(n)) n += 1;
  return `whisper${n}`;
}

function defaultWhisperServerPath(){
  return '/usr/local/bin/whisper-server';
}

function resetWhisperServerBuilderDefaults(){
  const modelId = document.getElementById('ws-model-id');
  const serverPath = document.getElementById('ws-server-path');
  if(modelId && !modelId.value.trim()) modelId.value = nextWhisperServerModelId();
  if(serverPath && !serverPath.value.trim()) serverPath.value = defaultWhisperServerPath();
}
```

- [ ] **Step 6: Call the builder default reset on page refresh and successful add**

```javascript
// at end of loadCachedModels()
resetWhisperServerBuilderDefaults();

// after successful addWhisperServerModel()
document.getElementById('ws-model-id').value = '';
document.getElementById('ws-server-path').value = '';
refreshLocal();
```

- [ ] **Step 7: Run the UI default-helper tests**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py -k "next_whisper_server_id_helper or path_default_logic" -v`
Expected: PASS.

- [ ] **Step 8: Commit the whisper-server builder UI defaults**

```bash
git add tests/test_whisper_server_local_models.py codai/admin/templates/models.html
git commit -m "feat: prefill whisper-server builder defaults"
```

## Task 3: Truncate configured whisper-server model paths in the Local Models table

**Files:**
- Modify: `tests/test_whisper_server_local_models.py`
- Modify: `codai/admin/templates/models.html`

- [ ] **Step 1: Write the failing template test for truncated whisper-server path rendering**

```python
def test_models_template_truncates_whisper_server_model_path_cells():
    template = Path("codai/admin/templates/models.html").read_text()

    assert 'text-overflow:ellipsis' in template
    assert 'title="${esc(m.model_path || \'—\')}"' in template
    assert 'max-width:' in template
```

- [ ] **Step 2: Run the truncation test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_models_template_truncates_whisper_server_model_path_cells -v`
Expected: FAIL because the current whisper-server path cell is unconstrained and lacks hover title.

- [ ] **Step 3: Implement minimal truncated path rendering in whisper-server rows**

```javascript
<td style="padding:.4rem .25rem;font-size:11px;color:var(--text-2);max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(m.model_path || '—')}">${esc(m.model_path || '—')}</td>
```

- [ ] **Step 4: Run the truncation template test to verify it passes**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_models_template_truncates_whisper_server_model_path_cells -v`
Expected: PASS.

- [ ] **Step 5: Commit the whisper-server path truncation UI**

```bash
git add tests/test_whisper_server_local_models.py codai/admin/templates/models.html
git commit -m "fix: constrain whisper-server path display"
```

## Task 4: Add detection-oriented UI coverage and run final verification

**Files:**
- Modify: `tests/test_whisper_server_local_models.py`
- Modify: `codai/admin/templates/models.html`
- Modify: `codai/admin/routes.py`

- [ ] **Step 1: Write the failing regression test that manual overrides remain allowed**

```python
def test_model_configure_preserves_explicit_whisper_server_model_id_and_server_path(monkeypatch, tmp_path):
    from codai.admin import routes
    from codai.api.app import app

    cfg = _build_config(tmp_path)
    monkeypatch.setattr(routes, "config_manager", cfg, raising=False)
    app.dependency_overrides[routes.require_admin] = lambda: "admin"

    client = TestClient(app)
    response = client.post(
        "/admin/api/model-configure",
        json={
            "model_id": "custom-whisper",
            "backend": "whisper-server",
            "model_type": "audio_models",
            "model_source": "manual-path",
            "server_path": "/opt/whisper-server/bin/whisper-server",
            "model_path": "/models/custom.gguf",
            "port": 8744,
            "gpu_device": 0,
        },
    )

    assert response.status_code == 200
    assert cfg.models_data["audio_models"][0]["id"] == "custom-whisper"
    assert cfg.models_data["audio_models"][0]["server_path"] == "/opt/whisper-server/bin/whisper-server"

    app.dependency_overrides.clear()
```

- [ ] **Step 2: Run the manual-override regression test to verify it passes or fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_model_configure_preserves_explicit_whisper_server_model_id_and_server_path -v`
Expected: PASS if defaults did not override explicit values; FAIL only if regression was introduced and then fix it.

- [ ] **Step 3: Write the failing template test for reset-after-add default refill hook**

```python
def test_models_template_resets_whisper_server_builder_defaults_after_add():
    template = Path("codai/admin/templates/models.html").read_text()

    assert "resetWhisperServerBuilderDefaults();" in template
    assert "document.getElementById('ws-model-id').value = '';" in template
    assert "document.getElementById('ws-server-path').value = '';" in template
```

- [ ] **Step 4: Run the reset-after-add test to verify it fails correctly**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py::test_models_template_resets_whisper_server_builder_defaults_after_add -v`
Expected: FAIL until the reset/default refill flow exists.

- [ ] **Step 5: If desired, improve UI path detection hint without expanding scope**

```javascript
// keep defaultWhisperServerPath() minimal for this task
// if no runtime path discovery is available in template JS, rely on backend + fallback and use placeholder/value reset
```

- [ ] **Step 6: Run the full whisper-server regression file**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_whisper_server_local_models.py -v`
Expected: PASS for all whisper-server tests, including the new defaults and truncation coverage.

- [ ] **Step 7: Inspect the final diff**

Run: `git diff -- codai/admin/templates/models.html codai/admin/routes.py tests/test_whisper_server_local_models.py`
Expected: only the planned builder-default and path-rendering changes.

- [ ] **Step 8: Commit the final defaults and rendering workflow**

```bash
git add tests/test_whisper_server_local_models.py codai/admin/templates/models.html codai/admin/routes.py
git commit -m "feat: add whisper-server builder defaults"
```
