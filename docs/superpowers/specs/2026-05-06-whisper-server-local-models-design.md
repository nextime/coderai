# Whisper-Server Local Model Integration Design

## Overview

Integrate `whisper-server` into the Local Models page as a first-class audio model configuration flow. Instead of maintaining a separate settings-only control panel and custom lifecycle semantics, whisper-server instances become normal persisted `audio_models` entries in `models.json` with `backend: "whisper-server"`. Each saved instance represents a simulated transcription model backed by its own whisper-server subprocess configuration.

## Goals

- Let admins add whisper-server-backed audio models directly from `/admin/models`.
- Make each whisper-server instance appear in Local Models as a configurable local model.
- Remove the old whisper-server configuration UI from `/admin/settings`.
- Make load/unload and on-request behavior match the rest of the model system.
- Drop backward compatibility for the old single-instance settings workflow and custom admin start/stop flow.

## Non-Goals

- Preserve the old `/admin/settings` whisper-server form.
- Preserve the legacy single-instance `whisper_server` fallback behavior.
- Generalize this work into a backend-agnostic custom model editor for all model types.

## Current State

- `codai/main.py` already registers `audio_models` entries with `backend: "whisper-server"` via `multi_model_manager.register_whisper_server(...)`.
- `codai/api/transcriptions.py` already supports on-demand startup for configured whisper-server model ids.
- `codai/admin/templates/models.html` only shows a separate status card for whisper-server, not real configurable model entries.
- `codai/admin/templates/settings.html` still contains a dedicated single-instance whisper-server configuration and manual start/stop controls.
- `codai/admin/routes.py` exposes custom whisper-server start/stop endpoints rather than routing through the generic model load/unload flow.

## Proposed Architecture

### Source of Truth

Persist whisper-server simulated models only in `models.json` under `audio_models`. A whisper-server entry is a dictionary with the normal model metadata plus backend-specific fields:

```json
{
  "id": "whisper-vulkan-base",
  "backend": "whisper-server",
  "server_path": "/usr/local/bin/whisper-server",
  "model_path": "/models/ggml-base.bin",
  "port": 8744,
  "gpu_device": 0,
  "load_mode": "on-request",
  "used_vram_gb": 1.8
}
```

Use `id` as the canonical identifier for whisper-server models. Do not rely on `path` for these entries because they are simulated models rather than cached artifacts.

### Registration and Runtime State

At startup, `codai/main.py` continues to register each whisper-server audio model from `audio_models`. `MultiModelManager` should treat those registrations as normal audio models, with runtime keys shaped as `audio:<id>`.

Runtime state lives in the existing `multi_model_manager.models` map and `whisper_servers` registry:

- `whisper_servers[model_id]` stores the manager and subprocess config.
- `models["audio:<model_id>"]` exists only when the subprocess is loaded/running.
- `request_model(..., model_type="audio")` remains the gate for on-request startup.
- `cleanup()` / unload stops the subprocess exactly like unloading any other model object.

The old single-instance `multi_model_manager.whisper_server` reference and related fallback behavior should be removed instead of maintained.

### Admin Lifecycle Unification

The generic admin lifecycle endpoints become the only control plane:

- `POST /admin/api/model-load` loads a whisper-server audio model by invoking the same request-model resolution flow as other models, then starting the subprocess when necessary.
- `POST /admin/api/model-unload` unloads a whisper-server audio model by removing `audio:<id>` from `multi_model_manager.models` and calling `WhisperServerManager.cleanup()`.
- `GET /admin/api/model-loaded-status` reports whisper-server loaded state through the same loaded-key list used by the Local Models page.

Dedicated `whisper-server/start` and `whisper-server/stop` endpoints should be removed, and the UI should stop calling them.

## UI Design

### Local Models Page

Add a new card on `codai/admin/templates/models.html` in the Local Models tab:

- Title: `Whisper-server simulated models`
- Description: explains these entries simulate transcription models for backends where `faster-whisper` is not suitable
- Visibility: show only when whisper-server is available enough to configure, using either detected binary availability or an existing configured whisper-server model entry

The card contains a compact form with:

- `Model ID`
- `whisper-server binary path`
- `Whisper model path`
- `Port`
- `GPU device index`
- `Load mode` (`load` or `on-request`)
- optional `Used VRAM (GB)`

Primary action: `Add model`

Submitting the form persists a new `audio_models` entry with `backend: "whisper-server"`, then refreshes the local model list. If an entry with the same `id` already exists, the API should reject it with a validation error rather than silently overwriting.

### Local Model Listing

Saved whisper-server entries should appear in Local Models alongside other configured models, specifically within the HuggingFace/local model rendering path as configured audio models rather than in a standalone whisper status box.

Each whisper-server row should display:

- model id
- backend badge showing `whisper-server`
- model path
- port and GPU device
- configured load mode
- loaded/running status inferred from loaded keys
- standard actions: `Load now`, `Unload`, `Configure`, `Remove`

`Configure` should reuse the existing model-configure modal where practical, but whisper-server entries need audio/backend-specific fields exposed. If reusing the generic modal would be too awkward, the whisper-server card can support edit-in-place or a dedicated edit modal, but the interaction must remain in the Models page.

### Settings Page

Remove the dedicated whisper-server section from `codai/admin/templates/settings.html`. The Settings API should stop returning and persisting whisper-specific admin form values for UI purposes. If the underlying `config.whisper` object remains temporarily in code, it should no longer be surfaced or described as a supported configuration path.

## Backend/API Design

### Persistence API

Extend or supplement `POST /admin/api/model-configure` so it can persist whisper-server audio model definitions using `id`-based entries instead of `path`-based cache entries.

Expected request shape for whisper-server model creation:

```json
{
  "model_id": "whisper-vulkan-base",
  "model_type": "audio_models",
  "backend": "whisper-server",
  "server_path": "/usr/local/bin/whisper-server",
  "model_path": "/models/ggml-base.bin",
  "port": 8744,
  "gpu_device": 0,
  "load_mode": "on-request",
  "used_vram_gb": 1.8
}
```

Validation rules:

- `model_id` required
- `backend` must equal `whisper-server`
- `server_path` required
- `port` must be integer in valid port range
- `gpu_device` must be integer >= 0
- duplicate `model_id` in `audio_models` rejected

The server should write an entry keyed by `id`, not `path`.

### Model Listing Payload

`multi_model_manager.list_models()` and/or any local-model listing payload used by the Models page should expose enough metadata for whisper-server rows:

- `backend`
- `load_mode`
- `server_path`
- `model_path`
- `port`
- `gpu_device`
- whether the model is configured in `models.json`

This metadata should come from `config_manager.models_data` for whisper-server entries and from runtime loaded keys for loaded state.

### Generic Load/Unload Support

`POST /admin/api/model-load` must detect whisper-server configured audio models and start them through `WhisperServerManager.start(...)` instead of falling through to Python audio backend loading.

`POST /admin/api/model-unload` must handle `WhisperServerManager` instances using the same generic key matching path, and should stop relying on separate whisper-only endpoints.

`codai/api/transcriptions.py` should continue to use `request_model()` and on-demand startup, but any single-instance fallback logic should be removed so only explicitly configured whisper-server model ids are valid.

## Data Flow

### Add Model

1. Admin fills whisper-server form on `/admin/models`.
2. UI posts model definition to model configuration endpoint.
3. Backend validates uniqueness and required whisper-server fields.
4. Backend appends the entry to `models.json` `audio_models`.
5. UI refreshes local models and loaded status.
6. The new model appears as a configured local audio model.

### Manual Load

1. Admin clicks `Load now` on a whisper-server model row.
2. UI posts `{path: <model-id>}` or a renamed generic identifier payload to `/admin/api/model-load`.
3. Backend resolves the model as an audio model with `backend: "whisper-server"`.
4. Backend starts the subprocess via `WhisperServerManager.start(...)`.
5. Backend stores the running manager under `audio:<model-id>` in `multi_model_manager.models`.
6. UI refreshes and shows the model as loaded.

### On-Request Load

1. Client calls `/v1/audio/transcriptions` with `model=<model-id>`.
2. `request_model(..., model_type="audio")` resolves the configured whisper-server model.
3. If not already running, transcription path starts the subprocess.
4. Audio is sent to whisper-server and the formatted transcription response is returned.

### Unload

1. Admin clicks `Unload`.
2. UI posts to `/admin/api/model-unload`.
3. Backend removes `audio:<model-id>` from loaded models and calls `cleanup()`.
4. Subprocess stops and loaded state disappears from the UI.

## Error Handling

- Missing `server_path` or `model_id` returns `400` with explicit field-specific detail.
- Duplicate whisper-server model id returns `409` or `400` with a clear duplicate message.
- Starting whisper-server failure returns `500` from model-load with the startup failure detail.
- If the binary path does not exist, the model may still be persisted, but manual load and on-request load should fail with a clear error. Prefer surfacing a warning on create if the path is missing.
- On page load, configured whisper-server entries should still render even if the binary is currently unavailable, so broken configs remain editable/removable.

## Testing Strategy

### Backend Tests

- Persisting a whisper-server model writes an `audio_models` dict entry with `id` and backend-specific fields.
- Duplicate `model_id` is rejected.
- `model-load` starts a whisper-server configured audio model and marks `audio:<id>` as loaded.
- `model-unload` stops a loaded whisper-server model and removes its loaded key.
- transcription on-request startup works only for configured whisper-server model ids.
- old whisper-server start/stop endpoints are absent.

### Frontend Tests / Verification

- Models page shows the new whisper-server add-model card.
- Settings page no longer shows whisper-server controls.
- After adding a whisper-server model, it appears in Local Models without page navigation.
- Load/unload buttons update row state exactly like other configured models.

## Files Expected to Change

- `codai/admin/templates/models.html`
- `codai/admin/templates/settings.html`
- `codai/admin/routes.py`
- `codai/models/manager.py`
- `codai/api/transcriptions.py`
- `codai/main.py`
- `codai/config.py` if removing persisted whisper UI settings from `config.json`

## Open Decisions Resolved

- The Local Models page is the only supported admin UI for whisper-server model creation.
- No backward compatibility is maintained for the old settings-based whisper-server workflow.
- Whisper-server lifecycle is unified with generic model load/unload and on-request semantics.
