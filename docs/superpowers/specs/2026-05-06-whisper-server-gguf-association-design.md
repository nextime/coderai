# Whisper-Server GGUF Association Design

## Overview

Extend the Local Models page so a whisper-server simulated model can use either:

- a manually entered GGUF filesystem path, or
- a GGUF file that was downloaded through the Models interface from HuggingFace

This keeps whisper-server model creation centered on `/admin/models` while making the common workflow direct: download a GGUF model, then associate that local cached file with a whisper-server instance.

## Goals

- Allow whisper-server model configuration to choose a downloaded local GGUF file.
- Preserve manual `model_path` entry for advanced or pre-existing local setups.
- Keep the existing persisted whisper-server model format in `models.json`.
- Reuse the current Models page cache inventory instead of introducing a second local-file discovery system.

## Non-Goals

- Introduce a new persistence schema for whisper-server models.
- Automatically start whisper-server when a GGUF file is downloaded.
- Copy, move, or duplicate downloaded GGUF files when associating them with whisper-server.
- Build a generic asset-linking system for all model backends.

## Current State

- `codai/admin/templates/models.html` already exposes a `Whisper-server simulated models` form with a manual `model_path` field.
- The same page already loads cached HuggingFace repos and local GGUF files through `/admin/api/cached-models`.
- GGUF files already appear as actionable local assets in the Local Models tab.
- `codai/admin/routes.py` persists whisper-server models through `POST /admin/api/model-configure`, storing `model_path` as a plain string.
- Runtime startup and on-request use only the resolved `model_path`; they do not care whether that path came from a manual entry or a downloaded local GGUF.

## Recommended Approach

Keep one whisper-server builder and add an explicit model-source choice inside it:

- `Downloaded GGUF`
- `Manual path`

The selected source controls how `model_path` is populated before submit. The saved backend entry remains unchanged.

This preserves a single mental model for whisper-server configuration while supporting both normal and advanced workflows.

## UI Design

### Whisper-Server Builder

Update the existing whisper-server builder card on `codai/admin/templates/models.html`.

Add a `Model source` control with two mutually exclusive modes:

1. `Downloaded GGUF`
2. `Manual path`

#### Downloaded GGUF mode

When selected:

- hide or disable the free-text `model_path` field
- show a dropdown populated from the current local GGUF cache list already loaded in `loadCachedModels()`
- store the selected GGUF file’s cached local path as the effective `model_path`
- optionally show the filename and size next to the selector for confidence

#### Manual path mode

When selected:

- show the existing free-text path field
- no GGUF dropdown is required
- submit the manual path as `model_path`

#### Shared fields

The following fields remain unchanged regardless of model source:

- `model_id`
- `server_path`
- `port`
- `gpu_device`
- `load_mode`
- `used_vram_gb`

### Optional GGUF Row Shortcut

A useful secondary enhancement is a GGUF table-row action such as `Use with whisper-server`.

This shortcut should:

- preselect `Downloaded GGUF` mode
- prefill the GGUF dropdown with the chosen file
- focus or scroll to the whisper-server builder
- not create the model immediately

This shortcut is optional and should use the same underlying builder instead of creating a parallel whisper-server configuration flow.

## Backend and Data Design

### Persistence

No new schema is needed.

Persist whisper-server models exactly as today under `audio_models` with:

- `backend: "whisper-server"`
- resolved `model_path`
- existing whisper-server metadata fields

Example:

```json
{
  "id": "whisper-vulkan-base",
  "backend": "whisper-server",
  "server_path": "/usr/local/bin/whisper-server",
  "model_path": "/storage/coderai/models/whisper/ggml-base.bin",
  "port": 8744,
  "gpu_device": 0,
  "load_mode": "on-request"
}
```

The backend should not persist whether the source was `Downloaded GGUF` or `Manual path`; that distinction is UI-only.

### API Behavior

Continue using `POST /admin/api/model-configure` as the single persistence endpoint.

Expected behavior:

- if `Downloaded GGUF` is chosen, the UI posts the selected cached file path as `model_path`
- if `Manual path` is chosen, the UI posts the typed path as `model_path`
- backend validation remains centered on the final resolved `model_path`

No runtime changes are required for load/start behavior, because runtime already only needs a concrete `model_path`.

## Validation Rules

- require exactly one source mode at a time
- `Downloaded GGUF` mode requires a selected GGUF file
- `Manual path` mode requires a non-empty path
- keep existing validation for `model_id`, `server_path`, `port`, and `gpu_device`
- reject malformed or missing `model_path` regardless of source mode

## Error Handling

- If a downloaded GGUF file is selected and later deleted from cache, the whisper-server model should still render in the UI but fail clearly during load/start with a file-not-found-style error.
- The UI may optionally validate that the selected cached path still exists before submit, but runtime validation is still required.
- Associating a downloaded GGUF file must not duplicate the file or alter its cache location.
- Removing a GGUF file from cache later does not automatically delete the whisper-server configuration entry; it becomes a broken reference until edited or removed.

## Data Flow

### Downloaded GGUF workflow

1. User downloads a GGUF file from the Models interface.
2. The file appears in the Local Models GGUF list.
3. User opens the whisper-server builder.
4. User selects `Downloaded GGUF`.
5. User picks the downloaded GGUF file from the dropdown.
6. UI submits the chosen cached path as `model_path`.
7. Backend saves the whisper-server model entry as a normal `audio_models` config.

### Manual path workflow

1. User opens the whisper-server builder.
2. User selects `Manual path`.
3. User enters a filesystem path.
4. UI submits the typed path as `model_path`.
5. Backend saves the whisper-server model entry with the same structure used by downloaded GGUF mode.

## Testing Strategy

### Backend tests

- whisper-server config creation succeeds when `model_path` comes from a GGUF cached path
- manual-path whisper-server config creation still succeeds unchanged
- missing selected GGUF path is rejected
- empty manual path is rejected

### UI/template tests

- whisper-server builder renders both `Downloaded GGUF` and `Manual path` source controls
- builder still includes the core whisper-server configuration fields
- current manual-path UI remains available

### Integration/regression tests

- cached GGUF entries can be surfaced to the whisper-server builder without changing existing local-model listing behavior
- optional GGUF row shortcut, if implemented, pre-fills the builder instead of creating a second flow

## Files Likely to Change

- `codai/admin/templates/models.html`
- `codai/admin/routes.py`
- `tests/test_whisper_server_local_models.py`

## Design Decisions Finalized

- Both downloaded-GGUF selection and manual path entry are supported.
- Downloaded GGUF is the primary guided workflow.
- Manual path remains fully supported.
- The persisted whisper-server model schema remains unchanged.
- The source mode is UI-only and is not stored in `models.json`.
