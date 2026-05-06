# Whisper-Server Builder Defaults and Path Rendering Design

## Overview

Improve the whisper-server builder on the Local Models page so it is easier to use without manual boilerplate and does not break layout when configured model paths are long.

The builder should:

- auto-suggest a `whisperN` model id when the user leaves the field unspecified
- auto-detect the `whisper-server` binary path from `PATH`, falling back to `/usr/local/bin/whisper-server`
- render long configured `model_path` values in the Local Models table using truncation with full-path hover disclosure

These changes extend the existing whisper-server Local Models workflow without changing the persisted model schema.

## Goals

- Reduce friction when adding whisper-server models.
- Keep default generation consistent between UI and backend.
- Prevent long `model_path` values from expanding the model table and damaging page layout.
- Preserve full `model_path` visibility via hover tooltip.

## Non-Goals

- Change the persisted `audio_models` whisper-server schema.
- Introduce a separate naming scheme beyond `whisperN`.
- Require binary existence validation before config creation.
- Add a new details modal solely for path display.

## Current State

- `codai/admin/templates/models.html` requires manual `model_id` entry in the whisper-server builder.
- `codai/admin/templates/models.html` requires manual `server_path` entry in the whisper-server builder.
- `codai/admin/routes.py` currently requires both `model_id` and `server_path` for whisper-server config creation.
- configured whisper-server rows currently render the full `model_path` inline in the table, which can make the row too wide.

## Recommended Approach

Apply defaults in both the UI and backend:

- UI pre-fills `model_id` with the next available `whisperN`
- UI pre-fills `server_path` with detected `whisper-server` or `/usr/local/bin/whisper-server`
- backend applies the same defaults if those fields are omitted
- configured whisper-server rows keep the full stored path but render it in a constrained, truncated cell with `title` hover text

This keeps the page helpful for interactive users while preserving correct behavior for non-UI callers.

## Defaults Design

### Model ID default

Use the pattern `whisperN` where `N` is the smallest non-negative integer not already used by configured whisper-server models.

Examples:

- no configured whisper-server models → `whisper0`
- `whisper0` exists → `whisper1`
- `whisper0`, `whisper1`, `whisper3` exist → `whisper2`

Only ids matching the exact `whisperN` pattern participate in this automatic sequence. Custom ids remain valid but do not block unrelated numbering except where they exactly match that pattern.

The UI should prefill this value when the page loads or refreshes. The backend should also apply this default if `model_id` is omitted or blank in the request.

### Binary path default

Resolve the whisper-server binary path using this order:

1. detect `whisper-server` from the current runtime `PATH`
2. if detection fails, use `/usr/local/bin/whisper-server`

The field remains editable. Failure to detect or validate the binary at prefill time should not block creation; runtime load/start remains responsible for surfacing executable-not-found errors.

### UI/backend symmetry

The UI should display the same defaults that the backend would apply.

This means:

- users can see the generated values before submit
- API callers omitting the same fields still get consistent behavior
- duplicate-id validation remains the final protection in concurrent cases

## Rendering Design

### Configured whisper-server path display

Keep the full stored `model_path` value unchanged.

Only the rendering of the `Model path` table column should change:

- apply a maximum visual width to the path cell
- use `overflow:hidden`
- use `text-overflow:ellipsis`
- use `white-space:nowrap`
- set `title` to the full path so hover reveals the complete string

This improves layout stability without reducing information availability.

### Hover behavior

The hover target should be the path cell text itself, not a separate icon. This keeps the interaction minimal and aligned with the existing table UI.

## Refresh and Post-Create Behavior

After a whisper-server model is added successfully:

- refresh the configured local models list as today
- recompute the next default `whisperN`
- reset the builder to the new next-free id
- reset `server_path` to the detected/fallback default if the form is cleared
- keep the chosen source mode behavior aligned with the current workflow

## Error Handling

- If the backend-generated `whisperN` collides because another whisper-server model was created concurrently, existing duplicate-id validation should reject the request cleanly.
- If the detected or fallback `server_path` is invalid, creation may still succeed, but later load/start should fail with a clear error.
- If path truncation is applied, hover must still expose the full path, so no information is lost.

## Testing Strategy

### Backend tests

- omitted `model_id` defaults to the next free `whisperN`
- omitted `server_path` defaults to detected path or `/usr/local/bin/whisper-server`
- duplicate-id validation still rejects collisions

### UI/template tests

- whisper-server builder includes logic for next-free `whisperN` defaults
- whisper-server builder includes binary-path detection/fallback logic
- configured whisper-server `model_path` cell renders truncated with hover title support

### Regression tests

- existing manual overrides for `model_id` and `server_path` still work
- model list remains visually bounded for long paths without changing stored data

## Files Likely to Change

- `codai/admin/templates/models.html`
- `codai/admin/routes.py`
- `tests/test_whisper_server_local_models.py`

## Design Decisions Finalized

- Default model ids use `whisperN` with smallest-missing-number allocation.
- Default `server_path` comes from `PATH` detection or `/usr/local/bin/whisper-server` fallback.
- Both defaults are applied in UI and backend.
- Long `model_path` values are truncated visually but fully available on hover.
- Persisted whisper-server schema remains unchanged.
