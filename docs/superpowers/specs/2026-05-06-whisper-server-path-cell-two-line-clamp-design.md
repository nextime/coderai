# Whisper-Server Path Cell Two-Line Clamp Design

## Overview

Tighten the configured whisper-server model row on the Local Models page by shrinking the rendered `Model path` cell to a maximum width of `160px` and allowing the path text to occupy up to two visible lines before truncation.

This is a rendering-only adjustment. Stored model data and whisper-server configuration behavior remain unchanged.

## Goals

- Reduce the horizontal space consumed by the configured whisper-server `Model path` column.
- Show more useful path information than a single-line ellipsis would at the same width.
- Preserve access to the full path via hover.
- Keep the change localized to the configured whisper-server row rendering.

## Non-Goals

- Change whisper-server model storage or API payloads.
- Rework the surrounding table layout or column order.
- Introduce a modal, expandable cell, or copy-to-clipboard affordance.
- Alter builder defaults, GGUF selection behavior, or load/unload controls.

## Current State

- `codai/admin/templates/models.html` renders the configured whisper-server `Model path` cell with `max-width:320px`.
- The current style uses single-line truncation via `overflow:hidden`, `text-overflow:ellipsis`, and `white-space:nowrap`.
- The full path is exposed through the cell `title` attribute.
- `tests/test_whisper_server_local_models.py` asserts the current `320px` single-line truncation style.

## Recommended Approach

Replace the current single-line truncation styling on the configured whisper-server `Model path` cell with a fixed-width, two-line clamp:

- set `max-width:160px`
- keep `overflow:hidden`
- remove `white-space:nowrap`
- use a two-line clamp style so the path can wrap onto a second line before truncating
- keep `title` set to the full path value

This keeps the row narrower while exposing more of the path than a single clipped line.

## Rendering Design

### Configured whisper-server path display

The configured whisper-server row in `codai/admin/templates/models.html` should render the `Model path` cell with:

- `max-width:160px`
- a wrapping/two-line layout
- truncation after the second line
- the existing full-path `title` tooltip

A WebKit-compatible line clamp approach is acceptable here because the current page already relies on inline styles and pragmatic browser behavior. The cell should remain readable in supported browsers and degrade to a bounded wrapped block if full clamp behavior is unavailable.

### Hover behavior

The path cell itself remains the hover target. No extra icon or interaction surface is added.

## Error Handling

- If clamp-specific CSS is not supported, the cell should still remain width-bounded and wrapped rather than returning to an unbounded single line.
- The full path remains available through the `title` attribute, so no information is lost.

## Testing Strategy

### Template regression test

Update `tests/test_whisper_server_local_models.py` to assert that the configured whisper-server path cell now includes:

- `max-width:160px`
- the two-line clamp CSS
- the existing `title` attribute with the full path

### Regression coverage boundary

No backend behavior changes are required. Existing whisper-server backend tests remain unchanged.

## Files Likely to Change

- `codai/admin/templates/models.html`
- `tests/test_whisper_server_local_models.py`

## Design Decisions Finalized

- The configured whisper-server `Model path` cell is capped at `160px`.
- The path may occupy up to two visible lines before truncation.
- Full-path hover disclosure remains in place.
- No backend or persisted-data behavior changes are part of this adjustment.
