# 2026-05-06 Web Admin Polish and Studio Recovery Design

## Summary
Repair the existing web admin dark-theme inconsistencies and recover broken visibility/usability in the overview and studio pages without introducing a new UI framework or route structure. The work stays inside the current admin stack built from shared CSS, Jinja templates, and existing admin/status APIs.

## Goals
- Make admin form controls and scrollable areas consistently match the dark theme.
- Make Overview > Recent Activity show real request history whenever tracked API traffic exists.
- Improve studio split-pane usability for non-chat, non-pipeline modes.
- Surface partial or unavailable capability gaps directly in the result/output areas.
- Rebuild a minimal usable pipeline browser/editor in the existing studio pipeline tab.

## Non-Goals
- Replacing the current admin templating system.
- Large-scale decomposition of `codai/admin/templates/chat.html` into multiple files in this pass.
- Building a full pipeline execution engine beyond what current backend support already exposes.
- Adding unrelated admin features.

## Current Context
- Shared admin theming lives in `codai/admin/static/style.css`.
- The overview page is rendered by `codai/admin/templates/dashboard.html` and reads status data from `/admin/api/status` in `codai/admin/routes.py`.
- Recent request history is sourced from the in-memory request log in `codai/api/log.py`.
- The studio UI, including chat, multimodal tabs, capability metadata, and pipeline tab, is implemented in the large single template `codai/admin/templates/chat.html`.

## Problem Breakdown

### 1. Inconsistent dark styling for selects and other controls
Some native select controls and possibly other field variants still render with a light/white browser-default background instead of the admin dark palette. This is especially visible where local component CSS uses custom classes such as `.fselect` in studio rather than the shared `.form-input` path. Scrollbars also use the browser default rather than a themed style, which stands out on dark pages.

### 2. Overview recent activity always empty
The Overview page currently displays the fallback empty row when `recent_activity` is empty. The backend route already attempts to read activity via `get_recent_activity()` from `codai/api/log.py`, so the bug is likely one of these:
- tracked requests do not include all relevant studio/API endpoints,
- request logging is not being exercised by the UI paths the user expects,
- or the returned payload shape is too narrow to represent the activity users care about.

### 3. Studio split layout too narrow on the control side
In multimodal tabs that use `.gen-wrap`, the left configuration column is fixed at `290px` via `.gen-ctrl`, which is too narrow for prompts, grouped controls, and parameter labels.

### 4. Partial/unavailable capability state not mirrored in output areas
The studio tabs already expose state metadata in buttons and capability cards, but the output panes do not persistently remind the user when the selected function is partial or unavailable, or which parts are missing.

### 5. Pipeline browser/editor usability regression
The pipeline tab no longer visibly exposes a “create new pipeline” action, and selectable/openable pipeline entries are not apparent in the current page. The desired target is not to restore an older hidden implementation, but to rebuild a minimal working browser/editor in the current UI.

## Proposed Approach

### A. Global admin theming polish
Update `codai/admin/static/style.css` to become the canonical source of dark styling for:
- native `select`, `option`, and any admin field variants that still inherit browser defaults,
- scrollbar colors for page-level and nested scroll containers,
- focus/hover states so the darker appearance remains readable and consistent.

For studio-only controls (`.fi`, `.fs`, `.fselect`, scrollable studio panes), add only minimal local alignment rules in `codai/admin/templates/chat.html` if the shared stylesheet cannot safely cover them. Prefer shared rules first.

#### Expected behavior
- Selects render with dark background, readable text, and dark dropdown arrow treatment.
- Textareas/inputs/selects in admin and studio appear visually related.
- Visible scrollbars use a subdued dark track and brighter thumb that remains readable against the page.

### B. Overview recent-activity recovery
Keep the dashboard rendering model intact in `codai/admin/templates/dashboard.html` and fix the data path at the source in `codai/admin/routes.py` and, if needed, `codai/api/log.py`.

#### Data strategy
1. Verify `/admin/api/status` continues returning `recent_activity` as an array of rows containing `time`, `model`, `type`, `status`, and `duration`.
2. Expand tracked request coverage in `codai/api/log.py` if current studio-relevant endpoints are missing from `_TRACKED_PATHS`.
3. Normalize entries in the status route if necessary so the dashboard template always receives a stable shape.
4. Preserve the empty-state row only for the true no-data case.

#### Expected behavior
- After actual tracked requests occur, Overview renders recent rows without requiring template changes beyond minor badge/style cleanup if needed.
- The list remains robust if a field is missing; the status route should normalize or default values before sending them.

### C. Studio split-pane usability improvement
In `codai/admin/templates/chat.html`, increase the left control column width for `.gen-ctrl` from a narrow fixed width to a more usable responsive range.

#### Layout rule
Use a larger default width with min/max constraints so the panel is clearly wider on desktop while still shrinking gracefully on smaller widths. The result/output area should still remain dominant, but no longer at the expense of prompt readability.

#### Expected behavior
- Prompt and options columns are visibly more comfortable in all non-chat/non-pipeline generation tabs.
- The layout remains stable on narrower screens and continues to scroll instead of overflowing horizontally where possible.

### D. Output-area capability warnings
Extend the studio output rendering in `codai/admin/templates/chat.html` so the selected mode’s capability state is echoed in the result area, not only the tabs.

#### Warning model
For any selected function/mode marked as:
- `partial`: show a persistent warning panel in the output region explaining that support is partial and listing the missing parts.
- `unavailable`: show a stronger warning/empty-state block explaining that the selected function is unavailable and listing the missing requirements or unsupported pieces.

#### Data source
Reuse capability metadata already present in the studio model/capability state logic instead of inventing a second source of truth. The output note should be generated from the same state object the tab badges/cards already use.

#### Expected behavior
- Users can understand limitations even after moving focus to the right-hand result area.
- The warning remains visible before and after generation attempts where appropriate.

### E. Minimal working pipeline browser/editor
Rebuild the pipeline tab in `codai/admin/templates/chat.html` as a minimal current-UI feature rather than trying to recover hidden legacy UI.

#### Minimal feature set
- A clearly visible primary action to create a new pipeline.
- A browsable list of existing pipelines rendered as cards/list items.
- A selectable/open state so a chosen pipeline can be inspected and edited.
- A basic editable ordered step list with key parameter inputs.
- Explicit empty-state messaging when there are no pipelines yet.

#### Interaction model
- The left or top portion of the pipeline area should make discovery obvious: create action + pipeline list/empty state.
- Selecting a pipeline should reveal its editor/details area in the existing pipeline panel.
- The UI should emphasize recoverability and visibility over advanced orchestration features.

#### Constraints
- Follow whatever backend/data model already exists for pipelines if present.
- If backend support is partial, still expose the UI with clear empty-state and unsupported-state messaging instead of silently hiding controls.

## File-Level Design

### `codai/admin/static/style.css`
Responsibilities:
- Shared dark form-control styling consistency.
- Shared scrollbar theme for admin pages and reusable containers.
- Any missing generic admin badge/style helpers needed by overview activity states.

### `codai/admin/templates/dashboard.html`
Responsibilities:
- Continue rendering recent activity rows from `recent_activity`.
- Only small presentation adjustments if required for normalized data or badge classes.

### `codai/admin/routes.py`
Responsibilities:
- Normalize `/admin/api/status` output for `recent_activity`.
- Potentially adapt backend activity entries into a stable frontend-friendly structure.

### `codai/api/log.py`
Responsibilities:
- Ensure tracked request coverage reflects the user-visible activity expected from studio/admin usage.
- Keep recent activity ring-buffer behavior simple and bounded.

### `codai/admin/templates/chat.html`
Responsibilities:
- Increase shared generation control-pane width.
- Add output warning blocks for partial/unavailable capabilities.
- Rebuild minimal pipeline browse/create/select/edit presentation.
- Keep new JS helpers focused and local: capability-note rendering, pipeline render/state helpers, and any layout glue.

## Error Handling
- If recent activity cannot be collected, Overview should still degrade to the existing empty state without JS failure.
- If capability metadata is incomplete, output warnings should degrade to a generic partial/unavailable message instead of rendering nothing.
- If no pipelines exist, the pipeline tab should explicitly say so and still present the create action.
- If pipeline backend support is incomplete, the editor area should surface what is unavailable rather than hiding the area.

## Testing Strategy

### Manual/UI verification
- Check Overview, Settings, Models, Tokens, Users, and Studio for dark select/input consistency.
- Check pages with nested scrolling, especially Studio, for themed scrollbar appearance.
- Generate representative requests from studio-relevant modes and verify Overview recent activity populates.
- Verify non-chat/non-pipeline studio tabs show a wider left control panel at common desktop widths.
- Verify partial/unavailable modes show notices in both navigation/capability areas and output areas.
- Verify pipeline tab shows create action, empty state, pipeline list once populated, and editable selected details.

### Automated verification targets
- Existing tests touching studio/admin surfaces should be reviewed and extended where practical.
- Add or update tests around recent-activity payload generation if backend tests already cover `api_status` or request logging.
- Add or update pipeline/studio surface tests if there is existing template/API coverage, especially around rendered pipeline states and capability notices.

## Risks and Mitigations
- `chat.html` is already large, so new logic could worsen maintainability. Mitigation: keep new helpers small and clearly separated by responsibility within the existing file.
- Native select/option styling differs by browser/OS. Mitigation: target a consistent dark baseline without relying on pixel-perfect native menu rendering.
- The user’s notion of “recent activity” may include endpoints not currently tracked. Mitigation: broaden tracked endpoint coverage based on actual studio-visible request flows.
- Pipeline support may be only partially implemented in the backend. Mitigation: build a minimal honest UI that always shows state and available actions, plus missing/unsupported notes.

## Success Criteria
- No obvious white-background form controls remain in the admin/studio UI under normal usage.
- Scrollbars visually match the dark admin theme.
- Overview recent activity shows actual request entries after tracked use.
- Studio control panes are noticeably less cramped in generation tabs.
- Partial/unavailable functionality is clearly disclosed inside output areas.
- The pipeline tab visibly supports creating, listing, selecting, and editing pipelines at a minimal usable level.
