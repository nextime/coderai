# Web Admin Polish and Studio Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore dark-theme consistency and fix the broken admin/studio usability regressions for recent activity, split-pane sizing, capability warnings, and pipelines.

**Architecture:** Keep the existing admin stack intact by making focused changes in the shared admin stylesheet, the overview status/data path, and the current studio template. Fix data issues at the backend source where possible, and keep studio UI additions inside small helper functions within the existing template rather than introducing a new rendering layer.

**Tech Stack:** FastAPI, Jinja templates, vanilla JavaScript, shared CSS, Python request logging

---

## File Structure

- Modify: `codai/admin/static/style.css`
  - Shared dark-theme form control rules and scrollbar styling for admin surfaces.
- Modify: `codai/api/log.py`
  - Recent-activity tracking coverage and stable log entry shape.
- Modify: `codai/admin/routes.py`
  - Normalize `/admin/api/status` recent-activity payload.
- Modify: `codai/admin/templates/dashboard.html`
  - Keep overview rendering aligned with normalized activity rows and valid badge classes.
- Modify: `codai/admin/templates/chat.html`
  - Wider generation control panel, output capability warnings, and rebuilt minimal pipeline browser/editor.
- Modify: `tests/test_studio_composed_surfaces.py`
  - Extend existing studio/admin surface assertions if applicable.
- Create or modify: backend/admin tests near existing route/log coverage once exact locations are confirmed during implementation.

## Task 1: Dark theme controls and scrollbar styling

**Files:**
- Modify: `codai/admin/static/style.css:1-253`
- Test: existing admin UI/manual verification surfaces

- [ ] **Step 1: Write the failing test or assertion target**

If there is an existing HTML/CSS surface test file for admin templates, add assertions that rendered studio/admin controls include the dark-theme classes/select markup you will rely on. If no CSS/template assertion test exists, record this as a manual-verification-only task and do not invent brittle CSS snapshot coverage.

Expected target checks:
```text
- select.form-input uses dark shared styling path
- studio .fselect/.fi/.fs remain dark-themed
- scrollbar styling declarations exist in shared stylesheet
```

- [ ] **Step 2: Run the relevant test target or confirm no automated CSS test exists**

Run one of the existing focused test commands if a matching test file already exists, for example:
```bash
pytest tests/test_studio_composed_surfaces.py -q
```
Expected: either PASS on unrelated existing assertions or confirmation that no direct CSS coverage exists yet.

- [ ] **Step 3: Implement the shared stylesheet updates**

Update `codai/admin/static/style.css` to:
```css
html {
  scroll-behavior: smooth;
  scrollbar-color: var(--border-2) var(--nav);
}

body,
.table-wrap,
.modal-box,
.chat-messages,
.studio .model-list,
.studio .chat-msgs,
.studio .gen-ctrl,
.studio .gen-out,
.studio .pipe-panel {
  scrollbar-width: thin;
  scrollbar-color: var(--border-2) transparent;
}

*::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

*::-webkit-scrollbar-track {
  background: transparent;
}

*::-webkit-scrollbar-thumb {
  background: var(--border-2);
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: padding-box;
}

*::-webkit-scrollbar-thumb:hover {
  background: #363b4f;
  border: 2px solid transparent;
  background-clip: padding-box;
}

button,
input,
select,
textarea {
  font-family: inherit;
  color-scheme: dark;
}

.form-input,
select.form-input,
textarea.form-input {
  background: var(--raised);
  color: var(--text);
}

select.form-input option,
select.form-input optgroup {
  background: var(--raised);
  color: var(--text);
}
```
Then mirror the same color treatment in the studio-local classes inside `chat.html` only if the shared stylesheet does not reach `.fi`, `.fs`, and `.fselect`.

- [ ] **Step 4: Run the focused verification**

Run:
```bash
pytest tests/test_studio_composed_surfaces.py -q
```
Expected: PASS if the file exists and remains green.

- [ ] **Step 5: Commit**

```bash
git add codai/admin/static/style.css codai/admin/templates/chat.html tests/test_studio_composed_surfaces.py
git commit -m "fix: align admin controls with dark theme"
```

## Task 2: Restore recent activity data in overview

**Files:**
- Modify: `codai/api/log.py:21-90`
- Modify: `codai/admin/routes.py:252-394`
- Modify: `codai/admin/templates/dashboard.html:55-142`
- Test: route/log tests in existing backend test suite

- [ ] **Step 1: Write the failing backend test**

Add or update a focused test that exercises the status payload after tracked activity is recorded. The test should create one or more log entries through the request logging path or by seeding the in-memory activity buffer, then assert `/admin/api/status` returns normalized rows.

Target shape:
```python
def test_admin_status_includes_recent_activity(client, admin_auth_headers):
    from codai.api import log as api_log
    api_log._activity.clear()
    api_log._activity.appendleft({
        "time": 1715000000,
        "model": "demo-model",
        "type": "chat",
        "status": 200,
        "duration": 1.23,
    })

    response = client.get("/admin/api/status", headers=admin_auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["recent_activity"][0]["model"] == "demo-model"
    assert payload["recent_activity"][0]["type"] == "chat"
    assert payload["recent_activity"][0]["status"] == 200
```
Use the project’s existing auth/client fixtures and exact test file location once discovered.

- [ ] **Step 2: Run the focused backend test to verify it fails**

Run the exact test node you added, for example:
```bash
pytest tests/<path-to-admin-or-routes-test>.py::test_admin_status_includes_recent_activity -q
```
Expected: FAIL before normalization/tracking fixes land.

- [ ] **Step 3: Expand request tracking and normalize the status payload**

In `codai/api/log.py`, extend `_TRACKED_PATHS` to cover all studio-visible endpoints that should appear in activity, keeping the same stable row shape:
```python
_TRACKED_PATHS = {
    "/v1/chat/completions": "chat",
    "/v1/completions": "completion",
    "/v1/images/generations": "image",
    "/v1/audio/speech": "tts",
    "/v1/audio/transcriptions": "transcription",
    "/v1/embeddings": "embedding",
    # add any currently used studio endpoints for video/audio/image edits if present
}
```
In `codai/admin/routes.py`, normalize rows before returning them:
```python
normalized_activity = []
for row in recent_activity:
    if not isinstance(row, dict):
        continue
    normalized_activity.append({
        "time": int(row.get("time", 0) or 0),
        "model": str(row.get("model") or "—"),
        "type": str(row.get("type") or "unknown"),
        "status": int(row.get("status", 0) or 0),
        "duration": round(float(row.get("duration", 0) or 0), 2),
    })
```
Return `normalized_activity` instead of the raw list.

If `dashboard.html` currently uses a nonexistent CSS class such as `badge-danger`, replace it with a valid class already defined in `style.css` or add a shared danger badge style in the stylesheet.

- [ ] **Step 4: Run the focused backend test to verify it passes**

Run:
```bash
pytest tests/<path-to-admin-or-routes-test>.py::test_admin_status_includes_recent_activity -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codai/api/log.py codai/admin/routes.py codai/admin/templates/dashboard.html codai/admin/static/style.css tests/<path-to-admin-or-routes-test>.py
git commit -m "fix: restore admin recent activity feed"
```

## Task 3: Widen studio generation control panes

**Files:**
- Modify: `codai/admin/templates/chat.html:109-126`
- Test: `tests/test_studio_composed_surfaces.py`

- [ ] **Step 1: Write the failing template assertion**

Extend `tests/test_studio_composed_surfaces.py` with an assertion that the studio template includes the updated wider `.gen-ctrl` rule.

Example assertion content:
```python
def test_studio_generation_panel_uses_wider_control_column():
    html = render_chat_template_somehow(...)
    assert ".gen-ctrl { width:min(380px, 36vw);" in html
```
Use the existing test style in the file rather than inventing a new render helper.

- [ ] **Step 2: Run the focused studio test to verify it fails**

Run:
```bash
pytest tests/test_studio_composed_surfaces.py::test_studio_generation_panel_uses_wider_control_column -q
```
Expected: FAIL before the CSS update.

- [ ] **Step 3: Implement the responsive width change**

In `codai/admin/templates/chat.html`, replace the current narrow fixed rule:
```css
.gen-ctrl {
  width: 290px;
  ...
}
```
with a responsive rule such as:
```css
.gen-ctrl {
  width: min(380px, 36vw);
  min-width: 340px;
  max-width: 420px;
  padding: .9rem 1rem;
  overflow-y: auto;
  border-right: 1px solid var(--border);
  background: var(--surface-1);
  display: flex;
  flex-direction: column;
  gap: .65rem;
  flex-shrink: 0;
}
```
Add a mobile media query if needed so small screens can drop below the desktop min-width without horizontal breakage.

- [ ] **Step 4: Run the focused studio test to verify it passes**

Run:
```bash
pytest tests/test_studio_composed_surfaces.py::test_studio_generation_panel_uses_wider_control_column -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codai/admin/templates/chat.html tests/test_studio_composed_surfaces.py
git commit -m "fix: widen studio control panels"
```

## Task 4: Mirror partial and unavailable capability notes in output panes

**Files:**
- Modify: `codai/admin/templates/chat.html`
- Test: `tests/test_studio_composed_surfaces.py`

- [ ] **Step 1: Write the failing studio surface test**

Add a test asserting that the output area rendering includes a warning container for partial/unavailable capability states.

Example target assertions:
```python
def test_studio_output_surfaces_capability_warnings():
    html = render_chat_template_somehow(...)
    assert "cap-output-note" in html
    assert "renderCapabilityOutputNote" in html
```
Use the project’s existing template-string assertions if the test file is static-surface based.

- [ ] **Step 2: Run the focused studio test to verify it fails**

Run:
```bash
pytest tests/test_studio_composed_surfaces.py::test_studio_output_surfaces_capability_warnings -q
```
Expected: FAIL.

- [ ] **Step 3: Implement the output-area warning block and helper**

In `codai/admin/templates/chat.html`:
1. Add a reusable output-note container style near existing capability card styles:
```css
.cap-output-note {
  width: 100%;
  max-width: 960px;
  border: 1px solid rgba(245, 158, 11, .24);
  background: rgba(58, 37, 16, .72);
  color: #f6d08a;
  border-radius: 8px;
  padding: .75rem .9rem;
  display: flex;
  flex-direction: column;
  gap: .35rem;
}

.cap-output-note.unavailable {
  border-color: rgba(248, 113, 113, .24);
  background: rgba(80, 28, 28, .62);
  color: #f3b2b2;
}
```
2. Add a JS helper that reuses existing capability metadata:
```javascript
function renderCapabilityOutputNote(modeKey, outEl) {
  const status = getCapabilityStatusForMode(modeKey);
  if (!status || status.state === 'available') return;
  const missing = Array.isArray(status.missing) ? status.missing : [];
  const title = status.state === 'unavailable' ? 'Feature unavailable' : 'Feature partially available';
  const detail = missing.length ? missing.join(', ') : 'Some required pieces are missing.';
  outEl.innerHTML = `
    <div class="cap-output-note ${status.state === 'unavailable' ? 'unavailable' : ''}">
      <strong>${title}</strong>
      <div>${detail}</div>
    </div>
  `;
}
```
3. Call the helper whenever a generation panel is selected or reset so the right-hand output area preserves the warning alongside empty-state content.

Use the real capability state getter names already present in `chat.html`; do not invent parallel metadata storage.

- [ ] **Step 4: Run the focused studio test to verify it passes**

Run:
```bash
pytest tests/test_studio_composed_surfaces.py::test_studio_output_surfaces_capability_warnings -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codai/admin/templates/chat.html tests/test_studio_composed_surfaces.py
git commit -m "fix: show studio capability notes in outputs"
```

## Task 5: Rebuild the minimal pipeline browser/editor

**Files:**
- Modify: `codai/admin/templates/chat.html`
- Test: `tests/test_studio_composed_surfaces.py`
- Test: any existing pipeline/admin API test files discovered during implementation

- [ ] **Step 1: Write the failing studio/pipeline surface tests**

Add assertions for the minimal pipeline flow being visibly present in the rendered studio surface.

Example assertions:
```python
def test_pipeline_tab_exposes_create_action_and_empty_state():
    html = render_chat_template_somehow(...)
    assert "Create pipeline" in html
    assert "pipe-empty-state" in html
    assert "renderPipelineList" in html


def test_pipeline_tab_exposes_editor_shell():
    html = render_chat_template_somehow(...)
    assert "pipe-editor" in html
    assert "openPipeline" in html
```
If there is backend pipeline data support with tests, add one focused API/status test for the list payload shape as well.

- [ ] **Step 2: Run the focused pipeline tests to verify they fail**

Run:
```bash
pytest tests/test_studio_composed_surfaces.py::test_pipeline_tab_exposes_create_action_and_empty_state -q
pytest tests/test_studio_composed_surfaces.py::test_pipeline_tab_exposes_editor_shell -q
```
Expected: FAIL.

- [ ] **Step 3: Implement the pipeline browse/create/select/edit shell**

In `codai/admin/templates/chat.html`, add:
1. A visible pipeline toolbar with a primary create button.
2. An explicit empty-state container.
3. A pipeline list container rendering cards/items for existing pipelines.
4. A details/editor area that opens when a pipeline is selected.
5. Focused helper functions for local state and rendering.

Minimum structure to add:
```html
<div class="pipe-toolbar">
  <button class="btn btn-primary" onclick="createPipeline()">Create pipeline</button>
  <div class="muted small" id="pipe-status"></div>
</div>
<div class="pipe-shell">
  <div class="pipe-list" id="pipe-list"></div>
  <div class="pipe-editor" id="pipe-editor">
    <div class="pipe-empty-state" id="pipe-empty-state">No pipeline selected.</div>
  </div>
</div>
```
Minimum helper shape:
```javascript
const pipelineState = {
  items: [],
  selectedId: null,
};

function createPipeline() { /* create local/default pipeline model or call existing backend hook */ }
function openPipeline(id) { /* select one pipeline and rerender */ }
function renderPipelineList() { /* render cards or empty state */ }
function renderPipelineEditor() { /* render selected pipeline steps and fields */ }
```
Use the existing backend/data hooks already present in the file if they exist; otherwise keep the shell honest with visible empty-state and minimal editable local state.

- [ ] **Step 4: Run the focused pipeline tests to verify they pass**

Run:
```bash
pytest tests/test_studio_composed_surfaces.py::test_pipeline_tab_exposes_create_action_and_empty_state -q
pytest tests/test_studio_composed_surfaces.py::test_pipeline_tab_exposes_editor_shell -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add codai/admin/templates/chat.html tests/test_studio_composed_surfaces.py tests/<any-pipeline-test-file>.py
git commit -m "feat: restore minimal studio pipeline editor"
```

## Task 6: Full verification and final integration pass

**Files:**
- Modify: any of the above if verification reveals issues
- Test: all relevant focused and broader checks

- [ ] **Step 1: Run focused tests for all touched areas**

Run:
```bash
pytest tests/test_studio_composed_surfaces.py -q
pytest tests/<path-to-admin-or-routes-test>.py -q
```
Expected: PASS.

- [ ] **Step 2: Run project lint and typecheck commands**

Determine the repository’s actual commands from existing project config/docs, then run them exactly. Example placeholders to replace with real commands once discovered:
```bash
<real-lint-command>
<real-typecheck-command>
```
Expected: PASS.

- [ ] **Step 3: Run any additional targeted regression tests**

Run the backend or admin test groups most closely related to routes, request logging, and studio surfaces. Example:
```bash
pytest tests -q
```
Only use a broader suite if it is practical in this repository.

- [ ] **Step 4: Fix any failures uncovered and rerun the affected commands**

For each failure:
```text
- adjust only the relevant file
- rerun the smallest failing test first
- rerun lint/typecheck if the fix touched Python or shared frontend template logic
```
Expected: all previously failing checks pass.

- [ ] **Step 5: Commit**

```bash
git add codai/admin/static/style.css codai/api/log.py codai/admin/routes.py codai/admin/templates/dashboard.html codai/admin/templates/chat.html tests
git commit -m "fix: polish admin studio experience"
```

## Self-Review

### Spec coverage check
- Dark-theme controls and scrollbars: covered by Task 1.
- Overview recent activity recovery: covered by Task 2.
- Wider studio split-pane controls: covered by Task 3.
- Output-area partial/unavailable notes: covered by Task 4.
- Minimal pipeline browser/editor rebuild: covered by Task 5.
- Verification, lint, typecheck, and regression checks: covered by Task 6.

### Placeholder scan
- Remaining placeholders intentionally require discovery of exact existing backend test file paths and exact project lint/typecheck commands before execution.
- These must be resolved during implementation by inspecting the existing repo structure before code changes are finalized.

### Type consistency check
- `recent_activity` row shape is consistently defined as `time`, `model`, `type`, `status`, `duration`.
- Capability output helper uses `state` and `missing` naming consistently.
- Pipeline state consistently uses `items` and `selectedId`.
