# Pipeline Accordion Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the studio pipeline tab so the custom pipeline area is again a single vertical accordion with the builder first and saved pipelines directly below it.

**Architecture:** Keep all existing backend pipeline endpoints and builder logic, but simplify the frontend template by removing the detached split list/editor shell. The pipeline section remains in `codai/admin/templates/chat.html`, with saved custom pipelines rendered inline as expandable cards beneath the builder and tested through template surface assertions.

**Tech Stack:** Jinja template, vanilla JavaScript, shared admin CSS, pytest

---

## File Structure

- Modify: `codai/admin/templates/chat.html`
  - Remove split custom pipeline layout markup and related shell-only CSS/JS helpers.
  - Restore builder-first accordion flow and inline saved-pipeline cards.
- Modify: `tests/test_studio_composed_surfaces.py`
  - Update and extend template-surface assertions for the restored accordion structure.

## Task 1: Write regression tests for the restored accordion structure

**Files:**
- Modify: `tests/test_studio_composed_surfaces.py:371-385`
- Test: `tests/test_studio_composed_surfaces.py`

- [ ] **Step 1: Write the failing tests**

Replace the split-layout assertions with accordion-focused checks.

Add/update tests to assert:
```python
def test_pipeline_tab_renders_builder_before_saved_custom_cards():
    template_path = "/storage/coderai/codai/admin/templates/chat.html"
    text = open(template_path, "r", encoding="utf-8").read()

    builder_index = text.index('id="pipe-builder-card"')
    custom_cards_index = text.index('id="custom-pipe-cards"')
    assert builder_index < custom_cards_index
    assert "＋ Build New Pipeline" in text


def test_pipeline_tab_drops_split_shell_layout():
    template_path = "/storage/coderai/codai/admin/templates/chat.html"
    text = open(template_path, "r", encoding="utf-8").read()

    assert 'class="pipe-shell"' not in text
    assert 'class="pipe-list"' not in text
    assert 'class="pipe-editor"' not in text
    assert 'id="pipe-empty-state"' not in text


def test_pipeline_tab_keeps_inline_saved_pipeline_cards():
    template_path = "/storage/coderai/codai/admin/templates/chat.html"
    text = open(template_path, "r", encoding="utf-8").read()

    assert "renderCustomPipelineCards" in text
    assert '<details class="pipe-card"' in text
    assert "editCustomPipeline" in text
    assert "deleteCustomPipeline" in text
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:
```bash
"/storage/coderai/venv_all/bin/python" -m pytest tests/test_studio_composed_surfaces.py::test_pipeline_tab_renders_builder_before_saved_custom_cards tests/test_studio_composed_surfaces.py::test_pipeline_tab_drops_split_shell_layout tests/test_studio_composed_surfaces.py::test_pipeline_tab_keeps_inline_saved_pipeline_cards -q
```
Expected: FAIL because the current template still contains the split shell markers.

- [ ] **Step 3: Commit the failing test changes only after the full task passes**

Do not commit yet. Continue to Task 2 and commit once implementation and verification are green.

## Task 2: Remove split-shell-only pipeline layout markup and CSS

**Files:**
- Modify: `codai/admin/templates/chat.html:1368-1405`
- Test: `tests/test_studio_composed_surfaces.py`

- [ ] **Step 1: Implement the layout restoration in the template markup**

Replace the current split block:
```html
<div class="pipe-toolbar">...</div>
<div class="pipe-shell">
  <div class="pipe-list" id="pipe-list"></div>
  <div class="pipe-editor" id="pipe-editor">
    <div class="pipe-empty-state" id="pipe-empty-state">No pipeline selected.</div>
    <details class="pipe-card" id="pipe-builder-card" open>...</details>
    <div id="custom-pipe-cards"></div>
  </div>
</div>
```
with the restored vertical flow:
```html
<details class="pipe-card" id="pipe-builder-card" open>
  <summary>＋ Build New Pipeline
    <span class="pipe-steps" style="margin:0;font-size:11px"><span class="pipe-step">Add steps</span><span class="pipe-arrow">→</span><span class="pipe-step">Configure</span><span class="pipe-arrow">→</span><span class="pipe-step">Save &amp; Run</span></span>
  </summary>
  <div class="pipe-card-body">
    ...existing builder fields/actions...
  </div>
</details>
<div id="custom-pipe-cards"></div>
```
Do not add a detached toolbar or empty-state wrapper around the builder.

- [ ] **Step 2: Remove split-layout-only CSS rules**

Delete the shell-only rules from `chat.html`, including these selectors if they are no longer used:
```css
.pipe-toolbar
.pipe-shell
.pipe-list
.pipe-editor
.pipe-empty-state
.pipe-list-card
.pipe-list-card.active
```
Also remove the small-screen `.pipe-shell { grid-template-columns:1fr; }` media-query override if it is no longer relevant.

Keep the core `.pipe-card`, `.pipe-head`, `.pipe-summary`, `.pipe-steps`, and tag styles that still serve the accordion cards.

- [ ] **Step 3: Run the focused tests to verify the layout change passes**

Run:
```bash
"/storage/coderai/venv_all/bin/python" -m pytest tests/test_studio_composed_surfaces.py::test_pipeline_tab_renders_builder_before_saved_custom_cards tests/test_studio_composed_surfaces.py::test_pipeline_tab_drops_split_shell_layout tests/test_studio_composed_surfaces.py::test_pipeline_tab_keeps_inline_saved_pipeline_cards -q
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add codai/admin/templates/chat.html tests/test_studio_composed_surfaces.py
git commit -m "fix: restore pipeline accordion layout"
```

## Task 3: Simplify pipeline JavaScript to match the accordion flow

**Files:**
- Modify: `codai/admin/templates/chat.html:3310-3475`
- Test: `tests/test_studio_composed_surfaces.py`

- [ ] **Step 1: Write one more failing assertion for shell-only JS removal**

Extend `test_pipeline_tab_drops_split_shell_layout()` or add a new test to assert split-shell-only helpers/state are gone:
```python
def test_pipeline_tab_drops_split_selection_helpers():
    template_path = "/storage/coderai/codai/admin/templates/chat.html"
    text = open(template_path, "r", encoding="utf-8").read()

    assert "renderPipelineList" not in text
    assert "pipelineState = {" not in text
    assert "selectedId" not in text
```

- [ ] **Step 2: Run the focused JS-structure test to verify it fails**

Run:
```bash
"/storage/coderai/venv_all/bin/python" -m pytest tests/test_studio_composed_surfaces.py::test_pipeline_tab_drops_split_selection_helpers -q
```
Expected: FAIL.

- [ ] **Step 3: Remove split-shell-only JS state and render helpers**

In `codai/admin/templates/chat.html`:
1. Delete the split-layout-only state:
```javascript
const pipelineState = { items: [], selectedId: null };
```
2. Delete `renderPipelineList()` entirely.
3. Remove the detached selection workflow from `openPipeline(id)`.
4. Update `initPipelineBuilder()` so it no longer populates `pipelineState` or calls `renderPipelineList()`.
5. Update `renderCustomPipelineCards()` so it renders all saved pipelines directly:
```javascript
function renderCustomPipelineCards() {
  const container = $('custom-pipe-cards');
  if (!container) return;
  container.innerHTML = _customPipelines.map(p => `
    <details class="pipe-card">
      <summary>
        <div class="pipe-head">
          <div class="pipe-title">${escapeHtml(p.name || p.id)}</div>
          <div class="pipe-summary">${escapeHtml(p.description || 'Custom multi-step pipeline for combining model and utility actions.')}</div>
          <div class="pipe-steps">
            ${(p.steps || []).map(s => `<span class="pipe-step">${escapeHtml(s.label || s.type)}</span>`).join('<span class="pipe-arrow">→</span>')}
          </div>
          <div class="pipe-tags"><span class="pipe-tag">custom</span><span class="pipe-tag">pipeline</span><span class="pipe-tag">${(p.steps || []).length} steps</span></div>
        </div>
      </summary>
      <div class="pipe-card-body">
        ${p.description ? `<p style="font-size:12px;color:var(--text-2);margin:0 0 .4rem">${escapeHtml(p.description)}</p>` : ''}
        <div class="frow"><label class="fl">Input</label><input id="cpr-input-${p.id}" class="fi" placeholder="{{ '{{' }}input{{ '}}' }} value"></div>
        <div style="display:flex;gap:.4rem;margin-top:.4rem;flex-wrap:wrap">
          <button class="btn btn-primary btn-sm" onclick="runCustomPipeline('${p.id}')">▶ Run</button>
          <button class="btn btn-ghost btn-sm" onclick="editCustomPipeline('${p.id}')">✎ Edit</button>
          <button class="btn btn-ghost btn-sm" style="color:var(--red)" onclick="deleteCustomPipeline('${p.id}')">✕ Delete</button>
        </div>
        <div class="progress" id="cpr-prog-${p.id}"></div>
        <div id="cpr-out-${p.id}"></div>
      </div>
    </details>`).join('');
}
```
6. Keep `createPipeline()` only as a builder reset/open helper:
```javascript
function createPipeline() {
  _editingPipelineId = null;
  _pbSteps = [];
  $('pb-name').value = '';
  $('pb-desc').value = '';
  $('pb-input').value = '';
  $('pb-prog').textContent = 'Creating a new pipeline draft.';
  renderBuilderSteps();
  $('pipe-builder-card').open = true;
  $('pipe-builder-card').scrollIntoView({behavior:'smooth', block:'start'});
}
```
7. Update `editCustomPipeline(id)` and `deleteCustomPipeline(id)` so they no longer reference `pipelineState` or `renderPipelineList()`.

- [ ] **Step 4: Run the focused JS-structure test to verify it passes**

Run:
```bash
"/storage/coderai/venv_all/bin/python" -m pytest tests/test_studio_composed_surfaces.py::test_pipeline_tab_drops_split_selection_helpers -q
```
Expected: PASS.

- [ ] **Step 5: Run the full studio surface suite**

Run:
```bash
"/storage/coderai/venv_all/bin/python" -m pytest tests/test_studio_composed_surfaces.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add codai/admin/templates/chat.html tests/test_studio_composed_surfaces.py
git commit -m "fix: simplify custom pipeline rendering"
```

## Task 4: Final verification

**Files:**
- Modify: any touched file only if verification reveals a real issue
- Test: `tests/test_studio_composed_surfaces.py`, plus any directly related suite if needed

- [ ] **Step 1: Run the complete relevant regression checks**

Run:
```bash
"/storage/coderai/venv_all/bin/python" -m pytest tests/test_studio_composed_surfaces.py -q
"/storage/coderai/venv_all/bin/python" -m pytest tests/test_manual_multimodal_test_client.py -q
```
Expected: PASS.

- [ ] **Step 2: Run project lint and typecheck commands if available in repo config**

Discover and run the real commands used by this repository. If none are configured or documented, record that explicitly during execution rather than inventing commands.

Candidate discovery targets:
```bash
grep -R "ruff\|mypy\|typecheck\|lint" README.md pyproject.toml setup.cfg tox.ini .
```
Then run the actual documented commands.

- [ ] **Step 3: Fix any verification failures and rerun the smallest affected checks first**

Use this order:
```text
1. rerun failing focused test
2. rerun tests/test_studio_composed_surfaces.py -q
3. rerun any documented lint/typecheck command affected by the change
```
Expected: all checks green.

- [ ] **Step 4: Commit**

```bash
git add codai/admin/templates/chat.html tests/test_studio_composed_surfaces.py
git commit -m "fix: restore custom pipeline tab flow"
```

## Self-Review

### Spec coverage check
- Builder first in the custom pipeline sequence: covered by Tasks 1 and 2.
- Saved pipelines rendered directly below builder: covered by Tasks 1, 2, and 3.
- Split shell removed: covered by Tasks 1, 2, and 3.
- Existing create/edit/run/delete functionality preserved: covered by Task 3.
- Artifact-history confusion reduced by restoring continuous flow: covered by Tasks 2 and 3.

### Placeholder scan
- All code changes referenced in the plan are concrete and tied to exact files.
- Verification commands use the repository’s existing `venv_all` Python path already established in this repo.
- Lint/typecheck discovery remains intentionally conditional because the actual commands must be confirmed from repo config during execution.

### Type consistency check
- The plan consistently keeps `_stepTypes`, `_pbSteps`, `_customPipelines`, and `_editingPipelineId`.
- `renderCustomPipelineCards()` remains the canonical saved-pipeline renderer.
- `createPipeline()` remains a builder reset/open helper instead of a detached list-selection entrypoint.
