# Whisper-Server Path Cell Two-Line Clamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Narrow the configured whisper-server model path column to `160px` and render up to two visible lines before truncation while preserving full-path hover disclosure.

**Architecture:** Keep the change localized to the configured whisper-server row renderer in the admin Models template and its existing template regression test. Replace the current single-line truncation inline style with a bounded two-line clamp style, then update the regression assertion to lock the new rendering contract in place.

**Tech Stack:** Jinja/HTML template, inline CSS, vanilla JavaScript-rendered table markup, pytest

---

### Task 1: Update the configured whisper-server path cell rendering

**Files:**
- Modify: `codai/admin/templates/models.html:997-1001`
- Test: `tests/test_whisper_server_local_models.py:735-741`

- [ ] **Step 1: Write the failing template regression expectation**

Update the existing test `test_models_template_truncates_configured_whisper_server_model_paths` in `tests/test_whisper_server_local_models.py` so it expects the new bounded two-line clamp style instead of the old single-line `320px` style:

```python
def test_models_template_truncates_configured_whisper_server_model_paths():
    template = Path("codai/admin/templates/models.html").read_text()

    assert '<th style="text-align:left;padding:.3rem .25rem;font-weight:700">Model path</th>' in template
    assert 'max-width:160px;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;line-height:1.25;max-height:2.5em' in template
    assert 'title="${esc(m.model_path || \"—\")}"' in template
    assert '>${esc(m.model_path || "—")}</td>' in template
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_whisper_server_local_models.py -k truncates_configured_whisper_server_model_paths`
Expected: FAIL because `codai/admin/templates/models.html` still contains the old `max-width:320px` single-line style.

- [ ] **Step 3: Apply the minimal template change**

Change the configured whisper-server row path cell in `codai/admin/templates/models.html` from the old single-line style to the new two-line bounded style:

```html
<td style="padding:.4rem .25rem;font-size:11px;color:var(--text-2);max-width:160px;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;line-height:1.25;max-height:2.5em" title="${esc(m.model_path || "—")}">${esc(m.model_path || "—")}</td>
```

This preserves the same rendered text source and `title` tooltip while halving the width and allowing two visible lines.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_whisper_server_local_models.py -k truncates_configured_whisper_server_model_paths`
Expected: PASS.

- [ ] **Step 5: Commit the localized rendering change**

```bash
git add tests/test_whisper_server_local_models.py codai/admin/templates/models.html
git commit -m "fix: clamp whisper-server paths to two lines"
```

### Task 2: Run the whisper-server regression file

**Files:**
- Verify: `tests/test_whisper_server_local_models.py`

- [ ] **Step 1: Run the full whisper-server regression file**

Run: `/storage/coderai/venv_all/bin/python -m pytest tests/test_whisper_server_local_models.py`
Expected: PASS with all whisper-server regression tests green; existing unrelated warnings from `codai/api/images.py` may still appear.

- [ ] **Step 2: Inspect git status for only intended changes**

Run: `git status --short`
Expected output includes only:

```text
M codai/admin/templates/models.html
M tests/test_whisper_server_local_models.py
```

If the spec document commit from the design step is already present, no additional docs file should appear as modified.

- [ ] **Step 3: Commit verification-complete state if Task 1 commit has not yet been created**

If Step 5 from Task 1 was already completed successfully, do not create another commit. Otherwise use:

```bash
git add tests/test_whisper_server_local_models.py codai/admin/templates/models.html
git commit -m "fix: clamp whisper-server paths to two lines"
```

## Self-Review

- Spec coverage: the plan updates only the configured whisper-server path rendering contract and its regression test, matching the approved design.
- Placeholder scan: all file paths, assertions, commands, and expected outcomes are explicit.
- Type consistency: the plan preserves the existing `m.model_path` rendering path and test name without introducing new identifiers.
