# 2026-05-06 Pipeline Accordion Restoration Design

## Summary
Restore the custom pipeline area in the studio pipeline tab to a single vertical accordion flow. The built-in runnable pipeline cards remain at the top, the “Create new pipeline” builder returns to being the first custom panel, and saved custom pipelines render immediately below it as expandable cards rather than through the newer split list/editor layout.

## Goals
- Make the pipeline tab read top-to-bottom again.
- Restore the visual pattern where the builder is the first custom pipeline panel.
- Show saved/custom pipelines directly below the builder in expandable cards.
- Keep the pipeline create/edit/run/delete functionality already wired to the current backend.
- Avoid making the saved pipeline area look replaced by the session artifact history section.

## Non-Goals
- Reworking the built-in runnable pipeline cards above the custom pipeline area.
- Changing the pipeline backend APIs or data model.
- Replacing the current builder form or actions with a new editor concept.
- Changing artifact history behavior outside of separating it visually from pipeline listing expectations.

## Current Context
- The pipeline tab lives in `codai/admin/templates/chat.html`.
- Built-in runnable pipeline cards still render above the custom pipeline section.
- The recent refactor introduced a split custom pipeline shell using `pipe-toolbar`, `pipe-shell`, `pipe-list`, and `pipe-editor`.
- That split layout moved the “Create pipeline” action into a detached toolbar and separated the builder from the saved pipeline cards, which no longer matches the expected original flow.
- The saved/custom pipeline rendering still uses the same backend fetch sources (`/v1/pipelines/custom`, `/v1/pipelines/step-types`) and can be reused.

## Problem Statement
The user expects the custom pipeline area to behave like a vertical stack of expandable panels. In the previous mental model and UI shape:
1. built-in runnable pipelines appeared first,
2. the first custom panel was the “Create new pipeline” builder,
3. existing saved pipelines followed immediately underneath,
4. each saved pipeline could be opened inline as its own panel.

The new split layout breaks that expectation by introducing a separate list/editor arrangement. This makes the custom pipelines harder to discover in context and visually displaces them, to the point that the area after “session artifact history” now feels like the saved pipeline section has gone missing.

## Proposed Approach

### A. Restore single-column custom pipeline flow
Remove the split custom pipeline shell and return to a single-column vertical stack directly in the pipeline panel.

#### Structure
- Keep the built-in runnable pipeline cards unchanged.
- Keep the builder card immediately after them.
- Place saved custom pipeline cards directly below the builder in `#custom-pipe-cards`.
- Preserve the existing card/accordion visual language already used in the pipeline tab.

#### Expected behavior
- The pipeline tab reads as one continuous accordion.
- The create flow is always visible in the expected position.
- Saved pipelines are visible where the user expects them, without needing a separate side list.

### B. Keep current backend wiring, simplify UI state
Reuse the existing fetch/state logic for step types and saved pipelines, but remove state that only exists to support the split shell.

#### Keep
- `_stepTypes`
- `_pbSteps`
- `_customPipelines`
- `_editingPipelineId`
- backend calls for save, run, load, and delete

#### Remove or simplify
- `pipelineState.items`
- `pipelineState.selectedId`
- `renderPipelineList()`
- `openPipeline()` as a separate selection workflow for a detached list/editor shell
- toolbar status messaging that only serves the detached list view

#### Expected behavior
- `createPipeline()` resets and opens the builder card.
- `editCustomPipeline(id)` fills the builder card from the selected saved pipeline and scrolls the user to that builder.
- `renderCustomPipelineCards()` directly renders expandable cards for every saved pipeline beneath the builder.

### C. Preserve inline saved-pipeline actions
Each saved custom pipeline card should remain self-contained and actionable.

#### Card contents
- pipeline name/title
- description
- step summary chips/arrows
- input field
- run button
- edit button
- delete button
- output/progress area

#### Expected behavior
- Users can browse and operate on saved pipelines without switching panes.
- Editing still uses the main builder as the canonical edit form.
- The saved pipeline cards remain visible while editing, preserving context.

### D. Re-separate from artifact history visually
The artifact history section should no longer appear to occupy the place where saved pipelines should be.

#### Approach
- Ensure the custom pipeline accordion begins immediately after the built-in pipeline cards.
- Avoid introducing detached empty-state placeholders or side containers that visually interrupt that flow.
- If needed, keep spacing and headings clear enough that artifact history remains understood as an output/history surface, not a pipeline browser.

## File-Level Design

### `codai/admin/templates/chat.html`
Responsibilities:
- Remove the split custom pipeline layout markup.
- Restore the builder card as the first custom accordion panel.
- Render saved pipelines directly below the builder in `#custom-pipe-cards`.
- Simplify JS helpers by deleting split-layout-only render/state functions.
- Keep builder/reset/edit/save/delete/run functionality intact.

### `tests/test_studio_composed_surfaces.py`
Responsibilities:
- Assert the builder card appears before `#custom-pipe-cards`.
- Assert split-shell markers (`pipe-shell`, `pipe-list`, `pipe-editor`) are absent.
- Assert saved pipeline rendering remains tied to expandable pipeline cards and custom card rendering helpers.

## Error Handling
- If no saved pipelines exist, render no saved cards below the builder rather than replacing the builder with a detached empty-state shell.
- If fetching saved pipelines fails, keep the builder available and surface the failure through lightweight status/progress messaging rather than hiding the pipeline area.
- If editing a saved pipeline fails to load, preserve the current builder state and show an error instead of collapsing the custom pipeline section.

## Testing Strategy

### Template/UI verification
- Verify the builder card remains the first custom pipeline panel after the built-in pipeline cards.
- Verify saved custom pipelines render below the builder as expandable cards.
- Verify no split-shell containers remain in the rendered template.
- Verify create/edit actions still point to the current builder functions.

### Behavioral verification
- Verify `createPipeline()` resets the builder and opens it.
- Verify `editCustomPipeline(id)` populates the builder and keeps the saved cards visible below.
- Verify saved pipeline cards still expose run/edit/delete controls inline.

## Risks and Mitigations
- Removing the split layout could accidentally drop some of the new pipeline discoverability work. Mitigation: keep the create builder always visible and preserve clear saved-card summaries.
- State cleanup could break save/edit flows if split-layout-only helpers are referenced elsewhere. Mitigation: remove only state/functions that are exclusive to the detached list/editor workflow and keep backend-facing helpers unchanged.
- The large single-file template makes small UI rewires easy to tangle. Mitigation: keep the restoration narrowly scoped to the custom pipeline section and its direct helper functions.

## Success Criteria
- The pipeline tab again shows a vertical sequence of expandable panels.
- “Create new pipeline” is the first custom pipeline panel.
- Saved custom pipelines appear directly below it.
- The detached split list/editor shell is gone.
- Saved pipelines no longer seem to be replaced by the artifact history area.
