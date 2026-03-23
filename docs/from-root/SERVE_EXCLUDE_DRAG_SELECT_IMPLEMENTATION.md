## Serve Exclude Drag Selection Implementation Plan

### Goal

Add click-and-drag range selection to the Serve `Exclude` page so a reviewer can mark multiple consecutive epochs as rejected or restored in one motion.

This is a quality-of-life feature for the browser-based Serve workflow. It should build on the current single-epoch focus/toggle behavior and preserve the existing autosave model.

### Desired User Experience

- User can press on an epoch row/card/slot in the epoch strip and drag across adjacent epochs.
- The UI shows a live preview of the selected range before commit.
- Releasing the pointer applies one bulk action to the whole range.
- Default drag behavior should be `reject selected epochs`.
- If the drag starts from an already rejected epoch, the UI may optionally switch to `restore selected epochs`, but that should be explicit in the UI if implemented.
- Single-click and keyboard toggle behavior should continue to work.
- Dragging should only affect contiguous epochs in the current loaded manifest.

### Current State

The current Serve Exclude page already has:

- epoch focus state
- bad epoch state in `badEpochs`
- autosave for epoch review changes
- keyboard toggle behavior for the focused epoch

Relevant files:

- [Exclude.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/pages/Exclude.tsx)
- [Exclude.test.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/pages/Exclude.test.tsx)

This means the feature should be implemented primarily in the frontend interaction layer. No backend API change should be required unless we later want explicit batch-action metadata.

### Recommended Product Behavior

#### Selection Model

- Drag gesture selects a contiguous epoch range.
- Range selection should be preview-only until pointer release.
- On release, convert the preview into a single state update to `badEpochs`.
- The page should schedule one autosave after the bulk update, not one save per epoch.

#### Bulk Action Rule

Use a simple, predictable rule:

- if drag starts on a clean epoch, mark the full range rejected
- if drag starts on a rejected epoch, mark the full range restored

This avoids mixed-mode ambiguity and keeps the gesture fast.

#### Scope

Phase 1 should support:

- mouse drag
- trackpad drag
- contiguous selection only
- current loaded epoch view only

Defer for later:

- non-contiguous multi-select
- modifier-key additive selection
- touch-specific gesture support
- drag selection across virtualized/unloaded windows

### Technical Design

#### Frontend State Additions

Add temporary interaction state in `Exclude.tsx`:

- `dragAnchorEpoch: number | null`
- `dragHoverEpoch: number | null`
- `dragMode: "reject" | "restore" | null`
- `isDraggingEpochRange: boolean`

Derived helper:

- `previewEpochRange: number[]`

This preview state should not immediately mutate persisted review state until drag end.

#### Event Handling

For each rendered epoch item:

- `onPointerDown`
- `onPointerEnter`
- `onPointerUp`

Page-level cleanup:

- cancel drag on pointer cancel
- cancel drag on pointer leave/window blur when needed

Recommended flow:

1. `pointerdown` on epoch `N`
2. determine drag mode from current rejected state of `N`
3. set anchor to `N`
4. while dragging, update hover epoch and recompute preview range
5. on `pointerup`, apply the full range in one state update
6. schedule existing autosave once

#### Data Update Logic

Create a small helper in `Exclude.tsx`:

- `applyEpochRangeAction(start: number, end: number, mode: "reject" | "restore")`

Behavior:

- compute inclusive epoch index range
- if `reject`, union with existing `badEpochs`
- if `restore`, remove those indices from `badEpochs`
- keep final array sorted and unique

This helper should be the only place that mutates epoch-review state for drag actions.

#### Visual Treatment

Each epoch item should support three visual states:

- normal
- preview-selected
- committed-rejected

Recommended styling:

- preview state uses a lighter temporary highlight
- committed rejected state keeps the existing stronger rejected style
- anchor epoch can optionally get a thin outline for clarity

The preview needs to be visually distinct so users can tell “not yet committed” versus “already rejected.”

### Architecture Notes

#### Keep It Frontend-Only First

Do not add backend endpoints for this feature in the first pass.

Reasons:

- current API already accepts full `bad_epoch_indices`
- bulk drag is just another client-side way to edit that array
- keeping it frontend-only reduces risk

#### Do Not Couple It to Zoom/Pan Yet

If the epoch view later gets richer gesture support, this feature should remain isolated as an interaction layer on the epoch selection elements themselves.

Avoid:

- mixing drag-select with graph pan in the same surface without a mode distinction
- hidden gesture rules that differ by browser

### Risks

#### Interaction Conflicts

Biggest UX risk is accidental selection when users meant to click once.

Mitigations:

- do not treat tiny pointer jitter as a full drag unless the pointer crosses into a second epoch
- preserve simple click-to-focus behavior
- only commit on pointer release

#### Autosave Churn

Bulk changes should not trigger repeated save calls while dragging.

Mitigation:

- only update persisted `badEpochs` at drag end
- reuse current debounced save path

#### Virtualization / Large Files

If the epoch list is only partially rendered or later becomes virtualized, drag selection across unloaded epochs becomes harder.

Mitigation:

- explicitly scope phase 1 to currently rendered/loaded epochs

### Implementation Phases

#### Phase 1: Core Drag Selection

- identify the epoch row/item component or mapped render block in `Exclude.tsx`
- add drag interaction state
- add pointer handlers
- add range preview computation
- add batch apply helper
- wire bulk update into the existing epoch autosave flow
- add temporary preview styling

#### Phase 2: UX Hardening

- distinguish click from drag more cleanly
- add cancel behavior for interrupted drags
- improve keyboard/focus behavior after bulk apply
- add small helper text such as `Drag across epochs to reject or restore a range`

#### Phase 3: Optional Enhancements

- shift-drag or modifier-based additive logic
- explicit reject/restore mode toggle
- click-and-drag across a minimap or overview strip
- touch support

### Testing Plan

Add frontend tests in:

- [Exclude.test.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/pages/Exclude.test.tsx)

Minimum test coverage:

- drag from clean epoch across multiple epochs marks full range rejected
- drag from rejected epoch across multiple epochs restores full range
- single click still behaves as before
- drag preview appears before commit
- only one save call is triggered for one completed drag interaction
- interrupted drag does not corrupt state

### Suggested Build Order

1. add pure helper for range application
2. add drag interaction state
3. wire pointer handlers to epoch elements
4. add preview styling
5. add tests for reject-range flow
6. add tests for restore-range flow
7. polish accidental-drag handling

### Acceptance Criteria

- reviewer can drag across consecutive epochs to reject them in one gesture
- reviewer can drag from a rejected epoch to restore a consecutive range
- existing single-epoch keyboard and click workflows still work
- drag action results in one coherent autosave operation
- no backend API changes are required for initial rollout

### Recommendation

This is worth doing, but it should stay a narrowly scoped frontend enhancement.

The best first version is:

- contiguous range drag only
- one clear rule for reject vs restore
- no backend changes
- solid tests around autosave and accidental-drag behavior
