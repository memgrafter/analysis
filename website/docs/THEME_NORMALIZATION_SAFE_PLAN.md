# Theme Normalization Safe Plan

## Context

A previous attempt to unify typography/layout introduced visual regressions (link colors, control borders, hover states), especially on search UI controls. The root issue is not Cabin itself; it is the interaction between:

- browser-native control rendering,
- `color-scheme: light dark`,
- and broad CSS/cascade changes made together.

This plan prioritizes visual stability and incremental change.

---

## Phase 0 — Freeze and Stabilize

- Keep the current known-good theme as-is.
- Do not mix typography/layout/color changes in one commit.
- No broad cross-page CSS rewrites.

---

## Phase 1 — Extract Tokens Only (No Behavior Change)

Create `assets/theme.css` containing only shared variables:

- color variables (copied exactly from current known-good values)
- font variables (`--font-sans`, `--font-mono`)

Import on pages, but do not alter existing selector behavior yet.

**Goal:** zero visual change.

---

## Phase 2 — Typography Pilot (Low Risk)

- Add Cabin via variable only (no broad forced application):
  - `--font-sans: "Cabin", ...`
  - keep `--font-mono` as existing mono stack initially
- Apply to one low-risk page first (About page).
- Validate in light/dark.

**Goal:** isolate typography impact.

---

## Phase 3 — Harden Form Controls (Critical for Search)

Before applying Cabin broadly on search page, explicitly normalize controls:

- explicit `appearance` handling where needed
- explicit `border`, `background`, `color`, `caret-color`
- explicit placeholder color
- avoid relying on UA defaults

This reduces browser-specific surprises from `color-scheme`.

---

## Phase 4 — Incremental Rollout

Roll out one page at a time:

1. Search (most sensitive)
2. Cloud
3. View
4. About (if not already complete)

For each page:

- one focused commit
- screenshot diff in light/dark
- quick manual checks for links, controls, borders, hover states

---

## Guardrails

- Avoid `!important` unless absolutely unavoidable.
- Never combine font + spacing + color updates in one commit.
- Keep a known-good tag/checkpoint before each phase.

---

## Acceptance Criteria

- No regressions in link colors/visited states.
- Search controls remain legible with stable borders in light/dark and focused/unfocused states.
- Theme remains visually consistent across pages.
- Typography changes can be reverted independently from theme tokens.
