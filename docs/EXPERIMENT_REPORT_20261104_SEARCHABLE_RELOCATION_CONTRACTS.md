# Searchable Relocation Navigation Contracts (2026-11-04)

## Scope

Implement searchable navigation contracts for verified non-local/relocation transitions in entity_options planning to unblock StructuredAgent on ls20 level 6, without relaxing state keys or hard-coding level-specific coordinates.

## Problem Statement

Level 6 failure mode: planner expects adjacent entry, observation shows relocation (e.g., enter toward (8,3) then end at (6,3)). Round 16-20 probes documented that these are real, reproducible gameplay transitions, often status-context sensitive.

The issue: `entity_options.py` line 225-226 rejects all non-local transitions with "Nonlocal transitions can use a different entity clock," aborting planning even when MechanismPolicy has already recorded consistent, conflict-free special transitions in `contextual_special_transitions`.

## Solution

Make repeated, conflict-free observed transitions in matching status/context become usable navigation contracts in entity options planning/prediction:

1. **Pass special transitions**: `MechanismPolicy` now passes `contextual_special_transitions` to `EntityOptionController.choose()` via new optional parameter
2. **Store in model**: `EntityOptionModel` stores special transitions and checks them in `movement()` method
3. **Allow searchable relocations**: `movement()` returns `("contextual_special_transition", destination)` when a matching contextual key exists
4. **Enable non-local planning**: `step()` method allows non-local transitions when backed by `"contextual_special_transition"` label

## Changes

### Code Changes

- `agents/entity_options.py`:
  - Added `special_transitions` dict to `EntityOptionController.__init__()` (line 481)
  - Added `special_transitions` dict to `EntityOptionModel.__init__()` (line 86)
  - Added optional `special_transitions` parameter to `choose()` (line 643)
  - Added contextual special transition check in `movement()` (lines 171-177)
  - Updated `step()` to allow non-local when label is `"contextual_special_transition"` (lines 229-234)

- `agents/mechanism_policy.py`:
  - Pass `special_transitions=self.contextual_special_transitions` to `entity_options.choose()` (line 3093)

### Test Changes

Added 5 comprehensive unit tests in `tests/unit/test_entity_options.py`:

1. `test_contextual_special_transition_allows_nonlocal_movement` - Consistent relocation becomes searchable
2. `test_contextual_special_transition_overrides_geometric_navigation` - Special edge used over geometric guess
3. `test_special_transition_rejected_for_wrong_status_context` - Only matches same pattern
4. `test_special_transition_rejected_for_wrong_color_context` - Only matches same color
5. `test_special_transition_works_in_goal_search` - Goal search uses special transitions

## Safety Constraints

- **No new flags**: Uses existing `STRUCTURED_ENTITY_OPTIONS=1` flag (already default-OFF)
- **No hard-coded coordinates**: Solution is generic; no ls20-specific logic
- **Status/context gated**: Only applies when (cell, action, pattern, color) matches exactly
- **No relaxed keys**: State-key ablation rejected per EXPERIMENT_REPORT_20261104_STATE_KEY_ABLATION.md
- **Frozen baseline preserved**: 5/7 baseline with full flag stack still runnable:
  ```bash
  export STRUCTURED_DYNAMIC_CONTACT=1
  export STRUCTURED_OPTION_COMPILER=1
  export STRUCTURED_ENTITY_TRACKING=1
  export STRUCTURED_CONTACT_EFFECTS=1
  export STRUCTURED_ENTITY_OPTIONS=1
  ```

## Test Results

- **424/424 unit tests pass** (405 existing + 5 new + 14 from state-key ablation)
- **All new tests pass** on first run
- **No regressions** in existing entity_options or mechanism_policy tests
- **Linting clean**: `ruff check` passes on all touched files
- **Whitespace clean**: `git diff --check` passes

## Reproducibility

To reproduce offline:

```bash
# Run full test suite
uv run pytest tests/unit/ -v

# Run only new special transition tests
uv run pytest tests/unit/test_entity_options.py::test_contextual_special_transition_allows_nonlocal_movement \
  tests/unit/test_entity_options.py::test_contextual_special_transition_overrides_geometric_navigation \
  tests/unit/test_entity_options.py::test_special_transition_rejected_for_wrong_status_context \
  tests/unit/test_entity_options.py::test_special_transition_rejected_for_wrong_color_context \
  tests/unit/test_entity_options.py::test_special_transition_works_in_goal_search -v

# Check linting
uv run ruff check agents/entity_options.py agents/mechanism_policy.py tests/unit/test_entity_options.py
```

## Level 6 Status

**Still unsolved.** This change enables the planner to use verified relocation transitions, but:

1. No level-6 checkpoint replay was performed (requires API credentials or local game environment)
2. No evidence yet that this specific change is sufficient for 6/7 clear
3. The frozen 5/7 baseline remains the verified reference

### Next Steps for Level 6 Verification

If API credentials become available:

1. Replay from row 620 checkpoint with full flag stack + new code
2. Check if planner now searches through observed (7,3) → ACTION4 → (6,3) relocation
3. Document whether level advances or new failure mode emerges
4. Do NOT claim 6/7 clear without demonstrated evidence

## Hypothesis Verification

Original hypothesis: "entity-options prediction / bad_edge invalidation may be aborting on player relocation instead of consulting contextual_special_transitions."

**Confirmed.** The issue was exactly as hypothesized:
- Line 225-226 rejected all non-local transitions unconditionally
- `contextual_special_transitions` existed in MechanismPolicy but weren't passed to entity_options
- Solution: pass special transitions and allow non-local when backed by consistent observations

## Files Changed

- `agents/entity_options.py` (+20 lines: dict field, movement check, step condition)
- `agents/mechanism_policy.py` (+1 line: pass special_transitions)
- `tests/unit/test_entity_options.py` (+87 lines: 5 new tests)

**Git diff**: 3 files changed, 108 insertions(+), 3 deletions(-)

## Open Questions

1. **Is this sufficient for level 6?** Unknown without replay evidence. This unblocks planning through relocations but doesn't guarantee the level is solvable.

2. **Do other levels regress?** Unit tests suggest no, but full integration testing (e.g., 700-action runs on levels 1-5) was not performed.

3. **Should this be behind a new flag?** Currently reuses existing `STRUCTURED_ENTITY_OPTIONS=1`. If integration testing shows regressions, add `STRUCTURED_RELOCATION_CONTRACTS=1` (default-OFF).

## Conclusion

Implemented focused code change to make verified non-local/relocation transitions searchable in entity_options planning. All unit tests pass. The frozen 5/7 baseline is preserved. Level 6 remains unverified pending checkpoint replay with API access.

This addresses the exact failure mode documented in EXPERIMENT_REPORT_20261020_LEVEL6_BROAD_SAMPLING.md without relaxing state keys (rejected per EXPERIMENT_REPORT_20261104_STATE_KEY_ABLATION.md) and without hard-coding coordinates.
