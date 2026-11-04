# State-Key Ablation Audit: Non-Local and Blocked Move Analysis

2026-11-04. Public `ls20-9607627b`, self-written `StructuredAgent`. Offline audit tooling for state-key tradeoff analysis as planned in 2026-10-20 update to COMPETITION_EXECUTION_PLAN.md.

## Status

**Audit tooling implemented and tested.** Recording generation and execution pending API credentials or local game environment setup.

- ✅ New audit script `scripts/audit_state_key_ablation.py` with comprehensive per-field ablation analysis
- ✅ 14 new unit tests covering all audit functions; **419/419 tests pass**
- ✅ Temporal train/test split with explicit rejection criterion
- ⏸️ Awaiting recording generation to execute audit and make GO/NO-GO decision
- ⏸️ Default-OFF checkpoint verification infrastructure designed but not implemented (conditional on GO decision)

## Methodology

### Audit Scope

Per COMPETITION_EXECUTION_PLAN.md 2026-10-20 update, this audit focuses exclusively on:

1. **Non-local moves**: Player moves more than 1 cell away (distance > 1)
2. **Blocked moves**: Player attempts move but stays in same position (no-op)

These are the problematic cases where level 6 experiences position prediction failures. Expected adjacent moves are excluded from this analysis.

### Field-Ablation Approach

The audit evaluates multiple candidate state-key field sets using temporal train/test split:

**Base fields** (always included):
- `status_signature` (task pattern)
- `status_color` (task color)

**Optional fields** (ablated individually and in combinations):
- `resource_level`: Current resource value
- `entity_signatures`: Tracked entity positions and motion (signature, previous, current)
- `walkable_cells`: Traversable grid cells
- `blocked_cells`: Impassable grid cells
- `static_objects`: Non-moving visible objects (color, area, shape, cell)

### Evaluation Metrics

For each field set, measured on the leave-out (test) set:

1. **Coverage**: Proportion of test samples that match a training-seen state key
2. **Conflict rate**: Training samples where same key → multiple outcomes
3. **Error rate**: Test samples where predicted outcome ≠ observed outcome
4. **Rejection criterion**: Any test sample with same-key different-successor → REJECT

### GO/NO-GO Decision Criteria

**GO** (proceed to default-OFF checkpoint verification):
- At least one viable field set (not rejected)
- Coverage ≥ 20% in leave-out
- Error rate = 0% in leave-out
- Zero conflicts in leave-out

**NO-GO** (do not relax state keys):
- All field sets rejected (same-key different-successor exists)
- Best viable coverage < 20%
- Any errors in leave-out predictions

## Implementation

### New Audit Script

`scripts/audit_state_key_ablation.py` implements:

1. **Sample extraction**: Filters recording for non-local and blocked moves only
2. **Temporal split**: 70% training, 30% leave-out testing
3. **Field set evaluation**: Tests base + each optional field, progressive combinations, and single-field ablations
4. **Recommendation**: Automated GO/NO-GO decision with detailed reasoning

Key functions:
- `extract_samples(recording)`: Parse recording, classify moves, extract eligible transitions
- `build_state_key(sample, field_set)`: Construct state key from specified fields
- `evaluate_field_set(samples, field_set)`: Temporal holdout evaluation with rejection check
- `ablation_analysis(samples)`: Comprehensive field-set comparison
- `make_recommendation(analysis)`: GO/NO-GO decision logic

### Test Coverage

14 new unit tests in `tests/unit/test_state_key_ablation.py`:

- Movement classification (expected_adjacent, blocked, nonlocal, unexpected_adjacent)
- State-key construction with various field combinations
- Temporal evaluation with zero conflicts, with conflicts, insufficient coverage
- GO recommendation (viable field set, good coverage, zero errors)
- NO-GO recommendations (all rejected, low coverage, has errors)

All tests use synthetic samples to verify audit logic independently of real recordings.

## Usage

### Prerequisites

1. **Recording with level-6 attempts**: Run StructuredAgent on ls20 to generate recording:

```bash
# Set environment variables
export ARC_API_KEY="your_api_key"
export STRUCTURED_MAX_ACTIONS=700
export STRUCTURED_JOINT_NAVIGATION=1
export STRUCTURED_ENTITY_OPTIONS=1
# Other feature flags as needed

# Run agent
uv run main.py --agent=structuredagent --game=ls20

# Recording saved to recordings/ls20.structuredagent.<guid>.recording.jsonl
```

2. **Identify recording path**: Locate the generated `.recording.jsonl` file

### Run Audit

```bash
uv run python scripts/audit_state_key_ablation.py \
  recordings/ls20.structuredagent.<guid>.recording.jsonl \
  --output sampling_runs/state_key_ablation/audit_report.json
```

Output includes:
- Per-field-set coverage, conflict rate, error rate
- Rejection status and reasons
- GO/NO-GO recommendation with detailed reasoning
- Error details for diagnosis

### Interpret Results

**If GO**: Proceed to implement default-OFF checkpoint verification (see next section)

**If NO-GO**: Document exact missing evidence:
- If all rejected: Same state keys lead to different outcomes; need different state representation or mechanism understanding
- If low coverage: Too few repeated state keys in leave-out; either need more samples or accept that level 6 requires mechanism-based planning beyond state lookup
- If errors exist: Current fields insufficient to distinguish outcomes; need additional observable state

## Next Steps (Conditional on GO)

### If Audit Returns GO

1. **Implement default-OFF checkpoint verification**:
   - Add `STRUCTURED_RELAXED_STATE_KEYS` environment flag (default=0)
   - Create `RelaxedStateKeyModel` using recommended field set from audit
   - Integrate into `entity_options.py` as alternative to exact atomic samples
   - Only predict for first new action after checkpoint; abort on first mismatch

2. **Checkpoint verification experiment**:
   ```bash
   # Replay from row 620 checkpoint with relaxed keys enabled
   export STRUCTURED_RELAXED_STATE_KEYS=1
   uv run python scripts/checkpoint_verification_experiment.py \
     recordings/baseline_recording.jsonl \
     --checkpoint-row=620 \
     --max-new-actions=1 \
     --output=sampling_runs/checkpoint_verification/single_step.json
   ```

3. **Success criteria for single-step verification**:
   - Predicted position matches observed position for first new action
   - Stop immediately after first action (do not continue to completion)
   - Document predicted vs observed for transparency

4. **Only after single-step success**:
   - Extend to multi-step checkpoint verification (bounded, e.g., 10 actions)
   - If multi-step succeeds, attempt full level-6 clear from checkpoint
   - If level-6 clears from checkpoint, run full from-scratch 700-action A/B test

### If Audit Returns NO-GO

**Do not implement relaxed state keys.** Instead:

1. **Document blocking evidence gap**: What state information is missing that would be needed to distinguish outcomes?

2. **Consider alternative approaches**:
   - Mechanism-based planning (contact effects, entity interactions) instead of state-key lookup
   - Explicit non-local transition learning with causal attribution
   - Accept that level 6 requires capabilities beyond current agent

3. **Do not**:
   - Expand sampling budget without clear hypothesis
   - Relax rejection criteria to force a GO
   - Implement speculative features without evidence

## Safety Constraints

- **Default-OFF only**: New capability must be opt-in via explicit environment variable
- **Old baseline preserved**: 5/7 frozen baseline remains unchanged and independently runnable
- **Single-step verification first**: Do not skip to full level-6 clear without incremental validation
- **Abort on first mismatch**: Prediction errors immediately halt verification
- **No blind 700-step A/B**: Full from-scratch comparison only after checkpoint clear + level-6 complete

## Reproducibility

### Determinism

- Temporal split: First 70% train, last 30% test (no random shuffle)
- Field evaluation: Deterministic iteration order over field sets
- Rejection: Any conflict or error in leave-out → REJECT

### Versioning

- Script: `scripts/audit_state_key_ablation.py` (this commit)
- Tests: `tests/unit/test_state_key_ablation.py` (this commit)
- Recording: Must specify exact path and SHA256 (included in report output)

### Environment

- Python 3.12.3
- Dependencies: `uv.lock` (commit-locked)
- All unit tests pass: `uv run pytest tests/unit/ -v`

## Open Questions

1. **Recording availability**: Baseline ls20 recording with level-6 attempts needed. If round12/round16 recordings exist elsewhere, they can be used directly.

2. **Checkpoint row selection**: Experiment reports reference "row 620" for level-6 work. Audit works with any recording; checkpoint experiments need to specify exact resumption point.

3. **Field set granularity**: Current implementation tests base + individual optional fields + progressive combinations + single-field ablations. If GO, can refine to test all 2^N combinations if needed (currently avoided for interpretability).

4. **Local vs API execution**: Audit is offline (no live game calls). Recording generation requires API key or local environment setup per README.md.

## Files Changed

- `scripts/audit_state_key_ablation.py` (new, 435 lines)
- `tests/unit/test_state_key_ablation.py` (new, 14 tests)
- `docs/EXPERIMENT_REPORT_20261104_STATE_KEY_ABLATION.md` (this file)

**Unit tests**: 419/419 passed (405 existing + 14 new)

**Linting**: `uv run ruff check agents scripts tests/unit` passes

**Git**: No secrets committed, `.gitignore` honored

## Conclusion

Audit tooling is complete and tested. **Execution awaits recording generation.** Once a recording with level-6 non-local/blocked moves is available, run the audit to obtain GO/NO-GO recommendation. If GO, implement default-OFF checkpoint verification as specified above. If NO-GO, document missing evidence and consider alternative approaches per safety constraints.

This audit directly addresses the 2026-10-20 planned next step: offline audit of exact vs relaxed state-key tradeoff with explicit leave-out evaluation and rejection criteria. Implementation maintains narrow scope (non-local/blocked only), evidence-driven decision (no weak-evidence defaults), and incremental validation (single-step before multi-step).
