# State-Key Ablation Audit: Handoff & Next Steps

**Date**: 2026-11-04  
**Branch**: `cursor/state-key-ablation-audit-c7ae`  
**PR**: https://github.com/songhongzheng3-cmyk/ARC-AGI-3-Agents/pull/1  
**Status**: ✅ Tooling complete and tested | ⏸️ Execution awaiting recording

---

## What Was Accomplished

Implemented the **2026-10-20 planned audit** from `COMPETITION_EXECUTION_PLAN.md`: offline analysis of state-key tradeoffs (exact vs relaxed) for level-6 non-local and blocked moves.

### Deliverables

1. **Audit Script** (`scripts/audit_state_key_ablation.py`)
   - Per-field ablation analysis with temporal train/test split
   - Automated GO/NO-GO recommendation
   - Explicit rejection criterion: same-key different-successor → REJECT

2. **Checkpoint Verification Framework** (`scripts/checkpoint_single_step_verification.py`)
   - Default-OFF infrastructure (requires `STRUCTURED_RELAXED_STATE_KEYS=1`)
   - Ready for use if audit returns GO
   - Single-step verification before any multi-step experiments

3. **Test Suite** (`tests/unit/test_state_key_ablation.py`)
   - 14 comprehensive unit tests
   - Synthetic data validates audit logic independently
   - All 419 tests pass (405 existing + 14 new)

4. **Experiment Report** (`docs/EXPERIMENT_REPORT_20261104_STATE_KEY_ABLATION.md`)
   - Complete methodology and decision criteria
   - Usage instructions and reproducibility notes
   - Conditional next steps (GO vs NO-GO paths)

### Validation

✅ **419/419 tests pass**  
✅ **Ruff linting passes**  
✅ **`git diff --check` passes**  
✅ **PR created** against `my-first-agent`

---

## How the Audit Works

### Scope
- **Only non-local and blocked moves** (level-6 problem cases)
- **Temporal split**: 70% training, 30% leave-out testing
- **Field sets tested**: base (status_signature, status_color) + optional fields (entity_signatures, resource_level, walkable/blocked cells, static objects)

### Decision Criteria

**GO** (proceed to checkpoint verification):
- At least one viable field set (not rejected)
- Coverage ≥ 20% in leave-out
- Error rate = 0% in leave-out
- Zero conflicts in leave-out

**NO-GO** (do not relax state keys):
- All field sets rejected (same-key different-successor exists)
- Best viable coverage < 20%
- Any errors in leave-out predictions

### Safety Constraints
- Default-OFF only (explicit environment variable required)
- Old 5/7 baseline preserved and independently runnable
- Single-step verification before multi-step
- Abort on first prediction mismatch
- No blind 700-step A/B without evidence

---

## What's Blocking Execution

**No recording available.** The audit needs a recording from running StructuredAgent on ls20 with level-6 attempts. This requires either:

1. **API credentials**: Set `ARC_API_KEY` in `.env` and run:
   ```bash
   export STRUCTURED_MAX_ACTIONS=700
   export STRUCTURED_JOINT_NAVIGATION=1
   export STRUCTURED_ENTITY_OPTIONS=1
   uv run main.py --agent=structuredagent --game=ls20
   ```

2. **Existing recordings**: If recordings like "round12", "round16", or "row 620 checkpoint" exist elsewhere, they can be used directly.

3. **Local environment setup**: Use local game execution per README.md (requires environment files).

---

## How to Execute the Audit (When Recording Available)

### 1. Generate or Locate Recording

If generating new recording:
```bash
# Configure environment
export ARC_API_KEY="your_key"
export STRUCTURED_MAX_ACTIONS=700
export STRUCTURED_ENTITY_OPTIONS=1

# Run agent
uv run main.py --agent=structuredagent --game=ls20

# Recording saved to: recordings/ls20.structuredagent.<guid>.recording.jsonl
```

### 2. Run Audit

```bash
uv run python scripts/audit_state_key_ablation.py \
  recordings/ls20.structuredagent.<guid>.recording.jsonl \
  --output sampling_runs/state_key_ablation/audit_report.json
```

Output includes:
- Per-field-set coverage, conflict rate, error rate
- Rejection status and reasons  
- **GO/NO-GO recommendation** with detailed reasoning
- First 10 error cases for diagnosis (if any)

### 3. Review Decision

Check `sampling_runs/state_key_ablation/audit_report.json`:
```json
{
  "recommendation": {
    "decision": "GO" | "NO-GO",
    "reason": "...",
    // ... detailed metrics
  }
}
```

---

## Next Steps (Conditional on Audit Result)

### If GO Decision

1. **Update checkpoint verification script**:
   - Implement `RelaxedStateKeyPredictor.observe()` and `.predict()` using recommended field set from audit
   - See TODO comments in `scripts/checkpoint_single_step_verification.py`

2. **Run single-step checkpoint verification**:
   ```bash
   export STRUCTURED_RELAXED_STATE_KEYS=1
   uv run python scripts/checkpoint_single_step_verification.py \
     recordings/baseline_recording.jsonl \
     --checkpoint-row=620 \
     --audit-report=sampling_runs/state_key_ablation/audit_report.json \
     --output=sampling_runs/checkpoint_verification/single_step.json
   ```

3. **Only if single-step MATCH**:
   - Extend to multi-step (bounded, e.g., 10 actions)
   - Document any mismatches and position predictions
   
4. **Only if multi-step succeeds**:
   - Attempt full level-6 clear from checkpoint
   - Compare predicted vs observed for transparency

5. **Only if level-6 clears from checkpoint**:
   - Run full from-scratch 700-action A/B test
   - Compare with frozen 5/7 baseline

### If NO-GO Decision

1. **Document missing evidence**:
   - What state information is missing to distinguish outcomes?
   - Are there patterns in the error cases?
   - What would be needed to make a safe migration?

2. **Consider alternative approaches**:
   - Mechanism-based planning (contact effects, entity interactions)
   - Explicit non-local transition learning with causal attribution
   - Accept that level 6 requires capabilities beyond state-key lookup

3. **Do NOT**:
   - Expand sampling budget without clear hypothesis
   - Relax rejection criteria to force a GO
   - Implement speculative features without evidence

---

## Recommended Single Next Step

**Priority 1**: Generate recording or locate existing recordings

- **If API key available**: Run StructuredAgent on ls20 with 700 actions
- **If recordings exist elsewhere**: Copy them to `recordings/` directory  
- **If neither**: Document this blocker and request access

**Why this is the blocker**: All subsequent work (audit execution, GO/NO-GO decision, checkpoint verification) depends on having recording data to analyze.

---

## Files to Review

| File | Purpose |
|------|---------|
| `docs/EXPERIMENT_REPORT_20261104_STATE_KEY_ABLATION.md` | Complete methodology and decision criteria |
| `scripts/audit_state_key_ablation.py` | Main audit implementation (435 lines) |
| `scripts/checkpoint_single_step_verification.py` | Checkpoint verification framework (placeholder predictor) |
| `tests/unit/test_state_key_ablation.py` | Comprehensive test coverage (14 tests) |
| `docs/COMPETITION_EXECUTION_PLAN.md` | Original 2026-10-20 audit requirement |

---

## Evidence of Narrow Scope

Per task requirements, this work maintains **narrow focus**:

✅ **Only non-local/blocked moves** (not all transitions)  
✅ **Offline audit** (no live game calls)  
✅ **Evidence-driven decision** (explicit rejection criterion)  
✅ **Default-OFF capability** (requires explicit opt-in)  
✅ **Incremental validation** (single-step before multi-step)  
✅ **Preserved baseline** (5/7 frozen, independently runnable)  
✅ **No blind experimentation** (each step requires evidence from prior)

Did NOT:
- ❌ Expand action budgets
- ❌ Re-run blind 700-step A/B
- ❌ Claim full level-6 clear
- ❌ Upload/submit Kaggle
- ❌ Implement new mechanisms without evidence

---

## Questions & Clarifications

### Q: Why not just run the agent to generate a recording?
**A**: No API credentials available in current cloud environment, and local execution requires environment files or API key. This is documented as the blocking step.

### Q: Can we use synthetic data for the audit?
**A**: No. The audit must use real recording data with level-6 non-local/blocked moves to make a valid GO/NO-GO decision. Synthetic data is only used for unit tests.

### Q: What if the audit returns NO-GO?
**A**: Document the exact missing evidence needed. Do not proceed with relaxing state keys. Consider mechanism-based alternatives per COMPETITION_EXECUTION_PLAN.md guidance.

### Q: Is the checkpoint verification complete?
**A**: The infrastructure is ready (script skeleton, environment variable, safety checks), but the predictor implementation (TODO comments) waits on GO decision from audit. This is intentional: don't implement until evidence supports it.

---

## Summary for Next Developer

**Done**: Audit tooling, tests, documentation, PR created  
**Blocking**: Need recording with level-6 attempts  
**Next**: Generate/locate recording → run audit → follow GO or NO-GO path  
**Critical**: Maintain narrow scope, default-OFF, incremental validation  

All tooling is ready. The audit will make an evidence-based recommendation. Follow that recommendation exactly—do not force a GO if evidence says NO-GO.

**PR**: https://github.com/songhongzheng3-cmyk/ARC-AGI-3-Agents/pull/1  
**Branch**: `cursor/state-key-ablation-audit-c7ae`  
**Tests**: 419/419 passing  
**Contact**: See EXPERIMENT_REPORT_20261104_STATE_KEY_ABLATION.md for full details
