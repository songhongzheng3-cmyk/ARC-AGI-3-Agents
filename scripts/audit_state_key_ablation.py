#!/usr/bin/env python3
"""State-key ablation audit for non-local and blocked moves.

Evaluate the tradeoff between exact full-action keys (sparse coverage) and
relaxed state keys (over-generalization risk). For each candidate field set,
measure leave-out prediction accuracy, conflict rate, and coverage. Reject
any key where the leave-out set contains same-key different-successor evidence.

This implements the 2026-10-20 planned audit from COMPETITION_EXECUTION_PLAN.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mechanism_state_tracker import (  # noqa: E402
    ACTION_DELTAS,
    MechanismStateTracker,
)
from agents.motion_context import MotionContextMemory  # noqa: E402
from scripts.shadow_replay_structured import action_name, load_frames  # noqa: E402


@dataclass
class TransitionSample:
    """A single state transition with all observable fields."""

    row: int
    level: int
    player_position: tuple[int, int]
    status_signature: str
    status_color: int | None
    resource_level: int
    action: str
    # Motion context: tracked entities with previous/current positions
    entity_signatures: tuple[tuple[str, tuple[int, int] | None, tuple[int, int]], ...]
    # Static visible scene (walkable, blocked, non-moving objects)
    walkable_cells: frozenset[tuple[int, int]]
    blocked_cells: frozenset[tuple[int, int]]
    static_objects: tuple[
        tuple[int, tuple[int, ...], str, tuple[int, int]], ...
    ]  # (color, area, shape_sig, cell)
    # Outcome
    next_player_position: tuple[int, int]
    next_status_signature: str
    next_status_color: int | None
    # Classification
    movement_class: str  # expected_adjacent, blocked_or_noop, nonlocal, unexpected_adjacent


def stable_scene(state, context):
    """Extract static scene components (non-moving objects)."""
    moving_signatures = {item.signature for item in context}
    static_objects = tuple(
        sorted(
            (item.color, item.area, item.shape_signature, item.cell)
            for item in state.objects
            if (item.color, item.area, item.shape_signature) not in moving_signatures
            and item.object_id not in state.player_component_ids
        )
    )
    return (
        state.walkable_cells,
        state.blocked_cells,
        static_objects,
    )


def classify_movement(before_pos, after_pos, action):
    """Classify movement type for this transition."""
    delta = ACTION_DELTAS.get(action)
    if delta is None:
        return "unclassified_action"

    expected = (before_pos[0] + delta[0], before_pos[1] + delta[1])
    distance = abs(after_pos[0] - before_pos[0]) + abs(after_pos[1] - before_pos[1])

    if after_pos == expected:
        return "expected_adjacent"
    elif after_pos == before_pos:
        return "blocked_or_noop"
    elif distance > 1:
        return "nonlocal"
    else:
        return "unexpected_adjacent"


def extract_samples(recording: Path) -> list[TransitionSample]:
    """Extract all eligible non-local and blocked transitions from recording."""
    frames = load_frames(recording)
    tracker = MechanismStateTracker()
    memory = MotionContextMemory()

    samples = []
    states = []
    contexts = []

    for frame in frames:
        state = tracker.observe(frame, action=action_name(frame))
        states.append(state)
        contexts.append(memory.observe(state, reset=action_name(frame) == "RESET"))

    for row, (before, after) in enumerate(zip(states, states[1:])):
        action = action_name(frames[row + 1])

        # Apply same filters as analyze_navigation_context.py
        if (
            before.level != after.level
            or action == "RESET"
            or before.player_position is None
            or after.player_position is None
            or min(before.player_confidence, after.player_confidence) < 0.8
            or (before.resource_capacity > 0 and before.resource_level == 0)
        ):
            continue

        movement_class = classify_movement(
            before.player_position, after.player_position, action
        )

        # Only include non-local and blocked moves (the focus of the audit)
        if movement_class not in ("nonlocal", "blocked_or_noop"):
            continue

        context_before = contexts[row]
        scene = stable_scene(before, context_before)
        entity_sigs = tuple(
            (item.signature, item.previous, item.current) for item in context_before
        )

        sample = TransitionSample(
            row=row,
            level=before.level,
            player_position=before.player_position,
            status_signature=before.status_signature,
            status_color=before.status_color,
            resource_level=before.resource_level,
            action=action,
            entity_signatures=entity_sigs,
            walkable_cells=scene[0],
            blocked_cells=scene[1],
            static_objects=scene[2],
            next_player_position=after.player_position,
            next_status_signature=after.status_signature,
            next_status_color=after.status_color,
            movement_class=movement_class,
        )
        samples.append(sample)

    return samples


def build_state_key(sample: TransitionSample, field_set: set[str]) -> tuple:
    """Build a state key including only the specified fields."""
    key_parts = [sample.level, sample.player_position, sample.action]

    if "status_signature" in field_set:
        key_parts.append(sample.status_signature)
    if "status_color" in field_set:
        key_parts.append(sample.status_color)
    if "resource_level" in field_set:
        key_parts.append(sample.resource_level)
    if "entity_signatures" in field_set:
        key_parts.append(sample.entity_signatures)
    if "walkable_cells" in field_set:
        key_parts.append(sample.walkable_cells)
    if "blocked_cells" in field_set:
        key_parts.append(sample.blocked_cells)
    if "static_objects" in field_set:
        key_parts.append(sample.static_objects)

    return tuple(key_parts)


def evaluate_field_set(
    samples: list[TransitionSample], field_set: set[str], split_fraction: float = 0.7
) -> dict[str, Any]:
    """Evaluate prediction performance with given field set using temporal holdout.

    Returns metrics including coverage, conflicts, and whether this field set
    passes the "same-key different-successor" rejection criterion.
    """
    # Time-split: train on first split_fraction, test on remainder
    split_index = int(len(samples) * split_fraction)
    training = samples[:split_index]
    testing = samples[split_index:]

    # Build evidence table from training data
    evidence: dict[tuple, Counter] = defaultdict(Counter)
    for sample in training:
        key = build_state_key(sample, field_set)
        outcome = (
            sample.next_player_position,
            sample.next_status_signature,
            sample.next_status_color,
        )
        evidence[key][outcome] += 1

    # Evaluate on test set
    checked = 0
    errors = 0
    unseen = 0
    ambiguous = 0
    error_details = []

    for sample in testing:
        key = build_state_key(sample, field_set)
        outcome = (
            sample.next_player_position,
            sample.next_status_signature,
            sample.next_status_color,
        )

        prior = evidence.get(key)
        if not prior:
            unseen += 1
        elif len(prior) != 1:
            ambiguous += 1
        else:
            checked += 1
            if outcome not in prior:
                errors += 1
                error_details.append(
                    {
                        "row": sample.row,
                        "level": sample.level,
                        "player": sample.player_position,
                        "action": sample.action,
                        "expected": list(prior.keys())[0],
                        "observed": outcome,
                        "movement_class": sample.movement_class,
                    }
                )

    # Count conflicts in training data (same key, multiple outcomes)
    training_conflicts = sum(1 for outcomes in evidence.values() if len(outcomes) > 1)
    training_keys = len(evidence)

    # Rejection criterion: any conflict in leave-out means REJECT
    rejected = ambiguous > 0 or errors > 0

    return {
        "field_set": sorted(field_set),
        "training_samples": len(training),
        "testing_samples": len(testing),
        "training_distinct_keys": training_keys,
        "training_conflicts": training_conflicts,
        "training_conflict_rate": (
            training_conflicts / training_keys if training_keys > 0 else 0.0
        ),
        "prediction_checks": checked,
        "prediction_errors": errors,
        "unseen_keys": unseen,
        "ambiguous_keys": ambiguous,
        "coverage": checked / len(testing) if testing else 0.0,
        "error_rate": errors / checked if checked > 0 else 0.0,
        "rejected": rejected,
        "rejection_reason": (
            "same-key different-successor in leave-out" if rejected else None
        ),
        "error_details": error_details[:10],  # First 10 errors for diagnosis
    }


def ablation_analysis(samples: list[TransitionSample]) -> dict[str, Any]:
    """Perform comprehensive field ablation analysis.

    Test multiple field combinations to understand which fields contribute to
    prediction accuracy without causing over-generalization.
    """
    # Define base fields and optional fields
    base_fields = {"status_signature", "status_color"}

    optional_fields = [
        "resource_level",
        "entity_signatures",
        "walkable_cells",
        "blocked_cells",
        "static_objects",
    ]

    results = []

    # Test base only
    results.append(evaluate_field_set(samples, base_fields))

    # Test base + each optional field individually
    for field in optional_fields:
        field_set = base_fields | {field}
        results.append(evaluate_field_set(samples, field_set))

    # Test progressive combinations (matching the context hierarchy)
    progressive = [
        base_fields,
        base_fields | {"entity_signatures"},
        base_fields | {"entity_signatures", "static_objects"},
        base_fields | {"entity_signatures", "static_objects", "walkable_cells"},
        base_fields
        | {"entity_signatures", "static_objects", "walkable_cells", "blocked_cells"},
        base_fields | set(optional_fields),  # All fields
    ]

    for field_set in progressive:
        if field_set not in [r["field_set"] for r in results]:
            results.append(evaluate_field_set(samples, set(field_set)))

    # Single-field ablation: remove one field at a time from the full set
    full_set = base_fields | set(optional_fields)
    for field in optional_fields:
        ablated_set = full_set - {field}
        results.append(evaluate_field_set(samples, ablated_set))

    return {
        "total_samples": len(samples),
        "movement_class_distribution": dict(
            Counter(s.movement_class for s in samples)
        ),
        "level_distribution": dict(Counter(s.level for s in samples)),
        "field_set_evaluations": results,
    }


def make_recommendation(analysis: dict[str, Any]) -> dict[str, Any]:
    """Make GO/NO-GO recommendation based on analysis results.

    GO criteria:
    - At least one field set has zero conflicts in leave-out
    - Coverage is meaningfully better than exact full keys (>20%)
    - Error rate is zero in leave-out

    NO-GO triggers:
    - All field sets have conflicts or errors in leave-out
    - Coverage improvement is marginal (<20%)
    """
    evaluations = analysis["field_set_evaluations"]

    # Find viable candidates (not rejected)
    viable = [e for e in evaluations if not e["rejected"]]

    if not viable:
        return {
            "decision": "NO-GO",
            "reason": "All field sets show same-key different-successor in leave-out",
            "recommendation": "Do not relax state keys. Need more evidence: either more varied samples or different state representation.",
            "next_step": "Document the exact missing evidence needed. Do not expand sampling without clear hypothesis.",
        }

    # Sort by coverage (descending)
    viable_sorted = sorted(viable, key=lambda x: x["coverage"], reverse=True)
    best = viable_sorted[0]

    # Check if coverage is meaningfully better than zero (threshold: 20%)
    if best["coverage"] < 0.20:
        return {
            "decision": "NO-GO",
            "reason": f"Best viable field set has only {best['coverage']:.1%} coverage in leave-out",
            "best_candidate": best["field_set"],
            "recommendation": "Coverage too low for safe migration. Exact keys remain too sparse, but relaxed keys haven't proven safe.",
            "next_step": "Need more samples showing same relaxed state with consistent outcomes, OR accept that level 6 requires mechanism understanding beyond state-key lookup.",
        }

    # Check error rate
    if best["error_rate"] > 0:
        return {
            "decision": "NO-GO",
            "reason": f"Best candidate has {best['error_rate']:.1%} error rate in leave-out",
            "best_candidate": best["field_set"],
            "errors": best["prediction_errors"],
            "recommendation": "Errors exist even in viable field sets. Not safe for migration.",
            "next_step": "Inspect error cases to understand what state information is missing.",
        }

    # We have a viable candidate with good coverage and zero errors
    return {
        "decision": "GO",
        "reason": f"Field set {best['field_set']} achieves {best['coverage']:.1%} coverage with zero conflicts and errors in leave-out",
        "recommended_field_set": best["field_set"],
        "coverage": best["coverage"],
        "training_samples": best["training_samples"],
        "testing_samples": best["testing_samples"],
        "next_step": "Implement default-OFF single-step checkpoint verification using this field set for level 6",
        "safety_note": "Keep old 5/7 baseline intact. Only verify first new action after checkpoint, abort on first prediction mismatch.",
    }


def analyze_recording(recording: Path) -> dict[str, Any]:
    """Main analysis pipeline."""
    samples = extract_samples(recording)

    if len(samples) == 0:
        return {
            "error": "No eligible non-local or blocked transitions found in recording",
            "recording": str(recording.resolve()),
        }

    analysis = ablation_analysis(samples)
    recommendation = make_recommendation(analysis)

    return {
        "schema_version": 1,
        "recording": str(recording.resolve()),
        "recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "focus": "non-local and blocked moves only",
        "methodology": "Temporal train/test split with per-field ablation analysis",
        "rejection_criterion": "same-key different-successor in leave-out",
        "analysis": analysis,
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="Path to recording file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    if not args.recording.exists():
        print(f"Error: Recording not found: {args.recording}", file=sys.stderr)
        sys.exit(1)

    report = analyze_recording(args.recording)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    # Print summary to stdout
    if "recommendation" in report:
        rec = report["recommendation"]
        print(json.dumps({"decision": rec["decision"], "reason": rec["reason"]}))
        print(f"\nFull report written to: {args.output}")
    else:
        print(json.dumps(report))


if __name__ == "__main__":
    main()
