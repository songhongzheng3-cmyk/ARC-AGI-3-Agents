#!/usr/bin/env python3
"""Compare navigation state abstractions in chronological recording replay.

Predict from earlier observations only, then record the actual outcome. A
conflicting key abstains thereafter. Coverage is reported alongside errors:
splitting every sample into its own key must not look like a perfect model.
This is a prediction audit, not a counterfactual policy/score evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Hashable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mechanism_state_tracker import (  # noqa: E402
    ACTION_DELTAS,
    MechanismStateTracker,
)
from agents.motion_context import MotionContextMemory  # noqa: E402
from scripts.shadow_replay_structured import action_name, load_frames  # noqa: E402


def audit(samples: list[tuple[int, Hashable, Hashable]]) -> dict[str, Any]:
    evidence: dict[Hashable, Counter] = defaultdict(Counter)
    checks = errors = conflicts = abstentions = 0
    error_rows = []
    for row, key, outcome in samples:
        prior = evidence[key]
        if len(prior) == 1:
            checks += 1
            if outcome not in prior:
                errors += 1
                error_rows.append(row)
        elif prior:
            abstentions += 1
        if prior and outcome not in prior:
            conflicts += 1
        prior[outcome] += 1
    return {
        "samples": len(samples),
        "distinct_keys": len(evidence),
        "prediction_checks": checks,
        "prediction_errors": errors,
        "error_rows": error_rows,
        "new_conflicting_outcomes": conflicts,
        "ambiguous_key_abstentions": abstentions,
        "unseen_keys": len(samples) - checks - abstentions,
        "coverage": round(checks / len(samples), 6) if samples else 0.0,
    }


def temporal_holdout(
    samples: list[tuple[int, Hashable, Hashable]], split_row: int
) -> dict[str, Any]:
    """Fit context outcomes before split_row and evaluate the frozen table after."""
    training = [sample for sample in samples if sample[0] < split_row]
    testing = [sample for sample in samples if sample[0] >= split_row]
    evidence: dict[Hashable, Counter] = defaultdict(Counter)
    for _, key, outcome in training:
        evidence[key][outcome] += 1

    checked = errors = unseen = ambiguous = 0
    error_rows = []
    for row, key, outcome in testing:
        prior = evidence.get(key)
        if not prior:
            unseen += 1
        elif len(prior) != 1:
            ambiguous += 1
        else:
            checked += 1
            if outcome not in prior:
                errors += 1
                error_rows.append(row)
    return {
        "split_row": split_row,
        "training_samples": len(training),
        "test_samples": len(testing),
        "prediction_checks": checked,
        "prediction_errors": errors,
        "error_rows": error_rows,
        "unseen_keys": unseen,
        "ambiguous_training_keys": ambiguous,
        "coverage": round(checked / len(testing), 6) if testing else 0.0,
    }


def repeated_noop_audit(frames, states, contexts, *, level: int) -> dict[str, Any]:
    """Find repeated costly no-ops under an exact visible-scene/action key.

    This is retrospective evidence only: it does not claim what an alternate
    action would have achieved. Resource is excluded from the key because a
    no-op may consume it; the board, task status, and tracked object positions
    must remain unchanged for a prior failure to apply.
    """
    failed: dict[Hashable, int] = {}
    failed_contexts: dict[Hashable, set[Hashable]] = defaultdict(set)
    repeat_attempts = []
    reopen_after_context_change = 0
    costly_noops = 0
    eligible = 0

    def stable_scene(state, context):
        moving_signatures = {item.signature for item in context}
        static_objects = tuple(
            sorted(
                (item.color, item.area, item.shape_signature, item.cell)
                for item in state.objects
                if (item.color, item.area, item.shape_signature)
                not in moving_signatures
                and item.object_id not in state.player_component_ids
            )
        )
        return (
            state.walkable_cells,
            state.blocked_cells,
            static_objects,
        )

    for row, (before, after) in enumerate(zip(states, states[1:])):
        action = action_name(frames[row + 1])
        if (
            before.level != level
            or before.level != after.level
            or action == "RESET"
            or before.player_position is None
            or after.player_position is None
            or min(before.player_confidence, after.player_confidence) < 0.8
            or (before.resource_capacity > 0 and before.resource_level == 0)
        ):
            continue

        context_before = contexts[row]
        context_after = contexts[row + 1]
        stable_before = stable_scene(before, context_before)
        stable_after = stable_scene(after, context_after)
        dynamic = tuple(
            (item.signature, item.previous, item.current) for item in context_before
        )
        key = (
            before.level,
            before.player_position,
            before.status_signature,
            before.status_color,
            stable_before,
            dynamic,
            action,
        )
        scene_key = key[:5] + (action,)
        if scene_key in failed_contexts and dynamic not in failed_contexts[scene_key]:
            reopen_after_context_change += 1
        eligible += 1
        previous_failure = failed.get(key)
        moved_or_changed = (
            before.player_position != after.player_position
            or before.status_signature != after.status_signature
            or before.status_color != after.status_color
            or stable_before != stable_after
        )
        if previous_failure is not None:
            repeat_attempts.append(
                {
                    "prior_noop_row": previous_failure,
                    "repeat_row": row,
                    "action": action,
                    "repeat_outcome": "changed" if moved_or_changed else "noop",
                    "resource_delta": after.resource_level - before.resource_level,
                }
            )
        if not moved_or_changed and after.resource_level < before.resource_level:
            costly_noops += 1
            failed[key] = row
            failed_contexts[scene_key].add(dynamic)
        elif moved_or_changed:
            failed.pop(key, None)

    return {
        "eligible_transitions": eligible,
        "costly_noop_transitions": costly_noops,
        "exact_context_repeated_attempts": len(repeat_attempts),
        "repeated_attempts": repeat_attempts,
        "repeated_attempts_that_changed_state": sum(
            item["repeat_outcome"] == "changed" for item in repeat_attempts
        ),
        "reopened_after_dynamic_context_change": reopen_after_context_change,
        "limitation": "Factual replay only; suppressed alternatives and score impact are unobserved.",
    }


def analyze(recording: Path) -> dict[str, Any]:
    frames = load_frames(recording)
    tracker = MechanismStateTracker()
    memory = MotionContextMemory()
    states, contexts = [], []
    for frame in frames:
        state = tracker.observe(frame, action=action_name(frame))
        states.append(state)
        contexts.append(memory.observe(state, reset=action_name(frame) == "RESET"))

    variants: dict[str, list] = defaultdict(list)
    by_level: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    feedback: dict[int, Counter] = defaultdict(Counter)
    anomaly_feedback: dict[int, Counter] = defaultdict(Counter)
    anomalies: dict[int, list[dict[str, Any]]] = defaultdict(list)
    exact_frames: dict[tuple, dict[tuple, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row, (before, after) in enumerate(zip(states, states[1:])):
        action = action_name(frames[row + 1])
        if (
            before.level != after.level
            or action == "RESET"
            or before.player_position is None
            or after.player_position is None
            or min(before.player_confidence, after.player_confidence) < 0.8
            or (before.resource_capacity > 0 and before.resource_level == 0)
        ):
            continue
        base = (
            before.level,
            before.player_position,
            before.status_signature,
            before.status_color,
            action,
        )
        outcome = (after.player_position, after.status_signature, after.status_color)
        feedback[before.level].update(event.event_type for event in after.events)
        feedback[before.level]["eligible_transitions"] += 1
        context = contexts[row]
        delta = ACTION_DELTAS.get(action)
        expected = (
            None
            if delta is None
            else (before.player_position[0] + delta[0],
                  before.player_position[1] + delta[1])
        )
        distance = abs(after.player_position[0] - before.player_position[0]) + abs(
            after.player_position[1] - before.player_position[1]
        )
        if expected is None:
            movement_class = "unclassified_action"
        elif after.player_position == expected:
            movement_class = "expected_adjacent"
        elif after.player_position == before.player_position:
            movement_class = "blocked_or_noop"
        elif distance > 1:
            movement_class = "nonlocal"
        else:
            movement_class = "unexpected_adjacent"
        anomaly_feedback[before.level][movement_class] += 1
        if movement_class != "expected_adjacent":
            anomalies[before.level].append(
                {
                    "row": row,
                    "action": action,
                    "source": before.player_position,
                    "expected": expected,
                    "observed": after.player_position,
                    "movement_class": movement_class,
                    "source_resource": before.resource_level,
                    "source_status": before.status_signature,
                    "source_color": before.status_color,
                    "motion_context": [asdict(item) for item in context],
                    "events": [event.event_type for event in after.events],
                }
            )
        extras = {
            "base": (),
            "current_object_positions": tuple(
                (m.signature, m.current) for m in context
            ),
            "motion_context": context,
            "motion_context_resource": (context, before.resource_level),
        }
        for window in (1, 3, 5, 6):
            extras[f"recent_actions_{window}"] = tuple(
                action_name(frame)
                for frame in frames[max(0, row - window + 1) : row + 1]
            )
        for name, extra in extras.items():
            sample = (row, (base, extra), outcome)
            variants[name].append(sample)
            by_level[before.level][name].append(sample)
        grid = tuple(tuple(line) for line in frames[row].frame[-1])
        exact_frames[(before.level, grid, action)][outcome].append(row)

    aliases = []
    for (level, _, action), outcomes in exact_frames.items():
        if len(outcomes) < 2:
            continue
        groups = []
        for outcome, rows in outcomes.items():
            for row in rows:
                groups.append(
                    {
                        "row": row,
                        "outcome": outcome,
                        "source_resource": states[row].resource_level,
                        "motion_context": [asdict(m) for m in contexts[row]],
                        "recent_actions_6": [
                            action_name(f) for f in frames[max(0, row - 5) : row + 1]
                        ],
                    }
                )
        aliases.append({"level": level, "action": action, "occurrences": groups})
    split_row = int(len(frames) * 0.7)
    level_holdout = {}
    for level, data in by_level.items():
        sample_count = len(data["base"])
        level_split_index = int(sample_count * 0.7)
        level_split_row = (
            data["base"][level_split_index][0]
            if level_split_index < sample_count
            else len(frames)
        )
        level_holdout[str(level)] = {
            name: temporal_holdout(samples, level_split_row)
            for name, samples in data.items()
        }
    return {
        "schema_version": 1,
        "recording": str(recording.resolve()),
        "recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "rows": len(frames),
        "exclusions": "level changes, RESET, missing/low-confidence player, empty-meter source",
        "limitation": "Chronological factual prediction only; no score or counterfactual policy claim.",
        "variants": {name: audit(samples) for name, samples in variants.items()},
        "temporal_holdout": {
            name: temporal_holdout(samples, split_row)
            for name, samples in variants.items()
        },
        "by_level_temporal_holdout": level_holdout,
        "by_level": {
            str(level): {name: audit(samples) for name, samples in data.items()}
            for level, data in by_level.items()
        },
        "eligible_transition_feedback": {
            str(level): dict(counts) for level, counts in feedback.items()
        },
        "movement_class_by_level": {
            str(level): dict(counts) for level, counts in anomaly_feedback.items()
        },
        "repeated_noop_audit_by_level": {
            str(level): repeated_noop_audit(
                frames,
                states,
                contexts,
                level=level,
            )
            for level in sorted({state.level for state in states})
        },
        "navigation_anomalies_by_level": {
            str(level): rows for level, rows in anomalies.items()
        },
        "pixel_identical_source_aliases": aliases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.recording)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "variants": report["variants"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
