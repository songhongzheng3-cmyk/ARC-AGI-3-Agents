#!/usr/bin/env python3
"""Freeze contact/motion hypotheses halfway through a recorded level.

The test suffix updates tracking from observations, but cannot change learned
motion outcomes or pattern operators. Motion/contact checks here condition on
the actual player displacement; they are not an end-to-end policy evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.dynamic_contact import DynamicContactModel, contact_relation  # noqa: E402
from agents.mechanism_state_tracker import MechanismStateTracker  # noqa: E402
from agents.motion_context import MotionContextMemory  # noqa: E402
from scripts.shadow_replay_structured import action_name, load_frames  # noqa: E402


def analyze(recording: Path, level: int) -> dict:
    frames = load_frames(recording)
    indices = [i for i, f in enumerate(frames) if int(f.levels_completed or 0) == level]
    if not indices:
        raise ValueError(f"recording has no level {level}")
    split = indices[0] + (indices[-1] - indices[0]) // 2
    tracker, model = MechanismStateTracker(), DynamicContactModel()
    counts = {"train": Counter(), "test": Counter()}
    trace = []
    before = None
    for row, frame in enumerate(frames):
        state = tracker.observe(frame, action=action_name(frame))
        part = "train" if row <= split else "test"
        summary = counts[part]
        details = []
        valid = (
            before is not None
            and before.level == state.level == level
            and before.player_position is not None
            and state.player_position is not None
            and min(before.player_confidence, state.player_confidence) >= 0.8
            and before.resource_level > 0
            and action_name(frame) != "RESET"
        )
        if valid:
            visible = MotionContextMemory.visible_positions(state)
            summary["transitions"] += 1
            changed = before.status_pattern != state.status_pattern
            summary["pattern_changes"] += changed
            any_contact = False
            for signature in sorted(model.moving):
                track = model.tracks.get(signature)
                if track is None or track.phase is None:
                    summary["unknown_phase"] += 1
                    continue
                moved = before.player_position != state.player_position
                predicted = model.next_phase(signature, track.phase, moved)
                if predicted is None:
                    summary["unknown_motion"] += 1
                    continue
                if signature in visible:
                    summary["motion_checks"] += 1
                    summary["motion_errors"] += predicted[1] != visible[signature]
                relation = contact_relation(
                    before.player_position,
                    state.player_position,
                    track.current,
                    predicted[1],
                )
                summary[f"relation_{relation}"] += 1
                if relation in {"crossing", "enter_departing_cell"}:
                    summary[f"pattern_changes_{relation}"] += changed
                contact = model.predict_contact(
                    signature,
                    track.phase,
                    before.player_position,
                    state.player_position,
                    before.status_pattern,
                )
                if contact is None or visible.get(signature) not in {
                    None,
                    state.player_position,
                }:
                    continue
                any_contact = True
                summary["candidate_contacts"] += 1
                if contact.pattern_after is not None:
                    summary["effect_checks"] += 1
                    summary["effect_errors"] += (
                        contact.pattern_after != state.status_pattern
                    )
                else:
                    summary["unknown_effect"] += 1
                details.append(
                    {
                        "signature": signature,
                        "phase": track.phase,
                        "phase_observed": track.observed,
                        "predicted_phase": predicted,
                        "relation": relation,
                        "predicted_pattern": contact.pattern_after,
                        "actual_pattern": state.status_pattern,
                    }
                )
            if changed and not any_contact:
                summary["pattern_changes_without_contact_attribution"] += 1
            if details or changed:
                trace.append(
                    {
                        "source_row": row - 1,
                        "part": part,
                        "action": action_name(frame),
                        "player_before": before.player_position,
                        "player_after": state.player_position,
                        "contacts": details,
                        "pattern_changed": changed,
                    }
                )
        model.observe(state, action=action_name(frame), learn=row <= split)
        before = state
    return {
        "schema_version": 1,
        "recording": str(recording.resolve()),
        "recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "level_zero_based": level,
        "train_through_row_inclusive": split,
        "split_method": "midpoint of the observed target level, chosen before fitting",
        "limitation": "Motion/effect prediction conditional on actual player displacement; not action or score evaluation.",
        "counts": {part: dict(summary) for part, summary in counts.items()},
        "learned_rules": [
            {
                "signature": sig,
                "player_moved": moved,
                "pattern_operations": sorted(ops),
                "support": model.effect_samples[(sig, moved)],
            }
            for (sig, moved), ops in model.operators.items()
        ],
        "tracking_metrics": dict(model.metrics),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.recording, args.level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "trace"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
