"""Audit component identities against public frames and a shadow-checked policy.

No engine, hidden game source or reference solver is loaded. Component masks
are appearances, not proven persistent entities. This report preserves their
pixel boxes, player relations and the observations that first labelled movement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.component_entities import extract_entities  # noqa: E402
from agents.dynamic_contact import DynamicContactModel  # noqa: E402
from agents.motion_context import MotionContextMemory  # noqa: E402
from scripts.shadow_replay_structured import (  # noqa: E402
    action_name,
    load_frames,
    make_agent,
)


def overlaps(first, second):
    return bool(
        first
        and second
        and max(first[0], second[0]) <= min(first[2], second[2])
        and max(first[1], second[1]) <= min(first[3], second[3])
    )


def audit_predictions(samples: list[tuple]) -> dict:
    """Fit on the first half, then check factual transitions without learning."""
    model = DynamicContactModel()
    split = samples[len(samples) // 2][0]
    counts = {"train": Counter(), "test": Counter()}
    changes, errors = [], []
    palette = defaultdict(Counter)
    for row, state, action in samples:
        before = model.previous
        part = "train" if row <= split else "test"
        counter = counts[part]
        valid = bool(
            before is not None
            and before.level == state.level
            and before.resource_level > 0
            and action in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
            and before.player_position is not None
            and state.player_position is not None
            and min(before.player_confidence, state.player_confidence) >= 0.8
        )
        changed = valid and (
            before.status_pattern != state.status_pattern
            or before.status_color != state.status_color
        )
        visible = MotionContextMemory.visible_positions(state)
        if valid:
            counter["transitions"] += 1
            counter["status_changes"] += changed
            moved = before.player_position != state.player_position
            for sig, track in model.tracks.items():
                if sig not in model.moving or track.phase is None:
                    continue
                expected = model.next_phase(sig, track.phase, moved)
                if expected is None:
                    counter["unknown_motion"] += 1
                elif sig in visible:
                    counter["motion_checks"] += 1
                    if expected[1] != visible[sig]:
                        counter["motion_errors"] += 1
                        errors.append(
                            {
                                "row": row,
                                "signature": sig,
                                "expected": expected[1],
                                "actual": visible[sig],
                            }
                        )
        model.observe(state, action=action, learn=row <= split)
        if changed:
            counter["attributed_changes"] += len(model.last_contacts) == 1
            changes.append(
                {
                    "row": row,
                    "part": part,
                    "before": [before.status_signature, before.status_color],
                    "after": [state.status_signature, state.status_color],
                    "contacts": [asdict(contact) for contact in model.last_contacts],
                }
            )
        if valid and len(model.last_contacts) == 1:
            sig = model.last_contacts[0].signature
            key = (sig, before.status_color)
            outputs = palette[key]
            if (
                part == "test"
                and sum(outputs.values()) >= model.MIN_SUPPORT
                and len(outputs) == 1
            ):
                counter["palette_checks"] += 1
                counter["palette_errors"] += state.status_color not in outputs
            if part == "train":
                outputs[state.status_color] += 1
    return {
        "freeze_row": split,
        "counts": {k: dict(v) for k, v in counts.items()},
        "moving_signatures": sorted(model.moving),
        "motion_errors": errors,
        "changes": changes,
        "metrics": dict(model.metrics),
        "palette_edges": [
            {"signature": sig, "before": color, "outputs": dict(outputs)}
            for (sig, color), outputs in palette.items()
            if outputs
        ],
        "boundary": "Frozen-prefix factual motion and conditional contact-palette checks, not action-policy rollout or causal proof.",
    }


def analyze(recording: Path, level: int) -> dict:
    frames = load_frames(recording)
    entry = next(
        i for i, frame in enumerate(frames) if int(frame.levels_completed or 0) == level
    )
    agent = make_agent()
    agent.mechanism_tracker.enable_entity_tracking = False
    policy = agent.mechanism_policy
    policy.enable_dynamic_contact = policy.enable_option_compiler = True
    policy.enable_motion_context = False
    policy.enable_joint_navigation = True
    snapshots, activations = [], []
    raw_samples, entity_samples = [], []
    signature_stats = defaultdict(
        lambda: {"visible_rows": [], "positions": Counter(), "co_visible": Counter()}
    )
    previous_objects = {}
    previous_state = None
    matched = 0
    for row in range(entry, len(frames)):
        if int(frames[row].levels_completed or 0) != level:
            break
        old_moving = set(policy.dynamic_contact.moving)
        selected = agent.choose_action(frames[entry : row + 1], frames[row]).name
        if row + 1 < len(frames):
            expected = action_name(frames[row + 1])
            if selected != expected:
                raise ValueError(
                    f"shadow divergence at row {row}: {selected} != {expected}"
                )
            matched += 1
        state = agent.mechanism_tracker.previous_state
        view = extract_entities(state, np.asarray(frames[row].frame[-1]))
        raw_samples.append((row, state, action_name(frames[row])))
        entity_samples.append(
            (
                row,
                replace(state, motion_entities=view.objects),
                action_name(frames[row]),
            )
        )
        visible = MotionContextMemory.visible_positions(state)
        current_objects = {}
        details = []
        for item in state.objects:
            signature = (item.color, item.area, item.shape_signature)
            if signature in visible and visible[signature] == item.cell:
                current_objects[signature] = item
                stats = signature_stats[signature]
                stats["visible_rows"].append(row)
                stats["positions"][item.cell] += 1
                stats["co_visible"].update(s for s in visible if s != signature)
            details.append(
                {
                    **asdict(item),
                    "motion_eligible": signature in visible,
                    "overlaps_player_bbox": overlaps(item.bbox, state.player_bbox),
                }
            )
        for signature in sorted(policy.dynamic_contact.moving - old_moving):
            before = previous_objects.get(signature)
            after = current_objects.get(signature)
            activations.append(
                {
                    "row": row,
                    "signature": signature,
                    "action": action_name(frames[row]),
                    "before": None if before is None else asdict(before),
                    "after": None if after is None else asdict(after),
                    "player_before": None
                    if previous_state is None
                    else previous_state.player_position,
                    "player_after": state.player_position,
                    "player_bbox_before": None
                    if previous_state is None
                    else previous_state.player_bbox,
                    "player_bbox_after": state.player_bbox,
                }
            )
        snapshots.append(
            {
                "row": row,
                "action": action_name(frames[row]),
                "player": state.player_position,
                "player_bbox": state.player_bbox,
                "player_components": state.player_component_ids,
                "player_confidence": state.player_confidence,
                "layout": None if state.layout is None else asdict(state.layout),
                "status": state.status_signature,
                "color": state.status_color,
                "resource": state.resource_level,
                "moving": sorted(policy.dynamic_contact.moving),
                "contacts": [
                    asdict(contact) for contact in policy.dynamic_contact.last_contacts
                ],
                "events": [event.event_type for event in state.events],
                "objects": details,
                "entities": [asdict(item) for item in view.objects],
                "memberships": [asdict(item) for item in view.memberships],
                "entity_exclusions": view.exclusions,
            }
        )
        previous_objects, previous_state = current_objects, state
    return {
        "schema_version": 1,
        "recording": str(recording.resolve()),
        "recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/audit_component_identity.py",
                "agents/mechanism_state_tracker.py",
                "agents/motion_context.py",
                "agents/dynamic_contact.py",
                "agents/mechanism_policy.py",
                "agents/option_compiler.py",
                "agents/templates/structured_agent.py",
                "agents/component_entities.py",
            )
        },
        "level_zero_based": level,
        "matched_decisions": matched,
        "boundary": "Recorded observation identity audit, not proof of physical identity or new game completion.",
        "activations": activations,
        "signatures": [
            {
                "signature": sig,
                "visible_rows": stats["visible_rows"],
                "positions": [
                    {"cell": cell, "count": count}
                    for cell, count in stats["positions"].items()
                ],
                "co_visible": [
                    {"signature": key, "count": count}
                    for key, count in stats["co_visible"].items()
                ],
            }
            for sig, stats in sorted(signature_stats.items())
        ],
        "dynamic_metrics": dict(policy.dynamic_contact.metrics),
        "dynamic_rules": [
            {
                "signature": sig,
                "player_moved": moved,
                "operators": sorted(ops),
                "support": policy.dynamic_contact.effect_samples[(sig, moved)],
            }
            for (sig, moved), ops in sorted(policy.dynamic_contact.operators.items())
        ],
        "snapshots": snapshots,
        "prediction_audit": {
            "raw_components": audit_predictions(raw_samples),
            "entities": audit_predictions(entity_samples),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("choose a new output path to preserve previous evidence")
    report = analyze(args.recording, args.level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"snapshots", "signatures", "source_sha256"}
            }
        )
    )


if __name__ == "__main__":
    main()
