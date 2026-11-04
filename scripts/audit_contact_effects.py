"""Freeze prefix-only delayed contact learning, check held-out factual contacts.

No engine, reference solver or hidden game source is loaded. A correct effect
given actual player movement and a unique contact is not a planned rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.contact_effects import ContactEffectMemory  # noqa: E402
from agents.dynamic_contact import DynamicContactModel  # noqa: E402
from agents.mechanism_state_tracker import MechanismStateTracker  # noqa: E402
from scripts.shadow_replay_structured import action_name, load_frames  # noqa: E402


def audit(recording: Path, *, level: int, freeze_row: int) -> dict:
    frames = load_frames(recording)
    rows = [i for i, f in enumerate(frames) if int(f.levels_completed or 0) == level]
    if not rows or not rows[0] < freeze_row < rows[-1]:
        raise ValueError("freeze must be strictly inside the selected level")
    tracker = MechanismStateTracker(enable_entity_tracking=True)
    dynamic = DynamicContactModel()
    effects = ContactEffectMemory()
    counts = Counter()
    checks, timeline = [], []
    frozen = None
    for row in rows:
        action = action_name(frames[row])
        state = tracker.observe(frames[row], action=action)
        before = dynamic.previous
        dynamic.observe(state, action=action, learn=row <= freeze_row)
        if row > freeze_row and before is not None:
            counts["transitions"] += 1
            changed = (before.status_pattern, before.status_color) != (
                state.status_pattern,
                state.status_color,
            )
            counts["status_changes"] += changed
            if len(dynamic.last_contacts) == 1:
                counts["unique_contacts"] += 1
                contact = dynamic.last_contacts[0]
                prediction = effects.predict(
                    contact.signature,
                    contact.relation,
                    before.player_position != state.player_position,
                    before.status_pattern,
                    before.status_color,
                )
                counts["pattern_checks"] += prediction.pattern is not None
                counts["color_checks"] += prediction.color is not None
                counts["pattern_unknown"] += prediction.pattern is None
                counts["color_unknown"] += prediction.color is None
                counts["pattern_errors"] += (
                    prediction.pattern is not None
                    and prediction.pattern != state.status_pattern
                )
                counts["color_errors"] += (
                    prediction.color is not None
                    and prediction.color != state.status_color
                )
                checks.append(
                    {
                        "row": row,
                        "contact": asdict(contact),
                        "prediction": asdict(prediction),
                        "actual_pattern": state.status_pattern,
                        "actual_color": state.status_color,
                    }
                )
        effects.observe(state, dynamic, action=action, learn=row <= freeze_row)
        if row <= freeze_row:
            timeline.append(
                {
                    "row": row,
                    "effect_samples": len(effects.evidence),
                    "delayed_samples": sum(
                        e.available_tick > e.transition_tick
                        for e in effects.evidence.values()
                    ),
                }
            )
        if row == freeze_row:
            frozen = effects.diagnostics()
    if effects.diagnostics()["evidence"] != frozen["evidence"]:
        raise AssertionError("held-out observations changed frozen effect evidence")
    return {
        "recording": str(recording.resolve()),
        "recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "agents/contact_effects.py",
                "agents/dynamic_contact.py",
                "agents/component_entities.py",
                "agents/mechanism_state_tracker.py",
                "agents/motion_context.py",
                "scripts/audit_contact_effects.py",
            )
        },
        "level_zero_based": level,
        "start_row": rows[0],
        "freeze_row": freeze_row,
        "tick_mapping": "recording_row = start_row + observation_tick (no reset in this audit)",
        "counts": dict(counts),
        "frozen": frozen,
        "timeline": timeline,
        "checks": checks,
        "boundary": "Frozen-prefix conditional effects on factual held-out contacts, not rollout, causal proof or new game completion.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--freeze-row", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path to preserve previous evidence")
    report = audit(args.recording, level=args.level, freeze_row=args.freeze_row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("freeze_row", "counts", "boundary")}))


if __name__ == "__main__":
    main()
