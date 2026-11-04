"""Inspect all public animation subframes without advancing policy state.

An action is one transition: intermediate animation frames must not be fed to
the action-level tracker as additional decisions or entity clock ticks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from arcengine import FrameData

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mechanism_state_tracker import (  # noqa: E402
    ACTION_DELTAS,
    MechanismStateTracker,
)
from scripts.shadow_replay_structured import action_name  # noqa: E402


def audit_animation(tracker, frame):
    layout = tracker.layout
    if layout is None or tracker.player_appearance is None:
        raise ValueError("player appearance must be bound from earlier observations")
    name = action_name(frame)
    source = tracker.player_position
    delta = ACTION_DELTAS.get(name)
    expected = (
        None
        if source is None or delta is None
        else (source[0] + delta[0], source[1] + delta[1])
    )
    previous_grid = tracker.previous_grid
    rows = []
    for index, raw in enumerate(frame.frame):
        grid = np.asarray(raw)
        matches = [
            bbox
            for bbox, appearance in tracker._composite_player_crops(grid, layout)
            if appearance == tracker.player_appearance
        ]
        # Scan the *same* bound appearance independently in every subframe.
        # Do not carry a command-predicted coordinate into an unobserved frame.
        bbox = matches[0] if len(matches) == 1 else None
        rows.append(
            {
                "subframe": index,
                "grid_sha256": hashlib.sha256(
                    grid.astype(np.uint8).tobytes()
                ).hexdigest(),
                "player_bbox": bbox,
                "player": None
                if bbox is None
                else tracker._bbox_to_player_cell(bbox, layout),
                "appearance_matches": len(matches),
                "pattern": tracker._status_pattern(grid, layout),
                "color": tracker._status_color(grid, layout),
                "pixel_changes": int(np.count_nonzero(grid != previous_grid)),
            }
        )
        previous_grid = grid
    return {
        "action": name,
        "source_player": source,
        "commanded_neighbor": expected,
        "subframes": rows,
        "final_player": rows[-1]["player"] if rows else None,
        "boundary": "Public pixels only. Animation audit, not a gate/teleport rule or score evaluation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--action-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tracker = MechanismStateTracker()
    sources = {}
    for index in range(1, args.action_index + 1):
        path = args.frames_dir / f"frame-{index:03}.json"
        payload = path.read_bytes()
        sources[path.name] = hashlib.sha256(payload).hexdigest()
        frame = FrameData.model_validate_json(payload)
        if index < args.action_index:
            tracker.observe(frame, action=action_name(frame))
    report = audit_animation(tracker, frame)
    report["source_sha256"] = sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "source_player": report["source_player"],
                "commanded_neighbor": report["commanded_neighbor"],
                "final_player": report["final_player"],
                "subframe_count": len(report["subframes"]),
            }
        )
    )


if __name__ == "__main__":
    main()
