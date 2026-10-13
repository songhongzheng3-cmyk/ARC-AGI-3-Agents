#!/usr/bin/env python3
"""Check the new binding against documented controlled public trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arcengine import FrameData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".external/ls20-solver/src"))

from ls20.abstract_state import GenericSceneParser  # noqa: E402

from agents.tile_scene_binding import TileSceneBinding  # noqa: E402
from scripts.verify_ls20_baseline import write_json  # noqa: E402


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directories", nargs="+", type=Path)
    args = cli.parse_args()
    results = []
    for directory in args.directories:
        for path in sorted((directory / "recordings").glob("ka59*.recording.jsonl")):
            frames = [
                FrameData.model_validate(json.loads(line)["data"])
                for line in path.read_text().splitlines()
            ]
            parser = GenericSceneParser()
            parser.parse(frames[0])
            if parser.layout is None:
                raise RuntimeError("reference could not infer geometry")
            binding = TileSceneBinding(frames[0], parser.layout.to_dict())
            expected = [(3, 3), (6, 3)]
            selected = 0
            previous = None
            rows = []
            for index, frame in enumerate(frames):
                action = frame.action_input.id.value if index else 0
                if action == 6:
                    x, y = frame.action_input.data["x"], frame.action_input.data["y"]
                    cell = binding.pixel_cell((x, y))
                    if cell in expected:
                        selected = expected.index(cell)
                elif action in (1, 2, 3, 4):
                    dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[action]
                    # Controlled feedback confirmed the second sprite cannot
                    # move vertically from its initial narrow-channel cell.
                    if not (selected == 1 and expected[1] == (6, 3) and dy):
                        expected[selected] = (
                            expected[selected][0] + dx,
                            expected[selected][1] + dy,
                        )
                scene = binding.parse(frame)
                if previous is not None and action in (1, 2, 3, 4):
                    binding.calibrate_active_marker(previous, scene)
                    scene = binding.parse(frame)
                actual = [actor.cell for actor in scene.actors]
                rows.append(
                    {
                        "step": index,
                        "expected_cells": list(expected),
                        "matches": actual == expected and not scene.warnings,
                        "scene": scene.to_dict(),
                    }
                )
                previous = scene
            results.append(
                {
                    "recording": str(path.resolve()),
                    "accepted": all(row["matches"] for row in rows),
                    "rows": rows,
                }
            )
    # Exclude unsupported studies from acceptance instead of silently claiming
    # correctness for cases whose expected values have not been documented.
    report = {
        "accepted": bool(results) and all(r["accepted"] for r in results),
        "evidence_boundary": "Fixture validation only, expected cells are development observation labels; no solver/goal claim.",
        "results": results,
    }
    write_json(args.directories[-1] / "tile_binding_validation.json", report)
    print(
        json.dumps(
            {
                "accepted": report["accepted"],
                "trials": len(results),
                "failed": [r["recording"] for r in results if not r["accepted"]],
            }
        )
    )
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
