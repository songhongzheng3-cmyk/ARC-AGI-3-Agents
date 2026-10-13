#!/usr/bin/env python3
"""Bounded second-level probes after a model-planned first-level setup.

Only public observations are inspected; the first-level plan is regenerated
for every trial and its actions count toward the experimental cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".external/ls20-solver/src"))

from ls20.search import bfs_search  # noqa: E402

from agents.agent import Agent  # noqa: E402
from agents.object_extractor import ObjectExtractor  # noqa: E402
from agents.selectable_navigation_model import SelectableNavigationModel  # noqa: E402
from agents.templates.selectable_navigation_agent import (  # noqa: E402
    SelectableNavigationAgent,
)
from scripts.run_selectable_transfer import binding_for, train  # noqa: E402
from scripts.verify_ls20_baseline import (  # noqa: E402
    git_info,
    snapshot_sources,
    write_json,
)


def visible_objects(frame):
    return [
        {
            "color": obj.color,
            "area": obj.area,
            "bbox": obj.bbox,
            "centroid": obj.centroid,
            "shape": obj.shape,
        }
        for obj in ObjectExtractor().extract(frame).objects
        if obj.color in (4, 14) and obj.area >= 2 and obj.bbox[1] < 63
    ]


def select_probes(frame, study="selection"):
    # Bboxes are public observation evidence. Each trial starts from the same
    # untouched second-level frame; no coordinate is used by the solver.
    shapes = [
        obj
        for obj in visible_objects(frame)
        if obj["color"] == 14 and obj["bbox"][0] >= 30
    ]
    by_size = {
        (o["bbox"][2] - o["bbox"][0] + 1, o["bbox"][3] - o["bbox"][1] + 1): o
        for o in shapes
    }
    if set(by_size) != {(3, 3), (3, 6), (6, 3), (6, 6)}:
        raise ValueError(f"unsupported second-level sprite sizes: {sorted(by_size)}")

    def click(size):
        x, y, _, _ = by_size[size]["bbox"]
        return (6, {"x": x + 1, "y": y + 1})

    if study in ("square-contact", "post-contact", "square-goal", "vertical-lift"):
        return [
            (
                "single-approaches-square-from-right",
                [(4, {})] * 4 + [(1, {})] * 3 + [(3, {})],
            )
        ]
    if study == "barriers":
        return [
            ("single-left-boundary", [(3, {})] * 3),
            ("horizontal-left-boundary", [click((6, 3))] + [(3, {})] * 5),
            ("vertical-left-boundary", [click((3, 6))] + [(3, {})] * 3),
            ("square-left-boundary", [click((6, 6))] + [(3, {})] * 6),
        ]
    return [
        ("default-single-horizontal", [(4, {}), (3, {})]),
        ("click-horizontal-then-return", [click((6, 3)), (4, {}), (3, {})]),
        ("click-vertical-then-return", [click((3, 6)), (1, {}), (2, {})]),
        ("click-square-then-return", [click((6, 6)), (4, {}), (3, {})]),
    ]


def square_marker(frame):
    grid = frame.frame[-1]
    candidates = [
        (x, y)
        for y in range(len(grid) - 1)
        for x in range(len(grid[0]) - 1)
        if all(grid[y + dy][x + dx] == 5 for dy in (0, 1) for dx in (0, 1))
    ]
    if len(candidates) != 1:
        raise ValueError(f"square marker is not unique: {candidates}")
    return candidates[0]


def step_summary(before, after, action, data):
    old, new = np.asarray(before.frame[-1]), np.asarray(after.frame[-1])
    if old.shape != new.shape:
        raise ValueError("unexpected frame dimensions")
    ys, xs = np.where(old != new)
    return {
        "action": action,
        "data": data,
        "changed_pixels": int(len(xs)),
        "changed_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        if len(xs)
        else None,
        "actors_and_outlines": visible_objects(after),
        "levels_completed": after.levels_completed,
        "state": after.state.name,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--study",
        choices=(
            "selection",
            "barriers",
            "square-contact",
            "post-contact",
            "square-goal",
            "vertical-lift",
        ),
        default="selection",
    )
    args = cli.parse_args()
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env.example")
    load_dotenv(ROOT / ".env", override=True)
    os.environ["OPERATION_MODE"] = "offline"
    output = (
        ROOT
        / "sampling_runs/ka59_level2_probes"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    os.environ["RECORDINGS_DIR"] = str(output / "recordings")
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.FileHandler(output / "run.log")]
    )
    movement = (
        ROOT / "sampling_runs/controlled_probes/20261008T073057031402Z/recordings"
    )
    contact = ROOT / "sampling_runs/controlled_probes/20261008T074433370992Z/recordings"
    paths = sorted(movement.glob("ka59*.direction-*.recording.jsonl")) + sorted(
        contact.glob("ka59*.first-actor-touch-other-actor*.recording.jsonl")
    )
    if len(paths) != 5:
        raise ValueError("expected five public training traces")
    write_json(
        output / "manifest.json",
        {
            "game": "ka59-38d34dbb",
            "seed": 0,
            "operation_mode": "offline",
            "preparation": "model-planned first-level completion, repeated for every independent trial",
            "probe_budget": "at most nineteen actions in second level per trial; no reset",
            "study": args.study,
            "hypotheses": [
                "click selects an entire same-color connected object",
                "direction translates the selected object rigidly",
                "blocked/no-op or partial movement contradicts that model",
                "barriers study: isolated object is either blocked at the region boundary or crosses it",
                "square-contact study: a single sprite can approach the square from the right and test passive relocation",
                "post-contact study: use observed square marker after contact, then test left movement/goal",
                "square-goal study: move square up into the lower-left outline after observed contact",
                "vertical-lift study: move the square below the left vertical sprite, then contact it from below",
            ],
            "project": git_info(ROOT),
            "backend": git_info(ROOT / ".external/ls20-solver"),
            "source_sha256": snapshot_sources(output, ROOT / ".external/ls20-solver"),
            "training_recordings": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            },
            "untouched_holdout": ["cn04", "tn36"],
        },
    )
    print(f"RESULT_DIRECTORY {output}", flush=True)
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE, logger=logging.getLogger("ka59-probe")
    )
    reports, probes = [], None
    for trial_number in range(
        1
        if args.study
        in ("square-contact", "post-contact", "square-goal", "vertical-lift")
        else 4
    ):
        card = arcade.open_scorecard(tags=["ka59-level2-probe", str(trial_number)])
        try:
            env = arcade.make("ka59-38d34dbb", seed=0, scorecard_id=card)
            if env is None:
                raise RuntimeError("no local environment")
            initial = Agent._convert_raw_frame_data(None, env.observation_space)
            model = SelectableNavigationModel(binding_for(initial), initial)
            train(model, paths)
            agent = SelectableNavigationAgent(
                card_id=card,
                game_id="ka59-38d34dbb",
                agent_name=f"level2-probe-{trial_number}",
                ROOT_URL="local",
                record=True,
                arc_env=env,
                model=model,
                search=bfs_search,
            )
            agent.main()
            if not agent.result["level_advanced"] or agent.action_counter != 11:
                raise RuntimeError(
                    f"first-level preparation failed: {agent.result['reason']}"
                )
            frame = agent.frames[-1]
            if probes is None:
                probes = select_probes(frame, args.study)
                write_json(output / "specifications.json", probes)
                write_json(
                    output / "second_level_initial_objects.json", visible_objects(frame)
                )
            name, actions = probes[trial_number]
            steps = []
            follow_up = [(3, {})] * 4
            if args.study in ("square-goal", "vertical-lift"):
                follow_up += [(1, {})]
            if args.study == "vertical-lift":
                follow_up += [(2, {})] * 2 + [(3, {})] * 2 + [(1, {})]
            extension = (
                1 + len(follow_up)
                if args.study in ("post-contact", "square-goal", "vertical-lift")
                else 0
            )
            for step_number in range(len(actions) + extension):
                if step_number < len(actions):
                    action_id, data = actions[step_number]
                elif step_number == len(actions):
                    x, y = square_marker(frame)
                    action_id, data = 6, {"x": x, "y": y}
                else:
                    action_id, data = follow_up[step_number - len(actions) - 1]
                if (
                    frame.state in (GameState.WIN, GameState.GAME_OVER)
                    or frame.levels_completed != 1
                ):
                    break
                if action_id not in frame.available_actions:
                    raise ValueError("probe action not legal")
                raw = env.step(
                    GameAction.from_id(action_id),
                    data=data,
                    reasoning={"source": "bounded-ka59-level2-probe", "trial": name},
                )
                if raw is None:
                    raise RuntimeError("no response")
                after = Agent._convert_raw_frame_data(None, raw)
                steps.append(step_summary(frame, after, action_id, data))
                agent.append_frame(after)
                agent.action_counter += 1
                frame = after
            summary = {
                "trial": name,
                "setup_actions": 11,
                "second_level_actions": len(steps),
                "levels_completed": frame.levels_completed,
                "recording": str(agent.recorder.filename),
                "steps": steps,
            }
            reports.append(summary)
            write_json(output / "report.json", reports)
            print(
                json.dumps(
                    {k: v for k, v in summary.items() if k != "steps"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        finally:
            score = arcade.close_scorecard(card)
            if score is not None:
                write_json(
                    output / f"trial-{trial_number}.local_scorecard.json",
                    score.model_dump(mode="json"),
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
