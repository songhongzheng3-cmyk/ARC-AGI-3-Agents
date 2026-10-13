#!/usr/bin/env python3
"""First-level learned-transition experiment, plus contact-rule ablation.

Training inputs are explicit prior ka59 public development trajectories.
This measures transfer from sampling into planning, not zero-shot games.
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

from arc_agi import Arcade, OperationMode
from arcengine import FrameData
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".external/ls20-solver/src"))

from ls20.abstract_state import GenericSceneParser  # noqa: E402
from ls20.search import bfs_search  # noqa: E402

from agents.agent import Agent  # noqa: E402
from agents.selectable_navigation_model import SelectableNavigationModel  # noqa: E402
from agents.templates.selectable_navigation_agent import (  # noqa: E402
    SelectableNavigationAgent,
)
from agents.tile_scene_binding import TileSceneBinding  # noqa: E402
from scripts.verify_ls20_baseline import (  # noqa: E402
    git_info,
    snapshot_sources,
    write_json,
)


def binding_for(frame):
    parser = GenericSceneParser()
    parser.parse(frame)
    if parser.layout is None:
        raise ValueError("no inferred geometry")
    return TileSceneBinding(frame, parser.layout.to_dict())


def read_frames(path):
    return [
        FrameData.model_validate(json.loads(line)["data"])
        for line in path.read_text().splitlines()
    ]


def train(model, paths):
    marker = None
    # First calibrate the selection marker from an observed unit movement;
    # do not infer selection from a colour name or palette order.
    for path in paths:
        frames = read_frames(path)
        binding = binding_for(frames[0])
        previous = binding.parse(frames[0])
        for frame in frames[1:]:
            scene = binding.parse(frame)
            old = {a.identity: a.cell for a in previous.actors}
            moved = [
                a
                for a in scene.actors
                if a.identity in old and a.cell != old[a.identity]
            ]
            if (
                len(moved) == 1
                and sum(
                    abs(x - y) for x, y in zip(moved[0].cell, old[moved[0].identity])
                )
                == 1
                and frame.action_input.id.value in (1, 2, 3, 4)
            ):
                if binding.calibrate_active_marker(previous, scene):
                    marker = binding.active_marker
                    break
            previous = scene
        if marker is not None:
            break
    if marker is None:
        raise ValueError("no selection calibration evidence")
    model.binding.active_marker = marker
    for path in paths:
        frames = read_frames(path)
        binding = binding_for(frames[0])
        binding.active_marker = marker
        previous = binding.parse(frames[0])
        for frame in frames[1:]:
            scene = binding.parse(frame)
            model.learn(previous, frame.action_input.id.value, scene)
            previous = scene
    if len(model.directions) != 4:
        raise ValueError("four input displacements not learned")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--movement", type=Path, required=True)
    cli.add_argument("--contact", type=Path, required=True)
    args = cli.parse_args()
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env.example")
    load_dotenv(ROOT / ".env", override=True)
    os.environ["OPERATION_MODE"] = "offline"
    output = (
        ROOT
        / "sampling_runs/selectable_transfer"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True)
    os.environ["RECORDINGS_DIR"] = str(output / "recordings")
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.FileHandler(output / "run.log")]
    )
    paths = sorted(
        (args.movement / "recordings").glob("ka59*.direction-*.recording.jsonl")
    ) + sorted(
        (args.contact / "recordings").glob(
            "ka59*.first-actor-touch-other-actor*.recording.jsonl"
        )
    )
    if len(paths) != 5:
        raise ValueError("expected four directional trials and one contact trial")
    write_json(
        output / "manifest.json",
        {
            "project": git_info(ROOT),
            "backend": git_info(ROOT / ".external/ls20-solver"),
            "source_sha256": snapshot_sources(output, ROOT / ".external/ls20-solver"),
            "training_recordings": {
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths
            },
            "game": "ka59-38d34dbb",
            "seed": 0,
            "max_actions_per_run": 200,
            "search": "unchanged reference bfs_search",
            "comparison": "only disable contact relocation rule; same binding, directions, goals, budget; planning ablation, not an online-discovery agent comparison",
            "goal": "occupancy hypothesis, requires actual levels_completed increase",
            "untouched_holdout": ["cn04", "tn36"],
        },
    )
    print(f"RESULT_DIRECTORY {output}", flush=True)
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE, logger=logging.getLogger("transfer-sdk")
    )
    reports = []
    for variant in ("transferred-contact", "without-contact"):
        card = arcade.open_scorecard(tags=["selectable-transfer", variant])
        try:
            env = arcade.make("ka59-38d34dbb", seed=0, scorecard_id=card)
            if env is None:
                raise RuntimeError("missing local game environment")
            initial = Agent._convert_raw_frame_data(None, env.observation_space)
            model = SelectableNavigationModel(binding_for(initial), initial)
            train(model, paths)
            if variant == "without-contact":
                model.contact_rules.clear()
            write_json(output / f"{variant}.initial_model.json", model.to_dict())
            agent = SelectableNavigationAgent(
                card_id=card,
                game_id="ka59-38d34dbb",
                agent_name=variant,
                ROOT_URL="local",
                record=True,
                arc_env=env,
                model=model,
                search=bfs_search,
            )
            agent.main()
            write_json(output / f"{variant}.result.json", agent.result)
            detail = {
                key: value
                for key, value in agent.result.items()
                if key not in ("plans", "transitions", "model")
            }
            detail["variant"] = variant
            reports.append(detail)
            write_json(output / "report.json", reports)
            print(json.dumps(detail, ensure_ascii=False), flush=True)
        finally:
            score = arcade.close_scorecard(card)
            if score is not None:
                write_json(
                    output / f"{variant}.local_scorecard.json",
                    score.model_dump(mode="json"),
                )
    return 0 if reports[0]["level_advanced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
