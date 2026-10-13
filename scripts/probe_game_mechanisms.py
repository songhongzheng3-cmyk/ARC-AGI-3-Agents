#!/usr/bin/env python3
"""Controlled, same-start experiments using public observations only.

The region/colour bindings below are explicit development-game hypotheses,
not a general-purpose solver. Each trial gets a fresh real local environment.
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

from agents.agent import Agent  # noqa: E402
from agents.object_extractor import ObjectExtractor  # noqa: E402
from agents.recorder import Recorder  # noqa: E402
from scripts.screen_game_mechanisms import component_moves, montage  # noqa: E402
from scripts.verify_ls20_baseline import (  # noqa: E402
    git_info,
    snapshot_sources,
    write_json,
)


def signature(frame):
    return hashlib.sha256(
        json.dumps(
            [
                frame.frame,
                frame.state.name,
                frame.levels_completed,
                frame.available_actions,
            ]
        ).encode()
    ).hexdigest()


def experiment_specs(game, initial, study="initial"):
    objects = ObjectExtractor().extract(initial).objects
    if study == "contact":
        # Development experiments derived from the documented initial public
        # layout, not fixed paths for a solver or evidence of goal semantics.
        actors = sorted(
            tuple(map(round, obj.centroid))
            for obj in objects
            if obj.area == 8 and obj.color == 14
        )
        if not game.startswith("ka59-") or len(actors) != 2:
            raise ValueError("contact study requires observed ka59 actors")
        x, y = actors[1]
        return [
            ("first-actor-touch-near-outline", [(3, {}), (3, {}), (2, {})]),
            ("second-actor-purple-crossing", [(6, {"x": x, "y": y})] + [(4, {})] * 4),
            ("first-actor-touch-other-actor", [(4, {})] * 3),
        ]
    if study == "control-switch":
        actors = sorted(
            tuple(map(round, obj.centroid))
            for obj in objects
            if obj.area == 8 and obj.color == 14
        )
        if not game.startswith("ka59-") or len(actors) != 2:
            raise ValueError("control-switch study requires two observed ka59 actors")
        first, second = ({"x": x, "y": y} for x, y in actors)
        return [
            ("no-click-vertical-control", [(1, {}), (2, {}), (1, {})]),
            ("click-first-vertical", [(6, first), (1, {}), (2, {})]),
            ("click-second-vertical", [(6, second), (1, {}), (2, {})]),
            ("click-second-horizontal", [(6, second), (4, {}), (3, {})]),
        ]
    if game.startswith("ka59-"):
        # Directions are hypotheses to test, inferred from the first screen.
        specs = [
            (f"direction-{a}-repeat-return", [(a, {})] * 2 + [(b, {})] * 2)
            for a, b in [(1, 2), (2, 1), (3, 4), (4, 3)]
        ]
        # Use observed centres; no game implementation inspection.
        for obj in objects:
            if obj.area == 8 and obj.color == 14:
                x, y = map(round, obj.centroid)
                specs.append((f"click-green-{x}-{y}", [(6, {"x": x, "y": y})]))
            if obj.area == 16 and obj.color == 4:
                x, y = map(round, obj.centroid)
                specs.append((f"click-outline-{x}-{y}", [(6, {"x": x, "y": y})]))
        return specs
    if game.startswith("ft09-"):
        # Public picture has four panels. Probe square interiors of the
        # bracketed lower-right panel; comparison panels stay untouched.
        return [
            (f"tile-{x}-{y}-twice", [(6, {"x": x, "y": y})] * 2)
            for obj in objects
            if obj.area == 36
            and obj.shape == "rectangle"
            and obj.bbox[0] > len(initial.frame[-1][0]) // 2
            and obj.bbox[1] > len(initial.frame[-1]) // 2
            for x, y in [tuple(map(round, obj.centroid))]
        ]
    raise ValueError(f"no registered controlled hypothesis for {game}")


def run_game(arcade, game, output, study="initial"):
    directory = output / game
    directory.mkdir()
    extractor = ObjectExtractor()
    summaries = []
    reference_signature = None
    specs = None
    index = 0
    while specs is None or index < len(specs):
        card = arcade.open_scorecard(tags=["controlled-mechanism-probe", game])
        try:
            env = arcade.make(game, seed=0, scorecard_id=card)
            if env is None:
                raise RuntimeError(f"missing local environment: {game}")
            initial = Agent._convert_raw_frame_data(None, env.observation_space)
            if specs is None:
                specs = experiment_specs(game, initial, study)
                if not specs:
                    raise RuntimeError("no observation-bound experiments")
                write_json(directory / "specifications.json", specs)
            trial, actions = specs[index]
            trial_directory = directory / trial
            trial_directory.mkdir()
            recorder = Recorder(prefix=f"{game}.{trial}")
            recorder.record(json.loads(initial.model_dump_json()))
            frames = [initial]
            steps = []
            if reference_signature is None:
                reference_signature = signature(initial)
            if signature(initial) != reference_signature:
                raise RuntimeError(
                    "initial observations differ; controlled trial invalid"
                )
            frame = initial
            for action_id, data in actions:
                if frame.state in (GameState.WIN, GameState.GAME_OVER):
                    break
                if action_id not in frame.available_actions:
                    raise RuntimeError("experimental action is not currently legal")
                before_grid = np.asarray(frame.frame[-1])
                raw = env.step(
                    GameAction.from_id(action_id),
                    data=data,
                    reasoning={"source": "controlled-probe", "trial": trial},
                )
                if raw is None:
                    raise RuntimeError("no observation after experimental action")
                after = Agent._convert_raw_frame_data(None, raw)
                after_grid = np.asarray(after.frame[-1])
                if before_grid.shape != after_grid.shape:
                    raise RuntimeError("grid dimensions changed; compare separately")
                changed = before_grid != after_grid
                # Explicit screen partition, not a claim about latent state.
                ys, xs = np.where(changed)
                steps.append(
                    {
                        "action": action_id,
                        "data": data,
                        "changed_pixels": int(changed.sum()),
                        "changed_pixels_excluding_bottom_row": int(changed[:-1].sum()),
                        "transitions": [
                            [
                                int(x),
                                int(y),
                                int(before_grid[y, x]),
                                int(after_grid[y, x]),
                            ]
                            for x, y in zip(xs, ys)
                        ],
                        "component_moves": component_moves(
                            extractor.extract(frame), extractor.extract(after)
                        ),
                        "levels_completed": after.levels_completed,
                        "state": after.state.name,
                    }
                )
                frames.append(after)
                recorder.record(json.loads(after.model_dump_json()))
                frame = after
                if frame.levels_completed > initial.levels_completed:
                    break
            summary = {
                "trial": trial,
                "same_initial_observation": True,
                "actions": len(steps),
                "final_state": frame.state.name,
                "levels_completed": frame.levels_completed,
                "returns_to_initial_except_bottom_row": bool(
                    np.array_equal(
                        np.asarray(frame.frame[-1])[:-1],
                        np.asarray(initial.frame[-1])[:-1],
                    )
                ),
                "steps": steps,
                "recording": str(recorder.filename),
            }
            write_json(trial_directory / "result.json", summary)
            montage(trial_directory / "observations.png", frames)
            summaries.append(summary)
        finally:
            score = arcade.close_scorecard(card)
            if score is not None:
                write_json(
                    directory / f"trial-{index}.local_scorecard.json",
                    score.model_dump(mode="json"),
                )
        index += 1
    write_json(directory / "report.json", summaries)
    return {
        "game": game,
        "independent_initializations": len(summaries),
        "total_actions": sum(row["actions"] for row in summaries),
        "all_starts_identical": True,
        "max_completed_levels": max(row["levels_completed"] for row in summaries),
        "report": str(directory / "report.json"),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--study", choices=("initial", "control-switch", "contact"), default="initial"
    )
    args = cli.parse_args()
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env.example")
    load_dotenv(ROOT / ".env", override=True)
    os.environ["OPERATION_MODE"] = "offline"
    output = (
        ROOT
        / "sampling_runs"
        / "controlled_probes"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True)
    os.environ["RECORDINGS_DIR"] = str(output / "recordings")
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.FileHandler(output / "run.log")]
    )
    games = (
        ["ka59-38d34dbb"]
        if args.study != "initial"
        else ["ka59-38d34dbb", "ft09-0d8bbf25"]
    )
    write_json(
        output / "manifest.json",
        {
            "project": git_info(ROOT),
            "source_sha256": snapshot_sources(
                output, ROOT / ".external" / "ls20-solver"
            ),
            "games": games,
            "environment": "real_local_engine",
            "seed": 0,
            "study": args.study,
            "budget": "ka59 contact: 3 fresh trials, at most 11 actions; control-switch: 4 fresh trials, at most 12 actions; zero extra resets"
            if args.study != "initial"
            else "ka59: 8 fresh trials, at most 20 actions; ft09: 8 fresh trials, at most 16 actions; zero extra resets",
            "hypotheses": [
                "touching an outline may occupy a destination",
                "purple cells may be blocked or traversable for the second selected actor",
                "contact between actors may block, push, or switch control",
            ]
            if args.study == "contact"
            else [
                "clicking the other green object selects which object responds to direction inputs",
                "alternative: click changes a marker but leaves control on the original object",
            ]
            if args.study == "control-switch"
            else [
                "ka59 has repeatable translation/return under paired inputs",
                "ft09 interior clicks have local repeatable binary colour effects",
            ],
            "observation_bindings": "ka59 colour/shape roles and ft09 lower-right panel are development hypotheses from public images; not game-independent parsing",
            "untouched_holdout": ["cn04", "tn36"],
        },
    )
    print(f"RESULT_DIRECTORY {output}", flush=True)
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE, logger=logging.getLogger("probe-sdk")
    )
    results = []
    for game in games:
        result = run_game(arcade, game, output, args.study)
        results.append(result)
        write_json(output / "report.json", results)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
