#!/usr/bin/env python3
"""Collect bounded public-observation evidence across diverse games."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import traceback
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from arc_agi import Arcade, OperationMode
from arc_agi.rendering import COLOR_MAP
from arcengine import GameAction, GameState
from dotenv import load_dotenv
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.agent import Agent  # noqa: E402
from agents.object_extractor import ObjectExtractor  # noqa: E402
from agents.recorder import Recorder  # noqa: E402
from scripts.verify_ls20_baseline import (  # noqa: E402
    git_info,
    snapshot_sources,
    write_json,
)

DEVELOPMENT = ("ls20", "ft09", "vc33", "tr87", "wa30", "ka59", "ar25", "lp85")
HOLDOUT = ("cn04", "tn36")
MAX_ACTIONS = 60


def click_points(snapshot: Any) -> list[tuple[int, int]]:
    points = []
    for obj in sorted(
        snapshot.objects,
        key=lambda obj: (obj.touches_boundary, obj.area > 512, -obj.area, obj.bbox),
    ):
        cx, cy = obj.centroid
        point = min(obj.pixels, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        if point not in points:
            points.append(point)
    return points[:40]


def component_moves(before: Any, after: Any) -> list[dict[str, Any]]:
    groups = []
    for snapshot in (before, after):
        grouped = defaultdict(list)
        for obj in snapshot.objects:
            min_x, min_y, _, _ = obj.bbox
            shape = tuple((x - min_x, y - min_y) for x, y in obj.pixels)
            grouped[(obj.color, shape)].append(obj)
        groups.append(grouped)
    moves = []
    for key, left in groups[0].items():
        right = groups[1].get(key, [])
        if len(left) == len(right) == 1 and left[0].bbox != right[0].bbox:
            moves.append(
                {
                    "color": key[0],
                    "area": left[0].area,
                    "before": left[0].bbox,
                    "after": right[0].bbox,
                }
            )
    return moves


def montage(path: Path, frames: list[Any]) -> None:
    indices = sorted(
        {0, min(1, len(frames) - 1), min(10, len(frames) - 1), len(frames) - 1}
    )
    sheet = Image.new("RGB", (len(indices) * 256, 278), "white")
    draw = ImageDraw.Draw(sheet)
    for column, index in enumerate(indices):
        grid = frames[index].frame[-1]
        pixels = [COLOR_MAP[int(value)][:7] for row in grid for value in row]
        picture = Image.new("RGB", (len(grid[0]), len(grid)))
        picture.putdata([tuple(bytes.fromhex(value[1:])) for value in pixels])
        sheet.paste(
            picture.resize((256, 256), Image.Resampling.NEAREST), (column * 256, 22)
        )
        draw.text(
            (column * 256 + 4, 4),
            f"step {index} / level {frames[index].levels_completed}",
            fill="black",
        )
    sheet.save(path)


def sample(arcade: Arcade, game: str, output: Path) -> dict[str, Any]:
    directory = output / game
    directory.mkdir()
    card = arcade.open_scorecard(
        tags=["mechanism-screening", "bounded-input-diversity"]
    )
    recorder = Recorder(prefix=game)
    extractor = ObjectExtractor()
    trials: Counter[str] = Counter()
    per_state: Counter[tuple[str, str]] = Counter()
    records = []
    frames = []
    resets = 0
    reason = "action_budget_exhausted"
    try:
        env = arcade.make(game, seed=0, scorecard_id=card)
        if env is None:
            raise RuntimeError("SDK could not create real local environment")
        frame = Agent._convert_raw_frame_data(None, env.observation_space)
        frames.append(frame)
        recorder.record(json.loads(frame.model_dump_json()))
        write_json(
            directory / "metadata.json", env.environment_info.model_dump(mode="json")
        )
        for step in range(1, MAX_ACTIONS + 1):
            if frame.state is GameState.WIN:
                reason = "win"
                break
            grid = np.asarray(frame.frame[-1])
            snapshot = extractor.extract(frame)
            state_key = hashlib.sha256(grid.tobytes()).hexdigest()
            if frame.state is GameState.GAME_OVER:
                if resets >= 1:
                    reason = "game_over_reset_budget_exhausted"
                    break
                resets += 1
                action, data, key = GameAction.RESET, {}, "RESET"
            else:
                candidates = []
                for value in frame.available_actions:
                    action = GameAction.from_id(int(value))
                    if action is GameAction.RESET:
                        continue
                    if action is GameAction.ACTION6:
                        for x, y in click_points(snapshot):
                            candidates.append(
                                (action, {"x": x, "y": y}, f"ACTION6({x},{y})")
                            )
                    else:
                        candidates.append((action, {}, action.name))
                if not candidates:
                    reason = "no_candidate_actions"
                    break
                # Balance action types first, then cover state/action pairs.
                by_type = Counter()
                for entry in records:
                    by_type[entry["action"]] += 1
                action, data, key = min(
                    candidates,
                    key=lambda c: (
                        by_type[c[0].name],
                        per_state[(state_key, c[2])],
                        trials[c[2]],
                    ),
                )
            raw = env.step(
                action,
                data=data,
                reasoning={"source": "mechanism-screening", "step": step},
            )
            if raw is None:
                raise RuntimeError(f"no observation after {key}")
            after = Agent._convert_raw_frame_data(None, raw)
            next_grid = np.asarray(after.frame[-1])
            next_snapshot = extractor.extract(after)
            changed = grid != next_grid
            ys, xs = np.where(changed)
            records.append(
                {
                    "step": step,
                    "action": action.name,
                    "data": data,
                    "before_level": frame.levels_completed,
                    "after_level": after.levels_completed,
                    "state": after.state.name,
                    "changed_pixels": int(changed.sum()),
                    "changed_bbox": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    ]
                    if len(xs)
                    else None,
                    "component_moves": component_moves(snapshot, next_snapshot),
                    "component_count": [
                        len(snapshot.objects),
                        len(next_snapshot.objects),
                    ],
                }
            )
            trials[key] += 1
            per_state[(state_key, key)] += 1
            frame = after
            frames.append(frame)
            recorder.record(json.loads(frame.model_dump_json()))
            if records[-1]["after_level"] > records[-1]["before_level"]:
                reason = "level_completed_preserve_evidence"
                break
        detail = {
            "game": game,
            "sampled_actions": len(records),
            "reason": reason,
            "levels_completed": frame.levels_completed,
            "total_levels": frame.win_levels,
            "final_state": frame.state.name,
            "extra_resets": resets,
            "changed_steps": sum(bool(row["changed_pixels"]) for row in records),
            "steps_with_component_displacement": sum(
                bool(row["component_moves"]) for row in records
            ),
            "action_trials": dict(trials),
            "initial_available_actions": frames[0].available_actions,
            "recording": str(recorder.filename),
            "hypothesis": "unclassified_pending_observation_review",
        }
        write_json(directory / "effects.json", records)
        montage(directory / "observations.png", frames)
    except Exception as exc:
        detail = {
            "game": game,
            "runtime_error": str(exc),
            "recording": str(recorder.filename),
        }
        (directory / "error.txt").write_text(traceback.format_exc())
    finally:
        score = arcade.close_scorecard(card)
        if score is not None:
            write_json(
                directory / "local_scorecard.json", score.model_dump(mode="json")
            )
    write_json(directory / "summary.json", detail)
    return detail


def main() -> int:
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env.example")
    load_dotenv(ROOT / ".env", override=True)
    os.environ["OPERATION_MODE"] = "normal"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "sampling_runs" / "game_screening" / stamp
    output.mkdir(parents=True)
    os.environ["RECORDINGS_DIR"] = str(output / "recordings")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(output / "run.log")],
    )
    arcade = Arcade(
        operation_mode=OperationMode.NORMAL, logger=logging.getLogger("screening-sdk")
    )
    inventory = arcade.get_environments()
    write_json(
        output / "inventory.json", [info.model_dump(mode="json") for info in inventory]
    )
    ids = {info.game_id.split("-")[0]: info.game_id for info in inventory}
    write_json(
        output / "manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "project": git_info(ROOT),
            "source_sha256": snapshot_sources(
                output, ROOT / ".external" / "ls20-solver"
            ),
            "development": [ids[key] for key in DEVELOPMENT],
            "untouched_holdout": [ids[key] for key in HOLDOUT],
            "max_actions_per_game": MAX_ACTIONS,
            "max_extra_resets_per_game": 1,
            "policy": "Balance legal action types, then state/action and global candidate coverage; object-pixel clicks.",
            "evidence_boundary": "Screening only. Frame changes and component shifts do not alone establish causality or game goals.",
        },
    )
    print(f"RESULT_DIRECTORY {output}", flush=True)
    results = []
    for key in DEVELOPMENT:
        result = sample(arcade, ids[key], output)
        results.append(result)
        write_json(output / "report.json", results)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return int(any("runtime_error" in result for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
