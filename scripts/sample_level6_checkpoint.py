#!/usr/bin/env python3
"""Broad, reproducible directional sampling from a level-6 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    package = ModuleType("agents")
    package.__path__ = [str(ROOT / "agents")]
    sys.modules["agents"] = package

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import FrameData, GameAction  # noqa: E402

from agents.agent import Agent  # noqa: E402
from agents.mechanism_state_tracker import MechanismStateTracker  # noqa: E402
from agents.motion_context import MotionContextMemory  # noqa: E402
from scripts.shadow_replay_structured import action_name, load_frames  # noqa: E402
from scripts.validate_compiled_candidate import same_observation  # noqa: E402


def _scene_key(state, context):
    dynamic_signatures = {item.signature for item in context}
    static_objects = tuple(
        sorted(
            (item.color, item.area, item.shape_signature, item.cell)
            for item in state.objects
            if (item.color, item.area, item.shape_signature) not in dynamic_signatures
            and item.object_id not in state.player_component_ids
        )
    )
    return (
        state.level,
        state.player_position,
        state.status_signature,
        state.status_color,
        state.walkable_cells,
        state.blocked_cells,
        static_objects,
        tuple((item.signature, item.previous, item.current) for item in context),
    )


def _choose_action(state, context, attempts, coarse_counts, global_counts, rng):
    available = {
        "ACTION1": GameAction.ACTION1,
        "ACTION2": GameAction.ACTION2,
        "ACTION3": GameAction.ACTION3,
        "ACTION4": GameAction.ACTION4,
    }
    scene = _scene_key(state, context)
    coarse = (state.level, state.player_position, state.status_signature, state.status_color)
    candidates = list(available)
    unseen = [name for name in candidates if (scene, name) not in attempts]
    if unseen:
        candidates = unseen
    return min(
        candidates,
        key=lambda name: (
            attempts[(scene, name)],
            coarse_counts[(coarse, name)],
            global_counts[name],
            rng.random(),
        ),
    )


def sample(recording, checkpoint_result, output, *, budget=120, seed=20261020):
    if not 1 <= budget <= 200:
        raise ValueError("budget must be 1..200")
    frames = load_frames(recording)
    branch = json.loads(checkpoint_result.read_text())
    freeze = int(branch["freeze_row"])
    checks = branch["checks"]
    if int(frames[freeze].levels_completed or 0) != 5:
        raise ValueError("freeze row is not the level-6 checkpoint")
    output.mkdir(parents=True, exist_ok=False)

    tracker = MechanismStateTracker(enable_entity_tracking=True)
    motion = MotionContextMemory()
    for frame in frames[: freeze + 1]:
        action = action_name(frame)
        state = tracker.observe(frame, action=action)
        motion.observe(state, reset=action == "RESET")

    result = {
        "recording": str(recording.resolve()),
        "recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "branch_result": str(checkpoint_result.resolve()),
        "branch_result_sha256": hashlib.sha256(checkpoint_result.read_bytes()).hexdigest(),
        "freeze_row": freeze,
        "verified_branch_actions": len(checks),
        "sampling_budget": budget,
        "seed": seed,
        "status": "restoring",
        "checks": [],
        "boundary": "Local offline checkpoint sampling; not Kaggle evaluation or game completion.",
    }
    result_path = output / "result.json"

    def save():
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    save()
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
        recordings_dir=str(output / "engine_recordings"),
    )
    card = arcade.open_scorecard(tags=["level6-broad-checkpoint-sampling"])
    try:
        env = arcade.make(frames[0].game_id, seed=0, scorecard_id=card)
        if env is None:
            raise RuntimeError("public game unavailable")
        actual = Agent._convert_raw_frame_data(None, env.observation_space)
        if not same_observation(actual, frames[0]):
            actual = Agent._convert_raw_frame_data(
                None, env.step(GameAction.from_name(action_name(frames[0])))
            )
            if not same_observation(actual, frames[0]):
                raise ValueError("initial recording observation mismatch")
        for row in range(1, freeze + 1):
            actual = Agent._convert_raw_frame_data(
                None, env.step(GameAction.from_name(action_name(frames[row])))
            )
            if not same_observation(actual, frames[row]):
                raise ValueError(f"checkpoint restoration diverged at row {row}")
        for index, check in enumerate(checks, start=1):
            actual = Agent._convert_raw_frame_data(
                None, env.step(GameAction.from_name(check["action"]))
            )
            expected = FrameData.model_validate_json(
                (checkpoint_result.parent / f"frame-{index:03}.json").read_text()
            )
            if not same_observation(actual, expected):
                raise ValueError(f"saved branch diverged at action {index}")
            state = tracker.observe(actual, action=check["action"])
            motion.observe(state)

        result.update(
            status="sampling",
            restored_prefix_observations=freeze + 1,
            matched_branch_observations=len(checks),
            source={
                "player": tracker.previous_state.player_position,
                "pattern": tracker.previous_state.status_signature,
                "color": tracker.previous_state.status_color,
                "resource": tracker.previous_state.resource_level,
                "levels_completed": int(actual.levels_completed or 0),
            },
        )
        save()
        rng = random.Random(seed)
        attempts = Counter()
        coarse_counts = Counter()
        global_counts = Counter()
        outcome_counts = Counter()
        for index in range(1, budget + 1):
            before = tracker.previous_state
            context_before = motion.context
            chosen = _choose_action(
                before, context_before, attempts, coarse_counts, global_counts, rng
            )
            scene = _scene_key(before, context_before)
            coarse = (before.level, before.player_position, before.status_signature, before.status_color)
            attempts[(scene, chosen)] += 1
            coarse_counts[(coarse, chosen)] += 1
            global_counts[chosen] += 1
            actual = Agent._convert_raw_frame_data(
                None, env.step(GameAction.from_name(chosen))
            )
            (output / f"frame-{index:03}.json").write_text(
                actual.model_dump_json() + "\n"
            )
            after = tracker.observe(actual, action=chosen)
            context_after = motion.observe(after)
            position_changed = before.player_position != after.player_position
            status_changed = (
                before.status_signature != after.status_signature
                or before.status_color != after.status_color
            )
            resource_delta = after.resource_level - before.resource_level
            dynamic_changed = context_before != context_after
            level_delta = int(actual.levels_completed or 0) - int(
                result["source"]["levels_completed"]
            )
            events = [event.event_type for event in after.events]
            if level_delta > 0:
                outcome = "level_advanced"
            elif status_changed:
                outcome = "status_changed"
            elif resource_delta > 0:
                outcome = "resource_increased"
            elif position_changed:
                outcome = "player_moved"
            elif dynamic_changed:
                outcome = "dynamic_context_changed_only"
            else:
                outcome = "no_detected_change"
            outcome_counts[outcome] += 1
            result["checks"].append(
                {
                    "index": index,
                    "action": chosen,
                    "before_player": before.player_position,
                    "after_player": after.player_position,
                    "before_pattern": before.status_signature,
                    "after_pattern": after.status_signature,
                    "before_color": before.status_color,
                    "after_color": after.status_color,
                    "before_resource": before.resource_level,
                    "after_resource": after.resource_level,
                    "resource_delta": resource_delta,
                    "levels_completed": int(actual.levels_completed or 0),
                    "player_moved": position_changed,
                    "status_changed": status_changed,
                    "dynamic_context_changed": dynamic_changed,
                    "outcome": outcome,
                    "events": events,
                    "attempts_for_context_action": attempts[(scene, chosen)],
                }
            )
            result["outcome_counts"] = dict(outcome_counts)
            result["action_counts"] = dict(global_counts)
            if int(actual.levels_completed or 0) > int(result["source"]["levels_completed"]):
                result["status"] = "level_completed"
                save()
                break
            if actual.state.name in {"GAME_OVER", "WIN", "GAME_WON"}:
                result["status"] = actual.state.name.lower()
                save()
                break
            if after.resource_capacity > 0 and after.resource_level == 0:
                result["status"] = "resource_exhausted"
                save()
                break
            save()
        else:
            result["status"] = "sampling_budget_exhausted"
    except Exception as exc:
        result.update(status="sampling_error", error=repr(exc))
        save()
        raise
    finally:
        arcade.close_scorecard(card)
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("checkpoint_result", type=Path)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.INFO)
    result = sample(
        args.recording,
        args.checkpoint_result,
        args.output,
        budget=args.budget,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": len(result["checks"]),
                "outcomes": result.get("outcome_counts", {}),
                "actions": result.get("action_counts", {}),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
