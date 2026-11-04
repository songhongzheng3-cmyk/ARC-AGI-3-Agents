"""Validate one frozen compiled candidate in the public engine, stopping on error.

Prefix actions restore an experimental checkpoint, not a competition policy.
Only the frozen audit's newly synthesized actions are tested after that point.
This is NOT an end-to-end Agent evaluation or an official competition score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

from agents.agent import Agent  # noqa: E402
from scripts.shadow_replay_structured import (  # noqa: E402
    action_name,
    load_frames,
    make_agent,
)


def same_observation(actual, expected) -> bool:
    return (
        actual.frame[-1] == expected.frame[-1]
        and actual.state == expected.state
        and actual.levels_completed == expected.levels_completed
        and actual.available_actions == expected.available_actions
    )


def validate(audit_path: Path, output: Path, *, target_index: int = 0) -> dict:
    audit = json.loads(audit_path.read_text())
    recording = Path(audit["recording"])
    if hashlib.sha256(recording.read_bytes()).hexdigest() != audit["recording_sha256"]:
        raise ValueError("recording changed since compilation")
    for name, expected in audit["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source changed since compilation: {name}")
    plan = audit["targets"][target_index]["compiled"]["plan"]
    if plan is None or not plan["steps"]:
        raise ValueError("audit has no nonempty candidate")
    frames = load_frames(recording)
    freeze = audit["freeze_row"]
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "audit": str(audit_path.resolve()),
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "boundary": "Checkpoint-restored candidate validation, not end-to-end Agent performance.",
        "prefix_actions": freeze,
        "candidate_actions": len(plan["steps"]),
        "checks": [],
        "status": "running",
    }

    def save():
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
        recordings_dir=str(output / "engine_recordings"),
    )
    card = arcade.open_scorecard(tags=["compiled-candidate-checkpoint-validation"])
    try:
        env = arcade.make(frames[0].game_id, seed=0, scorecard_id=card)
        if env is None:
            raise RuntimeError("public game not available locally")
        actual = Agent._convert_raw_frame_data(None, env.observation_space)
        if not same_observation(actual, frames[0]):
            # Agent recordings can start AFTER the first action, not at reset.
            first = GameAction.from_name(action_name(frames[0]))
            actual = Agent._convert_raw_frame_data(None, env.step(first))
            result["prefix_actions"] += 1
            result["initial_recorded_action_applied"] = first.name
            if not same_observation(actual, frames[0]):
                raise ValueError(
                    "initial observation differs from checkpoint recording"
                )
        for row in range(1, freeze + 1):
            action = GameAction.from_name(action_name(frames[row]))
            actual = Agent._convert_raw_frame_data(None, env.step(action))
            if not same_observation(actual, frames[row]):
                raise ValueError(f"checkpoint restoration diverged at row {row}")
        result["prefix_observations_matched"] = freeze + 1
        agent = make_agent()
        policy = agent.mechanism_policy
        policy.enable_dynamic_contact = True
        policy.enable_option_compiler = False
        policy.enable_motion_context = False
        policy.enable_joint_navigation = True
        for row in range(audit["training_start_row"], freeze + 1):
            chosen = agent.choose_action(frames[: row + 1], frames[row])
            if chosen.name != action_name(frames[row + 1]):
                raise ValueError(f"shadow policy diverged at row {row}")
        signature = tuple(audit["supported_dynamic_signatures"][0])
        start_level = int(actual.levels_completed or 0)
        for index, step in enumerate(plan["steps"]):
            name = step["action"]
            action = GameAction.from_name(name)
            actual = Agent._convert_raw_frame_data(None, env.step(action))
            (output / f"frame-{index + 1:03}.json").write_text(actual.model_dump_json())
            state = agent.mechanism_tracker.observe(actual, action=name)
            policy.dynamic_contact.observe(state, action=name, learn=False)
            track = policy.dynamic_contact.tracks.get(signature)
            observed = {
                "player": state.player_position,
                "status": {
                    "pattern": state.status_signature,
                    "color": state.status_color,
                },
                "resource": state.resource_level // audit["resource_cost"],
                "phase": None if track is None else track.phase,
            }
            # Normalize tuples before comparing with the saved JSON candidate.
            observed = json.loads(json.dumps(observed))
            mismatches = [
                field for field in observed if observed[field] != step["after"][field]
            ]
            completed = int(actual.levels_completed or 0) > start_level
            result["checks"].append(
                {
                    "index": index,
                    "action": name,
                    "expected": step["after"],
                    "actual": observed,
                    "phase_visible": bool(track and track.observed),
                    "mismatches": mismatches,
                    "levels_completed": actual.levels_completed,
                    "events": [asdict(event) for event in state.events],
                }
            )
            result["levels_completed"] = actual.levels_completed
            if completed:
                result["status"] = "level_completed"
                break
            if mismatches:
                result["status"] = "first_prediction_mismatch"
                break
            save()
        else:
            result["status"] = "candidate_exhausted_without_completion"
    except Exception as exc:
        result.update(status="validation_error", error=repr(exc))
        raise
    finally:
        arcade.close_scorecard(card)
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.INFO)
    result = validate(args.audit, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))


if __name__ == "__main__":
    main()
