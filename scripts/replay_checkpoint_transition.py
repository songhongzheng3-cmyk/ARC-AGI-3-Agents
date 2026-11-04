"""Repeat one recorded checkpoint action from an exactly matched prefix.

This local diagnostic uses only the public recording and local public engine.
It does not invoke the policy, update a competition run, or submit anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from arc_agi import Arcade, OperationMode
from arcengine import FrameData, GameAction

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agent import Agent  # noqa: E402
from agents.mechanism_state_tracker import MechanismStateTracker  # noqa: E402
from agents.motion_context import MotionContextMemory  # noqa: E402
from scripts.audit_action_animation import audit_animation  # noqa: E402
from scripts.shadow_replay_structured import action_name, load_frames  # noqa: E402
from scripts.validate_compiled_candidate import same_observation  # noqa: E402


def _context_payload(context) -> list[dict]:
    """Serialize the passive motion context at one action boundary."""
    return [
        {
            "signature": list(item.signature),
            "previous": item.previous,
            "current": item.current,
        }
        for item in context
    ]


def _state_payload(state, context, frame=None) -> dict:
    """Keep action-boundary signals compact and JSON serializable."""
    return {
        "player": state.player_position,
        "player_confidence": state.player_confidence,
        "level": state.level,
        "resource": state.resource_level,
        "status": state.status_signature,
        "color": state.status_color,
        "game_state": state.game_state,
        "available_actions": (
            None if frame is None else list(frame.available_actions)
        ),
        "objects": [
            {
                "object_id": item.object_id,
                "color": item.color,
                "cell": item.cell,
                "area": item.area,
                "shape": item.shape_signature,
                "static_target": item.static_target,
            }
            for item in state.objects
        ],
        "motion_context": _context_payload(context),
        "events": [
            {
                "type": event.event_type,
                "before": event.before,
                "after": event.after,
                "cells": event.cells,
                "object_id": event.object_id,
                "confidence": event.confidence,
            }
            for event in state.events
        ],
    }


def _load_result(path: Path) -> dict:
    return json.loads(path.read_text())


def _resolve_base_result(branch_result: Path, explicit: Path | None) -> Path | None:
    """Return the prerequisite branch result for a sampled branch, if any."""
    if explicit is not None:
        return explicit
    saved = _load_result(branch_result)
    referenced = saved.get("branch_result")
    if not referenced:
        return None
    path = Path(referenced)
    if not path.is_absolute():
        path = branch_result.parent / path
    return path


def repeat_transition(recording: Path, checkpoint_result: Path, output: Path,
                      *, check_number: int, trials: int,
                      prelude_actions: tuple[str, ...] = (),
                      base_checkpoint_result: Path | None = None,
                      target_action: str | None = None) -> dict:
    saved = _load_result(checkpoint_result)
    base_result_path = _resolve_base_result(
        checkpoint_result, base_checkpoint_result
    )
    base_saved = _load_result(base_result_path) if base_result_path else None
    frames = load_frames(recording)
    checks = saved["checks"]
    if not 1 <= check_number <= len(checks):
        raise ValueError("check number is outside the saved experiment")
    if not 1 <= trials <= 3:
        raise ValueError("trials must be 1..3")
    if any(action not in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
           for action in prelude_actions):
        raise ValueError("prelude actions must be ACTION1..ACTION4")
    freeze = saved["freeze_row"]
    prefix_checks = checks[: check_number - 1]
    target = checks[check_number - 1]
    if target_action is not None and target_action not in {
        "ACTION1", "ACTION2", "ACTION3", "ACTION4"
    }:
        raise ValueError("target action must be ACTION1..ACTION4")
    selected_target_action = target_action or target["action"]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "recording": str(recording.resolve()),
        "recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "checkpoint_result": str(checkpoint_result.resolve()),
        "checkpoint_result_sha256": hashlib.sha256(
            checkpoint_result.read_bytes()
        ).hexdigest(),
        "base_checkpoint_result": (
            str(base_result_path.resolve()) if base_result_path else None
        ),
        "base_replayed_experiment_actions": (
            len(base_saved["checks"]) if base_saved else 0
        ),
        "freeze_row": freeze,
        "replayed_experiment_actions": len(prefix_checks),
        "target_check_number": check_number,
        "recorded_target_action": target["action"],
        "target_action": selected_target_action,
        "prelude_actions": list(prelude_actions),
        "trials": [],
        "boundary": "Local checkpoint replay only; not competition evaluation.",
    }
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
        recordings_dir=str(output / "engine_recordings"),
    )
    for trial in range(1, trials + 1):
        card = arcade.open_scorecard(
            tags=["checkpoint-transition-repeat", f"trial-{trial}"]
        )
        try:
            env = arcade.make(frames[0].game_id, seed=0, scorecard_id=card)
            if env is None:
                raise RuntimeError("public game unavailable")
            actual = Agent._convert_raw_frame_data(None, env.observation_space)
            if not same_observation(actual, frames[0]):
                actual = Agent._convert_raw_frame_data(
                    None,
                    env.step(GameAction.from_name(action_name(frames[0]))),
                )
                if not same_observation(actual, frames[0]):
                    raise ValueError("initial recording observation mismatch")
            for row in range(1, freeze + 1):
                actual = Agent._convert_raw_frame_data(
                    None, env.step(GameAction.from_name(action_name(frames[row])))
                )
                if not same_observation(actual, frames[row]):
                    raise ValueError(f"checkpoint restoration diverged at {row}")

            tracker = MechanismStateTracker(enable_entity_tracking=True)
            motion_memory = MotionContextMemory()
            for frame in frames[: freeze + 1]:
                action = action_name(frame)
                state = tracker.observe(frame, action=action)
                motion_memory.observe(state, reset=action == "RESET")

            # A sampled branch is recorded after the checkpoint experiment that
            # produced its source state. Replaying only the sampled actions
            # changes the state before action 1 and makes every comparison
            # invalid, so restore the prerequisite branch first.
            if base_saved:
                if int(base_saved["freeze_row"]) != freeze:
                    raise ValueError("base and sampled checkpoints use different freeze rows")
                for index, base_check in enumerate(base_saved["checks"], start=1):
                    action = base_check["action"]
                    actual = Agent._convert_raw_frame_data(
                        None, env.step(GameAction.from_name(action))
                    )
                    saved_frame = FrameData.model_validate_json(
                        (base_result_path.parent / f"frame-{index:03}.json").read_text()
                    )
                    if not same_observation(actual, saved_frame):
                        raise ValueError(
                            f"base experimental prefix diverged at action {index}"
                        )
                    state = tracker.observe(actual, action=action)
                    motion_memory.observe(state, reset=action == "RESET")

            for index, saved_check in enumerate(prefix_checks, start=1):
                action = saved_check["action"]
                actual = Agent._convert_raw_frame_data(
                    None, env.step(GameAction.from_name(action))
                )
                saved_frame = FrameData.model_validate_json(
                    (checkpoint_result.parent / f"frame-{index:03}.json")
                    .read_text()
                )
                if not same_observation(actual, saved_frame):
                    raise ValueError(f"experimental prefix diverged at action {index}")
                state = tracker.observe(actual, action=action)
                motion_memory.observe(state, reset=action == "RESET")

            source_context = motion_memory.context
            source_state = state
            prelude_observations = []
            for action in prelude_actions:
                actual = Agent._convert_raw_frame_data(
                    None, env.step(GameAction.from_name(action))
                )
                state = tracker.observe(actual, action=action)
                context = motion_memory.observe(state, reset=action == "RESET")
                prelude_observations.append(
                    {"action": action, **_state_payload(state, context, actual)}
                )
            if prelude_actions:
                source_context = motion_memory.context
                source_state = state

            source = actual.model_dump_json()
            before_hash = hashlib.sha256(source.encode()).hexdigest()
            source_path = output / f"trial-{trial:02d}-source.json"
            source_path.write_text(source + "\n")
            source_gameplay_hash = hashlib.sha256(
                json.dumps(
                    actual.model_dump(mode="json", exclude={"guid"}),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            observed = Agent._convert_raw_frame_data(
                None, env.step(GameAction.from_name(selected_target_action))
            )
            observation_json = observed.model_dump_json()
            action_path = output / f"trial-{trial:02d}-action.json"
            action_path.write_text(observation_json + "\n")
            animation = audit_animation(tracker, observed)
            target_state = tracker.observe(observed, action=selected_target_action)
            target_context = motion_memory.observe(
                target_state, reset=selected_target_action == "RESET"
            )
            trial_result = {
                "trial": trial,
                "prefix_observation_sha256": before_hash,
                "target_action": selected_target_action,
                "source_player": animation["source_player"],
                "commanded_neighbor": animation["commanded_neighbor"],
                "final_player": animation["final_player"],
                "subframe_count": len(animation["subframes"]),
                "prelude_observations": prelude_observations,
                "source_gameplay_observation_sha256": source_gameplay_hash,
                "source_state": _state_payload(source_state, source_context, actual),
                "observation_sha256": hashlib.sha256(
                    observation_json.encode()
                ).hexdigest(),
                "gameplay_observation_sha256": hashlib.sha256(
                    json.dumps(
                        observed.model_dump(mode="json", exclude={"guid"}),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "animation": animation,
                "target_state": _state_payload(target_state, target_context, observed),
                "file": action_path.name,
                "source_file": source_path.name,
            }
            report["trials"].append(trial_result)
            (output / f"trial-{trial:02d}-animation.json").write_text(
                json.dumps(animation, indent=2) + "\n"
            )
        finally:
            arcade.close_scorecard(card)

    report["deterministic_raw_response"] = len(
        {trial["observation_sha256"] for trial in report["trials"]}
    ) == 1
    report["deterministic_gameplay_observation"] = len(
        {
            trial["gameplay_observation_sha256"]
            for trial in report["trials"]
        }
    ) == 1
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("checkpoint_result", type=Path)
    parser.add_argument("--check-number", type=int, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--prelude", nargs="*", default=[], metavar="ACTION",
        help="optional ACTION1..ACTION4 sequence before the target action",
    )
    parser.add_argument(
        "--base-checkpoint-result", type=Path,
        help="optional prerequisite branch result; inferred from branch_result when omitted",
    )
    parser.add_argument(
        "--target-action", choices=("ACTION1", "ACTION2", "ACTION3", "ACTION4"),
        help="replace the recorded target action for a controlled counterfactual",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.INFO)
    report = repeat_transition(
        args.recording,
        args.checkpoint_result,
        args.output,
        check_number=args.check_number,
        trials=args.trials,
        prelude_actions=tuple(args.prelude),
        base_checkpoint_result=args.base_checkpoint_result,
        target_action=args.target_action,
    )
    print(
        json.dumps(
            {
                "target_action": report["target_action"],
                "deterministic_raw_response": report[
                    "deterministic_raw_response"
                ],
                "deterministic_gameplay_observation": report[
                    "deterministic_gameplay_observation"
                ],
                "trials": [
                    {
                        "source": trial["source_player"],
                        "commanded": trial["commanded_neighbor"],
                        "final": trial["final_player"],
                        "gameplay_sha256": trial[
                            "gameplay_observation_sha256"
                        ],
                    }
                    for trial in report["trials"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
