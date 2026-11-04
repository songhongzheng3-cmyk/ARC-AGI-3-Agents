"""Checkpoint-restored Agent experiment; never included in competition sources.

Warm on a shadow-checked recorded prefix, then let the actual opt-in policy
choose fresh actions. Restore costs are separate from the bounded new budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
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
from arcengine import GameAction  # noqa: E402

from agents.agent import Agent  # noqa: E402
from scripts.shadow_replay_structured import (  # noqa: E402
    action_name,
    load_frames,
    make_agent,
)
from scripts.validate_compiled_candidate import same_observation  # noqa: E402


def validate(recording, output, *, freeze_row, budget, max_errors=1):
    if not 1 <= max_errors <= 3:
        raise ValueError("max_errors must be 1..3")
    frames = load_frames(recording)
    level = int(frames[freeze_row].levels_completed or 0)
    entry = next(
        i for i, f in enumerate(frames) if int(f.levels_completed or 0) == level
    )
    agent = make_agent()
    agent.mechanism_tracker.enable_entity_tracking = True
    policy = agent.mechanism_policy
    policy.enable_dynamic_contact = policy.enable_option_compiler = True
    policy.enable_contact_effects = True
    policy.enable_entity_options = False
    policy.enable_motion_context = False
    policy.enable_joint_navigation = True
    for row in range(entry, freeze_row):
        chosen = agent.choose_action(frames[entry : row + 1], frames[row])
        if chosen.name != action_name(frames[row + 1]):
            raise ValueError(f"shadow divergence at {row}")
        policy.entity_options.observe(
            agent.mechanism_tracker.previous_state,
            policy.dynamic_contact,
            action_name(frames[row]),
        )
    policy.enable_entity_options = True
    chosen = agent.choose_action(frames[entry : freeze_row + 1], frames[freeze_row])
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "recording": str(recording.resolve()),
        "recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "source_sha256": {
            n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
            for n in (
                "agents/entity_options.py",
                "agents/contact_effects.py",
                "agents/dynamic_contact.py",
                "agents/mechanism_policy.py",
                "agents/mechanism_state_tracker.py",
                "agents/component_entities.py",
                "agents/templates/structured_agent.py",
                "scripts/validate_entity_options.py",
                "scripts/shadow_replay_structured.py",
                "scripts/validate_compiled_candidate.py",
            )
        },
        "freeze_row": freeze_row,
        "new_budget": budget,
        "max_prediction_errors": max_errors,
        "shadow_matched": freeze_row - entry,
        "status": "running",
        "checks": [],
        "boundary": "Checkpoint-restored Agent diagnostic; not from-scratch game performance.",
    }

    def save():
        result["entity_options"] = policy.entity_options.diagnostics()
        result["contact_effects"] = policy.contact_effects.diagnostics()
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
        recordings_dir=str(output / "engine_recordings"),
    )
    card = arcade.open_scorecard(tags=["entity-options-checkpoint-validation"])
    try:
        env = arcade.make(frames[0].game_id, seed=0, scorecard_id=card)
        if env is None:
            raise RuntimeError("game unavailable")
        actual = Agent._convert_raw_frame_data(None, env.observation_space)
        restored = 0
        if not same_observation(actual, frames[0]):
            actual = Agent._convert_raw_frame_data(
                None, env.step(GameAction.from_name(action_name(frames[0])))
            )
            restored += 1
            if not same_observation(actual, frames[0]):
                raise ValueError("initial mismatch")
        for row in range(1, freeze_row + 1):
            actual = Agent._convert_raw_frame_data(
                None, env.step(GameAction.from_name(action_name(frames[row])))
            )
            restored += 1
            if not same_observation(actual, frames[row]):
                raise ValueError(f"restoration mismatch at {row}")
        result["restored_actions"] = restored
        result["matched_prefix_observations"] = freeze_row + 1
        history = frames[entry : freeze_row + 1]
        for index in range(budget):
            objective = policy.last_decision.objective if policy.last_decision else None
            contract = policy.entity_options.expected
            errors = policy.entity_options.metrics["prediction_errors"]
            actual = Agent._convert_raw_frame_data(None, env.step(chosen))
            (output / f"frame-{index + 1:03}.json").write_text(actual.model_dump_json())
            history.append(actual)
            selected = agent.choose_action(history, actual)
            state = agent.mechanism_tracker.previous_state
            result["checks"].append(
                {
                    "action": chosen.name,
                    "objective": objective,
                    "player": state.player_position,
                    "pattern": state.status_signature,
                    "color": state.status_color,
                    "resource": state.resource_level,
                    "levels_completed": actual.levels_completed,
                    "controller_errors": policy.entity_options.metrics[
                        "prediction_errors"
                    ],
                    "hypotheses": None if contract is None else contract.hypotheses,
                    "unknown_fields": None
                    if contract is None
                    else contract.unknown_fields,
                    "fixture_experiment": None
                    if contract is None
                    else contract.fixture,
                    "events": [event.event_type for event in state.events],
                }
            )
            if int(actual.levels_completed or 0) > level:
                result["status"] = "level_completed"
                break
            if (
                policy.entity_options.metrics["prediction_errors"] > errors
                and policy.entity_options.metrics["prediction_errors"] >= max_errors
            ):
                result["status"] = (
                    "first_prediction_mismatch"
                    if max_errors == 1
                    else "prediction_error_budget"
                )
                break
            chosen = selected
            save()
        else:
            result["status"] = "new_action_budget"
    except Exception as exc:
        result.update(status="validation_error", error=repr(exc))
        raise
    finally:
        arcade.close_scorecard(card)
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--freeze-row", type=int, required=True)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--max-errors", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.budget <= 120:
        parser.error("budget must be 1..120")
    if not 1 <= args.max_errors <= 3:
        parser.error("max-errors must be 1..3")
    logging.disable(logging.INFO)
    result = validate(
        args.recording,
        args.output,
        freeze_row=args.freeze_row,
        budget=args.budget,
        max_errors=args.max_errors,
    )
    print(json.dumps({k: result[k] for k in ("freeze_row", "status", "boundary")}))


if __name__ == "__main__":
    main()
