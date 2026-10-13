#!/usr/bin/env python3
"""Verify observation-trained shape/contact planning on ka59 level two."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".external/ls20-solver/src"))

from ls20.search import astar_search, bfs_search  # noqa: E402

from agents.agent import Agent  # noqa: E402
from agents.selectable_navigation_model import SelectableNavigationModel  # noqa: E402
from agents.shaped_contact_model import (  # noqa: E402
    ShapedContactModel,
    ShapedSceneBinding,
)
from agents.templates.selectable_navigation_agent import (  # noqa: E402
    SelectableNavigationAgent,
)
from scripts.run_selectable_transfer import (  # noqa: E402
    binding_for,
    read_frames,
    train,
)
from scripts.verify_ls20_baseline import (  # noqa: E402
    git_info,
    snapshot_sources,
    write_json,
)


def train_shapes(model, paths):
    trajectories = []
    for path in paths:
        frames = [f for f in read_frames(path) if f.levels_completed == 1]
        previous = model.binding.parse(frames[0])
        for frame in frames[1:]:
            after = model.binding.parse(frame)
            action = frame.action_input.id.value
            model.learn(previous, action, after)
            trajectories.append((previous, action, after))
            previous = after
    # Replay after learning: this checks generalised rules against all existing
    # observations, including no-ops, rather than merely fitting changed frames.
    for before, action, after in trajectories:
        if action == 6:
            continue
        predicted = model.contact_destination(before, action)
        if predicted is None:
            predicted = model.ordinary_destination(before, action) or before
        if predicted != after:
            raise ValueError("training trajectory prediction mismatch")
    return len(trajectories)


def execute_level(agent, model, output):
    frame = agent.frames[-1]
    plans, transitions = [], []
    started = time.monotonic()
    reason = "action_budget_exhausted"

    def finish():
        result = {
            "reason": reason,
            "setup_actions": 11,
            "second_level_actions": len(transitions),
            "total_actions": agent.action_counter,
            "levels_completed": frame.levels_completed,
            "final_state": frame.state.name,
            "elapsed_seconds": time.monotonic() - started,
            "recording": str(agent.recorder.filename),
            "plans": plans,
            "transitions": transitions,
        }
        write_json(output / "result.json", result)
        agent.cleanup()
        return result

    try:
        while len(transitions) < 200:
            current = model.binding.parse(frame)
            plan = astar_search(
                current,
                model.is_goal,
                model.macro_successors,
                lambda state: 0,
                max_expansions=50000,
            )
            plans.append(plan.to_dict())
            if not plan.found:
                reason = "no_plan_under_current_model"
                break
            if not plan.actions:
                reason = "goal_hypothesis_not_confirmed_by_environment"
                break
            for token in plan.actions[0].split("|"):
                if len(transitions) >= 200:
                    break
                choices = {t: s for t, s, _ in model.successors(current)}
                if token not in choices:
                    raise ValueError("macro primitive is absent from model")
                expected = choices[token]
                if token.startswith("select:"):
                    index = int(token.split(":")[1])
                    x, y = model.binding.pixel(current.positions[index])
                    action_id, data = 6, {"x": x + 1, "y": y + 1}
                else:
                    action_id, data = int(token.split(":")[1]), {}
                if action_id not in frame.available_actions:
                    raise ValueError("planned action is not legal")
                raw = agent.arc_env.step(
                    GameAction.from_id(action_id),
                    data=data,
                    reasoning={
                        "source": "observation-trained-shaped-contact",
                        "prediction": expected.to_dict(),
                    },
                )
                if raw is None:
                    raise RuntimeError("missing observation")
                frame = agent._convert_raw_frame_data(raw)
                agent.append_frame(frame)
                agent.action_counter += 1
                row = {
                    "action": action_id,
                    "data": data,
                    "expected": expected.to_dict(),
                    "levels_completed": frame.levels_completed,
                    "state": frame.state.name,
                    "prediction_checked": False,
                }
                transitions.append(row)
                if frame.levels_completed > 1:
                    reason = "level_completed"
                    return finish()
                if frame.state in (GameState.WIN, GameState.GAME_OVER):
                    reason = frame.state.name.lower()
                    return finish()
                actual = model.binding.parse(frame)
                row.update(
                    actual=actual.to_dict(),
                    prediction_checked=True,
                    prediction_matches=actual == expected,
                )
                if actual != expected:
                    reason = "prediction_mismatch_stop"
                    return finish()
                current = actual
            if current != plan.states[1]:
                if len(transitions) >= 200:
                    break
                raise ValueError("macro endpoint mismatch")
    except ValueError as exc:
        reason = str(exc)
    return finish()


def main():
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env.example")
    load_dotenv(ROOT / ".env", override=True)
    os.environ["OPERATION_MODE"] = "offline"
    output = (
        ROOT
        / "sampling_runs/shaped_transfer"
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
    first_paths = sorted(movement.glob("ka59*.direction-*.recording.jsonl")) + sorted(
        contact.glob("ka59*.first-actor-touch-other-actor*.recording.jsonl")
    )
    # Explicit development evidence; exclude future runs to preserve provenance.
    studies = [
        "20261008T092841265011Z",
        "20261008T092949849568Z",
        "20261008T093229405013Z",
        "20261008T093415559186Z",
        "20261008T093516733586Z",
        "20261008T093619834637Z",
    ]
    shape_paths = [
        p
        for study in studies
        for p in sorted(
            (ROOT / "sampling_runs/ka59_level2_probes" / study / "recordings").glob(
                "*jsonl"
            )
        )
    ]
    if len(first_paths) != 5 or len(shape_paths) != 12:
        raise ValueError("explicit training evidence is incomplete")
    write_json(
        output / "manifest.json",
        {
            "game": "ka59-38d34dbb",
            "seed": 0,
            "operation_mode": "offline",
            "second_level_action_limit": 200,
            "search_expansion_limit": 50000,
            "search": "unchanged reference A* with zero heuristic, navigation macros weighted by primitive action cost",
            "project": git_info(ROOT),
            "backend": git_info(ROOT / ".external/ls20-solver"),
            "source_sha256": snapshot_sources(output, ROOT / ".external/ls20-solver"),
            "training_recordings": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in first_paths + shape_paths
            },
            "scope": "development-trace-trained second-level experiment, not zero-shot or full-game agent",
            "untouched_holdout": ["cn04", "tn36"],
        },
    )
    print(f"RESULT_DIRECTORY {output}", flush=True)
    arcade = Arcade(operation_mode=OperationMode.OFFLINE)
    card = arcade.open_scorecard(tags=["shaped-contact-transfer"])
    try:
        env = arcade.make("ka59-38d34dbb", seed=0, scorecard_id=card)
        if env is None:
            raise RuntimeError("missing environment")
        initial = Agent._convert_raw_frame_data(None, env.observation_space)
        model = SelectableNavigationModel(binding_for(initial), initial)
        train(model, first_paths)
        agent = SelectableNavigationAgent(
            card_id=card,
            game_id="ka59-38d34dbb",
            agent_name="shaped-transfer",
            ROOT_URL="local",
            record=True,
            arc_env=env,
            model=model,
            search=bfs_search,
        )
        agent.main()
        if not agent.result["level_advanced"] or agent.action_counter != 11:
            raise RuntimeError("first-level model-planned preparation failed")
        shaped = ShapedContactModel(
            ShapedSceneBinding(agent.frames[-1]), model.directions
        )
        checked = train_shapes(shaped, shape_paths)
        write_json(output / "initial_model.json", shaped.to_dict())
        print(f"TRAINING_TRANSITIONS_CHECKED {checked}", flush=True)
        result = execute_level(agent, shaped, output)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("plans", "transitions")}
            ),
            flush=True,
        )
        return 0 if result["levels_completed"] == 2 else 1
    finally:
        score = arcade.close_scorecard(card)
        if score is not None:
            write_json(output / "local_scorecard.json", score.model_dump(mode="json"))


if __name__ == "__main__":
    raise SystemExit(main())
