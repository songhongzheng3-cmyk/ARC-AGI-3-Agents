"""Bounded, checkpoint-restored validation of a learned pattern-changing contact.

The diagnostic route is generated from the selected prefix, not replayed from
a successful trace. Prefix restoration IS replay, solely for this local test.
No part of this script is included in the competition bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The minimal public-engine environment deliberately has no LLM frameworks.
# Match the owned bundle's isolated package, rather than importing the dev
# agents/__init__.py registry and all of its unrelated optional agents.
if __name__ == "__main__":
    package = ModuleType("agents")
    package.__path__ = [str(ROOT / "agents")]
    sys.modules["agents"] = package

from agents.contact_effects import ContactEffectMemory  # noqa: E402
from agents.dynamic_contact import DynamicContactModel, contact_relation  # noqa: E402
from agents.mechanism_state_tracker import (  # noqa: E402
    ACTION_DELTAS,
    MechanismStateTracker,
)
from agents.motion_context import MotionContextMemory  # noqa: E402
from scripts.shadow_replay_structured import action_name, load_frames  # noqa: E402


def compile_probe(
    state, dynamic, effects, *, resource_cost, horizon=10, effect="pattern"
):
    """Stop at the first contact; never plan through an unknown effect."""
    signatures = tuple(sorted(dynamic.moving))
    if not signatures or any(
        sig not in dynamic.tracks or dynamic.tracks[sig].phase is None
        for sig in signatures
    ):
        return None
    phases = tuple(dynamic.tracks[sig].phase for sig in signatures)
    start = (state.player_position, phases)
    queue = deque([(start, ())])
    visited = {start}
    forbidden = {o.cell for o in state.objects if o.static_target}
    forbidden.update(
        o.cell
        for o in state.motion_entities or ()
        if (o.color, o.area, o.shape_signature) not in dynamic.moving
    )
    horizon = min(horizon, (state.resource_level - 1) // resource_cost)
    while queue:
        (player, phases), path = queue.popleft()
        if len(path) >= horizon:
            continue
        for action, delta in ACTION_DELTAS.items():
            target = player[0] + delta[0], player[1] + delta[1]
            if target not in state.navigation_projection_cells or target in forbidden:
                continue
            successors = tuple(
                dynamic.next_phase(sig, phase, True)
                for sig, phase in zip(signatures, phases)
            )
            if None in successors:
                continue
            contacts = [
                (sig, phase, nxt)
                for sig, phase, nxt in zip(signatures, phases, successors)
                if nxt[1] == target
            ]
            pattern, color = state.status_pattern, state.status_color
            prediction = None
            if contacts:
                if len(contacts) != 1:
                    continue
                sig, phase, nxt = contacts[0]
                relation = contact_relation(player, target, phase[1], nxt[1])
                prediction = effects.predict(sig, relation, True, pattern, color)
                if prediction.pattern is None or prediction.color is None:
                    continue
                if effect == "pattern" and prediction.pattern == pattern:
                    continue
                if effect == "color" and prediction.color == color:
                    continue
                pattern, color = prediction.pattern, prediction.color
            step = {
                "action": action,
                "player": target,
                "phases": successors,
                "pattern": pattern,
                "color": color,
                "resource": state.resource_level - (len(path) + 1) * resource_cost,
                "effect": None if prediction is None else asdict(prediction),
                "hypotheses": [
                    "geometric_navigation",
                    "noncontact_status_preservation",
                ],
            }
            next_path = path + (step,)
            if prediction is not None:
                return {"signatures": signatures, "steps": next_path}
            successor = target, successors
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, next_path))
    return None


def validate(
    recording, output, *, level, freeze_row, horizon=10, effect="pattern", contacts=1
):
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    from agents.agent import Agent
    from scripts.validate_compiled_candidate import same_observation

    frames = load_frames(recording)
    rows = [
        i
        for i, f in enumerate(frames[: freeze_row + 1])
        if int(f.levels_completed or 0) == level
    ]
    if not rows or rows[-1] != freeze_row:
        raise ValueError("freeze must be in the requested level")
    tracker = MechanismStateTracker(enable_entity_tracking=True)
    dynamic, effects = DynamicContactModel(), ContactEffectMemory()
    costs = Counter()
    for row in rows:
        name = action_name(frames[row])
        before = dynamic.previous
        state = tracker.observe(frames[row], action=name)
        if before is not None and effects._valid(before, state, name):
            cost = before.resource_level - state.resource_level
            if cost > 0:
                costs[cost] += 1
        dynamic.observe(state, action=name)
        effects.observe(state, dynamic, action=name)
    if len(costs) != 1 or sum(costs.values()) < 2:
        raise ValueError("no unique supported resource decrement")
    cost = next(iter(costs))
    plan = compile_probe(
        state, dynamic, effects, resource_cost=cost, horizon=horizon, effect=effect
    )
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "recording": str(recording.resolve()),
        "recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/probe_contact_effects.py",
                "agents/contact_effects.py",
                "agents/dynamic_contact.py",
                "agents/mechanism_state_tracker.py",
                "agents/component_entities.py",
                "agents/motion_context.py",
                "scripts/shadow_replay_structured.py",
                "scripts/validate_compiled_candidate.py",
            )
        },
        "level_zero_based": level,
        "freeze_row": freeze_row,
        "resource_cost": cost,
        "effect_dimension": effect,
        "requested_contacts": contacts,
        "validated_contacts": 0,
        "new_action_budget": 24,
        "resource_cost_support": dict(costs),
        "frozen_effects": effects.diagnostics(),
        "plan": plan,
        "generated_plans": [plan] if plan else [],
        "checks": [],
        "boundary": "Checkpoint-restored local experiment, not from-scratch Agent performance or game completion.",
        "status": "no_candidate" if plan is None else "running",
    }

    def save():
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    if plan is None:
        return result
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
        recordings_dir=str(output / "engine_recordings"),
    )
    card = arcade.open_scorecard(tags=["contact-effect-checkpoint-probe"])
    try:
        env = arcade.make(frames[0].game_id, seed=0, scorecard_id=card)
        if env is None:
            raise RuntimeError("public game unavailable")
        actual = Agent._convert_raw_frame_data(None, env.observation_space)
        restored = 0
        if not same_observation(actual, frames[0]):
            actual = Agent._convert_raw_frame_data(
                None, env.step(GameAction.from_name(action_name(frames[0])))
            )
            restored += 1
            if not same_observation(actual, frames[0]):
                raise ValueError("initial recorded observation does not match")
        for row in range(1, freeze_row + 1):
            actual = Agent._convert_raw_frame_data(
                None, env.step(GameAction.from_name(action_name(frames[row])))
            )
            restored += 1
            if not same_observation(actual, frames[row]):
                raise ValueError(f"checkpoint restoration diverged at row {row}")
        result["restored_actions"] = restored
        result["matched_prefix_observations"] = freeze_row + 1
        pending = deque(plan["steps"])
        for index in range(result["new_action_budget"]):
            if not pending:
                plan = compile_probe(
                    state,
                    dynamic,
                    effects,
                    resource_cost=cost,
                    horizon=min(horizon, result["new_action_budget"] - index),
                    effect=effect,
                )
                if plan is None:
                    result["status"] = "no_followup_candidate"
                    break
                result["generated_plans"].append(plan)
                pending.extend(plan["steps"])
            expected = pending.popleft()
            actual = Agent._convert_raw_frame_data(
                None, env.step(GameAction.from_name(expected["action"]))
            )
            (output / f"frame-{index + 1:03}.json").write_text(actual.model_dump_json())
            state = tracker.observe(actual, action=expected["action"])
            dynamic.observe(state, action=expected["action"], learn=False)
            visible = MotionContextMemory.visible_positions(state)
            observed = {
                "player": state.player_position,
                "pattern": state.status_pattern,
                "color": state.status_color,
                "resource": state.resource_level,
                "phases": tuple(
                    None if sig not in dynamic.tracks else dynamic.tracks[sig].phase
                    for sig in plan["signatures"]
                ),
            }
            mismatches = [key for key in observed if observed[key] != expected[key]]
            result["checks"].append(
                {
                    "action": expected["action"],
                    "expected": expected,
                    "actual": observed,
                    "mismatches": mismatches,
                    "visible_entities": [
                        sig for sig in plan["signatures"] if sig in visible
                    ],
                    "phase_boundary": "Hidden phases are beliefs, not direct visual checks.",
                    "levels_completed": actual.levels_completed,
                }
            )
            if mismatches or int(actual.levels_completed or 0) != level:
                result["status"] = "first_prediction_mismatch"
                break
            if not pending:
                result["validated_contacts"] += 1
                if result["validated_contacts"] == contacts:
                    result["status"] = "contact_effect_validated"
                    break
            save()
        else:
            result["status"] = "new_action_budget_exhausted"
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
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--freeze-row", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--effect", choices=("pattern", "color"), default="pattern")
    parser.add_argument("--contacts", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.horizon <= 12:
        parser.error("horizon must be between 1 and 12")
    if not 1 <= args.contacts <= 4:
        parser.error("contacts must be between 1 and 4")
    logging.disable(logging.INFO)
    result = validate(
        args.recording,
        args.output,
        level=args.level,
        freeze_row=args.freeze_row,
        horizon=args.horizon,
        effect=args.effect,
        contacts=args.contacts,
    )
    print(json.dumps({k: result[k] for k in ("freeze_row", "status", "boundary")}))


if __name__ == "__main__":
    main()
