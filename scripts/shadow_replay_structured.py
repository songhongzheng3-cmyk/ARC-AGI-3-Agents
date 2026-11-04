#!/usr/bin/env python3
"""Shadow-replay StructuredAgent against observations from a recording.

The recording supplies observations only.  At every decision point the current
StructuredAgent chooses an action, which is compared with the action that
produced the next recorded observation.  Replay stops at the first divergence
because later observations would no longer represent the agent's own state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from arcengine import FrameData

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.episode_memory import EpisodeMemory  # noqa: E402
from agents.mechanism_policy import MechanismPolicy  # noqa: E402
from agents.mechanism_state_tracker import MechanismStateTracker  # noqa: E402
from agents.navigation_frontier_policy import NavigationFrontierPolicy  # noqa: E402
from agents.templates.structured_agent import StructuredAgent  # noqa: E402


def load_frames(path: Path) -> list[FrameData]:
    frames: list[FrameData] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            normalized = dict(payload.get("data", payload))
            action_input = dict(normalized.get("action_input") or {})
            action_id = action_input.get("id", 0)
            if isinstance(action_id, str):
                name = action_id.upper()
                action_input["id"] = (
                    0 if name == "RESET" else int(name.removeprefix("ACTION"))
                )
            normalized["action_input"] = action_input
            frames.append(FrameData.model_validate(normalized))
    return frames


def action_name(frame: FrameData) -> str:
    value = frame.action_input.id
    if hasattr(value, "name"):
        return str(value.name).upper()
    if isinstance(value, int):
        return "RESET" if value == 0 else f"ACTION{value}"
    return str(value).upper()


def make_agent() -> StructuredAgent:
    """Construct only the policy state needed by choose_action()."""
    agent = StructuredAgent.__new__(StructuredAgent)
    agent.navigation_policy = NavigationFrontierPolicy()
    agent.mechanism_tracker = MechanismStateTracker(
        enable_entity_tracking=os.getenv("STRUCTURED_ENTITY_TRACKING", "0") == "1",
    )
    agent.mechanism_policy = MechanismPolicy(
        enable_joint_navigation=os.getenv("STRUCTURED_JOINT_NAVIGATION", "1") != "0",
        enable_motion_context=os.getenv("STRUCTURED_MOTION_CONTEXT", "0") == "1",
        enable_dynamic_contact=os.getenv("STRUCTURED_DYNAMIC_CONTACT", "0") == "1",
        enable_option_compiler=os.getenv("STRUCTURED_OPTION_COMPILER", "0") == "1",
        enable_contact_effects=os.getenv("STRUCTURED_CONTACT_EFFECTS", "0") == "1",
        enable_entity_options=os.getenv("STRUCTURED_ENTITY_OPTIONS", "0") == "1",
    )
    agent.episode_memory = EpisodeMemory()
    agent.last_action = None
    agent.fallback_counts = defaultdict(lambda: defaultdict(int))
    agent.fallback_clicks_by_level = defaultdict(set)
    return agent


def state_snapshot(agent: StructuredAgent, frame: FrameData) -> dict[str, Any]:
    state = agent.mechanism_tracker.previous_state
    decision = agent.mechanism_policy.last_decision
    plan = agent.mechanism_policy.task_plan_controller.active_plan
    stage = None if plan is None else plan.current_stage
    return {
        "player": None if state is None else state.player_position,
        "resource_level": None if state is None else state.resource_level,
        "resource_steps": (
            None if state is None else agent.mechanism_policy._resource_steps(state)
        ),
        "status_signature": None if state is None else state.status_signature,
        "status_color": None if state is None else state.status_color,
        "events": (
            [] if state is None else [event.event_type for event in state.events]
        ),
        "objective": None if decision is None else decision.objective,
        "target": None if decision is None else decision.target,
        "route_length": None if decision is None else decision.route_length,
        "known_refills": sorted(agent.mechanism_policy.refill_cells),
        "known_pattern_changers": sorted(agent.mechanism_policy.changer_cells),
        "known_palette_changers": sorted(agent.mechanism_policy.palette_changer_cells),
        "joint_transition_count": len(
            agent.mechanism_policy.joint_mechanism_graph.transitions()
        ),
        "navigation_edge_count": len(agent.mechanism_policy.joint_navigation.edges),
        "navigation_conflicts": agent.mechanism_policy.joint_navigation.counterexamples,
        "motion_context": [
            asdict(item) for item in agent.mechanism_policy.motion_context.context
        ],
        "dynamic_contact_metrics": dict(agent.mechanism_policy.dynamic_contact.metrics),
        "dynamic_effect_samples": sum(
            agent.mechanism_policy.dynamic_contact.effect_samples.values()
        ),
        "navigation_prediction_checks": agent.mechanism_policy.joint_prediction_checks,
        "navigation_prediction_errors": agent.mechanism_policy.joint_prediction_errors,
        "navigation_search": dict(agent.mechanism_policy.joint_navigation.last_search),
        "task_plan_id": None if plan is None else plan.plan_id,
        "task_plan_stage": None if stage is None else stage.kind.value,
        "task_plan_stage_index": None if plan is None else plan.current_stage_index,
        "task_plan_invalid_reason": (
            agent.mechanism_policy.task_plan_controller.last_invalid_reason
        ),
        "levels_completed": int(frame.levels_completed or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frames = load_frames(args.recording)
    try:
        entry = next(
            index
            for index, frame in enumerate(frames)
            if int(frame.levels_completed or 0) == args.level
        )
    except StopIteration as exc:
        raise SystemExit(f"recording does not contain level {args.level}") from exc

    agent = make_agent()
    trace: list[dict[str, Any]] = []
    first_difference: dict[str, Any] | None = None
    matched = 0
    decision_index = 0
    for row_index in range(entry, len(frames) - 1):
        observation = frames[row_index]
        following = frames[row_index + 1]
        if int(observation.levels_completed or 0) != args.level:
            break

        actual = agent.choose_action(frames[entry : row_index + 1], observation).name
        expected = action_name(following)
        detail = {
            "decision_zero_based": decision_index,
            "recording_row": row_index,
            "expected_action": expected,
            "actual_action": actual,
            "matched": actual == expected,
            **state_snapshot(agent, observation),
        }
        trace.append(detail)
        if actual != expected:
            first_difference = detail
            break
        matched += 1
        decision_index += 1

        if int(following.levels_completed or 0) != args.level:
            break

    report = {
        "schema_version": 1,
        "recording": str(args.recording),
        "level_zero_based": args.level,
        "matched_action_count": matched,
        "success": first_difference is None,
        "first_difference": first_difference,
        "trace": trace,
    }
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                "success": report["success"],
                "level_zero_based": args.level,
                "matched_action_count": matched,
                "first_difference": first_difference,
            },
            ensure_ascii=False,
        )
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
