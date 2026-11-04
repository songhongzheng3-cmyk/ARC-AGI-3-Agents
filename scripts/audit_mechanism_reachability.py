#!/usr/bin/env python3
"""Audit abstract mechanism chains versus concrete observed action contracts.

Only recorded public observations are read. The historical policy is shadow
checked through the cutoff; no future frames train the frozen search model.
There is no engine, solver backend, action replay to a game, or live action.
An abstract path is a hypothesis, not an executable route or a success claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.dynamic_contact import DynamicContactModel, TrackedPhase  # noqa: E402
from agents.mechanism_state_tracker import (  # noqa: E402
    MechanismState,
    pattern_key,
)
from agents.motion_context import ObjectSignature  # noqa: E402
from agents.option_compiler import OptionCompiler  # noqa: E402
from agents.task_planning import (  # noqa: E402
    EvidenceKind,
    JointMechanismState,
    JointMechanismTransition,
    ReachabilityOption,
    ReachabilityState,
    StateReachabilityPlanner,
)
from scripts.shadow_replay_structured import (  # noqa: E402
    action_name,
    load_frames,
    make_agent,
)


@dataclass(frozen=True)
class Snapshot:
    row: int
    state: MechanismState
    tracks: dict[ObjectSignature, TrackedPhase]
    contact_signatures: tuple[ObjectSignature, ...]


def joint(state: MechanismState) -> JointMechanismState:
    return JointMechanismState(state.status_signature, state.status_color)


def pattern(value: str) -> tuple[tuple[int, ...], ...]:
    if value == "empty":
        return ()
    return tuple(tuple(int(c) for c in row) for row in value.split("/"))


def abstract_search(
    start: JointMechanismState,
    goal: JointMechanismState,
    transitions: tuple[JointMechanismTransition, ...],
    dynamic: DynamicContactModel,
    signature: ObjectSignature | None,
    *,
    include_dynamic: bool,
) -> dict[str, Any]:
    """Status-only BFS; projections are explicitly labelled, never executed."""
    queue = deque([start])
    previous: dict[JointMechanismState, tuple | None] = {start: None}
    outcomes: dict[tuple, set] = defaultdict(set)
    for edge in transitions:
        outcomes[(edge.source, edge.actuator)].add(edge.destination)
    reached = None
    while queue and len(previous) < 512:
        status = queue.popleft()
        if status.pattern == goal.pattern and (
            goal.color is None or status.color == goal.color
        ):
            reached = status
            break
        candidates = []
        for edge in transitions:
            if len(outcomes[(edge.source, edge.actuator)]) != 1:
                continue
            evidence = "observed"
            after = None
            if edge.source == status:
                after = edge.destination
            elif (
                edge.source.pattern == edge.destination.pattern
                and edge.source.color == status.color
            ):
                after = JointMechanismState(status.pattern, edge.destination.color)
                evidence = "projected_palette"
            elif (
                edge.source.color == edge.destination.color
                and edge.source.pattern == status.pattern
            ):
                after = JointMechanismState(edge.destination.pattern, status.color)
                evidence = "projected_pattern"
            if after is not None:
                candidates.append(
                    (
                        after,
                        {
                            "evidence": evidence,
                            "actuator": edge.actuator,
                            "observed_source": asdict(edge.source),
                            "approach": edge.approach,
                            "contact_action": edge.contact_action,
                            "observed_result": edge.observed_result,
                            "phase_bound": False,
                        },
                    )
                )
        if include_dynamic and signature is not None:
            output = dynamic.pattern_after(signature, pattern(status.pattern), True)
            if output is not None and dynamic.preserves_color(signature, True):
                candidates.append(
                    (
                        JointMechanismState(pattern_key(output), status.color),
                        {
                            "evidence": "dynamic_operator_projection",
                            "signature": signature,
                            "operator_candidates": sorted(
                                dynamic.operators.get((signature, True), ())
                            ),
                            "support": dynamic.effect_samples[(signature, True)],
                            "approach": None,
                            "contact_action": None,
                            "phase_bound": False,
                        },
                    )
                )
        for after, detail in candidates:
            if after in previous:
                continue
            previous[after] = (status, detail)
            queue.append(after)
    path = None
    if reached is not None:
        path = []
        cursor = reached
        while (link := previous[cursor]) is not None:
            before, detail = link
            path.append({"before": asdict(before), "after": asdict(cursor), **detail})
            cursor = before
        path.reverse()
    return {
        "reachable": path is not None,
        "visited_statuses": len(previous),
        "contacts": None if path is None else len(path),
        "path": path,
        "actionable": False,
        "boundary": "Status alignment only: no navigation, phase, resource or final target contact proof.",
    }


def observed_options(
    snapshots: list[Snapshot],
    actions: dict[int, str],
    signature: ObjectSignature,
    resource_cost: int,
) -> tuple[list[ReachabilityOption], Counter]:
    """Extract actual single-action pre/post contracts, never infer navigation."""
    options = []
    excluded = Counter()
    for before, after in zip(snapshots, snapshots[1:]):
        a, b = before.state, after.state
        action = actions[after.row]
        if action == "RESET" or a.level != b.level or a.resource_level <= 0:
            excluded["reset_or_empty_resource"] += 1
            continue
        if (
            a.player_position is None
            or b.player_position is None
            or min(a.player_confidence, b.player_confidence) < 0.8
        ):
            excluded["uncertain_player"] += 1
            continue
        source, destination = before.tracks.get(signature), after.tracks.get(signature)
        if (
            source is None
            or destination is None
            or source.phase is None
            or destination.phase is None
        ):
            excluded["missing_phase"] += 1
            continue
        dynamic_effect = after.contact_signatures == (signature,)
        refill = b.resource_level if b.resource_level > a.resource_level else None
        cost = max(
            1,
            (a.resource_level - b.resource_level + resource_cost - 1) // resource_cost,
        )
        options.append(
            ReachabilityOption(
                option_id=f"row:{before.row}->{after.row}",
                source=joint(a),
                destination=joint(b),
                evidence=EvidenceKind.DYNAMIC
                if dynamic_effect
                else EvidenceKind.OBSERVED,
                approach=a.player_position,
                contact_action=action,
                observed_result=b.player_position,
                source_phase=source.phase,
                destination_phase=destination.phase,
                contact_cost=cost,
                # An occlusion belief may be useful for diagnosis, but is not a
                # directly observed action outcome.
                projected=not source.observed or not destination.observed,
                refill_to=None if refill is None else refill // resource_cost,
                required_resource_steps=a.resource_level // resource_cost,
            )
        )
    return options, excluded


def analyze(
    recording: Path,
    *,
    level: int,
    freeze_row: int | None,
    compile_actions: bool = False,
) -> dict[str, Any]:
    frames = load_frames(recording)
    entry = next(
        i for i, frame in enumerate(frames) if int(frame.levels_completed or 0) == level
    )
    cutoff = min(
        len(frames) - 2, freeze_row if freeze_row is not None else len(frames) - 2
    )
    if cutoff <= entry or int(frames[cutoff].levels_completed or 0) != level:
        raise ValueError(
            "cutoff must lie inside the requested level after its first observation"
        )
    agent = make_agent()
    # Reproduce the recorded candidate, not an ambient experimental setting.
    agent.mechanism_policy.enable_dynamic_contact = True
    agent.mechanism_policy.enable_option_compiler = False
    agent.mechanism_policy.enable_motion_context = False
    agent.mechanism_policy.enable_joint_navigation = True
    policy = agent.mechanism_policy
    snapshots = []
    actions = {i: action_name(frame) for i, frame in enumerate(frames[: cutoff + 1])}
    transition_history: dict[tuple, JointMechanismTransition] = {}
    first_seen: dict[tuple, int] = {}
    matched = 0
    for row in range(entry, cutoff + 1):
        actual = agent.choose_action(frames[entry : row + 1], frames[row]).name
        expected = action_name(frames[row + 1])
        if actual != expected:
            raise ValueError(
                f"shadow divergence at row {row}: {actual} != {expected}; audit stopped"
            )
        matched += 1
        state = agent.mechanism_tracker.previous_state
        if state is None:
            raise ValueError(f"missing structural state at row {row}")
        snapshots.append(
            Snapshot(
                row,
                state,
                dict(policy.dynamic_contact.tracks),
                tuple(
                    contact.signature
                    for contact in policy.dynamic_contact.last_contacts
                ),
            )
        )
        for edge in policy.joint_mechanism_graph.transitions():
            key = (
                edge.source,
                edge.actuator,
                edge.destination,
                edge.approach,
                edge.contact_action,
                edge.observed_result,
            )
            transition_history[key] = edge
            first_seen.setdefault(key, row)

    model = policy.dynamic_contact
    supported = sorted(
        sig
        for sig in model.moving
        if model.effect_samples[(sig, True)] >= model.MIN_SUPPORT
    )
    signature = supported[0] if len(supported) == 1 else None
    current = snapshots[-1].state
    targets = policy._static_targets(current)
    report: dict[str, Any] = {
        "schema_version": 1,
        "recording": str(recording.resolve()),
        "recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "scripts/audit_mechanism_reachability.py",
                "scripts/shadow_replay_structured.py",
                "agents/task_planning.py",
                "agents/option_compiler.py",
                "agents/dynamic_contact.py",
                "agents/mechanism_policy.py",
                "agents/mechanism_state_tracker.py",
                "agents/templates/structured_agent.py",
            )
        },
        "policy_config": {
            "dynamic_contact": True,
            "motion_context": False,
            "joint_navigation": True,
        },
        "resource_cost": policy.resource_cost,
        "level_zero_based": level,
        "training_start_row": entry,
        "freeze_row": cutoff,
        "shadow_matched_decisions": matched,
        "future_observations_used_for_training": 0,
        "supported_dynamic_signatures": supported,
        "static_transitions": [
            {**asdict(edge), "first_observed_row": first_seen[key]}
            for key, edge in transition_history.items()
        ],
        "start": {
            "player": current.player_position,
            "status": asdict(joint(current)),
            "resource_steps": policy._resource_steps(current),
        },
        "targets": [],
        "limitations": [
            "Offline model audit, not a live run, level completion, or Kaggle score.",
            "Only one identified moving contact object is supported; other latent variables may still exist.",
            "Observed action contracts are prefix-local facts, not a proof they generalize to unseen resources or contexts.",
            "Concrete observed contracts require their actual source resource; refill availability is not inferred from a permanent coordinate.",
            "Target-cell arrival is an alignment hypothesis; actual completion requires level-advance feedback.",
        ],
    }
    options, excluded = (
        ([], Counter())
        if signature is None
        else observed_options(snapshots, actions, signature, policy.resource_cost)
    )
    report["observed_action_options"] = len(options)
    report["excluded_transitions"] = dict(excluded)
    report["occlusion_dependent_options"] = sum(o.projected for o in options)
    by_contract: dict[tuple, list[ReachabilityOption]] = defaultdict(list)
    for option in options:
        if not option.projected:
            by_contract[StateReachabilityPlanner._contract_key(option)].append(option)
    by_row = {s.row: s.state for s in snapshots}
    report["observed_contract_conflicts"] = []
    for group in by_contract.values():
        if len({StateReachabilityPlanner._outcome(o) for o in group}) <= 1:
            continue
        report["observed_contract_conflicts"].append(
            [
                {
                    **asdict(o),
                    "source_resource": by_row[
                        int(o.option_id.split(":")[1].split("->")[0])
                    ].resource_level,
                }
                for o in group
            ]
        )
    # Separately show what would go wrong if source resource and consumable
    # object availability were ignored. These are representation aliases,
    # not automatically evidence for a new motion mechanism.
    coarse_groups: dict[tuple, list[ReachabilityOption]] = defaultdict(list)
    for option in options:
        if not option.projected:
            coarse_groups[StateReachabilityPlanner._contract_key(option)[:-1]].append(
                option
            )
    report["resource_marginal_aliases"] = []
    for group in coarse_groups.values():
        if len({StateReachabilityPlanner._outcome(o) for o in group}) <= 1:
            continue
        examples = []
        for option in group:
            before_row = int(option.option_id.split(":")[1].split("->")[0])
            after_row = before_row + 1
            before, after = by_row[before_row], by_row[after_row]
            examples.append(
                {
                    "row": before_row,
                    "player": before.player_position,
                    "status": asdict(option.source),
                    "phase": option.source_phase,
                    "action": option.contact_action,
                    "destination": option.observed_result,
                    "source_resource": before.resource_level,
                    "resource_after": after.resource_level,
                    "destination_objects_visible_before": [
                        {
                            "color": item.color,
                            "area": item.area,
                            "shape": item.shape_signature,
                        }
                        for item in before.objects
                        if item.cell == option.observed_result
                        and item.object_id not in before.player_component_ids
                    ],
                }
            )
        report["resource_marginal_aliases"].append(examples)
    track = None if signature is None else model.tracks.get(signature)
    report["start"]["phase"] = None if track is None else track.phase
    planner = StateReachabilityPlanner(options)
    compiler = None
    if compile_actions and signature is not None:
        compiler = OptionCompiler(
            model,
            signature,
            transition_history.values(),
            resource_cost=policy.resource_cost,
        )
        for before, after in zip(snapshots, snapshots[1:]):
            a, b = before.tracks.get(signature), after.tracks.get(signature)
            compiler.observe(
                before.state,
                actions[after.row],
                after.state,
                None if a is None else a.phase,
                None if b is None else b.phase,
                phase_observed=bool(
                    a is not None and b is not None and a.observed and b.observed
                ),
            )
    for cell, item in sorted(targets.items()):
        if cell in policy.dynamic_target_cells:
            continue
        goal = JointMechanismState(item.pattern_signature, item.pattern_color)
        result = {
            "cell": cell,
            "goal": asdict(goal),
            "static_only": abstract_search(
                joint(current),
                goal,
                tuple(transition_history.values()),
                model,
                signature,
                include_dynamic=False,
            ),
            "with_dynamic": abstract_search(
                joint(current),
                goal,
                tuple(transition_history.values()),
                model,
                signature,
                include_dynamic=True,
            ),
        }
        if signature is None or track is None or track.phase is None:
            result["concrete"] = {
                "path": None,
                "reason": "unique_dynamic_phase_unavailable",
                "actionable": False,
            }
        else:
            state = ReachabilityState(
                current.player_position,
                joint(current),
                track.phase,
                policy._resource_steps(current),
            )
            search = planner.search(
                state,
                goal,
                goal_cell=cell,
                reserve_steps=2,
                capacity_steps=current.resource_capacity
                // max(1, policy.resource_cost),
            )
            result["concrete"] = {**asdict(search), "actionable": search.actionable}
        if compiler is not None and track is not None:
            start = compiler.bind(current, track.phase)
            candidate = None if start is None else compiler.search(start, goal, cell)
            result["compiled"] = {
                "start": None if start is None else asdict(start),
                "plan": None if candidate is None else asdict(candidate),
                "metrics": dict(compiler.metrics),
                "boundary": "Falsifiable composed action predictions, not observed game completion.",
            }
        report["targets"].append(result)
    if compiler is not None:
        tracker = deepcopy(agent.mechanism_tracker)
        tracking = deepcopy(model)
        before = current
        before_track = tracking.tracks.get(signature)
        eligible = checks = 0
        errors = []
        for row in range(cutoff + 1, len(frames)):
            action = action_name(frames[row])
            after = tracker.observe(frames[row], action=action)
            if after.level != level:
                break
            tracking.observe(after, action=action, learn=False)
            after_track = tracking.tracks.get(signature)
            if (
                action in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
                and before.resource_level > 0
            ):
                eligible += 1
                bound = compiler.bind(
                    before, None if before_track is None else before_track.phase
                )
                predicted = None if bound is None else compiler.predict(bound, action)
                if predicted is not None:
                    checks += 1
                    expected = predicted.after
                    mismatches = []
                    if expected.player != after.player_position:
                        mismatches.append("player")
                    if expected.status != joint(after):
                        mismatches.append("status")
                    if (
                        after_track is not None
                        and after_track.phase is not None
                        and expected.phase != after_track.phase
                    ):
                        mismatches.append("phase")
                    if (
                        expected.resource
                        != after.resource_level // compiler.resource_cost
                    ):
                        mismatches.append("resource")
                    if mismatches:
                        errors.append(
                            {
                                "row": row,
                                "fields": mismatches,
                                "prediction": asdict(predicted),
                                "actual": {
                                    "player": after.player_position,
                                    "status": asdict(joint(after)),
                                    "resource": after.resource_level
                                    // compiler.resource_cost,
                                    "phase": None
                                    if after_track is None
                                    else after_track.phase,
                                },
                            }
                        )
            before, before_track = after, after_track
        report["compiled_frozen_suffix"] = {
            "eligible": eligible,
            "checks": checks,
            "errors": len(errors),
            "error_details": errors,
            "coverage": checks / eligible if eligible else 0,
            "boundary": "Frozen model, factual one-step checks on actual recorded states; not candidate-policy rollouts.",
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--level", type=int, default=4, help="zero-based level")
    parser.add_argument("--freeze-row", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-actions", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(
            "output already exists; choose a new path to preserve evidence"
        )
    report = analyze(
        args.recording,
        level=args.level,
        freeze_row=args.freeze_row,
        compile_actions=args.compile_actions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=sorted) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "freeze_row": report["freeze_row"],
                "shadow_matched_decisions": report["shadow_matched_decisions"],
                "observed_action_options": report["observed_action_options"],
                "targets": [
                    {
                        "goal": target["goal"],
                        "static_reachable": target["static_only"]["reachable"],
                        "composed_reachable": target["with_dynamic"]["reachable"],
                        "concrete_reason": target["concrete"]["reason"],
                        "compiled_steps": None
                        if not (target.get("compiled", {}).get("plan") or {}).get(
                            "steps"
                        )
                        else len(target["compiled"]["plan"]["steps"]),
                    }
                    for target in report["targets"]
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
