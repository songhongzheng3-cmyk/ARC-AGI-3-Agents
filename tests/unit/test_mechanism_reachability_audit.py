from dataclasses import replace
from types import SimpleNamespace

import pytest

from agents.dynamic_contact import DynamicContactModel, TrackedPhase
from agents.mechanism_policy import MechanismPolicy
from agents.mechanism_state_tracker import MechanismState, pattern_key
from agents.task_planning import JointMechanismState, JointMechanismTransition
from scripts.audit_mechanism_reachability import (
    Snapshot,
    abstract_search,
    analyze,
    observed_options,
    pattern,
)

SIGNATURE = (0, 3, "mask")


def _state(*, resource=20, status="10/11", color=12, player=(1, 1), level=4):
    return MechanismState(
        level=level,
        game_state="NOT_FINISHED",
        layout=None,
        player_position=player,
        player_confidence=1,
        walkable_cells=frozenset(),
        blocked_cells=frozenset(),
        objects=(),
        resource_level=resource,
        resource_capacity=84,
        status_pattern=pattern(status),
        matching_target_ids=(),
        status_color=color,
    )


def _dynamic():
    model = DynamicContactModel()
    model.moving.add(SIGNATURE)
    model.operators[(SIGNATURE, True)] = {"rotate1"}
    model.effect_samples[(SIGNATURE, True)] = 2
    model.color_evidence[(SIGNATURE, True)][True] = 2
    return model


def test_abstract_chain_composes_pattern_color_and_rotation_without_action_claim():
    a = JointMechanismState("010/110/011", 12)
    b = JointMechanismState("111/001/101", 12)
    c = JointMechanismState("110/011/101", 12)
    goal = JointMechanismState("101/110/011", 8)
    edges = (
        JointMechanismTransition(a, (2, 2), b),
        JointMechanismTransition(b, (2, 2), c),
        JointMechanismTransition(a, (4, 5), JointMechanismState(a.pattern, 8)),
    )
    static = abstract_search(
        a, goal, edges, _dynamic(), SIGNATURE, include_dynamic=False
    )
    composed = abstract_search(
        a, goal, edges, _dynamic(), SIGNATURE, include_dynamic=True
    )

    assert not static["reachable"]
    assert composed["reachable"]
    assert composed["contacts"] == 5
    assert composed["path"][-1]["after"] == {"pattern": goal.pattern, "color": 8}
    assert not composed["actionable"]
    assert all(not step["phase_bound"] for step in composed["path"])


def test_abstract_search_does_not_invent_missing_static_transition():
    a = JointMechanismState("010/110/011", 12)
    b = JointMechanismState("111/001/101", 12)
    goal = JointMechanismState("101/110/011", 12)
    result = abstract_search(
        a,
        goal,
        (JointMechanismTransition(a, (2, 2), b),),
        _dynamic(),
        SIGNATURE,
        include_dynamic=True,
    )

    assert not result["reachable"]


def test_abstract_search_abstains_on_contradictory_static_effect():
    a, b, c = (JointMechanismState(value, 12) for value in ("a", "b", "c"))
    result = abstract_search(
        a,
        b,
        (
            JointMechanismTransition(a, (2, 2), b),
            JointMechanismTransition(a, (2, 2), c),
        ),
        DynamicContactModel(),
        None,
        include_dynamic=False,
    )

    assert not result["reachable"]


def test_observed_options_update_phase_and_mark_occlusion_as_projection():
    first_phase = ((1, 3), (2, 3))
    next_phase = ((2, 3), (3, 3))
    before = _state()
    after = replace(before, resource_level=16, player_position=(2, 1))
    snapshots = [
        Snapshot(10, before, {SIGNATURE: TrackedPhase(first_phase, (2, 3))}, ()),
        Snapshot(11, after, {SIGNATURE: TrackedPhase(next_phase, (3, 3), False)}, ()),
    ]
    options, exclusions = observed_options(snapshots, {11: "ACTION4"}, SIGNATURE, 4)

    assert len(options) == 1
    assert not exclusions
    assert options[0].source_phase == first_phase
    assert options[0].destination_phase == next_phase
    assert options[0].projected
    assert not options[0].actionable
    assert options[0].source.pattern == pattern_key(before.status_pattern)


def test_observed_options_exclude_restart_empty_resource_and_missing_phase():
    track = {SIGNATURE: TrackedPhase(((1, 3), (2, 3)), (2, 3))}
    snapshots = [
        Snapshot(10, _state(resource=0), track, ()),
        Snapshot(11, _state(resource=84), track, ()),
        Snapshot(12, _state(resource=80), {}, ()),
    ]
    options, exclusions = observed_options(
        snapshots, {11: "ACTION4", 12: "ACTION4"}, SIGNATURE, 4
    )

    assert options == []
    assert exclusions == {"reset_or_empty_resource": 1, "missing_phase": 1}


def test_audit_trains_only_on_prefix_and_aborts_at_first_shadow_divergence(
    monkeypatch, tmp_path
):
    from scripts import audit_mechanism_reachability as audit

    recording = tmp_path / "recording.jsonl"
    recording.write_text("{}\n")
    frames = [
        SimpleNamespace(levels_completed=4, action_input=SimpleNamespace(id=4), row=i)
        for i in range(5)
    ]
    seen = []
    policy = MechanismPolicy()
    tracker = SimpleNamespace(previous_state=None)

    def choose(history, observation):
        seen.append(observation.row)
        tracker.previous_state = _state()
        if observation.row == 3:
            pytest.fail("the suffix must not train the model")
        assert len(history) == observation.row + 1
        return SimpleNamespace(name="ACTION4")

    agent = SimpleNamespace(
        mechanism_policy=policy, mechanism_tracker=tracker, choose_action=choose
    )
    monkeypatch.setattr(audit, "load_frames", lambda _path: frames)
    monkeypatch.setattr(audit, "make_agent", lambda: agent)
    result = analyze(recording, level=4, freeze_row=2)
    assert seen == [0, 1, 2]
    assert result["future_observations_used_for_training"] == 0
    assert result["shadow_matched_decisions"] == 3

    seen.clear()
    frames[2].action_input.id = 3
    with pytest.raises(ValueError, match="shadow divergence at row 1"):
        analyze(recording, level=4, freeze_row=2)
    assert seen == [0, 1]
