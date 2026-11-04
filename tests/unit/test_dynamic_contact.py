from collections import Counter
from copy import deepcopy
from dataclasses import replace

import pytest

from agents.dynamic_contact import (
    DynamicContactModel,
    TrackedPhase,
    contact_relation,
    rotate,
)
from agents.mechanism_policy import MechanismDecision, MechanismPolicy
from agents.mechanism_state_tracker import (
    GridLayout,
    MechanismEvent,
    MechanismObject,
    MechanismState,
)

PATTERN = ((0, 1, 0), (1, 1, 0), (0, 1, 1))
SIGNATURE = (2, 3, "tiny")
PHASE = ((1, 2), (2, 2))


def component(cell, **kwargs):
    x, y = cell
    values = dict(
        object_id="moving",
        cell=cell,
        color=2,
        area=3,
        shape_signature="tiny",
        bbox=(5 * x + 1, 5 * y + 1, 5 * x + 2, 5 * y + 2),
    )
    values.update(kwargs)
    return MechanismObject(**values)


def state(cell=(2, 2), *, player=(8, 8), pattern=PATTERN, **kwargs):
    values = dict(
        level=4,
        game_state="NOT_FINISHED",
        layout=GridLayout((0, 0), (5, 5), 10, 10, 3, 4, 0.95),
        player_position=player,
        player_confidence=0.99,
        walkable_cells=frozenset((x, y) for x in range(10) for y in range(10)),
        blocked_cells=frozenset(),
        objects=() if cell is None else (component(cell),),
        resource_level=60,
        resource_capacity=84,
        status_pattern=pattern,
        status_color=9,
        matching_target_ids=(),
        events=(),
    )
    values.update(kwargs)
    return MechanismState(**values)


def trained_model():
    model = DynamicContactModel()
    # Observe each shuttle phase under real movement and under blocked input.
    for i, x in enumerate([1, 2, 3, 2] * 4):
        observation = state((x, 2), player=(8 + i % 2, 8))
        for _ in range(3):
            model.observe(observation, action="ACTION4")
    return model


def ready_contact(model, pattern=PATTERN):
    model.reset_episode()
    model.observe(state((1, 2), player=(3, 4), pattern=pattern))
    before = state((2, 2), player=(3, 3), pattern=pattern)
    model.observe(before, action="ACTION1")
    return before


def learn_contact(model):
    ready_contact(model)
    model.observe(state(None, player=(3, 2), pattern=rotate(PATTERN)), action="ACTION1")


def test_heading_survives_blocked_steps_and_unknown_actions_break_it():
    model = trained_model()
    before = ready_contact(model)
    model.observe(before, action="ACTION1")
    assert model.tracks[SIGNATURE].phase == PHASE
    assert model.next_phase(SIGNATURE, PHASE, True) == ((2, 2), (3, 2))
    assert model.next_phase(SIGNATURE, PHASE, False) == PHASE
    model.observe(before, action="ACTION6")
    assert model.tracks[SIGNATURE].phase is None


def test_conflicting_motion_abstains_even_with_dominant_outcome():
    model = trained_model()
    model.motion[(SIGNATURE, PHASE, True)][((2, 2), (1, 2))] += 1
    assert model.next_phase(SIGNATURE, PHASE, True) is None


def test_occluded_positions_are_checked_beliefs_not_motion_training():
    model = trained_model()
    ready_contact(model)
    evidence = deepcopy(model.motion)
    model.observe(state(None, player=(3, 2), pattern=rotate(PATTERN)), action="ACTION1")
    assert not model.tracks[SIGNATURE].observed
    assert model.motion == evidence
    model.observe(
        state((2, 2), player=(4, 2), pattern=rotate(PATTERN)), action="ACTION4"
    )
    assert model.metrics["reappearance_checks"] == 1
    assert model.metrics["reappearance_errors"] == 0
    assert model.motion == evidence
    assert model.tracks[SIGNATURE].phase == ((3, 2), (2, 2))


def test_wrong_reappearance_discards_phase_without_learning_inferred_motion():
    model = trained_model()
    learn_contact(model)
    evidence = deepcopy(model.motion)
    model.observe(state((1, 2), player=(4, 2)), action="ACTION4")
    assert model.metrics["reappearance_errors"] == 1
    assert model.tracks[SIGNATURE].phase is None
    assert model.motion == evidence


def test_duplicate_identity_cannot_be_treated_as_occlusion():
    model = trained_model()
    ready_contact(model)
    duplicate = state(
        None,
        player=(3, 2),
        objects=(
            component((1, 2)),
            component((2, 2), object_id="copy"),
        ),
    )
    model.observe(duplicate, action="ACTION1")
    assert not model.last_contacts
    assert SIGNATURE not in model.tracks


def test_stationary_overlap_does_not_poison_moving_rotation():
    model = trained_model()
    for _ in range(2):
        learn_contact(model)
        model.observe(
            state(None, player=(3, 2), pattern=rotate(PATTERN)), action="ACTION1"
        )
    assert model.operators[(SIGNATURE, True)] == {"rotate1"}
    assert model.pattern_after(SIGNATURE, PATTERN) == rotate(PATTERN)
    assert model.pattern_after(SIGNATURE, PATTERN, moved=False) == PATTERN
    assert model.preserves_color(SIGNATURE)


def test_operator_contradiction_abstains_instead_of_majority_guess():
    model = trained_model()
    for _ in range(2):
        learn_contact(model)
    ready_contact(model)
    model.observe(state(None, player=(3, 2), pattern=PATTERN), action="ACTION1")
    assert model.pattern_after(SIGNATURE, PATTERN) is None


def test_frozen_observation_updates_belief_but_never_rules():
    model = trained_model()
    learn_contact(model)
    ready_contact(model)
    evidence = deepcopy(
        (
            model.motion,
            model.operators,
            model.effect_samples,
            model.color_evidence,
            model.moving,
        )
    )
    model.observe(
        state(None, player=(3, 2), pattern=PATTERN), action="ACTION1", learn=False
    )
    assert not model.tracks[SIGNATURE].observed
    assert evidence == (
        model.motion,
        model.operators,
        model.effect_samples,
        model.color_evidence,
        model.moving,
    )


def test_episode_keeps_rules_but_level_boundary_clears_them():
    model = trained_model()
    model.reset_episode()
    assert model.motion and not model.tracks and model.previous is None
    model.observe(state())
    model.observe(state(level=5), action="ACTION1")
    assert not model.motion and not model.operators and not model.moving


def test_contact_search_requires_known_phase_and_respects_horizon_and_effect():
    model = trained_model()
    for _ in range(2):
        learn_contact(model)
    before = ready_contact(model)

    def destination(player, action):
        return (player[0], player[1] - 1)

    route = model.contact_route(
        before, SIGNATURE, destination, ["ACTION1"], goal_pattern=rotate(PATTERN)
    )
    assert route is not None and route[0] == ("ACTION1",)
    assert route[1].relation == "meet_after_moving"
    assert (
        model.contact_route(before, SIGNATURE, destination, ["ACTION1"], horizon=0)
        is None
    )
    assert (
        model.contact_route(
            before, SIGNATURE, destination, ["ACTION1"], goal_pattern=PATTERN
        )
        is None
    )
    model.tracks[SIGNATURE] = TrackedPhase(None, (2, 2))
    assert model.contact_route(before, SIGNATURE, destination, ["ACTION1"]) is None


def test_stationary_overlap_is_not_a_new_contact_experiment():
    model = trained_model()
    learn_contact(model)
    after = model.previous
    assert (
        model.contact_route(
            after, SIGNATURE, lambda player, action: player, ["ACTION1"]
        )
        is None
    )


def test_contact_relations_do_not_conflate_crossing_and_overlap():
    assert contact_relation((1, 1), (2, 1), (2, 1), (1, 1)) == "crossing"
    assert contact_relation((2, 1), (2, 1), (2, 1), (2, 1)) == "persistent_overlap"
    assert contact_relation((1, 1), (2, 1), (2, 1), (3, 1)) == "enter_departing_cell"


def test_policy_flag_and_reset_lifetime():
    assert not MechanismPolicy().enable_dynamic_contact
    policy = MechanismPolicy(enable_dynamic_contact=True)
    policy.dynamic_contact = trained_model()
    policy.note_reset()
    assert policy.dynamic_contact.motion and not policy.dynamic_contact.tracks
    policy.reset()
    assert policy.enable_dynamic_contact and not policy.dynamic_contact.motion


def test_policy_selects_supported_rotation_but_not_low_resource_or_aligned_goal():
    policy = MechanismPolicy(enable_dynamic_contact=True)
    model = policy.dynamic_contact = trained_model()
    for _ in range(2):
        learn_contact(model)
    before = ready_contact(model)
    target = component(
        (7, 7), static_target=True, internal_pattern=rotate(PATTERN), pattern_color=9
    )
    targets = {target.cell: target}
    decision = policy._dynamic_contact_decision(
        before, targets, list(policy.DIRECTION_NAMES)
    )
    assert decision is not None and decision.action == "ACTION1"
    assert decision.objective == "align_dynamic_pattern"
    assert (
        policy._dynamic_contact_decision(
            replace(before, resource_level=4), targets, list(policy.DIRECTION_NAMES)
        )
        is None
    )
    assert (
        policy._dynamic_contact_decision(
            replace(before, status_pattern=rotate(PATTERN)),
            targets,
            list(policy.DIRECTION_NAMES),
        )
        is None
    )


def test_dynamic_effect_is_not_assigned_to_static_cell_or_graph():
    policy = MechanismPolicy(enable_dynamic_contact=True)
    model = policy.dynamic_contact = trained_model()
    before = ready_contact(model)
    policy._previous_navigation_state = before
    policy._last_player = before.player_position
    policy.last_status, policy.last_status_color = (
        before.status_signature,
        before.status_color,
    )
    policy.last_resource = before.resource_level
    policy._set_decision(
        before, MechanismDecision("ACTION1", "discover_pattern_changer", (3, 2), 1)
    )
    policy._update_evidence(
        state(
            None,
            player=(3, 2),
            pattern=rotate(PATTERN),
            events=(MechanismEvent("status_pattern_changed"),),
        ),
        actual_action="ACTION1",
    )
    assert model.effect_samples[(SIGNATURE, True)] == 1
    assert (3, 2) not in policy.changer_cells
    assert not policy.joint_mechanism_graph.transitions()


def test_stationary_operator_insufficient_support_abstains():
    model = DynamicContactModel()
    model.operators[(SIGNATURE, False)] = {"rotate0"}
    model.effect_samples = Counter({(SIGNATURE, False): 1})
    assert model.pattern_after(SIGNATURE, PATTERN, moved=False) is None


@pytest.mark.parametrize("method", ["_depart_action", "_continue_actuator_probe"])
def test_actuator_departure_never_repeats_known_blocked_edge_in_experiment(method):
    policy = MechanismPolicy(enable_dynamic_contact=True)
    observation = state(player=(4, 5))
    policy.observed_transitions[((4, 5), "ACTION1")] = (4, 5)
    policy.active_actuator_probe = ((4, 5), "palette")
    args = [observation, list(policy.DIRECTION_NAMES)]
    if method == "_depart_action":
        args.append((4, 5))
    decision = getattr(policy, method)(*args)
    assert decision is not None and decision.action == "ACTION2"


def test_palette_discovery_excludes_only_moving_tracks_not_static_candidates():
    policy = MechanismPolicy(enable_dynamic_contact=True)
    model = policy.dynamic_contact = trained_model()
    for _ in range(2):
        learn_contact(model)
    before = ready_contact(model)
    static = component(
        (4, 3), object_id="candidate", shape_signature="unknown", color=6
    )
    before = replace(before, objects=(component((2, 2)), static))
    model.tracks[(6, 3, "unknown")] = TrackedPhase(None, static.cell)
    target = component(
        (7, 7), static_target=True, internal_pattern=PATTERN, pattern_color=8
    )
    decision = policy._dynamic_contact_decision(
        before, {target.cell: target}, list(policy.DIRECTION_NAMES)
    )
    assert decision is not None
    assert decision.objective == "discover_palette_changer"
    assert decision.target == static.cell
