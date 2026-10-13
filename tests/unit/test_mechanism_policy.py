from dataclasses import replace

from agents.mechanism_policy import (
    MechanismDecision,
    MechanismPolicy,
    PendingInteraction,
)
from agents.mechanism_state_tracker import (
    GridLayout,
    MechanismEvent,
    MechanismObject,
    MechanismState,
)
from agents.task_planning import JointMechanismState, PlanStageKind


def _object(
    cell,
    area,
    color=11,
    pattern=(),
    static=False,
    pattern_color=None,
):
    x, y = cell
    return MechanismObject(
        object_id=f"{cell}-{area}-{color}",
        color=color,
        cell=cell,
        bbox=(x * 5, y * 5, x * 5 + 2, y * 5 + 2),
        area=area,
        shape_signature=f"area={area}",
        internal_pattern=pattern,
        multi_palette=bool(pattern),
        floor_ratio=0.5,
        static_target=static,
        pattern_color=pattern_color,
    )


def _state(
    player=(1, 1),
    resource=40,
    status=((1, 0), (0, 1)),
    objects=(),
    events=(),
    status_color=None,
    player_component_ids=(),
    player_bbox=None,
):
    cells = frozenset((x, y) for x in range(5) for y in range(5))
    return MechanismState(
        level=1,
        game_state="NOT_FINISHED",
        layout=GridLayout((0, 0), (5, 5), 5, 5, 3, 4, 0.95),
        player_position=player,
        player_confidence=0.95,
        walkable_cells=cells,
        blocked_cells=frozenset(),
        objects=tuple(objects),
        resource_level=resource,
        resource_capacity=40,
        status_pattern=status,
        matching_target_ids=(),
        events=tuple(events),
        status_color=status_color,
        player_component_ids=tuple(player_component_ids),
        player_bbox=player_bbox,
    )


def test_prefers_multi_part_candidate_and_learns_changer():
    policy = MechanismPolicy()
    changer = [
        _object((3, 1), 3, color=8),
        _object((3, 1), 2, color=9),
    ]
    refill = [_object((1, 4), 8)]
    action = policy.choose(_state(objects=changer + refill), [1, 2, 3, 4])
    assert action.name == "ACTION4"

    event = MechanismEvent("status_pattern_changed", action="ACTION4")
    policy.choose(
        _state(player=(3, 1), resource=36, objects=changer + refill, events=[event]),
        [1, 2, 3, 4],
    )
    assert policy.changer_cells == {(3, 1)}


def test_candidate_extraction_ignores_player_components_and_cross_cell_walls():
    policy = MechanismPolicy()
    player_top = MechanismObject(
        **{
            **_object((1, 1), 10, color=12).__dict__,
            "object_id": "player-top",
            "bbox": (5, 5, 9, 6),
        }
    )
    player_bottom = MechanismObject(
        **{
            **_object((1, 2), 15, color=9).__dict__,
            "object_id": "player-bottom",
            "bbox": (5, 7, 9, 9),
        }
    )
    wall = MechanismObject(
        **{
            **_object((3, 1), 5, color=1).__dict__,
            "bbox": (15, 5, 21, 5),
        }
    )
    compact = _object((4, 1), 8, color=11)
    state = _state(
        player=(1, 1),
        objects=[player_top, player_bottom, wall, compact],
        player_component_ids=("player-top", "player-bottom"),
    )

    assert policy._candidate_cells(state) == [(4, 1)]


def test_attributes_effect_to_candidate_hidden_under_player():
    policy = MechanismPolicy()
    actuator = [
        _object((3, 1), 3, color=8),
        _object((3, 1), 2, color=9),
    ]
    policy.choose(_state(player=(2, 1), objects=actuator), [1, 2, 3, 4])

    policy.choose(
        _state(
            player=(3, 1),
            objects=(),
            events=[MechanismEvent("status_pattern_changed", action="ACTION4")],
        ),
        [1, 2, 3, 4],
    )

    assert policy.changer_cells == {(3, 1)}
    assert (3, 1) in policy.contacted_cells


def test_attributes_refill_to_candidate_hidden_under_player():
    policy = MechanismPolicy()
    refill = [_object((3, 1), 8, color=11)]
    policy.choose(_state(player=(2, 1), resource=8, objects=refill), [1, 2, 3, 4])

    policy.choose(
        _state(
            player=(3, 1),
            resource=40,
            objects=(),
            events=[MechanismEvent("resource_increased", action="ACTION4")],
        ),
        [1, 2, 3, 4],
    )

    assert policy.refill_cells == {(3, 1)}


def test_keeps_contextual_relocation_out_of_navigation_graph():
    policy = MechanismPolicy()
    policy.level = 1
    policy._last_player = (1, 1)
    policy.pending_interaction = PendingInteraction(
        objective="visit_uncontacted_candidate",
        target=(2, 1),
        source=(1, 1),
        destination=(2, 1),
        action="ACTION4",
    )

    state = _state(player=(4, 4))
    policy._update_evidence(state)

    assert ((1, 1), "ACTION4") not in policy.observed_transitions
    assert policy.special_transitions[((1, 1), "ACTION4")] == (4, 4)
    assert policy.contextual_special_transitions[
        ((1, 1), "ACTION4", "empty", None)
    ] == (4, 4)


def test_records_local_move_and_blocked_transition_for_routing():
    policy = MechanismPolicy()
    policy.level = 1
    policy._last_player = (1, 1)
    policy.pending_interaction = PendingInteraction(
        objective="visit_uncontacted_candidate",
        target=(2, 1),
        source=(1, 1),
        destination=(2, 1),
        action="ACTION4",
    )
    policy._update_evidence(_state(player=(2, 1)))
    assert policy.observed_transitions[((1, 1), "ACTION4")] == (2, 1)

    policy._last_player = (2, 1)
    policy.pending_interaction = PendingInteraction(
        objective="contact_matching_static_object",
        target=(2, 0),
        source=(2, 1),
        destination=(2, 0),
        action="ACTION1",
    )
    policy._update_evidence(_state(player=(2, 1)))
    assert policy.observed_transitions[((2, 1), "ACTION1")] == (2, 1)


def test_route_avoids_special_transition_when_alternative_exists():
    policy = MechanismPolicy()
    policy.special_transitions[((1, 1), "ACTION4")] = (4, 4)

    route = policy._route(_state(player=(1, 1)), (2, 1))

    assert route is not None
    assert route != ["ACTION4"]
    assert route[0] != "ACTION4"


def test_route_uses_special_transition_in_matching_status_context():
    policy = MechanismPolicy()
    edge = ((1, 1), "ACTION4")
    policy.special_transitions[edge] = (4, 4)
    policy.contextual_special_transitions[
        ((1, 1), "ACTION4", "10/01", 9)
    ] = (4, 4)
    state = replace(
        _state(player=(1, 1), status=((1, 0), (0, 1)), status_color=9),
        walkable_cells=frozenset({(1, 1), (4, 4)}),
    )

    assert policy._route(state, (4, 4)) == ["ACTION4"]


def test_reuses_learned_relocation_edge_for_navigation():
    policy = MechanismPolicy()
    edge = ((1, 1), "ACTION4")
    policy.special_transitions[edge] = (4, 4)
    policy.contextual_special_transitions[
        ((1, 1), "ACTION4", "10/01", 9)
    ] = (4, 4)
    state = replace(
        _state(player=(1, 1), status=((1, 0), (0, 1)), status_color=9),
        walkable_cells=frozenset({(1, 1), (4, 4)}),
    )

    # Transport is useful topology. It must not be quarantined merely because
    # the observation also reports player_relocated.
    assert ((1, 1), "ACTION4", "10/01", 9) not in policy.unsafe_navigation_edges
    assert policy._route(state, (4, 4)) == ["ACTION4"]


def test_route_prefers_longer_observed_path_over_unknown_shortcut():
    policy = MechanismPolicy()
    policy.special_transitions[((4, 4), "ACTION1")] = (1, 1)
    policy.observed_transitions.update(
        {
            ((1, 1), "ACTION2"): (1, 2),
            ((1, 2), "ACTION4"): (2, 2),
            ((2, 2), "ACTION1"): (2, 1),
        }
    )

    route = policy._route(_state(player=(1, 1)), (2, 1))

    assert route == ["ACTION2", "ACTION4", "ACTION1"]


def test_route_rejects_special_transition_in_different_status_context():
    policy = MechanismPolicy()
    edge = ((1, 1), "ACTION4")
    policy.special_transitions[edge] = (4, 4)
    policy.contextual_special_transitions[
        ((1, 1), "ACTION4", "10/01", 9)
    ] = (4, 4)
    state = replace(
        _state(player=(1, 1), status=((0, 1), (1, 0)), status_color=9),
        walkable_cells=frozenset({(1, 1), (4, 4)}),
    )

    assert policy._route(state, (4, 4)) is None


def test_resource_wrap_does_not_mark_recent_candidate_as_refill():
    policy = MechanismPolicy()
    candidate = _object((2, 1), 8)
    policy.choose(
        _state(player=(1, 1), resource=2, objects=[candidate]),
        [1, 2, 3, 4],
    )
    policy.pending_interaction = PendingInteraction(
        objective="discover_resource_source",
        target=(2, 1),
        source=(1, 1),
        destination=(2, 1),
        action="ACTION1",
    )

    policy._update_evidence(
        _state(
            player=(1, 1),
            resource=40,
            objects=[candidate],
            events=[MechanismEvent("resource_increased", action="ACTION1")],
        )
    )

    assert (2, 1) not in policy.refill_cells


def test_route_returns_none_when_special_transition_is_only_geometric_edge():
    policy = MechanismPolicy()
    policy.special_transitions[((1, 1), "ACTION4")] = (4, 4)
    state = replace(
        _state(player=(1, 1)),
        walkable_cells=frozenset({(1, 1), (2, 1)}),
    )

    assert policy._route(state, (2, 1)) is None


def test_learns_palette_and_pattern_changers_separately():
    policy = MechanismPolicy()
    candidate = [
        _object((3, 1), 3, color=8),
        _object((3, 1), 2, color=9),
    ]
    policy.choose(_state(objects=candidate), [1, 2, 3, 4])
    policy.choose(
        _state(
            player=(3, 1),
            resource=36,
            objects=candidate,
            events=[MechanismEvent("status_palette_changed", action="ACTION4")],
        ),
        [1, 2, 3, 4],
    )

    assert policy.palette_changer_cells == {(3, 1)}
    assert not policy.changer_cells


def test_characterizes_new_actuator_until_output_repeats():
    policy = MechanismPolicy()
    candidate = [
        _object((3, 1), 3, color=8),
        _object((3, 1), 2, color=9),
    ]
    policy.choose(_state(objects=candidate), [1, 2, 3, 4])

    departed = policy.choose(
        _state(
            player=(3, 1),
            resource=36,
            status_color=9,
            objects=candidate,
            events=[MechanismEvent("status_palette_changed", action="ACTION4")],
        ),
        [1, 2, 3, 4],
    )
    assert departed.name == "ACTION1"
    assert policy.last_decision.objective == "depart_status_palette"

    returned = policy.choose(
        _state(player=(3, 0), resource=32, status_color=9, objects=candidate),
        [1, 2, 3, 4],
    )
    assert returned.name == "ACTION2"
    assert policy.last_decision.objective == "characterize_status_palette"

    policy.choose(
        _state(
            player=(3, 1),
            resource=28,
            status_color=12,
            objects=candidate,
            events=[MechanismEvent("status_palette_changed", action="ACTION2")],
        ),
        [1, 2, 3, 4],
    )
    policy.choose(
        _state(player=(3, 0), resource=24, status_color=12, objects=candidate),
        [1, 2, 3, 4],
    )
    policy.choose(
        _state(
            player=(3, 1),
            resource=20,
            status_color=9,
            objects=candidate,
            events=[MechanismEvent("status_palette_changed", action="ACTION2")],
        ),
        [1, 2, 3, 4],
    )

    assert ((3, 1), "palette") in policy.completed_actuator_probes
    assert policy.active_actuator_probe is None
    assert policy.actuator_probe_states[((3, 1), "palette")] == [9, 12]


def test_does_not_commit_when_target_palette_is_known_to_differ():
    policy = MechanismPolicy()
    pattern = ((1, 0), (0, 1))
    target = _object(
        (1, 3),
        20,
        pattern=pattern,
        static=True,
        pattern_color=12,
    )

    policy.choose(
        _state(
            player=(1, 1),
            status=pattern,
            status_color=9,
            objects=[target],
        ),
        [1, 2, 3, 4],
    )

    assert policy.committed_target is None
    assert policy.last_decision is None or (
        policy.last_decision.objective != "contact_matching_static_object"
    )


def test_object_lifecycle_evidence_prevents_no_effect_classification():
    policy = MechanismPolicy()
    target = (2, 1)
    policy.level = 1
    policy.pending_interaction = PendingInteraction(
        objective="probe_candidate_contact",
        target=target,
        source=(1, 1),
        destination=target,
        action="ACTION4",
    )
    policy.contact_attempts[target].update({"ACTION1", "ACTION2", "ACTION3"})
    policy.contact_probe_retries[(target, "ACTION4")] = 3

    policy.choose(
        _state(
            player=target,
            events=[
                MechanismEvent(
                    "object_disappeared",
                    action="ACTION4",
                    before=("candidate",),
                    after=("avatar",),
                    cells=(target,),
                )
            ],
        ),
        [1, 2, 3, 4],
    )

    assert target in policy.lifecycle_evidence_cells
    assert target not in policy.no_effect_cells


def test_repeats_delayed_candidate_contact_before_changing_probe_direction():
    policy = MechanismPolicy()
    target = (2, 0)
    policy.level = 1
    candidate = [
        _object(target, 3, color=8),
        _object(target, 2, color=9),
    ]
    state = _state(player=target, objects=candidate)
    policy.last_decision = MechanismDecision(
        action="ACTION4",
        objective="discover_both_changer",
        target=target,
        route_length=0,
    )

    first = policy.choose(state, [1, 2, 3, 4])
    assert first.name == "ACTION1"
    assert policy.last_decision.objective == "probe_candidate_contact"

    for attempt in range(3):
        repeated = policy.choose(
            _state(player=target, resource=36 - attempt * 4, objects=candidate),
            [1, 2, 3, 4],
        )
        assert repeated.name == "ACTION1"
        assert policy.last_decision.objective == "probe_candidate_contact"

    changed = policy.choose(
        _state(player=target, resource=24, objects=candidate),
        [1, 2, 3, 4],
    )
    assert changed.name != "ACTION1"


def test_enterable_probe_uses_untried_return_direction():
    policy = MechanismPolicy()
    target = (2, 2)
    policy.level = 1
    policy.contact_attempts[target].add("ACTION2")

    decision = policy._contact_probe(
        _state(player=target),
        target,
        ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
    )

    assert decision is not None
    assert decision.action == "ACTION2"
    assert policy._probe_return == (target, "ACTION1")


def test_exhausted_lifecycle_candidate_is_not_selected_again():
    policy = MechanismPolicy()
    target = (2, 1)
    policy.level = 1
    policy.lifecycle_evidence_cells.add(target)
    policy.contact_attempts[target].update(policy.DIRECTION_NAMES)
    policy.exhausted_probe_cells.add(target)
    candidate = [
        _object(target, 3, color=8),
        _object(target, 2, color=9),
    ]

    action = policy.choose(
        _state(player=(1, 1), objects=candidate),
        [1, 2, 3, 4],
    )

    assert action is None
    assert target not in policy.no_effect_cells


def test_four_blocked_contacts_exhaust_candidate_without_orbiting_it():
    policy = MechanismPolicy()
    target = (2, 1)
    policy.level = 1
    policy.pending_interaction = PendingInteraction(
        objective="probe_candidate_contact",
        target=target,
        source=target,
        destination=(2, 2),
        action="ACTION2",
    )
    policy.contact_probe_retries[(target, "ACTION2")] = 3

    policy.choose(_state(player=target), [1, 2, 3, 4])

    assert target in policy.exhausted_probe_cells


def test_autonomous_dynamic_scene_rotates_after_bounded_contact_window():
    policy = MechanismPolicy()
    target = (2, 1)
    policy.level = 1
    dynamic_signature = (12, 10, "moving-bar")
    policy.autonomous_dynamic_signatures.add(dynamic_signature)
    policy.exhausted_probe_cells.add((1, 1))
    policy.pending_interaction = PendingInteraction(
        objective="probe_candidate_contact",
        target=target,
        source=target,
        destination=(2, 2),
        action="ACTION2",
    )
    policy.contact_probe_retries[(target, "ACTION2")] = 3

    policy.choose(_state(player=target), [1, 2, 3, 4])

    assert target in policy.exhausted_probe_cells
    assert policy.contact_probe_retries[(target, "ACTION2")] == 4


def test_dynamic_probe_keeps_one_departure_then_rotates_after_four_returns():
    policy = MechanismPolicy()
    target = (2, 1)
    policy.level = 1
    policy.autonomous_dynamic_signatures.add((12, 10, "moving-bar"))
    state = _state(player=target)

    first = policy._contact_probe(state, target, list(policy.DIRECTION_NAMES))
    assert first is not None
    assert first.action == "ACTION1"

    return_action = policy._inverse_action(first.action)
    policy.contact_probe_retries[(target, return_action)] = 3
    repeated = policy._contact_probe(state, target, list(policy.DIRECTION_NAMES))
    assert repeated is not None
    assert repeated.action == first.action

    policy.contact_probe_retries[(target, return_action)] = 4
    rotated = policy._contact_probe(state, target, list(policy.DIRECTION_NAMES))
    assert rotated is not None
    assert rotated.action != first.action


def test_dynamic_interception_uses_recent_unit_motion_after_reset_jump():
    policy = MechanismPolicy()
    signature = (12, 10, "moving-bar")
    policy.dynamic_track_history[signature].extend([(6, 3), (6, 4), (9, 0), (9, 1)])

    assert policy._recent_unit_delta(signature) == (0, 1)


def test_dynamic_interception_does_not_reuse_velocity_while_track_is_still():
    policy = MechanismPolicy()
    signature = (12, 10, "moving-bar")
    policy.dynamic_track_history[signature].extend([(9, 0), (8, 0), (8, 0)])

    assert policy._recent_unit_delta(signature) is None


def test_dynamic_interception_plans_toward_future_track_contact():
    policy = MechanismPolicy()
    policy.level = 1
    policy.exhausted_probe_cells.add((2, 1))
    primary = (12, 10, "moving-bar")
    secondary = (9, 15, "moving-block")
    policy.autonomous_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (3, 1), secondary: (3, 2)})
    policy.dynamic_track_history[primary].extend([(4, 1), (3, 1)])
    policy.dynamic_track_history[secondary].extend([(4, 2), (3, 2)])

    decision = policy._dynamic_interception(
        _state(player=(1, 1)),
        list(policy.DIRECTION_NAMES),
        "palette",
    )

    assert decision is not None
    assert decision.action == "ACTION4"
    assert decision.objective == "contact_dynamic_palette"
    assert decision.dynamic_signature in {primary, secondary}


def test_dynamic_interception_prefers_less_tested_signature_over_shorter_path():
    policy = MechanismPolicy()
    policy.level = 1
    policy.exhausted_probe_cells.add((2, 1))
    near = (9, 15, "near-block")
    farther = (12, 10, "farther-bar")
    policy.autonomous_dynamic_signatures.update({near, farther})
    policy.dynamic_track_positions.update({near: (2, 1), farther: (3, 1)})
    policy.dynamic_track_history[near].extend([(2, 1), (2, 1)])
    policy.dynamic_track_history[farther].extend([(3, 1), (3, 1)])
    policy.dynamic_signature_attempts[near] = 2

    decision = policy._dynamic_interception(
        _state(player=(1, 1)),
        list(policy.DIRECTION_NAMES),
        "both",
    )

    assert decision is not None
    assert policy._dynamic_plan_signature == farther
    assert decision.route_length > 1


def test_resumes_committed_dynamic_interception_until_contact():
    policy = MechanismPolicy()
    signature = (12, 10, "moving-bar")
    policy._dynamic_plan.extend(["ACTION1", "ACTION4"])
    policy._dynamic_plan_signature = signature
    policy._dynamic_plan_target = (2, 1)
    policy._dynamic_plan_dimension = "palette"

    first = policy._resume_dynamic_plan(
        _state(player=(1, 2)),
        list(policy.DIRECTION_NAMES),
    )
    second = policy._resume_dynamic_plan(
        _state(player=(1, 1)),
        list(policy.DIRECTION_NAMES),
    )

    assert first is not None
    assert first.objective == "approach_dynamic_palette"
    assert first.dynamic_signature is None
    assert second is not None
    assert second.objective == "contact_dynamic_palette"
    assert second.dynamic_signature == signature


def test_dynamic_interception_abandons_repeated_blocked_contact_context():
    policy = MechanismPolicy()
    policy.level = 1
    policy.exhausted_probe_cells.add((2, 1))
    primary = (12, 10, "moving-bar")
    secondary = (9, 15, "moving-block")
    policy.autonomous_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 1), secondary: (2, 2)})
    policy.dynamic_track_history[primary].extend([(3, 1), (2, 1)])
    policy.dynamic_track_history[secondary].extend([(3, 2), (2, 2)])
    policy.dynamic_contact_retries[(primary, (1, 1), "ACTION4")] = 4

    decision = policy._dynamic_interception(
        _state(player=(1, 1)),
        list(policy.DIRECTION_NAMES),
        "palette",
    )

    assert decision is not None
    assert not (
        decision.dynamic_signature == primary
        and decision.action == "ACTION4"
        and decision.route_length == 1
    )


def test_dynamic_interception_approaches_confirmed_stationary_track():
    policy = MechanismPolicy()
    policy.level = 1
    policy.exhausted_probe_cells.add((2, 1))
    primary = (12, 10, "moving-bar")
    secondary = (9, 15, "moving-block")
    policy.autonomous_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (3, 1), secondary: (3, 2)})
    policy.dynamic_track_history[primary].extend([(3, 1), (3, 1)])
    policy.dynamic_track_history[secondary].extend([(3, 2), (3, 2)])

    decision = policy._dynamic_interception(
        _state(player=(1, 1)),
        list(policy.DIRECTION_NAMES),
        "pattern",
    )

    assert decision is not None
    assert decision.action == "ACTION4"
    assert decision.objective == "approach_dynamic_pattern"
    assert decision.route_length == 2
    assert decision.dynamic_signature is None


def test_dynamic_interception_does_not_route_through_exhausted_front_track():
    policy = MechanismPolicy()
    policy.level = 1
    policy.exhausted_probe_cells.add((2, 1))
    front = (12, 10, "front-bar")
    behind = (9, 15, "behind-block")
    policy.autonomous_dynamic_signatures.update({front, behind})
    policy.dynamic_track_positions.update({front: (2, 1), behind: (2, 2)})
    policy.dynamic_track_history[front].extend([(2, 1), (2, 1)])
    policy.dynamic_track_history[behind].extend([(2, 2), (2, 2)])
    policy.dynamic_contact_retries[(front, (1, 1), "ACTION4")] = 4

    decision = policy._dynamic_interception(
        _state(player=(1, 1)),
        list(policy.DIRECTION_NAMES),
        "both",
    )

    assert decision is not None
    assert decision.action != "ACTION4"


def test_dynamic_staging_returns_to_refill_instead_of_chasing_distant_track():
    policy = MechanismPolicy()
    policy.level = 1
    policy.refill_cells.add((1, 2))
    primary = (12, 10, "moving-bar")
    secondary = (9, 15, "moving-block")
    policy.autonomous_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (4, 4), secondary: (4, 3)})

    decision = policy._dynamic_staging(
        _state(player=(1, 1)),
        list(policy.DIRECTION_NAMES),
    )

    assert decision is not None
    assert decision.action == "ACTION2"
    assert decision.objective == "return_to_dynamic_staging"


def test_dynamic_staging_waits_with_known_blocked_action():
    policy = MechanismPolicy()
    policy.level = 1
    staging = (1, 1)
    policy.refill_cells.add(staging)
    policy.observed_transitions[(staging, "ACTION1")] = staging
    primary = (12, 10, "moving-bar")
    secondary = (9, 15, "moving-block")
    policy.autonomous_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (4, 4), secondary: (4, 3)})

    decision = policy._dynamic_staging(
        _state(player=staging),
        list(policy.DIRECTION_NAMES),
    )

    assert decision is not None
    assert decision.action == "ACTION1"
    assert decision.objective == "wait_dynamic_phase"


def test_dynamic_staging_ignores_command_controlled_pair():
    policy = MechanismPolicy()
    policy.level = 1
    staging = (1, 1)
    policy.refill_cells.add(staging)
    policy.observed_transitions[(staging, "ACTION1")] = staging
    primary = (12, 10, "moving-bar")
    secondary = (9, 15, "moving-block")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (4, 4), secondary: (4, 3)})

    decision = policy._dynamic_staging(
        _state(player=staging),
        list(policy.DIRECTION_NAMES),
    )

    assert decision is None


def test_staging_rejects_and_clears_long_dynamic_proposal():
    policy = MechanismPolicy()
    policy.level = 1
    staging = (1, 1)
    policy.refill_cells.add(staging)
    policy.observed_transitions[(staging, "ACTION1")] = staging
    primary_object = _object((4, 4), 10, color=12)
    secondary_object = _object((4, 3), 15, color=9)
    primary = (12, 10, primary_object.shape_signature)
    secondary = (9, 15, secondary_object.shape_signature)
    policy.autonomous_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (4, 4), secondary: (4, 3)})
    policy.dynamic_track_history[primary].extend([(4, 4), (4, 4)])
    policy.dynamic_track_history[secondary].extend([(4, 3), (4, 3)])
    policy.exhausted_probe_cells.add(staging)

    action = policy.choose(
        _state(
            player=staging,
            objects=[primary_object, secondary_object],
        ),
        [1, 2, 3, 4],
    )

    assert action is not None
    assert policy.last_decision.objective == "wait_dynamic_phase"
    assert not policy._dynamic_plan
    assert policy._dynamic_plan_signature is None


def test_first_dynamic_candidate_keeps_short_disconfirmation_window():
    policy = MechanismPolicy()
    target = (2, 1)
    policy.level = 1
    policy.autonomous_dynamic_signatures.add((12, 10, "moving-bar"))
    policy.pending_interaction = PendingInteraction(
        objective="probe_candidate_contact",
        target=target,
        source=target,
        destination=(2, 2),
        action="ACTION2",
    )
    policy.contact_probe_retries[(target, "ACTION2")] = 3

    policy.choose(_state(player=target), [1, 2, 3, 4])

    assert target in policy.exhausted_probe_cells


def test_tracks_large_object_motion_independent_of_player():
    policy = MechanismPolicy()
    policy.level = 1
    moving_before = _object((4, 1), 10, color=12)
    moving_after = _object((1, 4), 10, color=12)

    policy.choose(
        _state(player=(1, 1), objects=[moving_before]),
        [1, 2, 3, 4],
    )
    policy.choose(
        _state(player=(1, 2), resource=36, objects=[moving_after]),
        [1, 2, 3, 4],
    )

    assert policy.autonomous_dynamic_signatures


def test_tracks_large_object_motion_conditioned_on_action():
    policy = MechanismPolicy()
    policy.level = 1
    signature = (12, 10, "area=10")
    before = _object((3, 1), 10, color=12)
    middle = _object((2, 1), 10, color=12)
    after = _object((1, 1), 10, color=12)

    policy.choose(_state(player=(1, 3), objects=[before]), [1, 2, 3, 4])
    policy.pending_interaction = PendingInteraction(
        objective="test",
        target=(1, 3),
        source=(1, 3),
        destination=(0, 3),
        action="ACTION3",
    )
    policy._update_evidence(_state(player=(1, 3), objects=[middle]))
    policy.pending_interaction = PendingInteraction(
        objective="test",
        target=(1, 3),
        source=(1, 3),
        destination=(0, 3),
        action="ACTION3",
    )
    policy._update_evidence(_state(player=(1, 3), objects=[after]))

    assert signature in policy.command_dynamic_signatures
    assert policy.dynamic_action_transitions[(signature, (2, 1), "ACTION3")] == (1, 1)


def test_joint_command_planner_moves_objects_into_blocked_player():
    policy = MechanismPolicy()
    policy.level = 1
    policy.exhausted_probe_cells.add((2, 2))
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 0), secondary: (4, 0)})
    policy.observed_transitions[((2, 2), "ACTION2")] = (2, 2)
    policy.dynamic_action_transitions.update(
        {
            (primary, (2, 0), "ACTION2"): (2, 1),
            (primary, (2, 1), "ACTION2"): (2, 2),
            (secondary, (4, 0), "ACTION2"): (4, 1),
            (secondary, (4, 1), "ACTION2"): (4, 2),
        }
    )

    decision = policy._command_joint_interception(
        _state(player=(2, 2)),
        list(policy.DIRECTION_NAMES),
        "palette",
    )

    assert decision is not None
    assert decision.action == "ACTION2"
    assert decision.objective == "approach_command_dynamic_palette"
    assert decision.route_length == 2
    assert list(policy._dynamic_plan) == ["ACTION2"]
    assert policy._dynamic_plan_signature == primary


def test_joint_command_interception_does_not_wait_for_probe_exhaustion():
    policy = MechanismPolicy()
    policy.level = 1
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 0), secondary: (4, 0)})
    policy.observed_transitions[((2, 2), "ACTION2")] = (2, 2)
    policy.dynamic_action_transitions.update(
        {
            (primary, (2, 0), "ACTION2"): (2, 1),
            (primary, (2, 1), "ACTION2"): (2, 2),
            (secondary, (4, 0), "ACTION2"): (4, 1),
            (secondary, (4, 1), "ACTION2"): (4, 2),
        }
    )

    decision = policy._dynamic_interception(
        _state(player=(2, 2)),
        list(policy.DIRECTION_NAMES),
        "palette",
    )

    assert decision is not None
    assert decision.objective == "approach_command_dynamic_palette"
    assert not policy.exhausted_probe_cells


def test_command_model_experiment_rotates_actions_at_same_joint_state():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 0), secondary: (4, 0)})
    state = _state(player=(2, 2))

    first = policy._command_model_experiment(
        state, list(policy.DIRECTION_NAMES), "both"
    )
    second = policy._command_model_experiment(
        state, list(policy.DIRECTION_NAMES), "both"
    )

    assert first is not None
    assert second is not None
    assert first.action != second.action


def test_command_model_experiment_exhausts_object_state_across_player_positions():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update(
        {primary: (2, 0), secondary: (2, 1)}
    )
    positions = tuple(
        policy.dynamic_track_positions[signature]
        for signature in sorted(policy.command_dynamic_signatures)
    )
    for action in policy.DIRECTION_NAMES:
        policy.command_joint_experiments[(positions, action)] = 1

    decision = policy._command_model_experiment(
        _state(player=(4, 4)),
        list(policy.DIRECTION_NAMES),
        "both",
    )

    assert decision is None


def test_command_contact_frontier_targets_short_known_contact_route():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 1), secondary: (2, 2)})
    policy.observed_transitions[((2, 4), "ACTION2")] = (2, 4)
    policy.dynamic_action_transitions.update(
        {
            (primary, (2, 1), "ACTION2"): (2, 2),
            (secondary, (2, 2), "ACTION2"): (2, 3),
            (primary, (2, 2), "ACTION2"): (2, 3),
            (secondary, (2, 3), "ACTION2"): (2, 4),
        }
    )

    decision = policy._command_contact_frontier(
        _state(player=(2, 4)),
        list(policy.DIRECTION_NAMES),
        "both",
    )

    assert decision is not None
    assert decision.action == "ACTION2"
    assert decision.objective == "approach_command_frontier_both"
    assert decision.route_length == 2


def test_command_contact_frontier_does_not_treat_following_as_contact():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 1), secondary: (2, 2)})
    policy.observed_transitions[((2, 3), "ACTION1")] = (2, 2)
    policy.dynamic_action_transitions.update(
        {
            (primary, (2, 1), "ACTION1"): (2, 0),
            (secondary, (2, 2), "ACTION1"): (2, 1),
        }
    )

    decision = policy._command_contact_frontier(
        _state(player=(2, 3)),
        ["ACTION1"],
        "both",
    )

    assert decision is None


def test_joint_command_planner_does_not_treat_following_as_contact():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 1), secondary: (2, 2)})
    policy.observed_transitions[((2, 3), "ACTION1")] = (2, 2)
    policy.dynamic_action_transitions.update(
        {
            (primary, (2, 1), "ACTION1"): (2, 0),
            (secondary, (2, 2), "ACTION1"): (2, 1),
        }
    )

    decision = policy._command_joint_interception(
        _state(player=(2, 3)),
        ["ACTION1"],
        "both",
    )

    assert decision is None


def test_command_model_experiment_continues_equally_unknown_command():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 1), secondary: (2, 2)})
    policy.pending_interaction = PendingInteraction(
        objective="learn_command_dynamic_both",
        target=(2, 4),
        source=(2, 4),
        destination=(2, 5),
        action="ACTION2",
    )
    policy.command_experiment_action = "ACTION2"
    policy.command_experiment_streak = 1

    decision = policy._command_model_experiment(
        _state(player=(2, 4)),
        list(policy.DIRECTION_NAMES),
        "both",
    )

    assert decision is not None
    assert decision.action == "ACTION2"


def test_command_model_experiment_starts_with_last_effective_motion():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 1), secondary: (2, 2)})
    policy.last_command_motion_action = "ACTION2"

    decision = policy._command_model_experiment(
        _state(player=(2, 4)),
        list(policy.DIRECTION_NAMES),
        "both",
    )

    assert decision is not None
    assert decision.action == "ACTION2"


def test_command_model_experiment_rotates_after_ineffective_repeat():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    player = (2, 4)
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 1), secondary: (2, 2)})
    policy.pending_interaction = PendingInteraction(
        objective="learn_command_dynamic_both",
        target=player,
        source=player,
        destination=(1, 4),
        action="ACTION3",
    )
    policy.command_experiment_action = "ACTION3"
    policy.command_experiment_streak = 4

    decision = policy._command_model_experiment(
        _state(player=player),
        list(policy.DIRECTION_NAMES),
        "both",
    )

    assert decision is not None
    assert decision.action != "ACTION3"


def test_command_model_experiment_avoids_collapsing_pair_at_boundary():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 0), secondary: (2, 1)})
    policy.dynamic_action_transitions.update(
        {
            (primary, (2, 0), "ACTION2"): (2, 1),
            (secondary, (2, 1), "ACTION2"): (2, 2),
        }
    )

    decision = policy._command_model_experiment(
        _state(player=(2, 4)),
        list(policy.DIRECTION_NAMES),
        "both",
    )

    assert decision is not None
    assert decision.action == "ACTION3"


def test_joint_command_planner_requires_exact_object_transitions():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 0), secondary: (2, 1)})

    decision = policy._command_joint_interception(
        _state(player=(2, 3)),
        list(policy.DIRECTION_NAMES),
        "both",
    )

    assert decision is None


def test_confirmed_command_pair_skips_legacy_single_track_interception():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 0), secondary: (2, 1)})
    policy.exhausted_probe_cells.add((4, 4))

    decision = policy._dynamic_interception(
        _state(player=(2, 4)),
        list(policy.DIRECTION_NAMES),
        "both",
    )

    assert decision is None


def test_command_pair_confirmation_survives_transient_member_occlusion():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions[primary] = (2, 0)

    assert policy._has_command_pair()

    decision = policy._dynamic_interception(
        _state(player=(2, 4)),
        list(policy.DIRECTION_NAMES),
        "both",
    )

    assert decision is None


def test_command_visibility_recovery_reverses_occluding_action_then_rotates():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    player = (4, 4)
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions[primary] = (4, 3)
    policy.pending_interaction = PendingInteraction(
        objective="approach_command_frontier_both",
        target=(4, 4),
        source=(3, 4),
        destination=player,
        action="ACTION4",
    )

    first = policy._command_visibility_recovery(
        _state(player=player),
        list(policy.DIRECTION_NAMES),
    )
    second = policy._command_visibility_recovery(
        _state(player=player),
        list(policy.DIRECTION_NAMES),
    )

    assert first is not None
    assert first.action == "ACTION3"
    assert first.objective == "recover_command_visibility"
    assert second is not None
    assert second.action != first.action


def test_command_visibility_recovery_resets_when_pair_reappears():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions[primary] = (2, 0)
    policy.command_visibility_attempts["ACTION1"] = 1

    policy.dynamic_track_positions[secondary] = (2, 1)
    decision = policy._command_visibility_recovery(
        _state(player=(2, 4)),
        list(policy.DIRECTION_NAMES),
    )

    assert decision is None
    assert not policy.command_visibility_attempts


def test_joint_command_planner_rejects_member_overlap_prediction():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy.command_dynamic_signatures.update({primary, secondary})
    policy.dynamic_track_positions.update({primary: (2, 0), secondary: (2, 1)})
    policy.dynamic_action_transitions.update(
        {
            (primary, (2, 0), "ACTION1"): (2, 0),
            (secondary, (2, 1), "ACTION1"): (2, 0),
        }
    )

    decision = policy._command_joint_interception(
        _state(player=(2, 3)),
        ["ACTION1"],
        "both",
    )

    assert decision is None


def test_joint_command_plan_replans_when_predicted_member_state_is_wrong():
    policy = MechanismPolicy()
    primary = (12, 10, "primary")
    secondary = (9, 15, "secondary")
    policy._dynamic_plan.extend(["ACTION2"])
    policy._dynamic_plan_signature = primary
    policy._dynamic_plan_target = (2, 2)
    policy._dynamic_plan_dimension = "palette"
    policy._dynamic_plan_joint_signatures = (primary, secondary)
    policy._dynamic_plan_expected_states.append(((2, 2), ((2, 1), (4, 1))))
    policy.dynamic_track_positions.update({primary: (2, 1), secondary: (4, 0)})

    decision = policy._resume_dynamic_plan(
        _state(player=(2, 2)),
        list(policy.DIRECTION_NAMES),
    )

    assert decision is None
    assert not policy._dynamic_plan
    assert not policy._dynamic_plan_joint_signatures


def test_plain_object_disappearance_does_not_count_as_lifecycle_interaction():
    policy = MechanismPolicy()
    target = (2, 1)
    policy.level = 1
    policy.pending_interaction = PendingInteraction(
        objective="probe_candidate_contact",
        target=target,
        source=(1, 1),
        destination=target,
        action="ACTION4",
    )
    policy.contact_attempts[target].update({"ACTION1", "ACTION2", "ACTION3"})
    policy.contact_probe_retries[(target, "ACTION4")] = 3

    policy.choose(
        _state(
            player=target,
            events=[
                MechanismEvent(
                    "object_disappeared",
                    action="ACTION4",
                    before=("avatar",),
                    after=None,
                    cells=(target,),
                )
            ],
        ),
        [1, 2, 3, 4],
    )

    assert target not in policy.lifecycle_evidence_cells
    assert target in policy.no_effect_cells


def test_prefers_missing_palette_dimension_before_pattern_changer():
    policy = MechanismPolicy()
    pattern = ((1, 0), (0, 1))
    target = _object(
        (4, 1),
        20,
        pattern=pattern,
        static=True,
        pattern_color=12,
    )
    policy.level = 1
    policy.palette_changer_cells.add((1, 2))
    policy.changer_cells.add((3, 1))

    action = policy.choose(
        _state(
            player=(1, 1),
            status=pattern,
            status_color=9,
            objects=[target],
        ),
        [1, 2, 3, 4],
    )

    assert action.name == "ACTION2"
    assert policy.last_decision.objective == "align_status_palette"


def test_defers_ambiguous_matching_target_until_one_proves_dynamic():
    policy = MechanismPolicy()
    pattern = ((1, 0), (0, 1))
    matching = _object(
        (1, 3),
        20,
        pattern=pattern,
        static=True,
        pattern_color=9,
    )
    stable = _object(
        (4, 4),
        20,
        pattern=((1, 1), (0, 1)),
        static=True,
        pattern_color=9,
    )

    policy.choose(
        _state(
            status=pattern,
            status_color=9,
            objects=[matching, stable],
        ),
        [1, 2, 3, 4],
    )

    assert policy.committed_target is None

    changed_pattern = ((1, 1), (1, 0))
    changed_mirror = _object(
        (1, 3),
        20,
        pattern=changed_pattern,
        static=True,
        pattern_color=12,
    )
    policy.choose(
        _state(
            status=changed_pattern,
            status_color=12,
            objects=[changed_mirror, stable],
            events=[
                MechanismEvent("status_pattern_changed", action="ACTION4"),
                MechanismEvent("status_palette_changed", action="ACTION4"),
            ],
        ),
        [1, 2, 3, 4],
    )

    assert policy.dynamic_target_cells == {(1, 3)}


def test_commits_only_to_stable_target_after_mirror_is_identified():
    policy = MechanismPolicy()
    target_pattern = ((1, 1), (0, 1))
    mirror = _object(
        (1, 3),
        20,
        pattern=target_pattern,
        static=True,
        pattern_color=9,
    )
    target = _object(
        (4, 4),
        20,
        pattern=target_pattern,
        static=True,
        pattern_color=9,
    )
    policy.level = 1
    policy.dynamic_target_cells.add((1, 3))

    policy.choose(
        _state(
            player=(3, 4),
            status=target_pattern,
            status_color=9,
            objects=[mirror, target],
        ),
        [1, 2, 3, 4],
    )

    assert policy.committed_target == (4, 4)
    assert policy.last_decision.objective == "contact_matching_static_object"


def test_alignment_ignores_dynamic_mirror_pattern():
    policy = MechanismPolicy()
    status = ((1, 0), (0, 1))
    mirror = _object(
        (1, 3),
        20,
        pattern=status,
        static=True,
        pattern_color=9,
    )
    target = _object(
        (4, 4),
        20,
        pattern=((1, 1), (0, 1)),
        static=True,
        pattern_color=9,
    )
    policy.level = 1
    policy.dynamic_target_cells.add((1, 3))
    policy.changer_cells.add((2, 1))
    policy.palette_changer_cells.add((1, 2))

    action = policy.choose(
        _state(
            player=(1, 1),
            status=status,
            status_color=9,
            objects=[mirror, target],
        ),
        [1, 2, 3, 4],
    )

    assert action.name == "ACTION4"
    assert policy.last_decision.objective == "align_status_pattern"


def test_explores_when_required_changer_dimension_is_unknown():
    policy = MechanismPolicy()
    status = ((1, 0), (0, 1))
    target = _object(
        (4, 4),
        20,
        pattern=((1, 1), (0, 1)),
        static=True,
        pattern_color=9,
    )
    candidate = [
        _object((2, 1), 3, color=8),
        _object((2, 1), 2, color=9),
    ]
    policy.level = 1
    policy.palette_changer_cells.add((1, 2))

    action = policy.choose(
        _state(
            player=(1, 1),
            status=status,
            status_color=9,
            objects=[target, *candidate],
        ),
        [1, 2, 3, 4],
    )

    assert action.name == "ACTION4"
    assert policy.last_decision.objective == "discover_pattern_changer"


def test_both_dimensions_unknown_prefers_high_palette_diversity_candidate():
    policy = MechanismPolicy()
    target = _object(
        (4, 4),
        20,
        pattern=((1, 1), (0, 1)),
        static=True,
        pattern_color=9,
    )
    pattern_candidate = [
        _object((3, 1), 3, color=0),
        _object((3, 1), 1, color=1),
        _object((3, 1), 1, color=1),
    ]
    palette_candidate = [
        _object((1, 3), 2, color=8),
        _object((1, 3), 2, color=9),
        _object((1, 3), 2, color=12),
        _object((1, 3), 2, color=14),
    ]

    action = policy.choose(
        _state(
            player=(1, 1),
            status=((1, 0), (0, 1)),
            status_color=12,
            objects=[target, *pattern_candidate, *palette_candidate],
        ),
        [1, 2, 3, 4],
    )

    assert action.name == "ACTION2"
    assert policy.last_decision.target == (1, 3)
    assert policy.last_decision.objective == "discover_both_changer"


def test_unknown_changer_reserves_resource_for_controlled_experiment():
    policy = MechanismPolicy()
    target = _object(
        (4, 4),
        20,
        pattern=((1, 1), (0, 1)),
        static=True,
        pattern_color=9,
    )
    changer_candidate = [
        _object((3, 4), 3, color=8),
        _object((3, 4), 2, color=9),
    ]
    probable_refill = _object((1, 2), 8)

    action = policy.choose(
        _state(
            player=(3, 2),
            resource=24,
            status=((1, 0), (0, 1)),
            status_color=12,
            objects=[target, probable_refill, *changer_candidate],
        ),
        [1, 2, 3, 4],
    )

    assert action.name == "ACTION3"
    assert policy.last_decision.objective == "discover_resource_source"
    assert policy.last_decision.target == (1, 2)


def test_both_mismatched_dimensions_use_nearest_learned_changer():
    policy = MechanismPolicy()
    policy.level = 1
    policy.changer_cells.add((4, 1))
    policy.palette_changer_cells.add((1, 2))
    target = _object(
        (4, 4),
        20,
        pattern=((1, 1), (0, 1)),
        static=True,
        pattern_color=9,
    )

    action = policy.choose(
        _state(
            player=(1, 1),
            status=((1, 0), (0, 1)),
            status_color=12,
            objects=[target],
        ),
        [1, 2, 3, 4],
    )

    assert action.name == "ACTION2"
    assert policy.last_decision.objective == "align_status_palette"
    assert policy.last_decision.target == (1, 2)


def test_unknown_pattern_role_prefers_multipart_object_over_boundary_singleton():
    policy = MechanismPolicy()
    target = _object(
        (4, 4),
        20,
        pattern=((1, 1), (0, 1)),
        static=True,
        pattern_color=9,
    )
    boundary_singleton = [_object((0, 0), 5, color=1)]
    pattern_candidate = [
        _object((3, 1), 3, color=0),
        _object((3, 1), 1, color=1),
        _object((3, 1), 1, color=1),
    ]
    policy.level = 1
    policy.palette_changer_cells.add((1, 3))

    action = policy.choose(
        _state(
            player=(1, 1),
            status=((1, 0), (0, 1)),
            status_color=9,
            objects=[target, *boundary_singleton, *pattern_candidate],
        ),
        [1, 2, 3, 4],
    )

    assert action.name == "ACTION4"
    assert policy.last_decision.target == (3, 1)
    assert policy.last_decision.objective == "discover_pattern_changer"


def test_both_role_prefers_structural_controller_over_small_color_marker():
    policy = MechanismPolicy()
    target = _object(
        (4, 4),
        20,
        pattern=((1, 1), (0, 1)),
        static=True,
        pattern_color=9,
    )
    small_color_marker = [
        _object((2, 2), 1, color=0),
        _object((2, 2), 2, color=9),
        _object((2, 2), 2, color=14),
    ]
    structural_controller = [
        _object((3, 2), 5, color=1),
        _object((3, 2), 2, color=8),
    ]

    action = policy.choose(
        _state(
            player=(1, 2),
            status=((1, 0), (0, 1)),
            status_color=12,
            objects=[target, *small_color_marker, *structural_controller],
        ),
        [1, 2, 3, 4],
    )

    assert action.name == "ACTION4"
    assert policy.last_decision.target == (3, 2)
    assert policy.last_decision.objective == "discover_both_changer"


def test_both_role_does_not_treat_large_singleton_as_controller():
    policy = MechanismPolicy()
    target = _object(
        (4, 4),
        20,
        pattern=((1, 1), (0, 1)),
        static=True,
        pattern_color=9,
    )
    probable_refill = [_object((2, 2), 8, color=11)]
    compound_controller = [
        _object((3, 2), 3, color=0),
        _object((3, 2), 1, color=1),
        _object((3, 2), 1, color=1),
    ]

    policy.choose(
        _state(
            player=(1, 2),
            status=((1, 0), (0, 1)),
            status_color=12,
            objects=[target, *probable_refill, *compound_controller],
        ),
        [1, 2, 3, 4],
    )

    assert policy.last_decision.target == (3, 2)
    assert policy.last_decision.objective == "discover_both_changer"


def test_status_change_prevents_same_contact_from_becoming_refill():
    policy = MechanismPolicy()
    target = (2, 1)
    policy.level = 1
    policy.pending_interaction = PendingInteraction(
        objective="discover_both_changer",
        target=target,
        source=(1, 1),
        destination=target,
        action="ACTION4",
    )

    policy.choose(
        _state(
            player=target,
            resource=40,
            events=[
                MechanismEvent("resource_increased", action="ACTION4"),
                MechanismEvent("status_palette_changed", action="ACTION4"),
            ],
        ),
        [1, 2, 3, 4],
    )

    assert target in policy.palette_changer_cells
    assert target not in policy.refill_cells


def test_restores_resource_before_long_matching_target_route():
    policy = MechanismPolicy()
    policy.level = 1
    policy.refill_cells.add((1, 2))
    pattern = ((1, 0), (0, 1))
    target = _object((4, 4), 20, pattern=pattern, static=True)
    visible_refill = _object((1, 2), 8)

    action = policy.choose(
        _state(
            player=(1, 1),
            resource=8,
            status=pattern,
            objects=[target, visible_refill],
        ),
        [1, 2, 3, 4],
    )

    assert action.name == "ACTION2"
    assert policy.last_decision.objective == "restore_resource"


def test_restores_resource_before_actuator_and_completion_tail():
    policy = MechanismPolicy()
    policy.level = 1
    policy.changer_cells.add((3, 1))
    policy.refill_cells.add((1, 2))
    policy.special_transitions[((4, 4), "ACTION1")] = (1, 1)
    current = ((1, 0), (0, 1))
    required = ((1, 1), (0, 1))
    target = _object((4, 4), 20, pattern=required, static=True, pattern_color=9)
    visible_refill = _object((1, 2), 8)

    action = policy.choose(
        _state(
            player=(1, 1),
            resource=32,
            status=current,
            status_color=9,
            objects=[target, visible_refill],
        ),
        [1, 2, 3, 4],
    )

    assert action.name == "ACTION2"
    assert policy.last_decision.objective == "restore_resource"


def test_does_not_route_back_to_consumed_refill():
    policy = MechanismPolicy()
    policy.level = 1
    policy.refill_cells.add((2, 2))
    pattern = ((1, 0), (0, 1))
    target = _object((4, 4), 20, pattern=pattern, static=True)

    action = policy.choose(
        _state(player=(2, 2), resource=20, status=pattern, objects=[target]),
        [1, 2, 3, 4],
    )

    assert action is not None
    assert policy.last_decision.objective == "contact_matching_static_object"


def test_discovers_probable_refill_before_cycling_at_low_resource():
    policy = MechanismPolicy()
    policy.level = 1
    policy.changer_cells.add((3, 1))
    probable_refill = _object((1, 3), 8)

    action = policy.choose(
        _state(player=(3, 1), resource=16, objects=[probable_refill]),
        [1, 2, 3, 4],
    )

    assert action.name in {"ACTION2", "ACTION3"}
    assert policy.last_decision.objective == "discover_resource_source"


def test_contextual_topology_reserves_more_resource_for_unknown_mechanism():
    policy = MechanismPolicy()
    policy.special_transitions[((4, 4), "ACTION1")] = (1, 1)
    probable_refill = _object((1, 2), 8)
    state = _state(player=(1, 1), resource=24, objects=[probable_refill])

    assert policy._needs_discovery_refill(state, route_length=4)


def test_learned_joint_transition_installs_persistent_task_plan():
    policy = MechanismPolicy()
    policy.level = 1
    initial = ((1, 0), (0, 1))
    required = ((1, 1), (0, 1))
    policy.joint_mechanism_graph.record(
        JointMechanismState("10/01", 9),
        (3, 1),
        JointMechanismState("11/01", 9),
    )
    target = _object(
        (4, 1),
        20,
        pattern=required,
        static=True,
        pattern_color=9,
    )

    first = policy.choose(
        _state(
            player=(1, 1),
            status=initial,
            status_color=9,
            objects=[target],
        ),
        [1, 2, 3, 4],
    )

    assert first.name == "ACTION4"
    assert policy.last_decision.objective == "task_plan_apply_actuator"
    plan = policy.task_plan_controller.active_plan
    assert plan is not None
    assert plan.current_stage is not None
    assert plan.current_stage.kind is PlanStageKind.APPLY_ACTUATOR
    plan_id = plan.plan_id

    second = policy.choose(
        _state(
            player=(2, 1),
            resource=36,
            status=initial,
            status_color=9,
            objects=[target],
        ),
        [1, 2, 3, 4],
    )

    assert second.name == "ACTION4"
    assert policy.last_decision.objective == "task_plan_apply_actuator"
    assert policy.task_plan_controller.active_plan is plan
    assert policy.task_plan_controller.active_plan.plan_id == plan_id


def test_task_plan_uses_observed_approach_and_contact_action():
    policy = MechanismPolicy()
    policy.level = 1
    initial = ((1, 0), (0, 1))
    required = ((1, 1), (0, 1))
    policy.joint_mechanism_graph.record(
        JointMechanismState("10/01", 9),
        (3, 1),
        JointMechanismState("11/01", 9),
        approach=(2, 1),
        contact_action="ACTION4",
        observed_result=(3, 1),
    )
    target = _object(
        (4, 1),
        20,
        pattern=required,
        static=True,
        pattern_color=9,
    )

    approach = policy.choose(
        _state(
            player=(1, 1),
            status=initial,
            status_color=9,
            objects=[target],
        ),
        [1, 2, 3, 4],
    )
    assert approach.name == "ACTION4"
    assert policy.last_decision.objective == "task_plan_approach_actuator"
    assert policy.last_decision.target == (2, 1)

    contact = policy.choose(
        _state(
            player=(2, 1),
            resource=36,
            status=initial,
            status_color=9,
            objects=[target],
        ),
        [1, 2, 3, 4],
    )
    assert contact.name == "ACTION4"
    assert policy.last_decision.objective == "task_plan_apply_actuator"
    assert policy.last_decision.target == (3, 1)


def test_task_plan_quarantines_repeated_contextual_route_edge():
    policy = MechanismPolicy()
    policy.level = 1
    status = JointMechanismState("10/01", 9)
    policy.joint_mechanism_graph.record(
        status,
        (4, 4),
        JointMechanismState("11/01", 9),
    )
    policy.task_plan_controller.install(
        level=1,
        goal=JointMechanismState("11/01", 9),
        stages=(),
        topology_revision=0,
    )
    # Keep an active plan/stage so two identical task-plan transitions are
    # interpreted as a failed Option rather than generic exploration.
    from agents.task_planning import PlanStage

    policy.task_plan_controller.install(
        level=1,
        goal=JointMechanismState("11/01", 9),
        stages=(
            PlanStage(
                kind=PlanStageKind.APPLY_ACTUATOR,
                target=(4, 4),
                expected_before=status,
                expected_after=JointMechanismState("11/01", 9),
            ),
        ),
        topology_revision=0,
    )
    pending = PendingInteraction(
        objective="task_plan_apply_actuator",
        target=(4, 4),
        source=(2, 2),
        destination=(2, 3),
        action="ACTION2",
    )
    policy._last_player = (2, 2)
    policy.last_status = "10/01"
    policy.last_status_color = 9
    policy.pending_interaction = pending
    policy._update_evidence(
        _state(player=(2, 3), status=((1, 0), (0, 1)), status_color=9)
    )

    policy._last_player = (2, 2)
    policy.pending_interaction = pending
    policy._update_evidence(
        _state(player=(2, 3), status=((1, 0), (0, 1)), status_color=9)
    )

    assert ((2, 2), "ACTION2", "10/01", 9) in policy.unsafe_navigation_edges


def test_task_plan_inserts_refill_before_full_mission_when_needed():
    policy = MechanismPolicy()
    policy.level = 1
    initial = ((1, 0), (0, 1))
    required = ((1, 1), (0, 1))
    policy.refill_cells.add((1, 2))
    policy.joint_mechanism_graph.record(
        JointMechanismState("10/01", 9),
        (2, 1),
        JointMechanismState("11/01", 9),
    )
    target = _object(
        (3, 1),
        20,
        pattern=required,
        static=True,
        pattern_color=9,
    )
    refill = _object((1, 2), 8)

    action = policy.choose(
        _state(
            player=(1, 1),
            resource=12,
            status=initial,
            status_color=9,
            objects=[target, refill],
        ),
        [1, 2, 3, 4],
    )

    assert action.name == "ACTION2"
    assert policy.last_decision.objective == "task_plan_refill"
    plan = policy.task_plan_controller.active_plan
    assert plan is not None
    assert [stage.kind for stage in plan.stages] == [
        PlanStageKind.REFILL,
        PlanStageKind.APPLY_ACTUATOR,
        PlanStageKind.CONTACT_TARGET,
    ]


def test_returns_none_for_introductory_or_low_confidence_layout():
    policy = MechanismPolicy()
    state = _state()
    intro = MechanismState(**{**state.__dict__, "level": 0})
    low_layout = GridLayout((0, 0), (2, 2), 5, 5, 3, 4, 0.55, True)
    projected = MechanismState(**{**state.__dict__, "layout": low_layout})

    assert policy.choose(intro, [1, 2, 3, 4]) is None
    assert policy.choose(projected, [1, 2, 3, 4]) is None


def test_keeps_contacting_committed_target_when_avatar_occludes_it():
    policy = MechanismPolicy()
    pattern = ((1, 0), (0, 1))
    target = _object((1, 3), 20, pattern=pattern, static=True)

    first = policy.choose(
        _state(player=(1, 1), status=pattern, objects=[target]),
        [1, 2, 3, 4],
    )
    assert first.name == "ACTION2"
    assert policy.committed_target == (1, 3)

    # The target may disappear from object extraction while the avatar enters
    # its sprite.  Matching status has not changed, so contact must continue.
    second = policy.choose(
        _state(player=(1, 2), resource=36, status=pattern, objects=[]),
        [1, 2, 3, 4],
    )
    assert second.name == "ACTION2"
    assert policy.last_decision.objective == "contact_matching_static_object"
    assert policy.last_decision.target == (1, 3)


def test_contact_animation_does_not_learn_static_target_as_mechanism():
    policy = MechanismPolicy()
    pattern = ((1, 0), (0, 1))
    target = _object((1, 2), 20, pattern=pattern, static=True)

    policy.choose(
        _state(player=(1, 1), status=pattern, objects=[target]),
        [1, 2, 3, 4],
    )
    events = [
        MechanismEvent("resource_increased", action="ACTION2"),
        MechanismEvent("status_pattern_changed", action="ACTION2"),
    ]
    policy.choose(
        _state(
            player=(1, 2),
            resource=40,
            status=((1, 1), (0, 1)),
            objects=[],
            events=events,
        ),
        [1, 2, 3, 4],
    )

    assert (1, 2) not in policy.refill_cells
    assert (1, 2) not in policy.changer_cells


def test_probes_direction_when_candidate_cell_has_been_reached():
    policy = MechanismPolicy()
    candidate = [
        _object((2, 1), 3, color=8),
        _object((2, 1), 2, color=9),
    ]

    policy.choose(_state(player=(1, 1), objects=candidate), [1, 2, 3, 4])
    action = policy.choose(
        _state(player=(2, 1), resource=36, objects=candidate),
        [1, 2, 3, 4],
    )

    assert action is not None
    assert policy.last_decision.objective in {
        "probe_candidate_contact",
        "probe_candidate_depart",
    }
    assert policy.last_decision.target == (2, 1)


def test_probes_committed_completion_when_already_on_target_cell():
    policy = MechanismPolicy()
    pattern = ((1, 0), (0, 1))
    policy.level = 1
    policy.committed_target = (1, 2)
    policy.committed_pattern = "10/01"
    policy.static_target_cells.add((1, 2))

    action = policy.choose(
        _state(player=(1, 2), status=pattern, objects=[]),
        [1, 2, 3, 4],
    )

    assert action is not None
    assert policy.last_decision.objective in {
        "probe_candidate_contact",
        "probe_candidate_depart",
    }
    assert policy.last_decision.target == (1, 2)


def test_retries_blocked_completion_contact_before_abandoning_commitment():
    policy = MechanismPolicy()
    pattern = ((1, 0), (0, 1))
    target = _object((1, 2), 20, pattern=pattern, static=True)
    policy.level = 1
    policy.committed_target = (1, 2)
    policy.committed_pattern = "10/01"
    policy.pending_interaction = PendingInteraction(
        objective="contact_matching_static_object",
        target=(1, 2),
        source=(1, 1),
        destination=(1, 2),
        action="ACTION2",
    )

    action = policy.choose(
        _state(player=(1, 1), resource=4, status=pattern, objects=[target]),
        [1, 2, 3, 4],
    )

    assert action.name == "ACTION2"
    assert policy.last_decision.objective == "contact_matching_static_object"
