import json
from collections import Counter
from dataclasses import replace

from agents.contact_effects import ContactEffectMemory, EffectEvidence
from agents.dynamic_contact import DynamicContactModel, TrackedPhase, rotate
from agents.entity_options import EntityOptionController, EntityOptionModel
from agents.mechanism_policy import MechanismPolicy
from agents.mechanism_state_tracker import GridLayout, MechanismObject, MechanismState

PATTERN = ((0, 1, 0), (1, 1, 0), (0, 1, 1))
SIG = (2, 3, "sprite")
OTHER = (8, 3, "other")
PHASES = (
    ((1, 2), (2, 2)),
    ((2, 2), (3, 2)),
    ((3, 2), (2, 2)),
    ((2, 2), (1, 2)),
)
FAR = tuple(((a[0] + 5, a[1] + 5), (b[0] + 5, b[1] + 5)) for a, b in PHASES)
NAMES = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")


def sprite(cell, sig=SIG):
    return MechanismObject(str(sig), sig[0], cell, (0, 0, 1, 1), sig[1], sig[2])


def state(player=(3, 3), *, index=0, pattern=PATTERN, color=14, resource=60):
    entities = (sprite(PHASES[index][1]), sprite(FAR[index][1], OTHER))
    return MechanismState(
        level=5,
        game_state="NOT_FINISHED",
        layout=GridLayout((0, 0), (5, 5), 10, 10, 3, 4, 0.95),
        player_position=player,
        player_confidence=0.99,
        walkable_cells=frozenset(),
        blocked_cells=frozenset(),
        navigation_projection_cells=frozenset(
            (x, y) for x in range(10) for y in range(10)
        ),
        objects=entities,
        motion_entities=entities,
        resource_level=resource,
        resource_capacity=84,
        status_pattern=pattern,
        status_color=color,
        matching_target_ids=(),
    )


def dynamic(index=0):
    model = DynamicContactModel()
    for sig, phases in ((SIG, PHASES), (OTHER, FAR)):
        model.moving.add(sig)
        model.tracks[sig] = TrackedPhase(phases[index], phases[index][1])
        for phase, nxt in zip(phases, phases[1:] + phases[:1]):
            for moved in (True, False):
                model.motion[(sig, phase, moved)][nxt] = 2
    return model


def effects():
    memory = ContactEffectMemory()
    for tick in (1, 2):
        memory.evidence[tick] = EffectEvidence(
            tick,
            tick,
            SIG,
            "meet_after_moving",
            True,
            PATTERN,
            rotate(PATTERN),
            14,
            14,
            (),
        )
    return memory


def fixture(*, known=True, targets=None):
    controller = EntityOptionController()
    controller.costs[2] = 2
    observation, clock = state(), dynamic()
    memory = effects() if known else ContactEffectMemory()
    model = EntityOptionModel(controller, observation, clock, memory, targets or {})
    return controller, observation, clock, memory, model, model.bind(observation, clock)


def test_every_entity_advances_on_moves_and_blocked_actions():
    _, _, _, _, model, start = fixture()
    step = model.step(start, "ACTION1")
    assert step.contact == SIG
    assert step.after.phases == (PHASES[1], FAR[1])
    assert step.after.pattern == rotate(PATTERN)
    assert step.after.resource == 58
    model.navigation[(3, 3), "ACTION4"].append(
        (start.pattern, start.color, start.phases, start.player)
    )
    blocked = model.step(start, "ACTION4")
    assert blocked.after.player == start.player
    assert blocked.after.phases == (PHASES[1], FAR[1])


def test_unknown_contact_is_only_a_terminal_experiment_not_solution_state():
    _, _, _, _, model, start = fixture(known=False)
    assert model.step(start, "ACTION1") is None
    trial = model.step(start, "ACTION1", experiment=True)
    assert trial.unknown_fields == ("pattern", "color")
    path, key, _ = model.search_experiment(start, ("ACTION1",), Counter())
    assert len(path) == 1 and key == (SIG, "meet_after_moving", True, 14)
    assert model.search_experiment(start, ("ACTION1",), Counter({key: 3}))[0] is None


def test_simultaneous_contacts_duplicate_identity_and_unresolved_clocks_abstain():
    controller, observation, clock, memory, _, _ = fixture()
    clock.motion[(OTHER, FAR[0], True)] = Counter({PHASES[1]: 2})
    model = EntityOptionModel(controller, observation, clock, memory, {})
    assert (
        model.step(model.bind(observation, clock), "ACTION1", experiment=True) is None
    )
    duplicate = replace(
        observation, motion_entities=observation.motion_entities + (sprite((1, 1)),)
    )
    assert model.bind(duplicate, clock) is None
    clock.tracks[OTHER] = TrackedPhase(None, (7, 7))
    assert model.bind(observation, clock) is None


def test_pattern_rule_does_not_invent_color_rule_for_unseen_input():
    _, _, _, _, model, start = fixture()
    start = replace(start, color=8)
    assert model.step(start, "ACTION1") is None
    assert model.step(start, "ACTION1", experiment=True).unknown_fields == ("color",)


def test_experiment_fills_missing_color_before_destroying_modelled_pattern():
    _, _, _, _, model, start = fixture()
    path, _, _ = model.search_experiment(replace(start, color=8), NAMES, Counter())
    assert path and path[-1].contact == SIG
    assert path[-1].unknown_fields == ("color",)


def test_experiment_can_cross_known_contact_but_never_unknown_successor():
    controller, observation, clock, memory, _, _ = fixture()
    phases = (((5, 2), (4, 2)), ((4, 2), (4, 1)), ((4, 1), (4, 2)))
    clock.tracks[OTHER] = TrackedPhase(phases[0], phases[0][1])
    clock.motion[(OTHER, phases[0], True)] = Counter({phases[1]: 2})
    clock.motion[(OTHER, phases[1], True)] = Counter({phases[2]: 2})
    observation = replace(
        observation,
        motion_entities=(sprite(PHASES[0][1]), sprite(phases[0][1], OTHER)),
        navigation_projection_cells=frozenset({(3, 3), (3, 2), (4, 2)}),
    )
    model = EntityOptionModel(controller, observation, clock, memory, {})
    start = model.bind(observation, clock)
    path, key, _ = model.search_experiment(start, ("ACTION1", "ACTION4"), Counter())
    assert [step.action for step in path] == ["ACTION1", "ACTION4"]
    assert not path[0].unknown_fields and path[1].unknown_fields
    controller.observe(observation, clock, None)
    controller._install(model, path, experiment_key=key)
    after_clock = dynamic(1)
    after_clock.tracks[OTHER] = TrackedPhase(phases[1], phases[1][1])
    after = replace(
        state((3, 2), index=1, pattern=rotate(PATTERN), resource=58),
        motion_entities=(sprite(PHASES[1][1]), sprite(phases[1][1], OTHER)),
    )
    controller.observe(after, after_clock, "ACTION1")
    assert controller.metrics["validated_steps"] == 1
    assert not controller.attempts  # Known preparatory contact is not this experiment.


def test_conflicting_color_abstains_and_is_not_repeated_as_sparse_experiment():
    _, _, _, _, model, start = fixture()
    model.effects.evidence[3] = replace(model.effects.evidence[2], color_after=8)
    assert model.step(start, "ACTION1") is None
    assert model.search_experiment(start, ("ACTION1",), Counter())[0] is None


def test_complete_goal_requires_known_effect_and_exact_target_pattern_color():
    _, _, _, _, model, start = fixture(targets={(4, 2): (rotate(PATTERN), 14)})
    path, _ = model.search_goal(start, NAMES, horizon=3)
    assert path and path[-1].terminal
    assert [step.action for step in path] == ["ACTION1", "ACTION4"]
    model.targets = {(4, 2): (rotate(PATTERN), 8)}
    assert model.search_goal(start, NAMES, horizon=3)[0] is None


def test_observed_navigation_overrides_geometry_but_conflicts_and_teleports_abstain():
    _, _, _, _, model, start = fixture()
    matches = model.navigation[start.player, "ACTION4"]
    matches.append((start.pattern, start.color, start.phases, start.player))
    assert model.step(start, "ACTION4").after.player == start.player
    matches.append((start.pattern, start.color, start.phases, (4, 3)))
    assert model.step(start, "ACTION4") is None
    matches[:] = [(start.pattern, start.color, start.phases, (6, 3))]
    assert model.step(start, "ACTION4") is None


def test_refill_requires_visible_learned_inventory_and_is_single_use():
    controller, observation, clock, memory, _, _ = fixture()
    refill_sig = (10, 1, "refill")
    observation = replace(
        observation,
        motion_entities=observation.motion_entities + (sprite((4, 3), refill_sig),),
    )
    controller.refill_values[(4, 3)] = 84
    controller.refill_signatures[(4, 3)].add(refill_sig)
    model = EntityOptionModel(controller, observation, clock, memory, {})
    start = model.bind(observation, clock)
    step = model.step(start, "ACTION4")
    assert step.after.resource == 84 and not step.after.refills
    absent = replace(observation, motion_entities=observation.motion_entities[:2])
    assert not model.bind(absent, clock).refills
    assert model.step(replace(start, resource=2), "ACTION4") is None


def test_feedback_checks_known_fields_separately_from_unknown_outcomes():
    controller, observation, clock, memory, model, start = fixture(known=False)
    controller.observe(observation, clock, None)
    controller._install(model, (model.step(start, "ACTION1", experiment=True),))
    after = state((3, 2), index=1, pattern=rotate(PATTERN), color=8, resource=58)
    controller.observe(after, dynamic(1), "ACTION1")
    assert controller.metrics["validated_steps"] == 1
    assert controller.metrics["known_fields_checked"] == 5  # Includes fixtures.
    assert controller.metrics["unknown_fields_observed"] == 2
    assert controller.metrics["prediction_errors"] == 0
    assert controller.reacquire == SIG
    assert not controller.remaining


def test_action_or_position_deviation_invalidates_and_bans_failed_edge():
    controller, observation, clock, _, model, start = fixture()
    controller.observe(observation, clock, None)
    controller._install(model, (model.step(start, "ACTION4"),))
    controller.observe(state(index=1, resource=58), dynamic(1), "ACTION3")
    assert controller.metrics["prediction_errors"] == 1
    assert set(controller.failures[-1]["fields"]) == {"executed_action", "player"}
    assert ((3, 3), "ACTION4") in controller.bad_edges
    assert controller.expected is None and controller.model is None
    json.dumps(controller.diagnostics())


def test_install_clears_experiment_key_and_reset_clears_belief():
    controller, observation, clock, _, model, start = fixture()
    controller.observe(observation, clock, None)
    step = model.step(start, "ACTION4")
    controller._install(
        model, (step,), experiment_key=(SIG, "meet_after_moving", True, 14)
    )
    controller._install(model, (step,))
    assert controller.pending_key is None
    controller.reset_episode()
    assert controller.previous is None and controller.expected is None
    assert controller.model is None and not controller.remaining


def test_controller_requires_repeated_unique_resource_cost_and_bounded_budget():
    controller, observation, clock, memory, model, start = fixture()
    controller.costs = Counter({2: 1})
    assert controller.choose(observation, clock, memory, {}, NAMES) is None
    controller.costs = Counter({2: 3, 4: 1})
    assert controller.choose(observation, clock, memory, {}, NAMES) is None
    controller.costs = Counter({2: 3})
    controller.metrics["steps_selected"] = controller.MAX_STEPS
    controller.model = model
    controller.remaining.append(model.step(start, "ACTION4"))
    assert controller.choose(observation, clock, memory, {}, NAMES) is None
    assert not controller.remaining and controller.metrics["invalidated_action_budget"]


def test_option_feature_defaults_off_and_reset_retains_configuration():
    assert not MechanismPolicy().enable_entity_options
    policy = MechanismPolicy(enable_entity_options=True)
    policy.reset()
    assert policy.enable_entity_options


def test_contextual_special_transition_allows_nonlocal_movement():
    """Consistent relocation backed by special transition becomes searchable."""
    controller, observation, clock, memory, _, _ = fixture()
    # Record a non-local special transition: (3,3) + ACTION4 -> (6,3)
    controller.special_transitions = {
        ((3, 3), "ACTION4", PATTERN, 14): (6, 3),
    }
    model = EntityOptionModel(controller, observation, clock, memory, {})
    start = model.bind(observation, clock)

    # Should allow the non-local transition
    step = model.step(start, "ACTION4")
    assert step is not None
    assert step.after.player == (6, 3)
    assert "contextual_special_transition" in step.hypotheses


def test_contextual_special_transition_overrides_geometric_navigation():
    """Special edge is used instead of geometric guess when available."""
    controller, observation, clock, memory, _, _ = fixture()
    # Record a special transition that differs from adjacent cell
    controller.special_transitions = {
        ((3, 3), "ACTION4", PATTERN, 14): (5, 3),  # Not the adjacent (4,3)
    }
    model = EntityOptionModel(controller, observation, clock, memory, {})
    start = model.bind(observation, clock)

    # Should use the special transition, not geometric adjacent
    step = model.step(start, "ACTION4")
    assert step is not None
    assert step.after.player == (5, 3)
    assert "contextual_special_transition" in step.hypotheses


def test_special_transition_rejected_for_wrong_status_context():
    """Special transition only applies in matching status context."""
    controller, observation, clock, memory, _, _ = fixture()
    # Record a special transition for a different pattern
    wrong_pattern = ((1, 0), (0, 1))
    controller.special_transitions = {
        ((3, 3), "ACTION4", wrong_pattern, 14): (6, 3),
    }
    model = EntityOptionModel(controller, observation, clock, memory, {})
    start = model.bind(observation, clock)

    # Should not match because pattern differs
    movement = model.movement(start, "ACTION4")
    # Falls back to geometric or other navigation
    assert movement is None or movement[1] != "contextual_special_transition"


def test_special_transition_rejected_for_wrong_color_context():
    """Special transition only applies in matching color context."""
    controller, observation, clock, memory, _, _ = fixture()
    # Record a special transition for a different color
    controller.special_transitions = {
        ((3, 3), "ACTION4", PATTERN, 9): (6, 3),  # color 9, not 14
    }
    model = EntityOptionModel(controller, observation, clock, memory, {})
    start = model.bind(observation, clock)

    # Should not match because color differs
    movement = model.movement(start, "ACTION4")
    # Falls back to geometric or other navigation
    assert movement is None or movement[1] != "contextual_special_transition"


def test_special_transition_works_in_goal_search():
    """Goal search can find paths using special transitions."""
    controller, observation, clock, memory, _, _ = fixture()
    # Set up a special transition and a target
    controller.special_transitions = {
        ((3, 3), "ACTION4", PATTERN, 14): (6, 3),
    }
    targets = {(6, 3): (PATTERN, 14)}
    model = EntityOptionModel(controller, observation, clock, memory, targets)
    start = model.bind(observation, clock)

    # Should find a path using the special transition
    path, expanded = model.search_goal(start, NAMES)
    assert path is not None
    assert len(path) > 0
    assert path[-1].after.player == (6, 3)
    # Check that at least one step used the special transition
    assert any("contextual_special_transition" in step.hypotheses for step in path)


def test_mechanism_policy_signature_keys_converted_to_pattern():
    """MechanismPolicy passes signature strings; model converts to Pattern."""
    from agents.mechanism_state_tracker import pattern_key

    controller, observation, clock, memory, _, _ = fixture()
    # MechanismPolicy stores keys with signature strings like "010/111/010"
    signature_str = pattern_key(PATTERN)
    controller.special_transitions = {
        ((3, 3), "ACTION4", signature_str, 14): (6, 3),
    }
    model = EntityOptionModel(controller, observation, clock, memory, {})
    start = model.bind(observation, clock)

    # Model should convert signature string to Pattern and find the transition
    step = model.step(start, "ACTION4")
    assert step is not None
    assert step.after.player == (6, 3)
    assert "contextual_special_transition" in step.hypotheses

    # Verify the key was actually converted
    assert ((3, 3), "ACTION4", PATTERN, 14) in model.special_transitions
    # Original signature-string key should not be in model's dict
    assert ((3, 3), "ACTION4", signature_str, 14) not in model.special_transitions


def test_learned_special_transition_overrides_bad_edge_after_mismatch():
    """After prediction mismatch, special transition remains searchable."""
    from agents.mechanism_state_tracker import pattern_key

    controller, observation, clock, memory, _, _ = fixture()
    signature_str = pattern_key(PATTERN)

    # First: plan expects geometric move (3,3) + ACTION4 -> (4,3)
    # But observe actual relocation to (6,3)
    # This adds ((3,3), "ACTION4") to bad_edges
    controller.bad_edges.add(((3, 3), "ACTION4"))

    # Meanwhile, MechanismPolicy records the observed relocation as special transition
    controller.special_transitions = {
        ((3, 3), "ACTION4", signature_str, 14): (6, 3),
    }

    # Build new model after invalidation
    model = EntityOptionModel(controller, observation, clock, memory, {})
    start = model.bind(observation, clock)

    # Special transition should override bad_edge; movement should work
    movement = model.movement(start, "ACTION4")
    assert movement is not None
    assert movement[0] == (6, 3)
    assert movement[1] == "contextual_special_transition"

    # step() should also work and use the special transition
    step = model.step(start, "ACTION4")
    assert step is not None
    assert step.after.player == (6, 3)
    assert "contextual_special_transition" in step.hypotheses

    # Goal search should be able to use this edge
    targets = {(6, 3): (PATTERN, 14)}
    model_with_target = EntityOptionModel(controller, observation, clock, memory, targets)
    start_with_target = model_with_target.bind(observation, clock)
    path, _ = model_with_target.search_goal(start_with_target, NAMES)
    assert path is not None
    assert any("contextual_special_transition" in step.hypotheses for step in path)


def test_relocation_learned_in_context_c1_usable_under_c1():
    """After observing (c,a) relocate under C1, planning under C1 uses it."""
    from agents.mechanism_state_tracker import pattern_key

    controller, observation, clock, memory, _, _ = fixture()
    signature_str = pattern_key(PATTERN)

    # Observe (3,3)+ACTION4 relocates to (6,3) under pattern=PATTERN, color=14
    controller.special_transitions = {
        ((3, 3), "ACTION4", signature_str, 14): (6, 3),
    }
    # Mark edge as known non-local
    controller.known_nonlocal_edges.add(((3, 3), "ACTION4"))

    model = EntityOptionModel(controller, observation, clock, memory, {})
    start = model.bind(observation, clock)

    # Planning under same context C1 (PATTERN, 14) should return contextual_special_transition
    assert start.pattern == PATTERN
    assert start.color == 14
    movement = model.movement(start, "ACTION4")
    assert movement is not None
    assert movement == ((6, 3), "contextual_special_transition")

    step = model.step(start, "ACTION4")
    assert step is not None
    assert step.after.player == (6, 3)
    assert "contextual_special_transition" in step.hypotheses


def test_relocation_under_c1_suppresses_geometric_under_c2():
    """After observing (c,a) relocate under C1, different context C2 must not use geometric."""
    from agents.mechanism_state_tracker import pattern_key

    controller, observation, clock, memory, _, _ = fixture()
    signature_str = pattern_key(PATTERN)

    # Observe (3,3)+ACTION4 relocates under pattern=PATTERN, color=14
    controller.special_transitions = {
        ((3, 3), "ACTION4", signature_str, 14): (6, 3),
    }
    # Mark edge as known non-local (MechanismPolicy.special_transitions)
    controller.known_nonlocal_edges.add(((3, 3), "ACTION4"))

    # Create node with DIFFERENT context C2: different pattern
    different_pattern = ((1, 0), (0, 1))
    observation_c2 = replace(
        observation,
        status_pattern=different_pattern,
        status_color=14,  # same color, different pattern
    )
    model = EntityOptionModel(controller, observation_c2, clock, memory, {})
    start_c2 = model.bind(observation_c2, clock)

    # Planning under different context C2 must NOT return geometric
    assert start_c2.pattern == different_pattern
    assert start_c2.color == 14
    movement = model.movement(start_c2, "ACTION4")
    # Should return None (known non-local, no contextual match), not geometric (4,3)
    assert movement is None

    # step() should also return None
    step = model.step(start_c2, "ACTION4")
    assert step is None


def test_goal_beyond_bounce_pad_not_planned_as_geometric_after_observation():
    """Goal beyond a bounce pad is not planned geometrically through the pad."""
    from agents.mechanism_state_tracker import pattern_key

    controller, observation, clock, memory, _, _ = fixture()
    signature_str = pattern_key(PATTERN)

    # Observe bounce pad: (3,3)+ACTION4 relocates to (6,3) under context (PATTERN, 14)
    controller.special_transitions = {
        ((3, 3), "ACTION4", signature_str, 14): (6, 3),
    }
    controller.known_nonlocal_edges.add(((3, 3), "ACTION4"))

    # Goal is at (4,3) - geometrically adjacent through the bounce pad
    targets = {(4, 3): (PATTERN, 14)}
    model = EntityOptionModel(controller, observation, clock, memory, targets)
    start = model.bind(observation, clock)

    # Search should NOT find a geometric path through the bounce pad
    path, _ = model.search_goal(start, NAMES, max_expansions=500)

    # Either no path found, or path does not use geometric ACTION4 from (3,3)
    if path is not None:
        for step in path:
            # No step should claim geometric navigation through the bounce pad
            if step.before.player == (3, 3) and step.action == "ACTION4":
                assert "geometric" not in step.hypotheses
                # Should either use contextual_special_transition or not exist
                if "contextual_special_transition" not in step.hypotheses:
                    assert False, "Unexpected hypothesis for bounce pad edge"


def test_new_evidence_invalidates_retained_plan_before_next_action():
    controller, observation, clock, memory, model, start = fixture()
    controller.model = model
    controller.remaining.append(model.step(start, "ACTION4"))
    memory.evidence[3] = replace(memory.evidence[2], transition_tick=3)
    controller.choose(observation, clock, memory, {}, NAMES)
    assert controller.metrics["invalidated_effect_evidence_changed"] == 1


def test_missing_or_empty_entity_view_is_not_replaced_by_raw_objects():
    controller, observation, clock, memory, model, start = fixture()
    missing = replace(observation, motion_entities=None)
    assert model.bind(missing, clock) is None
    assert controller.choose(missing, clock, memory, {}, NAMES) is None
    empty = replace(observation, motion_entities=())
    assert not EntityOptionModel.objects_at(empty, PHASES[0][1])
    assert model.objects_at(observation, PHASES[0][1]) == {SIG}


def test_actual_level_advance_is_not_credited_to_inactive_controller():
    policy = MechanismPolicy(enable_entity_options=True)
    policy._reset_level(5)
    policy._reset_level(6)
    metrics = policy.entity_option_archives[-1]["metrics"]
    assert metrics["actual_level_advanced"] == 1
    assert not metrics.get("option_goal_completed", 0)


def test_complete_action_preserves_observed_nonlocal_clock_status_and_resource():
    controller, before, clock, memory, _, _ = fixture()
    controller.observe(before, clock, None)
    after = state((6, 3), index=2, pattern=rotate(PATTERN), color=8, resource=58)
    controller.observe(after, dynamic(2), "ACTION4")
    model = EntityOptionModel(controller, after, dynamic(2), memory, {})
    start = model.bind(before, clock)
    step = model.step(start, "ACTION4")
    assert step.hypotheses == ("observed_complete_action",)
    assert step.after == model.bind(after, dynamic(2))
    assert step.after.phases != (PHASES[1], FAR[1])  # No local-clock guess.
    assert not step.terminal


def test_complete_action_does_not_project_resource_status_phase_or_fixture_context():
    controller, before, clock, memory, _, _ = fixture()
    controller.observe(before, clock, None)
    controller.observe(state((6, 3), index=1, resource=58), dynamic(1), "ACTION4")
    model = EntityOptionModel(controller, before, clock, memory, {})
    start = model.bind(before, clock)
    for altered in (
        replace(start, resource=59),
        replace(start, color=8),
        replace(start, pattern=rotate(PATTERN)),
        replace(start, phases=(PHASES[1], FAR[1])),
        replace(start, fixtures=frozenset({((9, 9), OTHER)})),
    ):
        assert model.step(altered, "ACTION4") is None


def test_complete_conflicting_outcomes_abstain_without_fallback_or_majority():
    controller, before, clock, memory, _, _ = fixture()
    for dest in ((6, 3), (6, 3), (5, 3)):
        controller.observe(before, clock, None)
        controller.observe(state(dest, index=1, resource=58), dynamic(1), "ACTION4")
    model = EntityOptionModel(controller, before, clock, memory, {})
    assert model.step(model.bind(before, clock), "ACTION4") is None


def test_rejected_arrival_abstains_across_approaches_but_exact_fact_overrides_ban():
    controller, before, clock, memory, model, start = fixture()
    controller.observe(before, clock, None)
    controller._install(model, (model.step(start, "ACTION4"),))
    after = state((6, 3), index=1, resource=58)
    controller.observe(after, dynamic(1), "ACTION4")
    model = EntityOptionModel(controller, after, dynamic(1), memory, {})
    assert model.step(start, "ACTION4").after.player == (6, 3)
    other_approach = replace(start, player=(4, 2))
    assert model.step(other_approach, "ACTION2") is None
    # A rejected projection is not a universal gate rule for every status.
    assert model.step(replace(other_approach, color=8), "ACTION2") is not None
    json.dumps(controller.diagnostics())


def test_visible_fixture_changes_invalidate_the_predicted_state():
    controller, before, clock, _, model, start = fixture()
    controller.observe(before, clock, None)
    controller._install(model, (model.step(start, "ACTION4"),))
    after = state((4, 3), index=1, resource=58)
    after = replace(
        after, motion_entities=after.motion_entities + (sprite((9, 9), (7, 1, "new")),)
    )
    controller.observe(after, dynamic(1), "ACTION4")
    assert controller.failures[-1]["fields"] == ["fixtures"]
    assert not controller.rejected_entries


def test_incomplete_clocks_never_produce_complete_action_evidence():
    controller, before, clock, memory, _, _ = fixture()
    controller.observe(before, clock, None)
    after_clock = dynamic(1)
    after_clock.tracks[SIG] = TrackedPhase(None, (6, 3))
    controller.observe(state((6, 3), index=1, resource=58), after_clock, "ACTION4")
    model = EntityOptionModel(controller, before, clock, memory, {})
    assert not model.atomic


def test_episode_reset_clears_belief_but_level_change_clears_complete_facts():
    controller, before, clock, _, _, _ = fixture()
    controller.observe(before, clock, None)
    controller.observe(state((6, 3), index=1, resource=58), dynamic(1), "ACTION4")
    assert len(controller.atomic_samples) == 1
    controller.reset_episode()
    assert controller.previous is None and len(controller.atomic_samples) == 1
    controller.observe(replace(before, level=6), clock, None)
    assert not controller.atomic_samples


def test_unknown_fixture_is_a_bounded_experiment_not_a_solution_or_refill_fact():
    controller, before, clock, memory, _, _ = fixture()
    refill_sig = (11, 8, "ring")
    before = replace(
        before, motion_entities=before.motion_entities + (sprite((4, 3), refill_sig),)
    )
    controller.refill_signatures[(0, 0)].add(refill_sig)
    controller.refill_values[(0, 0)] = 84
    model = EntityOptionModel(controller, before, clock, memory, {})
    start = model.bind(before, clock)
    assert model.step(start, "ACTION4") is None
    assert model.step(start, "ACTION4", experiment=True) is None
    path, _ = model.search_fixture_experiment(start, ("ACTION4",), Counter())
    assert len(path) == 1 and path[0].fixture == ((4, 3), refill_sig)
    assert set(path[0].unknown_fields) == {
        "pattern",
        "color",
        "resource",
        "refills",
        "fixtures",
    }
    assert not path[0].terminal and not start.refills
    assert (
        model.search_fixture_experiment(
            start, ("ACTION4",), Counter({path[0].fixture: 1})
        )[0]
        is None
    )
    controller.observe(before, clock, None)
    controller._install(model, path)
    after = state((4, 3), index=1, resource=84)
    controller.observe(after, dynamic(1), "ACTION4")
    assert controller.metrics["fixture_outcomes_observed"] == 1
    assert controller.refill_values[(4, 3)] == 84  # Learned from actual feedback.
    assert controller.metrics["prediction_errors"] == 0
    assert controller.metrics["known_fields_checked"] == 2  # Position and all clocks.
    assert controller.fixture_attempts[path[0].fixture] == 1


def test_fixture_experiment_requires_same_appearance_evidence_and_abstains_on_mixed_contact():
    controller, before, clock, memory, _, _ = fixture()
    unknown_sig = (7, 1, "unknown")
    before = replace(
        before, motion_entities=before.motion_entities + (sprite((4, 3), unknown_sig),)
    )
    model = EntityOptionModel(controller, before, clock, memory, {})
    assert (
        model.search_fixture_experiment(model.bind(before, clock), NAMES, Counter())[0]
        is None
    )
    controller.refill_signatures[(0, 0)].add(unknown_sig)
    before = replace(
        before, motion_entities=before.motion_entities + (sprite((3, 2), unknown_sig),)
    )
    model = EntityOptionModel(controller, before, clock, memory, {})
    assert (
        model.step(
            model.bind(before, clock),
            "ACTION1",
            experiment=True,
            fixture_experiment=True,
        )
        is None
    )
