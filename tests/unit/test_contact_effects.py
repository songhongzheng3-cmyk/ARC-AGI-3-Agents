from copy import deepcopy
from dataclasses import replace

from agents.contact_effects import ContactEffectMemory
from agents.dynamic_contact import DynamicContactModel, TrackedPhase, rotate
from agents.mechanism_policy import MechanismPolicy
from agents.mechanism_state_tracker import GridLayout, MechanismObject, MechanismState
from scripts.probe_contact_effects import compile_probe

PATTERN = ((0, 1, 0), (1, 1, 0), (0, 1, 1))
SIG = (2, 3, "sprite")
OTHER = (8, 3, "other")
PHASE = ((1, 2), (2, 2))
NEXT = ((2, 2), (3, 2))


def sprite(cell, signature=SIG):
    x, y = cell
    return MechanismObject(
        str(signature),
        signature[0],
        cell,
        (5 * x + 1, 5 * y + 1, 5 * x + 2, 5 * y + 2),
        signature[1],
        signature[2],
    )


def state(cell=(1, 2), *, player=(3, 4), pattern=PATTERN, color=14):
    entities = () if cell is None else (sprite(cell),)
    return MechanismState(
        level=5,
        game_state="NOT_FINISHED",
        layout=GridLayout((0, 0), (5, 5), 10, 10, 3, 4, 0.95),
        player_position=player,
        player_confidence=0.99,
        walkable_cells=frozenset(),
        blocked_cells=frozenset(),
        objects=entities,
        motion_entities=entities,
        resource_level=60,
        resource_capacity=84,
        status_pattern=pattern,
        status_color=color,
        matching_target_ids=(),
    )


def model(*, clocks=True):
    dynamic = DynamicContactModel()
    dynamic.moving.add(SIG)
    if clocks:
        dynamic.motion[(SIG, PHASE, True)][NEXT] = 2
        dynamic.motion[(SIG, NEXT, True)][((3, 2), (2, 2))] = 2
    return dynamic


def episode(pattern=PATTERN, color=8):
    return [
        (state(pattern=pattern), "ACTION6"),
        (state((2, 2), player=(3, 3), pattern=pattern), "ACTION1"),
        (state(None, player=(3, 2), pattern=rotate(pattern), color=color), "ACTION1"),
        (state((2, 2), player=(4, 2), pattern=rotate(pattern), color=color), "ACTION4"),
    ]


def feed(memory, dynamic, samples):
    for observation, action in samples:
        memory.observe(observation, dynamic, action=action)


def test_gap_waits_for_reappearance_and_records_actual_availability():
    memory, dynamic = ContactEffectMemory(), model()
    samples = episode()
    feed(memory, dynamic, samples[:3])
    assert not memory.evidence
    feed(memory, dynamic, samples[3:])
    evidence = memory.evidence[2]
    assert evidence.available_tick == 3 and evidence.anchors == ((1, 3),)
    assert evidence.relation == "meet_after_moving"
    assert evidence.color_before == 14 and evidence.color_after == 8


def test_later_prefix_clocks_can_recover_earlier_samples_without_motion_training():
    memory, dynamic = ContactEffectMemory(), model(clocks=False)
    feed(memory, dynamic, episode())
    assert not memory.evidence
    dynamic.motion = model().motion
    rules = deepcopy(dynamic.motion)
    memory.observe(state(), dynamic, action="ACTION6")
    assert memory.evidence[2].available_tick == 4
    assert dynamic.motion == rules


def test_reanalysis_never_double_counts_and_two_distinct_samples_predict():
    memory, dynamic = ContactEffectMemory(), model()
    feed(memory, dynamic, episode())
    prediction = memory.predict(SIG, "meet_after_moving", True, PATTERN, 14)
    assert prediction.pattern is prediction.color is None
    feed(memory, dynamic, episode(rotate(PATTERN)))
    assert len(memory.evidence) == 2
    prediction = memory.predict(SIG, "meet_after_moving", True, PATTERN, 14)
    assert prediction.pattern == rotate(PATTERN) and prediction.color == 8
    assert prediction.pattern_support == prediction.color_support == 2
    for _ in range(8):
        memory.observe(state(), dynamic, action="ACTION6")
    assert len(memory.evidence) == 2
    assert memory.evidence[2].available_tick == 3


def test_new_motion_contradiction_retracts_old_inferred_effects():
    memory, dynamic = ContactEffectMemory(), model()
    feed(memory, dynamic, episode())
    dynamic.motion[(SIG, PHASE, True)][((2, 2), (1, 2))] = 1
    memory.observe(state(), dynamic, action="ACTION6")
    assert not memory.evidence and memory.metrics["retracted_samples"] == 1


def test_bad_reappearance_abstains_for_entire_occlusion_segment():
    memory, dynamic = ContactEffectMemory(), model()
    samples = episode()
    samples[-1] = (
        replace(samples[-1][0], motion_entities=(sprite((1, 2)),)),
        "ACTION4",
    )
    feed(memory, dynamic, samples)
    assert not memory.evidence


def test_unknown_other_mover_blocks_unique_attribution():
    memory, dynamic = ContactEffectMemory(), model()
    dynamic.moving.add(OTHER)
    feed(memory, dynamic, episode())
    assert not memory.evidence and memory.rejections["unresolved_mover"]


def test_simultaneous_contacts_are_not_assigned_to_either_entity():
    memory, dynamic = ContactEffectMemory(), model()
    dynamic.moving.add(OTHER)
    for key, outcomes in list(dynamic.motion.items()):
        dynamic.motion[(OTHER, key[1], key[2])] = outcomes.copy()
    samples = []
    for observation, action in episode():
        extra = tuple(sprite(item.cell, OTHER) for item in observation.motion_entities)
        samples.append(
            (
                replace(
                    observation, motion_entities=observation.motion_entities + extra
                ),
                action,
            )
        )
    feed(memory, dynamic, samples)
    assert not memory.evidence and memory.rejections["multiple_contacts"] == 1


def test_duplicate_identity_breaks_gap_even_when_later_reappearance_matches():
    memory, dynamic = ContactEffectMemory(), model()
    samples = episode()
    samples[2] = (
        replace(samples[2][0], motion_entities=(sprite((1, 2)), sprite((2, 2)))),
        "ACTION1",
    )
    feed(memory, dynamic, samples)
    assert not memory.evidence


def test_low_confidence_or_nonlocal_movement_cannot_bridge_gap():
    for change in ({"player_confidence": 0.1}, {"player_position": (7, 7)}):
        memory, dynamic = ContactEffectMemory(), model()
        samples = episode()
        samples[2] = (replace(samples[2][0], **change), "ACTION1")
        feed(memory, dynamic, samples)
        assert not memory.evidence


def test_gap_limit_and_raw_history_are_bounded():
    memory, dynamic = ContactEffectMemory(), model()
    memory.MAX_GAP = 0
    feed(memory, dynamic, episode())
    assert not memory.evidence
    for _ in range(memory.MAX_HISTORY + 10):
        memory.observe(state(), dynamic, action="ACTION6")
    assert len(memory.history) == memory.MAX_HISTORY


def test_frozen_suffix_cannot_train_or_be_staged_for_later_training():
    memory, dynamic = ContactEffectMemory(), model()
    feed(memory, dynamic, episode())
    frozen = deepcopy((memory.history, memory.evidence))
    for observation, action in episode(color=9):
        memory.observe(observation, dynamic, action=action, learn=False)
    assert (memory.history, memory.evidence) == frozen


def test_resuming_learning_cannot_bridge_unrecorded_frozen_observations():
    memory, dynamic = ContactEffectMemory(), model()
    feed(memory, dynamic, episode()[:2])
    memory.observe(state(), dynamic, action="ACTION6", learn=False)
    feed(memory, dynamic, episode()[2:])
    assert not memory.evidence


def test_relation_movement_and_input_color_are_separate_keys():
    memory, dynamic = ContactEffectMemory(), model()
    feed(memory, dynamic, episode() * 2)
    assert memory.predict(SIG, "crossing", True, PATTERN, 14).pattern is None
    assert memory.predict(SIG, "meet_after_moving", False, PATTERN, 14).color is None
    assert memory.predict(SIG, "meet_after_moving", True, PATTERN, 8).color is None


def test_effect_contradictions_abstain_instead_of_majority_guess():
    memory, dynamic = ContactEffectMemory(), model()
    feed(memory, dynamic, episode() * 2 + episode(color=9))
    assert memory.predict(SIG, "meet_after_moving", True, PATTERN, 14).color is None
    samples = episode()
    samples[2] = (replace(samples[2][0], status_pattern=PATTERN), "ACTION1")
    samples[3] = (replace(samples[3][0], status_pattern=PATTERN), "ACTION4")
    feed(memory, dynamic, samples)
    assert memory.predict(SIG, "meet_after_moving", True, PATTERN, 14).pattern is None


def test_reset_and_level_change_clear_evidence_not_share_hidden_heading():
    memory, dynamic = ContactEffectMemory(), model()
    feed(memory, dynamic, episode())
    memory.observe(state(), dynamic, action="RESET")
    assert not memory.evidence and len(memory.history) == 1
    feed(memory, dynamic, episode())
    assert memory.evidence
    memory.observe(replace(state(), level=6), dynamic, action="ACTION1")
    assert not memory.evidence and memory.tick == 0


def test_policy_feature_is_opt_in_and_preserved_on_reset():
    assert not MechanismPolicy().enable_contact_effects
    policy = MechanismPolicy(enable_contact_effects=True)
    feed(policy.contact_effects, model(), episode())
    policy.note_reset()
    assert not policy.contact_effects.evidence
    policy.reset()
    assert policy.enable_contact_effects


def test_missing_entity_view_inside_gap_cannot_seed_later_attribution():
    memory, dynamic = ContactEffectMemory(), model()
    samples = episode()
    samples[2] = (replace(samples[2][0], motion_entities=None), "ACTION1")
    feed(memory, dynamic, samples)
    assert not memory.evidence


def test_changed_layout_inside_gap_cannot_reuse_old_coordinates():
    memory, dynamic = ContactEffectMemory(), model()
    samples = episode()
    samples[2] = (
        replace(samples[2][0], layout=replace(samples[2][0].layout, origin=(1, 0))),
        "ACTION1",
    )
    feed(memory, dynamic, samples)
    assert not memory.evidence


def test_unknown_action_breaks_gap_even_if_player_positions_look_plausible():
    memory, dynamic = ContactEffectMemory(), model()
    samples = episode()
    samples[2] = (samples[2][0], "ACTION6")
    feed(memory, dynamic, samples)
    assert not memory.evidence


def test_automatic_empty_resource_restart_clears_effect_evidence():
    memory, dynamic = ContactEffectMemory(), model()
    feed(memory, dynamic, episode())
    memory.observe(replace(state(), resource_level=0), dynamic, action="ACTION6")
    memory.observe(state(), dynamic, action="ACTION1")
    assert not memory.evidence and len(memory.history) == 1


def test_policy_archives_effect_evidence_at_level_boundary():
    policy = MechanismPolicy(enable_contact_effects=True)
    policy.level = 5
    feed(policy.contact_effects, model(), episode())
    policy._reset_level(6)
    assert not policy.contact_effects.evidence
    assert policy.contact_effect_archives[0]["level"] == 5
    assert len(policy.contact_effect_archives[0]["evidence"]) == 1


def probe_fixture():
    memory, dynamic = ContactEffectMemory(), model()
    feed(memory, dynamic, episode() * 2)
    dynamic.tracks[SIG] = TrackedPhase(PHASE, (2, 2))
    current = replace(
        state((2, 2), player=(3, 3)),
        navigation_projection_cells=frozenset({(3, 2)}),
    )
    return memory, dynamic, current


def test_diagnostic_probe_compiles_new_first_contact_not_a_fixed_action_list():
    memory, dynamic, current = probe_fixture()
    plan = compile_probe(current, dynamic, memory, resource_cost=2)
    assert len(plan["steps"]) == 1
    step = plan["steps"][0]
    assert step["action"] == "ACTION1" and step["pattern"] == rotate(PATTERN)
    assert step["color"] == 8 and step["resource"] == 58


def test_diagnostic_probe_requires_every_phase_and_never_unknown_palette():
    memory, dynamic, current = probe_fixture()
    assert (
        compile_probe(
            replace(current, status_color=8), dynamic, memory, resource_cost=2
        )
        is None
    )
    dynamic.moving.add(OTHER)
    assert compile_probe(current, dynamic, memory, resource_cost=2) is None


def test_diagnostic_probe_obeys_budget_horizon_and_projection():
    memory, dynamic, current = probe_fixture()
    assert compile_probe(current, dynamic, memory, resource_cost=2, horizon=0) is None
    assert (
        compile_probe(
            replace(current, resource_level=2), dynamic, memory, resource_cost=2
        )
        is None
    )
    assert (
        compile_probe(
            replace(current, navigation_projection_cells=frozenset()),
            dynamic,
            memory,
            resource_cost=2,
        )
        is None
    )


def test_diagnostic_probe_does_not_cross_unknown_static_interactions():
    memory, dynamic, current = probe_fixture()
    obstacle = replace(sprite((3, 2), OTHER), static_target=True)
    current = replace(current, objects=current.objects + (obstacle,))
    assert compile_probe(current, dynamic, memory, resource_cost=2) is None


def test_diagnostic_probe_abstains_on_unmodelled_motion():
    memory, dynamic, current = probe_fixture()
    dynamic.motion[(SIG, PHASE, True)].clear()
    assert compile_probe(current, dynamic, memory, resource_cost=2) is None


def test_diagnostic_probe_never_arbitrarily_orders_simultaneous_contacts():
    memory, dynamic, current = probe_fixture()
    dynamic.moving.add(OTHER)
    dynamic.tracks[OTHER] = TrackedPhase(PHASE, (2, 2))
    dynamic.motion[(OTHER, PHASE, True)][NEXT] = 2
    assert compile_probe(current, dynamic, memory, resource_cost=2) is None


def test_diagnostic_color_probe_uses_learned_palette_not_pattern_change():
    memory, dynamic = ContactEffectMemory(), model()
    samples = episode() * 2
    samples = [(replace(s, status_pattern=PATTERN), a) for s, a in samples]
    feed(memory, dynamic, samples)
    dynamic.tracks[SIG] = TrackedPhase(PHASE, (2, 2))
    current = replace(
        state((2, 2), player=(3, 3)),
        navigation_projection_cells=frozenset({(3, 2)}),
    )
    assert compile_probe(current, dynamic, memory, resource_cost=2) is None
    plan = compile_probe(current, dynamic, memory, resource_cost=2, effect="color")
    assert plan["steps"][0]["color"] == 8
    assert plan["steps"][0]["pattern"] == PATTERN
