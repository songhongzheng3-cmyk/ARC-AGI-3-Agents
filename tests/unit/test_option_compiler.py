from dataclasses import replace

from agents.dynamic_contact import DynamicContactModel, TrackedPhase, transform
from agents.mechanism_state_tracker import MechanismObject, MechanismState, pattern_key
from agents.option_compiler import (
    CompilerState,
    OptionCompiler,
    OptionExecutionController,
)
from agents.task_planning import JointMechanismState, JointMechanismTransition

SIG = (0, 3, "triangle")
PHASES = (
    ((5, 1), (4, 1)),
    ((4, 1), (4, 0)),
    ((4, 0), (3, 0)),
    ((3, 0), (3, 1)),
    ((3, 1), (3, 2)),
)
A = "10/11"
B = "110/011/101"


def pattern(key):
    return tuple(tuple(map(int, row)) for row in key.split("/"))


def state(player, status=A, color=12, resource=20, objects=()):
    walkable = frozenset((x, y) for x in range(6) for y in range(6))
    return MechanismState(
        level=4,
        game_state="NOT_FINISHED",
        layout=None,
        player_position=player,
        player_confidence=1,
        walkable_cells=walkable,
        navigation_projection_cells=walkable,
        blocked_cells=frozenset(),
        objects=tuple(objects),
        resource_level=resource,
        resource_capacity=40,
        status_pattern=pattern(status),
        matching_target_ids=(),
        status_color=color,
    )


def dynamic(phases=PHASES):
    model = DynamicContactModel()
    for p, q in zip(phases, phases[1:] + phases[:1]):
        for moved in (True, False):
            model.motion[(SIG, p, moved)][q] = 2
    model.operators[(SIG, True)] = {"rotate1"}
    model.effect_samples[(SIG, True)] = 2
    model.color_evidence[(SIG, True)][True] = 2
    return model


def _composition_fixture():
    transitions = [
        JointMechanismTransition(
            JointMechanismState(A, 12),
            (1, 1),
            JointMechanismState(B, 12),
            approach=(0, 1),
            contact_action="ACTION4",
            observed_result=(1, 1),
        ),
        JointMechanismTransition(
            JointMechanismState(A, 12),
            (2, 1),
            JointMechanismState(A, 8),
            approach=(1, 1),
            contact_action="ACTION4",
            observed_result=(2, 1),
        ),
    ]
    model = OptionCompiler(dynamic(), SIG, transitions)
    rotated_a = pattern_key(transform(pattern(A), "rotate1"))
    observations = [
        (state((0, 1)), state((1, 1), B)),
        (state((1, 1)), state((2, 1), color=8)),
        (state((2, 1)), state((3, 1), rotated_a)),
        (state((3, 1), B), state((4, 1), B)),
    ]
    for i, (before, after) in enumerate(observations):
        model.observe(
            before,
            "ACTION4",
            replace(after, resource_level=16),
            PHASES[i],
            PHASES[i + 1],
        )
    return model


def test_compiler_composes_static_palette_and_dynamic_contact_with_phase_bound_actions():
    model = _composition_fixture()
    start = CompilerState((0, 1), JointMechanismState(A, 12), PHASES[0], 8)
    goal = JointMechanismState(pattern_key(transform(pattern(B), "rotate1")), 8)
    plan = model.search(start, goal, (4, 1))

    assert plan.reason is None
    assert len(plan.steps) == 4
    assert all(step.action == "ACTION4" for step in plan.steps)
    assert [step.after.phase for step in plan.steps] == list(PHASES[1:])
    assert plan.steps[-1].after.status == goal
    assert plan.steps[-1].after.resource == 4
    assert "static_dimension_projection" in plan.steps[1].hypotheses
    assert "dynamic_contact_projection" in plan.steps[2].hypotheses


def test_compiler_labels_unobserved_geometric_edges_and_enforces_resource_limit():
    model = _composition_fixture()
    start = CompilerState((0, 1), JointMechanismState(A, 12), PHASES[0], 8)

    assert model.predict(start, "ACTION2") is not None
    assert (
        "geometric_navigation_projection" in model.predict(start, "ACTION2").hypotheses
    )
    assert model.search(start, start.status, (5, 5)).steps is None


def test_compiler_abstains_on_strong_context_conflict_not_weaker_fallback():
    model = _composition_fixture()
    model.observe(state((0, 1)), "ACTION4", state((4, 4)), PHASES[0], PHASES[1])
    start = CompilerState((0, 1), JointMechanismState(A, 12), PHASES[0], 8)

    assert model.predict(start, "ACTION4") is None
    assert model.metrics["navigation_conflict"]


def test_compiler_does_not_use_geometric_projection_over_special_transition():
    model = OptionCompiler(dynamic(), SIG, [])
    model.observe(
        state((0, 1)), "ACTION4", state((4, 4), resource=16), PHASES[0], PHASES[1]
    )
    start = CompilerState((0, 1), JointMechanismState(A, 12), PHASES[0], 8)

    result = model.predict(start, "ACTION4")
    assert result.after.player == (4, 4)
    assert "geometric_navigation_projection" not in result.hypotheses


def test_compiler_does_not_use_geometric_projection_into_observed_blocked_cell():
    model = OptionCompiler(dynamic(), SIG, [])
    before = state((1, 1))
    before = replace(
        before,
        blocked_cells=frozenset({(1, 2)}),
        walkable_cells=frozenset({(1, 1)}),
        navigation_projection_cells=frozenset({(1, 1)}),
    )
    after = replace(before, blocked_cells=frozenset({(1, 2)}), resource_level=16)
    model.observe(before, "ACTION1", after, PHASES[0], PHASES[1])
    start = CompilerState((1, 1), JointMechanismState(A, 12), PHASES[0], 8)

    assert model.predict(start, "ACTION2") is None


def test_compiler_does_not_project_through_partially_background_tile():
    import numpy as np

    from agents.mechanism_state_tracker import GridLayout, MechanismStateTracker

    grid = np.full((10, 10), 3)
    grid[5, 5:10] = 4
    layout = GridLayout((0, 0), (5, 5), 2, 2, 3, 4, 1)
    walkable, _ = MechanismStateTracker._tile_cells(grid, layout)
    projection = MechanismStateTracker._navigation_projection_cells(
        grid, layout, walkable
    )
    assert (1, 1) in walkable
    assert (1, 1) not in projection
    assert (0, 1) in projection

    model = OptionCompiler(dynamic(), SIG, [])
    before = replace(state((0, 1)), navigation_projection_cells=projection)
    model.observe(
        before, "ACTION1", replace(before, resource_level=16), PHASES[0], PHASES[1]
    )
    start = CompilerState((0, 1), JointMechanismState(A, 12), PHASES[0], 8)
    assert model.predict(start, "ACTION4") is None


def test_compiler_blocked_input_advances_phase_using_no_movement_rule():
    model = OptionCompiler(dynamic(), SIG, [])
    model.observe(
        state((0, 1)), "ACTION1", state((0, 1), resource=16), PHASES[0], PHASES[1]
    )
    start = CompilerState((0, 1), JointMechanismState(A, 12), PHASES[0], 8)
    prediction = model.predict(start, "ACTION1")

    assert prediction.after.player == start.player
    assert prediction.after.phase == PHASES[1]


def test_compiler_abstains_on_unknown_stationary_overlap_operator():
    p = ((2, 1), (1, 1))
    model = dynamic()
    model.motion[(SIG, p, False)][p] = 2
    compiler = OptionCompiler(model, SIG, [])
    compiler.observe(state((1, 1)), "ACTION1", state((1, 1), resource=16), p, p)

    assert (
        compiler.predict(
            CompilerState((1, 1), JointMechanismState(A, 12), p, 8), "ACTION1"
        )
        is None
    )
    assert compiler.metrics["contact_effect_unknown"]


def test_compiler_freezes_rules_instead_of_mutating_them_during_search():
    model = dynamic()
    compiler = OptionCompiler(model, SIG, [])
    compiler.observe(
        state((0, 1)), "ACTION4", state((1, 1), resource=16), PHASES[0], PHASES[1]
    )
    model.motion[(SIG, PHASES[0], True)][PHASES[2]] = 2

    prediction = compiler.predict(
        CompilerState((0, 1), JointMechanismState(A, 12), PHASES[0], 8), "ACTION4"
    )
    assert prediction.after.phase == PHASES[1]


def test_compiler_consumes_visible_refill_and_does_not_recreate_it_on_return():
    far = tuple(((20 + i, 20), (21 + i, 20)) for i in range(5))
    compiler = OptionCompiler(dynamic(far), SIG, [])
    refill = MechanismObject("refill", 11, (1, 1), (5, 5, 7, 7), 8, "ring")
    compiler.observe(
        state((0, 1), resource=4, objects=[refill]),
        "ACTION4",
        state((1, 1), resource=40),
        far[0],
        far[1],
    )
    compiler.observe(
        state((1, 1)), "ACTION4", state((2, 1), resource=16), far[1], far[2]
    )
    compiler.observe(
        state((2, 1)), "ACTION3", state((1, 1), resource=16), far[2], far[3]
    )
    initial = compiler.bind(state((0, 1), resource=4, objects=[refill]), far[0])
    first = compiler.predict(initial, "ACTION4")
    second = compiler.predict(first.after, "ACTION4")
    third = compiler.predict(second.after, "ACTION3")

    assert initial.refills == {(1, 1)}
    assert first.after.resource == 10
    assert first.after.refills == frozenset()
    assert third.after.resource == 8
    absent = compiler.bind(state((0, 1), resource=4), far[0])
    assert compiler.predict(absent, "ACTION4").after.resource == 0


def test_compiler_reserve_and_expansion_limits_are_hard_constraints():
    model = _composition_fixture()
    start = CompilerState((0, 1), JointMechanismState(A, 12), PHASES[0], 4)
    goal = JointMechanismState(pattern_key(transform(pattern(B), "rotate1")), 8)

    assert model.search(start, goal, (4, 1), reserve=2).steps is None
    limited = model.search(replace(start, resource=8), goal, (4, 1), max_expansions=1)
    assert limited.reason == "expansion_budget"
    assert limited.expanded == 1


def test_compiler_does_not_learn_automatic_restart_as_navigation_or_refill():
    model = OptionCompiler(dynamic(), SIG, [])
    model.observe(
        state((0, 1), resource=0),
        "ACTION4",
        state((1, 1), resource=40),
        PHASES[0],
        PHASES[1],
    )

    assert not model.samples
    assert not model.refill_values


def _execution_fixture():
    compiler = _composition_fixture()
    controller = OptionExecutionController()
    model = dynamic()
    model.moving.add(SIG)
    model.tracks[SIG] = TrackedPhase(PHASES[0], PHASES[0][1])
    initial = state((0, 1), resource=32)
    controller.observe(initial, None, model, compiler.transitions)
    # Independent historical experiments, not a successful path replay.
    for (position, action), samples in compiler.samples.items():
        for sample in samples:
            before = state(position, sample.status.pattern, sample.status.color)
            after = state(
                sample.destination,
                sample.after_status.pattern,
                sample.after_status.color,
                16,
            )
            controller.samples.append(
                (
                    before,
                    action,
                    after,
                    {SIG: TrackedPhase(sample.phase, sample.phase[1])},
                    {SIG: TrackedPhase(sample.after_phase, sample.after_phase[1])},
                )
            )
    goals = [
        ((4, 1), JointMechanismState(pattern_key(transform(pattern(B), "rotate1")), 8))
    ]
    return controller, model, initial, goals


def test_execution_requires_real_feedback_before_second_action():
    controller, model, initial, goals = _execution_fixture()
    step = controller.choose(initial, model, goals, ["ACTION4"])
    assert step is not None
    assert controller.choose(initial, model, goals, ["ACTION4"]) is None

    model.tracks[SIG] = TrackedPhase(PHASES[1], PHASES[1][1])
    after = state(
        step.after.player,
        step.after.status.pattern,
        step.after.status.color,
        step.after.resource * 4,
    )
    controller.observe(after, step.action, model, [])
    assert controller.metrics["validated_steps"] == 1
    next_step = controller.choose(after, model, goals, ["ACTION4"])
    assert next_step is not None
    assert next_step.before == step.after


def test_execution_invalidates_entire_path_and_records_actual_counterexample():
    controller, model, initial, goals = _execution_fixture()
    step = controller.choose(initial, model, goals, ["ACTION4"])
    model.tracks[SIG] = TrackedPhase(PHASES[1], PHASES[1][1])
    unexpected = state(
        (4, 4),
        step.after.status.pattern,
        step.after.status.color,
        step.after.resource * 4,
    )
    controller.observe(unexpected, step.action, model, [])
    assert not controller.remaining
    assert controller.expected is None
    assert controller.compiler is None
    assert controller.metrics["prediction_errors"] == 1
    assert controller.failures[-1]["fields"] == ("player",)
    assert controller.samples[-1][2] == unexpected


def test_execution_uses_actual_fallback_action_and_preserves_evidence_on_reset():
    controller, model, initial, goals = _execution_fixture()
    controller.choose(initial, model, goals, ["ACTION4"])
    controller.observe(initial, "ACTION1", model, [])
    assert "executed_action" in controller.failures[-1]["fields"]
    assert controller.samples[-1][1] == "ACTION1"
    count = len(controller.samples)
    controller.reset_episode()
    assert len(controller.samples) == count
    assert controller.previous is None
    assert not controller.previous_tracks
    assert controller.expected is None


def test_execution_has_hard_search_and_mismatch_limits():
    controller, model, initial, goals = _execution_fixture()
    controller.metrics["searches"] = 32
    assert controller.choose(initial, model, goals, ["ACTION4"]) is None
    controller.metrics["searches"] = 0
    controller.metrics["prediction_errors"] = 3
    assert controller.choose(initial, model, goals, ["ACTION4"]) is None
