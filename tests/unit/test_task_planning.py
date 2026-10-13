from agents.task_planning import (
    JointMechanismGraph,
    JointMechanismState,
    PlanningState,
    PlanStage,
    PlanStageKind,
    TaskPlanController,
)


def _planning_state(
    *,
    status: JointMechanismState,
    player=(1, 1),
    topology_revision=0,
    events=(),
):
    return PlanningState(
        level=1,
        player=player,
        joint_status=status,
        resource_steps=10,
        topology_revision=topology_revision,
        event_types=frozenset(events),
    )


def test_joint_graph_finds_observed_multi_actuator_path():
    graph = JointMechanismGraph()
    initial = JointMechanismState("a", 8)
    intermediate = JointMechanismState("b", 8)
    goal = JointMechanismState("b", 9)

    graph.record(initial, (2, 1), intermediate)
    graph.record(intermediate, (3, 1), goal)

    path = graph.shortest_path(initial, goal)

    assert path is not None
    assert [transition.actuator for transition in path] == [(2, 1), (3, 1)]
    assert path[-1].destination == goal


def test_joint_graph_preserves_observed_contact_contract():
    graph = JointMechanismGraph()
    initial = JointMechanismState("a", 8)
    goal = JointMechanismState("b", 8)

    graph.record(
        initial,
        (2, 1),
        goal,
        approach=(1, 1),
        contact_action="ACTION4",
        observed_result=(2, 1),
    )

    path = graph.shortest_path(initial, goal)

    assert path is not None
    assert path[0].approach == (1, 1)
    assert path[0].contact_action == "ACTION4"
    assert path[0].observed_result == (2, 1)


def test_joint_graph_projects_independent_pattern_and_palette_effects():
    graph = JointMechanismGraph()
    pattern_a = "010/110/011"
    pattern_b = "111/001/101"

    # Palette cycle was observed only while pattern A was active.
    graph.record(
        JointMechanismState(pattern_a, 14),
        (6, 6),
        JointMechanismState(pattern_a, 8),
    )
    graph.record(
        JointMechanismState(pattern_a, 8),
        (6, 6),
        JointMechanismState(pattern_a, 12),
    )
    # Pattern transition was observed only while color 14 was active.
    graph.record(
        JointMechanismState(pattern_a, 14),
        (4, 6),
        JointMechanismState(pattern_b, 14),
    )

    path = graph.shortest_path(
        JointMechanismState(pattern_a, 8),
        JointMechanismState(pattern_b, 12),
    )

    assert path is not None
    assert {transition.actuator for transition in path} == {(4, 6), (6, 6)}
    assert path[-1].destination == JointMechanismState(pattern_b, 12)


def test_joint_graph_does_not_project_joint_effects():
    graph = JointMechanismGraph()
    graph.record(
        JointMechanismState("a", 1),
        (2, 2),
        JointMechanismState("b", 2),
    )

    assert (
        graph.shortest_path(
            JointMechanismState("a", 3),
            JointMechanismState("b", 2),
        )
        is None
    )


def test_task_plan_persists_until_observed_effect_advances_stage():
    controller = TaskPlanController()
    initial = JointMechanismState("a", 8)
    goal = JointMechanismState("b", 9)
    plan = controller.install(
        level=1,
        goal=goal,
        topology_revision=0,
        stages=(
            PlanStage(
                kind=PlanStageKind.APPLY_ACTUATOR,
                target=(2, 1),
                expected_before=initial,
                expected_after=goal,
            ),
            PlanStage(
                kind=PlanStageKind.CONTACT_TARGET,
                target=(4, 4),
                expected_before=goal,
                expected_after=goal,
            ),
        ),
    )

    unchanged = controller.observe(
        _planning_state(status=initial, player=(2, 1)),
        target_available=lambda _cell: True,
    )
    assert unchanged is plan
    assert plan.current_stage_index == 0

    advanced = controller.observe(
        _planning_state(
            status=goal,
            player=(2, 1),
            events=("status_pattern_changed", "status_palette_changed"),
        ),
        target_available=lambda _cell: True,
    )
    assert advanced is plan
    assert plan.current_stage_index == 1
    assert plan.current_stage is not None
    assert plan.current_stage.kind is PlanStageKind.CONTACT_TARGET


def test_task_plan_survives_topology_refinement():
    controller = TaskPlanController()
    status = JointMechanismState("a", 8)
    controller.install(
        level=1,
        goal=status,
        topology_revision=2,
        stages=(
            PlanStage(
                kind=PlanStageKind.CONTACT_TARGET,
                target=(4, 4),
                expected_before=status,
                expected_after=status,
            ),
        ),
    )

    advanced = controller.observe(
        _planning_state(status=status, topology_revision=3),
        target_available=lambda _cell: True,
    )
    assert advanced is not None
    assert controller.active_plan is advanced
    assert controller.active_plan.topology_revision == 3
    assert controller.last_invalid_reason is None


def test_task_plan_abandons_repeated_unproductive_stage():
    controller = TaskPlanController(maximum_stage_attempts=2)
    status = JointMechanismState("a", 8)
    controller.install(
        level=1,
        goal=JointMechanismState("b", 8),
        topology_revision=0,
        stages=(
            PlanStage(
                kind=PlanStageKind.APPLY_ACTUATOR,
                target=(2, 1),
                expected_before=status,
                expected_after=JointMechanismState("b", 8),
            ),
        ),
    )

    assert not controller.record_failed_attempt()
    assert controller.record_failed_attempt()
    assert controller.active_plan is None
    assert controller.last_invalid_reason == "stage_effect_not_observed"
