from agents.task_planning import (
    EvidenceKind,
    JointMechanismGraph,
    JointMechanismState,
    PlanningState,
    PlanStage,
    PlanStageKind,
    ReachabilityOption,
    ReachabilityState,
    StateReachabilityPlanner,
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


def _option(
    option_id,
    source,
    destination,
    *,
    approach=(1, 1),
    observed_result=(1, 1),
    evidence=EvidenceKind.OBSERVED,
    source_phase=None,
    destination_phase=None,
    contact_action="ACTION4",
    contact_cost=1,
    projected=False,
    refill_to=None,
):
    return ReachabilityOption(
        option_id=option_id,
        source=source,
        destination=destination,
        evidence=evidence,
        actuator=(2, 1),
        approach=approach,
        contact_action=contact_action,
        observed_result=observed_result,
        source_phase=source_phase,
        destination_phase=destination_phase,
        contact_cost=contact_cost,
        projected=projected,
        refill_to=refill_to,
    )


def test_reachability_planner_finds_resource_feasible_observed_chain():
    start = JointMechanismState("010", 12)
    middle = JointMechanismState("111", 12)
    goal = JointMechanismState("101", 8)
    planner = StateReachabilityPlanner(
        [
            _option("pattern", start, middle, observed_result=(2, 1)),
            _option("palette", middle, goal, approach=(2, 1)),
        ]
    )

    result = planner.search(
        ReachabilityState((1, 1), start, resource_steps=10),
        goal,
    )

    assert result.reason is None
    assert result.actionable
    assert [step.option.option_id for step in result.path or ()] == [
        "pattern",
        "palette",
    ]
    assert result.path[-1].to_state.resource_steps == 8


def test_reachability_planner_rejects_missing_contract_even_when_status_path_exists():
    start = JointMechanismState("a", 1)
    goal = JointMechanismState("b", 1)
    planner = StateReachabilityPlanner(
        [_option("unverified", start, goal, approach=None, contact_action=None)]
    )

    result = planner.search(ReachabilityState((1, 1), start, resource_steps=10), goal)

    assert result.path is None
    assert result.reason == "no_supported_path"
    assert result.rejected_missing_contract == 1


def test_reachability_planner_reports_projection_but_does_not_execute_it():
    start = JointMechanismState("a", 9)
    goal = JointMechanismState("b", 9)
    option = _option(
        "projected-pattern",
        start,
        goal,
        evidence=EvidenceKind.PROJECTED,
        projected=True,
    )
    planner = StateReachabilityPlanner([option])

    diagnostic = planner.search(
        ReachabilityState((1, 1), start, resource_steps=4),
        goal,
        allow_projected=True,
        require_actionable=False,
    )
    gated = planner.search(
        ReachabilityState((1, 1), start, resource_steps=4),
        goal,
        allow_projected=True,
        require_actionable=True,
    )

    assert diagnostic.path is not None
    assert diagnostic.projected_steps == 1
    assert not diagnostic.actionable
    assert gated.path is None
    assert gated.rejected_projection == 1


def test_reachability_planner_requires_dynamic_phase():
    start = JointMechanismState("a", 12)
    goal = JointMechanismState("b", 12)
    phase = ((2, 2), (2, 3))
    option = _option(
        "moving-contact",
        start,
        goal,
        evidence=EvidenceKind.DYNAMIC,
        source_phase=phase,
        destination_phase=((2, 3), (3, 3)),
    )
    planner = StateReachabilityPlanner([option])

    missing = planner.search(
        ReachabilityState((1, 1), start, phase=None, resource_steps=5), goal
    )
    valid = planner.search(
        ReachabilityState((1, 1), start, phase=phase, resource_steps=5), goal
    )

    assert missing.path is None
    assert missing.reason == "no_supported_path"
    assert missing.rejected_phase == 1
    assert valid.actionable


def test_reachability_planner_rejects_resource_shortfall():
    start = JointMechanismState("a", 1)
    goal = JointMechanismState("b", 1)
    planner = StateReachabilityPlanner(
        [_option("expensive", start, goal, contact_cost=4)]
    )

    result = planner.search(ReachabilityState((1, 1), start, resource_steps=3), goal)

    assert result.path is None
    assert result.reason == "no_supported_path"
    assert result.rejected_resource == 1


def test_reachability_planner_accepts_already_satisfied_goal():
    status = JointMechanismState("a", 1)
    result = StateReachabilityPlanner([]).search(
        ReachabilityState((1, 1), status, resource_steps=0), status
    )

    assert result.path == ()
    assert result.actionable


def test_reachability_planner_does_not_treat_unknown_color_as_goal():
    status = JointMechanismState("a", None)
    goal = JointMechanismState("a", 8)
    result = StateReachabilityPlanner([]).search(
        ReachabilityState((1, 1), status, resource_steps=0), goal
    )

    assert result.path is None
    assert result.reason == "no_supported_path"


def test_reachability_requires_navigation_edges_not_manhattan_distance():
    start, goal = JointMechanismState("a", 1), JointMechanismState("b", 1)
    result = StateReachabilityPlanner(
        [_option("far", start, goal, approach=(5, 1))]
    ).search(ReachabilityState((1, 1), start, resource_steps=100), goal)

    assert result.path is None


def test_reachability_updates_phase_during_navigation_and_rejects_phase_freeze():
    status, goal = JointMechanismState("a", 1), JointMechanismState("b", 1)
    phase = ((2, 2), (3, 2))
    next_phase = ((3, 2), (4, 2))
    planner = StateReachabilityPlanner(
        [
            _option("illegal-freeze", status, goal),
            _option(
                "walk",
                status,
                status,
                observed_result=(2, 1),
                source_phase=phase,
                destination_phase=next_phase,
            ),
            _option(
                "contact",
                status,
                goal,
                approach=(2, 1),
                evidence=EvidenceKind.DYNAMIC,
                source_phase=next_phase,
                destination_phase=((4, 2), (5, 2)),
            ),
        ]
    )
    result = planner.search(
        ReachabilityState((1, 1), status, phase=phase, resource_steps=3), goal
    )

    assert [step.option.option_id for step in result.path] == ["walk", "contact"]
    assert result.path[-1].to_state.phase == ((4, 2), (5, 2))
    assert result.rejected_phase > 0


def test_reachability_abstains_on_conflicting_observations_instead_of_latest_wins():
    start, goal = JointMechanismState("a", 1), JointMechanismState("b", 1)
    planner = StateReachabilityPlanner(
        [
            _option("first", start, goal),
            _option("contradiction", start, JointMechanismState("c", 1)),
        ]
    )
    result = planner.search(ReachabilityState((1, 1), start, resource_steps=10), goal)

    assert result.path is None
    assert result.rejected_conflicts == 2


def test_reachability_reserve_and_real_refill_allow_multi_segment_mission():
    status, goal = JointMechanismState("a", 1), JointMechanismState("b", 1)
    options = [
        _option("refill", status, status, observed_result=(2, 1), refill_to=8),
        _option("apply", status, goal, approach=(2, 1), observed_result=(3, 1)),
        _option("return", goal, goal, approach=(3, 1), observed_result=(4, 1)),
    ]
    planner = StateReachabilityPlanner(options)
    result = planner.search(
        ReachabilityState((1, 1), status, resource_steps=1),
        goal,
        goal_cell=(4, 1),
        reserve_steps=2,
        capacity_steps=8,
    )
    no_refill = StateReachabilityPlanner(options[1:]).search(
        ReachabilityState((2, 1), status, resource_steps=2),
        goal,
        goal_cell=(4, 1),
        reserve_steps=2,
    )

    assert result.actionable
    assert len(result.path) == 3
    assert result.path[-1].to_state.resource_steps == 6
    assert no_refill.path is None


def test_reachability_is_bounded_even_when_refill_and_cycles_exist():
    status, goal = JointMechanismState("a", 1), JointMechanismState("b", 1)
    planner = StateReachabilityPlanner(
        [
            _option("loop", status, status, refill_to=10),
        ]
    )
    result = planner.search(
        ReachabilityState((1, 1), status, resource_steps=1),
        goal,
        capacity_steps=10,
        max_expansions=1,
    )

    assert result.path is None
    assert result.reason == "expansion_budget"
    assert result.expanded_states == 1


def test_reachability_does_not_generalize_resource_guarded_refill_to_other_resource():
    from dataclasses import replace

    status, goal = JointMechanismState("a", 1), JointMechanismState("b", 1)
    refill = replace(
        _option("refill", status, goal, refill_to=10), required_resource_steps=2
    )
    planner = StateReachabilityPlanner([refill])
    unsupported = planner.search(
        ReachabilityState((1, 1), status, resource_steps=1), goal, capacity_steps=10
    )
    supported = planner.search(
        ReachabilityState((1, 1), status, resource_steps=2), goal, capacity_steps=10
    )

    assert unsupported.path is None
    assert supported.actionable
    assert supported.path[-1].to_state.resource_steps == 10


def test_reachability_uses_conservative_cost_not_the_cheapest_duplicate_sample():
    start, goal = JointMechanismState("a", 1), JointMechanismState("b", 1)
    planner = StateReachabilityPlanner(
        [
            _option("cheap", start, goal, contact_cost=1),
            _option("costly", start, goal, contact_cost=3),
        ]
    )
    result = planner.search(ReachabilityState((1, 1), start, resource_steps=2), goal)

    assert result.path is None
    assert result.rejected_resource == 2


def test_reachability_does_not_allow_a_projection_to_override_observed_failure():
    start, goal = JointMechanismState("a", 1), JointMechanismState("b", 1)
    planner = StateReachabilityPlanner(
        [
            _option("observed", start, start),
            _option(
                "wishful", start, goal, evidence=EvidenceKind.PROJECTED, projected=True
            ),
        ]
    )
    result = planner.search(
        ReachabilityState((1, 1), start, resource_steps=2),
        goal,
        allow_projected=True,
        require_actionable=False,
    )

    assert result.path is None
    assert result.rejected_conflicts > 0
