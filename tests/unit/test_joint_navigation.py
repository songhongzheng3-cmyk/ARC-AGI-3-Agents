from agents.joint_navigation import JointNavigationModel, NavigationNode
from agents.motion_context import MotionObservation
from agents.task_planning import JointMechanismState


def node(position, pattern="A", color=1):
    return NavigationNode(position, JointMechanismState(pattern, color))


def record(model, before, action, after, resource_before=20, resource_after=18):
    model.record(
        before,
        action,
        after,
        resource_before=resource_before,
        resource_after=resource_after,
    )


def test_joint_route_changes_status_and_uses_topology_in_new_context():
    model = JointNavigationModel()
    a, b, c, d = (
        node((0, 0)),
        node((3, 4), "B"),
        node((4, 4), "B"),
        node((4, 5), "B", 2),
    )
    record(model, a, "ACTION1", b)
    record(model, b, "ACTION4", c)
    record(model, c, "ACTION2", d)
    route = model.search(a, [d], resource=10, capacity=20, default_cost=2)

    assert route is not None
    assert [step.action for step in route.steps] == ["ACTION1", "ACTION4", "ACTION2"]
    assert all(step.kind == "observed" for step in route.steps)
    assert route.resource_remaining == 4
    # The edge from B is not a valid action from A even at the same position.
    assert (
        model.search(node((3, 4)), [d], resource=20, capacity=20, default_cost=2)
        is None
    )


def test_conflicting_exact_edge_is_retained_but_not_routed():
    model = JointNavigationModel()
    a, b, c = node((0, 0)), node((4, 4)), node((5, 5))
    record(model, a, "ACTION4", b)
    record(model, a, "ACTION4", c)
    record(model, a, "ACTION4", c)

    assert model.counterexamples == 1
    assert model.edges[(a, "ACTION4")].outcomes == {b: 1, c: 2}
    assert (
        model.search(
            a,
            [c],
            resource=20,
            capacity=20,
            default_cost=2,
            passable=[(0, 0), (5, 5)],
            allow_predictions=True,
        )
        is None
    )


def test_reset_is_never_an_action_model_observation():
    model = JointNavigationModel()
    a, b = node((0, 0)), node((5, 5))
    model.record(a, "ACTION1", b, resource_before=0, resource_after=20, reset=True)
    model.record(a, "RESET", b, resource_before=0, resource_after=20)
    assert not model.edges


def test_resource_constraints_require_refill_before_goal():
    model = JointNavigationModel()
    a, b, c, d = [node((x, 0)) for x in range(4)]
    record(model, a, "ACTION4", b)
    record(model, b, "ACTION4", c)
    record(model, a, "ACTION1", d, resource_before=2, resource_after=20)
    record(model, d, "ACTION2", b)
    route = model.search(a, [c], resource=2, capacity=20, default_cost=2)

    assert route is not None
    assert [step.action for step in route.steps] == ["ACTION1", "ACTION2", "ACTION4"]
    assert route.resource_remaining == 16
    assert model.search(a, [c], resource=1, capacity=20, default_cost=2) is None


def test_composition_is_a_prediction_and_exact_counterexample_overrides_it():
    model = JointNavigationModel()
    record(model, node((0, 0), "A", 1), "ACTION4", node((1, 0), "A", 2))
    start, goal = node((0, 0), "B", 1), node((1, 0), "B", 2)
    assert model.search(start, [goal], resource=20, capacity=20, default_cost=2) is None
    route = model.search(
        start, [goal], resource=20, capacity=20, default_cost=2, allow_predictions=True
    )
    assert route is not None and route.steps[0].kind == "projected"

    record(model, start, "ACTION4", node((2, 2), "C", 3))
    assert (
        model.search(
            start,
            [goal],
            resource=20,
            capacity=20,
            default_cost=2,
            allow_predictions=True,
        )
        is None
    )


def test_unknown_geometry_is_explicit_and_expansion_is_bounded():
    model = JointNavigationModel()
    start, goal = node((0, 0)), node((1, 0))
    route = model.search(
        start,
        [goal],
        resource=4,
        capacity=20,
        default_cost=2,
        passable=[(0, 0), (1, 0)],
        allow_predictions=True,
    )
    assert route is not None and route.steps[0].kind == "geometric"
    assert (
        model.search(
            start,
            [goal],
            resource=4,
            capacity=20,
            default_cost=2,
            passable=[(0, 0), (1, 0)],
            allow_predictions=True,
            max_expansions=1,
        )
        is None
    )
    assert model.last_search["status"] == "expansion_budget"


def test_shared_prediction_reserves_largest_observed_resource_cost():
    model = JointNavigationModel()
    record(
        model,
        node((0, 0), color=1),
        "ACTION4",
        node((1, 0), color=1),
        resource_before=20,
        resource_after=18,
    )
    record(
        model,
        node((0, 0), color=2),
        "ACTION4",
        node((1, 0), color=2),
        resource_before=20,
        resource_after=16,
    )
    start, goal = node((0, 0), color=3), node((1, 0), color=3)
    assert (
        model.search(
            start,
            [goal],
            resource=3,
            capacity=20,
            default_cost=2,
            allow_predictions=True,
        )
        is None
    )
    route = model.search(
        start, [goal], resource=6, capacity=20, default_cost=2, allow_predictions=True
    )
    assert route is not None and route.resource_remaining == 2


def phase(previous, current):
    return (MotionObservation((2, 3, "tiny"), previous, current),)


def test_opposite_motion_phases_do_not_conflict_or_share_status_effects():
    model = JointNavigationModel()
    right = phase((1, 7), (2, 7))
    left = phase((3, 7), (2, 7))
    a = NavigationNode((1, 8), JointMechanismState("A", 1), right)
    b = NavigationNode((1, 8), a.status, left)
    stays = NavigationNode((1, 7), a.status, phase((2, 7), (3, 7)))
    changes = NavigationNode((1, 7), JointMechanismState("B", 1), phase((2, 7), None))
    record(model, a, "ACTION1", stays)
    record(model, b, "ACTION1", changes)
    assert model.counterexamples == 0
    assert (
        model.search(a, [node((1, 7), "B")], resource=20, capacity=20, default_cost=2)
        is None
    )
    route = model.search(
        b, [node((1, 7), "B")], resource=20, capacity=20, default_cost=2
    )
    assert route is not None and route.steps[0].destination == changes


def test_phase_composition_preserves_observed_successor_context():
    model = JointNavigationModel()
    context, successor = phase((1, 7), (2, 7)), phase((2, 7), (3, 7))
    record(
        model,
        NavigationNode((0, 0), JointMechanismState("A", 1), context),
        "ACTION4",
        NavigationNode((1, 0), JointMechanismState("A", 2), successor),
    )
    start = NavigationNode((0, 0), JointMechanismState("B", 1), context)
    route = model.search(
        start,
        [node((1, 0), "B", 2)],
        resource=20,
        capacity=20,
        default_cost=2,
        allow_predictions=True,
    )
    assert route is not None and route.steps[0].kind == "projected"
    assert route.steps[0].destination.context == successor
    wrong_phase = NavigationNode(start.position, start.status, phase((3, 7), (2, 7)))
    assert (
        model.search(
            wrong_phase,
            [node((1, 0), "B", 2)],
            resource=20,
            capacity=20,
            default_cost=2,
            allow_predictions=True,
        )
        is None
    )


def test_geometric_prediction_cannot_freeze_or_majority_vote_a_motion_phase():
    model = JointNavigationModel()
    context, successor = phase((1, 7), (2, 7)), phase((2, 7), (3, 7))
    start = NavigationNode((0, 0), JointMechanismState("A", 1), context)
    args = dict(
        resource=20,
        capacity=20,
        default_cost=2,
        passable=[(0, 0), (1, 0)],
        allow_predictions=True,
    )
    assert model.search(start, [node((1, 0))], **args) is None
    other = NavigationNode((5, 5), start.status, context)
    record(model, other, "ACTION4", NavigationNode((6, 5), start.status, successor))
    route = model.search(start, [node((1, 0))], **args)
    assert route is not None and route.steps[0].destination.context == successor
    record(
        model,
        other,
        "ACTION4",
        NavigationNode((6, 5), start.status, phase((2, 7), None)),
    )
    assert model.search(start, [node((1, 0))], **args) is None
