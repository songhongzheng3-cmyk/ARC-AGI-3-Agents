"""Observation-derived navigation over position, status, motion and resource.

Exact observations, compositional predictions and unseen geometric edges are
kept separate. Conflicting observations disable an edge rather than replacing
one deterministic prediction with another. No successful trajectories or game
identifiers are part of the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Iterable

from .mechanism_state_tracker import ACTION_DELTAS, Cell
from .motion_context import MotionContext
from .task_planning import JointMechanismState


@dataclass(frozen=True)
class NavigationNode:
    position: Cell
    status: JointMechanismState
    context: MotionContext = ()


@dataclass
class NavigationEvidence:
    outcomes: dict[NavigationNode, int] = field(default_factory=dict)
    resource_cost: int = 0
    refill_to: int | None = None

    @property
    def destination(self) -> NavigationNode | None:
        return next(iter(self.outcomes)) if len(self.outcomes) == 1 else None


@dataclass(frozen=True)
class NavigationStep:
    source: NavigationNode
    action: str
    destination: NavigationNode
    kind: str
    resource_cost: int
    refill_to: int | None = None


@dataclass(frozen=True)
class JointRoute:
    steps: tuple[NavigationStep, ...]
    resource_remaining: int
    expanded_nodes: int


class JointNavigationModel:
    """Learn directed joint transitions and search a resource-feasible route."""

    def __init__(self) -> None:
        self.edges: dict[tuple[NavigationNode, str], NavigationEvidence] = {}
        self.by_position: dict[tuple[Cell, str], set[NavigationNode]] = {}
        self.counterexamples = 0
        self.last_search: dict[str, int | str] = {}
        self.context_transitions: dict[
            tuple[MotionContext, str], set[MotionContext]
        ] = {}

    def record(
        self,
        source: NavigationNode,
        action: str,
        destination: NavigationNode,
        *,
        resource_before: int,
        resource_after: int,
        reset: bool = False,
    ) -> None:
        if reset or action not in ACTION_DELTAS:
            return
        key = (source, action)
        evidence = self.edges.setdefault(key, NavigationEvidence())
        if evidence.outcomes and destination not in evidence.outcomes:
            self.counterexamples += 1
        evidence.outcomes[destination] = evidence.outcomes.get(destination, 0) + 1
        evidence.resource_cost = max(
            evidence.resource_cost, resource_before - resource_after, 0
        )
        if resource_after > resource_before:
            evidence.refill_to = (
                resource_after
                if evidence.refill_to is None
                else min(evidence.refill_to, resource_after)
            )
        self.by_position.setdefault((source.position, action), set()).add(source)
        self.context_transitions.setdefault((source.context, action), set()).add(
            destination.context
        )

    def _predict(
        self,
        node: NavigationNode,
        action: str,
        *,
        passable: set[Cell],
        default_cost: int,
        allow_predictions: bool,
    ) -> NavigationStep | None:
        exact = self.edges.get((node, action))
        if exact is not None:
            destination = exact.destination
            if destination is None:
                return None
            return NavigationStep(
                node,
                action,
                destination,
                "observed",
                max(default_cost, exact.resource_cost),
                exact.refill_to,
            )
        if not allow_predictions:
            return None
        sources = self.by_position.get((node.position, action), set())
        predictions: dict[tuple[NavigationNode, int | None], NavigationStep] = {}
        for source in sources:
            if source.context != node.context:
                continue
            evidence = self.edges[(source, action)]
            destination = evidence.destination
            if destination is None:
                return None
            before, after = source.status, destination.status
            projected: JointMechanismState | None = None
            if before == after:
                projected = node.status
            elif before.pattern == after.pattern and before.color == node.status.color:
                projected = JointMechanismState(node.status.pattern, after.color)
            elif before.color == after.color and before.pattern == node.status.pattern:
                projected = JointMechanismState(after.pattern, node.status.color)
            if projected is None:
                continue
            successor = NavigationNode(
                destination.position, projected, destination.context
            )
            prediction_key = (successor, evidence.refill_to)
            prior = predictions.get(prediction_key)
            predictions[prediction_key] = NavigationStep(
                node,
                action,
                successor,
                "projected",
                max(
                    default_cost,
                    evidence.resource_cost,
                    0 if prior is None else prior.resource_cost,
                ),
                evidence.refill_to,
            )
        if len(predictions) == 1:
            return next(iter(predictions.values()))
        if sources:
            # Known conflicting/context-dependent edges must never silently
            # fall back to an adjacent-cell guess.
            return None
        dx, dy = ACTION_DELTAS[action]
        neighbor = (node.position[0] + dx, node.position[1] + dy)
        if neighbor not in passable:
            return None
        context = node.context
        if context:
            successors = self.context_transitions.get((context, action), set())
            if len(successors) != 1:
                # Do not freeze an autonomous object while the player moves,
                # or choose a majority phase through contact/occlusion.
                return None
            context = next(iter(successors))
        return NavigationStep(
            node,
            action,
            NavigationNode(neighbor, node.status, context),
            "geometric",
            default_cost,
        )

    def search(
        self,
        start: NavigationNode,
        goals: Iterable[NavigationNode],
        *,
        resource: int,
        capacity: int,
        default_cost: int,
        passable: Iterable[Cell] = (),
        actions: Iterable[str] = tuple(ACTION_DELTAS),
        allow_predictions: bool = False,
        max_expansions: int = 4000,
    ) -> JointRoute | None:
        """Search joint nodes with Pareto labels for action cost and resource.

        Ordinary observed edges cost one; projected/geometric edges cost four
        in the search objective but still consume their predicted resource.
        Returned steps are predictions, not a commitment to replay all actions.
        A goal without motion context accepts any arrival phase; a contextual
        frontier goal requires that exact phase.
        """
        goals = frozenset(goals)
        any_phase_goals = {
            (goal.position, goal.status) for goal in goals if not goal.context
        }
        passable = set(passable)
        names = tuple(name for name in actions if name in ACTION_DELTAS)
        default_cost = max(1, default_cost)
        resource = max(0, min(resource, capacity))
        start_label = (start, resource)
        queue = [(0, 0, start_label)]
        best = {start_label: 0}
        labels: dict[NavigationNode, dict[int, int]] = {start: {resource: 0}}
        previous: dict[
            tuple[NavigationNode, int],
            tuple[tuple[NavigationNode, int], NavigationStep] | None,
        ] = {start_label: None}
        sequence = 0
        expanded = 0
        while queue and expanded < max_expansions:
            cost, _, label = heappop(queue)
            node, remaining = label
            if best.get(label) != cost:
                continue
            if labels.get(node, {}).get(remaining) != cost:
                continue
            expanded += 1
            if node in goals or (node.position, node.status) in any_phase_goals:
                steps = []
                cursor = label
                while (link := previous[cursor]) is not None:
                    parent, step = link
                    steps.append(step)
                    cursor = parent
                self.last_search = {"status": "found", "expanded": expanded}
                return JointRoute(tuple(reversed(steps)), remaining, expanded)
            for action in names:
                step = self._predict(
                    node,
                    action,
                    passable=passable,
                    default_cost=default_cost,
                    allow_predictions=allow_predictions,
                )
                if step is None or remaining < step.resource_cost:
                    continue
                after = remaining - step.resource_cost
                if step.refill_to is not None:
                    after = min(capacity, step.refill_to)
                successor = step.destination
                next_cost = cost + (1 if step.kind == "observed" else 4)
                existing = labels.setdefault(successor, {})
                if any(r >= after and c <= next_cost for r, c in existing.items()):
                    continue
                for r, c in tuple(existing.items()):
                    if r <= after and c >= next_cost:
                        del existing[r]
                existing[after] = next_cost
                next_label = (successor, after)
                best[next_label] = next_cost
                previous[next_label] = (label, step)
                sequence += 1
                heappush(queue, (next_cost, sequence, next_label))
        self.last_search = {
            "status": "expansion_budget" if queue else "no_path",
            "expanded": expanded,
        }
        return None
