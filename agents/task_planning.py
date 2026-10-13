"""Persistent task plans and learned joint mechanism transitions.

The classes in this module deliberately do not parse pixels or choose grid
routes.  They provide the protocol that sits between an observation-derived
world model and a domain policy:

* ``JointMechanismGraph`` stores causal actuator transitions.
* ``TaskPlan`` keeps a multi-stage commitment across observations.
* ``TaskPlanController`` advances or invalidates that commitment only when
  observed evidence warrants it.

Keeping this layer independent from ``MechanismPolicy`` makes it reusable by
future game-specific world models without importing ls20 coordinates or
palette semantics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable

from .mechanism_state_tracker import Cell


@dataclass(frozen=True, order=True)
class JointMechanismState:
    """Decision-relevant status shared by mechanisms and completion targets."""

    pattern: str
    color: int | None = None

    def matches(self, required: "JointMechanismState") -> bool:
        return self.pattern == required.pattern and (
            required.color is None
            or self.color is None
            or self.color == required.color
        )


@dataclass(frozen=True)
class JointMechanismTransition:
    source: JointMechanismState
    actuator: Cell
    destination: JointMechanismState
    observations: int = 1
    approach: Cell | None = None
    contact_action: str | None = None
    observed_result: Cell | None = None


class JointMechanismGraph:
    """A small causal graph learned only from observed actuator effects."""

    def __init__(self) -> None:
        self._destinations: dict[
            tuple[JointMechanismState, Cell], JointMechanismState
        ] = {}
        self._observation_counts: dict[
            tuple[JointMechanismState, Cell, JointMechanismState], int
        ] = {}
        self._interaction_contracts: dict[
            tuple[JointMechanismState, Cell, JointMechanismState],
            tuple[Cell | None, str | None, Cell | None],
        ] = {}

    def clear(self) -> None:
        self._destinations.clear()
        self._observation_counts.clear()
        self._interaction_contracts.clear()

    def record(
        self,
        source: JointMechanismState,
        actuator: Cell,
        destination: JointMechanismState,
        *,
        approach: Cell | None = None,
        contact_action: str | None = None,
        observed_result: Cell | None = None,
    ) -> None:
        """Record one real before/contact/after transition.

        A later contradictory observation replaces the currently usable edge,
        while both observations remain counted.  The policy can therefore
        replan immediately without pretending that an older deterministic
        model is still authoritative.
        """

        if source == destination:
            return
        self._destinations[(source, actuator)] = destination
        key = (source, actuator, destination)
        self._observation_counts[key] = self._observation_counts.get(key, 0) + 1
        if approach is not None and contact_action is not None:
            self._interaction_contracts[key] = (
                approach,
                contact_action,
                observed_result,
            )

    def transitions(self) -> tuple[JointMechanismTransition, ...]:
        result = []
        for (source, actuator), destination in self._destinations.items():
            approach, contact_action, observed_result = self._interaction_contracts.get(
                (source, actuator, destination),
                (None, None, None),
            )
            result.append(
                JointMechanismTransition(
                    source=source,
                    actuator=actuator,
                    destination=destination,
                    observations=self._observation_counts.get(
                        (source, actuator, destination), 1
                    ),
                    approach=approach,
                    contact_action=contact_action,
                    observed_result=observed_result,
                )
            )
        return tuple(
            sorted(
                result,
                key=lambda item: (
                    item.source.pattern,
                    -1 if item.source.color is None else item.source.color,
                    item.actuator,
                    item.destination.pattern,
                    (
                        -1
                        if item.destination.color is None
                        else item.destination.color
                    ),
                ),
            )
        )

    def shortest_path(
        self,
        start: JointMechanismState,
        goal: JointMechanismState,
        *,
        allow_projection: bool = True,
    ) -> tuple[JointMechanismTransition, ...] | None:
        """Return the fewest supported actuator contacts reaching ``goal``.

        A transition that changed only color is projected across the current
        pattern; a transition that changed only pattern is projected across
        the current color.  This is the minimum compositional rule justified
        by an observed single-dimension effect.  Transitions that changed both
        dimensions remain exact-context edges.
        """

        if start.matches(goal):
            return ()
        queue = deque([start])
        previous: dict[
            JointMechanismState,
            tuple[JointMechanismState, JointMechanismTransition] | None,
        ] = {start: None}
        reached: JointMechanismState | None = None
        while queue:
            current = queue.popleft()
            for transition in self._successors(
                current,
                allow_projection=allow_projection,
            ):
                successor = transition.destination
                if successor in previous:
                    continue
                previous[successor] = (current, transition)
                if successor.matches(goal):
                    reached = successor
                    queue.clear()
                    break
                queue.append(successor)
        if reached is None:
            return None

        path: list[JointMechanismTransition] = []
        current = reached
        while (step := previous[current]) is not None:
            prior, transition = step
            path.append(transition)
            current = prior
        path.reverse()
        return tuple(path)

    def _successors(
        self,
        current: JointMechanismState,
        *,
        allow_projection: bool,
    ) -> tuple[JointMechanismTransition, ...]:
        """Instantiate exact and independently composable observed effects."""

        result: list[JointMechanismTransition] = []
        seen: set[tuple[Cell, JointMechanismState]] = set()
        for observed in self.transitions():
            projected: JointMechanismTransition | None = None
            if observed.source == current:
                projected = observed
            elif allow_projection and (
                observed.source.pattern == observed.destination.pattern
                and current.color == observed.source.color
            ):
                # A palette-only actuator preserves whatever pattern is
                # currently active.
                projected = JointMechanismTransition(
                    source=current,
                    actuator=observed.actuator,
                    destination=JointMechanismState(
                        current.pattern,
                        observed.destination.color,
                    ),
                    observations=observed.observations,
                    approach=observed.approach,
                    contact_action=observed.contact_action,
                    observed_result=observed.observed_result,
                )
            elif allow_projection and (
                observed.source.color == observed.destination.color
                and current.pattern == observed.source.pattern
            ):
                # A pattern-only actuator preserves whatever palette is
                # currently active.
                projected = JointMechanismTransition(
                    source=current,
                    actuator=observed.actuator,
                    destination=JointMechanismState(
                        observed.destination.pattern,
                        current.color,
                    ),
                    observations=observed.observations,
                    approach=observed.approach,
                    contact_action=observed.contact_action,
                    observed_result=observed.observed_result,
                )
            if projected is None or projected.destination == current:
                continue
            key = (projected.actuator, projected.destination)
            if key in seen:
                continue
            seen.add(key)
            result.append(projected)
        return tuple(
            sorted(
                result,
                key=lambda item: (
                    item.actuator,
                    item.destination.pattern,
                    -1 if item.destination.color is None else item.destination.color,
                ),
            )
        )


class PlanStageKind(str, Enum):
    REFILL = "refill"
    APPLY_ACTUATOR = "apply_actuator"
    CONTACT_TARGET = "contact_target"


@dataclass(frozen=True)
class PlanStage:
    kind: PlanStageKind
    target: Cell
    expected_before: JointMechanismState | None = None
    expected_after: JointMechanismState | None = None
    estimated_steps: int = 0
    approach: Cell | None = None
    contact_action: str | None = None
    observed_result: Cell | None = None


@dataclass(frozen=True)
class PlanningState:
    level: int
    player: Cell | None
    joint_status: JointMechanismState
    resource_steps: int
    topology_revision: int
    event_types: frozenset[str] = frozenset()


@dataclass
class TaskPlan:
    plan_id: int
    level: int
    goal: JointMechanismState
    stages: tuple[PlanStage, ...]
    topology_revision: int
    current_stage_index: int = 0
    failed_stage_attempts: int = 0
    invalid_reason: str | None = None
    completed: bool = False

    @property
    def current_stage(self) -> PlanStage | None:
        if self.completed or self.invalid_reason is not None:
            return None
        if self.current_stage_index >= len(self.stages):
            return None
        return self.stages[self.current_stage_index]


class TaskPlanController:
    """Own one plan and update it from observed post-action evidence."""

    def __init__(self, *, maximum_stage_attempts: int = 2) -> None:
        self.maximum_stage_attempts = maximum_stage_attempts
        self.active_plan: TaskPlan | None = None
        self._next_plan_id = 1
        self.last_invalid_reason: str | None = None
        self.replan_requested = False

    def reset(self) -> None:
        self.active_plan = None
        self.last_invalid_reason = None
        self.replan_requested = False

    def install(
        self,
        *,
        level: int,
        goal: JointMechanismState,
        stages: Iterable[PlanStage],
        topology_revision: int,
    ) -> TaskPlan:
        plan = TaskPlan(
            plan_id=self._next_plan_id,
            level=level,
            goal=goal,
            stages=tuple(stages),
            topology_revision=topology_revision,
        )
        self._next_plan_id += 1
        self.active_plan = plan
        self.last_invalid_reason = None
        return plan

    def invalidate(self, reason: str) -> None:
        plan = self.active_plan
        if plan is not None:
            plan.invalid_reason = reason
        self.last_invalid_reason = reason
        self.active_plan = None

    def consume_replan_request(self) -> bool:
        requested = self.replan_requested
        self.replan_requested = False
        return requested

    def record_failed_attempt(self) -> bool:
        """Return ``True`` when the active stage exceeded its retry budget."""

        plan = self.active_plan
        if plan is None:
            return False
        plan.failed_stage_attempts += 1
        if plan.failed_stage_attempts >= self.maximum_stage_attempts:
            self.invalidate("stage_effect_not_observed")
            return True
        return False

    def observe(
        self,
        state: PlanningState,
        *,
        target_available: Callable[[Cell], bool],
    ) -> TaskPlan | None:
        """Advance completed stages or invalidate contradicted assumptions."""

        plan = self.active_plan
        if plan is None:
            return None
        if state.level != plan.level:
            plan.completed = state.level > plan.level
            self.active_plan = None
            return None
        if state.topology_revision != plan.topology_revision:
            # A newly discovered teleport/relocation edge refines navigation
            # rather than invalidating the mission.  The policy recomputes the
            # route at every decision, so keep the same stage sequence and let
            # it benefit from the richer topology on the next attempt.
            plan.topology_revision = state.topology_revision

        while (stage := plan.current_stage) is not None:
            if stage.kind is PlanStageKind.REFILL:
                if (
                    "resource_increased" not in state.event_types
                    or state.player != stage.target
                ):
                    break
            elif stage.kind is PlanStageKind.APPLY_ACTUATOR:
                expected = stage.expected_after
                if expected is None or not state.joint_status.matches(expected):
                    before = stage.expected_before
                    if before is not None and not state.joint_status.matches(before):
                        # The status drifted (usually the other dimension was
                        # changed by local navigation before the actuator was
                        # contacted).  That is a reason to recompute the plan
                        # from the current state, not a reason to abandon the
                        # whole mission.  Keep the controller's active plan so
                        # the policy can rebuild the stage sequence in place.
                        self.replan_requested = True
                    break
            elif stage.kind is PlanStageKind.CONTACT_TARGET:
                if "level_advanced" not in state.event_types:
                    if not state.joint_status.matches(plan.goal):
                        self.invalidate("goal_invariant_lost")
                        return None
                    if not target_available(stage.target):
                        # A target may be temporarily occluded by the player.
                        # Keep the plan only while the player is on that cell.
                        if state.player != stage.target:
                            self.invalidate("completion_target_lost")
                            return None
                    break

            plan.current_stage_index += 1
            plan.failed_stage_attempts = 0

        if plan.current_stage is None:
            plan.completed = True
            self.active_plan = None
            return None
        return plan

    def summary(self) -> str:
        plan = self.active_plan
        if plan is None:
            return f"task_plan=none last_invalid={self.last_invalid_reason}"
        stage = plan.current_stage
        return (
            f"task_plan={plan.plan_id} "
            f"stage={None if stage is None else stage.kind.value} "
            f"stage_index={plan.current_stage_index}/{len(plan.stages)} "
            f"attempts={plan.failed_stage_attempts}"
        )
