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
from heapq import heappop, heappush
from typing import Callable, Iterable

from .mechanism_state_tracker import ACTION_DELTAS, Cell


@dataclass(frozen=True, order=True)
class JointMechanismState:
    """Decision-relevant status shared by mechanisms and completion targets."""

    pattern: str
    color: int | None = None

    def matches(self, required: "JointMechanismState") -> bool:
        return self.pattern == required.pattern and (
            required.color is None or self.color is None or self.color == required.color
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
                    (-1 if item.destination.color is None else item.destination.color),
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


class EvidenceKind(str, Enum):
    """How strongly a mechanism option is supported by observations."""

    OBSERVED = "observed"
    PROJECTED = "projected"
    DYNAMIC = "dynamic"


DynamicPhase = tuple[Cell, Cell]


@dataclass(frozen=True)
class ReachabilityState:
    """State at an action boundary; resource is measured in action units."""

    player: Cell | None
    joint_status: JointMechanismState
    phase: DynamicPhase | None = None
    resource_steps: int = 0


@dataclass(frozen=True)
class ReachabilityOption:
    """One *atomic action*, including navigation, with an explicit contract.

    Approach is the exact pre-action position, not a cell we may magically
    route to. Every action updates phase, including ordinary walking and
    blocked inputs. A pair of ``None`` phases means there is no tracked
    dynamic object, not permission to freeze an existing one. Refill values
    are post-action resource levels, not a zero-cost bonus.
    """

    option_id: str
    source: JointMechanismState
    destination: JointMechanismState
    evidence: EvidenceKind = EvidenceKind.OBSERVED
    actuator: Cell | None = None
    approach: Cell | None = None
    contact_action: str | None = None
    observed_result: Cell | None = None
    source_phase: DynamicPhase | None = None
    destination_phase: DynamicPhase | None = None
    contact_cost: int = 1
    projected: bool = False
    refill_to: int | None = None
    required_resource_steps: int | None = None

    @property
    def has_contract(self) -> bool:
        return (
            self.approach is not None
            and self.contact_action in ACTION_DELTAS
            and self.observed_result is not None
            and self.contact_cost > 0
            and (self.refill_to is None or self.refill_to >= 0)
        )

    @property
    def actionable(self) -> bool:
        return (
            self.has_contract
            and not self.projected
            and self.evidence is not EvidenceKind.PROJECTED
            and (
                self.evidence is not EvidenceKind.DYNAMIC
                or (
                    self.source_phase is not None and self.destination_phase is not None
                )
            )
        )


@dataclass(frozen=True)
class ReachabilityStep:
    option: ReachabilityOption
    from_state: ReachabilityState
    to_state: ReachabilityState


@dataclass(frozen=True)
class ReachabilityResult:
    """Auditable result instead of a bare path or an optimistic ``None``."""

    path: tuple[ReachabilityStep, ...] | None
    expanded_states: int
    rejected_missing_contract: int = 0
    rejected_phase: int = 0
    rejected_resource: int = 0
    projected_steps: int = 0
    reason: str | None = None
    rejected_conflicts: int = 0
    rejected_projection: int = 0

    @property
    def actionable(self) -> bool:
        return bool(
            self.path is not None
            and not self.projected_steps
            and all(step.option.actionable for step in self.path)
        )


class StateReachabilityPlanner:
    """Bounded action-level search, with no implicit routing or phase freeze.

    A returned path is reachability under the supplied model, not proof of
    game completion or causality. Conflicting observations abstain; exact
    facts override projections. Resource/refill and final reserve are hard
    constraints. Projected paths are diagnostic and never actionable.
    """

    def __init__(
        self,
        options: Iterable[ReachabilityOption],
    ) -> None:
        self.options = tuple(options)
        self.by_status: dict[JointMechanismState, list[ReachabilityOption]] = {}
        self.observed_outcomes: dict[tuple, set[tuple]] = {}
        self.maximum_costs: dict[tuple, int] = {}
        for option in self.options:
            self.by_status.setdefault(option.source, []).append(option)
            if not option.projected and option.evidence is not EvidenceKind.PROJECTED:
                key = self._contract_key(option)
                self.observed_outcomes.setdefault(key, set()).add(self._outcome(option))
                self.maximum_costs[key] = max(
                    self.maximum_costs.get(key, 0), option.contact_cost
                )

    @staticmethod
    def _contract_key(option: ReachabilityOption) -> tuple:
        return (
            option.source,
            option.approach,
            option.source_phase,
            option.contact_action,
            option.required_resource_steps,
        )

    @staticmethod
    def _outcome(option: ReachabilityOption) -> tuple:
        return (
            option.destination,
            option.observed_result,
            option.destination_phase,
            option.refill_to,
        )

    @staticmethod
    def _key(state: ReachabilityState) -> tuple[object, ...]:
        return (
            state.player,
            state.joint_status,
            state.phase,
            state.resource_steps,
        )

    @staticmethod
    def _reaches_goal(state: ReachabilityState, goal: JointMechanismState) -> bool:
        # An unknown current colour cannot satisfy a colour-specific goal.
        # This is stricter than ``JointMechanismState.matches`` because that
        # helper is intentionally tolerant for partially observed online
        # plans.
        return state.joint_status.pattern == goal.pattern and (
            goal.color is None or state.joint_status.color == goal.color
        )

    def search(
        self,
        start: ReachabilityState,
        goal: JointMechanismState,
        *,
        allow_projected: bool = False,
        require_actionable: bool = True,
        goal_cell: Cell | None = None,
        reserve_steps: int = 0,
        capacity_steps: int | None = None,
        max_expansions: int = 10000,
    ) -> ReachabilityResult:
        """Return the fewest actions aligning status (and optionally position).

        Reaching a displayed target cell is still not a WIN; online execution
        must check the game's real level-advance feedback. The reserve is
        required at the goal, allowing a low-resource intermediate action
        to reach a refill when the action's own cost remains payable.
        """

        capacity = start.resource_steps if capacity_steps is None else capacity_steps
        if (
            start.resource_steps < 0
            or capacity < start.resource_steps
            or reserve_steps < 0
        ):
            raise ValueError("invalid resource bounds")

        queue: list[tuple[int, int, ReachabilityState]] = [(0, 0, start)]
        previous: dict[
            tuple[object, ...], tuple[tuple[object, ...], ReachabilityStep] | None
        ] = {self._key(start): None}
        distance: dict[tuple[object, ...], int] = {self._key(start): 0}
        serial = 0
        rejected_missing_contract = 0
        rejected_phase = 0
        rejected_resource = 0
        rejected_conflicts = 0
        rejected_projection = 0
        expanded = 0
        reached: tuple[object, ...] | None = None

        while queue and expanded < max_expansions:
            cost, _, current = heappop(queue)
            current_key = self._key(current)
            if cost != distance.get(current_key):
                continue
            expanded += 1
            if (
                self._reaches_goal(current, goal)
                and (goal_cell is None or current.player == goal_cell)
                and current.resource_steps >= reserve_steps
            ):
                reached = current_key
                break
            for option in self.by_status.get(current.joint_status, ()):
                if option.evidence is EvidenceKind.PROJECTED or option.projected:
                    if not allow_projected or require_actionable:
                        rejected_projection += 1
                        continue
                if not option.has_contract:
                    rejected_missing_contract += 1
                    continue
                if current.player != option.approach:
                    continue
                if (
                    option.required_resource_steps is not None
                    and option.required_resource_steps != current.resource_steps
                ):
                    rejected_resource += 1
                    continue
                if (
                    current.phase != option.source_phase
                    or (current.phase is not None and option.destination_phase is None)
                    or (
                        option.evidence is EvidenceKind.DYNAMIC
                        and option.source_phase is None
                    )
                ):
                    rejected_phase += 1
                    continue
                base = self._contract_key(option)[:-1]
                keys = (base + (None,), base + (current.resource_steps,))
                outcomes = set().union(
                    *(self.observed_outcomes.get(key, set()) for key in keys)
                )
                if len(outcomes) > 1 or (
                    outcomes and self._outcome(option) not in outcomes
                ):
                    rejected_conflicts += 1
                    continue
                action_cost = max(
                    option.contact_cost,
                    *(self.maximum_costs.get(key, 0) for key in keys),
                )
                if current.resource_steps < action_cost:
                    rejected_resource += 1
                    continue
                remaining = current.resource_steps - action_cost
                if option.refill_to is not None:
                    remaining = min(capacity, option.refill_to)
                next_state = ReachabilityState(
                    player=option.observed_result,
                    joint_status=option.destination,
                    phase=option.destination_phase,
                    resource_steps=remaining,
                )
                next_key = self._key(next_state)
                next_cost = cost + 1
                if next_cost >= distance.get(next_key, 10**9):
                    continue
                distance[next_key] = next_cost
                previous[next_key] = (
                    current_key,
                    ReachabilityStep(option, current, next_state),
                )
                serial += 1
                heappush(queue, (next_cost, serial, next_state))

        if reached is None:
            # Rejection counters are not a proof of *the* causal blocker:
            # several independent missing edges may exist. Report them all
            # rather than arbitrarily assigning one explanation.
            reason = "expansion_budget" if queue else "no_supported_path"
            return ReachabilityResult(
                None,
                expanded,
                rejected_missing_contract,
                rejected_phase,
                rejected_resource,
                0,
                reason,
                rejected_conflicts,
                rejected_projection,
            )

        path: list[ReachabilityStep] = []
        cursor = reached
        while (step := previous[cursor]) is not None:
            prior, item = step
            path.append(item)
            cursor = prior
        path.reverse()
        used_projected = sum(
            item.option.evidence is EvidenceKind.PROJECTED or item.option.projected
            for item in path
        )
        return ReachabilityResult(
            tuple(path),
            expanded,
            rejected_missing_contract,
            rejected_phase,
            rejected_resource,
            used_projected,
            None,
            rejected_conflicts,
            rejected_projection,
        )
