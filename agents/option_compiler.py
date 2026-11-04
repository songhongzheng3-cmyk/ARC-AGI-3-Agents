"""Compile observed mechanisms into falsifiable, phase-bound action paths.

No game identities, coordinates, hidden implementations or successful action
sequences are used. Cross-context navigation and independent status effects
are explicit hypotheses; execution must validate each predicted next frame.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from typing import Iterable

from .dynamic_contact import DynamicContactModel, Phase, TrackedPhase
from .mechanism_state_tracker import ACTION_DELTAS, Cell, MechanismState, pattern_key
from .motion_context import ObjectSignature
from .task_planning import JointMechanismState, JointMechanismTransition


def _pattern(value: str) -> tuple[tuple[int, ...], ...]:
    return (
        ()
        if value == "empty"
        else tuple(tuple(map(int, row)) for row in value.split("/"))
    )


def _status(state: MechanismState) -> JointMechanismState:
    return JointMechanismState(state.status_signature, state.status_color)


@dataclass(frozen=True)
class CompilerState:
    player: Cell
    status: JointMechanismState
    phase: Phase
    resource: int
    refills: frozenset[Cell] = frozenset()

    @property
    def node(self) -> tuple:
        return self.player, self.status, self.phase, self.refills


@dataclass(frozen=True)
class CompiledAction:
    before: CompilerState
    action: str
    after: CompilerState
    hypotheses: tuple[str, ...]
    support: int


@dataclass(frozen=True)
class CompiledPlan:
    steps: tuple[CompiledAction, ...] | None
    expanded: int
    reason: str | None


@dataclass(frozen=True)
class _Sample:
    position: Cell
    status: JointMechanismState
    phase: Phase
    destination: Cell
    after_status: JointMechanismState
    after_phase: Phase
    refill_visible: bool
    phase_observed: bool


class OptionCompiler:
    """A frozen compositional model and resource-aware A* planner.

    Observed position outcomes take precedence over explicitly labelled floor
    adjacency projections. The strongest available context is used; contradictions
    abstain instead of falling back to a weaker context. Refill inventory is
    finite within a plan. That is a conservative single-use hypothesis, not a
    claim that every game has consumable refills.
    """

    def __init__(
        self,
        dynamic: DynamicContactModel,
        signature: ObjectSignature,
        transitions: Iterable[JointMechanismTransition],
        *,
        resource_cost: int = 4,
    ) -> None:
        self.dynamic = deepcopy(dynamic)
        self.signature = signature
        self.transitions = tuple(transitions)
        self.resource_cost = max(1, resource_cost)
        self.samples: dict[tuple[Cell, str], list[_Sample]] = defaultdict(list)
        self.costs: dict[tuple[Cell, str], int] = defaultdict(lambda: 1)
        self.refill_signatures: dict[Cell, set[ObjectSignature]] = defaultdict(set)
        self.refill_values: dict[Cell, int] = {}
        self.walkable: set[Cell] = set()
        self.projection_cells: set[Cell] = set()
        self.blocked: set[Cell] = set()
        self.special_edges: set[tuple[Cell, str]] = set()
        self.metrics: Counter[str] = Counter()
        self._core_cache: dict[tuple, tuple | None] = {}

    @staticmethod
    def _objects(state: MechanismState, cell: Cell) -> set[ObjectSignature]:
        return {
            (o.color, o.area, o.shape_signature)
            for o in state.objects
            if o.cell == cell and o.object_id not in state.player_component_ids
        }

    def observe(
        self,
        before: MechanismState,
        action: str,
        after: MechanismState,
        source_phase: Phase | None,
        destination_phase: Phase | None,
        *,
        phase_observed: bool = True,
    ) -> None:
        """Collect a real action; callers must not supply inferred positions."""
        if (
            action not in ACTION_DELTAS
            or before.level != after.level
            or before.resource_level <= 0
            or before.player_position is None
            or after.player_position is None
            or min(before.player_confidence, after.player_confidence) < 0.8
        ):
            return
        self.walkable.update(before.walkable_cells)
        self.walkable.update(after.walkable_cells)
        self.projection_cells.update(before.navigation_projection_cells)
        self.projection_cells.update(after.navigation_projection_cells)
        self.blocked.update(before.blocked_cells)
        self.blocked.update(after.blocked_cells)
        self.blocked.difference_update(self.walkable)
        dx, dy = ACTION_DELTAS[action]
        geometric_destination = (
            before.player_position[0] + dx,
            before.player_position[1] + dy,
        )
        if after.player_position != geometric_destination:
            self.special_edges.add((before.player_position, action))
        refill = after.resource_level > before.resource_level
        visible = self._objects(before, after.player_position)
        if refill and visible:
            cell = after.player_position
            self.refill_signatures[cell].update(visible)
            amount = after.resource_level // self.resource_cost
            self.refill_values[cell] = min(self.refill_values.get(cell, amount), amount)
        # Resource/appearance evidence does not require a known motion phase.
        if source_phase is None or destination_phase is None:
            return
        key = (before.player_position, action)
        self.samples[key].append(
            _Sample(
                before.player_position,
                _status(before),
                source_phase,
                after.player_position,
                _status(after),
                destination_phase,
                bool(visible),
                phase_observed,
            )
        )
        if not refill:
            cost = (
                before.resource_level - after.resource_level + self.resource_cost - 1
            ) // self.resource_cost
            self.costs[key] = max(self.costs[key], cost)
        self._core_cache.clear()

    def bind(self, state: MechanismState, phase: Phase | None) -> CompilerState | None:
        if (
            state.player_position is None
            or phase is None
            or state.player_confidence < 0.8
        ):
            return None
        available = frozenset(
            cell
            for cell, signatures in self.refill_signatures.items()
            if self._objects(state, cell) & signatures
        )
        return CompilerState(
            state.player_position,
            _status(state),
            phase,
            state.resource_level // self.resource_cost,
            available,
        )

    def _movement(
        self, state: CompilerState, action: str
    ) -> tuple[Cell, int, str] | None:
        samples = self.samples.get((state.player, action), ())
        tiers = (
            (
                "observed_context",
                [
                    s
                    for s in samples
                    if s.status == state.status and s.phase == state.phase
                ],
            ),
            (
                "navigation_phase_projection",
                [s for s in samples if s.status == state.status],
            ),
            (
                "navigation_status_projection",
                [s for s in samples if s.phase == state.phase],
            ),
            ("navigation_context_projection", list(samples)),
        )
        for kind, matches in tiers:
            if not matches:
                continue
            outputs = {s.destination for s in matches}
            if len(outputs) != 1:
                self.metrics["navigation_conflict"] += 1
                return None
            if kind == "observed_context" and not any(
                s.phase_observed for s in matches
            ):
                kind = "navigation_belief_projection"
            return next(iter(outputs)), len(matches), kind
        if not samples:
            dx, dy = ACTION_DELTAS[action]
            neighbor = (state.player[0] + dx, state.player[1] + dy)
            if (
                neighbor in self.projection_cells
                and neighbor not in self.blocked
                and (state.player, action) not in self.special_edges
            ):
                self.metrics["geometric_navigation_projection"] += 1
                return neighbor, 0, "geometric_navigation_projection"
        self.metrics["navigation_unknown"] += 1
        return None

    def _static_effect(
        self, status: JointMechanismState, player: Cell, action: str, destination: Cell
    ) -> tuple[JointMechanismState, str] | None:
        candidates = [
            e
            for e in self.transitions
            if e.approach == player
            and e.contact_action == action
            and e.observed_result == destination
        ]
        if not candidates:
            if any(e.actuator == destination for e in self.transitions):
                self.metrics["static_approach_unknown"] += 1
                return None
            return status, "status_preservation_projection"
        exact = [e for e in candidates if e.source == status]
        if exact:
            outputs = {e.destination for e in exact}
            return (
                (next(iter(outputs)), "observed_static_effect")
                if len(outputs) == 1
                else None
            )
        outputs = set()
        for edge in candidates:
            if (
                edge.source.pattern == edge.destination.pattern
                and edge.source.color == status.color
            ):
                outputs.add(JointMechanismState(status.pattern, edge.destination.color))
            elif (
                edge.source.color == edge.destination.color
                and edge.source.pattern == status.pattern
            ):
                outputs.add(JointMechanismState(edge.destination.pattern, status.color))
        if len(outputs) == 1:
            return next(iter(outputs)), "static_dimension_projection"
        self.metrics["static_effect_unknown_or_conflicting"] += 1
        return None

    def _core(self, state: CompilerState, action: str) -> tuple | None:
        key = (state.player, state.status, state.phase, action)
        if key in self._core_cache:
            return self._core_cache[key]
        result = self._predict_core(state, action)
        self._core_cache[key] = result
        return result

    def _predict_core(self, state: CompilerState, action: str) -> tuple | None:
        movement = self._movement(state, action)
        if movement is None:
            return None
        destination, support, kind = movement
        moved = destination != state.player
        phase = self.dynamic.next_phase(self.signature, state.phase, moved)
        if phase is None:
            self.metrics["motion_rule_unknown"] += 1
            return None
        effect = self._static_effect(state.status, state.player, action, destination)
        if effect is None:
            return None
        status, effect_kind = effect
        hypotheses = [kind, effect_kind]
        if phase[1] == destination:
            if status != state.status:
                self.metrics["simultaneous_effect_order_unknown"] += 1
                return None
            output = self.dynamic.pattern_after(
                self.signature, _pattern(state.status.pattern), moved
            )
            if output is None or not self.dynamic.preserves_color(
                self.signature, moved
            ):
                self.metrics["contact_effect_unknown"] += 1
                return None
            status = JointMechanismState(pattern_key(output), state.status.color)
            hypotheses.append("dynamic_contact_projection")
        # An actual counterexample in this exact context blocks the composed
        # rule, even if another abstraction would have suggested a path.
        exact = [
            s
            for s in self.samples.get((state.player, action), ())
            if s.status == state.status and s.phase == state.phase and s.phase_observed
        ]
        if any(
            (s.destination, s.after_status, s.after_phase)
            != (destination, status, phase)
            for s in exact
        ):
            self.metrics["composed_prediction_counterexample"] += 1
            return None
        return destination, status, phase, tuple(hypotheses), support

    def predict(self, state: CompilerState, action: str) -> CompiledAction | None:
        core = self._core(state, action)
        if core is None:
            return None
        destination, status, phase, hypotheses, support = core
        cost = self.costs[(state.player, action)]
        if state.resource < cost:
            self.metrics["resource_insufficient"] += 1
            return None
        resource = state.resource - cost
        refills = state.refills
        if destination != state.player and destination in refills:
            resource = self.refill_values[destination]
            refills = refills - {destination}
            hypotheses += ("visible_single_use_refill_projection",)
        after = CompilerState(destination, status, phase, resource, refills)
        return CompiledAction(state, action, after, hypotheses, support)

    def _status_edges(self, status: JointMechanismState) -> set[JointMechanismState]:
        result = set()
        for e in self.transitions:
            if e.source == status:
                result.add(e.destination)
            elif (
                e.source.pattern == e.destination.pattern
                and e.source.color == status.color
            ):
                result.add(JointMechanismState(status.pattern, e.destination.color))
            elif (
                e.source.color == e.destination.color
                and e.source.pattern == status.pattern
            ):
                result.add(JointMechanismState(e.destination.pattern, status.color))
        for moved in (True, False):
            output = self.dynamic.pattern_after(
                self.signature, _pattern(status.pattern), moved
            )
            if output is not None and self.dynamic.preserves_color(
                self.signature, moved
            ):
                result.add(JointMechanismState(pattern_key(output), status.color))
        return result

    def _heuristics(
        self, start: CompilerState, goal: JointMechanismState, cell: Cell
    ) -> tuple[dict, dict]:
        # Relaxed mechanism and navigation graphs give lower bounds;
        # Manhattan distance is invalid in a world with observed relocations.
        reverse: dict[JointMechanismState, set] = defaultdict(set)
        queue = deque([start.status])
        seen = {start.status}
        while queue and len(seen) < 512:
            before = queue.popleft()
            for after in self._status_edges(before):
                reverse[after].add(before)
                if after not in seen:
                    seen.add(after)
                    queue.append(after)
        status_distance = {
            s: 0
            for s in seen
            if s.pattern == goal.pattern
            and (goal.color is None or s.color == goal.color)
        }
        queue = deque(status_distance)
        while queue:
            after = queue.popleft()
            for before in reverse[after]:
                if before not in status_distance:
                    status_distance[before] = status_distance[after] + 1
                    queue.append(before)
        reverse_position: dict[Cell, set[Cell]] = defaultdict(set)
        for samples in self.samples.values():
            for sample in samples:
                reverse_position[sample.destination].add(sample.position)
        for before in self.walkable:
            for dx, dy in ACTION_DELTAS.values():
                after = (before[0] + dx, before[1] + dy)
                if after in self.walkable and after not in self.blocked:
                    reverse_position[after].add(before)
        position_distance = {cell: 0}
        queue = deque([cell])
        while queue:
            after = queue.popleft()
            for before in reverse_position[after]:
                if before not in position_distance:
                    position_distance[before] = position_distance[after] + 1
                    queue.append(before)
        return status_distance, position_distance

    def search(
        self,
        start: CompilerState,
        goal: JointMechanismState,
        goal_cell: Cell,
        *,
        reserve: int = 2,
        max_expansions: int = 50000,
    ) -> CompiledPlan:
        status_distance, position_distance = self._heuristics(start, goal, goal_cell)
        if start.status not in status_distance:
            return CompiledPlan(None, 0, "abstract_status_unreachable")
        queue = [
            (
                max(
                    status_distance[start.status],
                    position_distance.get(start.player, 0),
                ),
                0,
                0,
                start,
            )
        ]
        labels = {start.node: {start.resource: 0}}
        previous: dict[CompilerState, tuple[CompilerState, CompiledAction] | None] = {
            start: None
        }
        serial = expanded = 0
        while queue and expanded < max_expansions:
            _, cost, _, state = heappop(queue)
            if labels.get(state.node, {}).get(state.resource) != cost:
                continue
            expanded += 1
            if (
                state.player == goal_cell
                and state.status.pattern == goal.pattern
                and (goal.color is None or state.status.color == goal.color)
                and state.resource >= reserve
            ):
                steps = []
                cursor = state
                while (link := previous[cursor]) is not None:
                    parent, step = link
                    steps.append(step)
                    cursor = parent
                return CompiledPlan(tuple(reversed(steps)), expanded, None)
            for action in ACTION_DELTAS:
                step = self.predict(state, action)
                if step is None or step.after.status not in status_distance:
                    continue
                after = step.after
                next_cost = cost + 1
                existing = labels.setdefault(after.node, {})
                if any(
                    r >= after.resource and c <= next_cost for r, c in existing.items()
                ):
                    continue
                for r, c in tuple(existing.items()):
                    if r <= after.resource and c >= next_cost:
                        del existing[r]
                existing[after.resource] = next_cost
                previous[after] = state, step
                serial += 1
                heuristic = max(
                    status_distance[after.status],
                    position_distance.get(after.player, 0),
                )
                heappush(queue, (next_cost + heuristic, next_cost, serial, after))
        return CompiledPlan(
            None, expanded, "expansion_budget" if queue else "no_compiled_path"
        )


class OptionExecutionController:
    """Level-local online evidence, bounded compilation and checked execution.

    Plans are disposable predictions. Every actual action, including another
    policy's fallback, is observed. An unexpected transition invalidates all
    remaining steps before any new action can be returned.
    """

    def __init__(self) -> None:
        self.metrics: Counter[str] = Counter()
        self.failures: deque[dict] = deque(maxlen=32)
        self.samples: list[tuple] = []
        self.transitions: dict[tuple, JointMechanismTransition] = {}
        self.previous: MechanismState | None = None
        self.previous_tracks: dict[ObjectSignature, TrackedPhase] = {}
        self.compiler: OptionCompiler | None = None
        self.remaining: deque[CompiledAction] = deque()
        self.expected: CompiledAction | None = None
        self.target: Cell | None = None
        self.ticks = 0
        self.next_search = 0

    def diagnostics(self) -> dict:
        failures = [
            {
                "expected": asdict(item["expected"]),
                "actual": None if item["actual"] is None else asdict(item["actual"]),
                "fields": item["fields"],
            }
            for item in self.failures
        ]
        return json.loads(
            json.dumps(
                {
                    "metrics": dict(self.metrics),
                    "failures": failures,
                    "samples": len(self.samples),
                    "remaining_steps": len(self.remaining),
                },
                default=sorted,
            )
        )

    def reset_episode(self) -> None:
        self.previous = None
        self.previous_tracks.clear()
        self.invalidate("episode_reset")

    def invalidate(self, reason: str) -> None:
        self.remaining.clear()
        self.expected = None
        self.compiler = None
        self.target = None
        self.metrics[f"invalidated_{reason}"] += 1

    def observe(
        self,
        state: MechanismState,
        action: str | None,
        dynamic: DynamicContactModel,
        transitions: Iterable[JointMechanismTransition],
    ) -> None:
        self.ticks += 1
        for edge in transitions:
            key = (
                edge.source,
                edge.actuator,
                edge.destination,
                edge.approach,
                edge.contact_action,
                edge.observed_result,
            )
            self.transitions[key] = edge
        if self.previous is not None and self.previous.level == state.level:
            if action in ACTION_DELTAS and self.previous.resource_level > 0:
                self.samples.append(
                    (
                        self.previous,
                        action,
                        state,
                        self.previous_tracks,
                        dict(dynamic.tracks),
                    )
                )
        if self.expected is not None and self.compiler is not None:
            track = dynamic.tracks.get(self.compiler.signature)
            actual = self.compiler.bind(state, None if track is None else track.phase)
            mismatches = []
            if action != self.expected.action:
                mismatches.append("executed_action")
            if actual is None:
                mismatches.append("unbound_observation")
            else:
                for name in ("player", "status", "phase", "resource", "refills"):
                    if getattr(actual, name) != getattr(self.expected.after, name):
                        mismatches.append(name)
            self.metrics["prediction_checks"] += 1
            if track is not None and track.observed:
                self.metrics["visible_phase_checks"] += 1
            if mismatches:
                self.metrics["prediction_errors"] += 1
                self.failures.append(
                    {
                        "expected": self.expected,
                        "actual": actual,
                        "fields": tuple(mismatches),
                    }
                )
                self.invalidate("prediction_mismatch")
                self.next_search = (
                    self.ticks
                )  # New factual evidence can repair the path.
            else:
                self.expected = None
                self.metrics["validated_steps"] += 1
        self.previous = state
        self.previous_tracks = dict(dynamic.tracks)

    def choose(
        self,
        state: MechanismState,
        dynamic: DynamicContactModel,
        goals: Iterable[tuple[Cell, JointMechanismState]],
        names: Iterable[str],
        *,
        resource_cost: int = 4,
        max_expansions: int = 50000,
    ) -> CompiledAction | None:
        names = set(names)
        if self.expected is not None:
            return None  # Must receive real feedback before another action.
        if self.remaining and self.compiler is not None:
            track = dynamic.tracks.get(self.compiler.signature)
            actual = self.compiler.bind(state, None if track is None else track.phase)
            step = self.remaining[0]
            if step.before == actual and step.action in names:
                self.expected = self.remaining.popleft()
                return self.expected
            self.invalidate("execution_context_changed")
        if (
            self.ticks < self.next_search
            or self.metrics["searches"] >= 32
            or self.metrics["prediction_errors"] >= 3
            or len(dynamic.moving) != 1
            or state.resource_level <= 0
        ):
            return None
        signature = next(iter(dynamic.moving))
        track = dynamic.tracks.get(signature)
        if (
            track is None
            or track.phase is None
            or dynamic.effect_samples[(signature, True)] < dynamic.MIN_SUPPORT
        ):
            return None
        self.next_search = self.ticks + 8
        self.metrics["searches"] += 1
        compiler = OptionCompiler(
            dynamic, signature, self.transitions.values(), resource_cost=resource_cost
        )
        for before, action, after, a, b in self.samples:
            first, last = a.get(signature), b.get(signature)
            compiler.observe(
                before,
                action,
                after,
                None if first is None else first.phase,
                None if last is None else last.phase,
                phase_observed=bool(
                    first and last and first.observed and last.observed
                ),
            )
        start = compiler.bind(state, track.phase)
        if start is None:
            return None
        candidates = []
        # The expansion budget is shared across all candidate targets.
        budget = max_expansions
        for cell, status in goals:
            if budget <= 0:
                break
            plan = compiler.search(start, status, cell, max_expansions=budget)
            budget -= plan.expanded
            self.metrics["expanded"] += plan.expanded
            if plan.steps and all(s.action in names for s in plan.steps):
                candidates.append((len(plan.steps), cell, plan.steps))
        if not candidates:
            self.metrics["no_plan"] += 1
            return None
        _, cell, steps = min(candidates, key=lambda item: (item[0], item[1]))
        self.compiler = compiler
        self.target = cell
        self.remaining = deque(steps)
        self.expected = self.remaining.popleft()
        self.metrics["installed_plans"] += 1
        return self.expected
