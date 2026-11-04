"""Bounded multi-entity options with explicit unknown effects and feedback.

Only observation-derived roles/rules enter this model. Search projections are
falsifiable hypotheses. Unknown contacts terminate experiments, never become
imaginary intermediate states in a solution path. No game IDs or route replay.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from heapq import heappop, heappush

from .contact_effects import ContactEffectMemory
from .dynamic_contact import Phase, contact_relation
from .mechanism_state_tracker import ACTION_DELTAS, Cell, Pattern
from .motion_context import MotionContextMemory, ObjectSignature


def _signature_to_pattern(signature: str) -> Pattern:
    """Convert status signature string to Pattern tuple structure."""
    if signature == "empty":
        return ()
    return tuple(
        tuple(int(value) for value in row)
        for row in signature.split("/")
        if row
    )


@dataclass(frozen=True)
class EntityNode:
    player: Cell
    pattern: Pattern
    color: int | None
    phases: tuple[Phase, ...]
    resource: int
    refills: frozenset[Cell] = frozenset()
    fixtures: frozenset[tuple[Cell, ObjectSignature]] = frozenset()

    @property
    def key(self):
        return (
            self.player,
            self.pattern,
            self.color,
            self.phases,
            self.refills,
            self.fixtures,
        )

    def diagnostic(self):
        return {
            **asdict(self),
            "refills": sorted(self.refills),
            "fixtures": sorted(self.fixtures),
        }


@dataclass(frozen=True)
class EntityStep:
    before: EntityNode
    action: str
    after: EntityNode
    hypotheses: tuple[str, ...]
    contact: ObjectSignature | None = None
    relation: str | None = None
    unknown_fields: tuple[str, ...] = ()
    terminal: bool = False
    fixture: tuple[Cell, ObjectSignature] | None = None

    def diagnostic(self):
        return {
            **asdict(self),
            "before": self.before.diagnostic(),
            "after": self.after.diagnostic(),
        }


class EntityOptionModel:
    """Frozen level-local evidence and phase-vector search representation."""

    def __init__(self, controller, state, dynamic, effects, targets):
        self.dynamic = deepcopy(dynamic)
        self.effects = deepcopy(effects)
        self.signatures = tuple(sorted(dynamic.moving))
        self.samples = tuple(controller.samples)
        self.cost = next(iter(controller.costs))
        self.projection = set(state.navigation_projection_cells)
        self.projection.add(state.player_position)
        self.targets = dict(targets)
        self.refill_values = dict(controller.refill_values)
        self.refill_signatures = deepcopy(controller.refill_signatures)
        self.failures = set(controller.bad_edges)
        self.rejected_entries = set(controller.rejected_entries)
        # Convert MechanismPolicy contextual_special_transitions keys from
        # (cell, action, signature_string, color) to (cell, action, Pattern, color)
        self.special_transitions = {}
        for (cell, action, pattern_or_sig, color), dest in controller.special_transitions.items():
            # Handle both Pattern tuples (from tests) and signature strings (from MechanismPolicy)
            if isinstance(pattern_or_sig, str):
                pattern = _signature_to_pattern(pattern_or_sig)
            else:
                pattern = pattern_or_sig
            self.special_transitions[(cell, action, pattern, color)] = dest
        self.atomic = defaultdict(set)
        for (
            before,
            action,
            after,
            phases,
            next_phases,
            signatures,
        ) in controller.atomic_samples:
            if signatures != self.signatures:
                continue
            source = self.bind_phases(before, phases)
            destination = self.bind_phases(after, next_phases)
            if source is not None and destination is not None:
                # Resource and visible fixture availability are part of this
                # exact key. Never project a special displacement's clock or
                # effects from its endpoint geometry alone.
                self.atomic[source, action].add(destination)
        self.navigation = defaultdict(list)
        for before, action, after, phases, signatures in self.samples:
            if signatures == self.signatures:
                self.navigation[before.player_position, action].append(
                    (
                        before.status_pattern,
                        before.status_color,
                        phases,
                        after.player_position,
                    )
                )

    @staticmethod
    def objects_at(state, cell):
        return {
            (o.color, o.area, o.shape_signature)
            for o in state.motion_entities or ()
            if o.cell == cell
        }

    def bind(self, state, dynamic):
        signatures = tuple(sorted(dynamic.moving))
        if signatures != self.signatures:
            return None
        return self.bind_phases(
            state,
            tuple(
                None if sig not in dynamic.tracks else dynamic.tracks[sig].phase
                for sig in signatures
            ),
        )

    def bind_phases(self, state, phases):
        if (
            state.player_position is None
            or state.player_confidence < 0.8
            or state.motion_entities is None
            or ContactEffectMemory._duplicates(state) & set(self.signatures)
            or len(phases) != len(self.signatures)
            or None in phases
        ):
            return None
        return EntityNode(
            state.player_position,
            state.status_pattern,
            state.status_color,
            tuple(phases),
            state.resource_level,
            frozenset(
                cell
                for cell, signatures in self.refill_signatures.items()
                if self.objects_at(state, cell) & signatures
            ),
            frozenset(
                (obj.cell, (obj.color, obj.area, obj.shape_signature))
                for obj in state.motion_entities
                if (obj.color, obj.area, obj.shape_signature) not in self.signatures
                and (obj.color, obj.area, obj.shape_signature)
                not in self.refill_signatures.get(obj.cell, ())
            ),
        )

    def movement(self, node, action):
        edge = node.player, action
        if edge in self.failures:
            return None

        # Check for consistent contextual special transitions first
        contextual_key = (node.player, action, node.pattern, node.color)
        if contextual_key in self.special_transitions:
            destination = self.special_transitions[contextual_key]
            return destination, "contextual_special_transition"

        samples = self.navigation.get(edge, ())
        tiers = (
            (
                "observed_navigation_context",
                [
                    s
                    for s in samples
                    if s[:3] == (node.pattern, node.color, node.phases)
                ],
            ),
            (
                "navigation_phase_projection",
                [s for s in samples if s[:2] == (node.pattern, node.color)],
            ),
            (
                "navigation_status_projection",
                [s for s in samples if s[2] == node.phases],
            ),
            ("navigation_context_projection", samples),
        )
        for label, matches in tiers:
            if matches:
                outputs = {s[3] for s in matches}
                return (next(iter(outputs)), label) if len(outputs) == 1 else None
        dx, dy = ACTION_DELTAS[action]
        target = node.player[0] + dx, node.player[1] + dy
        if target in self.projection or target in self.targets:
            return target, "geometric_navigation_projection"
        return None

    def step(self, node, action, *, experiment=False, fixture_experiment=False):
        if node.resource <= self.cost:
            return None
        exact = self.atomic.get((node, action))
        if exact:
            if len(exact) != 1:
                return None  # Conflicting complete outcomes do not vote.
            return EntityStep(
                node, action, next(iter(exact)), ("observed_complete_action",)
            )
        movement = self.movement(node, action)
        if movement is None:
            return None
        target, label = movement
        if (
            target,
            node.pattern,
            node.color,
            node.fixtures,
            node.refills,
        ) in self.rejected_entries and label not in (
            "observed_navigation_context",
            "contextual_special_transition",
        ):
            # Quarantine an unsupported arrival hypothesis across approaches;
            # this is abstention, not a learned universal gate/teleport rule.
            return None
        dx, dy = ACTION_DELTAS[action]
        if target not in {node.player, (node.player[0] + dx, node.player[1] + dy)}:
            # Nonlocal transitions can use a different entity clock, but allow
            # them when backed by a consistent contextual special transition.
            if label != "contextual_special_transition":
                return None
        moved = target != node.player
        fixtures = [(cell, sig) for cell, sig in node.fixtures if cell == target]
        fixture = None
        if fixtures and target not in node.refills:
            learned_refills = set().union(*self.refill_signatures.values())
            if (
                not fixture_experiment
                or not experiment
                or len(fixtures) != 1
                or fixtures[0][1] not in learned_refills
                or not moved
                or target in self.targets
            ):
                return None
            fixture = fixtures[0]
        phases = tuple(
            self.dynamic.next_phase(sig, phase, moved)
            for sig, phase in zip(self.signatures, node.phases)
        )
        if None in phases:
            return None
        contacts = [
            (sig, phase, nxt)
            for sig, phase, nxt in zip(self.signatures, node.phases, phases)
            if nxt[1] == target
        ]
        if len(contacts) > 1:
            return None
        if fixture is not None:
            if contacts:
                return None  # Do not confound a fixture and moving contact.
            return EntityStep(
                node,
                action,
                replace(
                    node,
                    player=target,
                    phases=phases,
                    resource=node.resource - self.cost,
                ),
                (label, "unknown_fixture_contact_experiment"),
                unknown_fields=("pattern", "color", "resource", "refills", "fixtures"),
                fixture=fixture,
            )
        pattern, color = node.pattern, node.color
        unknown, relation, signature = (), None, None
        hypotheses = [label, "noncontact_status_preservation"]
        if contacts:
            if target in node.refills or target in self.targets:
                return None  # No effect ordering assumption for mixed contacts.
            signature, phase, nxt = contacts[0]
            relation = contact_relation(node.player, target, phase[1], nxt[1])
            prediction = self.effects.predict(
                signature, relation, moved, pattern, color
            )
            unknown = tuple(
                field
                for field, value in (
                    ("pattern", prediction.pattern),
                    ("color", prediction.color),
                )
                if value is None
            )
            if unknown and not experiment:
                return None
            pattern = prediction.pattern if prediction.pattern is not None else pattern
            color = prediction.color if prediction.color is not None else color
            hypotheses.append("conditional_contact_effect")
        terminal = target in self.targets
        if terminal:
            required = self.targets[target]
            if (
                pattern != required[0]
                or color is None
                or (required[1] is not None and color != required[1])
            ):
                return None
            hypotheses.append("matching_target_completion_hypothesis")
        resource = node.resource - self.cost
        refills = node.refills
        if moved and target in refills:
            resource = self.refill_values[target]
            refills = refills - {target}
            hypotheses.append("visible_single_use_refill_projection")
        after = replace(
            node,
            player=target,
            pattern=pattern,
            color=color,
            phases=phases,
            resource=resource,
            refills=refills,
        )
        return EntityStep(
            node,
            action,
            after,
            tuple(hypotheses),
            signature,
            relation,
            unknown,
            terminal,
        )

    def search_goal(self, start, names, *, max_expansions=12000, horizon=48):
        if not self.targets:
            return None, 0

        def heuristic(node):
            return min(
                abs(node.player[0] - p[0]) + abs(node.player[1] - p[1])
                for p in self.targets
            )

        serial = 0
        queue = [(heuristic(start), 0, serial, start, ())]
        best = {start.key: (0, start.resource)}
        expanded = 0
        while queue and expanded < max_expansions:
            _, depth, _, node, path = heappop(queue)
            if depth >= horizon:
                continue
            expanded += 1
            for action in names:
                step = self.step(node, action)
                if step is None:
                    continue
                candidate = path + (step,)
                if step.terminal:
                    return candidate, expanded
                prior = best.get(step.after.key)
                if (
                    prior is not None
                    and prior[0] <= depth + 1
                    and prior[1] >= step.after.resource
                ):
                    continue
                best[step.after.key] = (depth + 1, step.after.resource)
                serial += 1
                heappush(
                    queue,
                    (
                        depth + 1 + heuristic(step.after),
                        depth + 1,
                        serial,
                        step.after,
                        candidate,
                    ),
                )
        return None, expanded

    def search_experiment(
        self, start, names, attempts, *, horizon=12, max_expansions=2500
    ):
        queue = deque([(start, ())])
        seen = {start.key}
        candidates = []
        expanded = 0
        while queue and expanded < max_expansions:
            node, path = queue.popleft()
            if len(path) >= horizon:
                continue
            expanded += 1
            for action in names:
                step = self.step(node, action, experiment=True)
                if step is None or step.terminal:
                    continue
                candidate = path + (step,)
                if step.contact is not None and step.unknown_fields:
                    moved = step.before.player != step.after.player
                    key = (step.contact, step.relation, moved, node.color)
                    # Only unknown contacts terminate the candidate. Supported
                    # effects can be traversed to reach missing input contexts.
                    if attempts[key] < 3:
                        evidence = [
                            e
                            for e in self.effects.evidence.values()
                            if e.signature == step.contact
                            and e.relation == step.relation
                            and e.moved == moved
                        ]
                        prediction = self.effects.predict(
                            step.contact, step.relation, moved, node.pattern, node.color
                        )
                        # >=2 samples and still unknown for this exact input
                        # may be an ambiguity/counterexample, not sparse data.
                        if prediction.color is None and prediction.color_support >= 2:
                            continue
                        score = (
                            # Preserve a modelled pattern while filling missing
                            # input-color evidence before changing the pattern
                            # through an entirely unidentified component.
                            "pattern" in step.unknown_fields,
                            len(step.unknown_fields),
                            0 if not evidence else 1,
                            len(candidate),
                            prediction.color_support,
                            step.contact,
                        )
                        candidates.append((score, candidate, key))
                    continue
                if step.after.key not in seen:
                    seen.add(step.after.key)
                    queue.append((step.after, candidate))
        if not candidates:
            return None, None, expanded
        _, path, key = min(candidates, key=lambda c: c[0])
        return path, key, expanded

    def search_fixture_experiment(
        self, start, names, attempts, *, horizon=8, max_expansions=1000
    ):
        """Probe one unvisited same-appearance fixture, never a solution edge.

        A previously observed refill appearance proposes the experiment only;
        no resource, status or environmental effect is inferred from its look.
        """
        queue = deque([(start, ())])
        seen = {start.key}
        expanded = 0
        while queue and expanded < max_expansions:
            node, path = queue.popleft()
            if len(path) >= horizon:
                continue
            expanded += 1
            for action in names:
                step = self.step(node, action, experiment=True, fixture_experiment=True)
                if step is None or step.terminal:
                    continue
                candidate = path + (step,)
                if step.fixture is not None:
                    if not attempts[step.fixture]:
                        return candidate, expanded
                    continue
                if step.unknown_fields or step.after.key in seen:
                    continue  # Unknown contacts cannot be intermediate states.
                seen.add(step.after.key)
                queue.append((step.after, candidate))
        return None, expanded


class EntityOptionController:
    """Executes one action at a time; unknown fields are experiment outcomes."""

    MAX_STEPS = 120
    MAX_CONTACTS = 12
    MAX_FIXTURE_EXPERIMENTS = 1

    def __init__(self):
        self.previous = None
        self.previous_phases = ()
        self.previous_signatures = ()
        self.samples = deque(maxlen=512)
        self.atomic_samples = deque(maxlen=512)
        self.costs = Counter()
        self.refill_values = {}
        self.refill_signatures = defaultdict(set)
        self.bad_edges = set()
        self.rejected_entries = set()
        self.special_transitions = {}
        self.attempts = Counter()
        self.fixture_attempts = Counter()
        self.metrics = Counter()
        self.failures = []
        self.model = None
        self.expected = None
        self.remaining = deque()
        self.objective = None
        self.pending_key = None
        self.reacquire = None
        self.tick = 0
        self.next_goal_search = 0
        self.next_experiment = 0
        self.level = None

    def invalidate(self, reason):
        self.metrics["invalidated_" + reason] += 1
        self.expected = None
        self.remaining.clear()
        self.model = None
        self.pending_key = None

    def reset_episode(self):
        self.invalidate("episode_reset")
        self.previous = None
        self.previous_phases = ()
        self.previous_signatures = ()
        self.reacquire = None

    def observe(self, state, dynamic, action):
        if self.level != state.level:
            self.__init__()
            self.level = state.level
        self.tick += 1
        if action == "RESET" or (
            self.previous is not None
            and self.previous.resource_level == 0
            and state.resource_level > 0
        ):
            self.reset_episode()
        before = self.previous
        signatures = tuple(sorted(dynamic.moving))
        phases = tuple(
            None if sig not in dynamic.tracks else dynamic.tracks[sig].phase
            for sig in signatures
        )
        if (
            before is not None
            and action in ACTION_DELTAS
            and before.resource_level > 0
            and min(before.player_confidence, state.player_confidence) >= 0.8
            and ContactEffectMemory._same_view(before, state)
            and before.player_position is not None
            and state.player_position is not None
        ):
            self.samples.append(
                (before, action, state, self.previous_phases, self.previous_signatures)
            )
            if signatures == self.previous_signatures:
                self.atomic_samples.append(
                    (before, action, state, self.previous_phases, phases, signatures)
                )
            delta = before.resource_level - state.resource_level
            if delta > 0:
                self.costs[delta] += 1
            if delta < 0 and before.player_position != state.player_position:
                objects = EntityOptionModel.objects_at(before, state.player_position)
                if objects:
                    self.refill_signatures[state.player_position].update(objects)
                    self.refill_values[state.player_position] = state.resource_level
        if self.expected is not None:
            expected = self.expected
            actual = self.model.bind(state, dynamic)
            mismatches = []
            if action != expected.action:
                mismatches.append("executed_action")
            if actual is None:
                mismatches.append("unbound_observation")
            else:
                for field in (
                    "player",
                    "pattern",
                    "color",
                    "phases",
                    "resource",
                    "refills",
                    "fixtures",
                ):
                    if field in expected.unknown_fields:
                        self.metrics["unknown_fields_observed"] += 1
                        continue
                    self.metrics["known_fields_checked"] += 1
                    if getattr(actual, field) != getattr(expected.after, field):
                        mismatches.append(field)
            self.metrics["prediction_checks"] += 1
            if expected.fixture is not None and action == expected.action:
                self.fixture_attempts[expected.fixture] += 1
                self.metrics["fixture_experiments_executed"] += 1
            self.metrics["visible_phase_checks"] += sum(
                dynamic.tracks[sig].observed
                for sig in signatures
                if sig in dynamic.tracks
            )
            self.metrics["belief_phase_checks"] += sum(
                not dynamic.tracks[sig].observed
                for sig in signatures
                if sig in dynamic.tracks
            )
            if mismatches:
                self.metrics["prediction_errors"] += 1
                self.failures.append(
                    {
                        "expected": expected.diagnostic(),
                        "actual": None if actual is None else actual.diagnostic(),
                        "fields": mismatches,
                    }
                )
                if "player" in mismatches:
                    self.bad_edges.add((expected.before.player, expected.action))
                    if action == expected.action and actual is not None:
                        self.rejected_entries.add(
                            (
                                expected.after.player,
                                expected.before.pattern,
                                expected.before.color,
                                expected.before.fixtures,
                                expected.before.refills,
                            )
                        )
                self.invalidate("prediction_mismatch")
                self.reacquire = None
            else:
                self.metrics["validated_steps"] += 1
                if expected.unknown_fields:
                    self.metrics["experimental_outcomes"] += 1
                if expected.fixture is not None:
                    self.metrics["fixture_outcomes_observed"] += 1
                    self.remaining.clear()
                if expected.contact is not None:
                    self.metrics["contacts_executed"] += 1
                    self.reacquire = expected.contact
                    if self.pending_key is not None and expected.unknown_fields:
                        self.attempts[self.pending_key] += 1
                    self.remaining.clear()
                if expected.terminal:
                    # Matching a target is only a hypothesis until the SDK
                    # advances the level. A failed contact must not loop.
                    self.bad_edges.add((expected.before.player, expected.action))
                    self.metrics["terminal_without_completion"] += 1
                    self.remaining.clear()
                self.expected = None
        self.previous = state
        self.previous_phases = phases
        self.previous_signatures = signatures

    def choose(self, state, dynamic, effects, targets, names, special_transitions=None):
        names = tuple(n for n in names if n in ACTION_DELTAS)
        if special_transitions is not None:
            self.special_transitions = special_transitions
        if self.metrics["steps_selected"] >= self.MAX_STEPS:
            if self.model is not None:
                self.invalidate("action_budget")
            return None
        if (
            self.expected is not None
            or len(dynamic.moving) < 2
            or len(dynamic.moving) > 5
            or state.motion_entities is None
            or state.player_position is None
            or state.player_confidence < 0.8
            or state.layout is None
            or state.layout.confidence < 0.8
            or state.layout.projected_view
            or not state.status_pattern
            or not self.costs
            or len(self.costs) != 1
            or sum(self.costs.values()) < 2
            or self.metrics["prediction_errors"] >= 3
        ):
            return None
        if self.model is not None and self.model.effects.evidence != effects.evidence:
            self.invalidate("effect_evidence_changed")
        if self.model is not None and self.remaining:
            actual = self.model.bind(state, dynamic)
            step = self.remaining[0]
            if step.before == actual and step.action in names:
                self.expected = self.remaining.popleft()
                self.metrics["steps_selected"] += 1
                return self.expected
            self.invalidate("execution_context_changed")
        model = EntityOptionModel(self, state, dynamic, effects, targets)
        start = model.bind(state, dynamic)
        if start is None:
            return None
        if (
            self.rejected_entries
            and sum(self.fixture_attempts.values()) < self.MAX_FIXTURE_EXPERIMENTS
        ):
            path, expanded = model.search_fixture_experiment(
                start, names, self.fixture_attempts
            )
            self.metrics["fixture_searches"] += 1
            self.metrics["fixture_expanded"] += expanded
            if path:
                self.objective = "test_visible_fixture_after_navigation_failure"
                return self._install(model, path)
        if self.reacquire is not None:
            visible = MotionContextMemory.visible_positions(state)
            if self.reacquire in visible:
                self.reacquire = None
            else:
                for name in names:
                    step = model.step(start, name)
                    if step is not None and step.contact is None and not step.terminal:
                        self.objective = "reacquire_entity_after_contact"
                        return self._install(model, (step,))
                self.metrics["reacquire_unavailable"] += 1
                return None
        if self.tick >= self.next_goal_search:
            path, expanded = model.search_goal(start, names)
            self.next_goal_search = self.tick + 4
            self.metrics["goal_searches"] += 1
            self.metrics["goal_expanded"] += expanded
            if path:
                self.objective = "execute_entity_goal_option"
                return self._install(model, path)
        if (
            self.tick < self.next_experiment
            or self.metrics["contacts_executed"] >= self.MAX_CONTACTS
        ):
            return None
        path, key, expanded = model.search_experiment(start, names, self.attempts)
        self.next_experiment = self.tick + 4
        self.metrics["experiment_searches"] += 1
        self.metrics["experiment_expanded"] += expanded
        if path:
            self.objective = "identify_entity_contact_effect"
            return self._install(model, path, experiment_key=key)
        return None

    def _install(self, model, path, experiment_key=None):
        self.model = model
        self.pending_key = experiment_key
        self.remaining = deque(path)
        self.expected = self.remaining.popleft()
        self.metrics["installed_plans"] += 1
        self.metrics["steps_selected"] += 1
        return self.expected

    def diagnostics(self):
        return {
            "metrics": dict(self.metrics),
            "failures": self.failures,
            "attempts": [
                {"key": key, "count": count} for key, count in self.attempts.items()
            ],
            "resource_cost_evidence": dict(self.costs),
            "remaining_steps": len(self.remaining),
            "objective": self.objective,
            "bad_edges": sorted(self.bad_edges),
            "rejected_entries": [
                {
                    "cell": cell,
                    "pattern": pattern,
                    "color": color,
                    "fixtures": sorted(fixtures),
                    "refills": sorted(refills),
                }
                for cell, pattern, color, fixtures, refills in sorted(
                    self.rejected_entries, key=repr
                )
            ],
            "complete_action_samples": len(self.atomic_samples),
            "fixture_attempts": [
                {"fixture": fixture, "count": count}
                for fixture, count in sorted(self.fixture_attempts.items())
            ],
        }
