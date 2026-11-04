"""Learn a small moving component's clock, contact and pattern operation.

Motion rules are learned only from consecutive visible observations. A hidden
position can be a checked belief, never a new motion-training sample. Object
identity, phase and effect are kept separate from a permanent actuator cell.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Iterable

from .mechanism_state_tracker import Cell, MechanismState, Pattern
from .motion_context import MotionContextMemory, ObjectSignature

Phase = tuple[Cell, Cell]  # Last distinct cell, current cell; pauses retain phase.


def rotate(pattern: Pattern) -> Pattern:
    return tuple(tuple(row) for row in zip(*pattern[::-1])) if pattern else ()


def transform(pattern: Pattern, operation: str) -> Pattern:
    result = (
        tuple(tuple(reversed(row)) for row in pattern)
        if operation.startswith("mirror")
        else pattern
    )
    for _ in range(int(operation[-1])):
        result = rotate(result)
    return result


OPERATIONS = tuple(
    f"{base}{turns}" for base in ("rotate", "mirror") for turns in range(4)
)


def contact_relation(
    player_before: Cell, player_after: Cell, object_before: Cell, object_after: Cell
) -> str:
    if (
        player_before != player_after
        and object_before == player_after
        and object_after == player_before
    ):
        return "crossing"
    if object_after == player_after:
        if player_before == player_after:
            return (
                "persistent_overlap"
                if object_before == object_after
                else "object_enters_player"
            )
        if object_before == object_after:
            return "player_enters_object"
        return "meet_after_moving"
    if object_before == player_after:
        return "enter_departing_cell"
    return "separate"


@dataclass(frozen=True)
class TrackedPhase:
    phase: Phase | None
    current: Cell
    observed: bool = True


@dataclass(frozen=True)
class ContactPrediction:
    signature: ObjectSignature
    source_phase: Phase
    destination_phase: Phase
    player_before: Cell
    player_after: Cell
    relation: str
    pattern_after: Pattern | None
    color_preserved: bool


class DynamicContactModel:
    """Conservative local hypotheses; contradictions cause abstention.

    The currently supported contact gate is overlap after both movements.
    Incoming-player and crossing relations are reported separately for audits.
    It is a candidate mechanism, not a prior claim about every moving object.
    """

    MIN_SUPPORT = 2

    def __init__(self) -> None:
        self.previous: MechanismState | None = None
        self.visible: dict[ObjectSignature, Cell] = {}
        self.tracks: dict[ObjectSignature, TrackedPhase] = {}
        self.moving: set[ObjectSignature] = set()
        self.motion: dict[tuple[ObjectSignature, Phase, bool], Counter[Phase]] = (
            defaultdict(Counter)
        )
        self.operators: dict[tuple[ObjectSignature, bool], set[str]] = {}
        self.effect_samples: Counter[tuple[ObjectSignature, bool]] = Counter()
        self.color_evidence: dict[tuple[ObjectSignature, bool], Counter[bool]] = (
            defaultdict(Counter)
        )
        self.metrics: Counter[str] = Counter()
        self.last_contacts: tuple[ContactPrediction, ...] = ()

    def reset_episode(self) -> None:
        """Retain learned level-local rules, discard episode-local beliefs."""
        self.previous = None
        self.visible.clear()
        self.tracks.clear()
        self.last_contacts = ()

    def next_phase(
        self, signature: ObjectSignature, phase: Phase, moved: bool
    ) -> Phase | None:
        evidence = self.motion.get((signature, phase, moved), {})
        if len(evidence) != 1 or sum(evidence.values()) < self.MIN_SUPPORT:
            return None
        return next(iter(evidence))

    def pattern_after(
        self, signature: ObjectSignature, pattern: Pattern, moved: bool = True
    ) -> Pattern | None:
        key = (signature, moved)
        if self.effect_samples[key] < self.MIN_SUPPORT:
            return None
        outputs = {transform(pattern, op) for op in self.operators.get(key, set())}
        return next(iter(outputs)) if len(outputs) == 1 else None

    def preserves_color(self, signature: ObjectSignature, moved: bool = True) -> bool:
        evidence = self.color_evidence.get((signature, moved), {})
        return evidence.get(True, 0) >= self.MIN_SUPPORT and not evidence.get(False, 0)

    def predict_contact(
        self,
        signature: ObjectSignature,
        phase: Phase,
        player_before: Cell,
        player_after: Cell,
        pattern: Pattern,
    ) -> ContactPrediction | None:
        successor = self.next_phase(signature, phase, player_before != player_after)
        if successor is None or successor[1] != player_after:
            return None
        relation = contact_relation(player_before, player_after, phase[1], successor[1])
        return ContactPrediction(
            signature,
            phase,
            successor,
            player_before,
            player_after,
            relation,
            self.pattern_after(signature, pattern, player_before != player_after),
            self.preserves_color(signature, player_before != player_after),
        )

    @staticmethod
    def _next_observed(track: TrackedPhase | None, current: Cell) -> TrackedPhase:
        if track is None:
            return TrackedPhase(None, current)
        if current == track.current:
            return TrackedPhase(track.phase, current)
        if sum(abs(a - b) for a, b in zip(current, track.current)) == 1:
            return TrackedPhase((track.current, current), current)
        return TrackedPhase(None, current)

    def observe(
        self, state: MechanismState, *, action: str | None = None, learn: bool = True
    ) -> None:
        before = self.previous
        boundary = before is not None and (
            before.level != state.level
            or (
                before.resource_capacity > 0
                and before.resource_level == 0
                and state.resource_level > 0
            )
        )
        if before is not None and before.level != state.level:
            self.__init__()
        if action == "RESET" or boundary:
            self.reset_episode()
        before = self.previous
        current_visible = MotionContextMemory.visible_positions(state)
        old_tracks = dict(self.tracks)
        self.last_contacts = ()
        valid = bool(
            before is not None
            and action in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
            and before.player_position is not None
            and state.player_position is not None
            and min(before.player_confidence, state.player_confidence) >= 0.8
            and (before.resource_capacity == 0 or before.resource_level > 0)
        )
        if before is not None and not valid:
            # An unmodelled action or uncertain player breaks phase continuity.
            old_tracks = {}
        duplicates = Counter(
            (item.color, item.area, item.shape_signature)
            for item in (
                state.objects
                if state.motion_entities is None
                else state.motion_entities
            )
            if item.object_id not in state.player_component_ids
        )
        proposed: dict[ObjectSignature, Phase] = {}
        contacts = []
        if valid:
            moved = before.player_position != state.player_position
            for signature, track in old_tracks.items():
                phase = track.phase
                if (
                    signature not in self.moving
                    or phase is None
                    or duplicates[signature] > 1
                ):
                    continue
                successor = self.next_phase(signature, phase, moved)
                actual = current_visible.get(signature)
                if successor is not None:
                    proposed[signature] = successor
                    if actual is not None:
                        self.metrics["visible_motion_checks"] += 1
                        self.metrics["visible_motion_errors"] += successor[1] != actual
                    if not track.observed and actual is not None:
                        self.metrics["reappearance_checks"] += 1
                        self.metrics["reappearance_errors"] += successor[1] != actual
                    predicted = self.predict_contact(
                        signature,
                        phase,
                        before.player_position,
                        state.player_position,
                        before.status_pattern,
                    )
                    # A visible object elsewhere falsifies this overlap. A
                    # disappearance alone is not evidence unless the predicted
                    # cell is occupied by the actual composite player.
                    if predicted is not None and actual in {
                        None,
                        state.player_position,
                    }:
                        contacts.append(predicted)
                # Never learn motion from an occlusion projection.
                if learn and track.observed and actual is not None:
                    destination = self._next_observed(track, actual)
                    if destination.phase is not None:
                        self.motion[(signature, phase, moved)][destination.phase] += 1
        self.last_contacts = tuple(contacts)
        if len(contacts) == 1 and learn:
            contact = contacts[0]
            key = (contact.signature, before.player_position != state.player_position)
            matching = {
                op
                for op in OPERATIONS
                if transform(before.status_pattern, op) == state.status_pattern
            }
            candidates = self.operators.setdefault(key, set(OPERATIONS))
            candidates.intersection_update(matching)
            self.effect_samples[key] += 1
            self.color_evidence[key][before.status_color == state.status_color] += 1
        elif len(contacts) > 1:
            self.metrics["ambiguous_contacts"] += 1

        next_tracks = {}
        for signature, current in current_visible.items():
            old = old_tracks.get(signature)
            if old is not None and not old.observed:
                prediction = proposed.get(signature)
                tracked = (
                    TrackedPhase(prediction, current)
                    if prediction is not None and prediction[1] == current
                    else TrackedPhase(None, current)
                )
            else:
                tracked = self._next_observed(old, current)
            if (
                learn
                and valid
                and old is not None
                and old.observed
                and tracked.phase is not None
                and current != old.current
            ):
                self.moving.add(signature)
            next_tracks[signature] = tracked
        for signature, successor in proposed.items():
            if signature in current_visible:
                continue
            if successor[1] == state.player_position:
                next_tracks[signature] = TrackedPhase(successor, successor[1], False)
                self.metrics["occluded_belief_steps"] += 1
            else:
                self.metrics["unexplained_disappearances"] += 1
        self.tracks = next_tracks
        self.visible = current_visible
        self.previous = state

    def contact_route(
        self,
        state: MechanismState,
        signature: ObjectSignature,
        destination: Callable[[Cell, str], Cell | None],
        actions: Iterable[str],
        *,
        goal_pattern: Pattern | None = None,
        horizon: int = 6,
    ) -> tuple[tuple[str, ...], ContactPrediction] | None:
        """Search a bounded contact path; return only its first-step contract.

        All future object phases come from observed rules. Player destinations
        are supplied by the caller and may be geometric hypotheses, so the
        route is always rechecked after executing one action.
        """
        track = self.tracks.get(signature)
        if track is None or track.phase is None or state.player_position is None:
            return None
        start = (state.player_position, track.phase)
        queue = deque([(start, ())])
        visited = {start}
        while queue:
            (player, phase), path = queue.popleft()
            if len(path) >= horizon:
                continue
            for action in actions:
                target = destination(player, action)
                if target is None:
                    continue
                successor = self.next_phase(signature, phase, player != target)
                if successor is None:
                    continue
                contact = self.predict_contact(
                    signature, phase, player, target, state.status_pattern
                )
                next_path = path + (action,)
                if contact is not None:
                    if player != target and (
                        goal_pattern is None or contact.pattern_after == goal_pattern
                    ):
                        return next_path, contact
                    # Never plan through an unsupported or unwanted effect.
                    continue
                next_state = (target, successor)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, next_path))
        return None
