"""Prefix-only, retractable contact-effect evidence; never selects an action.

Later level-local motion rules may explain earlier occlusions, but only a
visible reappearance can close a gap. This is conditional attribution, not
causal identification. Raw history, not yesterday's inferred positions, is
reanalysed when rules change. Inference never trains the motion model.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, replace

from .dynamic_contact import (
    OPERATIONS,
    DynamicContactModel,
    contact_relation,
    transform,
)
from .mechanism_state_tracker import ACTION_DELTAS, MechanismState, Pattern
from .motion_context import MotionContextMemory, ObjectSignature


@dataclass(frozen=True)
class EffectObservation:
    tick: int
    state: MechanismState
    action: str | None


@dataclass(frozen=True)
class EffectEvidence:
    transition_tick: int
    available_tick: int
    signature: ObjectSignature
    relation: str
    moved: bool
    pattern_before: Pattern
    pattern_after: Pattern
    color_before: int | None
    color_after: int | None
    # Inclusive observation ticks anchoring each inferred occlusion segment.
    anchors: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class EffectPrediction:
    pattern: Pattern | None
    color: int | None
    pattern_support: int
    color_support: int


class ContactEffectMemory:
    """Bounded episode evidence, keyed by entity, contact relation and motion.

    Requires the explicit composite-entity view. Reset clears episode evidence;
    level changes also reset tick numbering. No cross-level rule transfer is
    asserted. ``learn=False`` freezes evidence, including deferred attribution.
    """

    MIN_SUPPORT = 2
    MAX_HISTORY = 256
    MAX_GAP = 12

    def __init__(self) -> None:
        self.history: deque[EffectObservation] = deque(maxlen=self.MAX_HISTORY)
        self.evidence: dict[int, EffectEvidence] = {}
        self.tick = -1
        self.level: int | None = None
        self.metrics: Counter[str] = Counter()
        self.rejections: Counter[str] = Counter()

    def reset_episode(self) -> None:
        self.history.clear()
        self.evidence.clear()
        self.rejections.clear()

    @staticmethod
    def _same_view(before: MechanismState, after: MechanismState) -> bool:
        if any(
            s.motion_entities is None
            or s.layout is None
            or s.layout.confidence < 0.8
            or s.layout.projected_view
            for s in (before, after)
        ):
            return False
        return (
            before.layout.origin,
            before.layout.cell_size,
            before.layout.bounds,
        ) == (after.layout.origin, after.layout.cell_size, after.layout.bounds)

    @staticmethod
    def _valid(
        before: MechanismState, after: MechanismState, action: str | None
    ) -> bool:
        if (
            action not in ACTION_DELTAS
            or before.level != after.level
            or before.player_position is None
            or after.player_position is None
            or min(before.player_confidence, after.player_confidence) < 0.8
            or (before.resource_capacity > 0 and before.resource_level <= 0)
            or not before.status_pattern
            or not after.status_pattern
            or not ContactEffectMemory._same_view(before, after)
        ):
            return False
        delta = ACTION_DELTAS[action]
        source, target = before.player_position, after.player_position
        return target in {source, (source[0] + delta[0], source[1] + delta[1])}

    @staticmethod
    def _duplicates(state: MechanismState) -> set[ObjectSignature]:
        counts = Counter(
            (item.color, item.area, item.shape_signature)
            for item in state.motion_entities or ()
        )
        return {signature for signature, count in counts.items() if count > 1}

    def _positions(self, samples, dynamic):
        """Resolve positions only from pixels or two-sided checked gaps."""
        visible = [MotionContextMemory.visible_positions(s.state) for s in samples]
        positions = [dict(row) for row in visible]
        anchors: dict[tuple[int, ObjectSignature], tuple[int, int]] = {}
        duplicates = [self._duplicates(s.state) for s in samples]
        for signature in sorted(dynamic.moving):
            track = None
            pending = []
            source_tick = None
            for index, sample in enumerate(samples):
                current = visible[index].get(signature)
                # Visible displacement is factual even before player identity
                # resolves. It seeds heading, never conditional clock training.
                continuous = bool(
                    index
                    and sample.tick == samples[index - 1].tick + 1
                    and sample.action in ACTION_DELTAS
                    and samples[index - 1].state.level == sample.state.level
                    and self._same_view(samples[index - 1].state, sample.state)
                )
                if not continuous or signature in duplicates[index]:
                    track, pending, source_tick = None, [], None
                if current is not None:
                    if pending:
                        previous = samples[index - 1].state
                        moved = previous.player_position != sample.state.player_position
                        predicted = (
                            dynamic.next_phase(signature, track.phase, moved)
                            if track is not None
                            and track.phase is not None
                            and self._valid(previous, sample.state, sample.action)
                            else None
                        )
                        if predicted is not None and predicted[1] == current:
                            for missing_index, cell in pending:
                                positions[missing_index][signature] = cell
                                anchors[missing_index, signature] = (
                                    source_tick,
                                    sample.tick,
                                )
                            track = replace(track, phase=predicted, current=current)
                        else:
                            track = None
                        pending = []
                    track = dynamic._next_observed(track, current)
                    source_tick = sample.tick
                    continue
                if (
                    track is None
                    or track.phase is None
                    or not index
                    or len(pending) >= self.MAX_GAP
                    or not self._valid(
                        samples[index - 1].state, sample.state, sample.action
                    )
                ):
                    track, pending, source_tick = None, [], None
                    continue
                moved = (
                    samples[index - 1].state.player_position
                    != sample.state.player_position
                )
                predicted = dynamic.next_phase(signature, track.phase, moved)
                # Entity extraction can hide a sprite behind the player, but
                # arbitrary disappearance is not evidence of player contact.
                if predicted is None or predicted[1] != sample.state.player_position:
                    track, pending, source_tick = None, [], None
                    continue
                pending.append((index, predicted[1]))
                track = replace(track, phase=predicted, current=predicted[1])
            # Unclosed pending gaps deliberately contribute no inferred cells.
        return positions, anchors

    def observe(self, state, dynamic: DynamicContactModel, *, action=None, learn=True):
        if self.level != state.level:
            self.__init__()
            self.level = state.level
        self.tick += 1
        if action == "RESET" or (
            self.history
            and self.history[-1].state.resource_capacity > 0
            and self.history[-1].state.resource_level == 0
            and state.resource_level > 0
        ):
            self.reset_episode()
        if not learn:
            return  # Do not even stage held-out feedback for later training.
        self.history.append(EffectObservation(self.tick, state, action))
        samples = list(self.history)
        positions, anchors = self._positions(samples, dynamic)
        evidence = {}
        rejected = Counter()
        for index in range(1, len(samples)):
            before, after = samples[index - 1].state, samples[index].state
            if samples[index].tick != samples[index - 1].tick + 1:
                continue
            if not self._valid(before, after, samples[index].action):
                continue
            if any(
                s.motion_entities is None
                or s.layout is None
                or s.layout.confidence < 0.8
                or s.layout.projected_view
                for s in (before, after)
            ):
                rejected["no_entity_view"] += 1
                continue
            # Resource-changing and explicit target contacts can mix effects.
            if after.resource_level > before.resource_level or any(
                o.static_target and o.cell == after.player_position
                for s in (before, after)
                for o in s.objects
            ):
                rejected["other_interaction"] += 1
                continue
            if not dynamic.moving or any(
                sig not in positions[index - 1] or sig not in positions[index]
                for sig in dynamic.moving
            ):
                rejected["unresolved_mover"] += 1
                continue
            contacts = [
                sig
                for sig in dynamic.moving
                if positions[index][sig] == after.player_position
            ]
            if len(contacts) != 1:
                rejected["multiple_contacts" if contacts else "no_contact"] += 1
                continue
            signature = contacts[0]
            relation = contact_relation(
                before.player_position,
                after.player_position,
                positions[index - 1][signature],
                positions[index][signature],
            )
            dependencies = tuple(
                sorted(
                    {
                        anchors[i, sig]
                        for i in (index - 1, index)
                        for sig in dynamic.moving
                        if (i, sig) in anchors
                    }
                )
            )
            tick = samples[index].tick
            item = EffectEvidence(
                tick,
                self.tick,
                signature,
                relation,
                before.player_position != after.player_position,
                before.status_pattern,
                after.status_pattern,
                before.status_color,
                after.status_color,
                dependencies,
            )
            old = self.evidence.get(tick)
            if (
                old is not None
                and replace(item, available_tick=old.available_tick) == old
            ):
                item = old
            evidence[tick] = item
        self.metrics["retracted_samples"] += len(self.evidence.keys() - evidence.keys())
        self.metrics["analyses"] += 1
        self.evidence = evidence
        self.rejections = rejected

    def predict(self, signature, relation, moved, pattern, color) -> EffectPrediction:
        samples = [
            e
            for e in self.evidence.values()
            if (e.signature, e.relation, e.moved) == (signature, relation, moved)
        ]
        candidates = set(OPERATIONS)
        colors = Counter()
        for item in samples:
            candidates.intersection_update(
                op
                for op in OPERATIONS
                if transform(item.pattern_before, op) == item.pattern_after
            )
            if item.color_before == color and item.color_after is not None:
                colors[item.color_after] += 1
        outputs = {transform(pattern, op) for op in candidates}
        return EffectPrediction(
            next(iter(outputs))
            if len(samples) >= self.MIN_SUPPORT and len(outputs) == 1
            else None,
            next(iter(colors))
            if sum(colors.values()) >= self.MIN_SUPPORT and len(colors) == 1
            else None,
            len(samples),
            sum(colors.values()),
        )

    def diagnostics(self) -> dict:
        return {
            "tick": self.tick,
            "history_size": len(self.history),
            "metrics": dict(self.metrics),
            "rejections": dict(self.rejections),
            "evidence": [asdict(e) for e in self.evidence.values()],
            "boundary": "Prefix-only conditional contact effects, not causal proof or an action policy.",
        }
