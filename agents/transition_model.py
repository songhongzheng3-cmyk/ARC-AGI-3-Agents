"""Local empirical state-transition model for one ARC episode."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from .object_extractor import GridSnapshot


@dataclass(frozen=True)
class Transition:
    before_hash: str
    after_hash: str
    action: str
    data: dict[str, Any]
    changed: bool
    score_delta: int
    changed_cells: int
    object_changes: tuple[str, ...]


@dataclass
class _Pending:
    state_hash: str
    snapshot: GridSnapshot
    frame: Any
    action: str
    data: dict[str, Any]
    score: int


class TransitionModel:
    """Stores observed transitions and simple action-level statistics."""

    def __init__(self, history_limit: int = 32) -> None:
        self.transitions: deque[Transition] = deque(maxlen=max(1, history_limit))
        self.by_state_action: dict[tuple[str, str], list[Transition]] = defaultdict(list)
        self.snapshots: dict[str, GridSnapshot] = {}
        self._pending: _Pending | None = None

    @staticmethod
    def _object_signature(snapshot: GridSnapshot) -> set[str]:
        return {
            f"{item.color}:{item.bbox}:{item.area}:{item.shape}"
            for item in snapshot.objects
        }

    @staticmethod
    def _grid_delta(before: Any, after: Any) -> int:
        try:
            left = before.frame[-1]
            right = after.frame[-1]
            return sum(
                1
                for row_a, row_b in zip(left, right)
                for value_a, value_b in zip(row_a, row_b)
                if value_a != value_b
            )
        except (AttributeError, IndexError, TypeError):
            return 0

    def observe_state(
        self,
        state_hash: str,
        snapshot: GridSnapshot,
        frame: Any,
    ) -> Transition | None:
        """Close the pending action using the newly observed frame."""
        if self._pending is None:
            self.snapshots[state_hash] = snapshot
            return None
        pending = self._pending
        changed = pending.state_hash != state_hash
        before_objects = self._object_signature(pending.snapshot)
        after_objects = self._object_signature(snapshot)
        added = sorted(after_objects - before_objects)
        removed = sorted(before_objects - after_objects)
        score = int(getattr(frame, "levels_completed", 0))
        transition = Transition(
            before_hash=pending.state_hash,
            after_hash=state_hash,
            action=pending.action,
            data=dict(pending.data),
            changed=changed,
            score_delta=score - pending.score,
            changed_cells=self._grid_delta(pending.frame, frame),
            object_changes=tuple([f"added:{item}" for item in added] + [f"removed:{item}" for item in removed]),
        )
        self.transitions.append(transition)
        self.by_state_action[(pending.state_hash, pending.action)].append(transition)
        self.snapshots[state_hash] = snapshot
        self._pending = None
        return transition

    def record_action(
        self,
        state_hash: str,
        snapshot: GridSnapshot,
        frame: Any,
        action: str,
        data: dict[str, Any] | None,
        score: int,
    ) -> None:
        action_key = action.upper()
        if action_key == "ACTION6":
            action_key = f"ACTION6({int((data or {}).get('x', 0))},{int((data or {}).get('y', 0))})"
        self.snapshots[state_hash] = snapshot
        self._pending = _Pending(
            state_hash=state_hash,
            snapshot=snapshot,
            frame=frame,
            action=action_key,
            data=dict(data or {}),
            score=score,
        )

    def action_stats(self, state_hash: str, action: str) -> dict[str, float]:
        records = self.by_state_action.get((state_hash, action.upper()), [])
        if not records:
            return {"trials": 0.0, "change_rate": 0.0, "score_rate": 0.0}
        return {
            "trials": float(len(records)),
            "change_rate": sum(item.changed for item in records) / len(records),
            "score_rate": sum(item.score_delta for item in records) / len(records),
        }

    @staticmethod
    def _state_features(snapshot: GridSnapshot) -> set[str]:
        return {
            f"{item.color}:{item.shape}:{item.area}:{item.touches_boundary}"
            for item in snapshot.objects
        }

    @classmethod
    def _similarity(cls, left: GridSnapshot, right: GridSnapshot) -> float:
        if (left.width, left.height, left.background) != (
            right.width,
            right.height,
            right.background,
        ):
            return 0.0
        a = cls._state_features(left)
        b = cls._state_features(right)
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))

    @staticmethod
    def _outcome(records: list[Transition]) -> float | None:
        if not records:
            return None
        # Visible progress is stronger evidence than a state change alone.
        return sum(
            (1.0 if item.changed else -0.35) + 2.0 * item.score_delta
            for item in records
        ) / len(records)

    def score_action(
        self,
        state_hash: str,
        snapshot: GridSnapshot,
        action: str,
    ) -> tuple[float, str]:
        """Score an action using exact, similar, global, then exploration data."""
        action = action.upper()
        family = action.split("(", 1)[0]
        exact = self._outcome(self.by_state_action.get((state_hash, action), []))
        if exact is not None:
            return exact, "exact-state"

        similar: list[tuple[float, float]] = []
        for (other_hash, other_action), records in self.by_state_action.items():
            if other_action != action and not (
                family == "ACTION6" and other_action.startswith("ACTION6(")
            ):
                continue
            other = self.snapshots.get(other_hash)
            if other is None:
                continue
            similarity = self._similarity(snapshot, other)
            if similarity >= 0.5:
                outcome = self._outcome(records)
                if outcome is not None:
                    similar.append((similarity, outcome))
        if similar:
            weight = sum(item[0] for item in similar)
            return sum(similarity * outcome for similarity, outcome in similar) / weight, "similar-state"

        global_records = [
            item
            for (__, other_action), records in self.by_state_action.items()
            if other_action == action
            or (family == "ACTION6" and other_action.startswith("ACTION6("))
            for item in records
        ]
        global_score = self._outcome(global_records)
        if global_score is not None:
            return global_score, "episode"
        return 0.08, "exploration"

    def rank_actions(
        self,
        state_hash: str,
        snapshot: GridSnapshot,
        actions: list[str],
    ) -> list[str]:
        """Rank coordinate-expanded actions by state-conditional value."""
        order = {action: index for index, action in enumerate(actions)}
        scored = [
            (action, self.score_action(state_hash, snapshot, action))
            for action in actions
        ]
        return [
            action
            for action, __ in sorted(
                scored,
                key=lambda item: (item[1][0], -order[item[0]]),
                reverse=True,
            )
        ]

    def prompt_context(self, state_hash: str) -> str:
        lines: list[str] = []
        recent = [item for item in self.transitions if item.before_hash == state_hash][-4:]
        for item in recent:
            result = "changed" if item.changed else "no-change"
            effect = ",".join(item.object_changes[:2]) or "no-object-diff"
            lines.append(f"transition={item.action}:{result}:{effect}")
        return "\n".join(lines)
