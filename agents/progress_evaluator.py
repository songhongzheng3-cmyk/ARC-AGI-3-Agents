"""Game-type detection and conservative, explainable progress rewards."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .object_extractor import GridSnapshot


class GameType(str, Enum):
    UNKNOWN = "unknown"
    NAVIGATION = "navigation"
    REMOVAL = "removal"
    INTERACTION = "interaction"
    SWITCH = "switch"
    ARRANGEMENT = "arrangement"
    SEQUENCE = "sequence"


@dataclass(frozen=True)
class TypeEvidence:
    game_type: GameType
    score: float
    reason: str


@dataclass
class RewardVector:
    """A decomposed reward; total is deliberately not hidden."""

    total: float
    progress: float
    object_change: float
    novelty: float
    state_change: float
    repeat_penalty: float
    terminal: float
    game_type: GameType
    reason: str

    def summary(self) -> str:
        return (
            f"reward={self.total:.2f} type={self.game_type.value} "
            f"progress={self.progress:.2f} object={self.object_change:.2f} "
            f"novelty={self.novelty:.2f} repeat_penalty={self.repeat_penalty:.2f} "
            f"reason={self.reason}"
        )


class GameTypeDetector:
    """Infer a coarse task family from short-term structural evidence."""

    def __init__(self, history_limit: int = 12) -> None:
        self.history: deque[dict[str, Any]] = deque(maxlen=max(1, history_limit))
        self.scores: dict[GameType, float] = defaultdict(float)

    def reset(self) -> None:
        self.history.clear()
        self.scores.clear()

    def observe(
        self,
        action: str,
        before: GridSnapshot,
        after: GridSnapshot,
        changed_cells: int,
        levels_delta: int = 0,
    ) -> None:
        before_ids = {item.object_id for item in before.objects}
        after_ids = {item.object_id for item in after.objects}
        removed = len(before_ids - after_ids)
        added = len(after_ids - before_ids)
        before_shapes = {(item.color, item.shape, item.area) for item in before.objects}
        after_shapes = {(item.color, item.shape, item.area) for item in after.objects}
        shape_changed = before_shapes != after_shapes
        action = action.upper()

        if removed > added:
            self.scores[GameType.REMOVAL] += 2.0
        if action == "ACTION6":
            self.scores[GameType.INTERACTION] += 1.0
        if shape_changed and removed == 0 and added == 0:
            self.scores[GameType.SWITCH] += 0.8
        if changed_cells and len(after.objects) == len(before.objects):
            self.scores[GameType.NAVIGATION] += 0.3
            self.scores[GameType.ARRANGEMENT] += 0.2
        if levels_delta == 0 and len(self.history) >= 2:
            recent_actions = [item["action"] for item in self.history]
            if action in recent_actions[-2:]:
                self.scores[GameType.SEQUENCE] += 0.2

        self.history.append(
            {
                "action": action,
                "changed_cells": changed_cells,
                "removed": removed,
                "added": added,
                "shape_changed": shape_changed,
            }
        )

    def classify(self) -> tuple[GameType, list[TypeEvidence]]:
        if not self.scores:
            return GameType.UNKNOWN, []
        ranked = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        top_score = ranked[0][1]
        if top_score < 1.0:
            return GameType.UNKNOWN, [
                TypeEvidence(GameType.UNKNOWN, top_score, "insufficient evidence")
            ]
        evidence = [
            TypeEvidence(kind, score, f"observed {kind.value}-like transitions")
            for kind, score in ranked[:3]
        ]
        return ranked[0][0], evidence


class RewardEvaluator:
    """Convert one transition into conservative progress signals."""

    def __init__(self) -> None:
        self.seen_states: set[str] = set()

    def reset(self) -> None:
        self.seen_states.clear()

    @staticmethod
    def _diff_cells(before: Any, after: Any) -> int:
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

    def evaluate(
        self,
        before_frame: Any,
        after_frame: Any,
        before: GridSnapshot,
        after: GridSnapshot,
        game_type: GameType,
        after_hash: str,
        action: str = "",
        repeated_action: bool = False,
    ) -> RewardVector:
        changed_cells = self._diff_cells(before_frame, after_frame)
        if action.upper() == "RESET":
            return RewardVector(
                total=0.0,
                progress=0.0,
                object_change=0.0,
                novelty=0.0,
                state_change=0.0,
                repeat_penalty=0.0,
                terminal=0.0,
                game_type=game_type,
                reason="reset",
            )
        level_delta = int(getattr(after_frame, "levels_completed", 0)) - int(
            getattr(before_frame, "levels_completed", 0)
        )
        before_ids = {item.object_id for item in before.objects}
        after_ids = {item.object_id for item in after.objects}
        removed = len(before_ids - after_ids)
        added = len(after_ids - before_ids)
        self.seen_states.add(after_hash)

        state = getattr(after_frame, "state", "")
        state_name = getattr(state, "name", str(state))
        terminal = 1000.0 if state_name == "WIN" else 0.0
        terminal += 100.0 * max(0, level_delta)
        object_change = 0.0
        reason = "ordinary-change"
        if game_type is GameType.REMOVAL and removed > added:
            object_change = 2.0 * (removed - added)
            reason = "objects-removed"
        elif game_type in (GameType.INTERACTION, GameType.SWITCH) and (removed or added):
            object_change = 1.0 * (removed + added)
            reason = "interaction-object-change"
        elif game_type is GameType.ARRANGEMENT and changed_cells:
            object_change = 0.5
            reason = "arrangement-change"
        elif game_type is GameType.NAVIGATION and changed_cells:
            object_change = 0.25
            reason = "navigation-change"

        # Raw frame novelty and avatar animation are observations, not proof of
        # task progress.  Keep them in the vector for diagnostics but do not
        # pay reward for them until a mechanism-specific evaluator exists.
        state_change = 0.0
        novelty = 0.0
        repeat_penalty = 3.0 if repeated_action and not level_delta else 0.0
        total = terminal + object_change + novelty + state_change - repeat_penalty
        if not changed_cells and not terminal:
            total -= 2.0
            reason = "no-change"
        return RewardVector(
            total=total,
            progress=terminal,
            object_change=object_change,
            novelty=novelty,
            state_change=state_change,
            repeat_penalty=repeat_penalty,
            terminal=terminal,
            game_type=game_type,
            reason=reason,
        )
