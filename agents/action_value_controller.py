"""Online action-value controller for post-bootstrap execution."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class ActionValue:
    trials: int = 0
    total_reward: float = 0.0
    last_reward: float = 0.0

    @property
    def mean(self) -> float:
        return self.total_reward / self.trials if self.trials else 0.0


class ActionValueController:
    """Choose actions using online reward estimates and UCB exploration."""

    def __init__(self, exploration: float = 0.7, max_same_action: int = 2) -> None:
        self.exploration = max(0.0, exploration)
        self.max_same_action = max(1, max_same_action)
        self.values: dict[str, ActionValue] = {}
        self.total_trials = 0
        self.last_action = ""
        self.same_action_streak = 0
        self.task_signal_observed = False

    @staticmethod
    def key(action: str, data: dict[str, object] | None = None) -> str:
        action = action.upper()
        if action == "ACTION6":
            payload = data or {}
            return f"ACTION6({int(payload.get('x', 0))},{int(payload.get('y', 0))})"
        return action

    @staticmethod
    def family(action: str) -> str:
        return re.sub(r"\([^)]*\)$", "", action).upper()

    def reset(self) -> None:
        self.values.clear()
        self.total_trials = 0
        self.last_action = ""
        self.same_action_streak = 0
        self.task_signal_observed = False

    def observe(
        self,
        action: str,
        data: dict[str, object] | None,
        reward: float,
        *,
        task_signal: bool = False,
    ) -> None:
        key = self.key(action, data)
        value = self.values.setdefault(key, ActionValue())
        value.trials += 1
        value.total_reward += float(reward)
        value.last_reward = float(reward)
        self.total_trials += 1
        self.task_signal_observed = self.task_signal_observed or task_signal
        if key == self.last_action:
            self.same_action_streak += 1
        else:
            self.last_action = key
            self.same_action_streak = 1

    def score(self, action: str) -> float:
        value = self.values.get(action)
        if value is None or value.trials == 0:
            return self.exploration
        uncertainty = math.sqrt(
            math.log(max(1, self.total_trials)) / max(1, value.trials)
        )
        return value.mean + self.exploration * uncertainty

    def choose(self, candidates: list[str]) -> str | None:
        # Do not convert ordinary animation/grid changes into an action policy.
        # A controller becomes authoritative only after observing progress that
        # is tied to the task (level advance, win, or an explicit mechanism).
        if not candidates or not self.task_signal_observed:
            return None
        ordered = list(dict.fromkeys(candidates))
        if self.same_action_streak >= self.max_same_action:
            alternatives = [item for item in ordered if item != self.last_action]
            if alternatives:
                ordered = alternatives
        return max(
            ordered,
            key=lambda item: (self.score(item), -ordered.index(item)),
        )

    def summary(self, limit: int = 6) -> str:
        ranked = sorted(self.values.items(), key=lambda item: item[1].mean, reverse=True)
        return "; ".join(
            f"{key}:mean={value.mean:.2f}/n={value.trials}"
            for key, value in ranked[:limit]
        )
