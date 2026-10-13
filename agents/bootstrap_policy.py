"""Deterministic, low-budget exploration policy for a new ARC level."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arcengine import GameAction, GameState


@dataclass
class BootstrapStats:
    trials: int = 0
    changed: int = 0
    last_changed: bool = False
    best_score_delta: int = 0

    @property
    def mean_reward(self) -> float:
        if not self.trials:
            return 0.0
        return (self.changed * 0.2 + self.best_score_delta * 50) / self.trials


@dataclass
class BootstrapPolicy:
    """Cover actions and object candidates before delegating to an LLM.

    The policy is intentionally deterministic.  It gathers a small amount of
    comparable evidence instead of allowing a model to repeat one preferred
    action during the cold-start phase.
    """

    min_steps: int = 6
    max_steps: int = 14
    action6_trials: int = 2
    short_sequence_limit: int = 2
    no_information_limit: int = 3
    enabled: bool = True
    level: int | None = None
    steps: int = 0
    no_information_steps: int = 0
    covered: set[str] = field(default_factory=set)
    click_trials: set[tuple[int, int]] = field(default_factory=set)
    stats: dict[str, BootstrapStats] = field(default_factory=dict)

    def reset(self, level: int) -> None:
        self.level = level
        self.steps = 0
        self.no_information_steps = 0
        self.covered.clear()
        self.click_trials.clear()
        self.stats.clear()

    def observe(
        self,
        level: int,
        transition: Any | None,
    ) -> None:
        if self.level is None or level != self.level:
            self.reset(level)
        if transition is None:
            return
        if transition.action.upper() == "RESET":
            # RESET initializes the episode; it is not an exploration trial.
            return
        key = self._action_key(transition.action, transition.data)
        result = self.stats.setdefault(key, BootstrapStats())
        result.trials += 1
        result.changed += int(transition.changed)
        result.last_changed = bool(transition.changed)
        result.best_score_delta = max(result.best_score_delta, int(transition.score_delta))
        self.covered.add(key)
        self.steps += 1
        if transition.changed or transition.score_delta > 0 or transition.object_changes:
            self.no_information_steps = 0
        else:
            self.no_information_steps += 1

    @staticmethod
    def _action_key(action: str, data: dict[str, Any] | None = None) -> str:
        if action.upper() == "ACTION6":
            payload = data or {}
            return f"ACTION6({int(payload.get('x', 0))},{int(payload.get('y', 0))})"
        return action.upper()

    @staticmethod
    def _available_names(available: list[Any]) -> list[str]:
        names: list[str] = []
        for value in available:
            if isinstance(value, GameAction):
                name = value.name
            else:
                try:
                    name = GameAction.from_id(int(value)).name
                except (KeyError, TypeError, ValueError):
                    name = str(value).upper()
            if name not in names:
                names.append(name)
        return names

    def should_continue(self, state: GameState | Any, level: int) -> bool:
        if not self.enabled or state is GameState.WIN:
            return False
        if self.level is None or level != self.level:
            self.reset(level)
        if self.steps >= self.max_steps:
            return False
        if self.steps >= self.min_steps and self.no_information_steps >= self.no_information_limit:
            return False
        return True

    def choose(
        self,
        available: list[Any],
        candidates: list[dict[str, Any]],
        state: GameState | Any,
        level: int,
    ) -> GameAction | None:
        """Return the next bootstrap action, or None to delegate to the LLM."""
        if not self.should_continue(state, level):
            return None
        names = self._available_names(available)

        # First cover each simple action once in environment order.
        for name in names:
            if name in ("RESET", "ACTION6") or name in self.covered:
                continue
            action = GameAction.from_name(name)
            action.set_data({})
            return action

        # Then probe a small number of object-derived click points.
        clicks = [
            (int(item["x"]), int(item["y"]))
            for item in candidates
            if (int(item["x"]), int(item["y"])) not in self.click_trials
        ]
        if "ACTION6" in names and clicks and len(self.click_trials) < self.action6_trials:
            x, y = clicks[0]
            self.click_trials.add((x, y))
            action = GameAction.ACTION6
            action.set_data({"x": x, "y": y})
            return action

        # If coverage is complete but information is still arriving, repeat
        # the best-changing simple action only up to a short sequence limit.
        eligible = [name for name in names if name not in ("RESET", "ACTION6")]
        if eligible and self.steps < self.min_steps:
            best = max(
                eligible,
                key=lambda name: (
                    self.stats.get(name, BootstrapStats()).mean_reward,
                    -self.stats.get(name, BootstrapStats()).trials,
                ),
            )
            stats = self.stats.get(best, BootstrapStats())
            if stats.trials < self.short_sequence_limit:
                action = GameAction.from_name(best)
                action.set_data({})
                return action
        return None
