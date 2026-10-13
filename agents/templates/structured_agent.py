"""LLM-free agent composed from the learned structural policies."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from ..mechanism_policy import MechanismPolicy
from ..mechanism_state_tracker import MechanismStateTracker
from ..navigation_frontier_policy import NavigationFrontierPolicy

logger = logging.getLogger()


class StructuredAgent(Agent):
    """Exercise only observation-derived navigation and mechanism policies."""

    # The validated seven-level ls20 reference uses 637 actions.  Keep enough
    # headroom for online discovery while allowing shorter experiments to cap
    # the run through an environment variable.
    MAX_ACTIONS = int(os.getenv("STRUCTURED_MAX_ACTIONS", "700"))
    DIRECTION_NAMES = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.navigation_policy = NavigationFrontierPolicy()
        self.mechanism_tracker = MechanismStateTracker()
        self.mechanism_policy = MechanismPolicy()
        self.last_action: str | None = None
        self.fallback_counts: dict[tuple[Any, ...], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    @staticmethod
    def _available_names(frame: FrameData) -> list[str]:
        names: list[str] = []
        for value in frame.available_actions or []:
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

    @staticmethod
    def _fallback_state_key(frame: FrameData, state: Any) -> tuple[Any, ...]:
        """Group frames by decision-relevant state, not animated pixels.

        Resource bars and autonomous objects often change every frame.  A raw
        frame hash therefore makes every failed fallback action look like a
        brand-new state and repeatedly selects the first action.  Position and
        status describe the stable local decision context closely enough for
        bounded directional exploration.
        """
        return (
            int(frame.levels_completed or 0),
            getattr(state, "player_position", None),
            getattr(state, "status_signature", ""),
            getattr(state, "status_color", None),
        )

    def _fallback(self, frame: FrameData, state: Any) -> GameAction:
        names = [
            name
            for name in self._available_names(frame)
            if name in self.DIRECTION_NAMES
        ]
        if not names:
            return GameAction.RESET
        key = self._fallback_state_key(frame, state)
        counts = self.fallback_counts.setdefault(key, defaultdict(int))
        selected = min(names, key=lambda name: (counts[name], names.index(name)))
        counts[selected] += 1
        logger.info(
            "Structured fallback action=%s level=%s state=%s counts=%s",
            selected,
            key[0],
            key[1:],
            dict(counts),
        )
        return GameAction.from_name(selected)

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> GameAction:
        if latest_frame.state in {GameState.NOT_PLAYED, GameState.GAME_OVER}:
            self.last_action = GameAction.RESET.name
            return GameAction.RESET

        state = self.mechanism_tracker.observe(
            latest_frame,
            action=self.last_action,
        )
        logger.info("Mechanism state %s", self.mechanism_tracker.summary(state))

        action = self.navigation_policy.choose(latest_frame)
        if action is not None:
            logger.info(
                "Navigation frontier action=%s %s",
                action.name,
                self.navigation_policy.summary(),
            )
        else:
            action = self.mechanism_policy.choose(
                state,
                latest_frame.available_actions,
            )
            if action is not None:
                logger.info(
                    "Mechanism policy action=%s %s",
                    action.name,
                    self.mechanism_policy.summary(),
                )
            else:
                action = self._fallback(latest_frame, state)

        self.last_action = action.name
        return action
