"""LLM-free agent composed from the learned structural policies."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from ..episode_memory import EpisodeMemory
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
    MAX_FALLBACK_CLICK_TRIALS = 128

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.navigation_policy = NavigationFrontierPolicy()
        self.mechanism_tracker = MechanismStateTracker(
            enable_entity_tracking=os.getenv("STRUCTURED_ENTITY_TRACKING", "0") == "1",
        )
        self.mechanism_policy = MechanismPolicy(
            enable_joint_navigation=os.getenv("STRUCTURED_JOINT_NAVIGATION", "1")
            != "0",
            enable_motion_context=os.getenv("STRUCTURED_MOTION_CONTEXT", "0") == "1",
            enable_dynamic_contact=os.getenv("STRUCTURED_DYNAMIC_CONTACT", "0") == "1",
            enable_option_compiler=os.getenv("STRUCTURED_OPTION_COMPILER", "0") == "1",
            enable_contact_effects=os.getenv("STRUCTURED_CONTACT_EFFECTS", "0") == "1",
            enable_entity_options=os.getenv("STRUCTURED_ENTITY_OPTIONS", "0") == "1",
        )
        self.episode_memory = EpisodeMemory()
        self.last_action: str | None = None
        self.fallback_counts: dict[tuple[Any, ...], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.fallback_clicks_by_level: dict[int, set[tuple[int, int]]] = defaultdict(
            set
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
        names = [name for name in self._available_names(frame) if name != "RESET"]
        if not names:
            logger.warning("No non-RESET actions available for fallback")
            return GameAction.RESET
        key = self._fallback_state_key(frame, state)
        counts = self.fallback_counts.setdefault(key, defaultdict(int))

        if "ACTION6" in names:
            state_hash = self.episode_memory.state_hash(frame)
            level = int(frame.levels_completed or 0)
            tried_clicks = set(self.episode_memory.tried_clicks(state_hash))
            tried_clicks.update(self.fallback_clicks_by_level[level])
            if len(tried_clicks) < self.MAX_FALLBACK_CLICK_TRIALS:
                candidates = self.episode_memory.click_candidates(
                    frame,
                    tried=tried_clicks,
                )
            else:
                candidates = []
            if candidates:
                point = candidates[0]
                action = GameAction.ACTION6
                action.set_data({"x": int(point["x"]), "y": int(point["y"])})
                self.fallback_clicks_by_level[level].add(
                    (int(point["x"]), int(point["y"]))
                )
                counts[f"ACTION6({point['x']},{point['y']})"] += 1
                logger.info(
                    "Structured fallback action=ACTION6(%s,%s) level=%s state=%s",
                    point["x"],
                    point["y"],
                    key[0],
                    key[1:],
                )
                return action
            if len(tried_clicks) < self.MAX_FALLBACK_CLICK_TRIALS:
                grid = (getattr(frame, "frame", None) or [[]])[-1]
                flattened = [int(value) for row in grid for value in row]
                if flattened:
                    counts_by_color: dict[int, int] = defaultdict(int)
                    for value in flattened:
                        counts_by_color[value] += 1
                    background = max(counts_by_color, key=counts_by_color.get)
                    pixels = sorted(
                        {
                            (x, y)
                            for y, row in enumerate(grid)
                            for x, value in enumerate(row)
                            if int(value) != background and (x, y) not in tried_clicks
                        },
                        key=lambda point: (point[1], point[0]),
                    )
                    if pixels:
                        x, y = pixels[0]
                        action = GameAction.ACTION6
                        action.set_data({"x": x, "y": y})
                        self.fallback_clicks_by_level[level].add((x, y))
                        return action
            names.remove("ACTION6")
            if not names:
                if "RESET" in self._available_names(frame):
                    return GameAction.RESET
                logger.warning(
                    "No untried click candidate or reset at state=%s; cycling ACTION6",
                    key,
                )
                action = GameAction.ACTION6
                action.set_data({"x": 0, "y": 0})
                return action

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
        if self.last_action == "RESET":
            self.mechanism_policy.note_reset()
        logger.info("Mechanism state %s", self.mechanism_tracker.summary(state))
        state_hash = self.episode_memory.observe_frame(latest_frame)

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
                previous_action=self.last_action,
            )
            if action is not None:
                logger.info(
                    "Mechanism policy action=%s %s",
                    action.name,
                    self.mechanism_policy.summary(),
                )
            else:
                action = self._fallback(latest_frame, state)

        if action.name not in self._available_names(latest_frame):
            logger.info(
                "Policy proposed unavailable action=%s; using fallback",
                action.name,
            )
            action = self._fallback(latest_frame, state)
        if action.name not in self._available_names(latest_frame):
            logger.warning(
                "Fallback action=%s remains unavailable; using first legal action",
                action.name,
            )
            names = self._available_names(latest_frame)
            action = GameAction.from_name(names[0]) if names else GameAction.RESET

        self.last_action = action.name
        action_data = action.action_data.model_dump(exclude={"game_id"})
        self.episode_memory.record_action(
            action.name,
            action_data,
            state_hash,
            int(latest_frame.levels_completed or 0),
        )
        return action
