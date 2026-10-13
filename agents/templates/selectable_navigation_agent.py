"""Bounded first-level experimental agent using the unchanged reference BFS."""

from __future__ import annotations

import time
from typing import Any

from arcengine import GameAction, GameState

from ..agent import Agent
from ..selectable_navigation_model import MultiActorState, SelectableNavigationModel


class SelectableNavigationAgent(Agent):
    MAX_ACTIONS = 200

    def __init__(
        self, *args: Any, model: SelectableNavigationModel, search: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.search = search
        self.expected: MultiActorState | None = None
        self.plans = []
        self.transitions = []
        self.result = {}

    def is_done(self, frames, latest_frame):
        return (
            latest_frame.state in (GameState.WIN, GameState.GAME_OVER)
            or latest_frame.levels_completed > self.initial_level
        )

    def choose_action(self, frames, latest_frame):
        scene = self.model.binding.parse(latest_frame)
        state = self.model.state(scene)
        result = self.search(
            state, self.model.is_goal, self.model.successors, max_expansions=50000
        )
        self.plans.append(result.to_dict())
        if not result.found:
            raise ValueError("no_plan_under_current_model")
        if not result.actions:
            raise ValueError("goal_hypothesis_not_confirmed_by_environment")
        token = result.actions[0]
        self.expected = result.states[1]
        if token.startswith("select:"):
            index = int(token.split(":")[1])
            x, y = self.model.binding.cell_pixel(state.positions[index])
            action = GameAction.ACTION6
            action.set_data({"x": x, "y": y})
        else:
            action = GameAction.from_id(int(token.split(":")[1]))
        return action

    def main(self):
        self.timer = time.time()
        frame = self._convert_raw_frame_data(self.arc_env.observation_space)
        self.initial_level = frame.levels_completed
        self.append_frame(frame)
        reason = "action_budget_exhausted"
        try:
            for _ in range(self.MAX_ACTIONS):
                if self.is_done(self.frames, frame):
                    reason = (
                        "level_completed"
                        if frame.levels_completed > self.initial_level
                        else frame.state.name.lower()
                    )
                    break
                before = self.model.binding.parse(frame)
                action = self.choose_action(self.frames, frame)
                if action.value not in frame.available_actions:
                    raise ValueError("planned_action_is_not_legal")
                data = (
                    {"x": action.action_data.x, "y": action.action_data.y}
                    if action is GameAction.ACTION6
                    else {}
                )
                raw = self.arc_env.step(
                    action,
                    data=data,
                    reasoning={
                        "source": "selectable-navigation",
                        "hypothesis": "outline-occupancy",
                        "prediction": self.expected.to_dict(),
                    },
                )
                if raw is None:
                    raise RuntimeError("no observation returned")
                frame = self._convert_raw_frame_data(raw)
                self.action_counter += 1
                self.append_frame(frame)
                terminal = self.is_done(self.frames, frame)
                row = {
                    "action": action.value,
                    "data": data,
                    "expected": self.expected.to_dict(),
                    "levels_completed": frame.levels_completed,
                    "state": frame.state.name,
                    "prediction_checked": False,
                }
                if not terminal:
                    after = self.model.binding.parse(frame)
                    actual = self.model.state(after)
                    row.update(
                        actual=actual.to_dict(),
                        prediction_checked=True,
                        prediction_matches=actual == self.expected,
                    )
                    self.transitions.append(row)
                    self.model.learn(before, action.value, after)
                    if actual != self.expected:
                        reason = "prediction_mismatch_stop"
                        break
                else:
                    self.transitions.append(row)
                    reason = (
                        "level_completed"
                        if frame.levels_completed > self.initial_level
                        else frame.state.name.lower()
                    )
                    break
        except ValueError as exc:
            reason = str(exc)
        finally:
            self.result = {
                "reason": reason,
                "levels_completed": frame.levels_completed,
                "level_advanced": frame.levels_completed > self.initial_level,
                "final_state": frame.state.name,
                "actions": self.action_counter,
                "elapsed_seconds": time.time() - self.timer,
                "transitions": self.transitions,
                "plans": self.plans,
                "model": self.model.to_dict(),
                "recording": str(self.recorder.filename),
            }
            self.cleanup()
