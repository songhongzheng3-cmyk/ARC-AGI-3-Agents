"""Adapter for running an external observation-only ls20 solver backend."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger()


class _ObservedEnvironment:
    """Expose the existing observation and mirror backend steps to callbacks."""

    def __init__(
        self,
        env: Any,
        initial_observation: Any,
        on_observation: Callable[[Any], None],
    ) -> None:
        self._env = env
        self._initial_observation = initial_observation
        self._on_observation = on_observation
        self._first_reset = True

    def reset(self) -> Any:
        if self._first_reset:
            self._first_reset = False
            observation = self._initial_observation
        else:
            observation = self._env.reset()
        self._on_observation(observation)
        return observation

    def step(self, *args: Any, **kwargs: Any) -> Any:
        observation = self._env.step(*args, **kwargs)
        self._on_observation(observation)
        return observation

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class Ls20SolverAgent(Agent):
    """Run a full-game symbolic solver through the existing Swarm lifecycle.

    The verified ls20 solver owns a multi-level planning loop rather than a
    single-frame ``choose_action`` method.  This adapter deliberately invokes
    that native lifecycle instead of weakening it into the legacy Agent loop.

    The backend remains an external checkout.  Set ``ARC_LS20_SOLVER_PATH`` to
    its repository root; the checkout must contain ``src/ls20/agent.py``.
    """

    MAX_ACTIONS = 2500
    DEFAULT_LEVEL_BUDGETS = (100, 200, 250, 300, 400, 500, 600)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.result: Any | None = None
        self._observed_level: int | None = None

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    @staticmethod
    def _backend_src() -> Path:
        configured = os.getenv("ARC_LS20_SOLVER_PATH", "").strip()
        if not configured:
            raise RuntimeError(
                "ARC_LS20_SOLVER_PATH is required for ls20solveragent; "
                "point it at the external solver repository root"
            )
        root = Path(configured).expanduser().resolve()
        src = root / "src"
        required = src / "ls20" / "agent.py"
        if not required.is_file():
            raise RuntimeError(
                f"invalid ARC_LS20_SOLVER_PATH={root}: missing {required}"
            )
        return src

    @classmethod
    def _load_backend(cls) -> tuple[type[Any], type[Any]]:
        src = cls._backend_src()
        src_text = str(src)
        if src_text not in sys.path:
            sys.path.insert(0, src_text)

        agent_module = importlib.import_module("ls20.agent")
        session_module = importlib.import_module("ls20.session")
        module_path = Path(agent_module.__file__ or "").resolve()
        if src not in module_path.parents:
            raise RuntimeError(
                "a different 'ls20' package is already imported from "
                f"{module_path}; expected backend under {src}"
            )
        return agent_module.Phase6Agent, session_module.MultiLevelSession

    @classmethod
    def _level_budgets(cls) -> dict[int, int]:
        raw = os.getenv("ARC_LS20_LEVEL_BUDGETS", "").strip()
        if not raw:
            values = cls.DEFAULT_LEVEL_BUDGETS
        else:
            try:
                values = tuple(int(item.strip()) for item in raw.split(","))
            except ValueError as exc:
                raise RuntimeError(
                    "ARC_LS20_LEVEL_BUDGETS must be comma-separated integers"
                ) from exc
            if len(values) != 7 or any(value <= 0 for value in values):
                raise RuntimeError(
                    "ARC_LS20_LEVEL_BUDGETS must contain seven positive integers"
                )
        return dict(enumerate(values))

    def main(self) -> None:
        """Execute the backend's multi-level closed-loop solver."""
        if not self.game_id.startswith("ls20"):
            raise RuntimeError(
                f"Ls20SolverAgent only supports ls20 games, got {self.game_id}"
            )

        self.timer = time.time()
        try:
            phase6_agent, session_type = self._load_backend()
            budgets = self._level_budgets()
            target = int(os.getenv("ARC_LS20_TARGET_LEVELS", "7"))
            max_total = int(os.getenv("ARC_LS20_MAX_TOTAL_ACTIONS", "2500"))
            if not 1 <= target <= 7:
                raise RuntimeError("ARC_LS20_TARGET_LEVELS must be between 1 and 7")
            if max_total <= 0:
                raise RuntimeError("ARC_LS20_MAX_TOTAL_ACTIONS must be positive")

            backend = phase6_agent(planner="astar")
            session = session_type(
                planner="astar",
                level_solver_factory=backend._solver_for_level,
            )
            initial_observation = getattr(self.arc_env, "observation_space", None)
            if initial_observation is None:
                raise RuntimeError("ls20 environment has no initial observation")
            observed_env = _ObservedEnvironment(
                self.arc_env,
                initial_observation,
                self._record_backend_observation,
            )
            self.result = session.solve(
                observed_env,
                target_levels_completed=target,
                max_total_actions=max_total,
                max_actions_per_level=max(budgets.values()),
                level_action_budgets=budgets,
            )
            self.action_counter = int(self.result.total_actions)
            summary_path = self._write_run_summary()

            logger.info(
                "ls20 solver finished success=%s reason=%s levels=%s/%s "
                "actions=%s per_level=%s",
                self.result.success,
                self.result.reason,
                self.result.final_levels_completed,
                self.result.target_levels_completed,
                self.result.total_actions,
                [
                    archive.result.actions_executed
                    for archive in self.result.level_archives
                ],
            )
            logger.info("ls20 solver summary saved to %s", summary_path)
        finally:
            self.cleanup()

    def _record_backend_observation(self, raw: Any) -> None:
        frame = self._convert_raw_frame_data(raw)
        if frame.action_input.id is not GameAction.RESET:
            self.action_counter += 1
        level = int(frame.levels_completed or 0)
        if self._observed_level is None:
            self._observed_level = level
            logger.info("ls20 solver started at level=%s", level)
        elif level > self._observed_level:
            logger.info(
                "ls20 solver advanced level=%s -> %s observed_actions=%s",
                self._observed_level,
                level,
                self.action_counter,
            )
            self._observed_level = level
        self.append_frame(frame)

    def _write_run_summary(self) -> Path:
        if self.result is None:
            raise RuntimeError("cannot write summary before solver completion")
        output_dir = Path(
            os.getenv("ARC_LS20_RUN_DIR", "sampling_runs/solver_baselines")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = output_dir / f"{self.game_id}.{stamp}.summary.json"
        archives = list(self.result.level_archives)
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "game_id": self.game_id,
            "scorecard_id": self.card_id,
            "guid": self.guid,
            "backend": "Phase6Agent/GenericMechanismSolver",
            "success": bool(self.result.success),
            "reason": self.result.reason,
            "initial_levels_completed": self.result.initial_levels_completed,
            "final_levels_completed": self.result.final_levels_completed,
            "target_levels_completed": self.result.target_levels_completed,
            "total_actions": self.result.total_actions,
            "elapsed_seconds": self.result.elapsed_seconds,
            "actions_per_level": [
                archive.result.actions_executed for archive in archives
            ],
            "level_reasons": [archive.result.reason for archive in archives],
            "solver_types": [
                archive.world_model.get("solver_type") for archive in archives
            ],
            "recording": (
                str(self.recorder.filename) if hasattr(self, "recorder") else None
            ),
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> GameAction:
        raise RuntimeError(
            "Ls20SolverAgent uses its full-game main() loop, not choose_action()"
        )
