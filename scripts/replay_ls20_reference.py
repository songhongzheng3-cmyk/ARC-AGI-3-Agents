#!/usr/bin/env python3
"""Replay a successful ls20 recording through the symbolic solver offline."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from arcengine import FrameData, FrameDataRaw, GameAction


class RecordingEnvironment:
    """Serve recorded observations while checking every requested action."""

    def __init__(self, rows: list[dict[str, Any]], initial_index: int = 1) -> None:
        if len(rows) <= initial_index:
            raise ValueError("recording does not contain an initial observation")
        self.rows = rows
        self.initial_index = initial_index
        self.cursor = initial_index
        self.observation_space = self._raw(rows[initial_index])

    @staticmethod
    def _raw(row: dict[str, Any]) -> FrameDataRaw:
        normalized = dict(row)
        action_input = dict(normalized.get("action_input") or {})
        action_id = action_input.get("id", 0)
        if isinstance(action_id, str):
            name = action_id.upper()
            action_input["id"] = (
                0 if name == "RESET" else int(name.removeprefix("ACTION"))
            )
        normalized["action_input"] = action_input
        frame = FrameData.model_validate(normalized)
        raw = FrameDataRaw(
            game_id=frame.game_id,
            state=frame.state,
            levels_completed=frame.levels_completed,
            win_levels=frame.win_levels,
            action_input=frame.action_input,
            guid=frame.guid,
            full_reset=frame.full_reset,
            available_actions=frame.available_actions,
        )
        raw.frame = [np.asarray(layer, dtype=np.int8) for layer in frame.frame]
        return raw

    def reset(self) -> FrameDataRaw:
        self.cursor = self.initial_index
        self.observation_space = self._raw(self.rows[self.cursor])
        return self.observation_space

    def step(
        self,
        action: GameAction,
        data: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> FrameDataRaw:
        del data, reasoning
        next_index = self.cursor + 1
        if next_index >= len(self.rows):
            raise RuntimeError("solver requested an action after recording ended")
        expected = self.rows[next_index]["action_input"]["id"]
        expected_name = (
            "RESET" if expected == 0 else f"ACTION{expected}"
            if isinstance(expected, int)
            else str(expected).upper()
        )
        if action.name != expected_name:
            raise RuntimeError(
                "offline action divergence at recording row "
                f"{next_index}: expected {expected_name}, got {action.name}"
            )
        self.cursor = next_index
        self.observation_space = self._raw(self.rows[self.cursor])
        return self.observation_space


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            rows.append(payload.get("data", payload))
    return rows


def load_backend(root: Path) -> tuple[type[Any], type[Any]]:
    src = (root / "src").resolve()
    sys.path.insert(0, str(src))
    agent_module = importlib.import_module("ls20.agent")
    session_module = importlib.import_module("ls20.session")
    return agent_module.Phase6Agent, session_module.MultiLevelSession


def compact_archive(archive: Any) -> dict[str, Any]:
    world = archive.world_model
    return {
        "level_zero_based": archive.level_index,
        "actions_executed": archive.result.actions_executed,
        "reason": archive.result.reason,
        "actions": list(archive.result.actions),
        "objectives": world.get("objectives", []),
        "interactions": world.get("interactions", []),
        "active_probe_count": world.get("active_probe_count"),
        "causal_experiment_history": world.get("causal_experiment_history", []),
        "option_library": world.get("option_library"),
        "relational_option_planner": world.get("relational_option_planner"),
        "temporal_updates": world.get("temporal_updates", []),
        "evidence_ordering": world.get("evidence_ordering"),
        "state_graph": world.get("state_graph"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument(
        "--backend",
        type=Path,
        default=Path(".external/ls20-solver"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    phase6_type, session_type = load_backend(args.backend)
    environment = RecordingEnvironment(load_rows(args.recording))
    backend = phase6_type(planner="astar")
    session = session_type(
        planner="astar",
        level_solver_factory=backend._solver_for_level,
    )
    budgets = {0: 100, 1: 200, 2: 250, 3: 300, 4: 400, 5: 500, 6: 600}
    result = session.solve(
        environment,
        target_levels_completed=7,
        max_total_actions=2500,
        max_actions_per_level=600,
        level_action_budgets=budgets,
    )
    report = {
        "schema_version": 1,
        "recording": str(args.recording),
        "backend": str(args.backend),
        "success": result.success,
        "reason": result.reason,
        "final_levels_completed": result.final_levels_completed,
        "total_actions": result.total_actions,
        "actions_per_level": [
            archive.result.actions_executed for archive in result.level_archives
        ],
        "levels": [compact_archive(archive) for archive in result.level_archives],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "success": result.success,
                "levels": result.final_levels_completed,
                "total_actions": result.total_actions,
                "actions_per_level": report["actions_per_level"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
