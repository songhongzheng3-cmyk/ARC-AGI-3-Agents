#!/usr/bin/env python3
"""Verify the unmodified reference backend on the real local engine twice."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc_agi import Arcade, OperationMode  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from agents.templates.ls20_solver_agent import Ls20SolverAgent  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def git_info(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args]).decode()

    return {
        "head": git("rev-parse", "HEAD").strip(),
        "status": git("status", "--short").splitlines(),
    }


def snapshot_sources(output: Path, backend: Path) -> dict[str, str]:
    files = sorted(
        set(ROOT.glob("agents/**/*.py"))
        | set(ROOT.glob("scripts/**/*.py"))
        | {ROOT / "pyproject.toml", ROOT / "uv.lock"}
    )
    files.extend(sorted((backend / "src").rglob("*.py")))
    hashes = {}
    with zipfile.ZipFile(output / "solver_sources.zip", "w") as archive:
        for path in files:
            if "__pycache__" in path.parts:
                continue
            name = str(path.relative_to(ROOT))
            data = path.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
    return hashes


def normalized_recording(agent: Ls20SolverAgent) -> list[dict[str, Any]]:
    # Session GUIDs and timestamps are not game-state differences.
    normalized = []
    for row in agent.recorder.get():
        frame = row["data"]
        action = frame.get("action_input") or {}
        normalized.append(
            {
                "action": action.get("id"),
                "coordinates": {
                    key: value
                    for key, value in (action.get("data") or {}).items()
                    if key in {"x", "y"}
                },
                "state": frame.get("state"),
                "levels_completed": frame.get("levels_completed"),
                "win_levels": frame.get("win_levels"),
                "available_actions": frame.get("available_actions"),
                "frame_sha256": hashlib.sha256(
                    json.dumps(frame.get("frame"), separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ls20-9607627b")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error("at least two runs are required for baseline verification")

    os.chdir(ROOT)
    load_dotenv(ROOT / ".env.example")
    load_dotenv(ROOT / ".env", override=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "sampling_runs" / "baseline_verification" / stamp
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    backend = ROOT / ".external" / "ls20-solver"
    # Override inherited configuration for this experiment, after dotenv loads.
    os.environ.update(
        OPERATION_MODE="offline",
        ARC_LS20_SOLVER_PATH=str(backend),
        ARC_LS20_TARGET_LEVELS="7",
        ARC_LS20_MAX_TOTAL_ACTIONS="2500",
        ARC_LS20_LEVEL_BUDGETS="100,200,250,300,400,500,600",
        ARC_LS20_RUN_DIR=str(output),
        RECORDINGS_DIR=str(output / "recordings"),
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "project": git_info(ROOT),
        "backend": git_info(backend),
        "source_sha256": snapshot_sources(output, backend),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("arc-agi", "arcengine", "numpy", "pydantic")
        },
        "game": args.game,
        "seed": args.seed,
        "repeat": args.repeat,
        "environment": "real_local_engine",
        "agent": "Ls20SolverAgent / Phase6Agent native MultiLevelSession",
        "max_total_actions": 2500,
        "level_action_budgets": [100, 200, 250, 300, 400, 500, 600],
        "score_note": "Local SDK score is archived, not asserted as current Kaggle score.",
        "prediction_error_rate": None,
        "prediction_error_note": "Not uniformly instrumented; full world-model reports retained.",
    }
    write_json(output / "manifest.json", manifest)
    report: dict[str, Any] = {"runs": [], "accepted": False, "comparison": None}
    normalized_runs = []
    print(f"RESULT_DIRECTORY {output}", flush=True)
    for run in range(1, args.repeat + 1):
        arcade = Arcade(operation_mode=OperationMode.OFFLINE)
        card = arcade.open_scorecard(tags=["baseline-verification", f"run-{run}"])
        agent = None
        try:
            env = arcade.make(args.game, seed=args.seed, scorecard_id=card)
            if env is None:
                raise RuntimeError(
                    "local environment missing; download via SDK normal mode first"
                )
            if run == 1:
                write_json(
                    output / "game_metadata.json",
                    env.environment_info.model_dump(mode="json"),
                )
            agent = Ls20SolverAgent(
                card_id=card,
                game_id=args.game,
                agent_name="ls20solveragent",
                ROOT_URL="local",
                record=True,
                arc_env=env,
            )
            agent.main()
            result = agent.result
            normalized = normalized_recording(agent)
            normalized_runs.append(normalized)
            write_json(output / f"run-{run}.world_model.json", result.to_dict())
            write_json(output / f"run-{run}.normalized_trace.json", normalized)
            detail = {
                "run": run,
                "success": result.success,
                "reason": result.reason,
                "levels_completed": result.final_levels_completed,
                "total_actions": result.total_actions,
                "actions_per_level": [
                    item.result.actions_executed for item in result.level_archives
                ],
                "final_state": env.observation_space.state.name,
                "resets_after_initial": sum(
                    row["action"] in {0, "RESET"} for row in normalized[1:]
                ),
                "elapsed_seconds": result.elapsed_seconds,
                "recording": str(agent.recorder.filename),
            }
        except Exception as exc:
            detail = {"run": run, "success": False, "runtime_error": str(exc)}
            (output / f"run-{run}.error.txt").write_text(traceback.format_exc())
        finally:
            card_result = arcade.close_scorecard(card)
            if card_result is not None:
                write_json(
                    output / f"run-{run}.local_scorecard.json",
                    card_result.model_dump(mode="json"),
                )
        report["runs"].append(detail)
        write_json(output / "report.json", report)
        print(json.dumps(detail, ensure_ascii=False), flush=True)
        if not detail["success"]:
            print(
                "STOP: baseline failed; inspect evidence before another full run.",
                flush=True,
            )
            return 1

    first = normalized_runs[0]
    differences = []
    for run, other in enumerate(normalized_runs[1:], 2):
        index = next(
            (
                index
                for index, pair in enumerate(zip(first, other))
                if pair[0] != pair[1]
            ),
            min(len(first), len(other)) if len(first) != len(other) else None,
        )
        differences.append({"run": run, "first_difference_row": index})
    report["comparison"] = differences
    report["accepted"] = all(
        row["first_difference_row"] is None for row in differences
    ) and all(
        run["levels_completed"] == 7 and run["final_state"] == "WIN"
        for run in report["runs"]
    )
    report["evidence_boundary"] = (
        "Same seed/layout repetition; no cross-game generalization claim."
    )
    write_json(output / "report.json", report)
    print(
        json.dumps({"accepted": report["accepted"], "comparison": differences}),
        flush=True,
    )
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
