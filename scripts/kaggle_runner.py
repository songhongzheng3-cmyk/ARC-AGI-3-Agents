"""Run the frozen competition baseline sequentially through the public SDK.

This file is packaged into the offline notebook. It deliberately does not
load the project's .env, query an external LLM, or read game implementation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import sys
import time
from pathlib import Path

from arc_agi import Arcade, OperationMode

from agents.templates.ls20_solver_agent import Ls20SolverAgent
from agents.templates.structured_agent import StructuredAgent

LOG = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["offline", "online"], default="online")
    parser.add_argument("--game", help="One development game; omitted on Kaggle")
    parser.add_argument("--environments-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structured-budget", type=int, default=700)
    args = parser.parse_args()
    if args.structured_budget < 1:
        parser.error("--structured-budget must be positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    os.environ.update(
        ARC_LS20_SOLVER_PATH=str(root / "backend"),
        ARC_LS20_TARGET_LEVELS="7",
        ARC_LS20_MAX_TOTAL_ACTIONS="2500",
        ARC_LS20_LEVEL_BUDGETS="100,200,250,300,400,500,600",
        ARC_LS20_RUN_DIR=str(output / "ls20"),
        RECORDINGS_DIR=str(output / "recordings"),
    )
    # Legacy Agent.main uses <= MAX_ACTIONS. Keep the actual cap at budget.
    StructuredAgent.MAX_ACTIONS = args.structured_budget - 1
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    arcade = Arcade(
        operation_mode=(
            OperationMode.OFFLINE if args.mode == "offline" else OperationMode.ONLINE
        ),
        **(
            {"environments_dir": str(args.environments_dir.resolve())}
            if args.environments_dir
            else {}
        ),
    )
    games = [entry.game_id for entry in arcade.get_environments()]
    if args.game:
        games = [g for g in games if g == args.game or g.split("-")[0] == args.game]
    if not games:
        raise RuntimeError("no matching environments returned by the public SDK")

    report = {
        "mode": args.mode,
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("arc-agi", "arcengine", "numpy")
        },
        "manifest": json.loads((root / "manifest.json").read_text()),
        "runs": [],
        "score_note": "Only Kaggle competition rerun produces an official score.",
    }
    card = arcade.open_scorecard(tags=["kaggle-checkpoint-baseline"])
    errors = 0
    try:
        # GameAction members carry mutable arguments. Serial execution prevents
        # another game's thread from changing coordinates before step().
        for game_id in games:
            cls = (
                Ls20SolverAgent if game_id.split("-")[0] == "ls20" else StructuredAgent
            )
            started = time.monotonic()
            row = {"game_id": game_id, "agent": cls.__name__}
            agent = None
            try:
                env = arcade.make(game_id, scorecard_id=card)
                if env is None:
                    raise RuntimeError(f"arcade.make({game_id!r}) returned None")
                agent = cls(
                    card_id=card,
                    game_id=game_id,
                    agent_name=cls.__name__,
                    ROOT_URL=os.getenv("ARC_BASE_URL", "http://gateway:8001/"),
                    record=True,
                    arc_env=env,
                    tags=["kaggle-checkpoint-baseline"],
                )
                agent.main()
                row.update(
                    state=agent.state.name,
                    levels_completed=agent.levels_completed,
                    actions=agent.action_counter,
                )
                result = getattr(agent, "result", None)
                if result is not None:
                    row["solver_reason"] = result.reason
                    row["solver_success"] = result.success
            except Exception as exc:
                errors += 1
                row["error"] = f"{type(exc).__name__}: {exc}"
                LOG.exception("Game execution failed: %s", game_id)
            finally:
                if agent is not None:
                    agent.cleanup()
                row["elapsed_seconds"] = round(time.monotonic() - started, 3)
                report["runs"].append(row)
                (output / "report.json").write_text(
                    json.dumps(report, indent=2, ensure_ascii=False) + "\n"
                )
                LOG.info("GAME_RESULT %s", json.dumps(row))
    finally:
        scorecard = arcade.close_scorecard(card)
        if scorecard is not None:
            (output / "scorecard.json").write_text(scorecard.model_dump_json(indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
