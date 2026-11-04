"""Run the frozen agent and persist diagnostics before interacting with the SDK."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)
SOURCE_ROOT = Path(__file__).resolve().parent


def write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def public_data(data: Any) -> Any:
    """Do not persist SDK scorecard authentication fields."""
    if isinstance(data, dict):
        return {
            key: public_data(value)
            for key, value in data.items()
            if key.lower() not in {"api_key", "arc_api_key", "authorization"}
        }
    if isinstance(data, list):
        return [public_data(value) for value in data]
    return data


def observation(raw: Any) -> dict:
    if raw is None:
        return {"missing": True}
    grids = [grid.tolist() if hasattr(grid, "tolist") else grid for grid in raw.frame]
    return {
        "state": raw.state.name,
        "levels_completed": raw.levels_completed,
        "win_levels": raw.win_levels,
        "available_actions": [
            getattr(action, "value", action) for action in raw.available_actions or []
        ],
        "frame_sha256": hashlib.sha256(json.dumps(grids).encode()).hexdigest(),
    }


class ObservedEnvironment:
    """Observe actual SDK requests, including retries, without changing actions."""

    def __init__(self, env: Any, row: dict, event: Any, save: Any) -> None:
        self.env, self.row, self.event, self.save = env, row, event, save
        self.counts: Counter = Counter()
        self.last = observation(env.observation_space)
        self.row.update(
            initial_observation=self.last,
            step_requests=0,
            responses=0,
            missing_responses=0,
            unchanged_frames=0,
            max_levels_completed=self.last.get("levels_completed", 0),
            action_counts={},
        )
        self.event("initial_observation", game_id=row["game_id"], observation=self.last)
        # Agent's frame recorder starts after the first step.
        raw = env.observation_space
        if raw is not None:
            self.event(
                "initial_frame",
                game_id=row["game_id"],
                frame=[g.tolist() if hasattr(g, "tolist") else g for g in raw.frame],
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> Any:
        row = self.row
        row["step_requests"] += 1
        self.counts[action.name] += 1
        row["action_counts"] = dict(self.counts)
        request = {
            "game_id": row["game_id"],
            "request": row["step_requests"],
            "action": action.name,
            "data": data,
        }
        self.event("action_requested", **request, before=self.last)
        self.save()
        started = time.monotonic()
        try:
            raw = self.env.step(action, data=data, reasoning=reasoning)
        except Exception as exc:
            self.event("action_error", **request, error=f"{type(exc).__name__}: {exc}")
            raise
        after = observation(raw)
        if raw is None:
            row["missing_responses"] += 1
        else:
            row["responses"] += 1
            row["max_levels_completed"] = max(
                row["max_levels_completed"],
                raw.levels_completed,
            )
            row["unchanged_frames"] += int(
                self.last.get("frame_sha256") == after["frame_sha256"]
            )
        self.event(
            "action_result",
            **request,
            after=after,
            elapsed_seconds=round(time.monotonic() - started, 4),
        )
        self.last = after
        row["last_observation"] = after
        self.save()
        return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["offline", "online"], default="online")
    parser.add_argument("--game", help="One development game; omitted on Kaggle")
    parser.add_argument("--environments-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structured-budget", type=int, default=700)
    args = parser.parse_args(argv)
    if args.structured_budget < 1:
        parser.error("--structured-budget must be positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    report: dict = {
        "schema_version": 2,
        "status": "starting",
        "stage": "imports",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
        "runs": [],
        "structured_budget": args.structured_budget,
        "competition_rerun": bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN")),
        "score_note": "Only Kaggle competition rerun produces an official score.",
    }

    def save() -> None:
        write_json(output / "report.json", report)

    def event(kind: str, **fields: Any) -> None:
        with (output / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event": kind,
                        **fields,
                    }
                )
                + "\n"
            )

    def stage(name: str) -> None:
        report["stage"] = name
        save()
        event("stage", stage=name)
        LOG.info("EVAL_STAGE %s", name)

    save()
    card, arcade = None, None
    errors = 0
    try:
        from arc_agi import Arcade, OperationMode

        from agents.templates.structured_agent import StructuredAgent

        for package in ("arc-agi", "arcengine", "numpy"):
            report["packages"][package] = importlib.metadata.version(package)
        manifest = json.loads((SOURCE_ROOT / "manifest.json").read_text())
        report["manifest"] = manifest
        if manifest["agents"].get("ls20") == "Ls20SolverAgent":
            from agents.templates.ls20_solver_agent import Ls20SolverAgent

            os.environ.update(
                ARC_LS20_SOLVER_PATH=str(SOURCE_ROOT / "backend"),
                ARC_LS20_TARGET_LEVELS="7",
                ARC_LS20_MAX_TOTAL_ACTIONS="2500",
                ARC_LS20_LEVEL_BUDGETS="100,200,250,300,400,500,600",
                ARC_LS20_RUN_DIR=str(output / "ls20"),
            )
        else:
            Ls20SolverAgent = StructuredAgent
        os.environ["RECORDINGS_DIR"] = str(output / "recordings")
        StructuredAgent.MAX_ACTIONS = args.structured_budget - 1
        stage("discovering_games")
        arcade = Arcade(
            operation_mode=(
                OperationMode.OFFLINE
                if args.mode == "offline"
                else OperationMode.ONLINE
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
        report["games"] = games
        if not games:
            raise RuntimeError("no matching environments returned by the public SDK")
        stage("opening_scorecard")
        card = arcade.open_scorecard(tags=["kaggle-checkpoint-baseline"])
        if not card:
            raise RuntimeError("SDK did not return a scorecard id")
        report["status"] = "running"
        stage("running_games")
        # GameAction members carry mutable arguments: keep games serial.
        for game_id in games:
            cls = (
                Ls20SolverAgent if game_id.split("-")[0] == "ls20" else StructuredAgent
            )
            started = time.monotonic()
            row: dict = {
                "game_id": game_id,
                "agent": cls.__name__,
                "status": "starting",
                "stage": "creating_environment",
                "actions": 0,
            }
            report["runs"].append(row)
            event("game_start", game_id=game_id, agent=cls.__name__)
            save()
            agent = None
            try:
                env = arcade.make(game_id, scorecard_id=card)
                if env is None:
                    raise RuntimeError(f"arcade.make({game_id!r}) returned None")
                env = ObservedEnvironment(env, row, event, save)
                if row["initial_observation"].get("missing"):
                    raise RuntimeError("SDK returned no initial observation")
                row["stage"] = "constructing_agent"
                save()
                agent = cls(
                    card_id=card,
                    game_id=game_id,
                    agent_name=cls.__name__,
                    ROOT_URL=os.getenv("ARC_BASE_URL", "http://gateway:8001/"),
                    record=True,
                    arc_env=env,
                    tags=["kaggle-checkpoint-baseline"],
                )
                row.update(status="running", stage="agent_main")
                save()
                agent.main()
                row.update(
                    status="completed",
                    stop_reason=(
                        "win"
                        if agent.state.name == "WIN"
                        else "action_budget"
                        if agent.action_counter >= cls.MAX_ACTIONS + 1
                        else "agent_stopped"
                    ),
                )
                result = getattr(agent, "result", None)
                if result is not None:
                    row.update(
                        solver_reason=result.reason, solver_success=result.success
                    )
            except Exception as exc:
                errors += 1
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                LOG.exception("Game execution failed: %s", game_id)
            finally:
                if agent is not None:
                    row.update(
                        state=agent.state.name,
                        levels_completed=agent.levels_completed,
                        actions=agent.action_counter,
                    )
                    policy = getattr(agent, "mechanism_policy", None)
                    if policy is not None and policy.enable_entity_options:
                        row["entity_options"] = {
                            "completed_levels": policy.entity_option_archives,
                            "current_level": policy.level,
                            **policy.entity_options.diagnostics(),
                        }
                    if policy is not None and policy.enable_contact_effects:
                        row["contact_effects"] = {
                            "completed_levels": policy.contact_effect_archives,
                            "current_level": policy.level,
                            **policy.contact_effects.diagnostics(),
                        }
                    if policy is not None and policy.enable_option_compiler:
                        row["option_compiler"] = {
                            "completed_levels": policy.option_archives,
                            "current_level": policy.level,
                            **policy.option_controller.diagnostics(),
                        }
                    try:
                        agent.cleanup()
                    except Exception as exc:
                        errors += 1
                        row.update(status="failed", cleanup_error=str(exc))
                        LOG.exception("Agent cleanup failed: %s", game_id)
                row["elapsed_seconds"] = round(time.monotonic() - started, 3)
                save()
                event("game_result", **row)
                LOG.info("GAME_RESULT %s", json.dumps(row))
    except Exception as exc:
        errors += 1
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["failed_stage"] = report["stage"]
        event("run_error", stage=report["stage"], error=report["error"])
        LOG.exception("Evaluation failed at %s", report["stage"])
    finally:
        if arcade is not None and card:
            stage("closing_scorecard")
            try:
                scorecard = arcade.close_scorecard(card)
                if scorecard is None:
                    raise RuntimeError("SDK returned no closed scorecard")
                write_json(
                    output / "scorecard.json",
                    public_data(json.loads(scorecard.model_dump_json())),
                )
                report["scorecard_closed"] = True
            except Exception as exc:
                errors += 1
                report["scorecard_error"] = f"{type(exc).__name__}: {exc}"
                LOG.exception("Scorecard close failed")
        report.update(
            status="failed" if errors else "completed",
            errors=errors,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        stage("finished")
        LOG.info(
            "EVAL_RESULT status=%s games=%d errors=%d",
            report["status"],
            len(report["runs"]),
            errors,
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
