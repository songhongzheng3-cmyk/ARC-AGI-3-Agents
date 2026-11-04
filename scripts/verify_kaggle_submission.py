"""Exercise the built notebook against SDK-hosted public games over real HTTP.

This validates the source bundle and Phase B entry point, not Kaggle's private
sidecar, hidden games, wheel installation, or official scoring implementation.
Run using the Python/SDK environment you want to validate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from kaggle_runner import public_data, write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=ROOT / ".external/kaggle-starter/notebooks-owned/submission.ipynb",
    )
    parser.add_argument(
        "--environments-dir", type=Path, default=ROOT / "environment_files"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--game", help="Omit to test every locally cached public game")
    parser.add_argument("--budget", type=int, default=700)
    parser.add_argument(
        "--transport", choices=["offline", "gateway", "both"], default="both"
    )
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be positive")
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        runs_root = ROOT / "sampling_runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        prefix = "kaggle_verify_" + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ_"
        )
        output = Path(tempfile.mkdtemp(prefix=prefix, dir=runs_root)).resolve()
    notebook = args.notebook.resolve()
    print(f"Diagnostics: {output}", flush=True)
    summary: dict = {
        "python": platform.python_version(),
        "notebook": str(notebook),
        "status": "running",
        "checks": [],
        "experiment_environment": {
            name: os.environ[name]
            for name in (
                "STRUCTURED_JOINT_NAVIGATION",
                "STRUCTURED_MOTION_CONTEXT",
                "STRUCTURED_DYNAMIC_CONTACT",
                "STRUCTURED_OPTION_COMPILER",
                "STRUCTURED_ENTITY_TRACKING",
                "STRUCTURED_CONTACT_EFFECTS",
                "STRUCTURED_ENTITY_OPTIONS",
            )
            if name in os.environ
        },
        "limitation": "SDK public-game HTTP check; not the Kaggle image or hidden evaluation.",
    }
    write_json(output / "verification.json", summary)
    try:
        snapshot = output / "submission.ipynb"
        snapshot.write_bytes(notebook.read_bytes())
        summary.update(
            notebook_snapshot=str(snapshot),
            notebook_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        )
        with tempfile.TemporaryDirectory(prefix="arc-notebook-check-") as staging:
            # Disable inherited remote/dev settings. Both subprocesses run outside
            # the project so they import only the notebook's packaged source.
            env = dict(
                os.environ,
                ARC_API_KEY="test-key-123",
                OPERATION_MODE="offline",
                ENVIRONMENTS_DIR=str(args.environments_dir.resolve()),
                RECORDINGS_DIR=str(output / "server_recordings"),
                PYTHONPATH="",
                PYTHONUNBUFFERED="1",
                TMPDIR=staging,
                KAGGLE_IS_COMPETITION_RERUN="1",
                ARC_EVAL_BUDGET=str(args.budget),
            )
            env.pop("ARC_EVAL_GAME", None)
            if args.game:
                env["ARC_EVAL_GAME"] = args.game
            for transport in (
                ["offline", "gateway"] if args.transport == "both" else [args.transport]
            ):
                destination = output / transport
                destination.mkdir()
                child_env = dict(env, ARC_EVAL_OUTPUT=str(destination))
                if transport == "offline":
                    child_env.pop("KAGGLE_IS_COMPETITION_RERUN", None)
                server, worker = None, None
                counts: Counter = Counter()
                games: Counter = Counter()
                close_events: list = []
                http_errors: list = []
                try:
                    if transport == "gateway":
                        # Import only after setting a dedicated development config.
                        os.environ.update(
                            ARC_API_KEY="test-key-123",
                            OPERATION_MODE="offline",
                            ENVIRONMENTS_DIR=str(args.environments_dir.resolve()),
                            RECORDINGS_DIR=str(output / "server_recordings"),
                        )
                        from arc_agi import Arcade, OperationMode
                        from arc_agi.server import create_app
                        from flask import request
                        from werkzeug.serving import make_server

                        logging.getLogger("werkzeug").setLevel(logging.ERROR)
                        arcade = Arcade(
                            operation_mode=OperationMode.OFFLINE,
                            environments_dir=str(args.environments_dir.resolve()),
                            recordings_dir=str(output / "server_recordings"),
                        )
                        if args.game:
                            arcade.available_environments = [
                                e
                                for e in arcade.available_environments
                                if e.game_id == args.game
                                or e.game_id.split("-")[0] == args.game
                            ]
                        if not arcade.available_environments:
                            raise RuntimeError("no matching cached public games")

                        def closed(card):
                            close_events.append(True)
                            write_json(
                                destination / "gateway_scorecard.json",
                                public_data(
                                    json.loads(card.model_dump_json()),
                                ),
                            )

                        app, _ = create_app(
                            arcade, competition_mode=True, on_scorecard_close=closed
                        )

                        @app.before_request
                        def count_request():
                            counts[f"{request.method} {request.path}"] += 1
                            if request.path.startswith("/api/cmd/"):
                                data = request.get_json(silent=True) or {}
                                games[data.get("game_id", "unknown")] += 1

                        @app.after_request
                        def count_response(response):
                            if response.status_code >= 400:
                                http_errors.append(
                                    {
                                        "path": request.path,
                                        "status": response.status_code,
                                    }
                                )
                            return response

                        server = make_server("127.0.0.1", 0, app)
                        worker = threading.Thread(
                            target=server.serve_forever, daemon=True
                        )
                        worker.start()
                        child_env["ARC_EVAL_GATEWAY_URL"] = (
                            f"http://127.0.0.1:{server.server_port}/"
                        )

                    code = """
import json, os, subprocess, sys
from pathlib import Path
notebook = json.loads(Path(sys.argv[1]).read_text())
diagnostics = Path(os.environ['ARC_EVAL_OUTPUT'])
# Execute the exact generated unpack cell, with its bundle integrity assertion.
exec(compile(notebook['cells'][2]['source'], 'notebook-unpack', 'exec'))
if sys.argv[2] == 'gateway':
    exec(compile(notebook['cells'][3]['source'], 'notebook-phase-b', 'exec'))
else:
    command = [sys.executable, str(source_root / 'kaggle_runner.py'),
               '--mode', 'offline', '--output', str(diagnostics),
               '--environments-dir', os.environ['ENVIRONMENTS_DIR'],
               '--structured-budget', os.environ['ARC_EVAL_BUDGET']]
    if os.getenv('ARC_EVAL_GAME'):
        command.extend(['--game', os.environ['ARC_EVAL_GAME']])
    subprocess.run(command, cwd=source_root, check=True)
"""
                    with (destination / "notebook.log").open("w") as log:
                        result = subprocess.run(
                            [sys.executable, "-c", code, str(snapshot), transport],
                            cwd=staging,
                            env=child_env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=31750,
                        )
                    report = (
                        json.loads((destination / "report.json").read_text())
                        if (destination / "report.json").exists()
                        else {}
                    )
                    check = {
                        "transport": transport,
                        "exit_code": result.returncode,
                        "report_status": report.get("status"),
                        "runs": report.get("runs", []),
                        "packages": report.get("packages", {}),
                        "bundle_sha256": json.loads(
                            (destination / "bundle.json").read_text()
                        )["sha256"],
                    }
                    if transport == "gateway":
                        check.update(
                            http_requests=dict(counts),
                            game_requests=dict(games),
                            scorecard_close_callbacks=len(close_events),
                            http_errors=http_errors,
                        )
                    passed = (
                        result.returncode == 0
                        and report.get("status") == "completed"
                        and bool(report.get("runs"))
                        and all(
                            row.get("responses", 0) > 0
                            for row in report.get("runs", [])
                        )
                    )
                    if transport == "gateway":
                        passed = (
                            passed
                            and len(close_events) == 1
                            and not http_errors
                            and bool(games)
                        )
                    check["passed"] = bool(passed)
                    summary["checks"].append(check)
                    write_json(output / "verification.json", summary)
                    print(
                        json.dumps(
                            {
                                "transport": transport,
                                "passed": check["passed"],
                                "games": [
                                    {
                                        k: row.get(k)
                                        for k in [
                                            "game_id",
                                            "actions",
                                            "levels_completed",
                                            "status",
                                        ]
                                    }
                                    for row in report.get("runs", [])
                                ],
                            }
                        ),
                        flush=True,
                    )
                finally:
                    if server is not None:
                        server.shutdown()
                        server.server_close()
                    if worker is not None:
                        worker.join(timeout=5)
        if args.transport == "both" and all(c["passed"] for c in summary["checks"]):
            trajectories = {}
            for transport in ("offline", "gateway"):
                events = [
                    json.loads(line)
                    for line in (output / transport / "events.jsonl")
                    .read_text()
                    .splitlines()
                ]
                trajectories[transport] = [
                    (e["game_id"], e["action"], e["data"], e["after"])
                    for e in events
                    if e["event"] == "action_result"
                ]
            summary["offline_gateway_trajectories_match"] = (
                trajectories["offline"] == trajectories["gateway"]
            )
        summary["status"] = (
            "passed"
            if summary["checks"]
            and all(c["passed"] for c in summary["checks"])
            and summary.get("offline_gateway_trajectories_match", True)
            else "failed"
        )
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "verification.json", summary)
    print(f"Verification {summary['status']}: {output / 'verification.json'}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
