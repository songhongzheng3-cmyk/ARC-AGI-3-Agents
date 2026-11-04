"""Notebook Phase A/B entry point, also exercised by the local gateway check."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def run(source_root: Path, output: Path, gateway: str, timeout: float = 600) -> None:
    output.mkdir(parents=True, exist_ok=True)
    status = {
        "phase": "B" if os.getenv("KAGGLE_IS_COMPETITION_RERUN") else "A",
        "python": platform.python_version(),
        "stage": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    def checkpoint(stage: str) -> None:
        status["stage"] = stage
        (output / "runtime.json").write_text(json.dumps(status, indent=2) + "\n")
        print("NOTEBOOK_STAGE", json.dumps(status), flush=True)

    try:
        if status["phase"] == "A":
            checkpoint("import_check")
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from agents.templates.structured_agent import StructuredAgent; "
                    "import json; from pathlib import Path; "
                    "assert json.loads(Path('manifest.json').read_text())['agents']",
                ],
                cwd=source_root,
                check=True,
            )
            import pandas as pd

            pd.DataFrame(
                [["1_0", "1", True, 1]],
                columns=[
                    "row_id",
                    "game_id",
                    "end_of_game",
                    "score",
                ],
            ).to_parquet(output.parent / "submission.parquet", index=False)
            print(
                "Phase A import check passed; parquet is a placeholder, not an evaluation score."
            )
        else:
            checkpoint("waiting_for_gateway")
            deadline = time.monotonic() + timeout
            while True:
                try:
                    with urllib.request.urlopen(
                        gateway.rstrip("/") + "/api/games", timeout=10
                    ) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "competition gateway did not become ready"
                    ) from None
                time.sleep(min(5, max(0, deadline - time.monotonic())))
            checkpoint("running_agent")
            env = dict(
                os.environ,
                ARC_API_KEY="test-key-123",
                ARC_BASE_URL=gateway,
                OPERATION_MODE="online",
                ENVIRONMENTS_DIR="",
                MPLBACKEND="agg",
                PYTHONUNBUFFERED="1",
            )
            command = [
                sys.executable,
                str(source_root / "kaggle_runner.py"),
                "--mode",
                "online",
                "--output",
                str(output),
            ]
            if os.getenv("ARC_EVAL_GAME"):
                command.extend(["--game", os.environ["ARC_EVAL_GAME"]])
            if os.getenv("ARC_EVAL_BUDGET"):
                command.extend(["--structured-budget", os.environ["ARC_EVAL_BUDGET"]])
            with (output / "runner.log").open("w") as log:
                with subprocess.Popen(
                    command,
                    cwd=source_root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                ) as child:
                    timer = threading.Timer(31000, child.kill)
                    timer.daemon = True
                    timer.start()
                    try:
                        assert child.stdout is not None
                        for line in child.stdout:
                            log.write(line)
                            log.flush()
                            print(line, end="", flush=True)
                        code = child.wait()
                    finally:
                        timer.cancel()
                status["runner_exit_code"] = code
                if code:
                    raise subprocess.CalledProcessError(code, command)
        checkpoint("completed")
    except BaseException as exc:
        status.update(
            error=f"{type(exc).__name__}: {exc}", failed_stage=status["stage"]
        )
        checkpoint("failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.source_root.resolve(),
        Path(os.getenv("ARC_EVAL_OUTPUT", "/kaggle/working/arc_eval")).resolve(),
        os.getenv("ARC_EVAL_GATEWAY_URL", "http://gateway:8001/"),
    )


if __name__ == "__main__":
    main()
