"""Upload/check the prepared notebook without rebuilding the random starter."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_notebook(starter: Path) -> None:
    notebook = json.loads((starter / "notebooks/submission.ipynb").read_text())
    bundle = (ROOT / "sampling_runs/kaggle_bundle/sources.zip").read_bytes()
    encoded = base64.b64encode(bundle).decode()
    source = "\n".join(
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in notebook["cells"]
    )
    if encoded not in source or hashlib.sha256(bundle).hexdigest() not in source:
        raise ValueError(
            "notebook does not contain the prepared source bundle; rebuild it"
        )
    if notebook["metadata"]["kaggle"]["isInternetEnabled"]:
        raise ValueError("competition notebook must have internet disabled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["push", "status"])
    parser.add_argument(
        "--starter", type=Path, default=ROOT / ".external/kaggle-starter"
    )
    parser.add_argument(
        "--acknowledge-backend-rights",
        action="store_true",
        help="Confirm permission to distribute the third-party reference backend to Kaggle",
    )
    args = parser.parse_args()
    if args.command == "push" and not args.acknowledge_backend_rights:
        raise SystemExit(
            "Reference solver has no explicit redistribution license. "
            "Obtain permission before uploading its source; then use "
            "--acknowledge-backend-rights."
        )
    starter = args.starter.resolve()
    metadata = json.loads((starter / "notebooks/kernel-metadata.json").read_text())
    token_path = starter / ".kaggle/access_token"
    if not token_path.is_file() or not token_path.read_text().strip():
        raise SystemExit(
            f"Save your Kaggle API token locally at {token_path}; do not send it in chat."
        )
    cli = starter / ".venv/bin/kaggle"
    if not cli.is_file():
        raise SystemExit(
            "install the Kaggle CLI into the Starter's separate .venv first"
        )
    if "REPLACE_WITH_YOUR_USERNAME" in metadata["id"]:
        raise SystemExit("configure your Kaggle username first")
    env = dict(os.environ, KAGGLE_API_TOKEN=token_path.read_text().strip())
    if args.command == "push":
        validate_notebook(starter)
        command = [str(cli), "kernels", "push", "-p", str(starter / "notebooks")]
    else:
        command = [str(cli), "kernels", "status", metadata["id"]]
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
