"""Upload/check the prepared notebook without rebuilding the random starter."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import subprocess
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def validate_notebook(starter: Path, *, owned_only: bool = False) -> None:
    directory = starter / ("notebooks-owned" if owned_only else "notebooks")
    notebook = json.loads((directory / "submission.ipynb").read_text())
    bundle = (ROOT / (
        "sampling_runs/kaggle_bundle_owned/sources.zip"
        if owned_only else "sampling_runs/kaggle_bundle/sources.zip"
    )).read_bytes()
    if owned_only:
        with ZipFile(BytesIO(bundle)) as archive:
            names = set(archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
        if any(name.startswith("backend/") or "ls20_solver_agent" in name for name in names):
            raise ValueError("owned-only bundle includes the reference backend")
        if manifest["agents"] != {"all": "StructuredAgent"}:
            raise ValueError("owned-only manifest selects an external solver")
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
    parser.add_argument("--owned-only", action="store_true")
    args = parser.parse_args()
    if args.command == "push" and not (
        args.owned_only or args.acknowledge_backend_rights
    ):
        raise SystemExit(
            "Reference solver has no explicit redistribution license. "
            "Obtain permission before uploading its source; then use "
            "--acknowledge-backend-rights."
        )
    starter = args.starter.resolve()
    directory = starter / ("notebooks-owned" if args.owned_only else "notebooks")
    metadata = json.loads((directory / "kernel-metadata.json").read_text())
    token_path = starter / ".kaggle/access_token"
    if not token_path.is_file():
        token_path = Path.home() / ".kaggle/access_token"
    if not token_path.is_file() or not token_path.read_text().strip():
        raise SystemExit(
            f"Save your Kaggle API token locally at {token_path}; do not send it in chat."
        )
    if stat.S_IMODE(token_path.stat().st_mode) & 0o077:
        raise SystemExit("Kaggle token must be private (chmod 600)")
    cli = starter / ".venv/bin/kaggle"
    if not cli.is_file():
        raise SystemExit(
            "install the Kaggle CLI into the Starter's separate .venv first"
        )
    if "REPLACE_WITH_YOUR_USERNAME" in metadata["id"]:
        raise SystemExit("configure your Kaggle username first")
    env = dict(os.environ, KAGGLE_API_TOKEN=token_path.read_text().strip())
    if args.command == "push":
        validate_notebook(starter, owned_only=args.owned_only)
        command = [str(cli), "kernels", "push", "-p", str(directory)]
    else:
        command = [str(cli), "kernels", "status", metadata["id"]]
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
