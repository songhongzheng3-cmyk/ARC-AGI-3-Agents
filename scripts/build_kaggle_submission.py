"""Build an offline notebook from an explicit, credential-free source bundle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
BACKEND_REVISION = "bcf25348fd88fddbaaabb5c332f6bf3bb3a3fb60"
SOURCE_FILES = (
    "agents/agent.py",
    "agents/recorder.py",
    "agents/tracing.py",
    "agents/game_model.py",
    "agents/object_extractor.py",
    "agents/mechanism_policy.py",
    "agents/mechanism_state_tracker.py",
    "agents/navigation_frontier_policy.py",
    "agents/task_planning.py",
    "agents/templates/ls20_solver_agent.py",
    "agents/templates/structured_agent.py",
)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def collect_sources(root: Path, backend: Path) -> tuple[dict[str, bytes], dict]:
    revision = git(backend, "rev-parse", "HEAD")
    if revision != BACKEND_REVISION or git(backend, "status", "--porcelain"):
        raise ValueError("reference backend must be the unchanged pinned checkout")
    sources = {name: (root / name).read_bytes() for name in SOURCE_FILES}
    sources["agents/__init__.py"] = b'"""Offline submission modules only."""\n'
    sources["agents/templates/__init__.py"] = b""
    sources["kaggle_runner.py"] = (root / "scripts/kaggle_runner.py").read_bytes()
    for path in sorted((backend / "src/ls20").glob("*.py")):
        sources[f"backend/src/ls20/{path.name}"] = path.read_bytes()
    if "backend/src/ls20/agent.py" not in sources:
        raise ValueError("missing reference agent")
    sources["backend/README.md"] = (backend / "README.md").read_bytes()
    manifest = {
        "schema_version": 1,
        "project_revision": git(root, "rev-parse", "HEAD"),
        "project_dirty": bool(git(root, "status", "--porcelain")),
        "agents": {"ls20": "Ls20SolverAgent", "other": "StructuredAgent"},
        "backend_url": "https://github.com/zzZYHarold/arc-agi-3-ls20-agent",
        "backend_revision": revision,
        "source_sha256": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(sources.items())
        },
        "excluded": [
            "credentials",
            "game implementations",
            "development trajectories",
            "LLM agents",
        ],
    }
    sources["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return sources, manifest


def make_bundle(sources: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(sources.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return buffer.getvalue()


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "execution_count": None,
        "outputs": [],
    }


def build_notebook(bundle: bytes) -> dict:
    encoded = base64.b64encode(bundle).decode()
    digest = hashlib.sha256(bundle).hexdigest()
    install = dedent("""\
        import subprocess, sys
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links',
            '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels',
            'arc-agi', 'python-dotenv',
        ], check=True)
        """)
    unpack = dedent(f"""\
        import base64, hashlib, io, tempfile, zipfile
        from pathlib import Path
        bundle = base64.b64decode({encoded!r})
        assert hashlib.sha256(bundle).hexdigest() == {digest!r}
        source_root = Path(tempfile.mkdtemp(prefix='arc_submission_'))
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            archive.extractall(source_root)
        print('Source bundle:', {digest!r})
        """)
    run = dedent("""\
        import os, subprocess, sys, time, urllib.request
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            deadline = time.monotonic() + 600
            while True:
                try:
                    with urllib.request.urlopen('http://gateway:8001/api/games', timeout=10) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('competition gateway did not become ready')
                    time.sleep(5)
            env = dict(os.environ, SCHEME='http', HOST='gateway', PORT='8001',
                       ARC_API_KEY='test-key-123', ARC_BASE_URL='http://gateway:8001/',
                       OPERATION_MODE='online', ENVIRONMENTS_DIR='', MPLBACKEND='agg')
            subprocess.run([sys.executable, str(source_root / 'kaggle_runner.py'),
                            '--mode', 'online', '--output', '/tmp/arc_eval'],
                           cwd=source_root, env=env, check=True, timeout=31000)
        else:
            # Phase A checks imports but cannot play the hidden gateway games.
            subprocess.run([sys.executable, '-c',
                            'from agents.templates.ls20_solver_agent import Ls20SolverAgent; '
                            'from agents.templates.structured_agent import StructuredAgent; '
                            'import sys; from pathlib import Path; '
                            'sys.path.insert(0, str(Path("backend/src").resolve())); '
                            'from ls20.agent import Phase6Agent; from ls20.session import MultiLevelSession'],
                           cwd=source_root, check=True)
            import pandas as pd
            pd.DataFrame([['1_0', '1', True, 1]],
                         columns=['row_id', 'game_id', 'end_of_game', 'score']).to_parquet(
                         '/kaggle/working/submission.parquet', index=False)
            print('Phase A import check passed; parquet is a placeholder, not an evaluation score.')
        """)
    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "kaggle": {
                "accelerator": "none",
                "isInternetEnabled": False,
                "isGpuEnabled": False,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# Current agent competition baseline\n\n"
                "ls20: unchanged public reference backend; other games: StructuredAgent.\n"
                "Reference: https://github.com/zzZYHarold/arc-agi-3-ls20-agent\n"
                "No claim of cross-game generalization. Phase A output is a placeholder.",
            },
            code_cell(install),
            code_cell(unpack),
            code_cell(run),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--starter", type=Path, default=ROOT / ".external/kaggle-starter"
    )
    parser.add_argument("--backend", type=Path, default=ROOT / ".external/ls20-solver")
    args = parser.parse_args()
    sources, manifest = collect_sources(ROOT, args.backend.resolve())
    bundle = make_bundle(sources)
    notebook = build_notebook(bundle)
    directory = args.starter.resolve() / "notebooks"
    metadata_path = directory / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if "REPLACE_WITH_YOUR_USERNAME" in metadata["id"]:
        raise SystemExit("configure your Kaggle username in kernel-metadata.json first")
    metadata.update(enable_gpu=False, enable_internet=False)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    (directory / "submission.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    output = ROOT / "sampling_runs/kaggle_bundle"
    output.mkdir(parents=True, exist_ok=True)
    (output / "sources.zip").write_bytes(bundle)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {directory / 'submission.ipynb'}")
    print(
        f"Bundle: {len(sources)} files, {len(bundle)} bytes, sha256={hashlib.sha256(bundle).hexdigest()}"
    )
    print(
        "CPU, internet disabled. Upload this notebook directly; official make submit would rebuild the random starter."
    )


if __name__ == "__main__":
    main()
