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
    "agents/episode_memory.py",
    "agents/game_model.py",
    "agents/object_extractor.py",
    "agents/mechanism_policy.py",
    "agents/dynamic_contact.py",
    "agents/component_entities.py",
    "agents/contact_effects.py",
    "agents/entity_options.py",
    "agents/joint_navigation.py",
    "agents/mechanism_state_tracker.py",
    "agents/motion_context.py",
    "agents/option_compiler.py",
    "agents/navigation_frontier_policy.py",
    "agents/task_planning.py",
    "agents/templates/ls20_solver_agent.py",
    "agents/templates/structured_agent.py",
)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def collect_sources(root: Path, backend: Path | None) -> tuple[dict[str, bytes], dict]:
    revision = None
    if backend is not None:
        revision = git(backend, "rev-parse", "HEAD")
        if revision != BACKEND_REVISION or git(backend, "status", "--porcelain"):
            raise ValueError("reference backend must be the unchanged pinned checkout")
    sources = {
        name: (root / name).read_bytes()
        for name in SOURCE_FILES
        if backend is not None or name != "agents/templates/ls20_solver_agent.py"
    }
    sources["agents/__init__.py"] = b'"""Offline submission modules only."""\n'
    sources["agents/templates/__init__.py"] = b""
    for name in ("kaggle_runner.py", "kaggle_runtime.py"):
        sources[name] = (root / "scripts" / name).read_bytes()
    if backend is not None:
        for path in sorted((backend / "src/ls20").glob("*.py")):
            sources[f"backend/src/ls20/{path.name}"] = path.read_bytes()
        if "backend/src/ls20/agent.py" not in sources:
            raise ValueError("missing reference agent")
        sources["backend/README.md"] = (backend / "README.md").read_bytes()
    manifest = {
        "schema_version": 1,
        "project_revision": git(root, "rev-parse", "HEAD"),
        "project_dirty": bool(git(root, "status", "--porcelain")),
        "agents": (
            {"ls20": "Ls20SolverAgent", "other": "StructuredAgent"}
            if backend is not None
            else {"all": "StructuredAgent"}
        ),
        "backend_url": (
            "https://github.com/zzZYHarold/arc-agi-3-ls20-agent"
            if backend is not None
            else None
        ),
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


def build_notebook(bundle: bytes, *, owned_only: bool = False) -> dict:
    encoded = base64.b64encode(bundle).decode()
    digest = hashlib.sha256(bundle).hexdigest()
    install = dedent("""\
        import json, os, platform, subprocess, sys
        from pathlib import Path
        diagnostics = Path(os.getenv('ARC_EVAL_OUTPUT', '/kaggle/working/arc_eval'))
        diagnostics.mkdir(parents=True, exist_ok=True)
        installation = {'python': platform.python_version(), 'stage': 'installing'}
        (diagnostics / 'installation.json').write_text(json.dumps(installation))
        try:
            with (diagnostics / 'install.log').open('w') as log:
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links',
                    '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels',
                    'arc-agi', 'python-dotenv',
                ], stdout=log, stderr=subprocess.STDOUT)
            print((diagnostics / 'install.log').read_text())
            result.check_returncode()
            installation['stage'] = 'completed'
        except BaseException as exc:
            installation.update(stage='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            (diagnostics / 'installation.json').write_text(json.dumps(installation, indent=2))
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
        (diagnostics / 'bundle.json').write_text(json.dumps({{'sha256': {digest!r}}}))
        """)
    run = dedent("""\
        import os, subprocess, sys
        # KAGGLE_IS_COMPETITION_RERUN selects Phase B inside the shared runtime.
        # That runtime launches kaggle_runner.py and retains both files and stdout.
        subprocess.run([sys.executable, str(source_root / 'kaggle_runtime.py'),
                        '--source-root', str(source_root)],
                       cwd=source_root, check=True, timeout=31700)
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
                + (
                    "All games: project-owned StructuredAgent.\n"
                    if owned_only
                    else (
                        "ls20: unchanged public reference backend; other games: StructuredAgent.\n"
                        "Reference: https://github.com/zzZYHarold/arc-agi-3-ls20-agent\n"
                    )
                )
                + "No claim of cross-game generalization. Phase A output is a placeholder.",
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
    parser.add_argument("--owned-only", action="store_true")
    args = parser.parse_args()
    sources, manifest = collect_sources(
        ROOT, None if args.owned_only else args.backend.resolve()
    )
    bundle = make_bundle(sources)
    notebook = build_notebook(bundle, owned_only=args.owned_only)
    starter = args.starter.resolve()
    directory = starter / ("notebooks-owned" if args.owned_only else "notebooks")
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "kernel-metadata.json"
    metadata = json.loads((starter / "notebooks/kernel-metadata.json").read_text())
    if "REPLACE_WITH_YOUR_USERNAME" in metadata["id"]:
        raise SystemExit("configure your Kaggle username in kernel-metadata.json first")
    metadata.update(enable_gpu=False, enable_internet=False)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    (directory / "submission.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    output = ROOT / (
        "sampling_runs/kaggle_bundle_owned"
        if args.owned_only
        else "sampling_runs/kaggle_bundle"
    )
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
