"""Verify reproducible offline packaging and guard against random rebuilds."""

import base64
import hashlib
import io
import json
import zipfile

import pytest

from scripts.build_kaggle_submission import build_notebook, collect_sources, make_bundle
from scripts.kaggle_submit import validate_notebook


def test_bundle_preserves_sources_and_is_deterministic():
    sources = {"agents/example.py": b"print('agent')\n", "manifest.json": b"{}"}
    bundle = make_bundle(sources)
    assert bundle == make_bundle(dict(reversed(list(sources.items()))))
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert set(archive.namelist()) == set(sources)
        for name, data in sources.items():
            assert archive.read(name) == data


def test_notebook_has_checked_offline_runtime_and_compiles():
    bundle = make_bundle({"manifest.json": b"{}"})
    notebook = build_notebook(bundle)
    assert not notebook["metadata"]["kaggle"]["isInternetEnabled"]
    assert not notebook["metadata"]["kaggle"]["isGpuEnabled"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], "notebook-cell", "exec")
    unpack = notebook["cells"][2]["source"]
    assert base64.b64encode(bundle).decode() in unpack
    assert hashlib.sha256(bundle).hexdigest() in unpack
    run = notebook["cells"][3]["source"]
    assert "check=True" in run
    assert "KAGGLE_IS_COMPETITION_RERUN" in run
    assert "kaggle_runner.py" in run


def test_upload_guard_rejects_overwritten_notebook(tmp_path, monkeypatch):
    from scripts import kaggle_submit

    monkeypatch.setattr(kaggle_submit, "ROOT", tmp_path)
    output = tmp_path / "sampling_runs/kaggle_bundle"
    output.mkdir(parents=True)
    bundle = make_bundle({"manifest.json": b"{}"})
    (output / "sources.zip").write_bytes(bundle)
    starter = tmp_path / "starter"
    notebooks = starter / "notebooks"
    notebooks.mkdir(parents=True)
    notebook = build_notebook(bundle)
    target = notebooks / "submission.ipynb"
    target.write_text(json.dumps(notebook))
    validate_notebook(starter)
    notebook["cells"][2]["source"] = "print('random starter')"
    target.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="does not contain"):
        validate_notebook(starter)


def test_upload_requires_backend_distribution_rights(monkeypatch):
    import sys

    from scripts.kaggle_submit import main

    monkeypatch.setattr(sys, "argv", ["kaggle_submit.py", "push"])
    with pytest.raises(SystemExit, match="no explicit redistribution license"):
        main()


def test_owned_only_bundle_excludes_reference_backend(tmp_path, monkeypatch):
    from scripts import kaggle_submit

    root = tmp_path / "project"
    from scripts.build_kaggle_submission import SOURCE_FILES

    for name in SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n")
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts/kaggle_runner.py").write_text("# runner\n")
    (root / "scripts/kaggle_runtime.py").write_text("# runtime\n")
    monkeypatch.setattr("scripts.build_kaggle_submission.git", lambda *_: "revision")
    sources, manifest = collect_sources(root, None)
    assert manifest["agents"] == {"all": "StructuredAgent"}
    assert not any(
        "ls20_solver_agent" in name or name.startswith("backend/") for name in sources
    )

    bundle = make_bundle(sources)
    monkeypatch.setattr(kaggle_submit, "ROOT", tmp_path)
    output = tmp_path / "sampling_runs/kaggle_bundle_owned"
    output.mkdir(parents=True)
    (output / "sources.zip").write_bytes(bundle)
    starter = tmp_path / "starter"
    directory = starter / "notebooks-owned"
    directory.mkdir(parents=True)
    (directory / "submission.ipynb").write_text(
        json.dumps(build_notebook(bundle, owned_only=True))
    )
    validate_notebook(starter, owned_only=True)

    sources["backend/src/ls20/agent.py"] = b"unlicensed\n"
    (output / "sources.zip").write_bytes(make_bundle(sources))
    with pytest.raises(ValueError, match="reference backend"):
        validate_notebook(starter, owned_only=True)
