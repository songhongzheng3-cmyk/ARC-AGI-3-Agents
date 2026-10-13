#!/usr/bin/env python3
"""Replay public recordings through the unchanged reference scene parser.

This audits observation bindings; it executes no game actions and is not a
solver evaluation. Input is the project's serialized FrameData, not FrameDataRaw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from arcengine import FrameData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".external" / "ls20-solver" / "src"))

from ls20.abstract_state import GenericSceneParser  # noqa: E402

from scripts.verify_ls20_baseline import git_info, write_json  # noqa: E402


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("probe_directory", type=Path)
    args = cli.parse_args()
    directory = args.probe_directory.resolve()
    results = []
    for recording in sorted((directory / "recordings").glob("*.recording.jsonl")):
        parser = GenericSceneParser()
        rows = []
        for index, line in enumerate(recording.read_text().splitlines()):
            frame = FrameData.model_validate(json.loads(line)["data"])
            if not frame.frame:
                raise RuntimeError("empty replay input; cannot audit parser")
            action = frame.action_input.id.name if index else None
            state = parser.parse(frame, action=action)
            rows.append(
                {
                    "step": index,
                    "action": action,
                    "layout": parser.layout.to_dict() if parser.layout else None,
                    "player_position": state.player_position,
                    "player_bbox": parser.player_bbox,
                    "object_count": len(state.objects),
                    "warnings": state.warnings,
                    "state": state.to_dict(),
                }
            )
        results.append({"recording": str(recording), "rows": rows})
    if not results:
        raise RuntimeError("no public recordings found")
    report = directory / "reference_parser_audit.json"
    write_json(
        report,
        {
            "backend": git_info(ROOT / ".external" / "ls20-solver"),
            "audit_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "recording_sha256": {
                result["recording"]: hashlib.sha256(
                    Path(result["recording"]).read_bytes()
                ).hexdigest()
                for result in results
            },
            "parser": "Unchanged GenericSceneParser, default parameters",
            "evidence_boundary": "Offline observation audit only; no goal labels or action selection and no cross-game success claim.",
            "results": results,
        },
    )
    print(f"AUDIT_REPORT {report}")
    print(f"recordings={len(results)}, frames={sum(len(r['rows']) for r in results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
