#!/usr/bin/env python3
"""Compare two ls20 recordings from the start of a selected level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            rows.append(payload.get("data", payload))
    return rows


def action_name(row: dict[str, Any]) -> str:
    value = (row.get("action_input") or {}).get("id")
    if isinstance(value, int):
        return "RESET" if value == 0 else f"ACTION{value}"
    return str(value or "UNKNOWN").upper()


def level_actions(rows: list[dict[str, Any]], level: int) -> list[str]:
    entry = next(
        index
        for index, row in enumerate(rows)
        if int(row.get("levels_completed") or 0) == level
    )
    actions: list[str] = []
    for row in rows[entry + 1 :]:
        if int(row.get("levels_completed") or 0) != level:
            break
        actions.append(action_name(row))
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = level_actions(load_rows(args.reference), args.level)
    candidate = level_actions(load_rows(args.candidate), args.level)
    common = min(len(reference), len(candidate))
    first_difference = next(
        (index for index in range(common) if reference[index] != candidate[index]),
        None,
    )
    if first_difference is None and len(reference) != len(candidate):
        first_difference = common

    report = {
        "schema_version": 1,
        "level_zero_based": args.level,
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_action_count": len(reference),
        "candidate_action_count": len(candidate),
        "first_difference_step_zero_based": first_difference,
        "reference_action": (
            reference[first_difference]
            if first_difference is not None and first_difference < len(reference)
            else None
        ),
        "candidate_action": (
            candidate[first_difference]
            if first_difference is not None and first_difference < len(candidate)
            else None
        ),
        "reference_prefix": reference[:12],
        "candidate_prefix": candidate[:12],
    }
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
