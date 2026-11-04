#!/usr/bin/env python3
"""Single-step checkpoint verification with relaxed state keys (default-OFF).

Only executes if STRUCTURED_RELAXED_STATE_KEYS=1. Replays a recording up to
a specified checkpoint row, then executes ONE new action predicted using
relaxed state keys. Immediately reports match/mismatch and halts.

This is the first incremental verification step after a GO decision from
audit_state_key_ablation.py. Do not skip to multi-step or full level clear
without first validating single-step prediction accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from arcengine import FrameData

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.episode_memory import EpisodeMemory  # noqa: E402
from agents.mechanism_policy import MechanismPolicy  # noqa: E402
from agents.mechanism_state_tracker import MechanismStateTracker  # noqa: E402
from agents.navigation_frontier_policy import NavigationFrontierPolicy  # noqa: E402
from agents.templates.structured_agent import StructuredAgent  # noqa: E402
from scripts.shadow_replay_structured import action_name, load_frames  # noqa: E402


class RelaxedStateKeyPredictor:
    """Prediction using relaxed state keys from audit recommendation.

    This is a placeholder implementation. If audit returns GO with a specific
    field set, implement prediction logic here using that field set.
    """

    def __init__(self, recommended_fields: list[str]):
        self.fields = set(recommended_fields)
        self.evidence = defaultdict(lambda: defaultdict(int))
        self.enabled = os.getenv("STRUCTURED_RELAXED_STATE_KEYS", "0") == "1"

    def observe(self, state, action, next_state):
        """Record an observed transition (during checkpoint replay)."""
        if not self.enabled:
            return

        # TODO: Build state key using self.fields
        # key = build_key_from(state, self.fields)
        # outcome = (next_state.player_position, next_state.status_signature, next_state.status_color)
        # self.evidence[key][outcome] += 1
        pass

    def predict(self, state, action):
        """Predict next state using learned evidence.

        Returns None if no prediction available (unseen key or ambiguous).
        Returns (player_position, status_signature, status_color) if confident.
        """
        if not self.enabled:
            return None

        # TODO: Build state key using self.fields
        # key = build_key_from(state, self.fields)
        # outcomes = self.evidence.get(key)
        # if outcomes and len(outcomes) == 1:
        #     return list(outcomes.keys())[0]
        return None


def replay_to_checkpoint(
    frames: list[FrameData], checkpoint_row: int
) -> tuple[StructuredAgent, RelaxedStateKeyPredictor]:
    """Replay recording up to checkpoint row, building relaxed-key evidence."""
    agent = StructuredAgent.__new__(StructuredAgent)
    agent.navigation_policy = NavigationFrontierPolicy()
    agent.mechanism_tracker = MechanismStateTracker(
        enable_entity_tracking=os.getenv("STRUCTURED_ENTITY_TRACKING", "0") == "1",
    )
    agent.mechanism_policy = MechanismPolicy(
        enable_joint_navigation=os.getenv("STRUCTURED_JOINT_NAVIGATION", "1") != "0",
        enable_motion_context=os.getenv("STRUCTURED_MOTION_CONTEXT", "0") == "1",
        enable_dynamic_contact=os.getenv("STRUCTURED_DYNAMIC_CONTACT", "0") == "1",
        enable_option_compiler=os.getenv("STRUCTURED_OPTION_COMPILER", "0") == "1",
        enable_contact_effects=os.getenv("STRUCTURED_CONTACT_EFFECTS", "0") == "1",
        enable_entity_options=os.getenv("STRUCTURED_ENTITY_OPTIONS", "0") == "1",
    )
    agent.episode_memory = EpisodeMemory()
    agent.last_action = None
    agent.fallback_counts = defaultdict(lambda: defaultdict(int))
    agent.fallback_clicks_by_level = defaultdict(set)

    # TODO: Get recommended fields from audit report
    # For now, use placeholder
    predictor = RelaxedStateKeyPredictor([])

    # Replay up to checkpoint
    for i in range(min(checkpoint_row + 1, len(frames))):
        frame = frames[i]
        agent.mechanism_tracker.observe(frame, action=action_name(frame))
        agent.episode_memory.add(frame)

        # Record transition for predictor (TODO: implement when audit returns GO)
        # if i > 0:
        #     state = agent.mechanism_tracker.previous_state
        #     predictor.observe(state, action_name(frame), agent.mechanism_tracker.current_state)

    return agent, predictor


def verify_single_step(
    agent: StructuredAgent,
    predictor: RelaxedStateKeyPredictor,
    checkpoint_frame: FrameData,
    next_frame: FrameData,
) -> dict[str, Any]:
    """Execute single-step verification.

    Returns match/mismatch verdict and diagnostic info.
    """
    if not predictor.enabled:
        return {
            "error": "STRUCTURED_RELAXED_STATE_KEYS not enabled",
            "verdict": "SKIPPED",
        }

    state = agent.mechanism_tracker.previous_state
    if state is None:
        return {"error": "No state available at checkpoint", "verdict": "ERROR"}

    # Predict using relaxed keys
    # TODO: Implement actual prediction
    predicted_outcome = predictor.predict(state, action_name(next_frame))

    if predicted_outcome is None:
        return {
            "verdict": "NO_PREDICTION",
            "reason": "Unseen key or ambiguous training evidence",
            "checkpoint_row": checkpoint_frame.action_input.id,
            "checkpoint_player": state.player_position,
        }

    # Compare with actual outcome
    actual_outcome = (
        next_frame.player[0],  # Actual player position from recording
        state.status_signature,  # Actual status
        state.status_color,
    )

    match = predicted_outcome == actual_outcome

    return {
        "verdict": "MATCH" if match else "MISMATCH",
        "checkpoint_row": checkpoint_frame.action_input.id,
        "action": action_name(next_frame),
        "checkpoint_state": {
            "player": state.player_position,
            "status_signature": state.status_signature,
            "status_color": state.status_color,
            "resource": state.resource_level,
        },
        "predicted": {
            "player": predicted_outcome[0],
            "status_signature": predicted_outcome[1],
            "status_color": predicted_outcome[2],
        },
        "observed": {
            "player": actual_outcome[0],
            "status_signature": actual_outcome[1],
            "status_color": actual_outcome[2],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="Recording file")
    parser.add_argument(
        "--checkpoint-row",
        type=int,
        required=True,
        help="Row number to resume from (e.g., 620)",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        help="Path to audit_state_key_ablation.py output with GO recommendation",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON")
    args = parser.parse_args()

    if os.getenv("STRUCTURED_RELAXED_STATE_KEYS", "0") != "1":
        print(
            "ERROR: STRUCTURED_RELAXED_STATE_KEYS=1 required for this experiment",
            file=sys.stderr,
        )
        print(
            "This is a default-OFF feature. Set the environment variable explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.recording.exists():
        print(f"ERROR: Recording not found: {args.recording}", file=sys.stderr)
        sys.exit(1)

    frames = load_frames(args.recording)

    if args.checkpoint_row >= len(frames) - 1:
        print(
            f"ERROR: Checkpoint row {args.checkpoint_row} beyond recording length {len(frames)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Replay to checkpoint
    agent, predictor = replay_to_checkpoint(frames, args.checkpoint_row)

    # Verify single next action
    checkpoint_frame = frames[args.checkpoint_row]
    next_frame = frames[args.checkpoint_row + 1]

    result = verify_single_step(agent, predictor, checkpoint_frame, next_frame)

    # Save report
    report = {
        "schema_version": 1,
        "recording": str(args.recording.resolve()),
        "checkpoint_row": args.checkpoint_row,
        "audit_report": str(args.audit_report.resolve()) if args.audit_report else None,
        "relaxed_keys_enabled": predictor.enabled,
        "verification": result,
        "safety_note": "Single-step only. Do not proceed to multi-step without MATCH verdict.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"Verdict: {result['verdict']}")
    if result["verdict"] == "MATCH":
        print("✓ Single-step prediction matches observation.")
        print("Next: Extend to bounded multi-step verification.")
    elif result["verdict"] == "MISMATCH":
        print("✗ Prediction mismatch. Do not proceed.")
        print("Review field set or return to audit.")
    else:
        print(f"Cannot verify: {result.get('reason', 'unknown')}")

    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
