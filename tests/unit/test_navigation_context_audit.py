from types import SimpleNamespace

from scripts.analyze_navigation_context import (
    audit,
    repeated_noop_audit,
    temporal_holdout,
)


def _noop_replay(dynamic_contexts):
    frames = [
        SimpleNamespace(action_input=SimpleNamespace(id=1)) for _ in range(4)
    ]
    states = [
        SimpleNamespace(
            level=5,
            player_position=(0, 0),
            player_confidence=1.0,
            resource_capacity=100,
            resource_level=100 - index * 2,
            status_signature="010/010/010",
            status_color=4,
            objects=(),
            player_component_ids=(),
            walkable_cells=frozenset({(0, 0), (0, 1)}),
            blocked_cells=frozenset(),
        )
        for index in range(4)
    ]
    return frames, states, dynamic_contexts


def test_audit_predicts_before_learning_and_abstains_after_a_conflict():
    report = audit(
        [(1, "same", "A"), (2, "same", "B"), (3, "same", "B"), (4, "new", "B")]
    )
    assert report["samples"] == 4
    assert report["prediction_checks"] == 1
    assert report["prediction_errors"] == 1
    assert report["error_rows"] == [2]
    assert report["ambiguous_key_abstentions"] == 1
    assert report["new_conflicting_outcomes"] == 1
    assert report["unseen_keys"] == 2


def test_unique_contexts_are_zero_coverage_not_perfect_prediction():
    report = audit([(1, "left", "A"), (2, "right", "B")])
    assert report["prediction_errors"] == 0
    assert report["prediction_checks"] == 0
    assert report["coverage"] == 0


def test_temporal_holdout_freezes_training_evidence_at_split():
    report = temporal_holdout(
        [(1, "known", "A"), (2, "known", "A"), (8, "known", "B"),
         (9, "unseen", "C")],
        split_row=5,
    )
    assert report["training_samples"] == 2
    assert report["test_samples"] == 2
    assert report["prediction_checks"] == 1
    assert report["prediction_errors"] == 1
    assert report["error_rows"] == [8]
    assert report["unseen_keys"] == 1


def test_temporal_holdout_abstains_on_conflicting_training_key():
    report = temporal_holdout(
        [(1, "known", "A"), (2, "known", "B"), (8, "known", "A")],
        split_row=5,
    )
    assert report["prediction_checks"] == 0
    assert report["ambiguous_training_keys"] == 1
    assert report["prediction_errors"] == 0


def test_repeated_costly_noop_is_flagged_only_for_same_exact_context():
    moving = SimpleNamespace(signature=(8, 3, "1/1"), previous=(2, 2), current=(2, 3))
    frames, states, contexts = _noop_replay([(moving,)] * 4)

    report = repeated_noop_audit(frames, states, contexts, level=5)

    assert report["costly_noop_transitions"] == 3
    assert report["exact_context_repeated_attempts"] == 2
    assert report["repeated_attempts_that_changed_state"] == 0
    assert [item["repeat_row"] for item in report["repeated_attempts"]] == [1, 2]


def test_dynamic_motion_change_reopens_previously_failed_action():
    contexts = [
        (SimpleNamespace(signature=(8, 3, "1/1"), previous=(2, 1), current=(2, 2)),),
        (SimpleNamespace(signature=(8, 3, "1/1"), previous=(2, 2), current=(2, 3)),),
        (SimpleNamespace(signature=(8, 3, "1/1"), previous=(2, 3), current=(2, 4)),),
        (SimpleNamespace(signature=(8, 3, "1/1"), previous=(2, 4), current=(2, 5)),),
    ]
    frames, states, _ = _noop_replay(contexts)

    report = repeated_noop_audit(frames, states, contexts, level=5)

    assert report["exact_context_repeated_attempts"] == 0
    assert report["reopened_after_dynamic_context_change"] == 2
