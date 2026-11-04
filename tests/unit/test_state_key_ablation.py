"""Unit tests for state-key ablation audit script."""

from __future__ import annotations

from scripts.audit_state_key_ablation import (
    TransitionSample,
    build_state_key,
    classify_movement,
    evaluate_field_set,
    make_recommendation,
)


def test_classify_movement_expected_adjacent():
    """Player moves to expected adjacent cell."""
    movement = classify_movement((5, 5), (5, 6), "ACTION2")
    assert movement == "expected_adjacent"


def test_classify_movement_blocked():
    """Player doesn't move (blocked or no-op)."""
    movement = classify_movement((5, 5), (5, 5), "ACTION1")
    assert movement == "blocked_or_noop"


def test_classify_movement_nonlocal():
    """Player moves more than 1 cell away (nonlocal)."""
    movement = classify_movement((5, 5), (3, 5), "ACTION1")
    assert movement == "nonlocal"


def test_classify_movement_unexpected_adjacent():
    """Player moves to adjacent cell but not the expected one."""
    movement = classify_movement((5, 5), (6, 5), "ACTION1")  # Expected (4,5), got (6,5)
    assert movement == "unexpected_adjacent"


def test_build_state_key_minimal():
    """Minimal state key includes only level, player, action."""
    sample = TransitionSample(
        row=0,
        level=1,
        player_position=(5, 5),
        status_signature="test_sig",
        status_color=10,
        resource_level=50,
        action="ACTION1",
        entity_signatures=(),
        walkable_cells=frozenset(),
        blocked_cells=frozenset(),
        static_objects=(),
        next_player_position=(5, 5),
        next_status_signature="test_sig",
        next_status_color=10,
        movement_class="blocked_or_noop",
    )

    key = build_state_key(sample, set())
    assert key == (1, (5, 5), "ACTION1")


def test_build_state_key_with_status():
    """State key includes status fields when specified."""
    sample = TransitionSample(
        row=0,
        level=1,
        player_position=(5, 5),
        status_signature="test_sig",
        status_color=10,
        resource_level=50,
        action="ACTION1",
        entity_signatures=(),
        walkable_cells=frozenset(),
        blocked_cells=frozenset(),
        static_objects=(),
        next_player_position=(5, 5),
        next_status_signature="test_sig",
        next_status_color=10,
        movement_class="blocked_or_noop",
    )

    key = build_state_key(sample, {"status_signature", "status_color"})
    assert key == (1, (5, 5), "ACTION1", "test_sig", 10)


def test_build_state_key_with_all_fields():
    """State key includes all fields when specified."""
    sample = TransitionSample(
        row=0,
        level=1,
        player_position=(5, 5),
        status_signature="test_sig",
        status_color=10,
        resource_level=50,
        action="ACTION1",
        entity_signatures=(("entity1", (3, 3), (3, 4)),),
        walkable_cells=frozenset({(4, 5), (6, 5)}),
        blocked_cells=frozenset({(5, 4)}),
        static_objects=((12, (1,), "shape1", (7, 7)),),
        next_player_position=(5, 5),
        next_status_signature="test_sig",
        next_status_color=10,
        movement_class="blocked_or_noop",
    )

    all_fields = {
        "status_signature",
        "status_color",
        "resource_level",
        "entity_signatures",
        "walkable_cells",
        "blocked_cells",
        "static_objects",
    }
    key = build_state_key(sample, all_fields)
    assert len(key) == 10  # 3 base + 7 optional


def test_evaluate_field_set_zero_conflicts():
    """Field set with zero conflicts in leave-out is viable."""
    # Create samples where same key always leads to same outcome
    samples = []
    for i in range(10):
        sample = TransitionSample(
            row=i,
            level=1,
            player_position=(5, 5),
            status_signature="sig_a",
            status_color=10,
            resource_level=50,
            action="ACTION1",
            entity_signatures=(),
            walkable_cells=frozenset(),
            blocked_cells=frozenset(),
            static_objects=(),
            next_player_position=(5, 5),  # Always same outcome
            next_status_signature="sig_a",
            next_status_color=10,
            movement_class="blocked_or_noop",
        )
        samples.append(sample)

    result = evaluate_field_set(samples, {"status_signature", "status_color"})

    assert result["rejected"] is False
    assert result["prediction_errors"] == 0
    assert result["ambiguous_keys"] == 0


def test_evaluate_field_set_with_conflicts():
    """Field set with conflicts in leave-out is rejected."""
    samples = []
    # First 7 samples: same key, same outcome
    for i in range(7):
        samples.append(
            TransitionSample(
                row=i,
                level=1,
                player_position=(5, 5),
                status_signature="sig_a",
                status_color=10,
                resource_level=50,
                action="ACTION1",
                entity_signatures=(),
                walkable_cells=frozenset(),
                blocked_cells=frozenset(),
                static_objects=(),
                next_player_position=(5, 5),
                next_status_signature="sig_a",
                next_status_color=10,
                movement_class="blocked_or_noop",
            )
        )

    # Last 3 samples: same key, DIFFERENT outcome (in leave-out after 70% split)
    for i in range(7, 10):
        samples.append(
            TransitionSample(
                row=i,
                level=1,
                player_position=(5, 5),
                status_signature="sig_a",
                status_color=10,
                resource_level=50,
                action="ACTION1",
                entity_signatures=(),
                walkable_cells=frozenset(),
                blocked_cells=frozenset(),
                static_objects=(),
                next_player_position=(4, 5),  # DIFFERENT outcome
                next_status_signature="sig_a",
                next_status_color=10,
                movement_class="nonlocal",
            )
        )

    result = evaluate_field_set(samples, {"status_signature", "status_color"})

    # Should be rejected due to prediction errors in test set
    # (same key leads to different outcome than seen in training)
    assert result["rejected"] is True
    assert result["prediction_errors"] > 0


def test_evaluate_field_set_insufficient_coverage():
    """Field set with very low coverage should be detectable."""
    samples = []
    # Create 10 samples with all unique keys (no reuse)
    for i in range(10):
        samples.append(
            TransitionSample(
                row=i,
                level=1,
                player_position=(i, i),  # Each at different position
                status_signature=f"sig_{i}",
                status_color=i,
                resource_level=50,
                action="ACTION1",
                entity_signatures=(),
                walkable_cells=frozenset(),
                blocked_cells=frozenset(),
                static_objects=(),
                next_player_position=(i, i),
                next_status_signature=f"sig_{i}",
                next_status_color=i,
                movement_class="blocked_or_noop",
            )
        )

    result = evaluate_field_set(samples, {"status_signature", "status_color"})

    # All test samples will have unseen keys
    assert result["coverage"] == 0.0
    assert result["unseen_keys"] == result["testing_samples"]


def test_make_recommendation_go():
    """GO recommendation when viable field set with good coverage exists."""
    analysis = {
        "total_samples": 100,
        "movement_class_distribution": {"nonlocal": 50, "blocked_or_noop": 50},
        "level_distribution": {5: 100},
        "field_set_evaluations": [
            {
                "field_set": ["status_signature", "status_color"],
                "training_samples": 70,
                "testing_samples": 30,
                "training_distinct_keys": 20,
                "training_conflicts": 0,
                "training_conflict_rate": 0.0,
                "prediction_checks": 8,
                "prediction_errors": 0,
                "unseen_keys": 22,
                "ambiguous_keys": 0,
                "coverage": 0.267,  # > 20% threshold
                "error_rate": 0.0,
                "rejected": False,
                "rejection_reason": None,
                "error_details": [],
            }
        ],
    }

    recommendation = make_recommendation(analysis)

    assert recommendation["decision"] == "GO"
    assert "coverage" in recommendation
    assert recommendation["coverage"] > 0.2


def test_make_recommendation_no_go_all_rejected():
    """NO-GO when all field sets are rejected."""
    analysis = {
        "total_samples": 100,
        "movement_class_distribution": {"nonlocal": 50, "blocked_or_noop": 50},
        "level_distribution": {5: 100},
        "field_set_evaluations": [
            {
                "field_set": ["status_signature", "status_color"],
                "training_samples": 70,
                "testing_samples": 30,
                "rejected": True,
                "rejection_reason": "same-key different-successor in leave-out",
            },
            {
                "field_set": ["status_signature", "status_color", "entity_signatures"],
                "training_samples": 70,
                "testing_samples": 30,
                "rejected": True,
                "rejection_reason": "same-key different-successor in leave-out",
            },
        ],
    }

    recommendation = make_recommendation(analysis)

    assert recommendation["decision"] == "NO-GO"
    assert "all field sets" in recommendation["reason"].lower()


def test_make_recommendation_no_go_low_coverage():
    """NO-GO when best viable field set has coverage below threshold."""
    analysis = {
        "total_samples": 100,
        "movement_class_distribution": {"nonlocal": 50, "blocked_or_noop": 50},
        "level_distribution": {5: 100},
        "field_set_evaluations": [
            {
                "field_set": ["status_signature", "status_color"],
                "training_samples": 70,
                "testing_samples": 30,
                "training_distinct_keys": 25,
                "training_conflicts": 0,
                "training_conflict_rate": 0.0,
                "prediction_checks": 3,
                "prediction_errors": 0,
                "unseen_keys": 27,
                "ambiguous_keys": 0,
                "coverage": 0.10,  # < 20% threshold
                "error_rate": 0.0,
                "rejected": False,
                "rejection_reason": None,
                "error_details": [],
            }
        ],
    }

    recommendation = make_recommendation(analysis)

    assert recommendation["decision"] == "NO-GO"
    assert "coverage" in recommendation["reason"].lower()


def test_make_recommendation_no_go_has_errors():
    """NO-GO when best viable field set has errors."""
    analysis = {
        "total_samples": 100,
        "movement_class_distribution": {"nonlocal": 50, "blocked_or_noop": 50},
        "level_distribution": {5: 100},
        "field_set_evaluations": [
            {
                "field_set": ["status_signature", "status_color"],
                "training_samples": 70,
                "testing_samples": 30,
                "training_distinct_keys": 20,
                "training_conflicts": 0,
                "training_conflict_rate": 0.0,
                "prediction_checks": 10,
                "prediction_errors": 2,  # Has errors
                "unseen_keys": 20,
                "ambiguous_keys": 0,
                "coverage": 0.333,  # Good coverage but...
                "error_rate": 0.2,
                "rejected": False,
                "rejection_reason": None,
                "error_details": [],
            }
        ],
    }

    recommendation = make_recommendation(analysis)

    assert recommendation["decision"] == "NO-GO"
    assert "error" in recommendation["reason"].lower()
