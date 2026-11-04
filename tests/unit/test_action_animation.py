from types import SimpleNamespace

import numpy as np
import pytest

from agents.mechanism_state_tracker import GridLayout, MechanismStateTracker
from scripts.audit_action_animation import audit_animation
from scripts.validate_entity_options import validate


def grid_with_player(col):
    grid = np.full((20, 20), 3)
    grid[5:7, col * 5 : col * 5 + 5] = 12
    grid[7:10, col * 5 : col * 5 + 5] = 9
    return grid


def bound_tracker():
    tracker = MechanismStateTracker()
    tracker.layout = GridLayout((0, 0), (5, 5), 4, 4, 3, 4, 0.95)
    tracker.previous_grid = grid_with_player(1)
    tracker.player_position = (1, 1)
    tracker.player_bbox = (5, 5, 9, 9)
    tracker.player_appearance = tracker._normalized_appearance(
        tracker.previous_grid[5:10, 5:10]
    )
    return tracker


def test_animation_tracks_actual_pixels_without_advancing_command_or_clock():
    tracker = bound_tracker()
    before = tracker.previous_grid.copy()
    frame = SimpleNamespace(
        action_input=SimpleNamespace(id=4),
        frame=[grid_with_player(col).tolist() for col in (2, 1, 0)],
    )
    report = audit_animation(tracker, frame)
    assert report["commanded_neighbor"] == (2, 1)
    assert [row["player"] for row in report["subframes"]] == [(2, 1), (1, 1), (0, 1)]
    assert report["final_player"] == (0, 1)
    assert tracker.player_position == (1, 1)
    assert tracker.player_bbox == (5, 5, 9, 9)
    np.testing.assert_array_equal(tracker.previous_grid, before)


def test_ambiguous_animation_appearance_does_not_invent_a_player_coordinate():
    tracker = bound_tracker()
    grid = grid_with_player(2)
    grid[5:10, :5] = grid_with_player(0)[5:10, :5]
    frame = SimpleNamespace(action_input=SimpleNamespace(id=4), frame=[grid.tolist()])
    report = audit_animation(tracker, frame)
    assert report["subframes"][0]["appearance_matches"] == 2
    assert report["final_player"] is None


@pytest.mark.parametrize("max_errors", (0, 4))
def test_checkpoint_error_limit_is_bounded_before_loading_any_recording(max_errors):
    with pytest.raises(ValueError, match="max_errors must be 1..3"):
        validate(None, None, freeze_row=620, budget=40, max_errors=max_errors)
