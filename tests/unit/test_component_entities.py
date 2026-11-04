from dataclasses import replace

import numpy as np

from agents.component_entities import extract_entities
from agents.mechanism_state_tracker import (
    GridLayout,
    MechanismState,
    MechanismStateTracker,
)
from agents.motion_context import MotionContextMemory
from scripts.audit_component_identity import audit_predictions


def make_state(grid, *, player=(5, 5), player_bbox=None):
    layout = GridLayout((0, 0), (5, 5), 6, 6, 3, 4, 0.95)
    objects = MechanismStateTracker._objects(grid, layout)
    return MechanismState(
        level=5,
        game_state="NOT_FINISHED",
        layout=layout,
        player_position=player,
        player_confidence=0.99,
        walkable_cells=frozenset(),
        blocked_cells=frozenset(),
        objects=objects,
        resource_level=80,
        resource_capacity=84,
        status_pattern=((1, 0), (1, 1)),
        status_color=14,
        matching_target_ids=(),
        player_bbox=player_bbox,
    )


def coloured_sprite(grid, x, y):
    grid[y : y + 3, x : x + 3] = [[9, 14, 14], [9, 0, 8], [12, 12, 8]]


def test_multicolour_fragment_cells_become_one_entity_with_stable_pixel_identity():
    first, second = np.full((30, 30), 3), np.full((30, 30), 3)
    coloured_sprite(first, 6, 6)
    coloured_sprite(second, 11, 6)
    before, after = make_state(first), make_state(second)
    assert len(before.objects) == 5
    one, two = extract_entities(before, first), extract_entities(after, second)
    assert len(one.objects) == len(two.objects) == 1
    assert one.objects[0].area == 9
    assert one.objects[0].shape_signature == two.objects[0].shape_signature
    assert one.objects[0].cell == (1, 1)
    assert two.objects[0].cell == (2, 1)
    assert len(one.memberships[0].component_ids) == 5


def test_diagonally_touching_parts_can_form_compact_entity():
    grid = np.full((30, 30), 3)
    grid[6, 6] = 0
    grid[7, 7:9] = 0
    grid[8, 7] = 0
    view = extract_entities(make_state(grid), grid)
    assert len(view.objects) == 1
    assert view.objects[0].area == 4
    assert len(view.memberships[0].component_ids) == 2


def test_unconnected_nearby_sprites_are_not_merged_by_rounded_cell():
    grid = np.full((30, 30), 3)
    grid[6, 6] = 8
    grid[6, 4] = 9
    view = extract_entities(make_state(grid), grid)
    assert len(view.objects) == 2
    assert view.objects[0].cell == view.objects[1].cell


def test_static_target_owns_its_inner_pixels_and_player_owns_overlaps():
    grid = np.full((30, 30), 3)
    grid[6:13, 6:13] = 5
    coloured_sprite(grid, 8, 8)
    coloured_sprite(grid, 20, 20)
    state = make_state(grid, player_bbox=(19, 19, 23, 23))
    assert any(o.static_target for o in state.objects)
    view = extract_entities(state, grid)
    assert not view.objects
    assert {reason for _, reason in view.exclusions} == {
        "target_owned",
        "player_owned_or_overlapped",
    }


def test_connected_sprites_larger_than_cell_are_ambiguous_not_arbitrarily_split():
    grid = np.full((30, 30), 3)
    grid[10:13, 10:14] = 8
    grid[10:13, 14:18] = 9
    view = extract_entities(make_state(grid), grid)
    assert not view.objects
    assert all(reason == "oversized_connected_group" for _, reason in view.exclusions)


def test_ambiguous_same_colour_mask_is_not_filled_in_as_entity():
    grid = np.full((30, 30), 3)
    grid[6:11, 6] = grid[6, 6:11] = grid[10, 6:11] = 8
    grid[8, 8] = 8
    view = extract_entities(make_state(grid), grid)
    assert any(reason == "ambiguous_pixel_mask" for _, reason in view.exclusions)


def test_duplicate_composite_entities_still_abstain_and_empty_view_never_falls_back():
    grid = np.full((30, 30), 3)
    coloured_sprite(grid, 6, 6)
    coloured_sprite(grid, 16, 16)
    state = make_state(grid)
    view = extract_entities(state, grid)
    assert len(view.objects) == 2
    assert not MotionContextMemory.visible_positions(
        replace(state, motion_entities=view.objects)
    )
    assert not MotionContextMemory.visible_positions(replace(state, motion_entities=()))


def test_entity_mode_is_explicit_and_survives_tracker_reset():
    assert not MechanismStateTracker().enable_entity_tracking
    tracker = MechanismStateTracker(enable_entity_tracking=True)
    tracker.reset()
    assert tracker.enable_entity_tracking


def test_frozen_audit_does_not_learn_palette_edges_from_suffix():
    grid = np.full((30, 30), 3)
    initial = make_state(grid)
    samples = [
        (row, replace(initial, status_color=14 if row < 4 else 8), "ACTION1")
        for row in range(8)
    ]
    audit = audit_predictions(samples)
    assert audit["freeze_row"] == 4
    assert not audit["palette_edges"]
    assert audit["counts"]["train"]["status_changes"] == 1
    assert not audit["moving_signatures"]
