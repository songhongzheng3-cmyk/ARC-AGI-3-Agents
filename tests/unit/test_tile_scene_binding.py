from types import SimpleNamespace

from agents.tile_scene_binding import TileSceneBinding

LAYOUT = {
    "origin": [2, 2],
    "cell_size": [3, 3],
    "cols": 8,
    "rows": 5,
    "floor_color": 1,
    "background_color": 2,
}


def picture(first=(1, 2), second=(4, 2), selected=0, changed_shape=False):
    grid = [[2] * 32 for _ in range(32)]
    for y in range(2, 17):
        for x in range(2, 26):
            grid[y][x] = 1
    for i, cell in enumerate((first, second)):
        cx, cy = 3 + cell[0] * 3, 3 + cell[1] * 3
        for y in range(cy - 1, cy + 2):
            for x in range(cx - 1, cx + 2):
                grid[y][x] = 14
        grid[cy][cx] = 0 if i == selected else 5
        if i == 0 and changed_shape:
            for y in range(cy - 1, cy + 2):
                grid[y][cx + 1] = 0
    # An identically shaped sprite in the HUD must not enter identity matching.
    for y in range(27, 30):
        for x in range(9, 12):
            grid[y][x] = 14
    grid[28][10] = 0
    return SimpleNamespace(frame=[grid])


def test_shape_change_keeps_identity_and_learns_selection_from_movement():
    binding = TileSceneBinding(picture(), LAYOUT)
    before = binding.parse(picture())
    assert before.active_identity is None
    after = binding.parse(picture(first=(2, 2), changed_shape=True))
    assert [(a.identity, a.cell) for a in after.actors] == [
        ("actor-0", (2, 2)),
        ("actor-1", (4, 2)),
    ]
    assert not after.warnings
    assert binding.calibrate_active_marker(before, after)
    assert (
        binding.parse(picture(first=(2, 2), changed_shape=True)).active_identity
        == "actor-0"
    )


def test_click_selection_preserves_ids_then_other_actor_moves():
    binding = TileSceneBinding(picture(), LAYOUT)
    before = binding.parse(picture())
    moved = binding.parse(picture(first=(2, 2)))
    assert binding.calibrate_active_marker(before, moved)
    new_marker = picture(first=(2, 2), selected=1)
    new_marker.frame[-1][9][9] = 4
    selected = binding.parse(new_marker)
    assert selected.active_identity == "actor-1"
    after = binding.parse(picture(first=(2, 2), second=(5, 2), selected=1))
    assert [(a.identity, a.cell) for a in after.actors] == [
        ("actor-0", (2, 2)),
        ("actor-1", (5, 2)),
    ]


def test_equal_cost_identity_assignment_stays_uncertain():
    binding = TileSceneBinding(picture(first=(1, 2), second=(3, 2)), LAYOUT)
    binding.parse(picture(first=(1, 2), second=(3, 2)))
    ambiguous = binding.parse(picture(first=(2, 1), second=(2, 3)))
    assert ambiguous.warnings == ("ambiguous_identity_assignment",)
    assert ambiguous.active_identity is None
    assert not ambiguous.actors
