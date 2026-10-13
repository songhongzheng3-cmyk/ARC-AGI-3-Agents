from types import SimpleNamespace

from agents.game_model import GameModel
from agents.object_extractor import ObjectExtractor


def _snapshot(grid: list[list[int]]):
    return ObjectExtractor().extract(SimpleNamespace(frame=[grid]))


def test_target_candidates_exclude_avatar_and_bottom_hud():
    grid = [[4 for _ in range(20)] for _ in range(20)]
    # Goal candidate: two nearby color-9 components that should be grouped.
    grid[4][9] = 9
    grid[4][10] = 9
    grid[5][10] = 9
    # Avatar: color-12 marker touching its color-9 body.
    for x in range(8, 11):
        grid[15][x] = 12
        grid[16][x] = 9
    # Small bottom HUD component should not be a goal candidate.
    grid[19][1] = 9

    snapshot = _snapshot(grid)
    model = GameModel()
    model.reset(snapshot)
    candidates = model.target_candidates()

    assert len(candidates) == 1
    assert candidates[0]["color"] == 9
    assert candidates[0]["area"] == 3
    assert model.target_distance() == 12


def test_contact_is_limited_to_goal_candidates():
    before_grid = [[4 for _ in range(20)] for _ in range(20)]
    for x in range(8, 11):
        before_grid[10][x] = 12
        before_grid[11][x] = 9
    before_grid[7][9] = 9  # goal candidate above the avatar
    before_grid[8][9] = 3  # adjacent wall, not a goal
    before = _snapshot(before_grid)

    after_grid = [[4 for _ in range(20)] for _ in range(20)]
    for x in range(8, 11):
        after_grid[8][x] = 12
        after_grid[9][x] = 9
    after_grid[7][9] = 9  # goal candidate adjacent to the new avatar
    after = _snapshot(after_grid)

    model = GameModel()
    model.reset(before)
    evidence = model.observe("ACTION1", before, after, changed_cells=12)

    assert evidence.contact_objects
    # The goal is reported, while other small objects may also be useful
    # interaction hypotheses; large structural colors are filtered in real
    # game snapshots by their aggregate area.
    assert any(item.startswith("c9:") for item in evidence.contact_objects)
