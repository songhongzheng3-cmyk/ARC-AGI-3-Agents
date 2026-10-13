from types import SimpleNamespace

from arcengine import GameAction

from agents.navigation_frontier_policy import NavigationFrontierPolicy


def _frame(grid, level=0):
    return SimpleNamespace(
        frame=[grid],
        levels_completed=level,
        available_actions=[
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ],
    )


def test_records_successful_path_on_level_change():
    policy = NavigationFrontierPolicy()
    policy.level = 0
    policy.action_history = ["ACTION3", "ACTION1"]
    snapshot = policy.extractor.extract([[4]])

    policy._reset_for_level(1, snapshot)

    assert policy.successful_paths[0] == ("ACTION3", "ACTION1")
    assert policy.action_history == []


def test_non_navigation_frame_does_not_take_over():
    grid = [[4 for _ in range(8)] for _ in range(8)]
    grid[3][3] = 12

    policy = NavigationFrontierPolicy()

    assert policy.choose(_frame(grid)) is None
    assert not policy.applicable


def test_intro_navigation_policy_yields_to_later_mechanism_levels():
    policy = NavigationFrontierPolicy()

    assert policy.choose(_frame([[3]], level=1)) is None
    assert policy.level is None
