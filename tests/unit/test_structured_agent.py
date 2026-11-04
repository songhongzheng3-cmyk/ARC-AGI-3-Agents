from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

from arcengine import GameAction, GameState

from agents.episode_memory import EpisodeMemory
from agents.templates.structured_agent import StructuredAgent


def _agent():
    agent = StructuredAgent.__new__(StructuredAgent)
    agent.navigation_policy = Mock()
    agent.mechanism_tracker = Mock()
    agent.mechanism_policy = Mock()
    agent.last_action = "ACTION3"
    agent.fallback_counts = {}
    agent.fallback_clicks_by_level = defaultdict(set)
    agent.episode_memory = EpisodeMemory()
    return agent


def _frame(state=GameState.NOT_FINISHED):
    return SimpleNamespace(
        state=state,
        frame=[[[3, 3], [3, 3]]],
        levels_completed=1,
        available_actions=[1, 2, 3, 4],
    )


def test_prefers_navigation_then_records_selected_action():
    agent = _agent()
    frame = _frame()
    mechanism_state = object()
    agent.mechanism_tracker.observe.return_value = mechanism_state
    agent.mechanism_tracker.summary.return_value = "state"
    agent.navigation_policy.choose.return_value = GameAction.ACTION2
    agent.navigation_policy.summary.return_value = "nav"

    action = agent.choose_action([], frame)

    assert action is GameAction.ACTION2
    assert agent.last_action == "ACTION2"
    agent.mechanism_tracker.observe.assert_called_once_with(
        frame,
        action="ACTION3",
    )
    agent.mechanism_policy.choose.assert_not_called()


def test_uses_mechanism_policy_when_navigation_is_done():
    agent = _agent()
    frame = _frame()
    mechanism_state = object()
    agent.mechanism_tracker.observe.return_value = mechanism_state
    agent.mechanism_tracker.summary.return_value = "state"
    agent.navigation_policy.choose.return_value = None
    agent.mechanism_policy.choose.return_value = GameAction.ACTION4
    agent.mechanism_policy.summary.return_value = "mechanism"

    action = agent.choose_action([], frame)

    assert action is GameAction.ACTION4
    agent.mechanism_policy.choose.assert_called_once_with(
        mechanism_state,
        frame.available_actions,
        previous_action="ACTION3",
    )


def test_resets_without_querying_policies_after_game_over():
    agent = _agent()

    action = agent.choose_action([], _frame(GameState.GAME_OVER))

    assert action is GameAction.RESET
    agent.navigation_policy.choose.assert_not_called()
    agent.mechanism_tracker.observe.assert_not_called()


def test_default_budget_can_cover_validated_seven_level_reference():
    assert StructuredAgent.MAX_ACTIONS >= 637


def test_fallback_cycles_actions_when_only_resource_pixels_change():
    agent = _agent()
    state = SimpleNamespace(
        player_position=(9, 3),
        status_signature="010/110/011",
        status_color=14,
    )
    first = _frame()
    second = _frame()
    second.frame = [[[8, 8], [8, 8]]]

    assert agent._fallback(first, state) is GameAction.ACTION1
    assert agent._fallback(second, state) is GameAction.ACTION2


def test_fallback_keeps_distinct_player_positions_separate():
    agent = _agent()
    frame = _frame()
    first_state = SimpleNamespace(
        player_position=(9, 3),
        status_signature="010/110/011",
        status_color=14,
    )
    second_state = SimpleNamespace(
        player_position=(9, 4),
        status_signature="010/110/011",
        status_color=14,
    )

    assert agent._fallback(frame, first_state) is GameAction.ACTION1
    assert agent._fallback(frame, second_state) is GameAction.ACTION1


def test_fallback_executes_action6_with_extracted_coordinates():
    agent = _agent()
    frame = _frame()
    frame.available_actions = [6]
    frame.frame = [[[0] * 8 for _ in range(8)]]
    frame.frame[0][3][4] = 9
    frame.frame[0][3][5] = 9
    state = SimpleNamespace(
        player_position=None,
        status_signature="empty",
        status_color=None,
    )

    action = agent._fallback(frame, state)

    assert action is GameAction.ACTION6
    assert 0 <= action.action_data.x < 8
    assert 0 <= action.action_data.y < 8


def test_fallback_does_not_repeat_an_unchanged_click_coordinate():
    agent = _agent()
    frame = _frame()
    frame.available_actions = [6]
    frame.frame = [[[0] * 8 for _ in range(8)]]
    frame.frame[0][3][4] = 9
    frame.frame[0][3][5] = 9
    state = SimpleNamespace(
        player_position=None,
        status_signature="empty",
        status_color=None,
    )

    first = agent._fallback(frame, state)
    first_xy = (first.action_data.x, first.action_data.y)
    state_hash = agent.episode_memory.state_hash(frame)
    agent.episode_memory.record_action(
        first.name,
        {"x": first_xy[0], "y": first_xy[1]},
        state_hash,
        score=1,
    )
    agent.episode_memory.observe_frame(frame)
    second = agent._fallback(frame, state)

    assert (second.action_data.x, second.action_data.y) != first_xy


def test_fallback_click_dedup_survives_animated_grid_changes():
    agent = _agent()
    frame = _frame()
    frame.available_actions = [6]
    frame.frame = [[[0] * 8 for _ in range(8)]]
    frame.frame[0][3][4] = 9
    frame.frame[0][3][5] = 9
    state = SimpleNamespace(
        player_position=None,
        status_signature="empty",
        status_color=None,
    )
    first = agent._fallback(frame, state)
    first_xy = (first.action_data.x, first.action_data.y)

    animated = _frame()
    animated.available_actions = [6]
    animated.frame = [[row[:] for row in frame.frame[0]]]
    animated.frame[0][0][0] = 1
    second = agent._fallback(animated, state)

    assert (second.action_data.x, second.action_data.y) != first_xy


def test_fallback_uses_legal_click_after_candidate_budget_is_exhausted():
    agent = _agent()
    frame = _frame()
    frame.available_actions = [6]
    frame.frame = [[[0] * 8 for _ in range(8)]]
    state = SimpleNamespace(
        player_position=None,
        status_signature="empty",
        status_color=None,
    )
    agent.fallback_clicks_by_level[1].update(
        (x, 0) for x in range(StructuredAgent.MAX_FALLBACK_CLICK_TRIALS)
    )

    action = agent._fallback(frame, state)

    assert action is GameAction.ACTION6


def test_policy_action_not_in_available_actions_uses_fallback():
    agent = _agent()
    frame = _frame()
    frame.available_actions = [2]
    agent.mechanism_tracker.observe.return_value = object()
    agent.mechanism_tracker.summary.return_value = "state"
    agent.navigation_policy.choose.return_value = GameAction.ACTION4

    action = agent.choose_action([], frame)

    assert action is GameAction.ACTION2
