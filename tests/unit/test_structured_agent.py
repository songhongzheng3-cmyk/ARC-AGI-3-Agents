from types import SimpleNamespace
from unittest.mock import Mock

from arcengine import GameAction, GameState

from agents.templates.structured_agent import StructuredAgent


def _agent():
    agent = StructuredAgent.__new__(StructuredAgent)
    agent.navigation_policy = Mock()
    agent.mechanism_tracker = Mock()
    agent.mechanism_policy = Mock()
    agent.last_action = "ACTION3"
    agent.fallback_counts = {}
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
