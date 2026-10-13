from pathlib import Path
from types import SimpleNamespace

import pytest
from arcengine import ActionInput, GameAction, GameState

from agents import AVAILABLE_AGENTS
from agents.templates.ls20_solver_agent import Ls20SolverAgent, _ObservedEnvironment


class _FakeEnvironment:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self) -> str:
        self.reset_calls += 1
        return "fresh"

    def step(self, action: str) -> str:
        self.step_calls += 1
        return f"after-{action}"


def test_agent_is_registered() -> None:
    assert AVAILABLE_AGENTS["ls20solveragent"] is Ls20SolverAgent


def test_backend_path_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARC_LS20_SOLVER_PATH", raising=False)
    with pytest.raises(RuntimeError, match="ARC_LS20_SOLVER_PATH is required"):
        Ls20SolverAgent._backend_src()


def test_backend_path_must_contain_solver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ARC_LS20_SOLVER_PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="missing"):
        Ls20SolverAgent._backend_src()


def test_level_budgets_use_verified_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARC_LS20_LEVEL_BUDGETS", raising=False)
    assert Ls20SolverAgent._level_budgets() == {
        0: 100,
        1: 200,
        2: 250,
        3: 300,
        4: 400,
        5: 500,
        6: 600,
    }


def test_level_budgets_validate_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARC_LS20_LEVEL_BUDGETS", "100,200")
    with pytest.raises(RuntimeError, match="seven positive integers"):
        Ls20SolverAgent._level_budgets()


def test_observed_environment_reuses_existing_initial_observation() -> None:
    env = _FakeEnvironment()
    seen: list[str] = []
    wrapped = _ObservedEnvironment(env, "existing", seen.append)

    assert wrapped.reset() == "existing"
    assert env.reset_calls == 0
    assert seen == ["existing"]

    assert wrapped.step("ACTION1") == "after-ACTION1"
    assert env.step_calls == 1
    assert seen[-1] == "after-ACTION1"

    assert wrapped.reset() == "fresh"
    assert env.reset_calls == 1
    assert seen[-1] == "fresh"


def test_raw_frame_conversion_preserves_action_input() -> None:
    raw = SimpleNamespace(
        game_id="ls20-test",
        frame=[],
        state=GameState.NOT_FINISHED,
        levels_completed=2,
        win_levels=7,
        action_input=ActionInput(
            id=GameAction.ACTION4,
            data={"game_id": "ls20-test"},
            reasoning={"source": "test"},
        ),
        guid="guid",
        full_reset=False,
        available_actions=[1, 2, 3, 4],
    )

    frame = Ls20SolverAgent._convert_raw_frame_data(object(), raw)

    assert frame.action_input.id is GameAction.ACTION4
    assert frame.action_input.data["game_id"] == "ls20-test"
    assert frame.action_input.reasoning == {"source": "test"}
