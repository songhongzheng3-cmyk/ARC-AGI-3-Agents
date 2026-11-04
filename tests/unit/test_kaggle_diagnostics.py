"""Failures must leave actionable diagnostics, including partial game progress."""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from arcengine import GameAction, GameState

from scripts import kaggle_runner, kaggle_runtime


def frame(levels=0):
    return SimpleNamespace(
        state=GameState.NOT_FINISHED,
        levels_completed=levels,
        win_levels=7,
        frame=[[[levels, 1]]],
        available_actions=[1, 2, 3, 4],
    )


def setup_runner(tmp_path, monkeypatch, games):
    import arc_agi

    from agents.templates import structured_agent

    (tmp_path / "manifest.json").write_text(
        json.dumps({"agents": {"all": "StructuredAgent"}})
    )
    monkeypatch.setattr(kaggle_runner, "SOURCE_ROOT", tmp_path)
    arcade = Mock()
    arcade.get_environments.return_value = [SimpleNamespace(game_id=g) for g in games]
    arcade.open_scorecard.return_value = "test-card"
    arcade.close_scorecard.return_value.model_dump_json.return_value = json.dumps(
        {
            "api_key": "must-not-save",
            "score": 0,
        }
    )
    monkeypatch.setattr(arc_agi, "Arcade", lambda **_: arcade)

    class Agent:
        MAX_ACTIONS = 700

        def __init__(self, arc_env, game_id, **kwargs):
            self.env, self.game_id = arc_env, game_id
            self.action_counter, self.levels_completed = 0, 0
            self.state = GameState.NOT_FINISHED

        def main(self):
            raw = self.env.step(GameAction.ACTION1, data={"x": 3, "y": 5})
            self.action_counter += 1
            self.levels_completed = raw.levels_completed
            if self.game_id == "broken":
                raise RuntimeError("injected policy failure")

        def cleanup(self):
            pass

    monkeypatch.setattr(structured_agent, "StructuredAgent", Agent)
    return arcade


def test_discovery_failure_still_writes_report(tmp_path, monkeypatch):
    arcade = setup_runner(tmp_path, monkeypatch, [])
    assert kaggle_runner.main(["--output", str(tmp_path / "output")]) == 1
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["status"] == "failed"
    assert report["failed_stage"] == "discovering_games"
    assert "no matching environments" in report["error"]
    arcade.open_scorecard.assert_not_called()


def test_partial_progress_and_later_games_survive_policy_failure(tmp_path, monkeypatch):
    arcade = setup_runner(tmp_path, monkeypatch, ["broken", "working"])
    arcade.make.side_effect = [
        SimpleNamespace(observation_space=frame(), step=Mock(return_value=frame(1)))
        for _ in range(2)
    ]
    assert kaggle_runner.main(["--output", str(tmp_path / "output")]) == 1
    report = json.loads((tmp_path / "output/report.json").read_text())
    broken, working = report["runs"]
    assert broken["status"] == "failed"
    assert broken["levels_completed"] == broken["max_levels_completed"] == 1
    assert broken["actions"] == broken["responses"] == 1
    assert "injected policy failure" in broken["error"]
    assert working["status"] == "completed"
    arcade.close_scorecard.assert_called_once_with("test-card")
    assert "must-not-save" not in (tmp_path / "output/scorecard.json").read_text()
    events = [
        json.loads(line)
        for line in (tmp_path / "output/events.jsonl").read_text().splitlines()
    ]
    action = next(e for e in events if e["event"] == "action_requested")
    assert action["data"] == {"x": 3, "y": 5}
    assert any(e["event"] == "initial_frame" for e in events)


def test_scorecard_failure_does_not_erase_game_result(tmp_path, monkeypatch):
    arcade = setup_runner(tmp_path, monkeypatch, ["working"])
    arcade.make.return_value = SimpleNamespace(
        observation_space=frame(),
        step=Mock(return_value=frame(2)),
    )
    arcade.close_scorecard.side_effect = RuntimeError("close rejected")
    assert kaggle_runner.main(["--output", str(tmp_path / "output")]) == 1
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["status"] == "failed"
    assert report["runs"][0]["levels_completed"] == 2
    assert "close rejected" in report["scorecard_error"]


def test_missing_step_response_is_counted_without_changing_request():
    raw = frame()
    env = SimpleNamespace(observation_space=raw, step=Mock(side_effect=[None, raw]))
    events, row = [], {"game_id": "example"}
    observed = kaggle_runner.ObservedEnvironment(
        env,
        row,
        lambda kind, **data: events.append((kind, data)),
        lambda: None,
    )
    assert observed.step(GameAction.ACTION6, {"x": 7, "y": 9}) is None
    assert observed.step(GameAction.ACTION6, {"x": 7, "y": 9}) is raw
    assert row["step_requests"] == 2
    assert row["missing_responses"] == row["responses"] == 1
    assert row["action_counts"] == {"ACTION6": 2}
    env.step.assert_called_with(
        GameAction.ACTION6, data={"x": 7, "y": 9}, reasoning=None
    )


def test_runtime_gateway_timeout_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    monkeypatch.setattr(
        kaggle_runtime.urllib.request,
        "urlopen",
        Mock(side_effect=OSError("unreachable")),
    )
    with pytest.raises(RuntimeError, match="gateway did not become ready"):
        kaggle_runtime.run(tmp_path, tmp_path / "output", "http://unused/", timeout=0)
    status = json.loads((tmp_path / "output/runtime.json").read_text())
    assert status["stage"] == "failed"
    assert status["failed_stage"] == "waiting_for_gateway"


def test_runtime_retains_subprocess_output_and_exit_code(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    response = Mock()
    response.__enter__ = Mock(return_value=SimpleNamespace(status=200))
    response.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(
        kaggle_runtime.urllib.request, "urlopen", Mock(return_value=response)
    )
    (tmp_path / "kaggle_runner.py").write_text(
        "print('injected runtime crash', flush=True)\nraise SystemExit(3)\n"
    )
    with pytest.raises(subprocess.CalledProcessError):
        kaggle_runtime.run(tmp_path, tmp_path / "output", "http://unused/")
    status = json.loads((tmp_path / "output/runtime.json").read_text())
    assert status["runner_exit_code"] == 3
    assert status["failed_stage"] == "running_agent"
    assert "injected runtime crash" in (tmp_path / "output/runner.log").read_text()
