from unittest.mock import Mock, patch

import pytest
from arc_agi.scorecard import EnvironmentScorecard

from agents.swarm import Swarm
from agents.templates.random_agent import Random


@pytest.fixture
def mock_arcade():
    with patch("agents.swarm.Arcade") as arcade_type:
        arcade = arcade_type.return_value
        arcade.open_scorecard.return_value = "test-card-123"
        arcade.close_scorecard.return_value = EnvironmentScorecard(
            card_id="test-card-123"
        )
        yield arcade


@pytest.mark.unit
class TestSwarmInitialization:
    def test_swarm_init(self, mock_arcade):
        with patch.dict("os.environ", {"ARC_API_KEY": "test-api-key"}):
            swarm = Swarm(
                agent="random", ROOT_URL="https://example.com", games=["game1", "game2"]
            )

            assert swarm.agent_name == "random"
            assert swarm.ROOT_URL == "https://example.com"
            assert swarm.GAMES == ["game1", "game2"]
            assert swarm.agent_class == Random
            assert len(swarm.threads) == 0
            assert len(swarm.agents) == 0

            assert swarm.headers["X-API-Key"] == "test-api-key"
            assert swarm.headers["Accept"] == "application/json"
            assert swarm._arc is mock_arcade


@pytest.mark.unit
class TestSwarmScorecard:
    def test_open_scorecard(self, mock_arcade):
        swarm = Swarm(agent="random", ROOT_URL="https://example.com", games=["game1"])

        card_id = swarm.open_scorecard()
        assert card_id == "test-card-123"
        mock_arcade.open_scorecard.assert_called_once_with(tags=["agent", "random"])

    def test_close_scorecard(self, mock_arcade):
        swarm = Swarm(agent="random", ROOT_URL="https://example.com", games=["game1"])
        swarm.card_id = "test-card-123"

        scorecard = swarm.close_scorecard("test-card-123")
        assert isinstance(scorecard, EnvironmentScorecard)
        assert scorecard.card_id == "test-card-123"
        assert swarm.card_id is None
        mock_arcade.close_scorecard.assert_called_once_with("test-card-123")


@pytest.mark.unit
class TestSwarmAgentManagement:
    @patch("agents.swarm.Swarm.open_scorecard")
    @patch("agents.swarm.Swarm.close_scorecard")
    @patch("agents.swarm.Thread")
    @patch("agents.swarm.Arcade")
    def test_agent_threading(self, mock_arcade_type, mock_thread, mock_close, mock_open):
        mock_open.return_value = "test-card-123"
        mock_close.return_value = EnvironmentScorecard(card_id="test-card-123")
        mock_arcade_type.return_value.make.return_value = Mock()

        mock_thread_instances = [Mock() for _ in range(3)]
        mock_thread.side_effect = mock_thread_instances

        swarm = Swarm(
            agent="random",
            ROOT_URL="https://example.com",
            games=["game1", "game2", "game3"],
        )

        assert swarm.agent_name == "random"
        assert swarm.agent_class == Random
        assert swarm.GAMES == ["game1", "game2", "game3"]

        with patch.object(Random, "main") as mock_agent_main:
            mock_agent_main.return_value = None

            swarm.main()

            assert mock_thread.call_count == 3
            for mock_thread_instance in mock_thread_instances:
                mock_thread_instance.start.assert_called_once()
                mock_thread_instance.join.assert_called_once()


@pytest.mark.unit
class TestSwarmCleanup:
    def test_cleanup(self, mock_arcade):
        swarm = Swarm(
            agent="random", ROOT_URL="https://example.com", games=["game1", "game2"]
        )

        mock_agent1 = Mock()
        mock_agent2 = Mock()
        swarm.agents = [mock_agent1, mock_agent2]

        mock_session = Mock()
        swarm._session = mock_session

        scorecard = EnvironmentScorecard(card_id="test-card-123")
        swarm.cleanup(scorecard)

        mock_agent1.cleanup.assert_called_once_with(scorecard)
        mock_agent2.cleanup.assert_called_once_with(scorecard)

        mock_session.close.assert_called_once()

        mock_agent = Mock()
        swarm.agents = [mock_agent]

        swarm.cleanup()
        mock_agent.cleanup.assert_called_once_with(None)

        delattr(swarm, "_session")
        swarm.cleanup()


@pytest.mark.unit
class TestSwarmTags:
    def test_open_scorecard_with_custom_tags(self, mock_arcade):
        """Test that custom tags are sent when opening a scorecard"""
        custom_tags = ["experiment1", "version2", "test"]

        swarm = Swarm(
            agent="random",
            ROOT_URL="https://example.com",
            games=["game1"],
            tags=custom_tags,
        )

        card_id = swarm.open_scorecard()
        assert card_id == "test-card-123"

        mock_arcade.open_scorecard.assert_called_once_with(
            tags=custom_tags + ["agent", "random"]
        )

    def test_open_scorecard_with_empty_tags(self, mock_arcade):
        """Test that default tags are sent when no custom tags are provided"""
        swarm = Swarm(
            agent="random", ROOT_URL="https://example.com", games=["game1"], tags=[]
        )

        card_id = swarm.open_scorecard()
        assert card_id == "test-card-123"

        mock_arcade.open_scorecard.assert_called_once_with(tags=["agent", "random"])

    def test_open_scorecard_with_default_and_custom_tags(self, mock_arcade):
        """Test that tags include both defaults and custom tags when set from main.py"""
        custom_tags = ["experiment1", "version2"]

        swarm = Swarm(
            agent="random",
            ROOT_URL="https://example.com",
            games=["game1"],
            tags=custom_tags,
        )

        card_id = swarm.open_scorecard()
        assert card_id == "test-card-123"

        mock_arcade.open_scorecard.assert_called_once_with(
            tags=custom_tags + ["agent", "random"]
        )
