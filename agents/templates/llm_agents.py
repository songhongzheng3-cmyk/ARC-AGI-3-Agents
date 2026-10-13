import json
import logging
import os
import textwrap
from typing import Any, Optional

import openai
from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient

from ..action_value_controller import ActionValueController
from ..agent import Agent
from ..bootstrap_policy import BootstrapPolicy
from ..episode_memory import EpisodeMemory
from ..hypothesis_manager import HypothesisManager
from ..mechanism_policy import MechanismPolicy
from ..mechanism_state_tracker import MechanismState, MechanismStateTracker
from ..navigation_frontier_policy import NavigationFrontierPolicy
from ..object_extractor import ObjectExtractor
from ..progress_evaluator import GameTypeDetector, RewardEvaluator, RewardVector
from ..transition_model import TransitionModel

logger = logging.getLogger()


class LLM(Agent):
    """An agent that uses a base LLM model to play games."""

    MAX_ACTIONS: int = 80
    DO_OBSERVATION: bool = True
    REASONING_EFFORT: Optional[str] = None
    MODEL_REQUIRES_TOOLS: bool = True

    # Keep the interactive context bounded.  The default values are deliberately
    # conservative for large reasoning models; override them in .env for ablations.
    MESSAGE_LIMIT: int = int(os.getenv("ARC_MESSAGE_LIMIT", "4"))
    MAX_MESSAGE_CHARS: int = int(os.getenv("ARC_MAX_MESSAGE_CHARS", "20000"))
    OBSERVE_EVERY: int = max(1, int(os.getenv("ARC_OBSERVE_EVERY", "2")))
    OBSERVE_MAX_TOKENS: int = int(os.getenv("ARC_OBSERVE_MAX_TOKENS", "768"))
    ACTION_MAX_TOKENS: int = int(os.getenv("ARC_ACTION_MAX_TOKENS", "256"))
    NOOP_LIMIT: int = max(1, int(os.getenv("ARC_NOOP_LIMIT", "2")))
    MEMORY_HISTORY_LEN: int = max(1, int(os.getenv("ARC_MEMORY_HISTORY_LEN", "8")))
    CLICK_CANDIDATE_LIMIT: int = max(
        1, int(os.getenv("ARC_CLICK_CANDIDATE_LIMIT", "12"))
    )
    ACTION_VALIDATION: bool = os.getenv("ARC_ACTION_VALIDATION", "1") != "0"
    BOOTSTRAP_ENABLED: bool = os.getenv("ARC_BOOTSTRAP_ENABLED", "1") != "0"
    BOOTSTRAP_MIN_STEPS: int = max(0, int(os.getenv("ARC_BOOTSTRAP_MIN_STEPS", "6")))
    BOOTSTRAP_MAX_STEPS: int = max(0, int(os.getenv("ARC_BOOTSTRAP_MAX_STEPS", "14")))
    BOOTSTRAP_ACTION6_TRIALS: int = max(
        0, int(os.getenv("ARC_BOOTSTRAP_ACTION6_TRIALS", "2"))
    )
    BOOTSTRAP_NO_INFORMATION_LIMIT: int = max(
        1, int(os.getenv("ARC_BOOTSTRAP_NO_INFORMATION_LIMIT", "3"))
    )
    VALUE_CONTROLLER_ENABLED: bool = os.getenv("ARC_VALUE_CONTROLLER", "1") != "0"
    VALUE_EXPLORATION: float = float(os.getenv("ARC_VALUE_EXPLORATION", "0.7"))
    VALUE_MAX_SAME_ACTION: int = max(
        1, int(os.getenv("ARC_VALUE_MAX_SAME_ACTION", "2"))
    )
    NAVIGATION_FRONTIER_ENABLED: bool = (
        os.getenv("ARC_NAVIGATION_FRONTIER", "1") != "0"
    )
    MECHANISM_POLICY_ENABLED: bool = os.getenv("ARC_MECHANISM_POLICY", "1") != "0"
    LLM_TIMEOUT: float = float(os.getenv("ARC_LLM_TIMEOUT", "90"))
    MODEL: str = "qwen3-235b-a22b"
    messages: list[dict[str, Any]]
    token_counter: int

    _latest_tool_call_id: str = "call_12345"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages = []
        self.token_counter = 0
        self._client: Optional[OpenAIClient] = None
        self._last_frame_signature: Optional[str] = None
        self._same_frame_count = 0
        self._last_action_name: Optional[str] = None
        self._last_action_key: Optional[tuple[Any, ...]] = None
        self._last_action_frame_signature: Optional[str] = None
        self._blocked_action_keys: set[tuple[Any, ...]] = set()
        self.memory = EpisodeMemory(history_limit=self.MEMORY_HISTORY_LEN)
        self.bootstrap = BootstrapPolicy(
            min_steps=self.BOOTSTRAP_MIN_STEPS,
            max_steps=self.BOOTSTRAP_MAX_STEPS,
            action6_trials=self.BOOTSTRAP_ACTION6_TRIALS,
            no_information_limit=self.BOOTSTRAP_NO_INFORMATION_LIMIT,
            enabled=self.BOOTSTRAP_ENABLED,
        )
        self.value_controller = ActionValueController(
            exploration=self.VALUE_EXPLORATION,
            max_same_action=self.VALUE_MAX_SAME_ACTION,
        )
        self.navigation_policy = NavigationFrontierPolicy(
            enabled=self.NAVIGATION_FRONTIER_ENABLED
        )
        self.mechanism_policy = MechanismPolicy(enabled=self.MECHANISM_POLICY_ENABLED)
        self.object_extractor = ObjectExtractor()
        self.game_type_detector = GameTypeDetector()
        self.reward_evaluator = RewardEvaluator()
        self.transition_model = TransitionModel(history_limit=self.MEMORY_HISTORY_LEN * 4)
        self.hypothesis_manager = HypothesisManager()
        self.mechanism_tracker = MechanismStateTracker()
        self._mechanism_state: MechanismState | None = None
        self._reasoning_snapshot = None
        self._reasoning_level: int | None = None
        self._memory_state_hash = ""
        self._memory_context = ""
        self._click_candidates: list[dict[str, Any]] = []
        self._last_action_frame: FrameData | None = None
        self._last_action_snapshot: Any = None
        self._last_reward: RewardVector | None = None
        self._game_type = "unknown"

    @property
    def name(self) -> str:
        obs = "with-observe" if self.DO_OBSERVATION else "no-observe"
        sanitized_model_name = self.MODEL.replace("/", "-").replace(":", "-")
        name = f"{super().name}.{sanitized_model_name}.{obs}"
        if self.REASONING_EFFORT:
            name += f".{self.REASONING_EFFORT}"
        return name

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment below to only let the agent play one time
                # latest_frame.state is GameState.GAME_OVER,
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""

        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

        if self._client is None:
            self._client = OpenAIClient(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                timeout=self.LLM_TIMEOUT,
                max_retries=0,
            )
        client = self._client

        # A repeated frame is usually a blocked move.  Avoid paying for another
        # long model call when the same action just produced no state change.
        frame_signature = self._frame_signature(latest_frame)
        self._memory_state_hash = self.memory.observe_frame(latest_frame)
        current_level = int(getattr(latest_frame, "levels_completed", 0))
        previous_action = None
        action_input = getattr(latest_frame, "action_input", None)
        action_id = getattr(action_input, "id", None)
        if action_id is not None:
            previous_action = str(getattr(action_id, "name", action_id)).upper()
        self._mechanism_state = self.mechanism_tracker.observe(
            latest_frame,
            action=previous_action,
        )
        logger.info(
            "Mechanism state %s",
            self.mechanism_tracker.summary(self._mechanism_state),
        )
        if self._reasoning_level is not None and current_level != self._reasoning_level:
            # Do not close an action from the previous level against this one.
            self.hypothesis_manager = HypothesisManager()
            self.transition_model = TransitionModel(
                history_limit=self.MEMORY_HISTORY_LEN * 4
            )
            self.game_type_detector.reset()
            self.reward_evaluator.reset()
            self.value_controller.reset()
            self._last_reward = None
        self._reasoning_level = current_level
        self._last_frame_for_reasoning = latest_frame
        reasoning_snapshot = self.object_extractor.extract(latest_frame)
        transition = self.transition_model.observe_state(
            self._memory_state_hash,
            reasoning_snapshot,
            latest_frame,
        )
        if transition is not None:
            self.hypothesis_manager.observe(transition)
            if self._last_action_frame is not None and self._last_action_snapshot is not None:
                changed_cells = transition.changed_cells
                self.game_type_detector.observe(
                    transition.action,
                    self._last_action_snapshot,
                    reasoning_snapshot,
                    changed_cells,
                    transition.score_delta,
                )
                game_type, __ = self.game_type_detector.classify()
                self._game_type = game_type.value
                self._last_reward = self.reward_evaluator.evaluate(
                    self._last_action_frame,
                    latest_frame,
                    self._last_action_snapshot,
                    reasoning_snapshot,
                    game_type,
                    self._memory_state_hash,
                    action=transition.action,
                    repeated_action=(
                        len(self.memory.history) >= 2
                        and self.memory.history[-1].action
                        == self.memory.history[-2].action
                    ),
                )
                self.value_controller.observe(
                    transition.action,
                    transition.data,
                    self._last_reward.total,
                    task_signal=(
                        transition.score_delta > 0
                        or self._last_reward.terminal > 0
                        or self._mechanism_state.task_signal_observed
                    ),
                )
                logger.info("Progress %s", self._last_reward.summary())
        self.bootstrap.observe(current_level, transition)
        self._reasoning_snapshot = reasoning_snapshot
        tried_clicks = set(self.memory.tried_clicks(self._memory_state_hash))
        self._click_candidates = self.memory.click_candidates(
            latest_frame,
            tried=tried_clicks,
            max_candidates=self.CLICK_CANDIDATE_LIMIT,
        )
        self._memory_context = self.memory.prompt_context(
            self._memory_state_hash,
            getattr(latest_frame, "available_actions", []),
            click_candidates=self._click_candidates,
        )
        reasoning_context = self._reasoning_context(reasoning_snapshot)
        if reasoning_context:
            self._memory_context += "\n" + reasoning_context
        if self._mechanism_state is not None:
            self._memory_context += "\n" + self.mechanism_tracker.summary(
                self._mechanism_state
            )
        logger.info(
            "Memory state=%s candidates=%s tried_clicks=%s",
            self._memory_state_hash[:8],
            len(self._click_candidates),
            len(tried_clicks),
        )
        if frame_signature == self._last_frame_signature:
            self._same_frame_count += 1
        else:
            self._same_frame_count = 0
            self._blocked_action_keys.clear()
        self._last_frame_signature = frame_signature
        if (
            self._same_frame_count >= self.NOOP_LIMIT
            and self._last_action_name not in (None, GameAction.RESET.name)
        ):
            recovery = self._recovery_action(latest_frame)
            if recovery is not None:
                logger.warning(
                    "No-op circuit breaker: frame unchanged %s times; "
                    "using %s without another LLM call",
                    self._same_frame_count,
                    recovery.name,
                )
                self._remember_action(recovery, latest_frame)
                return self._finalize_action(recovery, frame_signature)

        functions = self.build_functions()
        tools = self.build_tools()
        # RESET is only meaningful before the first move or after GAME_OVER.
        # Removing it from the schema prevents the model from wasting a run.
        if not self._reset_allowed(latest_frame):
            functions = [f for f in functions if f["name"] != GameAction.RESET.name]
            tools = [
                t for t in tools
                if t.get("function", {}).get("name") != GameAction.RESET.name
            ]

        # if latest_frame.state in [GameState.NOT_PLAYED]:
        if len(self.messages) == 0:
            # have to manually trigger the first reset to kick off agent
            user_prompt = self._prompt_with_memory(latest_frame)
            message0 = {"role": "user", "content": user_prompt}
            self.push_message(message0)
            if self.MODEL_REQUIRES_TOOLS:
                message1 = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": self._latest_tool_call_id,
                            "type": "function",
                            "function": {
                                "name": GameAction.RESET.name,
                                "arguments": json.dumps({}),
                            },
                        }
                    ],
                }
            else:
                message1 = {
                    "role": "assistant",
                    "function_call": {"name": "RESET", "arguments": json.dumps({})},  # type: ignore
                }
            self.push_message(message1)
            action = GameAction.RESET
            self._remember_action(action, latest_frame)
            return self._finalize_action(action, frame_signature)

        navigation_action = self.navigation_policy.choose(latest_frame)
        if navigation_action is not None:
            logger.info(
                "Navigation frontier action=%s %s",
                navigation_action.name,
                self.navigation_policy.summary(),
            )
            self._remember_action(navigation_action, latest_frame)
            return self._finalize_action(navigation_action, frame_signature)

        mechanism_action = self.mechanism_policy.choose(
            self._mechanism_state,
            getattr(latest_frame, "available_actions", []),
        )
        if mechanism_action is not None:
            logger.info(
                "Mechanism policy action=%s %s",
                mechanism_action.name,
                self.mechanism_policy.summary(),
            )
            self._remember_action(mechanism_action, latest_frame)
            return self._finalize_action(mechanism_action, frame_signature)

        bootstrap_action = self.bootstrap.choose(
            getattr(latest_frame, "available_actions", []),
            self._click_candidates,
            latest_frame.state,
            int(latest_frame.levels_completed),
        )
        if bootstrap_action is not None:
            logger.info(
                "Bootstrap action=%s step=%s/%s",
                bootstrap_action.name,
                self.bootstrap.steps + 1,
                self.bootstrap.max_steps,
            )
            self._remember_action(bootstrap_action, latest_frame)
            return self._finalize_action(bootstrap_action, frame_signature)

        if self.VALUE_CONTROLLER_ENABLED:
            ranked = self._available_ranked_actions(latest_frame)
            selected = self.value_controller.choose(ranked)
            if selected is not None:
                action = self._action_from_ranked_name(selected, latest_frame)
                action = self._validate_action(action, latest_frame)
                logger.info(
                    "Value controller action=%s candidates=%s values=%s",
                    action.name,
                    ",".join(ranked[:8]),
                    self.value_controller.summary(),
                )
                self._remember_action(action, latest_frame)
                return self._finalize_action(action, frame_signature)

        # let the agent comment observations before choosing action
        # on the first turn, this will be in response to RESET action
        function_name = latest_frame.action_input.id.name
        function_response = self.build_func_resp_prompt(latest_frame)
        self._ensure_tool_response_parent(latest_frame)
        if self.MODEL_REQUIRES_TOOLS:
            message2 = {
                "role": "tool",
                "tool_call_id": self._latest_tool_call_id,
                "content": str(function_response),
            }
        else:
            message2 = {
                "role": "function",
                "name": function_name,
                "content": str(function_response),
            }
        self.push_message(message2)

        should_observe = self.DO_OBSERVATION and (
            self.action_counter <= 1 or self.action_counter % self.OBSERVE_EVERY == 0
        )
        if should_observe:
            self._ensure_chat_history(latest_frame)
            logger.info("Sending to Assistant for observation...")
            logger.info("LLM context: messages=%s chars=%s", len(self.messages), self._message_chars())
            try:
                create_kwargs: dict[str, Any] = {
                    "model": self.MODEL,
                    "messages": self.messages,
                    "max_tokens": self.OBSERVE_MAX_TOKENS,
                    "temperature": 0,
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                response = client.chat.completions.create(**create_kwargs)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e
            self.track_response_tokens(response, response.choices[0].message.content)
            message3 = {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
            logger.info(f"Assistant: {response.choices[0].message.content}")
            self.push_message(message3)

        # now ask for the next action
        user_prompt = self._prompt_with_memory(latest_frame)
        message4 = {"role": "user", "content": user_prompt}
        self.push_message(message4)
        self._ensure_chat_history(latest_frame)

        name = GameAction.ACTION5.name  # default action if LLM doesnt call one
        arguments = None
        message5 = None

        if self.MODEL_REQUIRES_TOOLS:
            logger.info("Sending to Assistant for action...")
            logger.info("LLM context: messages=%s chars=%s", len(self.messages), self._message_chars())
            try:
                create_kwargs = {
                    "model": self.MODEL,
                    "messages": self.messages,
                    "tools": tools,
                    "tool_choice": "required",
                    "max_tokens": self.ACTION_MAX_TOKENS,
                    "temperature": 0,
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                response = client.chat.completions.create(**create_kwargs)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e
            self.track_response_tokens(response)
            message5 = response.choices[0].message
            logger.debug(f"... got response {message5}")
            if not message5.tool_calls:
                logger.warning("LLM returned no tool call; using recovery action")
                recovery = self._recovery_action(latest_frame)
                if recovery is not None:
                    self._remember_action(recovery, latest_frame)
                    return self._finalize_action(recovery, frame_signature)
                self._remember_action(GameAction.ACTION5, latest_frame)
                return self._finalize_action(GameAction.ACTION5, frame_signature)
            tool_call = message5.tool_calls[0]
            self._latest_tool_call_id = tool_call.id
            logger.info(
                "LLM raw action: name=%s arguments=%s",
                tool_call.function.name,
                tool_call.function.arguments,
            )
            name = tool_call.function.name
            arguments = tool_call.function.arguments

            # sometimes the model will call multiple tools which isnt allowed
            extra_tools = message5.tool_calls[1:]
            for tc in extra_tools:
                logger.info(
                    "Error: assistant called more than one action, only using the first."
                )
                message_extra = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Error: assistant can only call one action (tool) at a time. default to only the first chosen action.",
                }
                self.push_message(message_extra)
        else:
            logger.info("Sending to Assistant for action...")
            logger.info("LLM context: messages=%s chars=%s", len(self.messages), self._message_chars())
            try:
                create_kwargs = {
                    "model": self.MODEL,
                    "messages": self.messages,
                    "functions": functions,
                    "function_call": "auto",
                    "max_tokens": self.ACTION_MAX_TOKENS,
                    "temperature": 0,
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                response = client.chat.completions.create(**create_kwargs)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e
            self.track_response_tokens(response)
            message5 = response.choices[0].message
            function_call = message5.function_call
            logger.debug(f"Assistant: {function_call.name} {function_call.arguments}")
            name = function_call.name
            arguments = function_call.arguments

        if message5:
            self.push_message(message5)
        action_id = name
        if arguments:
            try:
                data = json.loads(arguments) or {}
            except Exception as e:
                data = {}
                logger.warning(f"JSON parsing error on LLM function response: {e}")
        else:
            data = {}

        try:
            action = GameAction.from_name(action_id)
            action.set_data(data)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Invalid LLM action %r/%r: %s", action_id, data, exc)
            recovery = self._recovery_action(latest_frame)
            if recovery is not None:
                action = recovery
            else:
                action = GameAction.ACTION5

        if action is GameAction.RESET and not self._reset_allowed(latest_frame):
            logger.warning(
                "Ignoring RESET during an active game (state=%s)",
                latest_frame.state.name,
            )
            action = self._recovery_action(latest_frame) or GameAction.ACTION1

        if self.ACTION_VALIDATION:
            action = self._validate_action(action, latest_frame)

        action_key = self._action_key(action)
        click = self._click_coordinates(action)
        if click is not None and click in set(self.memory.tried_clicks(self._memory_state_hash)):
            logger.warning("Skipping previously ineffective click: (%s, %s)", *click)
            action = self._recovery_action(latest_frame) or GameAction.ACTION1
        elif action_key in self._blocked_action_keys:
            logger.warning("Skipping previously ineffective action: %s", action_key)
            action = self._recovery_action(latest_frame) or GameAction.ACTION1
        logger.info("Parsed action: %s data=%s", action.name, data)
        self._remember_action(action, latest_frame)
        return self._finalize_action(action, frame_signature)

    def _frame_signature(self, frame: FrameData) -> str:
        """Return a compact, stable signature for no-op detection."""
        return json.dumps(
            {
                "state": frame.state.name,
                "levels": frame.levels_completed,
                "frame": frame.frame,
            },
            separators=(",", ":"),
        )

    @staticmethod
    def _action_key(action: GameAction) -> tuple[Any, ...]:
        """Identify an action including coordinates for click actions."""
        if action.is_complex():
            data = action.action_data
            return (action.name, getattr(data, "x", None), getattr(data, "y", None))
        return (action.name,)

    @staticmethod
    def _click_coordinates(action: GameAction) -> Optional[tuple[int, int]]:
        if action is not GameAction.ACTION6:
            return None
        data = action.action_data
        return (int(getattr(data, "x", 0)), int(getattr(data, "y", 0)))

    @staticmethod
    def _reset_allowed(frame: FrameData) -> bool:
        return frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER)

    def _finalize_action(self, action: GameAction, frame_signature: str) -> GameAction:
        """Remember the action before it is submitted to the environment."""
        # If the previous action left the frame unchanged, remember that exact
        # action as ineffective so the model cannot repeat it forever.
        if (
            self._last_action_frame_signature == frame_signature
            and self._last_action_key is not None
        ):
            self._blocked_action_keys.add(self._last_action_key)
        self._last_action_name = action.name
        self._last_action_key = self._action_key(action)
        self._last_action_frame_signature = frame_signature
        return action

    def _validate_action(self, action: GameAction, frame: FrameData) -> GameAction:
        """Reject only actions contradicted by exact local evidence."""
        available = getattr(frame, "available_actions", []) or []
        legal_names: set[str] = set()
        for value in available:
            if isinstance(value, GameAction):
                legal_names.add(value.name)
                continue
            try:
                legal_names.add(GameAction.from_id(int(value)).name)
            except (KeyError, TypeError, ValueError):
                legal_names.add(str(value).upper())
        if legal_names and action.name not in legal_names:
            recovery = self._recovery_action(frame)
            if recovery is not None:
                logger.warning("Validator replaced unavailable action %s with %s", action.name, recovery.name)
                return recovery

        if action is GameAction.ACTION6 and self._click_candidates:
            click = self._click_coordinates(action)
            candidates = {
                (int(item["x"]), int(item["y"])) for item in self._click_candidates
            }
            # An out-of-grid or non-candidate point is less useful than a
            # deterministic object-derived point, unless no candidate exists.
            if click is None or click not in candidates:
                replacement = self._candidate_action(self._click_candidates[0])
                logger.warning(
                    "Validator replaced ACTION6(%s,%s) with candidate (%s,%s)",
                    *(click or ("?", "?")),
                    int(self._click_candidates[0]["x"]),
                    int(self._click_candidates[0]["y"]),
                )
                return replacement

        selected_key = self._action_key(action)
        if selected_key in self._blocked_action_keys:
            recovery = self._recovery_action(frame)
            if recovery is not None:
                logger.warning("Validator replaced blocked action %s with %s", action.name, recovery.name)
                return recovery

        ranked = self.transition_model.rank_actions(
            self._memory_state_hash,
            self._reasoning_snapshot,
            self._available_ranked_actions(frame),
        ) if self._reasoning_snapshot is not None else []
        if ranked and selected_key == (action.name,):
            selected_score = self.transition_model.score_action(
                self._memory_state_hash, self._reasoning_snapshot, action.name
            )[0]
            top = ranked[0]
            top_score = self.transition_model.score_action(
                self._memory_state_hash, self._reasoning_snapshot, top
            )[0]
            if selected_score < 0 and top_score > selected_score:
                logger.warning("Validator replaced low-value action %s with %s", action.name, top)
                return self._action_from_ranked_name(top, frame)
        return action

    def _available_ranked_actions(self, frame: FrameData) -> list[str]:
        ranked = self.memory.ranked_actions(
            self._memory_state_hash,
            getattr(frame, "available_actions", []),
            click_candidates=self._click_candidates,
        )
        return ranked

    @staticmethod
    def _candidate_action(candidate: dict[str, Any]) -> GameAction:
        action = GameAction.ACTION6
        action.set_data({"x": int(candidate["x"]), "y": int(candidate["y"])})
        return action

    def _action_from_ranked_name(self, name: str, frame: FrameData) -> GameAction:
        if name.startswith("ACTION6("):
            coords = name.removeprefix("ACTION6(").removesuffix(")").split(",")
            return self._candidate_action({"x": int(coords[0]), "y": int(coords[1])})
        try:
            return GameAction.from_name(name)
        except (KeyError, ValueError):
            return self._recovery_action(frame) or GameAction.ACTION1

    def _remember_action(self, action: GameAction, frame: FrameData) -> None:
        """Record the action in exact episode memory before environment.step."""
        data = action.action_data.model_dump()
        self.memory.record_action(
            action.name,
            data,
            self._memory_state_hash,
            int(frame.levels_completed),
        )
        if self._reasoning_snapshot is not None:
            self.transition_model.record_action(
                self._memory_state_hash,
                self._reasoning_snapshot,
                frame,
                action.name,
                data,
                int(frame.levels_completed),
            )
        self._last_action_frame = frame
        self._last_action_snapshot = self._reasoning_snapshot

    def _reasoning_context(self, snapshot: Any) -> str:
        """Build a bounded object/rule summary for the model prompt."""
        lines = [
            "objects="
            + ", ".join(
                f"{item.object_id}/shape={item.shape}/area={item.area}"
                for item in snapshot.objects[:12]
            )
            if snapshot.objects
            else "objects=none",
        ]
        transition_context = self.transition_model.prompt_context(self._memory_state_hash)
        hypothesis_context = self.hypothesis_manager.prompt_context()
        if transition_context:
            lines.append(transition_context)
        if hypothesis_context:
            lines.append(hypothesis_context)
        lines.append(f"game_type={self._game_type}")
        if self._last_reward is not None:
            lines.append("last_reward=" + self._last_reward.summary())
        available = getattr(self._last_frame_for_reasoning, "available_actions", []) if hasattr(self, "_last_frame_for_reasoning") else []
        if available:
            ranked = self.memory.ranked_actions(
                self._memory_state_hash,
                available,
                click_candidates=self._click_candidates,
            )
            policy_ranked = self.transition_model.rank_actions(
                self._memory_state_hash,
                snapshot,
                ranked,
            )
            lines.append("policy_ranked_actions=" + ", ".join(policy_ranked[:12]))
        return "\n".join(lines)

    def _ensure_tool_response_parent(self, frame: FrameData) -> None:
        """Keep a valid assistant tool call before each tool response."""
        if not self.messages:
            self.push_message(
                {"role": "user", "content": self._prompt_with_memory(frame)}
            )
        last = self.messages[-1] if self.messages else None
        if self._message_role(last) == "assistant" and isinstance(last, dict):
            calls = last.get("tool_calls") or []
            if calls:
                name = calls[0].get("function", {}).get("name")
                if name == frame.action_input.id.name:
                    return
        self._latest_tool_call_id = f"local_{self.action_counter}"
        self.push_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": self._latest_tool_call_id,
                        "type": "function",
                        "function": {
                            "name": frame.action_input.id.name,
                            "arguments": json.dumps(frame.action_input.data or {}),
                        },
                    }
                ],
            }
        )

    def _ensure_chat_history(self, frame: FrameData) -> None:
        """Recover a minimal valid chat when aggressive trimming removed context."""
        if not self.messages or not any(
            self._message_role(message) == "user" for message in self.messages
        ):
            self.messages = [
                {"role": "user", "content": self._prompt_with_memory(frame)}
            ]
            return
        # Keep the current tool/observation messages intact.  If trimming left
        # an assistant/tool message at the front, add a short user anchor rather
        # than deleting the frame that the model needs to act on.
        if self._message_role(self.messages[0]) != "user":
            self.messages.insert(
                0, {"role": "user", "content": self._prompt_with_memory(frame)}
            )

    def _recovery_action(self, frame: FrameData) -> Optional[GameAction]:
        """Choose a different legal simple action after repeated no-ops."""
        available = getattr(frame, "available_actions", None)
        candidates: list[GameAction] = []
        if available:
            for value in available:
                try:
                    if isinstance(value, GameAction):
                        candidate = value
                    elif isinstance(value, int):
                        candidate = GameAction.from_id(value)
                    else:
                        candidate = GameAction.from_name(str(value))
                except (KeyError, TypeError, ValueError):
                    continue
                if not candidate.is_complex() and candidate is not GameAction.RESET:
                    candidates.append(candidate)
        if not candidates:
            candidates = [
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
                GameAction.ACTION5,
            ]
        for candidate in candidates:
            if self._action_key(candidate) not in self._blocked_action_keys:
                return candidate
        return candidates[0] if candidates else None

    def track_tokens(self, tokens: int, message: str = "") -> None:
        self.token_counter += tokens
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "tokens": tokens,
                    "total_tokens": self.token_counter,
                    "assistant": message,
                }
            )
        logger.info(f"Received {tokens} tokens, new total {self.token_counter}")
        # handle tool to debug messages:
        # with open("messages.json", "w") as f:
        #     json.dump(
        #         [
        #             msg if isinstance(msg, dict) else msg.model_dump()
        #             for msg in self.messages
        #         ],
        #         f,
        #         indent=2,
        #     )

    def push_message(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Push a message onto stack, store up to MESSAGE_LIMIT with FIFO."""
        self.messages.append(message)
        self._trim_message_history()
        return self.messages

    @staticmethod
    def _message_role(message: Any) -> Optional[str]:
        if isinstance(message, dict):
            return message.get("role")
        return getattr(message, "role", None)

    @staticmethod
    def _message_chars_for(message: Any) -> int:
        if isinstance(message, dict):
            content = message.get("content", "")
            return len(str(content)) + len(str(message.get("tool_calls", "")))
        try:
            return len(json.dumps(message.model_dump(), default=str))
        except AttributeError:
            return len(str(message))

    def _message_chars(self) -> int:
        return sum(self._message_chars_for(message) for message in self.messages)

    def _trim_message_history(self) -> None:
        """Bound both message count and serialized size without orphaned tools."""
        while self.messages and (
            len(self.messages) > self.MESSAGE_LIMIT
            or self._message_chars() > self.MAX_MESSAGE_CHARS
        ):
            self.messages.pop(0)
        # A tool result must be paired with its preceding assistant tool call.
        # Drop leading results once after size trimming; don't cascade-pop the
        # whole conversation when a local recovery generated several results.
        while self.messages and self._message_role(self.messages[0]) == "tool":
            self.messages.pop(0)

    def build_functions(self) -> list[dict[str, Any]]:
        """Build JSON function description of game actions for LLM."""
        empty_params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        functions: list[dict[str, Any]] = [
            {
                "name": GameAction.RESET.name,
                "description": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION1.name,
                "description": "Send this simple input action (1, W, Up).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION2.name,
                "description": "Send this simple input action (2, S, Down).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION3.name,
                "description": "Send this simple input action (3, A, Left).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION4.name,
                "description": "Send this simple input action (4, D, Right).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION5.name,
                "description": "Send this simple input action (5, Enter, Spacebar, Delete).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION6.name,
                "description": "Send this complex input action (6, Click, Point). Prefer one of the local click_candidates in the prompt; do not repeat no_change_clicks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "string",
                            "description": "Coordinate X which must be Int<0,63>",
                        },
                        "y": {
                            "type": "string",
                            "description": "Coordinate Y which must be Int<0,63>",
                        },
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
            },
        ]
        return functions

    def build_tools(self) -> list[dict[str, Any]]:
        """Support models that expect tool_call format."""
        functions = self.build_functions()
        tools: list[dict[str, Any]] = []
        for f in functions:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f["name"],
                        "description": f["description"],
                        "parameters": f.get("parameters", {}),
                        "strict": True,
                    },
                }
            )
        return tools

    def track_response_tokens(self, response: Any, message: str = "") -> None:
        """Record prompt and completion usage separately for latency tuning."""
        usage = getattr(response, "usage", None)
        total = int(getattr(usage, "total_tokens", 0) or 0)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        self.track_tokens(total, message)
        logger.info(
            "LLM usage: prompt=%s completion=%s total=%s",
            prompt,
            completion,
            total,
        )

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        prompt = textwrap.dedent(
            """
# State:
{state}

# Score:
{score}

# Last action:
{last_action}

# Frame:
{latest_frame}

# TURN:
Return at most 120 words. Do not restate the grid. Mention only the last action's
effect, one verified fact, and the next action hypothesis.
        """.format(
                latest_frame=self.pretty_print_3d(latest_frame.frame),
                score=latest_frame.levels_completed,
                state=latest_frame.state.name,
                last_action=latest_frame.action_input.id.name,
            )
        )
        if self._memory_context:
            prompt += f"\n\n# STRUCTURED MEMORY:\n{self._memory_context}"
        return prompt

    def _prompt_with_memory(self, latest_frame: FrameData) -> str:
        prompt = self.build_user_prompt(latest_frame)
        if self._memory_context:
            prompt += f"\n\n# STRUCTURED MEMORY:\n{self._memory_context}"
        return prompt

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Build the user prompt for the LLM. Override this method to customize the prompt."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

Current state: {state}. The previous action was {last_action}. Do not call
RESET during an active game; use it only before the first move or after
GAME_OVER.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action. Use the tool call only; do not emit a long explanation.
        """.format(
            state=latest_frame.state.name,
            last_action=latest_frame.action_input.id.name,
        )
        )

    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
            lines.append("")
        return "\n".join(lines)

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        if self._cleanup:
            if hasattr(self, "recorder") and not self.is_playback:
                meta = {
                    "llm_user_prompt": self.build_user_prompt(self.frames[-1]),
                    "llm_tools": self.build_tools()
                    if self.MODEL_REQUIRES_TOOLS
                    else self.build_functions(),
                    "llm_tool_resp_prompt": self.build_func_resp_prompt(
                        self.frames[-1]
                    ),
                }
                self.recorder.record(meta)
        super().cleanup(*args, **kwargs)


class ReasoningLLM(LLM, Agent):
    """An LLM agent that uses o4-mini and captures reasoning metadata in the action.reasoning field."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MODEL = "o4-mini"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field
        action.reasoning = {
            "model": self.MODEL,
            "action_chosen": action.name,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.levels_completed,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "response_preview": self._last_response_content[:200] + "..."
            if len(self._last_response_content) > 200
            else self._last_response_content,
        }

        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Override to capture reasoning token information from reasoning models."""
        super().track_tokens(tokens, message)

        # Store the response content for reasoning context (avoid empty or JSON strings)
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens

    def capture_reasoning_from_response(self, response: Any) -> None:
        """Helper method to capture reasoning tokens from OpenAI API response.

        This should be called from the parent class if we have access to the raw response.
        For reasoning models, reasoning tokens are in response.usage.completion_tokens_details.reasoning_tokens
        """
        if hasattr(response, "usage") and hasattr(
            response.usage, "completion_tokens_details"
        ):
            if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                self._last_reasoning_tokens = (
                    response.usage.completion_tokens_details.reasoning_tokens
                )
                self._total_reasoning_tokens += self._last_reasoning_tokens
                logger.debug(
                    f"Captured {self._last_reasoning_tokens} reasoning tokens from {self.MODEL} response"
                )


class FastLLM(LLM, Agent):
    """Similar to LLM, but skips observations."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = False
    MODEL = "gpt-4o-mini"

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
        """.format()
        )


class GuidedLLM(LLM, Agent):
    """Similar to LLM, with explicit human-provided rules in the user prompt to increase success rate."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = True
    MODEL = "o3"
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 10
    REASONING_EFFORT = "high"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field
        action.reasoning = {
            "model": self.MODEL,
            "action_chosen": action.name,
            "reasoning_effort": self.REASONING_EFFORT,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.levels_completed,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "agent_type": "guided_llm",
            "game_rules": "locksmith",
            "response_preview": self._last_response_content[:200] + "..."
            if len(self._last_response_content) > 200
            else self._last_response_content,
        }

        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Override to capture reasoning token information from o3 models."""
        super().track_tokens(tokens, message)

        # Store the response content for reasoning context (avoid empty or JSON strings)
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens

    def capture_reasoning_from_response(self, response: Any) -> None:
        """Helper method to capture reasoning tokens from OpenAI API response.

        This should be called from the parent class if we have access to the raw response.
        For o3 models, reasoning tokens are in response.usage.completion_tokens_details.reasoning_tokens
        """
        if hasattr(response, "usage") and hasattr(
            response.usage, "completion_tokens_details"
        ):
            if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                self._last_reasoning_tokens = (
                    response.usage.completion_tokens_details.reasoning_tokens
                )
                self._total_reasoning_tokens += self._last_reasoning_tokens
                logger.debug(
                    f"Captured {self._last_reasoning_tokens} reasoning tokens from o3 response"
                )

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

You are playing a game called LockSmith. Rules and strategy:
* RESET: start over, ACTION1: move up, ACTION2: move down, ACTION3: move left, ACTION4: move right (ACTION5 and ACTION6 do nothing in this game)
* you may may one action per turn
* your goal is find and collect a matching key then touch the exit door
* 6 levels total, score shows which level, complete all levels to win (grid row 62)
* start each level with limited energy. you GAME_OVER if you run out (grid row 61)
* the player is a 4x4 square: [[X,X,X,X],[0,0,0,X],[4,4,4,X],[4,4,4,X]] where X is transparent to the background
* the grid represents a birds-eye view of the level
* walls are made of INT<10>, you cannot move through a wall
* walkable floor area is INT<8>
* you can refill energy by touching energy pills (a 2x2 of INT<6>)
* current key is shown in bottom-left of entire grid
* the exit door is a 4x4 square with INT<11> border
* to find a new key shape, touch the key rotator, a 4x4 square denoted by INT<9> and INT<4> in the top-left corner of the square
* to find a new key color, touch the color rotator, a 4x4 square denoted by INT<9> and INT<2> and in the bottom-left corner of the square
* to rotate more than once, move 1 space away from the rotator and back on
* continue rotating the shape and color of the key until the key matches the one inside the exit door (scaled down 2X)
* if the grid does not change after an action, you probably tried to move into a wall

An example of a good strategy observation:
The player 4x4 made of INT<4> and INT<0> is standing below a wall of INT<10>, so I cannot move up anymore and should
move left towards the rotator with INT<11>.

# TURN:
Call exactly one action.
        """.format()
        )


# Example of a custom LLM agent
class MyCustomLLM(LLM):
    """Template for creating your own custom LLM agent."""

    MAX_ACTIONS = 80
    MODEL = "gpt-4o-mini"
    DO_OBSERVATION = True

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Customize this method to provide instructions to the LLM."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# CUSTOM INSTRUCTIONS:
Add your game instructions and strategy here.
For example, explain the game rules, objectives, and optimal strategies.

# TURN:
Call exactly one action.
        """.format()
        )
