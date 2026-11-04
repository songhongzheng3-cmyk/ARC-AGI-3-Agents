"""Closed-loop policy for resource-and-pattern grid mechanisms."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from heapq import heappop, heappush
from math import gcd
from typing import Any, Iterable

from arcengine import GameAction

from .contact_effects import ContactEffectMemory
from .dynamic_contact import DynamicContactModel
from .entity_options import EntityOptionController
from .joint_navigation import JointNavigationModel, NavigationNode, NavigationStep
from .mechanism_state_tracker import (
    ACTION_DELTAS,
    Cell,
    MechanismObject,
    MechanismState,
)
from .motion_context import MotionContextMemory
from .option_compiler import OptionExecutionController
from .task_planning import (
    JointMechanismGraph,
    JointMechanismState,
    JointMechanismTransition,
    PlanningState,
    PlanStage,
    PlanStageKind,
    TaskPlanController,
)


@dataclass(frozen=True)
class MechanismDecision:
    action: str
    objective: str
    target: Cell | None = None
    route_length: int | None = None
    dynamic_signature: tuple[int, int, str] | None = None


@dataclass(frozen=True)
class PendingInteraction:
    """One executed step that may produce evidence about its target."""

    objective: str
    target: Cell
    source: Cell
    destination: Cell
    action: str
    dynamic_signature: tuple[int, int, str] | None = None
    expected_status: JointMechanismState | None = None


class MechanismPolicy:
    """Learn object roles from effects and plan one action at a time.

    The policy contains no level coordinates or successful action sequence.
    It uses structural priors only to choose the order of experiments; roles
    become authoritative only after an observed resource or status event.
    """

    DIRECTION_NAMES = tuple(ACTION_DELTAS)

    def __init__(
        self,
        enabled: bool = True,
        *,
        enable_joint_navigation: bool = True,
        enable_motion_context: bool = False,
        enable_dynamic_contact: bool = False,
        enable_option_compiler: bool = False,
        enable_contact_effects: bool = False,
        enable_entity_options: bool = False,
    ) -> None:
        self.enabled = enabled
        self.enable_joint_navigation = enable_joint_navigation
        self.enable_motion_context = enable_motion_context
        self.enable_dynamic_contact = enable_dynamic_contact
        self.enable_option_compiler = enable_option_compiler
        self.enable_contact_effects = enable_contact_effects
        self.enable_entity_options = enable_entity_options
        self.entity_options = EntityOptionController()
        self.entity_option_archives: list[dict] = []
        self.contact_effects = ContactEffectMemory()
        self.contact_effect_archives: list[dict] = []
        self.option_controller = OptionExecutionController()
        self.option_archives: list[dict] = []
        self.dynamic_contact = DynamicContactModel()
        self.motion_context = MotionContextMemory()
        self.joint_navigation = JointNavigationModel()
        self._previous_navigation_state: MechanismState | None = None
        self._expected_navigation_step: NavigationStep | None = None
        self.joint_prediction_checks = 0
        self.joint_prediction_errors = 0
        self.level: int | None = None
        self.known_walkable: set[Cell] = set()
        self.known_blocked: set[Cell] = set()
        self.observed_transitions: dict[tuple[Cell, str], Cell] = {}
        self.special_transitions: dict[tuple[Cell, str], Cell] = {}
        self.contextual_special_transitions: dict[
            tuple[Cell, str, str, int | None], Cell
        ] = {}
        self.transition_effects: dict[
            tuple[Cell, str, str, int | None], frozenset[str]
        ] = {}
        self.unsafe_navigation_edges: set[tuple[Cell, str, str, int | None]] = set()
        self.topology_revision = 0
        self.joint_mechanism_graph = JointMechanismGraph()
        self.task_plan_controller = TaskPlanController()
        self.refill_cells: set[Cell] = set()
        self.changer_cells: set[Cell] = set()
        self.palette_changer_cells: set[Cell] = set()
        self.contacted_cells: set[Cell] = set()
        self.lifecycle_evidence_cells: set[Cell] = set()
        self.no_effect_cells: set[Cell] = set()
        self.exhausted_probe_cells: set[Cell] = set()
        self.dynamic_track_positions: dict[tuple[int, int, str], Cell] = {}
        self.dynamic_track_history: dict[tuple[int, int, str], deque[Cell]] = (
            defaultdict(lambda: deque(maxlen=16))
        )
        self.autonomous_dynamic_signatures: set[tuple[int, int, str]] = set()
        self.command_dynamic_signatures: set[tuple[int, int, str]] = set()
        self.dynamic_command_evidence: dict[tuple[int, int, str], int] = defaultdict(
            int
        )
        self.dynamic_action_transitions: dict[
            tuple[tuple[int, int, str], Cell, str], Cell
        ] = {}
        self.dynamic_pattern_signatures: set[tuple[int, int, str]] = set()
        self.dynamic_palette_signatures: set[tuple[int, int, str]] = set()
        self.dynamic_contact_retries: dict[
            tuple[tuple[int, int, str], Cell, str], int
        ] = defaultdict(int)
        self.dynamic_signature_attempts: dict[tuple[int, int, str], int] = defaultdict(
            int
        )
        self.command_joint_experiments: dict[tuple[tuple[Cell, ...], str], int] = (
            defaultdict(int)
        )
        self.command_experiment_action: str | None = None
        self.command_experiment_streak = 0
        self.last_command_motion_action: str | None = None
        self.command_visibility_attempts: dict[str, int] = defaultdict(int)
        self.dynamic_phase_waits: dict[Cell, int] = defaultdict(int)
        self.identity_probes: set[str] = set()
        self.resource_cost = 4
        self.last_resource: int | None = None
        self.last_status = "empty"
        self.last_status_color: int | None = None
        self.last_decision: MechanismDecision | None = None
        self.pending_interaction: PendingInteraction | None = None
        self.recent_interactions: deque[PendingInteraction] = deque(maxlen=4)
        self.static_target_cells: set[Cell] = set()
        self.dynamic_target_cells: set[Cell] = set()
        self.target_states: dict[Cell, tuple[str, int | None]] = {}
        self.target_object_memory: dict[Cell, MechanismObject] = {}
        self.committed_target: Cell | None = None
        self.committed_pattern = "empty"
        self.committed_color: int | None = None
        self.completion_contact_retries: dict[tuple[Cell, str], int] = defaultdict(int)
        self.contact_attempts: dict[Cell, set[str]] = defaultdict(set)
        self.contact_probe_retries: dict[tuple[Cell, str], int] = defaultdict(int)
        self.actuator_probe_states: dict[tuple[Cell, str], list[str | int | None]] = (
            defaultdict(list)
        )
        self.completed_actuator_probes: set[tuple[Cell, str]] = set()
        self.active_actuator_probe: tuple[Cell, str] | None = None
        self._probe_return: tuple[Cell, str] | None = None
        self._dynamic_probe_departures: dict[Cell, str] = {}
        self._dynamic_plan: deque[str] = deque()
        self._dynamic_plan_signature: tuple[int, int, str] | None = None
        self._dynamic_plan_target: Cell | None = None
        self._dynamic_plan_dimension: str | None = None
        self._dynamic_plan_joint_signatures: tuple[tuple[int, int, str], ...] = ()
        self._dynamic_plan_expected_states: deque[tuple[Cell, tuple[Cell, ...]]] = (
            deque()
        )
        self._depart_and_return: Cell | None = None
        self._last_player: Cell | None = None
        self._previous_candidate_cells: set[Cell] = set()
        self._task_plan_route_failures = 0
        self._task_plan_recent_steps: deque[tuple[Cell, str, JointMechanismState]] = (
            deque(maxlen=8)
        )
        self._task_plan_recent_signature: tuple[int, int] | None = None

    def reset(self) -> None:
        enabled = self.enabled
        self.__init__(
            enabled=enabled,
            enable_joint_navigation=self.enable_joint_navigation,
            enable_motion_context=self.enable_motion_context,
            enable_dynamic_contact=self.enable_dynamic_contact,
            enable_option_compiler=self.enable_option_compiler,
            enable_contact_effects=self.enable_contact_effects,
            enable_entity_options=self.enable_entity_options,
        )

    def note_reset(self) -> None:
        """A reset is an episode boundary, never a learned movement edge."""
        self._previous_navigation_state = None
        self.motion_context.reset()
        self.dynamic_contact.reset_episode()
        self.contact_effects.reset_episode()
        self.entity_options.reset_episode()
        self.option_controller.reset_episode()
        self._expected_navigation_step = None
        self.pending_interaction = None
        self.recent_interactions.clear()
        self._previous_candidate_cells.clear()
        self._last_player = None
        self.last_resource = None
        self.dynamic_track_positions.clear()
        self.dynamic_track_history.clear()
        self.active_actuator_probe = None
        self._probe_return = None
        self._depart_and_return = None
        self._dynamic_plan.clear()
        self._dynamic_plan_expected_states.clear()
        self.task_plan_controller.invalidate("episode_reset")
        self._task_plan_recent_steps.clear()
        self._task_plan_recent_signature = None

    def _reset_level(self, level: int) -> None:
        if self.enable_entity_options and self.level is not None:
            if level > self.level:
                self.entity_options.metrics["actual_level_advanced"] += 1
                pending = self.entity_options.expected
                if pending is not None and pending.terminal:
                    self.entity_options.metrics["option_goal_completed"] += 1
            self.entity_option_archives.append(
                {"level": self.level, **self.entity_options.diagnostics()}
            )
        if self.enable_contact_effects and self.level is not None:
            self.contact_effect_archives.append(
                {"level": self.level, **self.contact_effects.diagnostics()}
            )
        if self.enable_option_compiler and self.level is not None:
            self.option_archives.append(
                {"level": self.level, **self.option_controller.diagnostics()}
            )
        self.level = level
        self.entity_options = EntityOptionController()
        self.contact_effects = ContactEffectMemory()
        self.joint_navigation = JointNavigationModel()
        self.motion_context.reset()
        self.dynamic_contact = DynamicContactModel()
        self.option_controller = OptionExecutionController()
        self._previous_navigation_state = None
        self._expected_navigation_step = None
        self.joint_prediction_checks = 0
        self.joint_prediction_errors = 0
        self.known_walkable.clear()
        self.known_blocked.clear()
        self.observed_transitions.clear()
        self.special_transitions.clear()
        self.contextual_special_transitions.clear()
        self.transition_effects.clear()
        self.unsafe_navigation_edges.clear()
        self.topology_revision = 0
        self.joint_mechanism_graph.clear()
        self.task_plan_controller.reset()
        self.refill_cells.clear()
        self.changer_cells.clear()
        self.palette_changer_cells.clear()
        self.contacted_cells.clear()
        self.lifecycle_evidence_cells.clear()
        self.no_effect_cells.clear()
        self.exhausted_probe_cells.clear()
        self.dynamic_track_positions.clear()
        self.dynamic_track_history.clear()
        self.autonomous_dynamic_signatures.clear()
        self.command_dynamic_signatures.clear()
        self.dynamic_command_evidence.clear()
        self.dynamic_action_transitions.clear()
        self.dynamic_pattern_signatures.clear()
        self.dynamic_palette_signatures.clear()
        self.dynamic_contact_retries.clear()
        self.dynamic_signature_attempts.clear()
        self.command_joint_experiments.clear()
        self.command_experiment_action = None
        self.command_experiment_streak = 0
        self.last_command_motion_action = None
        self.command_visibility_attempts.clear()
        self.dynamic_phase_waits.clear()
        self.identity_probes.clear()
        self.resource_cost = 4
        self.last_resource = None
        self.last_status = "empty"
        self.last_status_color = None
        self.last_decision = None
        self.pending_interaction = None
        self.recent_interactions.clear()
        self.static_target_cells.clear()
        self.dynamic_target_cells.clear()
        self.target_states.clear()
        self.target_object_memory.clear()
        self.committed_target = None
        self.committed_pattern = "empty"
        self.committed_color = None
        self.completion_contact_retries.clear()
        self.contact_attempts.clear()
        self.contact_probe_retries.clear()
        self.actuator_probe_states.clear()
        self.completed_actuator_probes.clear()
        self.active_actuator_probe = None
        self._probe_return = None
        self._dynamic_probe_departures.clear()
        self._dynamic_plan.clear()
        self._dynamic_plan_signature = None
        self._dynamic_plan_target = None
        self._dynamic_plan_dimension = None
        self._dynamic_plan_joint_signatures = ()
        self._dynamic_plan_expected_states.clear()
        self._depart_and_return = None
        self._last_player = None
        self._previous_candidate_cells.clear()
        self._task_plan_route_failures = 0
        self._task_plan_recent_steps.clear()
        self._task_plan_recent_signature = None

    @staticmethod
    def _available_names(values: Iterable[Any]) -> list[str]:
        result: list[str] = []
        for value in values or []:
            if isinstance(value, GameAction):
                name = value.name
            else:
                try:
                    name = GameAction.from_id(int(value)).name
                except (KeyError, TypeError, ValueError):
                    name = str(value).upper()
            if name in ACTION_DELTAS and name not in result:
                result.append(name)
        return result

    @staticmethod
    def _inside(item: MechanismObject, state: MechanismState) -> bool:
        if state.layout is None:
            return False
        x1, y1, x2, y2 = state.layout.bounds
        # Sprites centred on a boundary cell may render one pixel beyond the
        # inferred floor rectangle.  Admit that narrow overhang while keeping
        # larger HUD panels outside the logical scene.
        tolerance = 1
        bx1, by1, bx2, by2 = item.bbox
        return (
            x1 - tolerance <= bx1
            and y1 - tolerance <= by1
            and bx2 <= x2 + tolerance
            and by2 <= y2 + tolerance
        )

    def _object_groups(
        self, state: MechanismState
    ) -> dict[Cell, tuple[MechanismObject, ...]]:
        grouped: dict[Cell, list[MechanismObject]] = defaultdict(list)
        for item in state.objects:
            if (
                self._inside(item, state)
                and item.object_id not in state.player_component_ids
            ):
                grouped[item.cell].append(item)
        return {
            cell: tuple(sorted(items, key=lambda item: (item.area, item.color)))
            for cell, items in grouped.items()
        }

    def _static_targets(self, state: MechanismState) -> dict[Cell, MechanismObject]:
        def covered(bbox: tuple[int, int, int, int]) -> bool:
            player = state.player_bbox
            return player is not None and not (
                bbox[2] < player[0]
                or player[2] < bbox[0]
                or bbox[3] < player[1]
                or player[3] < bbox[1]
            )

        # Pixel overlap invalidates the observed target pattern: the player's
        # sprite can become an extra connected part inside the goal's crop.
        # Preserve the last unobstructed observation only while it is covered;
        # once uncovered, real target changes remain observable.
        result = {}
        for item in state.objects:
            if (
                item.static_target
                and self._inside(item, state)
                and not covered(item.bbox)
            ):
                result[item.cell] = item
                self.target_object_memory[item.cell] = item
        for cell, item in self.target_object_memory.items():
            if covered(item.bbox):
                result[cell] = item
        return result

    def _candidate_cells(self, state: MechanismState) -> list[Cell]:
        groups = self._object_groups(state)
        targets = set(self._static_targets(state))
        candidates: list[tuple[tuple[int, int, int, int, Cell], Cell]] = []
        for cell, items in groups.items():
            if cell == state.player_position or cell in targets:
                continue
            total_area = sum(item.area for item in items)
            colors = {item.color for item in items}
            if total_area > 20 or not (1 <= total_area <= 16):
                continue
            min_x = min(item.bbox[0] for item in items)
            min_y = min(item.bbox[1] for item in items)
            max_x = max(item.bbox[2] for item in items)
            max_y = max(item.bbox[3] for item in items)
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            if state.layout is not None:
                cell_width, cell_height = state.layout.cell_size
                if width > cell_width or height > cell_height:
                    continue
            # Multi-part/multi-palette compact objects are the most informative
            # experiments because they often manipulate another visible token.
            likely_changer = len(items) > 1 or len(colors) > 1
            if state.layout is not None:
                likely_changer = likely_changer or (
                    total_area <= 6
                    and cell[0] in {0, state.layout.cols - 1}
                    and cell[1] in {0, state.layout.rows - 1}
                )
            likely_refill = total_area >= 7 and len(items) == 1
            candidates.append(
                (
                    (
                        0 if likely_changer else 1 if likely_refill else 2,
                        0 if cell not in self.contacted_cells else 1,
                        max(width, height),
                        total_area,
                        cell,
                    ),
                    cell,
                )
            )
        return [cell for _, cell in sorted(candidates)]

    def _role_discovery_cells(
        self,
        state: MechanismState,
        dimension: str,
    ) -> list[Cell]:
        """Rank unexplained objects by evidence for the missing role."""
        groups = self._object_groups(state)
        candidates = [
            cell
            for cell in self._candidate_cells(state)
            if cell not in self.no_effect_cells
            and cell not in self.exhausted_probe_cells
            and cell not in self.refill_cells
            and cell not in self.changer_cells
            and cell not in self.palette_changer_cells
        ]

        def role_score(cell: Cell) -> tuple[int, int, int, int, Cell]:
            items = groups[cell]
            colors = len({item.color for item in items})
            total_area = sum(item.area for item in items)
            largest_component = max(item.area for item in items)
            if dimension == "pattern":
                return (
                    0 if len(items) > 1 else 1,
                    -len(items),
                    0 if cell not in self.contacted_cells else 1,
                    total_area,
                    cell,
                )
            if dimension == "both":
                # A compound controller can advertise its state either through
                # several palette tokens or through a substantial structural
                # backing plus one state token.  Ranking only by color count
                # mistakes small multicolor markers for the primary mechanism
                # in later levels.  Treat both forms as equally informative.
                control_richness = max(
                    colors,
                    largest_component if len(items) > 1 else 1,
                )
                return (
                    -control_richness,
                    -colors,
                    0 if cell not in self.contacted_cells else 1,
                    total_area,
                    cell,
                )
            return (
                -colors,
                -len(items),
                0 if cell not in self.contacted_cells else 1,
                total_area,
                cell,
            )

        return sorted(candidates, key=role_score)

    def _likely_refill_cells(self, state: MechanismState) -> list[Cell]:
        groups = self._object_groups(state)
        targets = set(self._static_targets(state))
        candidates: list[Cell] = []
        for cell, items in groups.items():
            if (
                cell == state.player_position
                or cell in targets
                or cell in self.contacted_cells
                or cell in self.no_effect_cells
                or cell in self.exhausted_probe_cells
            ):
                continue
            total_area = sum(item.area for item in items)
            if len(items) == 1 and 7 <= total_area <= 16:
                candidates.append(cell)
        return candidates

    def _visible_refill_cells(self, state: MechanismState) -> set[Cell]:
        visible_candidates = set(self._likely_refill_cells(state))
        return self.refill_cells.intersection(visible_candidates)

    def _update_evidence(
        self, state: MechanismState, *, actual_action: str | None = None
    ) -> None:
        # An action after an observed empty meter can restart the scene even
        # while the SDK continues to report NOT_FINISHED. This is not a refill
        # or an actuator/motion outcome of the pending action.
        if (
            self._previous_navigation_state is not None
            and self._previous_navigation_state.level == state.level
            and self._previous_navigation_state.resource_capacity > 0
            and self._previous_navigation_state.resource_level == 0
            and state.resource_level > 0
        ):
            self.note_reset()
        dynamic_effect = False
        if self.enable_dynamic_contact:
            self.dynamic_contact.observe(
                state,
                action=actual_action
                or (
                    None
                    if self.pending_interaction is None
                    else self.pending_interaction.action
                ),
            )
            dynamic_effect = len(self.dynamic_contact.last_contacts) == 1 and any(
                event.event_type in {"status_pattern_changed", "status_palette_changed"}
                for event in state.events
            )
            if self.enable_contact_effects:
                self.contact_effects.observe(
                    state,
                    self.dynamic_contact,
                    action=actual_action
                    or (
                        None
                        if self.pending_interaction is None
                        else self.pending_interaction.action
                    ),
                )
        self.known_walkable.update(state.walkable_cells)
        self.known_blocked.update(state.blocked_cells)
        self.known_blocked.difference_update(self.known_walkable)
        player = state.player_position
        player_delta = None
        if player is not None and self._last_player is not None:
            player_delta = (
                player[0] - self._last_player[0],
                player[1] - self._last_player[1],
            )
        current_dynamic_positions: dict[tuple[int, int, str], Cell] = {}
        previous_action = (
            None
            if self.pending_interaction is None
            else self.pending_interaction.action
        )
        for item in state.objects:
            if (
                not self._inside(item, state)
                or item.static_target
                or item.area < 8
                or item.cell == player
                or item.object_id in state.player_component_ids
            ):
                continue
            signature = (item.color, item.area, item.shape_signature)
            current_dynamic_positions[signature] = item.cell
            history = self.dynamic_track_history[signature]
            history.append(item.cell)
            previous_cell = self.dynamic_track_positions.get(signature)
            if previous_cell is None:
                continue
            if previous_action is not None:
                self.dynamic_action_transitions[
                    (signature, previous_cell, previous_action)
                ] = item.cell
            if previous_cell == item.cell:
                continue
            object_delta = (
                item.cell[0] - previous_cell[0],
                item.cell[1] - previous_cell[1],
            )
            commanded_delta = (
                None if previous_action is None else ACTION_DELTAS[previous_action]
            )
            if commanded_delta is not None and object_delta == commanded_delta:
                self.last_command_motion_action = previous_action
                self.dynamic_command_evidence[signature] += 1
                if self.dynamic_command_evidence[signature] >= 2:
                    self.command_dynamic_signatures.add(signature)
                    self.autonomous_dynamic_signatures.discard(signature)
            elif (
                signature not in self.command_dynamic_signatures
                and player_delta is not None
                and object_delta != player_delta
                and object_delta != (0, 0)
            ):
                self.autonomous_dynamic_signatures.add(signature)
        self.dynamic_track_positions = current_dynamic_positions
        visible_static_targets = set(self._static_targets(state))
        self.static_target_cells.update(visible_static_targets)
        event_types = {event.event_type for event in state.events}
        current_candidate_cells = set(self._candidate_cells(state))
        resource_wrapped = bool(
            "resource_increased" in event_types
            and self.last_resource is not None
            and self.last_resource <= max(1, self.resource_cost)
            and state.resource_level >= state.resource_capacity
        )

        # Compact actuators and refill tokens are often completely covered by
        # the 5x5 composite player on contact.  Their current observation then
        # disappears in exactly the frame that reports the effect.  Preserve
        # the prior visible-candidate set for one transition so the effect can
        # still be attributed to the contacted relation without depending on
        # palette values or fixed coordinates.
        covered_candidate = (
            player
            if player is not None and player in self._previous_candidate_cells
            else None
        )
        if covered_candidate is not None:
            status_changed = bool(
                event_types.intersection(
                    {"status_pattern_changed", "status_palette_changed"}
                )
            )
            if (
                "resource_increased" in event_types
                and not status_changed
                and not resource_wrapped
            ):
                self.refill_cells.add(covered_candidate)
                self.contacted_cells.add(covered_candidate)
            if "status_pattern_changed" in event_types and not dynamic_effect:
                self.changer_cells.add(covered_candidate)
                self.contacted_cells.add(covered_candidate)
            if "status_palette_changed" in event_types and not dynamic_effect:
                self.palette_changer_cells.add(covered_candidate)
                self.contacted_cells.add(covered_candidate)

        visible_targets = self._static_targets(state)
        status_changed = bool(
            event_types.intersection(
                {"status_pattern_changed", "status_palette_changed"}
            )
        )
        for cell, item in visible_targets.items():
            target_state = (item.pattern_signature, item.pattern_color)
            previous_target_state = self.target_states.get(cell)
            if (
                status_changed
                and previous_target_state is not None
                and previous_target_state != target_state
            ):
                self.dynamic_target_cells.add(cell)
            self.target_states[cell] = target_state

        if (
            self.last_resource is not None
            and 0 < state.resource_level < self.last_resource
        ):
            delta = self.last_resource - state.resource_level
            self.resource_cost = gcd(self.resource_cost, delta) or delta

        pending = self.pending_interaction
        previous_navigation = self._previous_navigation_state
        previous_context = self.motion_context.context
        current_context = (
            self.motion_context.observe(state) if self.enable_motion_context else ()
        )
        if (
            previous_navigation is not None
            and previous_navigation.level == state.level
            and previous_navigation.player_position is not None
            and player is not None
            and min(previous_navigation.player_confidence, state.player_confidence)
            >= 0.8
            and pending is not None
            and pending.source == previous_navigation.player_position
            and not resource_wrapped
        ):
            before_node = NavigationNode(
                previous_navigation.player_position,
                self._joint_state(previous_navigation),
                previous_context,
            )
            after_node = NavigationNode(
                player, self._joint_state(state), current_context
            )
            self.joint_navigation.record(
                before_node,
                pending.action,
                after_node,
                resource_before=previous_navigation.resource_level,
                resource_after=state.resource_level,
            )
            prediction = self._expected_navigation_step
            if prediction is not None and prediction.source == before_node:
                self.joint_prediction_checks += 1
                self.joint_prediction_errors += prediction.destination != after_node
        self._previous_navigation_state = state
        self._expected_navigation_step = None
        if pending is not None:
            context_key = (
                pending.source,
                pending.action,
                self.last_status,
                self.last_status_color,
            )
            self.transition_effects[context_key] = frozenset(event_types)
            # A relocation is a learned transport edge, not an unsafe edge by
            # itself.  The level-2 layout uses this edge to return from a
            # teleport pocket; quarantining it makes every later route
            # unreachable.  Only effects that change the mechanism state are
            # unsafe for an ordinary coordinate-only route.
            if event_types.intersection(
                {
                    "resource_increased",
                    "status_pattern_changed",
                    "status_palette_changed",
                    "level_advanced",
                }
            ):
                # These transitions are useful mechanism evidence, but they
                # are not ordinary navigation edges.  Reusing one inside a
                # coordinate-only route can silently violate the Option's
                # status/resource preconditions.
                self.unsafe_navigation_edges.add(context_key)

            plan = self.task_plan_controller.active_plan
            if pending.objective.startswith("task_plan_") and plan is not None:
                stage_signature = (plan.plan_id, plan.current_stage_index)
                if self._task_plan_recent_signature != stage_signature:
                    self._task_plan_recent_steps.clear()
                    self._task_plan_recent_signature = stage_signature
                step_signature = (
                    pending.source,
                    pending.action,
                    JointMechanismState(self.last_status, self.last_status_color),
                )
                if step_signature in self._task_plan_recent_steps:
                    # A closed-loop Option must make progress.  Repeating the
                    # same state/action edge within one stage is direct
                    # evidence that the coordinate route is cycling through
                    # contextual topology.  Quarantine only that edge/context
                    # and let the next decision search for an alternative.
                    self.unsafe_navigation_edges.add(context_key)
                    self.observed_transitions.pop(
                        (pending.source, pending.action), None
                    )
                    self.contextual_special_transitions.pop(context_key, None)
                    self.topology_revision += 1
                self._task_plan_recent_steps.append(step_signature)
        if (
            pending is not None
            and player is not None
            and self._last_player == pending.source
        ):
            expected_delta = ACTION_DELTAS[pending.action]
            observed_delta = (
                player[0] - pending.source[0],
                player[1] - pending.source[1],
            )
            transition_key = (pending.source, pending.action)
            if observed_delta in {expected_delta, (0, 0)}:
                # Only deterministic local motion belongs in the navigation
                # graph.  Mechanisms can relocate the player several cells;
                # treating that contextual effect as a reusable edge makes
                # ordinary routes depend on a transition that may not fire in
                # the next status/phase.
                self.observed_transitions[transition_key] = player
                removed = self.special_transitions.pop(transition_key, None)
                if removed is not None:
                    self.topology_revision += 1
            else:
                contextual_key = (
                    pending.source,
                    pending.action,
                    self.last_status,
                    self.last_status_color,
                )
                previous_special = self.special_transitions.get(transition_key)
                previous_contextual = self.contextual_special_transitions.get(
                    contextual_key
                )
                self.special_transitions[transition_key] = player
                self.contextual_special_transitions[contextual_key] = player
                if previous_special != player or previous_contextual != player:
                    self.topology_revision += 1
                self.observed_transitions.pop(transition_key, None)
            self.known_walkable.add(player)
        if pending is not None:
            self.recent_interactions.append(pending)
        interactions = list(reversed(self.recent_interactions))
        if pending is not None and pending.target in {pending.destination, player}:
            expected_event = {
                "task_plan_refill": "resource_increased",
                "task_plan_apply_actuator": "status_changed",
                "task_plan_contact_target": "level_advanced",
            }.get(pending.objective)
            observed_expected = (
                expected_event in event_types
                if expected_event != "status_changed"
                else bool(
                    event_types.intersection(
                        {"status_pattern_changed", "status_palette_changed"}
                    )
                )
            )
            if expected_event is not None and not observed_expected:
                self.task_plan_controller.record_failed_attempt()
        if pending is not None:
            reached = pending.target in {pending.destination, player}
            exploratory = pending.objective in {
                "visit_uncontacted_candidate",
                "discover_resource_source",
                "restore_resource",
                "align_status_pattern",
                "align_status_palette",
                "discover_pattern_changer",
                "discover_palette_changer",
                "discover_both_changer",
                "probe_candidate_contact",
            }
            can_assign_role = (
                reached
                and exploratory
                and pending.target not in self.static_target_cells
                and pending.objective != "contact_matching_static_object"
                and not dynamic_effect
            )
            if reached:
                self.contacted_cells.add(pending.target)
            status_changed = bool(
                event_types.intersection(
                    {"status_pattern_changed", "status_palette_changed"}
                )
            )
            if (
                can_assign_role
                and "resource_increased" in event_types
                and not status_changed
                and not resource_wrapped
            ):
                self.refill_cells.add(pending.target)
            if can_assign_role and "status_pattern_changed" in event_types:
                self.changer_cells.add(pending.target)
            if can_assign_role and "status_palette_changed" in event_types:
                self.palette_changer_cells.add(pending.target)
            if pending.dynamic_signature is not None:
                if "status_pattern_changed" in event_types:
                    self.dynamic_pattern_signatures.add(pending.dynamic_signature)
                if "status_palette_changed" in event_types:
                    self.dynamic_palette_signatures.add(pending.dynamic_signature)
                if not event_types.intersection(
                    {
                        "status_pattern_changed",
                        "status_palette_changed",
                        "level_advanced",
                    }
                ):
                    dynamic_key = (
                        pending.dynamic_signature,
                        pending.source,
                        pending.action,
                    )
                    self.dynamic_contact_retries[dynamic_key] += 1
                    self.dynamic_signature_attempts[pending.dynamic_signature] += 1
            lifecycle_at_target = any(
                event.event_type in {"object_appeared", "object_disappeared"}
                and pending.target in event.cells
                and event.before is not None
                and event.after is not None
                for event in state.events
            )
            if lifecycle_at_target:
                self.lifecycle_evidence_cells.add(pending.target)
            if (
                reached
                and pending.objective == "probe_candidate_contact"
                and not event_types.intersection(
                    {
                        "resource_increased",
                        "status_pattern_changed",
                        "status_palette_changed",
                        "level_advanced",
                    }
                )
            ):
                retry_key = (pending.target, pending.action)
                self.contact_probe_retries[retry_key] += 1
                # A short repeat window distinguishes delayed effects from a
                # blocked edge.  Dynamic scenes must revisit the relation from
                # another side or phase after this window instead of spending
                # most of the resource budget on one unchanged contact.
                probe_limit = 4
                if self.contact_probe_retries[retry_key] >= probe_limit:
                    self.contact_attempts[pending.target].add(pending.action)
                    blocked_contact = (
                        player == pending.source
                        and pending.destination != pending.target
                    )
                    if blocked_contact:
                        self.exhausted_probe_cells.add(pending.target)
                if len(self.contact_attempts[pending.target]) >= len(
                    self.DIRECTION_NAMES
                ):
                    self.exhausted_probe_cells.add(pending.target)
                    if pending.target not in self.lifecycle_evidence_cells:
                        self.no_effect_cells.add(pending.target)

        if not dynamic_effect and event_types.intersection(
            {
                "resource_increased",
                "status_pattern_changed",
                "status_palette_changed",
            }
        ):
            for interaction in interactions:
                exploratory = interaction.objective in {
                    "visit_uncontacted_candidate",
                    "discover_resource_source",
                    "restore_resource",
                    "align_status_pattern",
                    "align_status_palette",
                    "discover_pattern_changer",
                    "discover_palette_changer",
                    "discover_both_changer",
                    "probe_candidate_contact",
                }
                contacted = interaction.target in {
                    interaction.source,
                    interaction.destination,
                }
                if (
                    not exploratory
                    or not contacted
                    or interaction.target in self.static_target_cells
                ):
                    continue
                status_changed = bool(
                    event_types.intersection(
                        {"status_pattern_changed", "status_palette_changed"}
                    )
                )
                if (
                    "resource_increased" in event_types
                    and not status_changed
                    and not resource_wrapped
                ):
                    self.refill_cells.add(interaction.target)
                if "status_pattern_changed" in event_types:
                    self.changer_cells.add(interaction.target)
                if "status_palette_changed" in event_types:
                    self.palette_changer_cells.add(interaction.target)
                self.contacted_cells.add(interaction.target)
                break

        # Treat a newly discovered status actuator as a small observable state
        # machine.  Re-enter it until an output repeats, which establishes a
        # full cycle without assuming how many states or colours it contains.
        # A matching completion target can interrupt this experiment later in
        # choose(), so pattern actuators need not be cycled past a solved state.
        if pending is not None and not dynamic_effect:
            reached = pending.target in {pending.destination, player}
            probe_cell = pending.target
            dimensions: list[tuple[str, str | int | None]] = []
            if "status_pattern_changed" in event_types:
                dimensions.append(("pattern", state.status_signature))
            if "status_palette_changed" in event_types:
                dimensions.append(("palette", state.status_color))
            for dimension, value in dimensions:
                key = (probe_cell, dimension)
                learned_role = (
                    dimension == "pattern" and probe_cell in self.changer_cells
                ) or (
                    dimension == "palette" and probe_cell in self.palette_changer_cells
                )
                if (
                    not reached
                    or not learned_role
                    or probe_cell in self.static_target_cells
                    or key in self.completed_actuator_probes
                ):
                    continue
                observed = self.actuator_probe_states[key]
                if value in observed:
                    self.completed_actuator_probes.add(key)
                    if self.active_actuator_probe == key:
                        self.active_actuator_probe = None
                else:
                    observed.append(value)
                    self.active_actuator_probe = key

        # A cell that has ever been identified as a static pattern target must
        # not also become a learned actuator merely because contact animates it.
        self.refill_cells.difference_update(self.static_target_cells)
        self.changer_cells.difference_update(self.static_target_cells)
        self.palette_changer_cells.difference_update(self.static_target_cells)

        # Learn a joint mechanism edge only from a real contact whose observed
        # effect identified the contacted cell as an actuator.  This preserves
        # the causal before/contact/after boundary required by task planning;
        # ordinary movement and visual animation never become graph edges.
        if pending is not None and status_changed and not dynamic_effect:
            reached = pending.target in {pending.destination, player}
            learned_actuator = (
                pending.target in self.changer_cells
                or pending.target in self.palette_changer_cells
            )
            if reached and learned_actuator:
                self.joint_mechanism_graph.record(
                    JointMechanismState(self.last_status, self.last_status_color),
                    pending.target,
                    JointMechanismState(
                        state.status_signature,
                        state.status_color,
                    ),
                    approach=pending.source,
                    contact_action=pending.action,
                    observed_result=player,
                )

        if self._depart_and_return is not None and player == self._depart_and_return:
            self._depart_and_return = None
        self.last_resource = state.resource_level
        self.last_status = state.status_signature
        self.last_status_color = state.status_color
        self._last_player = player
        self._previous_candidate_cells = current_candidate_cells

    def _set_decision(
        self,
        state: MechanismState,
        decision: MechanismDecision,
    ) -> GameAction:
        self.last_decision = decision
        source = state.player_position
        if source is None or decision.target is None:
            self.pending_interaction = None
        else:
            dx, dy = ACTION_DELTAS[decision.action]
            self.pending_interaction = PendingInteraction(
                objective=decision.objective,
                target=decision.target,
                source=source,
                destination=(source[0] + dx, source[1] + dy),
                action=decision.action,
                dynamic_signature=decision.dynamic_signature,
                expected_status=(
                    self.task_plan_controller.active_plan.current_stage.expected_after
                    if decision.objective == "task_plan_apply_actuator"
                    and self.task_plan_controller.active_plan is not None
                    and self.task_plan_controller.active_plan.current_stage is not None
                    else None
                ),
            )
        return GameAction.from_name(decision.action)

    def _resume_probe(
        self,
        state: MechanismState,
        names: list[str],
    ) -> MechanismDecision | None:
        if self._probe_return is None:
            return None
        target, action = self._probe_return
        if action not in names:
            self._probe_return = None
            return None
        self._probe_return = None
        return MechanismDecision(
            action=action,
            objective="probe_candidate_contact",
            target=target,
            route_length=1,
        )

    def _matching_commitment(
        self,
        state: MechanismState,
        visible_targets: dict[Cell, MechanismObject],
    ) -> Cell | None:
        matching = [
            cell
            for cell, item in visible_targets.items()
            if cell not in self.dynamic_target_cells
            and self._target_matches_state(item, state)
        ]
        unresolved_ambiguity = (
            len(visible_targets) > 1 and not self.dynamic_target_cells
        )
        if matching and not unresolved_ambiguity:
            route_choice = self._nearest_route(state, matching)
            if route_choice is not None:
                self.committed_target = route_choice[0]
                self.committed_pattern = state.status_signature
                self.committed_color = state.status_color
        elif not self._committed_state_matches(state):
            self.committed_target = None
            self.committed_pattern = "empty"
            self.committed_color = None
        return self.committed_target

    @staticmethod
    def _colors_compatible(left: int | None, right: int | None) -> bool:
        return left is None or right is None or left == right

    @classmethod
    def _target_matches_state(
        cls,
        item: MechanismObject,
        state: MechanismState,
    ) -> bool:
        return bool(state.status_pattern) and (
            item.internal_pattern == state.status_pattern
            and cls._colors_compatible(item.pattern_color, state.status_color)
        )

    def _committed_state_matches(self, state: MechanismState) -> bool:
        return (
            self.committed_pattern == state.status_signature
            and self._colors_compatible(
                self.committed_color,
                state.status_color,
            )
        )

    def _alignment_dimension(
        self,
        state: MechanismState,
        visible_targets: dict[Cell, MechanismObject],
    ) -> str | None:
        targets = tuple(
            item
            for cell, item in visible_targets.items()
            if cell not in self.dynamic_target_cells
        )
        if not targets or not state.status_pattern:
            return None
        if any(item.internal_pattern == state.status_pattern for item in targets):
            return "palette"
        if state.status_color is not None and any(
            item.pattern_color == state.status_color for item in targets
        ):
            return "pattern"
        return "both"

    @staticmethod
    def _inverse_action(action: str) -> str:
        return {
            "ACTION1": "ACTION2",
            "ACTION2": "ACTION1",
            "ACTION3": "ACTION4",
            "ACTION4": "ACTION3",
        }[action]

    def _contact_probe(
        self,
        state: MechanismState,
        target: Cell,
        names: list[str],
    ) -> MechanismDecision | None:
        """Test one untried directional contact around an reached object."""
        player = state.player_position
        if player is None:
            return None

        tried = self.contact_attempts[target]
        ordered = [name for name in names if name not in tried]
        # A blocked/out-of-grid direction is a direct contact experiment and
        # does not require spending another action to return to the object.
        for action in ordered:
            dx, dy = ACTION_DELTAS[action]
            neighbor = (player[0] + dx, player[1] + dy)
            if neighbor not in state.walkable_cells:
                return MechanismDecision(
                    action=action,
                    objective="probe_candidate_contact",
                    target=target,
                    route_length=1,
                )
        # For enterable objects, depart once and re-enter from a new direction.
        # Autonomous mechanisms can be phase-delayed.  Commit to one side for
        # a short observation window, then rotate to another side if it still
        # produces no effect.  Re-ranking the side on every frame changes the
        # experiment before it has accumulated enough evidence.
        dynamic_scene = bool(self.autonomous_dynamic_signatures)
        committed_departure = self._dynamic_probe_departures.get(target)
        departure_actions = list(names)
        if dynamic_scene and committed_departure in departure_actions:
            return_action = self._inverse_action(committed_departure)
            retries = self.contact_probe_retries[(target, return_action)]
            if retries < 4:
                departure_actions.remove(committed_departure)
                departure_actions.insert(0, committed_departure)
            else:
                self._dynamic_probe_departures.pop(target, None)

        for action in departure_actions:
            dx, dy = ACTION_DELTAS[action]
            neighbor = (player[0] + dx, player[1] + dy)
            return_action = self._inverse_action(action)
            phase_exhausted = (
                dynamic_scene
                and self.contact_probe_retries[(target, return_action)] >= 4
            )
            if (
                neighbor in state.walkable_cells
                and return_action not in tried
                and not phase_exhausted
            ):
                if dynamic_scene:
                    self._dynamic_probe_departures[target] = action
                self._probe_return = (target, return_action)
                return MechanismDecision(
                    action=action,
                    objective="probe_candidate_depart",
                    target=target,
                    route_length=1,
                )
        self.exhausted_probe_cells.add(target)
        return None

    def _recent_unit_delta(
        self,
        signature: tuple[int, int, str],
    ) -> Cell | None:
        """Return the immediately observed unit displacement for a track.

        Dynamic mechanisms can disappear, wrap, or be reset by contact.  A
        large jump is therefore not a velocity estimate.  A stationary latest
        frame is equally important: it invalidates an older velocity rather
        than silently carrying that motion forward indefinitely.
        """
        history = self.dynamic_track_history.get(signature)
        if history is None or len(history) < 2:
            return None
        source, destination = history[-2], history[-1]
        delta = (
            destination[0] - source[0],
            destination[1] - source[1],
        )
        return delta if abs(delta[0]) + abs(delta[1]) == 1 else None

    def _clear_dynamic_plan(self) -> None:
        self._dynamic_plan.clear()
        self._dynamic_plan_signature = None
        self._dynamic_plan_target = None
        self._dynamic_plan_dimension = None
        self._dynamic_plan_joint_signatures = ()
        self._dynamic_plan_expected_states.clear()

    def _has_command_pair(self) -> bool:
        """Return whether a substantial command-coupled pair was confirmed.

        Confirmation is historical evidence, not a visibility predicate.  A
        member can disappear for one frame when the player occludes it or the
        parser merges its pixels.  Forgetting the pair in that frame lets
        incompatible static-discovery policies take control.  The individual
        joint planners still require both current positions before acting.
        """
        return (
            sum(signature[1] >= 10 for signature in self.command_dynamic_signatures)
            >= 2
        )

    def _command_model_experiment(
        self,
        state: MechanismState,
        names: list[str],
        dimension: str | None,
    ) -> MechanismDecision | None:
        """Probe one under-observed joint command instead of static wandering."""
        player = state.player_position
        signatures = tuple(
            sorted(
                signature
                for signature in self.command_dynamic_signatures
                if signature[1] >= 10 and signature in self.dynamic_track_positions
            )
        )
        if player is None or len(signatures) < 2:
            return None
        positions = tuple(
            self.dynamic_track_positions[signature] for signature in signatures
        )
        actions = [name for name in names if name in ACTION_DELTAS]
        if not actions:
            return None

        previous_action = (
            self.pending_interaction.action
            if self.pending_interaction is not None
            and self.pending_interaction.objective.startswith("learn_command_dynamic_")
            else self.last_command_motion_action
        )
        can_continue = (
            previous_action is not None
            and self.command_experiment_streak < 4
            and (
                self.command_experiment_action is None
                or previous_action == self.command_experiment_action
            )
        )

        untried_actions = [
            action
            for action in actions
            if self.command_joint_experiments[(positions, action)] == 0
        ]
        if not untried_actions:
            return None

        def score(action: str) -> tuple[int, int, int, int]:
            predicted = tuple(
                self._predict_command_destination(state, signature, position, action)
                for signature, position in zip(signatures, positions)
            )
            current_offsets = tuple(
                (
                    position[0] - positions[0][0],
                    position[1] - positions[0][1],
                )
                for position in positions[1:]
            )
            predicted_offsets = tuple(
                (
                    position[0] - predicted[0][0],
                    position[1] - predicted[0][1],
                )
                for position in predicted[1:]
            )
            invalid_pair = int(
                len(set(predicted)) != len(predicted)
                or predicted_offsets != current_offsets
            )
            exact_evidence = sum(
                (signature, position, action) in self.dynamic_action_transitions
                for signature, position in zip(signatures, positions)
            )
            player_transition_known = int((player, action) in self.observed_transitions)
            return (
                invalid_pair,
                exact_evidence,
                0 if can_continue and action == previous_action else 1,
                player_transition_known * len(actions) + actions.index(action),
            )

        action = min(untried_actions, key=score)
        self.command_joint_experiments[(positions, action)] += 1
        if action == self.command_experiment_action:
            self.command_experiment_streak += 1
        else:
            self.command_experiment_action = action
            self.command_experiment_streak = 1
        return MechanismDecision(
            action=action,
            objective=f"learn_command_dynamic_{dimension or 'unknown'}",
            target=player,
            route_length=1,
        )

    def _command_visibility_recovery(
        self,
        state: MechanismState,
        names: list[str],
    ) -> MechanismDecision | None:
        """Recover a confirmed pair after one member is temporarily hidden.

        Occlusion is common when the player enters a command-coupled object's
        cell.  The pair identity remains valid, but joint planning must pause
        until both positions are observed again.  Use a bounded directional
        experiment episode instead of returning control to a stateless outer
        fallback.  First try to undo the action that caused the occlusion,
        then rotate through the remaining directions.
        """
        player = state.player_position
        if player is None:
            return None
        substantial = tuple(
            sorted(
                signature
                for signature in self.command_dynamic_signatures
                if signature[1] >= 10
            )
        )
        visible = tuple(
            signature
            for signature in substantial
            if signature in self.dynamic_track_positions
        )
        if len(substantial) < 2 or len(visible) >= len(substantial):
            self.command_visibility_attempts.clear()
            return None

        actions = [name for name in names if name in ACTION_DELTAS]
        if not actions:
            return None
        reverse = {
            "ACTION1": "ACTION2",
            "ACTION2": "ACTION1",
            "ACTION3": "ACTION4",
            "ACTION4": "ACTION3",
        }
        previous_action = (
            None
            if self.pending_interaction is None
            else self.pending_interaction.action
        )
        preferred = reverse.get(previous_action or "")
        first_recovery_step = not self.command_visibility_attempts
        action = min(
            actions,
            key=lambda candidate: (
                self.command_visibility_attempts[candidate],
                0 if first_recovery_step and candidate == preferred else 1,
                actions.index(candidate),
            ),
        )
        self.command_visibility_attempts[action] += 1
        return MechanismDecision(
            action=action,
            objective="recover_command_visibility",
            target=player,
            route_length=1,
        )

    def _command_contact_frontier(
        self,
        state: MechanismState,
        names: list[str],
        dimension: str | None,
    ) -> MechanismDecision | None:
        """Choose an informative edge on a short route to dynamic contact.

        Exact joint planning is intentionally conservative, but falling back
        to arbitrary state-action coverage can spend the entire resource
        budget learning transitions that do not approach a task-relevant
        interaction.  This search uses the learned local motion model as an
        optimistic hypothesis, replans after every real observation, and
        selects the shortest low-uncertainty route that would make the player
        and one command-coupled member touch or cross.

        No predicted transition is committed as fact.  Unknown edges merely
        identify which next experiment has a plausible causal purpose.
        """
        player = state.player_position
        signatures = tuple(
            sorted(
                signature
                for signature in self.command_dynamic_signatures
                if signature[1] >= 10 and signature in self.dynamic_track_positions
            )
        )
        if player is None or len(signatures) < 2 or state.layout is None:
            return None
        actions = [name for name in names if name in ACTION_DELTAS]
        if not actions:
            return None

        initial_positions = tuple(
            self.dynamic_track_positions[signature] for signature in signatures
        )
        initial_offsets = tuple(
            (
                position[0] - initial_positions[0][0],
                position[1] - initial_positions[0][1],
            )
            for position in initial_positions[1:]
        )
        passable = set(self.known_walkable) | set(state.walkable_cells) | {player}
        horizon = min(12, max(0, self._resource_steps(state) - 3))
        if horizon <= 0:
            return None

        SearchState = tuple[Cell, tuple[Cell, ...], tuple[str, ...], int, int]
        queue: deque[SearchState] = deque([(player, initial_positions, (), 0, 0)])
        best_cost: dict[tuple[Cell, tuple[Cell, ...]], tuple[int, int, int]] = {
            (player, initial_positions): (0, 0, 0)
        }
        candidates: list[
            tuple[
                tuple[int, int, int, tuple[str, ...]],
                tuple[str, ...],
                tuple[int, int, str],
                Cell,
            ]
        ] = []

        while queue:
            source, positions, path, unknown_object_edges, unknown_player_edges = (
                queue.popleft()
            )
            if len(path) >= horizon:
                continue
            for action in actions:
                moving_to = tuple(
                    self._predict_command_destination(
                        state, signature, position, action
                    )
                    for signature, position in zip(signatures, positions)
                )
                next_offsets = tuple(
                    (
                        position[0] - moving_to[0][0],
                        position[1] - moving_to[0][1],
                    )
                    for position in moving_to[1:]
                )
                if (
                    len(set(moving_to)) != len(moving_to)
                    or next_offsets != initial_offsets
                ):
                    continue

                object_edge_unknown = sum(
                    (signature, position, action) not in self.dynamic_action_transitions
                    for signature, position in zip(signatures, positions)
                )
                observed_player = self.observed_transitions.get((source, action))
                dx, dy = ACTION_DELTAS[action]
                attempted = (source[0] + dx, source[1] + dy)
                if observed_player is not None:
                    destination = observed_player
                    player_edge_unknown = 0
                else:
                    destination = attempted if attempted in passable else source
                    player_edge_unknown = 1

                next_path = path + (action,)
                next_unknown_objects = unknown_object_edges + object_edge_unknown
                next_unknown_player = unknown_player_edges + player_edge_unknown
                contacted: list[tuple[tuple[int, int, str], Cell]] = []
                for signature, moving_from, moved_to in zip(
                    signatures, positions, moving_to
                ):
                    if (destination == source and moved_to == source) or (
                        source == moved_to and destination == moving_from
                    ):
                        retry_key = (signature, source, action)
                        if self.dynamic_contact_retries[retry_key] < 4:
                            contacted.append((signature, moved_to))
                for signature, contact_cell in contacted:
                    candidates.append(
                        (
                            (
                                next_unknown_objects,
                                len(next_path),
                                next_unknown_player,
                                next_path,
                            ),
                            next_path,
                            signature,
                            contact_cell,
                        )
                    )
                if contacted:
                    continue

                key = (destination, moving_to)
                cost = (
                    next_unknown_objects,
                    len(next_path),
                    next_unknown_player,
                )
                previous_cost = best_cost.get(key)
                if previous_cost is not None and previous_cost <= cost:
                    continue
                best_cost[key] = cost
                queue.append(
                    (
                        destination,
                        moving_to,
                        next_path,
                        next_unknown_objects,
                        next_unknown_player,
                    )
                )

        if not candidates:
            return None
        _, path, signature, contact_cell = min(candidates, key=lambda item: item[0])
        immediate_contact = len(path) == 1
        return MechanismDecision(
            action=path[0],
            objective=(
                f"probe_command_dynamic_{dimension or 'unknown'}"
                if immediate_contact
                else f"approach_command_frontier_{dimension or 'unknown'}"
            ),
            target=contact_cell,
            route_length=len(path),
            dynamic_signature=signature if immediate_contact else None,
        )

    def _predict_command_destination(
        self,
        state: MechanismState,
        signature: tuple[int, int, str],
        source: Cell,
        action: str,
    ) -> Cell:
        """Predict one command-coupled member from local observed dynamics.

        Exact state-action evidence is authoritative. When it is absent, use
        the nearest observation of the same member under the same command if
        it was a local move or a blocked move. This preserves learned walls
        and lanes instead of assuming every command translates every object.
        Pure geometric extrapolation is only the final prior.
        """
        exact = self.dynamic_action_transitions.get((signature, source, action))
        if exact is not None:
            return exact

        nearby: list[tuple[int, Cell]] = []
        for (
            seen_signature,
            seen_source,
            seen_action,
        ), destination in self.dynamic_action_transitions.items():
            if seen_signature != signature or seen_action != action:
                continue
            delta = (
                destination[0] - seen_source[0],
                destination[1] - seen_source[1],
            )
            if abs(delta[0]) + abs(delta[1]) > 1:
                continue
            distance = abs(source[0] - seen_source[0]) + abs(source[1] - seen_source[1])
            nearby.append((distance, delta))
        if nearby:
            nearest_distance = min(distance for distance, _ in nearby)
            deltas = [
                delta for distance, delta in nearby if distance == nearest_distance
            ]
            counts: dict[Cell, int] = defaultdict(int)
            for delta in deltas:
                counts[delta] += 1
            dx, dy = min(
                counts,
                key=lambda delta: (
                    -counts[delta],
                    0 if delta == ACTION_DELTAS[action] else 1,
                    delta,
                ),
            )
        else:
            dx, dy = ACTION_DELTAS[action]

        proposed = (source[0] + dx, source[1] + dy)
        if state.layout is None:
            return source
        if (
            0 <= proposed[0] < state.layout.cols
            and 0 <= proposed[1] < state.layout.rows
        ):
            return proposed
        return source

    def _command_joint_interception(
        self,
        state: MechanismState,
        names: list[str],
        dimension: str | None,
    ) -> MechanismDecision | None:
        """Search player and synchronized command-coupled members jointly."""
        player = state.player_position
        signatures = tuple(
            sorted(
                signature
                for signature in self.command_dynamic_signatures
                if signature[1] >= 10 and signature in self.dynamic_track_positions
            )
        )
        if player is None or len(signatures) < 2:
            return None
        horizon = min(6, max(0, self._resource_steps(state) - 2))
        if horizon <= 0:
            return None

        preferred: set[tuple[int, int, str]] = set()
        if dimension in {"pattern", "both"}:
            preferred.update(self.dynamic_pattern_signatures)
        if dimension in {"palette", "both"}:
            preferred.update(self.dynamic_palette_signatures)

        initial_positions = tuple(
            self.dynamic_track_positions[signature] for signature in signatures
        )
        initial_offsets = tuple(
            (
                position[0] - initial_positions[0][0],
                position[1] - initial_positions[0][1],
            )
            for position in initial_positions[1:]
        )
        passable = set(self.known_walkable) | set(state.walkable_cells) | {player}
        action_order = [name for name in names if name in ACTION_DELTAS]
        predicted_path = tuple[tuple[Cell, tuple[Cell, ...]], ...]
        queue: deque[tuple[Cell, tuple[Cell, ...], tuple[str, ...], predicted_path]] = (
            deque([(player, initial_positions, (), ())])
        )
        visited = {(player, initial_positions)}
        candidates: list[
            tuple[
                tuple[
                    int,
                    int,
                    int,
                    int,
                    tuple[str, ...],
                    tuple[int, int, str],
                ],
                tuple[str, ...],
                tuple[int, int, str],
                Cell,
                predicted_path,
            ]
        ] = []

        while queue:
            source, positions, path, predicted_states = queue.popleft()
            if len(path) >= horizon:
                continue
            for action in action_order:
                # Joint planning is deliberately conservative: only execute a
                # multi-step contact plan through object transitions observed
                # at these exact cells. Unknown commands are handled below as
                # one-step model experiments and replanned after observation.
                if not all(
                    (signature, position, action) in self.dynamic_action_transitions
                    for signature, position in zip(signatures, positions)
                ):
                    continue
                moving_to = tuple(
                    self._predict_command_destination(
                        state, signature, position, action
                    )
                    for signature, position in zip(signatures, positions)
                )
                next_offsets = tuple(
                    (
                        position[0] - moving_to[0][0],
                        position[1] - moving_to[0][1],
                    )
                    for position in moving_to[1:]
                )
                # The observed members form one rigid command-coupled pair.
                # Independent fallback extrapolation must not collapse them
                # into one cell or change their relative arrangement. Exact
                # wrap/reset transitions preserve this invariant as well.
                if (
                    len(set(moving_to)) != len(moving_to)
                    or next_offsets != initial_offsets
                ):
                    continue
                dx, dy = ACTION_DELTAS[action]
                attempted = (source[0] + dx, source[1] + dy)
                observed_player = self.observed_transitions.get((source, action))
                if observed_player is not None:
                    destination = observed_player
                elif (
                    attempted in passable
                    and attempted not in positions
                    and attempted not in moving_to
                ):
                    destination = attempted
                else:
                    destination = source

                next_path = path + (action,)
                next_states = predicted_states + ((destination, moving_to),)
                contacted: list[tuple[tuple[int, int, str], Cell, int]] = []
                for signature, moving_from, moved_to in zip(
                    signatures, positions, moving_to
                ):
                    if (destination == source and moved_to == source) or (
                        source == moved_to and destination == moving_from
                    ):
                        retry_key = (signature, source, action)
                        if self.dynamic_contact_retries[retry_key] < 4:
                            # Prefer the mechanism moving into a stationary
                            # player. Walking the player into a token is still
                            # a useful fallback, but it does not test the
                            # command-coupled mechanism as directly.
                            contact_kind = 0 if destination == source else 1
                            contacted.append((signature, moved_to, contact_kind))
                for signature, contact_cell, contact_kind in contacted:
                    player_moves = sum(
                        previous_player != next_player
                        for previous_player, (next_player, _) in zip(
                            (player,)
                            + tuple(
                                predicted_player
                                for predicted_player, _ in predicted_states
                            ),
                            next_states,
                        )
                    )
                    priority = (
                        0 if signature in preferred else 1,
                        self.dynamic_signature_attempts[signature],
                        contact_kind,
                        player_moves,
                        next_path,
                        signature,
                    )
                    candidates.append(
                        (
                            priority,
                            next_path,
                            signature,
                            contact_cell,
                            next_states,
                        )
                    )
                if contacted:
                    continue

                key = (destination, moving_to)
                if key in visited:
                    continue
                visited.add(key)
                queue.append((destination, moving_to, next_path, next_states))

        if not candidates:
            return None
        _, path, signature, contact_cell, predicted_states = min(
            candidates,
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2],
                item[0][3],
                len(item[1]),
                item[0][4],
                item[0][5],
            ),
        )
        self._dynamic_plan = deque(path[1:])
        self._dynamic_plan_signature = signature
        self._dynamic_plan_target = contact_cell
        self._dynamic_plan_dimension = dimension
        self._dynamic_plan_joint_signatures = signatures
        self._dynamic_plan_expected_states = deque(predicted_states)
        return MechanismDecision(
            action=path[0],
            objective=(
                f"contact_command_dynamic_{dimension or 'unknown'}"
                if len(path) == 1
                else f"approach_command_dynamic_{dimension or 'unknown'}"
            ),
            target=contact_cell,
            route_length=len(path),
            dynamic_signature=signature if len(path) == 1 else None,
        )

    def _dynamic_interception(
        self,
        state: MechanismState,
        names: list[str],
        dimension: str | None,
    ) -> MechanismDecision | None:
        """Plan a short observation-only interception of a moving mechanism."""
        player = state.player_position
        if player is None:
            return None
        joint = self._command_joint_interception(state, names, dimension)
        if joint is not None:
            return joint
        # A confirmed synchronized pair must only be reasoned about in the
        # joint state space.  The legacy single-track interception below can
        # otherwise chase one member while ignoring the other, producing
        # contradictory ``approach_dynamic`` and static-discovery objectives.
        if self._has_command_pair():
            return None
        if not self.exhausted_probe_cells:
            return None
        dynamic_signatures = (
            self.autonomous_dynamic_signatures | self.command_dynamic_signatures
        )
        substantial = {
            signature
            for signature in dynamic_signatures
            if signature[1] >= 10 and signature in self.dynamic_track_positions
        }
        if len(substantial) < 2:
            return None

        preferred: set[tuple[int, int, str]] = set()
        if dimension in {"pattern", "both"}:
            preferred.update(self.dynamic_pattern_signatures)
        if dimension in {"palette", "both"}:
            preferred.update(self.dynamic_palette_signatures)
        signatures = sorted(
            substantial,
            key=lambda signature: (
                0 if signature in preferred else 1,
                signature[1],
                signature,
            ),
        )
        horizon = min(6, max(0, self._resource_steps(state) - 2))
        if horizon <= 0:
            return None
        passable = set(self.known_walkable) | set(state.walkable_cells) | {player}
        action_order = [name for name in names if name in ACTION_DELTAS]
        current_dynamic_occupancy = {
            cell: signature
            for signature, cell in self.dynamic_track_positions.items()
            if signature in substantial
        }
        candidates: list[tuple[int, tuple[str, ...], tuple[int, int, str], Cell]] = []
        for signature in signatures:
            current = self.dynamic_track_positions[signature]
            command_coupled = signature in self.command_dynamic_signatures
            delta = self._recent_unit_delta(signature)
            # A confirmed autonomous track can pause at a phase boundary or
            # after contact.  Treat it as stationary for approach planning,
            # while reserving velocity extrapolation for an immediately
            # observed unit displacement.
            if delta is None:
                delta = (0, 0)
            projected = [
                (current[0] + delta[0] * step, current[1] + delta[1] * step)
                for step in range(horizon + 1)
            ]
            queue: deque[tuple[Cell, Cell, tuple[str, ...]]] = deque(
                [(player, current, ())]
            )
            visited = {(player, current, 0)}
            while queue:
                source, moving_from, path = queue.popleft()
                elapsed = len(path)
                if elapsed >= horizon:
                    continue
                next_elapsed = elapsed + 1
                for action in action_order:
                    if command_coupled:
                        moving_to = self.dynamic_action_transitions.get(
                            (signature, moving_from, action)
                        )
                        if moving_to is None:
                            mdx, mdy = ACTION_DELTAS[action]
                            proposed = (
                                moving_from[0] + mdx,
                                moving_from[1] + mdy,
                            )
                            moving_to = (
                                proposed
                                if state.layout is not None
                                and 0 <= proposed[0] < state.layout.cols
                                and 0 <= proposed[1] < state.layout.rows
                                else moving_from
                            )
                    else:
                        moving_from = projected[elapsed]
                        moving_to = projected[next_elapsed]
                    dx, dy = ACTION_DELTAS[action]
                    attempted = (source[0] + dx, source[1] + dy)
                    destination = attempted if attempted in passable else source
                    contact_cell: Cell | None = None
                    if attempted == moving_from:
                        contact_cell = moving_from
                    elif attempted == moving_to or destination == moving_to:
                        contact_cell = moving_to
                    elif source == moving_to and destination == moving_from:
                        contact_cell = moving_from
                    next_path = path + (action,)
                    retry_key = (signature, source, action)
                    if (
                        contact_cell is not None
                        and self.dynamic_contact_retries[retry_key] < 4
                    ):
                        candidates.append(
                            (len(next_path), next_path, signature, contact_cell)
                        )
                        queue.clear()
                        break
                    # Never route through another dynamic member, or through a
                    # target contact relation whose bounded retry budget is
                    # exhausted.  The floor detector can expose the underlying
                    # tile as walkable even though the rendered object blocks
                    # the actual action.
                    occupying_signature = current_dynamic_occupancy.get(attempted)
                    if occupying_signature is not None:
                        continue
                    key = (destination, moving_to, next_elapsed)
                    if key in visited:
                        continue
                    visited.add(key)
                    queue.append((destination, moving_to, next_path))
        if not candidates:
            return None
        _, path, signature, contact_cell = min(
            candidates,
            key=lambda item: (
                0 if item[2] in preferred else 1,
                self.dynamic_signature_attempts[item[2]],
                item[0],
                item[1],
                item[2],
            ),
        )
        self._dynamic_plan = deque(path[1:])
        self._dynamic_plan_signature = signature
        self._dynamic_plan_target = contact_cell
        self._dynamic_plan_dimension = dimension
        return MechanismDecision(
            action=path[0],
            objective=(
                f"contact_dynamic_{dimension or 'unknown'}"
                if len(path) == 1
                else f"approach_dynamic_{dimension or 'unknown'}"
            ),
            target=contact_cell,
            route_length=len(path),
            dynamic_signature=signature if len(path) == 1 else None,
        )

    def _resume_dynamic_plan(
        self,
        state: MechanismState,
        names: list[str],
    ) -> MechanismDecision | None:
        if not self._dynamic_plan:
            return None
        if self._dynamic_plan_joint_signatures:
            expected = (
                None
                if not self._dynamic_plan_expected_states
                else self._dynamic_plan_expected_states.popleft()
            )
            actual_positions = tuple(
                self.dynamic_track_positions.get(signature)
                for signature in self._dynamic_plan_joint_signatures
            )
            if (
                expected is None
                or state.player_position != expected[0]
                or actual_positions != expected[1]
            ):
                self._clear_dynamic_plan()
                return None
        action = self._dynamic_plan[0]
        if action not in names or self._resource_steps(state) <= 2:
            self._clear_dynamic_plan()
            return None
        self._dynamic_plan.popleft()
        final_step = not self._dynamic_plan
        decision = MechanismDecision(
            action=action,
            objective=(
                f"contact_dynamic_{self._dynamic_plan_dimension or 'unknown'}"
                if final_step
                else f"approach_dynamic_{self._dynamic_plan_dimension or 'unknown'}"
            ),
            target=self._dynamic_plan_target,
            route_length=len(self._dynamic_plan) + 1,
            dynamic_signature=(self._dynamic_plan_signature if final_step else None),
        )
        if final_step:
            self._dynamic_plan_signature = None
            self._dynamic_plan_target = None
            self._dynamic_plan_dimension = None
            self._dynamic_plan_joint_signatures = ()
            self._dynamic_plan_expected_states.clear()
        return decision

    def _dynamic_staging(
        self,
        state: MechanismState,
        names: list[str],
    ) -> MechanismDecision | None:
        """Wait at a learned safe junction instead of chasing distant tracks."""
        player = state.player_position
        positions = [
            cell
            for signature, cell in self.dynamic_track_positions.items()
            if signature in self.autonomous_dynamic_signatures and signature[1] >= 10
        ]
        if player is None or len(positions) < 2:
            return None
        distance = min(
            abs(player[0] - cell[0]) + abs(player[1] - cell[1]) for cell in positions
        )
        if distance <= 2:
            return None

        staging_cells = set(self.refill_cells) | set(self.exhausted_probe_cells)
        choice = self._nearest_route(state, staging_cells)
        if choice is None:
            return None
        staging, route = choice
        if route:
            return self._decision_for_route(
                route,
                "return_to_dynamic_staging",
                staging,
                names,
            )
        if self.dynamic_phase_waits[staging] >= 16:
            return None
        blocked = [
            action
            for action in names
            if self.observed_transitions.get((staging, action)) == staging
        ]
        if not blocked:
            return None
        self.dynamic_phase_waits[staging] += 1
        return MechanismDecision(
            action=blocked[0],
            objective="wait_dynamic_phase",
            target=staging,
            route_length=1,
        )

    def _route(
        self,
        state: MechanismState,
        target: Cell,
        *,
        allow_unsafe_target_transition: bool = False,
    ) -> list[str] | None:
        start = state.player_position
        if start is None:
            return None
        if start == target:
            return []
        passable = (
            set(self.known_walkable) | set(state.walkable_cells) | {start, target}
        )
        blocked = set(self.known_blocked) - passable
        uncertainty_cost = 4 if self.special_transitions else 1
        # Prefer topology that has been observed. In mechanism levels an
        # apparently adjacent move may be a portal; treating an untried
        # geometric edge as equally reliable can invalidate the route at every
        # step and consume the whole resource bar. Unknown edges remain usable
        # for exploration, but a slightly longer verified route now wins.
        frontier: list[tuple[int, int, int, Cell]] = [(0, 0, 0, start)]
        insertion_order = 0
        best_cost: dict[Cell, int] = {start: 0}
        best_steps: dict[Cell, int] = {start: 0}
        previous: dict[Cell, Cell | None] = {start: None}
        previous_action: dict[Cell, str] = {}
        while frontier:
            cost, steps, _, cell = heappop(frontier)
            if cost != best_cost.get(cell) or steps != best_steps.get(cell):
                continue
            if cell == target:
                break
            for action, (dx, dy) in ACTION_DELTAS.items():
                edge = (cell, action)
                contextual_edge = (
                    cell,
                    action,
                    state.status_signature,
                    state.status_color,
                )
                if contextual_edge in self.contextual_special_transitions:
                    # Reuse a nonlocal transition only in the status context
                    # where its actual destination was observed.
                    neighbor = self.contextual_special_transitions[contextual_edge]
                    edge_cost = 1
                elif edge in self.special_transitions:
                    # We know geometric motion is wrong for this edge, but the
                    # observed relocation belongs to another context.
                    continue
                else:
                    observed = edge in self.observed_transitions
                    neighbor = self.observed_transitions.get(
                        edge,
                        (cell[0] + dx, cell[1] + dy),
                    )
                    edge_cost = 1 if observed else uncertainty_cost
                if contextual_edge in self.unsafe_navigation_edges and not (
                    allow_unsafe_target_transition and neighbor == target
                ):
                    continue
                if neighbor in blocked or neighbor not in passable:
                    continue
                next_cost = cost + edge_cost
                next_steps = steps + 1
                known = (
                    (best_cost[neighbor], best_steps[neighbor])
                    if neighbor in best_cost
                    else None
                )
                if known is not None and known <= (next_cost, next_steps):
                    continue
                best_cost[neighbor] = next_cost
                best_steps[neighbor] = next_steps
                previous[neighbor] = cell
                previous_action[neighbor] = action
                insertion_order += 1
                heappush(
                    frontier,
                    (next_cost, next_steps, insertion_order, neighbor),
                )
        if target not in previous:
            return None
        actions: list[str] = []
        current = target
        while previous[current] is not None:
            actions.append(previous_action[current])
            current = previous[current]  # type: ignore[assignment]
        actions.reverse()
        return actions

    def _nearest_route(
        self, state: MechanismState, cells: Iterable[Cell]
    ) -> tuple[Cell, list[str]] | None:
        routes = [
            (cell, route)
            for cell in sorted(set(cells))
            if (route := self._route(state, cell)) is not None
        ]
        if not routes:
            return None
        return min(routes, key=lambda item: (len(item[1]), item[0]))

    def _resource_steps(self, state: MechanismState) -> int:
        return state.resource_level // max(1, self.resource_cost)

    def _needs_refill(self, state: MechanismState, route_length: int) -> bool:
        return bool(
            self._visible_refill_cells(state)
            and self._resource_steps(state) <= route_length + 3
        )

    def _needs_discovery_refill(
        self,
        state: MechanismState,
        route_length: int,
    ) -> bool:
        """Reserve enough resource to test an unknown mechanism after arrival.

        Reaching an unexplained object is not completion: a controlled test may
        require contact, departure, re-entry and at least one follow-up move.
        Treat those steps as part of the route cost when a probable refill is
        available.  Known actuators and completion targets keep the smaller
        generic safety margin used by ``_needs_refill``.
        """
        # A controlled unknown-object experiment needs enough room for contact,
        # departure/re-entry and an escape/refill route.  Use an absolute
        # seven-step floor as well as the ordinary route-relative margin.  This
        # avoids abandoning exploration too early merely because a refill is
        # nearby, while still turning back before the experiment becomes unsafe.
        return bool(
            self._likely_refill_cells(state)
            and self._resource_steps(state) <= max(route_length + 3, 7)
        )

    def _route_to_refill_for_goal(
        self,
        state: MechanismState,
        goal: Cell,
    ) -> tuple[Cell, list[str]] | None:
        """Choose a visible or untested refill candidate that unlocks a goal."""
        steps = self._resource_steps(state)
        visible = self._visible_refill_cells(state)
        candidates = visible | set(self._likely_refill_cells(state))
        choices: list[tuple[tuple[int, int, int, Cell], Cell, list[str]]] = []
        for cell in candidates:
            route = self._route(state, cell)
            if route is None or len(route) > steps:
                continue
            after_refill = self._route_from(state, cell, goal)
            if after_refill is None:
                continue
            choices.append(
                (
                    (
                        0 if cell in visible else 1,
                        len(route),
                        len(after_refill),
                        cell,
                    ),
                    cell,
                    route,
                )
            )
        if not choices:
            return None
        _, cell, route = min(choices, key=lambda item: item[0])
        return cell, route

    def _route_from(
        self,
        state: MechanismState,
        start: Cell,
        target: Cell,
        *,
        status_signature: str | None = None,
        status_color: int | None = None,
    ) -> list[str] | None:
        original = state.player_position
        if original is None:
            return None
        projected = MechanismState(
            level=state.level,
            game_state=state.game_state,
            layout=state.layout,
            player_position=start,
            player_confidence=state.player_confidence,
            walkable_cells=state.walkable_cells,
            blocked_cells=state.blocked_cells,
            objects=state.objects,
            resource_level=state.resource_level,
            resource_capacity=state.resource_capacity,
            status_pattern=(
                state.status_pattern
                if status_signature is None
                else tuple(
                    tuple(int(value) for value in row)
                    for row in status_signature.split("/")
                    if row
                )
            ),
            matching_target_ids=state.matching_target_ids,
            events=state.events,
            warnings=state.warnings,
            status_color=(state.status_color if status_color is None else status_color),
        )
        return self._route(projected, target)

    def _completion_tail_steps(
        self,
        state: MechanismState,
        start: Cell,
        targets: dict[Cell, MechanismObject],
    ) -> int | None:
        """Estimate post-actuator travel in each target's required context."""
        choices: list[int] = []
        for cell, item in targets.items():
            if cell in self.dynamic_target_cells:
                continue
            route = self._route_from(
                state,
                start,
                cell,
                status_signature=item.pattern_signature,
                status_color=item.pattern_color,
            )
            if route is not None:
                choices.append(len(route))
        return min(choices) if choices else None

    def _decision_for_route(
        self,
        route: list[str],
        objective: str,
        target: Cell,
        names: list[str],
    ) -> MechanismDecision | None:
        if not route or route[0] not in names:
            return None
        return MechanismDecision(
            action=route[0],
            objective=objective,
            target=target,
            route_length=len(route),
        )

    @staticmethod
    def _joint_state(state: MechanismState) -> JointMechanismState:
        return JointMechanismState(state.status_signature, state.status_color)

    def _joint_navigation_decision(
        self,
        state: MechanismState,
        targets: dict[Cell, MechanismObject],
        names: list[str],
    ) -> MechanismDecision | None:
        """Search across status-changing motion before splitting it into Options.

        If a complete route is unsupported, probe a reachable missing actuator
        edge instead of repeating an already known state/action. The model is
        level-local and each chosen step is checked on the next observation.
        """
        if (
            not self.enable_joint_navigation
            or state.player_position is None
            or not self.special_transitions
            or self._has_command_pair()
            or self.active_actuator_probe is not None
            or self.task_plan_controller.active_plan is not None
        ):
            return None
        goals = {
            NavigationNode(
                cell, JointMechanismState(item.pattern_signature, item.pattern_color)
            )
            for cell, item in targets.items()
            if cell not in self.dynamic_target_cells and item.pattern_color is not None
        }
        model = self.joint_navigation
        effect_edges = [
            (source, action, evidence.destination)
            for (source, action), evidence in model.edges.items()
            if evidence.destination is not None
            and source.status != evidence.destination.status
        ]
        if not goals or not effect_edges:
            return None
        start = NavigationNode(
            state.player_position, self._joint_state(state), self.motion_context.context
        )
        search_args = dict(
            resource=state.resource_level,
            capacity=state.resource_capacity,
            default_cost=self.resource_cost,
            passable=self.known_walkable | set(state.walkable_cells) | set(targets),
            actions=names,
        )
        route = model.search(start, goals, **search_args)
        if route is None:
            route = model.search(start, goals, allow_predictions=True, **search_args)
        if route is not None and route.steps:
            step = route.steps[0]
            self._expected_navigation_step = step
            # Joint planning owns the entire mission; an old coordinate-only
            # commitment must not resume underneath it on the next frame.
            self.task_plan_controller.invalidate("joint_navigation_selected")
            return MechanismDecision(
                step.action,
                "joint_navigation_goal",
                route.steps[-1].destination.position,
                len(route.steps),
            )

        # Only existing before/after statuses seed experiments. An unreachable
        # desired status is not invented as an observed state.
        statuses = {start.status}
        for source, _, destination in effect_edges:
            statuses.update((source.status, destination.status))
        patterns = {status.pattern for status in statuses}
        colors = {status.color for status in statuses}
        statuses = {JointMechanismState(p, c) for p in patterns for c in colors}
        frontier: dict[NavigationNode, str] = {}
        for source, action, destination in effect_edges:
            for status in statuses:
                relevant = any(
                    (
                        source.status.color != destination.status.color
                        and start.status.color != g.status.color
                        and status.color != g.status.color
                    )
                    or (
                        source.status.pattern != destination.status.pattern
                        and start.status.pattern != g.status.pattern
                        and status.pattern != g.status.pattern
                    )
                    for g in goals
                )
                node = NavigationNode(source.position, status, source.context)
                if relevant and (node, action) not in model.edges:
                    frontier.setdefault(node, action)
        if not frontier:
            return None
        route = model.search(start, frontier, allow_predictions=True, **search_args)
        if route is None:
            return None
        node = start if not route.steps else route.steps[-1].destination
        if route.resource_remaining < self.resource_cost:
            return None
        self.task_plan_controller.invalidate("joint_navigation_experiment")
        if route.steps:
            step = route.steps[0]
            self._expected_navigation_step = step
            return MechanismDecision(
                step.action,
                "joint_navigation_frontier",
                node.position,
                len(route.steps) + 1,
            )
        action = frontier[node]
        dx, dy = ACTION_DELTAS[action]
        return MechanismDecision(
            action,
            "joint_navigation_experiment",
            (node.position[0] + dx, node.position[1] + dy),
            1,
        )

    def _planning_state(self, state: MechanismState) -> PlanningState:
        return PlanningState(
            level=state.level,
            player=state.player_position,
            joint_status=self._joint_state(state),
            resource_steps=self._resource_steps(state),
            topology_revision=self.topology_revision,
            event_types=frozenset(event.event_type for event in state.events),
        )

    def _observe_task_plan(
        self,
        state: MechanismState,
        targets: dict[Cell, MechanismObject],
    ) -> None:
        self.task_plan_controller.observe(
            self._planning_state(state),
            target_available=lambda cell: (
                cell in targets
                or cell in self.static_target_cells
                or state.player_position == cell
            ),
        )
        if self.task_plan_controller.consume_replan_request():
            # Rebuilding can fail because the remaining resource or newly
            # learned topology makes the mission infeasible.  The old plan's
            # precondition is already contradicted; never keep executing it
            # when no replacement can be installed.
            self.task_plan_controller.invalidate("stage_precondition_changed")
            self._build_task_plan(state, targets)

    def _estimate_plan_steps(
        self,
        state: MechanismState,
        *,
        start: Cell,
        transitions: tuple[JointMechanismTransition, ...],
        target: Cell,
        goal: JointMechanismState,
    ) -> int | None:
        """Estimate a complete learned actuator-to-target mission."""

        position = start
        total = 0
        for transition in transitions:
            approach = transition.approach or transition.actuator
            route = self._route_from(
                state,
                position,
                approach,
                status_signature=transition.source.pattern,
                status_color=transition.source.color,
            )
            if route is None:
                return None
            contact_steps = 1 if transition.contact_action is not None else 0
            total += len(route) + contact_steps
            if not route and contact_steps == 0:
                total += 2
            position = transition.observed_result or transition.actuator
        route = self._route_from(
            state,
            position,
            target,
            status_signature=goal.pattern,
            status_color=goal.color,
        )
        if route is None:
            return None
        return total + len(route) + 2

    def _build_task_plan(
        self,
        state: MechanismState,
        targets: dict[Cell, MechanismObject],
    ) -> bool:
        """Install the cheapest resource-feasible plan supported by evidence.

        Planning is intentionally gated on at least one learned joint
        transition.  Before that point the existing causal exploration policy
        remains responsible for discovering actuator roles and effects.
        """

        player = state.player_position
        transitions_known = self.joint_mechanism_graph.transitions()
        if player is None or not targets or not transitions_known:
            return False

        current = self._joint_state(state)
        candidates: list[
            tuple[
                int,
                Cell,
                JointMechanismState,
                tuple[JointMechanismTransition, ...],
            ]
        ] = []
        for target, item in targets.items():
            if target in self.dynamic_target_cells:
                continue
            goal = JointMechanismState(item.pattern_signature, item.pattern_color)
            transitions = self.joint_mechanism_graph.shortest_path(current, goal)
            if transitions is None:
                continue
            estimate = self._estimate_plan_steps(
                state,
                start=player,
                transitions=transitions,
                target=target,
                goal=goal,
            )
            if estimate is not None:
                candidates.append((estimate, target, goal, transitions))
        if not candidates:
            return False

        estimate, target, goal, transitions = min(
            candidates,
            key=lambda item: (item[0], len(item[3]), item[1]),
        )
        stages: list[PlanStage] = []
        plan_start = player
        available_steps = self._resource_steps(state)
        safety_margin = 2

        if available_steps <= estimate + safety_margin:
            refill_choices: list[tuple[int, Cell, int]] = []
            for refill in sorted(self._visible_refill_cells(state)):
                route = self._route(state, refill)
                if route is None or len(route) > available_steps:
                    continue
                after_refill = self._estimate_plan_steps(
                    state,
                    start=refill,
                    transitions=transitions,
                    target=target,
                    goal=goal,
                )
                if after_refill is None:
                    continue
                refill_capacity = state.resource_capacity // max(1, self.resource_cost)
                if refill_capacity <= after_refill + safety_margin:
                    continue
                refill_choices.append((len(route), refill, after_refill))
            if not refill_choices:
                return False
            refill_distance, refill, estimate = min(refill_choices)
            stages.append(
                PlanStage(
                    kind=PlanStageKind.REFILL,
                    target=refill,
                    expected_before=current,
                    expected_after=current,
                    estimated_steps=refill_distance,
                )
            )
            plan_start = refill

        position = plan_start
        for transition in transitions:
            approach = transition.approach or transition.actuator
            route = self._route_from(
                state,
                position,
                approach,
                status_signature=transition.source.pattern,
                status_color=transition.source.color,
            )
            if route is None:
                return False
            stages.append(
                PlanStage(
                    kind=PlanStageKind.APPLY_ACTUATOR,
                    target=transition.actuator,
                    expected_before=transition.source,
                    expected_after=transition.destination,
                    estimated_steps=(
                        len(route) + (1 if transition.contact_action else 0)
                        if route or transition.contact_action
                        else 2
                    ),
                    approach=transition.approach,
                    contact_action=transition.contact_action,
                    observed_result=transition.observed_result,
                )
            )
            position = transition.observed_result or transition.actuator
        final_route = self._route_from(
            state,
            position,
            target,
            status_signature=goal.pattern,
            status_color=goal.color,
        )
        if final_route is None:
            return False
        stages.append(
            PlanStage(
                kind=PlanStageKind.CONTACT_TARGET,
                target=target,
                expected_before=goal,
                expected_after=goal,
                estimated_steps=len(final_route),
            )
        )
        self.task_plan_controller.install(
            level=state.level,
            goal=goal,
            stages=stages,
            topology_revision=self.topology_revision,
        )
        return True

    def _task_plan_decision(
        self,
        state: MechanismState,
        names: list[str],
    ) -> MechanismDecision | None:
        plan = self.task_plan_controller.active_plan
        if plan is None or plan.current_stage is None:
            return None
        stage = plan.current_stage
        navigation_target = (
            stage.approach
            if stage.kind is PlanStageKind.APPLY_ACTUATOR
            and stage.approach is not None
            and stage.contact_action is not None
            else stage.target
        )
        route = self._route(state, navigation_target)
        if route is None:
            fallback = self._nearest_route(
                state,
                {
                    navigation_target,
                    *self.changer_cells,
                    *self.palette_changer_cells,
                },
            )
            if fallback is None:
                self._task_plan_route_failures += 1
                if self._task_plan_route_failures >= 8:
                    self.task_plan_controller.invalidate("stage_unreachable")
                    self._task_plan_route_failures = 0
                return None
            target, route = fallback
            return self._decision_for_route(
                route,
                "task_plan_stage_approach",
                target,
                names,
            )
        self._task_plan_route_failures = 0
        objective = {
            PlanStageKind.REFILL: "task_plan_refill",
            PlanStageKind.APPLY_ACTUATOR: "task_plan_apply_actuator",
            PlanStageKind.CONTACT_TARGET: "task_plan_contact_target",
        }[stage.kind]
        if route:
            route_objective = (
                "task_plan_approach_actuator"
                if navigation_target != stage.target
                else objective
            )
            return self._decision_for_route(
                route,
                route_objective,
                navigation_target,
                names,
            )
        if (
            stage.kind is PlanStageKind.APPLY_ACTUATOR
            and stage.approach is not None
            and stage.contact_action is not None
            and state.player_position == stage.approach
            and stage.contact_action in names
        ):
            return MechanismDecision(
                action=stage.contact_action,
                objective=objective,
                target=stage.target,
                route_length=1,
            )
        if stage.kind is PlanStageKind.APPLY_ACTUATOR:
            decision = self._depart_action(state, names, stage.target)
            if decision is not None:
                return MechanismDecision(
                    action=decision.action,
                    objective="task_plan_reenter_actuator",
                    target=decision.target,
                    route_length=decision.route_length,
                )
        contact = self._contact_probe(state, stage.target, names)
        if contact is None:
            return None
        return MechanismDecision(
            action=contact.action,
            objective=objective,
            target=stage.target,
            route_length=contact.route_length,
        )

    def _depart_action(
        self, state: MechanismState, names: list[str], target: Cell
    ) -> MechanismDecision | None:
        for action in names:
            if (
                self.enable_dynamic_contact
                and self.observed_transitions.get((target, action)) == target
            ):
                continue
            dx, dy = ACTION_DELTAS[action]
            neighbor = (target[0] + dx, target[1] + dy)
            if neighbor in self.known_walkable or neighbor in state.walkable_cells:
                self._depart_and_return = target
                return MechanismDecision(
                    action=action,
                    objective="depart_status_changer",
                    target=neighbor,
                    route_length=1,
                )
        return None

    def _continue_actuator_probe(
        self,
        state: MechanismState,
        names: list[str],
    ) -> MechanismDecision | None:
        """Continue the active depart/re-enter experiment until a cycle closes."""
        probe = self.active_actuator_probe
        if probe is None or probe in self.completed_actuator_probes:
            self.active_actuator_probe = None
            return None
        target, dimension = probe
        route = self._route(state, target, allow_unsafe_target_transition=True)
        if route is None:
            self.active_actuator_probe = None
            return None
        if route:
            return self._decision_for_route(
                route,
                f"characterize_status_{dimension}",
                target,
                names,
            )
        for action in names:
            dx, dy = ACTION_DELTAS[action]
            neighbor = (target[0] + dx, target[1] + dy)
            if (
                self.enable_dynamic_contact
                and self.observed_transitions.get((target, action)) == target
            ):
                # An apparently open floor cell is weaker evidence than the
                # real failed departure. Do not burn the meter on that edge.
                continue
            if neighbor in self.known_walkable or neighbor in state.walkable_cells:
                return MechanismDecision(
                    action=action,
                    objective=f"depart_status_{dimension}",
                    target=neighbor,
                    route_length=1,
                )
        return None

    def _dynamic_contact_decision(
        self,
        state: MechanismState,
        targets: dict[Cell, MechanismObject],
        names: list[str],
    ) -> MechanismDecision | None:
        """Use a supported moving contact, never a permanent actuator cell.

        Two contacts are a bounded identification experiment. Thereafter only
        a supported pattern orbit or discovery of a still-missing role may
        interrupt the ordinary policy. All routes execute one action at a time.
        """
        model = self.dynamic_contact
        if (
            not self.enable_dynamic_contact
            or self._has_command_pair()
            or state.player_position is None
            or state.player_confidence < 0.8
            or len(model.moving) != 1
            or self.task_plan_controller.active_plan is not None
        ):
            return None
        signature = next(iter(model.moving))
        track = model.tracks.get(signature)
        if track is None or track.phase is None:
            return None
        # Do not disturb a fully aligned completion objective.
        goals = [
            item
            for cell, item in targets.items()
            if cell not in self.dynamic_target_cells
        ]
        if any(
            item.internal_pattern == state.status_pattern
            and self._colors_compatible(item.pattern_color, state.status_color)
            for item in goals
        ):
            return None

        def destination(source: Cell, action: str) -> Cell | None:
            edge = (source, action)
            context = (source, action, state.status_signature, state.status_color)
            # Nonlocal/unsafe edges may advance the clock differently. Their
            # effects are outside this deliberately local contact search.
            if (
                edge in self.special_transitions
                or context in self.unsafe_navigation_edges
            ):
                return None
            observed = self.observed_transitions.get(edge)
            dx, dy = ACTION_DELTAS[action]
            target = (
                observed if observed is not None else (source[0] + dx, source[1] + dy)
            )
            if target not in (
                self.known_walkable | set(state.walkable_cells) | {source}
            ):
                return None
            return target

        samples = model.effect_samples[(signature, True)]
        desired = None
        objective = "identify_dynamic_contact"
        if samples >= model.MIN_SUPPORT:
            next_pattern = model.pattern_after(signature, state.status_pattern)
            if next_pattern is None:
                return None
            # Find a reachable target in the learned D4 orbit, not merely an
            # arbitrary status change that would repeat forever.
            orbit = {state.status_pattern}
            pattern = state.status_pattern
            reachable = False
            for _ in range(7):
                pattern = model.pattern_after(signature, pattern)
                if pattern is None or pattern in orbit:
                    break
                orbit.add(pattern)
                if any(item.internal_pattern == pattern for item in goals):
                    reachable = True
                    break
            if reachable and next_pattern != state.status_pattern:
                desired = next_pattern
                objective = "align_dynamic_pattern"
            elif (
                model.preserves_color(signature)
                and any(
                    not self._colors_compatible(item.pattern_color, state.status_color)
                    for item in goals
                )
                and not self.palette_changer_cells
            ):
                # This mechanism cannot correct the missing palette. Test an
                # unexplained static candidate instead of rotating indefinitely.
                moving_cells = {
                    item.current
                    for key, item in model.tracks.items()
                    if key in model.moving
                }
                for cell in self._role_discovery_cells(state, "palette"):
                    if cell in moving_cells:
                        continue
                    route = self._route(state, cell)
                    if route is None:
                        continue
                    if self._needs_discovery_refill(state, len(route)):
                        refill = self._route_to_refill_for_goal(state, cell)
                        if refill is not None and refill[0] != state.player_position:
                            return self._decision_for_route(
                                refill[1],
                                "restore_resource"
                                if refill[0] in self._visible_refill_cells(state)
                                else "discover_resource_source",
                                refill[0],
                                names,
                            )
                    if route:
                        return self._decision_for_route(
                            route, "discover_palette_changer", cell, names
                        )
                    return self._contact_probe(state, cell, names)
                return None
            else:
                return None
        horizon = min(6, max(0, self._resource_steps(state) - 3))
        proposal = model.contact_route(
            state,
            signature,
            destination,
            names,
            goal_pattern=desired,
            horizon=horizon,
        )
        if proposal is None:
            return None
        path, contact = proposal
        if self._needs_discovery_refill(state, len(path)):
            return None
        return MechanismDecision(
            path[0],
            objective,
            contact.player_after,
            len(path),
            signature if len(path) == 1 else None,
        )

    def choose(
        self,
        state: MechanismState | None,
        available_actions: Iterable[Any],
        *,
        previous_action: str | None = None,
    ) -> GameAction | None:
        if not self.enabled or state is None or state.layout is None:
            return None
        names = self._available_names(available_actions)
        if (
            state.level < 1
            or state.layout.confidence < 0.8
            or state.layout.projected_view
            or not all(name in names for name in self.DIRECTION_NAMES)
        ):
            return None
        if self.level != state.level:
            self._reset_level(state.level)
        self._update_evidence(state, actual_action=previous_action)
        if (
            self.enable_entity_options
            and self.enable_dynamic_contact
            and self.enable_contact_effects
        ):
            self.entity_options.observe(state, self.dynamic_contact, previous_action)
        if self.enable_option_compiler and self.enable_dynamic_contact:
            self.option_controller.observe(
                state,
                previous_action,
                self.dynamic_contact,
                self.joint_mechanism_graph.transitions(),
            )

        # A synchronized command pair is stronger evidence than the compact
        # static-looking cell beneath the player. Stop any old depart/re-enter
        # probe immediately so joint planning receives the remaining budget.
        if self._has_command_pair():
            self._probe_return = None
            self._dynamic_probe_departures.clear()

        if state.player_position is None:
            for name in names:
                if name not in self.identity_probes:
                    self.identity_probes.add(name)
                    decision = MechanismDecision(
                        action=name,
                        objective="resolve_player_identity",
                    )
                    return self._set_decision(state, decision)
            return None

        targets = self._static_targets(state)
        self._observe_task_plan(state, targets)

        if (
            self.enable_entity_options
            and self.enable_dynamic_contact
            and self.enable_contact_effects
            and not self._has_command_pair()
        ):
            step = self.entity_options.choose(
                state,
                self.dynamic_contact,
                self.contact_effects,
                {
                    cell: (item.internal_pattern, item.pattern_color)
                    for cell, item in targets.items()
                    if cell not in self.dynamic_target_cells
                },
                names,
                special_transitions=self.contextual_special_transitions,
                known_nonlocal_edges=set(self.special_transitions.keys()),
                nonlocal_edge_destinations=self.special_transitions,
            )
            if step is not None:
                self.task_plan_controller.invalidate("entity_option_selected")
                self._probe_return = None
                self.active_actuator_probe = None
                self._clear_dynamic_plan()
                return self._set_decision(
                    state,
                    MechanismDecision(
                        step.action,
                        self.entity_options.objective,
                        step.after.player,
                        len(self.entity_options.remaining) + 1,
                        step.contact,
                    ),
                )

        if (
            self.enable_option_compiler
            and self.enable_dynamic_contact
            and not self._has_command_pair()
        ):
            step = self.option_controller.choose(
                state,
                self.dynamic_contact,
                (
                    (
                        cell,
                        JointMechanismState(item.pattern_signature, item.pattern_color),
                    )
                    for cell, item in sorted(targets.items())
                    if cell not in self.dynamic_target_cells
                ),
                names,
                resource_cost=self.resource_cost,
            )
            if step is not None:
                self.task_plan_controller.invalidate("compiled_option_selected")
                self._probe_return = None
                self.active_actuator_probe = None
                self._clear_dynamic_plan()
                return self._set_decision(
                    state,
                    MechanismDecision(
                        action=step.action,
                        objective="execute_compiled_option",
                        target=step.after.player,
                        route_length=len(self.option_controller.remaining) + 1,
                    ),
                )

        dynamic_contact_decision = self._dynamic_contact_decision(state, targets, names)
        if dynamic_contact_decision is not None:
            self.task_plan_controller.invalidate("dynamic_contact_selected")
            self._probe_return = None
            self.active_actuator_probe = None
            return self._set_decision(state, dynamic_contact_decision)

        # Execute an installed multi-stage plan before allowing local
        # objectives to compete again.  Dynamic command-coupled mechanisms
        # retain their dedicated joint planner; this first task-planning slice
        # covers only learned static actuator transitions.
        if not self._has_command_pair():
            planned = self._task_plan_decision(state, names)
            if planned is not None:
                return self._set_decision(state, planned)

        joint_decision = self._joint_navigation_decision(state, targets, names)
        if joint_decision is not None:
            return self._set_decision(state, joint_decision)

        pending_completion = self.pending_interaction
        event_types = {event.event_type for event in state.events}
        if event_types.intersection(
            {"status_pattern_changed", "status_palette_changed", "level_advanced"}
        ):
            self._clear_dynamic_plan()
        if (
            pending_completion is not None
            and pending_completion.objective == "contact_matching_static_object"
            and state.player_position == pending_completion.source
            and self._committed_state_matches(state)
            and "level_advanced" not in event_types
        ):
            retry_key = (
                pending_completion.target,
                pending_completion.action,
            )
            if self.completion_contact_retries[retry_key] < 1:
                self.completion_contact_retries[retry_key] += 1
                return self._set_decision(
                    state,
                    MechanismDecision(
                        action=pending_completion.action,
                        objective="contact_matching_static_object",
                        target=pending_completion.target,
                        route_length=1,
                    ),
                )

        resumed_probe = self._resume_probe(state, names)
        if resumed_probe is not None:
            return self._set_decision(state, resumed_probe)

        resumed_dynamic = self._resume_dynamic_plan(state, names)
        if resumed_dynamic is not None:
            return self._set_decision(state, resumed_dynamic)

        reached_decision = self.last_decision
        if (
            reached_decision is not None
            and not self._has_command_pair()
            and reached_decision.target == state.player_position
            and reached_decision.objective
            in {
                "visit_uncontacted_candidate",
                "discover_resource_source",
                "restore_resource",
                "discover_pattern_changer",
                "discover_palette_changer",
                "discover_both_changer",
                "probe_candidate_contact",
            }
            and reached_decision.target not in self.refill_cells
            and reached_decision.target not in self.changer_cells
            and reached_decision.target not in self.palette_changer_cells
            and reached_decision.target not in self.static_target_cells
        ):
            decision = self._contact_probe(
                state,
                reached_decision.target,
                names,
            )
            if decision is not None:
                return self._set_decision(state, decision)

        committed_target = self._matching_commitment(state, targets)
        if committed_target is not None:
            route_choice = self._nearest_route(state, [committed_target])
            if route_choice is not None:
                target, route = route_choice
                if self._resource_steps(state) <= len(route) + 3:
                    refill = self._route_to_refill_for_goal(state, target)
                    if refill is not None and refill[0] != state.player_position:
                        decision = self._decision_for_route(
                            refill[1],
                            (
                                "restore_resource"
                                if refill[0] in self._visible_refill_cells(state)
                                else "discover_resource_source"
                            ),
                            refill[0],
                            names,
                        )
                        if decision is not None:
                            return self._set_decision(state, decision)
                decision = self._decision_for_route(
                    route, "contact_matching_static_object", target, names
                )
                if decision is None and not route:
                    decision = self._contact_probe(state, target, names)
                if decision is not None:
                    return self._set_decision(state, decision)

        actuator_probe = self._continue_actuator_probe(state, names)
        if actuator_probe is not None:
            return self._set_decision(state, actuator_probe)

        alignment_dimension = self._alignment_dimension(state, targets)

        dynamic_interception = self._dynamic_interception(
            state,
            names,
            alignment_dimension,
        )
        if (
            dynamic_interception is not None
            and (dynamic_interception.route_length or 0) == 1
        ):
            return self._set_decision(state, dynamic_interception)

        dynamic_staging = self._dynamic_staging(state, names)
        if dynamic_staging is not None:
            # _dynamic_interception() evaluates a complete candidate and stores
            # its continuation.  If staging wins instead, that proposal was
            # rejected and must not leak into the next decision.
            self._clear_dynamic_plan()
            return self._set_decision(state, dynamic_staging)

        if dynamic_interception is not None:
            return self._set_decision(state, dynamic_interception)

        # Once the controllable pair is known, never fall back to alternating
        # between a generic candidate and a staging cell. If joint search does
        # not yet know a contact path, learn the missing command transition at
        # the current joint state.
        if self._has_command_pair():
            visibility_recovery = self._command_visibility_recovery(state, names)
            if visibility_recovery is not None:
                return self._set_decision(state, visibility_recovery)

            command_frontier = self._command_contact_frontier(
                state,
                names,
                alignment_dimension,
            )
            if command_frontier is not None:
                return self._set_decision(state, command_frontier)

            command_experiment = self._command_model_experiment(
                state,
                names,
                alignment_dimension,
            )
            if command_experiment is not None:
                return self._set_decision(state, command_experiment)

            # Do not hand control back to static role discovery after the
            # dynamic mechanism has been identified.  Returning no decision
            # lets the outer agent's exact-state exploration choose a bounded
            # fallback without reviving incompatible object hypotheses.
            return None

        if (
            alignment_dimension is not None
            and self.active_actuator_probe is None
            and self._build_task_plan(state, targets)
        ):
            planned = self._task_plan_decision(state, names)
            if planned is not None:
                return self._set_decision(state, planned)

        missing_role = (
            (alignment_dimension == "palette" and not self.palette_changer_cells)
            or (alignment_dimension == "pattern" and not self.changer_cells)
            or (
                alignment_dimension == "both"
                and (not self.palette_changer_cells or not self.changer_cells)
            )
        )
        if missing_role:
            role_candidates = self._role_discovery_cells(
                state,
                alignment_dimension,
            )
            for target in role_candidates:
                route = self._route(state, target)
                if route is None:
                    continue
                if self._needs_discovery_refill(state, len(route)):
                    refill = self._route_to_refill_for_goal(state, target)
                    if refill is not None and refill[0] != state.player_position:
                        decision = self._decision_for_route(
                            refill[1],
                            (
                                "restore_resource"
                                if refill[0] in self._visible_refill_cells(state)
                                else "discover_resource_source"
                            ),
                            refill[0],
                            names,
                        )
                        if decision is not None:
                            return self._set_decision(state, decision)
                decision = self._decision_for_route(
                    route,
                    f"discover_{alignment_dimension}_changer",
                    target,
                    names,
                )
                if decision is None and not route:
                    decision = self._contact_probe(state, target, names)
                if decision is not None:
                    return self._set_decision(state, decision)

        # A first changer contact is often reached near the end of the initial
        # budget.  Before spending the last moves cycling it, test the nearest
        # compact ring-like candidate as a possible refill.  Its role is still
        # learned only from a real resource_increased event.
        if not self.refill_cells and self._resource_steps(state) <= 5:
            emergency = self._nearest_route(state, self._likely_refill_cells(state))
            if emergency is not None:
                target, route = emergency
                decision = self._decision_for_route(
                    route, "discover_resource_source", target, names
                )
                if decision is not None:
                    return self._set_decision(state, decision)

        changer_cells: set[Cell] = set()
        alignment_objective = "align_status_pattern"
        if alignment_dimension == "palette" and self.palette_changer_cells:
            changer_cells = self.palette_changer_cells
            alignment_objective = "align_status_palette"
        elif alignment_dimension == "pattern" and self.changer_cells:
            changer_cells = self.changer_cells
        elif alignment_dimension == "both":
            # Pattern and palette actuators are independent in the observed
            # mechanism family.  When both dimensions differ, choose the
            # cheapest known correction now; the next observation recomputes
            # the remaining dimension.  Previously this case fell through to
            # generic candidate exploration even after both roles were learned.
            alignment_options: list[tuple[int, str, Cell, list[str]]] = []
            pattern_route = self._nearest_route(state, self.changer_cells)
            if pattern_route is not None:
                alignment_options.append(
                    (
                        len(pattern_route[1]),
                        "align_status_pattern",
                        pattern_route[0],
                        pattern_route[1],
                    )
                )
            palette_route = self._nearest_route(state, self.palette_changer_cells)
            if palette_route is not None:
                alignment_options.append(
                    (
                        len(palette_route[1]),
                        "align_status_palette",
                        palette_route[0],
                        palette_route[1],
                    )
                )
            if alignment_options:
                _, alignment_objective, selected, _ = min(
                    alignment_options,
                    key=lambda item: (item[0], item[1], item[2]),
                )
                changer_cells = {selected}
        elif alignment_dimension is None and self.changer_cells:
            changer_cells = self.changer_cells
        elif alignment_dimension is None and self.palette_changer_cells:
            changer_cells = self.palette_changer_cells
            alignment_objective = "align_status_palette"

        if changer_cells:
            changer = self._nearest_route(state, changer_cells)
            if changer is not None:
                target, route = changer
                tail_steps = self._completion_tail_steps(state, target, targets)
                mission_steps = len(route) + 2
                if tail_steps is not None:
                    mission_steps += tail_steps + 2
                # Full actuator-to-completion reservation is needed once the
                # environment has demonstrated contextual topology. On plain
                # grids the ordinary route margin is sufficient and preserves
                # the faster actuator-characterization sequence.
                reserve_full_mission = bool(self.special_transitions)
                if (
                    reserve_full_mission
                    and self._resource_steps(state) <= mission_steps
                ):
                    refill = self._route_to_refill_for_goal(state, target)
                    if refill is not None and refill[0] != state.player_position:
                        decision = self._decision_for_route(
                            refill[1],
                            (
                                "restore_resource"
                                if refill[0] in self._visible_refill_cells(state)
                                else "discover_resource_source"
                            ),
                            refill[0],
                            names,
                        )
                        if decision is not None:
                            return self._set_decision(state, decision)
                if not route:
                    decision = self._depart_action(state, names, target)
                elif self._needs_refill(state, len(route) + 2):
                    refill = self._nearest_route(
                        state, self._visible_refill_cells(state)
                    )
                    decision = (
                        None
                        if refill is None or refill[0] == state.player_position
                        else self._decision_for_route(
                            refill[1], "restore_resource", refill[0], names
                        )
                    )
                else:
                    decision = self._decision_for_route(
                        route, alignment_objective, target, names
                    )
                if decision is not None:
                    return self._set_decision(state, decision)

        candidates = [
            cell
            for cell in self._candidate_cells(state)
            if cell not in self.no_effect_cells
            and cell not in self.exhausted_probe_cells
        ]
        for target in candidates:
            route = self._route(state, target)
            if route:
                if self._needs_refill(state, len(route)):
                    refill = self._nearest_route(
                        state, self._visible_refill_cells(state)
                    )
                    if refill is not None and refill[0] != state.player_position:
                        decision = self._decision_for_route(
                            refill[1], "restore_resource", refill[0], names
                        )
                        if decision is not None:
                            return self._set_decision(state, decision)
                decision = self._decision_for_route(
                    route, "visit_uncontacted_candidate", target, names
                )
                if decision is not None:
                    return self._set_decision(state, decision)
            elif route == []:
                decision = self._contact_probe(state, target, names)
                if decision is not None:
                    return self._set_decision(state, decision)
        self.pending_interaction = None
        return None

    def summary(self) -> str:
        decision = self.last_decision
        return (
            f"mechanism_policy_level={self.level} "
            f"refills={sorted(self.refill_cells)} "
            f"changers={sorted(self.changer_cells)} "
            f"palette_changers={sorted(self.palette_changer_cells)} "
            f"active_actuator_probe={self.active_actuator_probe} "
            f"exhausted_probes={sorted(self.exhausted_probe_cells)} "
            f"dynamic_tracks={len(self.autonomous_dynamic_signatures)} "
            f"command_tracks={len(self.command_dynamic_signatures)} "
            f"dynamic_targets={sorted(self.dynamic_target_cells)} "
            f"joint_edges={len(self.joint_mechanism_graph.transitions())} "
            f"navigation_edges={len(self.joint_navigation.edges)} "
            f"navigation_conflicts={self.joint_navigation.counterexamples} "
            f"motion_context_tracks={len(self.motion_context.context)} "
            f"contact_tracks={len(self.dynamic_contact.moving)} "
            f"contact_effect_samples={sum(self.dynamic_contact.effect_samples.values())} "
            f"compiled_plans={self.option_controller.metrics['installed_plans']} "
            f"compiled_checks={self.option_controller.metrics['prediction_checks']} "
            f"compiled_errors={self.option_controller.metrics['prediction_errors']} "
            f"navigation_predictions={self.joint_prediction_checks} "
            f"navigation_errors={self.joint_prediction_errors} "
            f"navigation_search={self.joint_navigation.last_search} "
            f"{self.task_plan_controller.summary()} "
            f"objective={None if decision is None else decision.objective} "
            f"target={None if decision is None else decision.target}"
        )
