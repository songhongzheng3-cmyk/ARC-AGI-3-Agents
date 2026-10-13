"""Run bounded, LLM-free sampling experiments against one ARC game.

This tool deliberately stops before the agent/controller layer.  It records
the observations needed to decide whether a sampling policy learned anything:
state hashes, object summaries, changed cells, level progress, and action
effects.  A run is useful even when it does not win, because its JSONL output
can be replayed when designing a mode-specific solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any

from arc_agi import Arcade
from arcengine import GameAction, GameState
from dotenv import load_dotenv

from agents.episode_memory import EpisodeMemory
from agents.game_model import GameModel
from agents.object_extractor import GridSnapshot, ObjectExtractor

logger = logging.getLogger("sampling_probe")


@dataclass
class SampleRecord:
    step: int
    action: str
    data: dict[str, Any]
    before_hash: str
    after_hash: str
    changed: bool
    changed_cells: int
    changed_bbox: tuple[int, int, int, int] | None
    color_transitions: dict[str, int]
    player_before: tuple[int, int] | None
    player_after: tuple[int, int] | None
    player_delta: tuple[int, int]
    blocked: bool
    contact_objects: list[str]
    before_object_count: int
    after_object_count: int
    before_objects: list[dict[str, Any]]
    after_objects: list[dict[str, Any]]
    levels_before: int
    levels_after: int
    state_after: str
    available_after: list[str]
    target_distance_before: int | None
    target_distance_after: int | None
    interaction_distance_before: int | None
    interaction_distance_after: int | None


def _action_name(value: Any) -> str:
    if isinstance(value, GameAction):
        return value.name
    try:
        return GameAction.from_id(int(value)).name
    except (KeyError, TypeError, ValueError):
        return str(value).upper()


def _available(frame: Any) -> list[str]:
    names: list[str] = []
    for value in getattr(frame, "available_actions", []) or []:
        name = _action_name(value)
        if name not in names:
            names.append(name)
    return names


def _grid(frame: Any) -> list[list[int]]:
    layers = _to_lists(getattr(frame, "frame", []) or [])
    grid = layers[-1] if layers else []
    return [list(row) for row in grid]


def _to_lists(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _to_lists(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_to_lists(item) for item in value]
    return value


def _normalize_frame(raw: Any) -> Any:
    """Match Agent._convert_raw_frame_data for direct environment calls."""
    layers = _to_lists(getattr(raw, "frame", []) or [])
    return SimpleNamespace(
        frame=layers,
        state=getattr(raw, "state", None),
        levels_completed=int(getattr(raw, "levels_completed", 0)),
        win_levels=getattr(raw, "win_levels", None),
        available_actions=getattr(raw, "available_actions", []) or [],
    )


def _state_hash(frame: Any) -> str:
    payload = json.dumps(_grid(frame), separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _changed_cells(before: Any, after: Any) -> int:
    left = _grid(before)
    right = _grid(after)
    return sum(
        1
        for row_left, row_right in zip(left, right)
        for value_left, value_right in zip(row_left, row_right)
        if value_left != value_right
    )


def _change_details(
    before: Any, after: Any
) -> tuple[tuple[int, int, int, int] | None, dict[str, int]]:
    left = _grid(before)
    right = _grid(after)
    points: list[tuple[int, int]] = []
    transitions: dict[str, int] = {}
    for y, (row_left, row_right) in enumerate(zip(left, right)):
        for x, (value_left, value_right) in enumerate(zip(row_left, row_right)):
            if value_left == value_right:
                continue
            points.append((x, y))
            key = f"{int(value_left)}->{int(value_right)}"
            transitions[key] = transitions.get(key, 0) + 1
    if not points:
        return None, transitions
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys)), transitions


def _object_dict(snapshot: GridSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "id": item.object_id,
            "color": item.color,
            "area": item.area,
            "bbox": item.bbox,
            "centroid": item.centroid,
            "shape": item.shape,
            "touches_boundary": item.touches_boundary,
        }
        for item in snapshot.objects
    ]


class SamplingProbe:
    """A bounded sampler with several deterministic exploration policies."""

    def __init__(
        self,
        game_id: str,
        strategy: str,
        max_steps: int,
        click_limit: int,
        output_dir: Path,
        probe_action: str | None = None,
        sequence: list[str] | None = None,
    ) -> None:
        self.game_id = game_id
        self.strategy = strategy
        self.max_steps = max(1, max_steps)
        self.click_limit = max(0, click_limit)
        self.output_dir = output_dir
        self.probe_action = probe_action.upper() if probe_action else None
        self.sequence = [item.upper() for item in (sequence or [])]
        self.sequence_index = 0
        self.dfs_stack: list[tuple[str, str | None]] = []
        self.dfs_visited: set[str] = set()
        self.dfs_tried: dict[str, set[str]] = {}
        self.dfs_last_before: str | None = None
        self.dfs_last_action: str | None = None
        self.nav_edges: dict[tuple[tuple[int, int], str], tuple[int, int]] = {}
        self.nav_tried: dict[tuple[int, int], set[str]] = {}
        self.nav_seen: set[tuple[int, int]] = set()
        self.nav_pending: tuple[tuple[int, int], str, tuple[int, int]] | None = None
        self.nav_position: tuple[int, int] = (0, 0)
        self.nav_pixel_position: tuple[int, int] | None = None
        self.nav_landmark_target: tuple[int, int] | None = None
        self.nav_landmark_cell: tuple[int, int] | None = None
        self.nav_landmark_contacted = False
        self.nav_static_initialized = False
        self.nav_preferred_frontier: str | None = None
        self.extractor = ObjectExtractor()
        self.model = GameModel()
        self.global_trials: dict[str, int] = {}
        self.state_trials: dict[tuple[str, str], int] = {}
        self.clicks_by_state: dict[str, set[tuple[int, int]]] = {}
        self.blocked_actions_by_navigation: dict[str, set[str]] = {}
        self.interaction_centers: list[tuple[int, int]] = []
        self.interaction_contacted: set[tuple[int, int]] = set()
        self.round_robin_index = 0
        self.records: list[SampleRecord] = []

    @staticmethod
    def _key(name: str, data: dict[str, Any] | None = None) -> str:
        return EpisodeMemory.action_key(name, data)

    def _candidates(self, frame: Any, state_hash: str) -> list[dict[str, Any]]:
        tried = self.clicks_by_state.get(state_hash, set())
        return EpisodeMemory.click_candidates(
            frame, tried=tried, max_candidates=self.click_limit
        )

    @staticmethod
    def _inverse(name: str) -> str | None:
        return {
            "ACTION1": "ACTION2",
            "ACTION2": "ACTION1",
            "ACTION3": "ACTION4",
            "ACTION4": "ACTION3",
        }.get(name.upper())

    def _navigation_key(self, frame: Any) -> str:
        """Ignore HUD/animation churn and retain the moving avatar geometry."""
        snapshot = self.extractor.extract(frame)
        dynamic = [
            (item.color, item.bbox, item.area)
            for item in snapshot.objects
            if item.color in {9, 12}
        ]
        return json.dumps(dynamic, separators=(",", ":"))

    def _navigation_position(self, frame: Any) -> tuple[int, int] | None:
        """Position-only state, intentionally insensitive to sprite animation."""
        snapshot = self.extractor.extract(frame)
        avatar = self.model._avatar(snapshot)
        return self.model._center(avatar) if avatar is not None else None

    def _nearest_frontier_action(
        self,
        start: tuple[int, int],
        names: list[str],
    ) -> str | None:
        """Route through observed edges to the nearest node with an unknown edge."""
        queue = deque([start])
        first_action: dict[tuple[int, int], str | None] = {start: None}
        distance: dict[tuple[int, int], int] = {start: 0}
        frontiers: list[tuple[tuple[int, int], str]] = []
        while queue:
            node = queue.popleft()
            untried = [
                name
                for name in names
                if name not in self.nav_tried.setdefault(node, set())
            ]
            frontiers.extend((node, name) for name in untried)
            for name in names:
                neighbor = self.nav_edges.get((node, name))
                if neighbor is None or neighbor == node or neighbor in first_action:
                    continue
                first_action[neighbor] = first_action[node] or name
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
        if not frontiers:
            return None
        preferred = self.nav_preferred_frontier
        node, action = min(
            frontiers,
            key=lambda item: (
                0 if item[1] == preferred else 1,
                distance[item[0]],
                item[0][0],
                item[0][1],
                names.index(item[1]),
            ),
        )
        return first_action[node] or action

    @staticmethod
    def _logical_delta(action: str) -> tuple[int, int]:
        return {
            "ACTION1": (0, -1),
            "ACTION2": (0, 1),
            "ACTION3": (-1, 0),
            "ACTION4": (1, 0),
        }[action]

    def _landmark_action(
        self,
        frame: Any,
        names: list[str],
    ) -> str | None:
        """Approach the nearest salient object once before frontier search."""
        if self.nav_landmark_contacted:
            return None
        snapshot = self.extractor.extract(frame)
        avatar = self.model._avatar(snapshot)
        if avatar is None:
            return None
        player = self.model._center(avatar)
        if self.nav_landmark_target is None:
            candidates = self.model.interaction_candidates(self.model.initial)
            if not candidates:
                self.nav_landmark_contacted = True
                return None
            self.nav_landmark_target = tuple(candidates[0]["center"])
        target = self.nav_landmark_target
        if self.nav_landmark_cell is not None and self.nav_position == self.nav_landmark_cell:
            self.nav_landmark_contacted = True
            return None
        if abs(player[0] - target[0]) + abs(player[1] - target[1]) <= 4:
            self.nav_landmark_contacted = True
            return None
        if self.nav_landmark_cell is None:
            return None
        queue = deque([self.nav_position])
        first: dict[tuple[int, int], str | None] = {self.nav_position: None}
        while queue:
            node = queue.popleft()
            if node == self.nav_landmark_cell:
                return first[node]
            for name in names:
                neighbor = self.nav_edges.get((node, name))
                if neighbor is None or neighbor == node or neighbor in first:
                    continue
                first[neighbor] = first[node] or name
                queue.append(neighbor)
        return None

    def _initialize_static_navigation(self, frame: Any) -> None:
        if self.nav_static_initialized:
            return
        self.nav_static_initialized = True
        grid = _grid(frame)
        snapshot = self.extractor.extract(frame)
        avatar = self.model._avatar(snapshot)
        if not grid or avatar is None:
            return
        cell = max(1, avatar.bbox[2] - avatar.bbox[0] + 1)
        structural = [
            item
            for item in snapshot.objects
            if item.color not in self.model._avatar_colors and not item.touches_boundary
        ]
        floor = max(structural, key=lambda item: item.area, default=None)
        if floor is None:
            return
        row_counts = [sum(value == floor.color for value in row) for row in grid]
        threshold = cell * 4
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, count in enumerate([*row_counts, 0]):
            if count >= threshold and start is None:
                start = index
            elif count < threshold and start is not None:
                runs.append((start, index - 1))
                start = None
        if not runs:
            return
        y0, y1 = max(runs, key=lambda run: run[1] - run[0])
        floor_x = [
            x
            for y in range(y0, y1 + 1)
            for x, value in enumerate(grid[y])
            if value == floor.color
        ]
        if not floor_x:
            return
        x0, x1 = min(floor_x), max(floor_x)
        cols = (x1 - x0 + 1) // cell
        rows = (y1 - y0 + 1) // cell
        avatar_center = self.model._center(avatar)
        start_cell = ((avatar_center[0] - x0) // cell, (avatar_center[1] - y0) // cell)
        known: set[tuple[int, int]] = set()
        for row in range(rows):
            for col in range(cols):
                tile = [
                    grid[y][x]
                    for y in range(y0 + row * cell, y0 + (row + 1) * cell)
                    for x in range(x0 + col * cell, x0 + (col + 1) * cell)
                ]
                if sum(value != snapshot.background for value in tile) >= len(tile) // 2:
                    known.add((col - start_cell[0], row - start_cell[1]))
        for node in known:
            for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
                dx, dy = self._logical_delta(name)
                neighbor = (node[0] + dx, node[1] + dy)
                absolute = (neighbor[0] + start_cell[0], neighbor[1] + start_cell[1])
                if neighbor in known:
                    self.nav_edges[(node, name)] = neighbor
                    self.nav_tried.setdefault(node, set()).add(name)
                elif 0 <= absolute[0] < cols and 0 <= absolute[1] < rows:
                    self.nav_edges[(node, name)] = node
                    self.nav_tried.setdefault(node, set()).add(name)

        interactions = self.model.interaction_candidates(self.model.initial)
        if interactions:
            tx, ty = interactions[0]["center"]
            self.nav_landmark_cell = (
                (int(tx) - x0) // cell - start_cell[0],
                (int(ty) - y0) // cell - start_cell[1],
            )
        targets = self.model.target_candidates(self.model.initial)
        if targets:
            tx, ty = targets[0]["center"]
            if ty < y0:
                self.nav_preferred_frontier = "ACTION1"
            elif ty > y1:
                self.nav_preferred_frontier = "ACTION2"
            elif tx < x0:
                self.nav_preferred_frontier = "ACTION3"
            elif tx > x1:
                self.nav_preferred_frontier = "ACTION4"

    def _choose_frontier_nav(
        self, frame: Any
    ) -> tuple[GameAction, dict[str, Any]] | None:
        names = [
            name
            for name in _available(frame)
            if name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
        ]
        position = self._navigation_position(frame)
        if position is None or not names:
            return None
        self._initialize_static_navigation(frame)
        if self.nav_pixel_position is None:
            self.nav_pixel_position = position
        self.nav_seen.add(self.nav_position)

        if self.nav_pending is not None:
            previous, action, previous_pixel = self.nav_pending
            self.nav_tried.setdefault(previous, set()).add(action)
            moved = position != previous_pixel
            if moved:
                delta = self._logical_delta(action)
                current = (previous[0] + delta[0], previous[1] + delta[1])
            else:
                current = previous
            self.nav_position = current
            self.nav_edges[(previous, action)] = current
            if current != previous:
                reverse = self._inverse(action)
                if reverse is not None:
                    self.nav_edges[(current, reverse)] = previous
                    self.nav_tried.setdefault(current, set()).add(reverse)
            self.nav_seen.add(current)
            self.nav_pending = None
            self.nav_pixel_position = position

        action = self._landmark_action(frame, names)
        if action is None:
            action = self._nearest_frontier_action(self.nav_position, names)
        if action is None:
            return None
        self.nav_pending = (self.nav_position, action, position)
        return GameAction.from_name(action), {}

    def _choose_graph_dfs(
        self, frame: Any, state_hash: str
    ) -> tuple[GameAction, dict[str, Any]] | None:
        node_key = self._navigation_key(frame)
        if not self.dfs_stack:
            self.dfs_stack.append((node_key, None))
            self.dfs_visited.add(node_key)

        # Close the edge taken on the previous turn and update the DFS path.
        if self.dfs_last_before is not None and self.dfs_last_action is not None:
            previous = self.dfs_last_before
            key = self._key(self.dfs_last_action)
            self.dfs_tried.setdefault(previous, set()).add(key)
            if node_key != previous:
                stack_states = [item[0] for item in self.dfs_stack]
                if node_key in stack_states:
                    # Returning through the inverse edge means the parent
                    # already knows this direction; mark it before popping
                    # the child so DFS explores the next branch.
                    parent_state = node_key
                    self.dfs_tried.setdefault(parent_state, set()).add(
                        self._key(self.dfs_last_action)
                    )
                    while self.dfs_stack and self.dfs_stack[-1][0] != node_key:
                        self.dfs_stack.pop()
                elif node_key not in self.dfs_visited:
                    self.dfs_stack.append((node_key, self.dfs_last_action))
                    self.dfs_visited.add(node_key)

        names = [
            name
            for name in _available(frame)
            if name not in {"RESET", "ACTION6"}
        ]
        tried = self.dfs_tried.setdefault(node_key, set())
        for name in names:
            if self._key(name) not in tried:
                self.dfs_last_before = node_key
                self.dfs_last_action = name
                return GameAction.from_name(name), {}

        # Dead-end: undo the action that entered this node and continue at its
        # parent. The reverse edge is intentionally allowed even if it was
        # already observed at the child.
        if len(self.dfs_stack) > 1:
            _, entry_action = self.dfs_stack[-1]
            reverse = self._inverse(entry_action or "")
            self.dfs_last_before = node_key
            self.dfs_last_action = reverse
            if reverse is not None and reverse in _available(frame):
                return GameAction.from_name(reverse), {}
        return None

    def _choose(self, frame: Any, state_hash: str) -> tuple[GameAction, dict[str, Any]] | None:
        names = [
            name
            for name in _available(frame)
            if name not in {"RESET", "ACTION6"}
        ]
        if not names:
            names = [name for name in _available(frame) if name != "RESET"]
        if not names:
            return None

        if self.strategy == "graph_dfs":
            return self._choose_graph_dfs(frame, state_hash)

        if self.strategy == "frontier_nav":
            return self._choose_frontier_nav(frame)

        if self.strategy == "sequence":
            if self.sequence_index >= len(self.sequence):
                return None
            name = self.sequence[self.sequence_index]
            self.sequence_index += 1
            if name not in _available(frame):
                raise ValueError(f"sequence action {name!r} is not available")
            return GameAction.from_name(name), {}

        if self.strategy == "single":
            if self.probe_action not in names:
                raise ValueError(
                    f"probe action {self.probe_action!r} is not available; got {names}"
                )
            return GameAction.from_name(self.probe_action), {}

        if self.strategy == "target_probe":
            # Calibrate all directional effects once, then prefer actions that
            # reduce distance to the filtered target.  A blocked action is
            # still recorded, but receives a temporary penalty in this state.
            for name in names:
                if self.global_trials.get(name, 0) == 0:
                    return GameAction.from_name(name), {}
            snapshot = self.extractor.extract(frame)
            target_distance = self.model.target_distance(snapshot)
            effects = self.model.action_effects
            if target_distance is not None and effects:
                candidates: list[tuple[tuple[int, int, int], str]] = []
                navigation_key = self._navigation_key(frame)
                blocked_at_navigation = self.blocked_actions_by_navigation.get(
                    navigation_key, set()
                )
                for name in names:
                    effect = effects.get(name)
                    if effect is None or not effect.dx:
                        continue
                    avatar = self.model._avatar(snapshot)
                    if avatar is None:
                        continue
                    center = self.model._center(avatar)
                    projected = (center[0] + int(median(effect.dx)), center[1] + int(median(effect.dy)))
                    targets = self.model.target_candidates(snapshot)
                    projected_distance = min(
                        abs(projected[0] - int(item["center"][0]))
                        + abs(projected[1] - int(item["center"][1]))
                        for item in targets
                    ) if targets else target_distance
                    blocked = self.state_trials.get((state_hash, name), 0)
                    if name in blocked_at_navigation:
                        # A repeated action in the exact same navigation state
                        # already proved that edge blocked.  Explore another
                        # edge before spending more samples on it.
                        projected_distance += 100
                    # Distance is primary; local retries and globally common
                    # actions are secondary tie-breakers.
                    candidates.append(((projected_distance, blocked, self.global_trials.get(name, 0)), name))
                if candidates:
                    _, name = min(candidates)
                    return GameAction.from_name(name), {}

        if self.strategy == "candidate_probe":
            for name in names:
                if self.global_trials.get(name, 0) == 0:
                    return GameAction.from_name(name), {}
            if not self.interaction_centers:
                self.interaction_centers = [
                    tuple(candidate["center"])
                    for candidate in self.model.interaction_candidates(self.model.initial)
                ]
            remaining = [
                center
                for center in self.interaction_centers
                if center not in self.interaction_contacted
            ]
            snapshot = self.extractor.extract(frame)
            avatar = self.model._avatar(snapshot)
            effects = self.model.action_effects
            if remaining and avatar is not None and effects:
                center = self.model._center(avatar)
                navigation_key = self._navigation_key(frame)
                blocked_at_navigation = self.blocked_actions_by_navigation.get(
                    navigation_key, set()
                )
                candidates: list[tuple[tuple[int, int, int], str]] = []
                for name in names:
                    effect = effects.get(name)
                    if effect is None or not effect.dx:
                        continue
                    projected = (
                        center[0] + int(median(effect.dx)),
                        center[1] + int(median(effect.dy)),
                    )
                    distance = min(
                        abs(projected[0] - target[0])
                        + abs(projected[1] - target[1])
                        for target in remaining
                    )
                    if name in blocked_at_navigation:
                        distance += 100
                    candidates.append(
                        ((distance, self.global_trials.get(name, 0), self.state_trials.get((state_hash, name), 0)), name)
                    )
                if candidates:
                    _, name = min(candidates)
                    return GameAction.from_name(name), {}

        candidates = self._candidates(frame, state_hash)
        if self.strategy == "coverage":
            for name in names:
                if self.global_trials.get(name, 0) == 0:
                    return GameAction.from_name(name), {}
            if "ACTION6" in _available(frame) and candidates:
                point = candidates[0]
                return GameAction.ACTION6, {"x": int(point["x"]), "y": int(point["y"])}

        if self.strategy == "frontier":
            # First learn the global action vocabulary, then exhaust each
            # action once in the exact state before revisiting it.
            for name in names:
                if self.global_trials.get(name, 0) == 0:
                    return GameAction.from_name(name), {}
            ordered = sorted(
                names,
                key=lambda name: self.state_trials.get((state_hash, name), 0),
            )
            for name in ordered:
                if self.state_trials.get((state_hash, name), 0) == 0:
                    return GameAction.from_name(name), {}
            if "ACTION6" in _available(frame) and candidates:
                point = candidates[0]
                return GameAction.ACTION6, {"x": int(point["x"]), "y": int(point["y"])}

        if self.strategy == "round_robin":
            name = names[self.round_robin_index % len(names)]
            self.round_robin_index += 1
            return GameAction.from_name(name), {}

        # Once coverage/frontier is exhausted, prefer the least-tested action.
        name = min(
            names,
            key=lambda item: (
                self.global_trials.get(item, 0),
                self.state_trials.get((state_hash, item), 0),
            ),
        )
        return GameAction.from_name(name), {}

    def run(self) -> dict[str, Any]:
        arcade = Arcade()
        scorecard_id = arcade.open_scorecard(tags=["sampling", self.strategy])
        env = arcade.make(self.game_id, scorecard_id=scorecard_id)
        if env is None:
            raise RuntimeError(f"Unable to create environment for {self.game_id}")

        frame = _normalize_frame(env.observation_space)
        reset_raw = env.step(GameAction.RESET, data={}, reasoning={})
        if reset_raw is None:
            raise RuntimeError("RESET returned no frame")
        frame = _normalize_frame(reset_raw)
        state_hash = _state_hash(frame)
        initial_snapshot = self.extractor.extract(frame)
        self.model.reset(initial_snapshot)
        logger.info(
            "initial game=%s strategy=%s state=%s grid=%sx%s objects=%s actions=%s",
            self.game_id,
            self.strategy,
            state_hash[:8],
            initial_snapshot.width,
            initial_snapshot.height,
            len(initial_snapshot.objects),
            ",".join(_available(frame)),
        )

        for step in range(1, self.max_steps + 1):
            if getattr(frame, "state", None) is GameState.WIN:
                break
            choice = self._choose(frame, state_hash)
            if choice is None:
                logger.warning("No legal sample action at step=%s", step)
                break
            action, data = choice
            before_hash = state_hash
            before_snapshot = self.extractor.extract(frame)
            target_distance_before = self.model.target_distance(before_snapshot)
            interaction_distance_before = self.model.interaction_distance(before_snapshot)
            before_level = int(getattr(frame, "levels_completed", 0))
            next_raw = env.step(action, data=data, reasoning={"source": "sampling_probe"})
            if next_raw is None:
                logger.warning("Action %s returned no frame", action.name)
                break
            next_frame = _normalize_frame(next_raw)
            after_hash = _state_hash(next_frame)
            after_snapshot = self.extractor.extract(next_frame)
            target_distance_after = self.model.target_distance(after_snapshot)
            interaction_distance_after = self.model.interaction_distance(after_snapshot)
            changed_bbox, color_transitions = _change_details(frame, next_frame)
            model_evidence = self.model.observe(
                action.name,
                before_snapshot,
                after_snapshot,
                changed_cells=_changed_cells(frame, next_frame),
                level_delta=int(getattr(next_frame, "levels_completed", 0)) - before_level,
            )
            if model_evidence.blocked:
                self.blocked_actions_by_navigation.setdefault(
                    self._navigation_key(frame), set()
                ).add(action.name)
            if model_evidence.contact_objects and self.interaction_centers:
                for candidate in self.model.interaction_candidates(self.model.initial):
                    component_ids = set(candidate.get("component_ids", []))
                    if component_ids.intersection(model_evidence.contact_objects):
                        self.interaction_contacted.add(tuple(candidate["center"]))
            key = self._key(action.name, data)
            self.global_trials[key] = self.global_trials.get(key, 0) + 1
            self.state_trials[(before_hash, key)] = self.state_trials.get(
                (before_hash, key), 0
            ) + 1
            if action is GameAction.ACTION6:
                self.clicks_by_state.setdefault(before_hash, set()).add(
                    (int(data["x"]), int(data["y"]))
                )
            record = SampleRecord(
                step=step,
                action=action.name,
                data=data,
                before_hash=before_hash,
                after_hash=after_hash,
                changed=before_hash != after_hash,
                changed_cells=_changed_cells(frame, next_frame),
                changed_bbox=changed_bbox,
                color_transitions=color_transitions,
                player_before=model_evidence.player_before,
                player_after=model_evidence.player_after,
                player_delta=model_evidence.delta,
                blocked=model_evidence.blocked,
                contact_objects=list(model_evidence.contact_objects),
                before_object_count=len(before_snapshot.objects),
                after_object_count=len(after_snapshot.objects),
                before_objects=_object_dict(before_snapshot),
                after_objects=_object_dict(after_snapshot),
                levels_before=before_level,
                levels_after=int(getattr(next_frame, "levels_completed", 0)),
                state_after=getattr(getattr(next_frame, "state", None), "name", str(getattr(next_frame, "state", ""))),
                available_after=_available(next_frame),
                target_distance_before=target_distance_before,
                target_distance_after=target_distance_after,
                interaction_distance_before=interaction_distance_before,
                interaction_distance_after=interaction_distance_after,
            )
            self.records.append(record)
            logger.info(
                "step=%s action=%s changed=%s cells=%s objects=%s->%s levels=%s->%s state=%s",
                step,
                action.name,
                record.changed,
                record.changed_cells,
                record.before_object_count,
                record.after_object_count,
                record.levels_before,
                record.levels_after,
                record.state_after,
            )
            frame = next_frame
            state_hash = after_hash
            if record.levels_after > record.levels_before:
                logger.info(
                    "level advanced at step=%s; stopping and preserving path",
                    step,
                )
                break

        scorecard = arcade.close_scorecard(scorecard_id)
        summary: dict[str, Any] = {
            "game_id": self.game_id,
            "strategy": self.strategy,
            "requested_steps": self.max_steps,
            "sampled_steps": len(self.records),
            "changed_steps": sum(record.changed for record in self.records),
            "total_changed_cells": sum(record.changed_cells for record in self.records),
            "level_progress": max(
                [record.levels_after for record in self.records] or [0]
            ),
            "won": bool(self.records and self.records[-1].state_after == "WIN"),
            "action_trials": self.global_trials,
            "final_state": self.records[-1].state_after if self.records else "RESET",
            "successful_path": (
                [record.action for record in self.records]
                if any(record.levels_after > record.levels_before for record in self.records)
                else []
            ),
            "scorecard": scorecard.model_dump() if scorecard is not None else None,
            "inference": self.model.summary(after_snapshot if self.records else initial_snapshot),
            "target_distance_start": self.records[0].target_distance_before if self.records else self.model.target_distance(initial_snapshot),
            "target_distance_min": min(
                [value for record in self.records for value in (record.target_distance_before, record.target_distance_after) if value is not None]
                or [None]
            ),
            "target_contact_steps": [
                record.step for record in self.records if record.contact_objects
            ],
            "interaction_candidates_start": self.interaction_centers,
            "interaction_distance_start": self.records[0].interaction_distance_before if self.records else self.model.interaction_distance(initial_snapshot),
            "interaction_distance_min": min(
                [value for record in self.records for value in (record.interaction_distance_before, record.interaction_distance_after) if value is not None]
                or [None]
            ),
            "interaction_contacted": sorted(self.interaction_contacted),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        variant = self.strategy
        if self.strategy == "single" and self.probe_action:
            variant = f"{variant}.{self.probe_action.lower()}"
        prefix = self.output_dir / f"{self.game_id}.{variant}.{stamp}"
        trajectory_path = prefix.parent / f"{prefix.name}.jsonl"
        summary_path = prefix.parent / f"{prefix.name}.summary.json"
        with trajectory_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=True, indent=2)
        summary["trajectory"] = str(trajectory_path)
        summary["summary_file"] = str(summary_path)
        return summary


def main() -> None:
    load_dotenv(dotenv_path=".env.example")
    load_dotenv(dotenv_path=".env", override=True)
    parser = argparse.ArgumentParser(description="Run LLM-free ARC sampling")
    parser.add_argument("--game", default="ls20-9607627b")
    parser.add_argument(
        "--strategy",
        choices=(
            "coverage",
            "frontier",
            "round_robin",
            "single",
            "sequence",
            "graph_dfs",
            "frontier_nav",
            "target_probe",
            "candidate_probe",
        ),
        default="frontier",
    )
    parser.add_argument("--probe-action", choices=("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"))
    parser.add_argument(
        "--sequence",
        help="Comma-separated actions for strategy=sequence, e.g. ACTION3,ACTION1",
    )
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--click-limit", type=int, default=8)
    parser.add_argument("--output-dir", default="sampling_runs")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    summary = SamplingProbe(
        game_id=args.game,
        strategy=args.strategy,
        max_steps=args.steps,
        click_limit=args.click_limit,
        output_dir=Path(args.output_dir),
        probe_action=args.probe_action,
        sequence=args.sequence.split(",") if args.sequence else None,
    ).run()
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
