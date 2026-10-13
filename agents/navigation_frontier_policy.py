"""Observation-only online frontier planning for keyboard navigation games."""

from __future__ import annotations

from collections import deque
from typing import Any

from arcengine import GameAction

from .game_model import GameModel
from .object_extractor import ObjectExtractor


class NavigationFrontierPolicy:
    """Explore a parsed logical grid without memorizing an answer sequence.

    The policy derives a coarse tile map from the current observation, visits
    one salient interior landmark to gather interaction evidence, and then
    routes through known edges to the nearest unexplored boundary edge.  Every
    action is validated against the next observation before updating the map.
    """

    DIRECTION_NAMES = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")

    def __init__(self, enabled: bool = True, max_level: int = 0) -> None:
        self.enabled = enabled
        self.max_level = max_level
        self.extractor = ObjectExtractor()
        self.model = GameModel()
        self.successful_paths: dict[int, tuple[str, ...]] = {}
        self.level: int | None = None
        self.action_history: list[str] = []
        self._reset_local()

    def _reset_local(self) -> None:
        self.edges: dict[tuple[tuple[int, int], str], tuple[int, int]] = {}
        self.tried: dict[tuple[int, int], set[str]] = {}
        self.position = (0, 0)
        self.pixel_position: tuple[int, int] | None = None
        self.pending: tuple[tuple[int, int], str, tuple[int, int]] | None = None
        self.landmark_target: tuple[int, int] | None = None
        self.landmark_cell: tuple[int, int] | None = None
        self.landmark_contacted = False
        self.preferred_frontier: str | None = None
        self.static_initialized = False
        self.applicable = False

    @staticmethod
    def _available_names(frame: Any) -> list[str]:
        names: list[str] = []
        for value in getattr(frame, "available_actions", []) or []:
            if isinstance(value, GameAction):
                name = value.name
            else:
                try:
                    name = GameAction.from_id(int(value)).name
                except (KeyError, TypeError, ValueError):
                    name = str(value).upper()
            if name not in names:
                names.append(name)
        return names

    @staticmethod
    def _grid(frame: Any) -> list[list[int]]:
        layers = getattr(frame, "frame", []) or []
        grid = layers[-1] if layers else []
        return [list(map(int, row)) for row in grid]

    @staticmethod
    def _delta(action: str) -> tuple[int, int]:
        return {
            "ACTION1": (0, -1),
            "ACTION2": (0, 1),
            "ACTION3": (-1, 0),
            "ACTION4": (1, 0),
        }[action]

    @staticmethod
    def _inverse(action: str) -> str:
        return {
            "ACTION1": "ACTION2",
            "ACTION2": "ACTION1",
            "ACTION3": "ACTION4",
            "ACTION4": "ACTION3",
        }[action]

    def _reset_for_level(self, level: int, snapshot: Any) -> None:
        if self.level is not None and level > self.level and self.action_history:
            self.successful_paths[self.level] = tuple(self.action_history)
        self.level = level
        self.action_history = []
        self._reset_local()
        self.model.reset(snapshot)

    def _initialize_static(self, frame: Any) -> None:
        if self.static_initialized:
            return
        self.static_initialized = True
        grid = self._grid(frame)
        snapshot = self.extractor.extract(frame)
        avatar = self.model._avatar(snapshot)
        names = self._available_names(frame)
        if (
            not grid
            or avatar is None
            or not all(name in names for name in self.DIRECTION_NAMES)
        ):
            return

        cell = max(1, avatar.bbox[2] - avatar.bbox[0] + 1)
        structural = [
            item
            for item in snapshot.objects
            if item.color not in self.model._avatar_colors
            and not item.touches_boundary
        ]
        floor = max(structural, key=lambda item: item.area, default=None)
        if floor is None:
            return

        row_counts = [sum(value == floor.color for value in row) for row in grid]
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, count in enumerate([*row_counts, 0]):
            if count >= cell * 4 and start is None:
                start = index
            elif count < cell * 4 and start is not None:
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
        if cols < 2 or rows < 2:
            return

        avatar_center = self.model._center(avatar)
        start_cell = (
            (avatar_center[0] - x0) // cell,
            (avatar_center[1] - y0) // cell,
        )
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
        if len(known) < 3:
            return

        for node in known:
            for name in self.DIRECTION_NAMES:
                dx, dy = self._delta(name)
                neighbor = (node[0] + dx, node[1] + dy)
                absolute = (
                    neighbor[0] + start_cell[0],
                    neighbor[1] + start_cell[1],
                )
                if neighbor in known:
                    self.edges[(node, name)] = neighbor
                    self.tried.setdefault(node, set()).add(name)
                elif 0 <= absolute[0] < cols and 0 <= absolute[1] < rows:
                    self.edges[(node, name)] = node
                    self.tried.setdefault(node, set()).add(name)

        interactions = self.model.interaction_candidates(self.model.initial)
        if interactions:
            tx, ty = interactions[0]["center"]
            self.landmark_target = (int(tx), int(ty))
            self.landmark_cell = (
                (int(tx) - x0) // cell - start_cell[0],
                (int(ty) - y0) // cell - start_cell[1],
            )
        else:
            self.landmark_contacted = True

        targets = self.model.target_candidates(self.model.initial)
        if targets:
            tx, ty = targets[0]["center"]
            if ty < y0:
                self.preferred_frontier = "ACTION1"
            elif ty > y1:
                self.preferred_frontier = "ACTION2"
            elif tx < x0:
                self.preferred_frontier = "ACTION3"
            elif tx > x1:
                self.preferred_frontier = "ACTION4"
        self.applicable = bool(self.edges and self.preferred_frontier)

    def _close_pending(self, pixel_position: tuple[int, int]) -> None:
        if self.pending is None:
            self.pixel_position = pixel_position
            return
        previous, action, previous_pixel = self.pending
        self.tried.setdefault(previous, set()).add(action)
        if pixel_position != previous_pixel:
            dx, dy = self._delta(action)
            current = (previous[0] + dx, previous[1] + dy)
        else:
            current = previous
        self.position = current
        self.edges[(previous, action)] = current
        if current != previous:
            reverse = self._inverse(action)
            self.edges[(current, reverse)] = previous
            self.tried.setdefault(current, set()).add(reverse)
        self.pending = None
        self.pixel_position = pixel_position

    def _route_to(self, target: tuple[int, int], names: list[str]) -> str | None:
        queue = deque([self.position])
        first: dict[tuple[int, int], str | None] = {self.position: None}
        while queue:
            node = queue.popleft()
            if node == target:
                return first[node]
            for name in names:
                neighbor = self.edges.get((node, name))
                if neighbor is None or neighbor == node or neighbor in first:
                    continue
                first[neighbor] = first[node] or name
                queue.append(neighbor)
        return None

    def _landmark_action(self, pixel: tuple[int, int], names: list[str]) -> str | None:
        if self.landmark_contacted:
            return None
        if self.landmark_target is not None:
            distance = abs(pixel[0] - self.landmark_target[0]) + abs(
                pixel[1] - self.landmark_target[1]
            )
            if distance <= 4:
                self.landmark_contacted = True
                return None
        if self.landmark_cell is None:
            self.landmark_contacted = True
            return None
        return self._route_to(self.landmark_cell, names)

    def _frontier_action(self, names: list[str]) -> str | None:
        queue = deque([self.position])
        first: dict[tuple[int, int], str | None] = {self.position: None}
        distance: dict[tuple[int, int], int] = {self.position: 0}
        frontiers: list[tuple[tuple[int, int], str]] = []
        while queue:
            node = queue.popleft()
            frontiers.extend(
                (node, name)
                for name in names
                if name not in self.tried.setdefault(node, set())
            )
            for name in names:
                neighbor = self.edges.get((node, name))
                if neighbor is None or neighbor == node or neighbor in first:
                    continue
                first[neighbor] = first[node] or name
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
        if not frontiers:
            return None
        node, action = min(
            frontiers,
            key=lambda item: (
                0 if item[1] == self.preferred_frontier else 1,
                distance[item[0]],
                item[0][0],
                item[0][1],
                names.index(item[1]),
            ),
        )
        return first[node] or action

    def choose(self, frame: Any) -> GameAction | None:
        if not self.enabled:
            return None
        level = int(getattr(frame, "levels_completed", 0) or 0)
        # This policy solves the introductory navigation-only level.  Later
        # levels may expose the same four movement actions and a parseable map,
        # but their objective is controlled by resource/status mechanisms.  A
        # frontier walk there spends budget before the mechanism policy can act.
        if level > self.max_level:
            return None
        snapshot = self.extractor.extract(frame)
        if self.level is None or level != self.level:
            self._reset_for_level(level, snapshot)
        self._initialize_static(frame)
        if not self.applicable:
            return None

        avatar = self.model._avatar(snapshot)
        if avatar is None:
            return None
        pixel = self.model._center(avatar)
        self._close_pending(pixel)
        names = [
            name
            for name in self._available_names(frame)
            if name in self.DIRECTION_NAMES
        ]
        action_name = self._landmark_action(pixel, names)
        if action_name is None:
            action_name = self._frontier_action(names)
        if action_name is None:
            return None

        self.pending = (self.position, action_name, pixel)
        self.action_history.append(action_name)
        action = GameAction.from_name(action_name)
        action.set_data({})
        return action

    def summary(self) -> str:
        return (
            f"level={self.level} position={self.position} "
            f"landmark_contacted={self.landmark_contacted} "
            f"known_edges={len(self.edges)} actions={len(self.action_history)}"
        )
