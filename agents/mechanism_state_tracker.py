"""Observation-only structural state and mechanism tracking for grid games.

The tracker is deliberately passive: it parses public frames and action
transitions but never selects an action.  It gives planners a compact state
containing the logical grid, resource/status signals, interaction objects and
task-level events without assigning semantic meaning to concrete colours.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any, Iterable

import numpy as np

Cell = tuple[int, int]
Pattern = tuple[tuple[int, ...], ...]
BBox = tuple[int, int, int, int]

ACTION_DELTAS: dict[str, Cell] = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}


@dataclass(frozen=True)
class GridLayout:
    origin: Cell
    cell_size: Cell
    cols: int
    rows: int
    floor_color: int
    background_color: int
    confidence: float
    projected_view: bool = False

    @property
    def bounds(self) -> BBox:
        x, y = self.origin
        width, height = self.cell_size
        return (x, y, x + self.cols * width - 1, y + self.rows * height - 1)


@dataclass(frozen=True)
class MechanismObject:
    object_id: str
    color: int
    cell: Cell
    bbox: BBox
    area: int
    shape_signature: str
    internal_pattern: Pattern = ()
    multi_palette: bool = False
    floor_ratio: float = 0.0
    static_target: bool = False
    pattern_color: int | None = None

    @property
    def pattern_signature(self) -> str:
        return pattern_key(self.internal_pattern)


@dataclass(frozen=True)
class MechanismEvent:
    event_type: str
    action: str | None = None
    before: Any = None
    after: Any = None
    cells: tuple[Cell, ...] = ()
    object_id: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class MechanismState:
    level: int
    game_state: str
    layout: GridLayout | None
    player_position: Cell | None
    player_confidence: float
    walkable_cells: frozenset[Cell]
    blocked_cells: frozenset[Cell]
    objects: tuple[MechanismObject, ...]
    resource_level: int
    resource_capacity: int
    status_pattern: Pattern
    matching_target_ids: tuple[str, ...]
    events: tuple[MechanismEvent, ...] = ()
    warnings: tuple[str, ...] = ()
    status_color: int | None = None
    player_component_ids: tuple[str, ...] = ()
    player_bbox: BBox | None = None

    @property
    def status_signature(self) -> str:
        return pattern_key(self.status_pattern)

    @property
    def task_signal_observed(self) -> bool:
        return any(
            event.event_type
            in {
                "level_advanced",
                "resource_increased",
                "status_pattern_changed",
                "status_palette_changed",
                "status_pattern_matched_static_pattern",
            }
            for event in self.events
        )


@dataclass(frozen=True)
class _Component:
    color: int
    pixels: tuple[Cell, ...]
    bbox: BBox

    @property
    def area(self) -> int:
        return len(self.pixels)

    @property
    def center(self) -> tuple[float, float]:
        return (
            sum(point[0] for point in self.pixels) / self.area,
            sum(point[1] for point in self.pixels) / self.area,
        )


def pattern_key(pattern: Pattern) -> str:
    if not pattern:
        return "empty"
    return "/".join("".join(str(value) for value in row) for row in pattern)


def _compress_pattern(mask: np.ndarray) -> Pattern:
    if not mask.size or not bool(mask.any()):
        return ()
    ys, xs = np.nonzero(mask)
    cropped = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = cropped.shape
    common = gcd(height, width)
    for scale in range(common, 1, -1):
        if height % scale or width % scale:
            continue
        blocks = cropped.reshape(height // scale, scale, width // scale, scale)
        minimum = blocks.min(axis=(1, 3))
        maximum = blocks.max(axis=(1, 3))
        if np.array_equal(minimum, maximum):
            cropped = maximum
            break
    return tuple(tuple(int(value) for value in row) for row in cropped)


def _grid(frame: Any) -> np.ndarray:
    value = getattr(frame, "frame", frame)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not value:
        return np.empty((0, 0), dtype=np.int16)
    array = np.asarray(value)
    if array.ndim == 3:
        array = array[-1]
    if array.ndim != 2:
        return np.empty((0, 0), dtype=np.int16)
    return np.ascontiguousarray(array)


def _components(
    grid: np.ndarray, colors: Iterable[int] | None = None
) -> list[_Component]:
    if not grid.size:
        return []
    allowed = None if colors is None else {int(value) for value in colors}
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    result: list[_Component] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x]:
                continue
            color = int(grid[y, x])
            if allowed is not None and color not in allowed:
                seen[y, x] = True
                continue
            queue = [(x, y)]
            seen[y, x] = True
            pixels: list[Cell] = []
            for cx, cy in queue:
                pixels.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and not seen[ny, nx]
                        and int(grid[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        queue.append((nx, ny))
            xs = [point[0] for point in pixels]
            ys = [point[1] for point in pixels]
            result.append(
                _Component(
                    color=color,
                    pixels=tuple(pixels),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                )
            )
    result.sort(key=lambda item: (-item.area, item.color, item.bbox))
    return result


def _shape_signature(mask: np.ndarray) -> str:
    pattern = _compress_pattern(mask)
    return f"mask={pattern_key(pattern)}:area={int(mask.sum())}:size={mask.shape[1]}x{mask.shape[0]}"


class MechanismStateTracker:
    """Incrementally infer stable mechanism signals from public observations."""

    def __init__(self) -> None:
        self.layout: GridLayout | None = None
        self.previous_grid: np.ndarray | None = None
        self.previous_state: MechanismState | None = None
        self.resource_capacity = 0
        self.resource_right_endpoint: int | None = None
        self.resource_row_band: tuple[int, int] | None = None
        self.player_position: Cell | None = None
        self.player_confidence = 0.0
        self.player_bbox: BBox | None = None
        self.player_appearance: tuple[int, ...] | None = None
        self._level: int | None = None

    def reset(self) -> None:
        self.layout = None
        self.previous_grid = None
        self.previous_state = None
        self.resource_capacity = 0
        self.resource_right_endpoint = None
        self.resource_row_band = None
        self.player_position = None
        self.player_confidence = 0.0
        self.player_bbox = None
        self.player_appearance = None
        self._level = None

    def _reset_level_local(self) -> None:
        """Drop geometry-dependent state while retaining the prior transition."""
        self.layout = None
        self.previous_grid = None
        self.resource_capacity = 0
        self.resource_right_endpoint = None
        self.resource_row_band = None
        self.player_position = None
        self.player_confidence = 0.0
        self.player_bbox = None
        self.player_appearance = None

    @staticmethod
    def _player_components(
        objects: Iterable[MechanismObject],
        layout: GridLayout | None,
        player: Cell | None,
        player_bbox: BBox | None = None,
    ) -> tuple[str, ...]:
        """Identify color components belonging to the composite player sprite.

        The visual player can span two adjacent logical cells.  Treating its
        individual color components as independent command-coupled objects
        creates a fictitious multi-object mechanism.  Once player identity is
        resolved, use the observed full-cell-width stacked bands as a compact,
        palette-neutral signature of that composite sprite.
        """
        if layout is None or player is None:
            return ()
        if player_bbox is not None:
            px1, py1, px2, py2 = player_bbox
            return tuple(
                sorted(
                    item.object_id
                    for item in objects
                    if px1 <= item.bbox[0]
                    and py1 <= item.bbox[1]
                    and item.bbox[2] <= px2
                    and item.bbox[3] <= py2
                )
            )
        cell_width, cell_height = layout.cell_size
        candidates: list[MechanismObject] = []
        for item in objects:
            x1, y1, x2, y2 = item.bbox
            width = x2 - x1 + 1
            height = y2 - y1 + 1
            close = abs(item.cell[0] - player[0]) <= 1 and abs(
                item.cell[1] - player[1]
            ) <= 1
            if (
                close
                and width >= cell_width - 1
                and 1 <= height < cell_height
                and item.area >= cell_width
                and not item.static_target
            ):
                candidates.append(item)
        for first_index, first in enumerate(candidates):
            fx1, fy1, fx2, fy2 = first.bbox
            for second in candidates[first_index + 1 :]:
                sx1, sy1, sx2, sy2 = second.bbox
                aligned = abs(fx1 - sx1) <= 1 and abs(fx2 - sx2) <= 1
                touching = fy2 + 1 >= sy1 and sy2 + 1 >= fy1
                stacked_height = max(fy2, sy2) - min(fy1, sy1) + 1
                covers_cell = stacked_height >= cell_height - 1
                if aligned and touching and covers_cell:
                    return tuple(sorted((first.object_id, second.object_id)))
        return ()

    @staticmethod
    def _normalized_appearance(crop: np.ndarray) -> tuple[int, ...]:
        mapping: dict[int, int] = {}
        normalized: list[int] = []
        for value in crop.reshape(-1):
            token = int(value)
            if token not in mapping:
                mapping[token] = len(mapping)
            normalized.append(mapping[token])
        return tuple(normalized)

    @staticmethod
    def _bbox_to_player_cell(bbox: BBox, layout: GridLayout) -> Cell:
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        origin_x, origin_y = layout.origin
        cell_width, cell_height = layout.cell_size
        return (
            int(round((center_x - origin_x) / cell_width)),
            int(round((center_y - origin_y) / cell_height)),
        )

    @staticmethod
    def _composite_player_crops(
        grid: np.ndarray,
        layout: GridLayout,
    ) -> list[tuple[BBox, tuple[int, ...]]]:
        """Return full-cell, multi-palette sprite candidates.

        A player candidate occupies one complete action-lattice cell with no
        structural pixels.  The crop may contain any palette values; identity
        is based on its normalized arrangement and commanded translation.
        """
        cell_width, cell_height = layout.cell_size
        structural = {layout.floor_color, layout.background_color}
        x1, y1, x2, y2 = layout.bounds
        candidates: list[tuple[BBox, tuple[int, ...]]] = []
        for top in range(max(0, y1 - cell_height), min(grid.shape[0], y2 + 2)):
            bottom = top + cell_height
            if bottom > grid.shape[0]:
                break
            for left in range(max(0, x1 - cell_width), min(grid.shape[1], x2 + 2)):
                right = left + cell_width
                if right > grid.shape[1]:
                    break
                crop = grid[top:bottom, left:right]
                values = {int(value) for value in crop.reshape(-1)}
                if len(values) < 2 or values.intersection(structural):
                    continue
                bbox = (left, top, right - 1, bottom - 1)
                candidates.append(
                    (bbox, MechanismStateTracker._normalized_appearance(crop))
                )
        return candidates

    def _resolve_composite_player(
        self,
        grid: np.ndarray,
        layout: GridLayout,
        action: str | None,
    ) -> tuple[Cell, float] | None:
        if action not in ACTION_DELTAS:
            return None
        cell_width, cell_height = layout.cell_size
        dx, dy = ACTION_DELTAS[action]

        if self.player_bbox is None or self.player_appearance is None:
            if self.previous_grid is None or self.previous_grid.shape != grid.shape:
                return None
            matches: list[tuple[BBox, tuple[int, ...]]] = []
            for source_bbox, appearance in self._composite_player_crops(
                self.previous_grid, layout
            ):
                destination_bbox = (
                    source_bbox[0] + dx * cell_width,
                    source_bbox[1] + dy * cell_height,
                    source_bbox[2] + dx * cell_width,
                    source_bbox[3] + dy * cell_height,
                )
                x1, y1, x2, y2 = destination_bbox
                if not (
                    0 <= x1 <= x2 < grid.shape[1]
                    and 0 <= y1 <= y2 < grid.shape[0]
                ):
                    continue
                destination = grid[y1 : y2 + 1, x1 : x2 + 1]
                if self._normalized_appearance(destination) != appearance:
                    continue
                sx1, sy1, sx2, sy2 = source_bbox
                source_after = grid[sy1 : sy2 + 1, sx1 : sx2 + 1]
                source_before = self.previous_grid[sy1 : sy2 + 1, sx1 : sx2 + 1]
                if float(np.mean(source_before != source_after)) < 0.50:
                    continue
                matches.append((destination_bbox, appearance))
            if len(matches) != 1:
                return None
            self.player_bbox, self.player_appearance = matches[0]
            return self._bbox_to_player_cell(self.player_bbox, layout), 0.99

        current_candidates = self._composite_player_crops(grid, layout)
        matching = [
            bbox
            for bbox, appearance in current_candidates
            if appearance == self.player_appearance
        ]
        expected = (
            self.player_bbox[0] + dx * cell_width,
            self.player_bbox[1] + dy * cell_height,
            self.player_bbox[2] + dx * cell_width,
            self.player_bbox[3] + dy * cell_height,
        )
        selected: BBox | None = None
        if expected in matching:
            selected = expected
        elif self.player_bbox in matching:
            selected = self.player_bbox
        elif len(matching) == 1:
            selected = matching[0]
        if selected is None:
            return None
        self.player_bbox = selected
        return self._bbox_to_player_cell(selected, layout), 0.99

    @staticmethod
    def _standard_hud_present(grid: np.ndarray, background: int) -> bool:
        if grid.shape[0] < 61 or grid.shape[1] < 12:
            return False
        crop = grid[53:61, 1:11]
        if not crop.size:
            return False
        values, counts = np.unique(crop, return_counts=True)
        non_background = sum(
            int(count)
            for value, count in zip(values, counts)
            if int(value) != background
        )
        return non_background >= int(crop.size * 0.65)

    def _infer_layout(self, grid: np.ndarray) -> GridLayout | None:
        if not grid.size:
            return None
        boundary = np.concatenate(
            (grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1])
        )
        boundary_values, boundary_counts = np.unique(boundary, return_counts=True)
        background = int(boundary_values[int(np.argmax(boundary_counts))])
        candidates = [
            component
            for component in _components(grid)
            if component.color != background and component.area >= 20
        ]
        if not candidates:
            return None
        height, width = grid.shape

        def score(component: _Component) -> float:
            x1, y1, x2, y2 = component.bbox
            box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            centrality = 1.0 - abs(component.center[0] - (width - 1) / 2) / max(
                1, width
            )
            upper = 1.0 if y2 < int(height * 0.9) else 0.2
            return (
                component.area
                * (0.5 + component.area / box_area)
                * (0.5 + centrality)
                * upper
            )

        floor = max(candidates, key=score)
        standard_hud = self._standard_hud_present(grid, background)
        cell = 5 if standard_hud else 2
        origin_x, origin_y = floor.bbox[0], floor.bbox[1]
        active: list[Cell] = []
        max_cols = (width - origin_x) // cell
        max_rows = (height - origin_y) // cell
        for row in range(max_rows):
            for col in range(max_cols):
                tile = grid[
                    origin_y + row * cell : origin_y + (row + 1) * cell,
                    origin_x + col * cell : origin_x + (col + 1) * cell,
                ]
                if (
                    tile.shape == (cell, cell)
                    and float(np.mean(tile == floor.color)) >= 0.30
                ):
                    active.append((col, row))
        if not active:
            return None
        cols = max(point[0] for point in active) + 1
        rows = max(point[1] for point in active) + 1
        confidence = 0.95 if standard_hud and cols >= 2 and rows >= 2 else 0.55
        return GridLayout(
            origin=(origin_x, origin_y),
            cell_size=(cell, cell),
            cols=cols,
            rows=rows,
            floor_color=floor.color,
            background_color=background,
            confidence=confidence,
            projected_view=not standard_hud,
        )

    @staticmethod
    def _tile_cells(
        grid: np.ndarray, layout: GridLayout
    ) -> tuple[frozenset[Cell], frozenset[Cell]]:
        ox, oy = layout.origin
        cell_w, cell_h = layout.cell_size
        walkable: set[Cell] = set()
        blocked: set[Cell] = set()
        for row in range(layout.rows):
            for col in range(layout.cols):
                tile = grid[
                    oy + row * cell_h : oy + (row + 1) * cell_h,
                    ox + col * cell_w : ox + (col + 1) * cell_w,
                ]
                if tile.shape != (cell_h, cell_w):
                    continue
                point = (col, row)
                if float(np.mean(tile == layout.floor_color)) >= 0.30:
                    walkable.add(point)
                else:
                    blocked.add(point)
        return frozenset(walkable), frozenset(blocked)

    @staticmethod
    def _status_pattern(grid: np.ndarray, layout: GridLayout) -> Pattern:
        if layout.projected_view or grid.shape[0] < 61 or grid.shape[1] < 12:
            return ()
        crop = grid[53:61, 1:11]
        values, counts = np.unique(crop, return_counts=True)
        base = int(values[int(np.argmax(counts))])
        visible = (crop != base) & (crop != layout.background_color)
        if not bool(visible.any()):
            return ()
        foreground_values, foreground_counts = np.unique(
            crop[visible], return_counts=True
        )
        foreground = int(foreground_values[int(np.argmax(foreground_counts))])
        return _compress_pattern(crop == foreground)

    @staticmethod
    def _status_color(grid: np.ndarray, layout: GridLayout) -> int | None:
        if layout.projected_view or grid.shape[0] < 61 or grid.shape[1] < 12:
            return None
        crop = grid[53:61, 1:11]
        values, counts = np.unique(crop, return_counts=True)
        base = int(values[int(np.argmax(counts))])
        visible = (crop != base) & (crop != layout.background_color)
        if not bool(visible.any()):
            return None
        foreground_values, foreground_counts = np.unique(
            crop[visible], return_counts=True
        )
        return int(foreground_values[int(np.argmax(foreground_counts))])

    def _resource_level(self, grid: np.ndarray, layout: GridLayout) -> int:
        candidates: list[_Component] = []
        for component in _components(grid):
            x1, y1, x2, y2 = component.bbox
            width = x2 - x1 + 1
            height = y2 - y1 + 1
            if y1 <= layout.bounds[3] or height > 3:
                continue
            if self.resource_right_endpoint is not None:
                if (
                    x2 == self.resource_right_endpoint
                    and (y1, y2) == self.resource_row_band
                ):
                    return component.area
                continue
            if width >= 4 and width >= 2 * height:
                candidates.append(component)
        if self.resource_right_endpoint is not None:
            return 0
        if not candidates:
            return 0
        selected = max(
            candidates, key=lambda item: (item.bbox[2], item.bbox[0], item.area)
        )
        self.resource_right_endpoint = selected.bbox[2]
        self.resource_row_band = (selected.bbox[1], selected.bbox[3])
        return selected.area

    @staticmethod
    def _objects(grid: np.ndarray, layout: GridLayout) -> tuple[MechanismObject, ...]:
        result: list[MechanismObject] = []
        ox, oy = layout.origin
        cell_w, cell_h = layout.cell_size
        structural = {layout.floor_color, layout.background_color}
        x1, y1, x2, y2 = layout.bounds
        for component in _components(grid):
            if component.color in structural:
                continue
            bx1, by1, bx2, by2 = component.bbox
            width = bx2 - bx1 + 1
            height = by2 - by1 + 1
            cx, cy = component.center
            if not (
                x1 - cell_w <= cx <= x2 + cell_w and y1 - cell_h <= cy <= y2 + cell_h
            ):
                continue
            crop = grid[by1 : by2 + 1, bx1 : bx2 + 1]
            other = ~np.isin(crop, list(structural | {component.color}))
            internal = _compress_pattern(other)
            pattern_color = None
            if bool(other.any()):
                internal_values, internal_counts = np.unique(
                    crop[other], return_counts=True
                )
                pattern_color = int(internal_values[int(np.argmax(internal_counts))])
            floor_ratio = float(np.mean(crop == layout.floor_color))
            # Compact objects and the composite player are centred on action-
            # lattice intersections, not inside floor-tile interiors.  A
            # shared rounded coordinate system makes physical overlap equal
            # logical contact instead of leaving a one-cell attribution gap.
            cell = (
                max(0, min(layout.cols - 1, int(round((cx - ox) / cell_w)))),
                max(0, min(layout.rows - 1, int(round((cy - oy) / cell_h)))),
            )
            mask = np.zeros((height, width), dtype=bool)
            for px, py in component.pixels:
                mask[py - by1, px - bx1] = True
            static_target = (
                bool(internal)
                and width >= cell_w + 2
                and height >= cell_h + 2
                and floor_ratio < 0.30
            )
            result.append(
                MechanismObject(
                    object_id=f"r{cell[1]}c{cell[0]}:{_shape_signature(mask)}",
                    color=component.color,
                    cell=cell,
                    bbox=component.bbox,
                    area=component.area,
                    shape_signature=_shape_signature(mask),
                    internal_pattern=internal,
                    multi_palette=bool(internal),
                    floor_ratio=floor_ratio,
                    static_target=static_target,
                    pattern_color=pattern_color,
                )
            )
        return tuple(sorted(result, key=lambda item: (item.cell, item.shape_signature)))

    def _resolve_player(
        self,
        grid: np.ndarray,
        layout: GridLayout,
        action: str | None,
    ) -> tuple[Cell | None, float]:
        composite = self._resolve_composite_player(grid, layout, action)
        if composite is not None:
            return composite
        if self.previous_grid is None or action not in ACTION_DELTAS:
            return self.player_position, self.player_confidence
        if self.previous_grid.shape != grid.shape:
            return self.player_position, self.player_confidence
        dx, dy = ACTION_DELTAS[action]
        ox, oy = layout.origin
        cell_w, cell_h = layout.cell_size
        if self.player_position is not None:
            px, py = self.player_position
            template = self.previous_grid[
                oy + py * cell_h : oy + (py + 1) * cell_h,
                ox + px * cell_w : ox + (px + 1) * cell_w,
            ]
            if template.shape == (cell_h, cell_w) and len(np.unique(template)) >= 2:
                source_after = grid[
                    oy + py * cell_h : oy + (py + 1) * cell_h,
                    ox + px * cell_w : ox + (px + 1) * cell_w,
                ]
                global_matches: list[tuple[float, Cell]] = []
                for row in range(layout.rows):
                    for col in range(layout.cols):
                        if (col, row) == self.player_position:
                            continue
                        tile = grid[
                            oy + row * cell_h : oy + (row + 1) * cell_h,
                            ox + col * cell_w : ox + (col + 1) * cell_w,
                        ]
                        if tile.shape != template.shape:
                            continue
                        similarity = float(np.mean(template == tile))
                        if similarity >= 0.98:
                            global_matches.append((similarity, (col, row)))
                source_changed = float(np.mean(template != source_after)) >= 0.35
                if source_changed and global_matches:
                    best_similarity = max(item[0] for item in global_matches)
                    best = [
                        item for item in global_matches if item[0] == best_similarity
                    ]
                    if len(best) == 1:
                        return best[0][1], min(
                            0.99,
                            0.82 + 0.15 * best_similarity,
                        )
        candidates: list[tuple[float, float, int, Cell]] = []
        for row in range(layout.rows):
            for col in range(layout.cols):
                nx, ny = col + dx, row + dy
                if not (0 <= nx < layout.cols and 0 <= ny < layout.rows):
                    continue
                before = self.previous_grid[
                    oy + row * cell_h : oy + (row + 1) * cell_h,
                    ox + col * cell_w : ox + (col + 1) * cell_w,
                ]
                destination = grid[
                    oy + ny * cell_h : oy + (ny + 1) * cell_h,
                    ox + nx * cell_w : ox + (nx + 1) * cell_w,
                ]
                source_after = grid[
                    oy + row * cell_h : oy + (row + 1) * cell_h,
                    ox + col * cell_w : ox + (col + 1) * cell_w,
                ]
                if before.shape != (cell_h, cell_w) or len(np.unique(before)) < 2:
                    continue
                similarity = float(np.mean(before == destination))
                changed = float(np.mean(before != source_after))
                if similarity >= 0.90 and changed >= 0.35:
                    candidates.append(
                        (similarity, changed, len(np.unique(before)), (nx, ny))
                    )
        if candidates:
            if self.player_position is not None:
                expected = (
                    self.player_position[0] + dx,
                    self.player_position[1] + dy,
                )
                aligned = [item for item in candidates if item[3] == expected]
                if aligned:
                    similarity, changed, _, position = max(
                        aligned,
                        key=lambda item: (item[0], item[1], -item[2]),
                    )
                    return position, min(0.99, 0.7 + 0.15 * similarity + 0.1 * changed)
                # Once identity is resolved, continuity is stronger evidence
                # than an unrelated translated object elsewhere in the grid.
                # Compact sprites may be occluded while entering an object, so
                # retain the commanded one-cell move when topology permits it.
                if expected in self._tile_cells(grid, layout)[0]:
                    return expected, max(0.55, self.player_confidence - 0.02)
                return self.player_position, max(0.55, self.player_confidence - 0.03)
            similarity, changed, _, position = max(
                candidates,
                key=lambda item: (item[0], item[1], -item[2]),
            )
            return position, min(0.99, 0.7 + 0.15 * similarity + 0.1 * changed)
        if self.player_position is not None:
            expected = (
                self.player_position[0] + dx,
                self.player_position[1] + dy,
            )
            if expected in self._tile_cells(grid, layout)[0]:
                return expected, max(0.55, self.player_confidence - 0.02)
        return self.player_position, self.player_confidence

    @staticmethod
    def _event(
        event_type: str,
        action: str | None,
        before: Any,
        after: Any,
        confidence: float = 1.0,
    ) -> MechanismEvent:
        return MechanismEvent(
            event_type=event_type,
            action=action,
            before=before,
            after=after,
            confidence=confidence,
        )

    @staticmethod
    def _object_groups(
        objects: Iterable[MechanismObject],
    ) -> dict[Cell, tuple[str, ...]]:
        groups: dict[Cell, list[str]] = {}
        for item in objects:
            signature = ":".join(
                (
                    str(item.color),
                    item.shape_signature,
                    item.pattern_signature,
                    str(item.pattern_color),
                    str(int(item.static_target)),
                )
            )
            groups.setdefault(item.cell, []).append(signature)
        return {cell: tuple(sorted(signatures)) for cell, signatures in groups.items()}

    def observe(self, frame: Any, *, action: str | None = None) -> MechanismState:
        grid = _grid(frame)
        level = int(getattr(frame, "levels_completed", 0) or 0)
        if self._level is not None and level < self._level:
            self.reset()
        elif self._level is not None and level != self._level:
            self._reset_level_local()
        previous = self.previous_state
        if not grid.size:
            state = MechanismState(
                level=level,
                game_state=str(getattr(frame, "state", "UNKNOWN")),
                layout=None,
                player_position=None,
                player_confidence=0.0,
                walkable_cells=frozenset(),
                blocked_cells=frozenset(),
                objects=(),
                resource_level=0,
                resource_capacity=self.resource_capacity,
                status_pattern=(),
                matching_target_ids=(),
                warnings=("empty_frame",),
            )
            self.previous_state = state
            self.previous_grid = grid
            self._level = level
            return state

        if self.layout is None:
            self.layout = self._infer_layout(grid)
        layout = self.layout
        warnings: list[str] = []
        if layout is None:
            warnings.append("layout_not_inferred")
            walkable = blocked = frozenset()
            objects: tuple[MechanismObject, ...] = ()
            status = ()
            status_color = None
            resource = 0
        else:
            if layout.confidence < 0.8:
                warnings.append("low_confidence_projected_layout")
            walkable, blocked = self._tile_cells(grid, layout)
            objects = self._objects(grid, layout)
            status = self._status_pattern(grid, layout)
            status_color = self._status_color(grid, layout)
            resource = self._resource_level(grid, layout)
            self.resource_capacity = max(self.resource_capacity, resource)
            self.player_position, self.player_confidence = self._resolve_player(
                grid, layout, action
            )

        target_ids = tuple(
            item.object_id
            for item in objects
            if item.static_target
            and item.internal_pattern == status
            and bool(status)
            and (
                item.pattern_color is None
                or status_color is None
                or item.pattern_color == status_color
            )
        )
        player_component_ids = self._player_components(
            objects,
            layout,
            self.player_position,
            self.player_bbox,
        )
        events: list[MechanismEvent] = []
        if previous is not None:
            if level > previous.level:
                events.append(
                    self._event("level_advanced", action, previous.level, level)
                )
            if resource > previous.resource_level:
                events.append(
                    self._event(
                        "resource_increased",
                        action,
                        previous.resource_level,
                        resource,
                        0.95,
                    )
                )
            elif 0 < resource < previous.resource_level:
                events.append(
                    self._event(
                        "resource_decreased",
                        action,
                        previous.resource_level,
                        resource,
                        0.95,
                    )
                )
            if status and previous.status_pattern and status != previous.status_pattern:
                events.append(
                    self._event(
                        "status_pattern_changed",
                        action,
                        pattern_key(previous.status_pattern),
                        pattern_key(status),
                        0.95,
                    )
                )
            if (
                status_color is not None
                and previous.status_color is not None
                and status_color != previous.status_color
            ):
                events.append(
                    self._event(
                        "status_palette_changed",
                        action,
                        previous.status_color,
                        status_color,
                        0.95,
                    )
                )
            if target_ids and not previous.matching_target_ids:
                events.append(
                    self._event(
                        "status_pattern_matched_static_pattern",
                        action,
                        (),
                        target_ids,
                        0.95,
                    )
                )
            if level == previous.level:
                before_groups = self._object_groups(previous.objects)
                after_groups = self._object_groups(objects)
                for cell in sorted(before_groups.keys() | after_groups.keys()):
                    before_group = before_groups.get(cell)
                    after_group = after_groups.get(cell)
                    if before_group == after_group:
                        continue
                    if before_group is not None:
                        events.append(
                            MechanismEvent(
                                event_type="object_disappeared",
                                action=action,
                                before=before_group,
                                after=after_group,
                                cells=(cell,),
                                confidence=0.85,
                            )
                        )
                    if after_group is not None:
                        events.append(
                            MechanismEvent(
                                event_type="object_appeared",
                                action=action,
                                before=before_group,
                                after=after_group,
                                cells=(cell,),
                                confidence=0.85,
                            )
                        )
            if (
                action in ACTION_DELTAS
                and previous.player_position is not None
                and self.player_position is not None
            ):
                dx, dy = ACTION_DELTAS[action]
                expected = (
                    previous.player_position[0] + dx,
                    previous.player_position[1] + dy,
                )
                if self.player_position not in {
                    previous.player_position,
                    expected,
                }:
                    events.append(
                        MechanismEvent(
                            event_type="player_relocated",
                            action=action,
                            before=previous.player_position,
                            after=self.player_position,
                            cells=(previous.player_position, self.player_position),
                            confidence=self.player_confidence,
                        )
                    )

        state_value = getattr(frame, "state", "UNKNOWN")
        game_state = str(
            getattr(state_value, "name", getattr(state_value, "value", state_value))
        )
        state = MechanismState(
            level=level,
            game_state=game_state,
            layout=layout,
            player_position=self.player_position,
            player_confidence=self.player_confidence,
            walkable_cells=walkable,
            blocked_cells=blocked,
            objects=objects,
            resource_level=resource,
            resource_capacity=self.resource_capacity,
            status_pattern=status,
            matching_target_ids=target_ids,
            events=tuple(events),
            warnings=tuple(warnings),
            status_color=status_color,
            player_component_ids=player_component_ids,
            player_bbox=self.player_bbox,
        )
        self.previous_state = state
        self.previous_grid = grid.copy()
        self._level = level
        return state

    @staticmethod
    def summary(state: MechanismState | None) -> str:
        if state is None:
            return "mechanism_state=unavailable"
        layout = state.layout
        layout_text = (
            "unknown"
            if layout is None
            else f"{layout.cols}x{layout.rows}@{layout.cell_size[0]}"
        )
        events = ",".join(event.event_type for event in state.events) or "none"
        return (
            f"mechanism_layout={layout_text} "
            f"player={state.player_position or 'unresolved'} "
            f"resource={state.resource_level}/{state.resource_capacity} "
            f"status={state.status_signature} "
            f"palette={state.status_color} "
            f"matches={len(state.matching_target_ids)} events={events}"
        )
