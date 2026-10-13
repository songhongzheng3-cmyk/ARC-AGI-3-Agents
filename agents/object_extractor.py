"""Deterministic object-centric representation for ARC grids."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

Point = tuple[int, int]


@dataclass(frozen=True)
class GridObject:
    """One same-colour 4-connected component in a settled grid."""

    object_id: str
    color: int
    pixels: tuple[Point, ...]
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    shape: str
    touches_boundary: bool

    @property
    def area(self) -> int:
        return len(self.pixels)


@dataclass(frozen=True)
class GridSnapshot:
    width: int
    height: int
    background: int | None
    objects: tuple[GridObject, ...]

    def by_id(self) -> dict[str, GridObject]:
        return {item.object_id: item for item in self.objects}


class ObjectExtractor:
    """Extract stable, compact objects from the final animation grid."""

    @staticmethod
    def _grid(frame_or_grid: Any) -> list[list[int]]:
        layers = getattr(frame_or_grid, "frame", frame_or_grid) or []
        if layers and isinstance(layers[0], (list, tuple)):
            # FrameData.frame is [animation_layer][row][column]. A plain grid
            # is accepted too for offline tests and replay tooling.
            first = layers[0]
            if first and isinstance(first[0], (list, tuple)):
                layers = layers[-1]
        return [list(map(int, row)) for row in layers if row]

    @staticmethod
    def _shape(pixels: Iterable[Point], width: int, height: int) -> str:
        points = list(pixels)
        min_x = min(x for x, _ in points)
        min_y = min(y for _, y in points)
        max_x = max(x for x, _ in points)
        max_y = max(y for _, y in points)
        normalized = {(x - min_x, y - min_y) for x, y in points}
        if len(normalized) == (max_x - min_x + 1) * (max_y - min_y + 1):
            if max_x == min_x or max_y == min_y:
                return "line"
            return "rectangle"
        if len(points) == 1:
            return "point"
        if all(x == min_x or x == max_x or y == min_y or y == max_y for x, y in points):
            return "outline"
        return "irregular"

    def extract(self, frame_or_grid: Any) -> GridSnapshot:
        grid = self._grid(frame_or_grid)
        if not grid:
            return GridSnapshot(0, 0, None, ())
        height = len(grid)
        width = max(len(row) for row in grid)
        if any(len(row) != width for row in grid):
            raise ValueError("ARC grid must be rectangular")

        counts: dict[int, int] = {}
        for row in grid:
            for value in row:
                counts[value] = counts.get(value, 0) + 1
        background = max(counts, key=counts.get)
        visited: set[Point] = set()
        objects: list[GridObject] = []
        for y in range(height):
            for x in range(width):
                if (x, y) in visited or grid[y][x] == background:
                    continue
                color = grid[y][x]
                queue = [(x, y)]
                visited.add((x, y))
                pixels: list[Point] = []
                while queue:
                    cx, cy = queue.pop()
                    pixels.append((cx, cy))
                    for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                        if 0 <= nx < width and 0 <= ny < height:
                            if (nx, ny) not in visited and grid[ny][nx] == color:
                                visited.add((nx, ny))
                                queue.append((nx, ny))
                pixels.sort(key=lambda point: (point[1], point[0]))
                min_x = min(px for px, _ in pixels)
                min_y = min(py for _, py in pixels)
                max_x = max(px for px, _ in pixels)
                max_y = max(py for _, py in pixels)
                object_id = f"c{color}:{min_x},{min_y}:{max_x},{max_y}"
                objects.append(
                    GridObject(
                        object_id=object_id,
                        color=color,
                        pixels=tuple(pixels),
                        bbox=(min_x, min_y, max_x, max_y),
                        centroid=(
                            sum(px for px, _ in pixels) / len(pixels),
                            sum(py for _, py in pixels) / len(pixels),
                        ),
                        shape=self._shape(pixels, width, height),
                        touches_boundary=any(
                            px in (0, width - 1) or py in (0, height - 1)
                            for px, py in pixels
                        ),
                    )
                )
        objects.sort(key=lambda item: (item.color, item.bbox[1], item.bbox[0]))
        return GridSnapshot(width, height, background, tuple(objects))
