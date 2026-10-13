"""Observation binding for a tested family of selectable tiled sprites.

Palette and geometry are level-local hypotheses discovered from a public
initial image. This does not assign universal semantics to action IDs or
claim a goal from an outline alone.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Any

from .object_extractor import ObjectExtractor

Cell = tuple[int, int]


@dataclass(frozen=True)
class TileActor:
    identity: str
    cell: Cell
    marker: int
    pattern: tuple[int, ...]


@dataclass(frozen=True)
class TileScene:
    actors: tuple[TileActor, ...]
    active_identity: str | None
    goal_candidates: tuple[Cell, ...]
    floor_cells: frozenset[Cell]
    uncertain_cells: frozenset[Cell]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "actors": [asdict(actor) for actor in self.actors],
            "active_identity": self.active_identity,
            "goal_candidates": self.goal_candidates,
            "floor_cells": sorted(self.floor_cells),
            "uncertain_cells": sorted(self.uncertain_cells),
            "warnings": self.warnings,
        }


class TileSceneBinding:
    """Match full-cell sprite patterns and preserve multi-object identities.

    The initial sprite ring must cover all but its centre. Later shapes may
    change: a majority of perimeter pixels is sufficient for a candidate,
    while new centre markers remain observed state. Ambiguous assignments
    stop identity-dependent planning.
    """

    def __init__(self, initial: Any, layout: dict[str, Any]) -> None:
        self.layout = layout
        self.origin = tuple(layout["origin"])
        self.cell_size = tuple(layout["cell_size"])
        self.floor_color = layout["floor_color"]
        self.background_color = layout["background_color"]
        if self.cell_size != (3, 3):
            raise ValueError("binding currently validates only 3x3 sprite cells")
        snapshot = ObjectExtractor().extract(initial)
        rings = [
            obj
            for obj in snapshot.objects
            if obj.area == 8
            and obj.bbox[2] - obj.bbox[0] == 2
            and obj.bbox[3] - obj.bbox[1] == 2
            and self.contains_pixel(tuple(map(round, obj.centroid)))
        ]
        counts = Counter(obj.color for obj in rings)
        if not counts or counts.most_common(1)[0][1] < 2:
            raise ValueError("insufficient repeated ring-sprite evidence")
        self.actor_color = counts.most_common(1)[0][0]
        self.actor_count = sum(obj.color == self.actor_color for obj in rings)
        grid = initial.frame[-1]
        self.markers = {
            grid[round(obj.centroid[1])][round(obj.centroid[0])]
            for obj in rings
            if obj.color == self.actor_color
        } | {self.actor_color}
        # Candidate destinations remain hypotheses until contact/completion.
        self.goals = tuple(
            sorted(
                self.pixel_cell(tuple(map(round, obj.centroid)))
                for obj in snapshot.objects
                if obj.area == 16
                and obj.bbox[2] - obj.bbox[0] == 4
                and obj.bbox[3] - obj.bbox[1] == 4
                and self.contains_pixel(tuple(map(round, obj.centroid)))
            )
        )
        self.previous: TileScene | None = None
        self.active_marker: int | None = None

    def contains_pixel(self, point: Cell) -> bool:
        x, y = point
        ox, oy = self.origin
        sx, sy = self.cell_size
        return (
            ox <= x < ox + sx * self.layout["cols"]
            and oy <= y < oy + sy * self.layout["rows"]
        )

    def pixel_cell(self, point: Cell) -> Cell:
        return ((point[0] - self.origin[0]) // 3, (point[1] - self.origin[1]) // 3)

    def cell_pixel(self, cell: Cell) -> Cell:
        return (self.origin[0] + 3 * cell[0] + 1, self.origin[1] + 3 * cell[1] + 1)

    def parse(self, frame: Any) -> TileScene:
        grid = frame.frame[-1]
        found = []
        floor, uncertain = set(), set()
        for row in range(self.layout["rows"]):
            for column in range(self.layout["cols"]):
                cx, cy = self.cell_pixel((column, row))
                patch = tuple(
                    grid[y][x]
                    for y in range(cy - 1, cy + 2)
                    for x in range(cx - 1, cx + 2)
                )
                perimeter = patch[:4] + patch[5:]
                if sum(p == self.actor_color for p in perimeter) >= 5:
                    found.append(((column, row), patch[4], patch))
                if patch[4] == self.floor_color:
                    floor.add((column, row))
                elif patch[4] != self.background_color:
                    uncertain.add((column, row))
        warnings = []
        actors = []
        if len(found) != self.actor_count:
            warnings.append(f"actor_count_changed:{len(found)}")
        if self.previous is None:
            actors = [
                TileActor(f"actor-{i}", cell, marker, pattern)
                for i, (cell, marker, pattern) in enumerate(sorted(found))
            ]
        elif len(found) == len(self.previous.actors) == self.actor_count:
            assignments = []
            for assignment in permutations(found):
                cost = sum(
                    abs(old.cell[0] - new[0][0]) + abs(old.cell[1] - new[0][1])
                    for old, new in zip(self.previous.actors, assignment)
                )
                assignments.append((cost, assignment))
            assignments.sort(key=lambda pair: pair[0])
            if len(assignments) > 1 and assignments[0][0] == assignments[1][0]:
                warnings.append("ambiguous_identity_assignment")
            else:
                actors = [
                    TileActor(old.identity, cell, marker, pattern)
                    for old, (cell, marker, pattern) in zip(
                        self.previous.actors, assignments[0][1]
                    )
                ]
        else:
            warnings.append("identity_not_recoverable")
        active = (
            [actor.identity for actor in actors if actor.marker == self.active_marker]
            if self.active_marker is not None
            else []
        )
        for actor in actors:
            floor.discard(actor.cell)
            uncertain.discard(actor.cell)
        scene = TileScene(
            tuple(actors),
            active[0] if len(active) == 1 else None,
            self.goals,
            frozenset(floor),
            frozenset(uncertain),
            tuple(warnings),
        )
        if not warnings:
            self.previous = scene
        return scene

    def calibrate_active_marker(self, before: TileScene, after: TileScene) -> bool:
        """Bind selection indication only after exactly one actor moves."""
        old = {actor.identity: actor for actor in before.actors}
        moved = [
            actor
            for actor in after.actors
            if actor.identity in old and actor.cell != old[actor.identity].cell
        ]
        if before.warnings or after.warnings or len(moved) != 1:
            return False
        marker = moved[0].marker
        if sum(actor.marker == marker for actor in after.actors) != 1:
            return False
        self.active_marker = marker
        return True
