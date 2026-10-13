"""Experimental, observation-bound rigid shapes and multi-object contacts.

Unlike colour connected components, sprite templates keep touching actors
separate. Contact effects are evidence-scoped, not an assumed physics engine.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from functools import reduce
from math import gcd
from typing import Any

from .object_extractor import ObjectExtractor
from .selectable_navigation_model import MultiActorState

Cell = tuple[int, int]


@dataclass(frozen=True)
class Shape:
    width: int
    height: int
    marker_width: int
    marker_height: int


class ShapedSceneBinding:
    """Bind unique rectangular sprites and matching padded outlines locally.

    Palette and rectangular-template assumptions are explicitly limited to
    this development experiment; unsupported or ambiguous frames fail closed.
    """

    def __init__(self, initial: Any):
        grid = initial.frame[-1]
        self.sprite_color, self.marker_color, self.active_color = 14, 5, 0
        self.goal_color = 4
        components = ObjectExtractor().extract(initial).objects
        sprites = [
            o for o in components if o.color == self.sprite_color and o.bbox[1] < 63
        ]
        sizes = [
            (o.bbox[2] - o.bbox[0] + 1, o.bbox[3] - o.bbox[1] + 1) for o in sprites
        ]
        if not sizes or len(set(sizes)) != len(sizes):
            raise ValueError("shaped binding requires unique initial sprite sizes")
        self.pitch = reduce(gcd, [v for size in sizes for v in size])
        if self.pitch < 3:
            raise ValueError("unsupported sprite geometry")
        self.origin = (sprites[0].bbox[0] % self.pitch, sprites[0].bbox[1] % self.pitch)
        self.shapes = []
        for w, h in sorted(sizes):
            self.shapes.append(
                Shape(w, h, max(1, w // self.pitch), max(1, h // self.pitch))
            )
        self.shapes = tuple(self.shapes)
        self.goals = []
        outlines = [
            o for o in components if o.color == self.goal_color and o.bbox[3] < 63
        ]
        for shape in self.shapes:
            matches = [
                o
                for o in outlines
                if o.bbox[2] - o.bbox[0] + 1 == shape.width + 2
                and o.bbox[3] - o.bbox[1] + 1 == shape.height + 2
            ]
            if len(matches) != 1:
                raise ValueError("outline-to-shape assignment is not unique")
            x, y, _, _ = matches[0].bbox
            self.goals.append(self.cell((x + 1, y + 1)))
        self.goals = tuple(self.goals)
        if len(outlines) != len(self.shapes):
            raise ValueError("additional unmatched outlines require a new scene model")
        self.cols = (len(grid[0]) - self.origin[0]) // self.pitch
        # Exclude the observed bottom HUD row and partial tiles.
        self.rows = (len(grid) - 1 - self.origin[1]) // self.pitch
        self.floor_color = Counter(
            grid[y][x]
            for y in range(self.rows * self.pitch)
            for x in range(self.cols * self.pitch)
            if grid[y][x]
            not in (
                self.sprite_color,
                self.marker_color,
                self.active_color,
                self.goal_color,
            )
        ).most_common(1)[0][0]
        self.terrain = {}
        for y in range(self.rows):
            for x in range(self.cols):
                px, py = self.pixel((x, y))
                color = grid[py + self.pitch // 2][px + self.pitch // 2]
                self.terrain[(x, y)] = (
                    self.floor_color
                    if color
                    in (
                        self.sprite_color,
                        self.marker_color,
                        self.active_color,
                        self.goal_color,
                    )
                    else color
                )
        self.initial = self.parse(initial)

    def pixel(self, cell: Cell) -> Cell:
        return (
            self.origin[0] + cell[0] * self.pitch,
            self.origin[1] + cell[1] * self.pitch,
        )

    def cell(self, pixel: Cell) -> Cell:
        dx, dy = pixel[0] - self.origin[0], pixel[1] - self.origin[1]
        if dx % self.pitch or dy % self.pitch:
            raise ValueError("sprite/outline is not on the inferred lattice")
        return dx // self.pitch, dy // self.pitch

    def footprint(self, index: int, cell: Cell) -> frozenset[Cell]:
        s = self.shapes[index]
        return frozenset(
            (cell[0] + dx, cell[1] + dy)
            for dx in range(s.width // self.pitch)
            for dy in range(s.height // self.pitch)
        )

    def parse(self, frame: Any) -> MultiActorState:
        grid = frame.frame[-1]
        positions, selected = [], []
        for index, shape in enumerate(self.shapes):
            candidates = []
            mx = (shape.width - shape.marker_width) // 2
            my = (shape.height - shape.marker_height) // 2
            for cy in range(self.rows - shape.height // self.pitch + 1):
                for cx in range(self.cols - shape.width // self.pitch + 1):
                    x, y = self.pixel((cx, cy))
                    marker = grid[y + my][x + mx]
                    # The single-tile passive marker becomes yellow after a
                    # selection change; multi-tile markers remain grey in the
                    # public calibration traces. Preserve either observation.
                    if marker not in (
                        self.marker_color,
                        self.goal_color,
                        self.active_color,
                    ):
                        continue
                    valid = True
                    green = 0
                    for dy in range(shape.height):
                        for dx in range(shape.width):
                            value = grid[y + dy][x + dx]
                            centre = (
                                mx <= dx < mx + shape.marker_width
                                and my <= dy < my + shape.marker_height
                            )
                            border = dx in (0, shape.width - 1) or dy in (
                                0,
                                shape.height - 1,
                            )
                            if centre:
                                valid &= value == marker
                            else:
                                valid &= value == self.sprite_color or (
                                    marker == self.active_color
                                    and border
                                    and value == self.active_color
                                )
                                green += value == self.sprite_color
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid and green:
                        candidates.append(((cx, cy), marker == self.active_color))
            if len(candidates) != 1:
                raise ValueError(f"ambiguous/missing sprite {index}: {candidates}")
            position, active = candidates[0]
            positions.append(position)
            if active:
                selected.append(index)
        if len(selected) != 1:
            raise ValueError(f"selection is ambiguous: {selected}")
        covered = set()
        for i, p in enumerate(positions):
            cells = self.footprint(i, p)
            if covered & cells:
                raise ValueError("sprite templates overlap")
            covered.update(cells)
        return MultiActorState(tuple(positions), selected[0])


@dataclass(frozen=True)
class ContactEffect:
    index: int
    relative_position: Cell
    delta: Cell
    terrain_signature: tuple[int, ...]


@dataclass(frozen=True)
class ShapedContactRule:
    action: int
    selected: int
    effects: tuple[ContactEffect, ...]


class ShapedContactModel:
    def __init__(self, binding: ShapedSceneBinding, directions: dict[int, Cell]):
        self.binding = binding
        self.directions = dict(directions)
        self.rules: list[ShapedContactRule] = []
        self.support = Counter()

    def terrain_signature(
        self, index: int, start: Cell, delta: Cell
    ) -> tuple[int, ...]:
        distance = abs(delta[0]) + abs(delta[1])
        if not distance or (delta[0] and delta[1]):
            raise ValueError("non-axial contact displacement")
        direction = (delta[0] // distance, delta[1] // distance)
        values = []
        for n in range(distance + 1):
            pos = (start[0] + n * direction[0], start[1] + n * direction[1])
            values.extend(
                self.binding.terrain.get(c, -1)
                for c in sorted(self.binding.footprint(index, pos))
            )
        return tuple(values)

    def ordinary_destination(
        self, state: MultiActorState, action: int
    ) -> MultiActorState | None:
        dx, dy = self.directions[action]
        index = state.selected
        x, y = state.positions[index]
        position = (x + dx, y + dy)
        cells = self.binding.footprint(index, position)
        if any(self.binding.terrain.get(c) != self.binding.floor_color for c in cells):
            return None
        if any(
            cells & self.binding.footprint(i, p)
            for i, p in enumerate(state.positions)
            if i != index
        ):
            return None
        positions = list(state.positions)
        positions[index] = position
        return MultiActorState(tuple(positions), index)

    def learn(
        self, before: MultiActorState, action: int, after: MultiActorState
    ) -> None:
        if action == 6:
            if before.positions != after.positions:
                raise ValueError("click changed object positions")
            return
        if action not in self.directions or before.selected != after.selected:
            raise ValueError("unsupported action or selection change")
        moved = [
            i
            for i, (a, b) in enumerate(zip(before.positions, after.positions))
            if a != b
        ]
        if not moved:
            if self.ordinary_destination(before, action) is not None:
                raise ValueError("observed no-op contradicts ordinary movement")
            return
        if moved == [before.selected]:
            if self.ordinary_destination(before, action) != after:
                raise ValueError("ordinary translation contradicted")
            return
        if before.selected in moved:
            raise ValueError("active and passive simultaneous movement is unsupported")
        sx, sy = before.positions[before.selected]
        effects = []
        for i in moved:
            x, y = before.positions[i]
            ax, ay = after.positions[i]
            delta = (ax - x, ay - y)
            direction = self.directions[action]
            if delta != (
                direction[0] * (abs(delta[0]) + abs(delta[1])),
                direction[1] * (abs(delta[0]) + abs(delta[1])),
            ):
                raise ValueError("contact displacement is not along input direction")
            effects.append(
                ContactEffect(
                    i, (x - sx, y - sy), delta, self.terrain_signature(i, (x, y), delta)
                )
            )
        rule = ShapedContactRule(action, before.selected, tuple(effects))
        if rule not in self.rules:
            self.rules.append(rule)
        self.support[rule] += 1

    def contact_destination(
        self, state: MultiActorState, action: int
    ) -> MultiActorState | None:
        predictions = set()
        sx, sy = state.positions[state.selected]
        dx, dy = self.directions[action]
        active_next = self.binding.footprint(state.selected, (sx + dx, sy + dy))
        for rule in self.rules:
            if (rule.action, rule.selected) != (action, state.selected):
                continue
            positions = list(state.positions)
            valid = any(
                active_next & self.binding.footprint(e.index, positions[e.index])
                for e in rule.effects
            )
            affected = {e.index for e in rule.effects}
            for effect in rule.effects:
                p = positions[effect.index]
                if (
                    p[0] - sx,
                    p[1] - sy,
                ) != effect.relative_position or self.terrain_signature(
                    effect.index, p, effect.delta
                ) != effect.terrain_signature:
                    valid = False
                    break
                # Unobserved actors on the swept path could produce another
                # chain. Reject rather than silently using the recorded effect.
                distance = abs(effect.delta[0]) + abs(effect.delta[1])
                for n in range(1, distance + 1):
                    swept = self.binding.footprint(
                        effect.index, (p[0] + n * dx, p[1] + n * dy)
                    )
                    if any(
                        swept & self.binding.footprint(i, q)
                        for i, q in enumerate(state.positions)
                        if i not in affected and i != state.selected
                    ):
                        valid = False
                        break
                positions[effect.index] = (
                    p[0] + effect.delta[0],
                    p[1] + effect.delta[1],
                )
            occupied = set()
            for i, p in enumerate(positions):
                cells = self.binding.footprint(i, p)
                if occupied & cells or any(
                    self.binding.terrain.get(c) != self.binding.floor_color
                    for c in cells
                ):
                    valid = False
                occupied.update(cells)
            if valid:
                predictions.add(MultiActorState(tuple(positions), state.selected))
        if len(predictions) > 1:
            raise ValueError("conflicting contact predictions")
        return next(iter(predictions), None)

    def successors(self, state: MultiActorState):
        for i in range(len(state.positions)):
            if i != state.selected:
                yield f"select:{i}", MultiActorState(state.positions, i), 1
        for action in sorted(self.directions):
            destination = self.contact_destination(state, action)
            if destination is None:
                destination = self.ordinary_destination(state, action)
            if destination is not None and destination != state:
                yield f"move:{action}", destination, 1

    def is_goal(self, state: MultiActorState) -> bool:
        return state.positions == self.binding.goals

    def macro_successors(self, state: MultiActorState):
        """Navigation closure to observed contacts or candidate goal anchors.

        Each macro carries its actual primitive actions/cost. Arbitrary
        intermediate states are explored by the local BFS, not handed to the
        outer search. Execution must still verify every primitive transition.
        """
        for index in range(len(state.positions)):
            start = MultiActorState(state.positions, index)
            prefix = () if state.selected == index else (f"select:{index}",)
            targets = {self.binding.goals[index]}
            for rule in self.rules:
                if rule.selected != index:
                    continue
                anchors = {
                    (
                        state.positions[e.index][0] - e.relative_position[0],
                        state.positions[e.index][1] - e.relative_position[1],
                    )
                    for e in rule.effects
                }
                if len(anchors) == 1:
                    targets.update(anchors)
            queue = deque([(start, prefix)])
            seen = {start.positions[index]}
            while queue:
                current, path = queue.popleft()
                position = current.positions[index]
                if position in targets:
                    if position == self.binding.goals[index] and path:
                        yield "|".join(path), current, len(path)
                    for action in sorted(self.directions):
                        destination = self.contact_destination(current, action)
                        if destination is not None:
                            actions = (*path, f"move:{action}")
                            yield "|".join(actions), destination, len(actions)
                for action in sorted(self.directions):
                    destination = self.ordinary_destination(current, action)
                    if (
                        destination is not None
                        and destination.positions[index] not in seen
                    ):
                        seen.add(destination.positions[index])
                        queue.append((destination, (*path, f"move:{action}")))

    def to_dict(self):
        return {
            "shapes": [asdict(s) for s in self.binding.shapes],
            "goals": self.binding.goals,
            "directions": self.directions,
            "rules": [{**asdict(r), "support": self.support[r]} for r in self.rules],
            "assumptions": [
                "rigid sprites",
                "floor under initially hidden cells",
                "matching-size outline occupancy",
                "observed contact effects only",
            ],
        }
