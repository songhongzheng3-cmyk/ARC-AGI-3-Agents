"""Small, observation-only memory of uniquely identifiable moving components.

Position alone aliases the two directions of a shuttle. Keep its last two
observed cells, without extrapolating through occlusion or duplicate identities.
This passive memory assigns no game-specific role to an object or palette.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .mechanism_state_tracker import Cell, MechanismState

ObjectSignature = tuple[int, int, str]


@dataclass(frozen=True)
class MotionObservation:
    signature: ObjectSignature
    previous: Cell | None
    current: Cell | None


MotionContext = tuple[MotionObservation, ...]


class MotionContextMemory:
    """Select compact components by observed movement, not a minimum area.

    Only unique, non-player components fully inside the board are eligible.
    Two components with the same signature are ambiguous and are not matched.
    Missing observations remain explicit, rather than implying stationarity.
    """

    def __init__(self) -> None:
        self.context: MotionContext = ()
        self.positions: dict[ObjectSignature, Cell] = {}
        self.moving: set[ObjectSignature] = set()
        self.level: int | None = None
        self.previous_resource: int | None = None

    def reset(self) -> None:
        self.__init__()

    @staticmethod
    def visible_positions(state: MechanismState) -> dict[ObjectSignature, Cell]:
        layout = state.layout
        if layout is None:
            return {}
        grouped: dict[ObjectSignature, list[Cell]] = defaultdict(list)
        left, top, right, bottom = layout.bounds
        width, height = layout.cell_size
        objects = (
            state.objects if state.motion_entities is None else state.motion_entities
        )
        for item in objects:
            x1, y1, x2, y2 = item.bbox
            if (
                item.static_target
                or item.object_id in state.player_component_ids
                or item.cell == state.player_position
                or not (left <= x1 <= x2 <= right and top <= y1 <= y2 <= bottom)
                or x2 - x1 + 1 > width
                or y2 - y1 + 1 > height
            ):
                continue
            grouped[(item.color, item.area, item.shape_signature)].append(item.cell)
        return {key: cells[0] for key, cells in grouped.items() if len(cells) == 1}

    def observe(self, state: MechanismState, *, reset: bool = False) -> MotionContext:
        if (
            reset
            or self.level != state.level
            or (
                self.previous_resource == 0
                and state.resource_capacity > 0
                and state.resource_level > 0
            )
        ):
            self.reset()
        current = self.visible_positions(state)
        for signature, cell in current.items():
            previous = self.positions.get(signature)
            if (
                previous is not None
                and sum(abs(a - b) for a, b in zip(previous, cell)) == 1
            ):
                self.moving.add(signature)
        self.context = tuple(
            MotionObservation(
                signature, self.positions.get(signature), current.get(signature)
            )
            for signature in sorted(self.moving)
        )
        self.positions = current
        self.level = state.level
        self.previous_resource = state.resource_level
        return self.context
