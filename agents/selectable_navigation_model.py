"""Level-local learned transitions for multiple selectable grid objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .tile_scene_binding import Cell, TileScene, TileSceneBinding


@dataclass(frozen=True)
class MultiActorState:
    positions: tuple[Cell, ...]
    selected: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContactRule:
    direction: Cell
    displacement: Cell
    terrain_ray: tuple[int, ...]


class SelectableNavigationModel:
    """Moves, selection, and contact relocation learned from public traces.

    A relocation rule is scoped to its observed terrain sequence and level;
    it is not a universal push/jump rule. Goal occupancy is a hypothesis.
    """

    def __init__(self, binding: TileSceneBinding, initial: Any) -> None:
        self.binding = binding
        self.terrain = {}
        for row in range(binding.layout["rows"]):
            for col in range(binding.layout["cols"]):
                x, y = binding.cell_pixel((col, row))
                self.terrain[(col, row)] = initial.frame[-1][y][x]
        scene = binding.parse(initial)
        # Occupied tiles are a traversability hypothesis, subsequently checked
        # on execution. The hidden terrain under a sprite is not observable.
        for actor in scene.actors:
            self.terrain[actor.cell] = binding.floor_color
        self.floor = frozenset(
            cell for cell, color in self.terrain.items() if color == binding.floor_color
        )
        self.directions: dict[int, Cell] = {}
        self.contact_rules: list[ContactRule] = []
        self.learning_events: list[dict[str, Any]] = []

    @staticmethod
    def state(scene: TileScene) -> MultiActorState:
        if scene.warnings or scene.active_identity is None:
            raise ValueError("cannot plan with uncertain identities or selection")
        return MultiActorState(
            tuple(a.cell for a in scene.actors),
            [a.identity for a in scene.actors].index(scene.active_identity),
        )

    def ray(self, start: Cell, direction: Cell, length: int) -> tuple[int, ...]:
        return tuple(
            self.terrain.get(
                (start[0] + direction[0] * n, start[1] + direction[1] * n), -1
            )
            for n in range(1, length + 1)
        )

    def learn(self, before: TileScene, action: int, after: TileScene) -> None:
        if before.warnings or after.warnings or before.active_identity is None:
            return
        old = {a.identity: a for a in before.actors}
        moved = [
            a
            for a in after.actors
            if a.identity in old and a.cell != old[a.identity].cell
        ]
        if len(moved) != 1 or action == 6:
            return
        actor = moved[0]
        delta = (
            actor.cell[0] - old[actor.identity].cell[0],
            actor.cell[1] - old[actor.identity].cell[1],
        )
        if (
            actor.identity == before.active_identity
            and abs(delta[0]) + abs(delta[1]) == 1
        ):
            if action in self.directions and self.directions[action] != delta:
                raise ValueError("direction model contradicted; stop and retain trace")
            self.directions[action] = delta
        elif actor.identity != before.active_identity and action in self.directions:
            direction = self.directions[action]
            sender = old[before.active_identity].cell
            receiver = old[actor.identity].cell
            if (sender[0] + direction[0], sender[1] + direction[1]) != receiver:
                return
            distance = abs(delta[0]) + abs(delta[1])
            if delta != (direction[0] * distance, direction[1] * distance):
                return
            rule = ContactRule(
                direction, delta, self.ray(receiver, direction, distance)
            )
            if rule not in self.contact_rules:
                self.contact_rules.append(rule)
                self.learning_events.append(
                    {
                        "effect": "contact_relocation",
                        "action": action,
                        "rule": asdict(rule),
                        "supporting_transitions": 1,
                    }
                )

    def successors(self, state: MultiActorState):
        for index in range(len(state.positions)):
            if index != state.selected:
                yield f"select:{index}", MultiActorState(state.positions, index), 1
        source = state.positions[state.selected]
        for action, direction in sorted(self.directions.items()):
            destination = (source[0] + direction[0], source[1] + direction[1])
            positions = list(state.positions)
            if destination in positions:
                receiver = positions.index(destination)
                for rule in self.contact_rules:
                    length = abs(rule.displacement[0]) + abs(rule.displacement[1])
                    landing = (
                        destination[0] + rule.displacement[0],
                        destination[1] + rule.displacement[1],
                    )
                    if (
                        rule.direction == direction
                        and self.ray(destination, direction, length) == rule.terrain_ray
                        and landing in self.floor
                        and landing not in positions
                    ):
                        positions[receiver] = landing
                        yield (
                            f"move:{action}",
                            MultiActorState(tuple(positions), state.selected),
                            1,
                        )
            elif destination in self.floor:
                positions[state.selected] = destination
                yield (
                    f"move:{action}",
                    MultiActorState(tuple(positions), state.selected),
                    1,
                )

    def is_goal(self, state: MultiActorState) -> bool:
        goals = set(self.binding.goals)
        return bool(goals) and goals.issubset(state.positions)

    def to_dict(self):
        return {
            "directions": self.directions,
            "contact_rules": [asdict(r) for r in self.contact_rules],
            "goal_hypothesis": "all observed outline centres occupied",
            "goals": self.binding.goals,
            "floor_hypothesis": sorted(self.floor),
            "learning_events": self.learning_events,
        }
