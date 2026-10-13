"""Compact, auditable game-rule inference from observed grid transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

from .object_extractor import GridObject, GridSnapshot


@dataclass
class ActionEffect:
    trials: int = 0
    moved: int = 0
    blocked: int = 0
    dx: list[int] = field(default_factory=list)
    dy: list[int] = field(default_factory=list)

    def observe(self, delta: tuple[int, int], blocked: bool) -> None:
        self.trials += 1
        self.dx.append(delta[0])
        self.dy.append(delta[1])
        if blocked:
            self.blocked += 1
        else:
            self.moved += 1

    def summary(self) -> dict[str, Any]:
        return {
            "trials": self.trials,
            "moved": self.moved,
            "blocked": self.blocked,
            "median_dx": int(median(self.dx)) if self.dx else 0,
            "median_dy": int(median(self.dy)) if self.dy else 0,
        }


@dataclass(frozen=True)
class TransitionEvidence:
    action: str
    player_before: tuple[int, int] | None
    player_after: tuple[int, int] | None
    delta: tuple[int, int]
    blocked: bool
    contact_objects: tuple[str, ...]
    level_delta: int


class GameModel:
    """Infer control semantics and goal candidates without claiming certainty."""

    def __init__(self) -> None:
        self.action_effects: dict[str, ActionEffect] = {}
        self.evidence: list[TransitionEvidence] = []
        self.initial: GridSnapshot | None = None
        self._avatar_colors = {9, 12}

    def reset(self, snapshot: GridSnapshot) -> None:
        self.action_effects.clear()
        self.evidence.clear()
        self.initial = snapshot

    @staticmethod
    def _bbox_gap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
        """Chebyshev gap between two rectangles (0 means touching/overlap)."""
        left_x0, left_y0, left_x1, left_y1 = left
        right_x0, right_y0, right_x1, right_y1 = right
        dx = max(left_x0 - right_x1 - 1, right_x0 - left_x1 - 1, 0)
        dy = max(left_y0 - right_y1 - 1, right_y0 - left_y1 - 1, 0)
        return max(dx, dy)

    def _avatar_parts(self, snapshot: GridSnapshot) -> set[str]:
        """Return all color-9/12 components belonging to the player sprite."""
        marker = self._avatar(snapshot)
        if marker is None:
            return set()
        parts = {marker.object_id}
        for item in snapshot.objects:
            if item.color not in self._avatar_colors:
                continue
            # The player's 9 body touches the 12 marker and is below it in the
            # observed sprite.  Restricting by relative position keeps an
            # above-positioned goal from being absorbed when approached.
            marker_center = self._center(marker)
            item_center = self._center(item)
            if self._bbox_gap(marker.bbox, item.bbox) <= 0 and item_center[1] >= marker_center[1]:
                parts.add(item.object_id)
        return parts

    def _group_same_color(self, snapshot: GridSnapshot, item: GridObject) -> list[GridObject]:
        """Group nearby components split by a one-cell gap or a marker pixel."""
        group = [item]
        for other in snapshot.objects:
            if other.object_id == item.object_id or other.color != item.color:
                continue
            if self._bbox_gap(item.bbox, other.bbox) <= 1:
                group.append(other)
        return group

    @staticmethod
    def _center(item: GridObject) -> tuple[int, int]:
        return (
            int(round(item.centroid[0])),
            int(round(item.centroid[1])),
        )

    def _avatar(self, snapshot: GridSnapshot) -> GridObject | None:
        # Color 12 is a stable avatar marker in the observed keyboard game.
        markers = [item for item in snapshot.objects if item.color == 12]
        if markers:
            return max(markers, key=lambda item: item.area)
        # Generic fallback for games without the marker color: use the largest
        # color-9 object away from the grid boundary.
        candidates = [
            item
            for item in snapshot.objects
            if item.color == 9 and not item.touches_boundary
        ]
        return max(candidates, key=lambda item: item.area, default=None)

    def _contacts(
        self,
        before: GridSnapshot,
        after: GridSnapshot,
        avatar_before: GridObject | None,
        avatar_after: GridObject | None,
    ) -> tuple[str, ...]:
        if avatar_before is None or avatar_after is None:
            return ()
        # IDs include the bounding box and therefore change whenever an object
        # moves.  Use geometry instead: an object is contacted when the new
        # avatar sprite overlaps it or comes directly adjacent to it.
        avatar_box = avatar_after.bbox
        contacted: list[str] = []
        target_ids = {
            component_id
            for candidate in self.interaction_candidates(before)
            for component_id in candidate.get("component_ids", [candidate["object_id"]])
        }
        for item in before.objects:
            if item.object_id not in target_ids:
                continue
            if self._bbox_gap(avatar_box, item.bbox) <= 1:
                contacted.append(item.object_id)
        return tuple(sorted(contacted))

    def observe(
        self,
        action: str,
        before: GridSnapshot,
        after: GridSnapshot,
        changed_cells: int,
        level_delta: int = 0,
    ) -> TransitionEvidence:
        action = action.upper()
        avatar_before = self._avatar(before)
        avatar_after = self._avatar(after)
        before_center = self._center(avatar_before) if avatar_before else None
        after_center = self._center(avatar_after) if avatar_after else None
        if before_center is None or after_center is None:
            delta = (0, 0)
        else:
            delta = (
                after_center[0] - before_center[0],
                after_center[1] - before_center[1],
            )
        blocked = delta == (0, 0)
        effect = self.action_effects.setdefault(action, ActionEffect())
        effect.observe(delta, blocked)
        item = TransitionEvidence(
            action=action,
            player_before=before_center,
            player_after=after_center,
            delta=delta,
            blocked=blocked,
            contact_objects=self._contacts(before, after, avatar_before, avatar_after),
            level_delta=level_delta,
        )
        self.evidence.append(item)
        return item

    def target_candidates(self, snapshot: GridSnapshot | None = None) -> list[dict[str, Any]]:
        snapshot = snapshot or self.initial
        if snapshot is None:
            return []
        avatar = self._avatar(snapshot)
        avatar_center = self._center(avatar) if avatar else None
        avatar_parts = self._avatar_parts(snapshot)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in snapshot.objects:
            if item.object_id in seen or item.color not in self._avatar_colors or item.color == 12:
                continue
            if item.object_id in avatar_parts:
                continue
            group = self._group_same_color(snapshot, item)
            seen.update(part.object_id for part in group)
            center = self._center(item)
            if avatar_center and center == avatar_center:
                continue
            min_x = min(part.bbox[0] for part in group)
            min_y = min(part.bbox[1] for part in group)
            max_x = max(part.bbox[2] for part in group)
            max_y = max(part.bbox[3] for part in group)
            group_area = sum(part.area for part in group)
            group_center = (
                int(round(sum(part.centroid[0] * part.area for part in group) / group_area)),
                int(round(sum(part.centroid[1] * part.area for part in group) / group_area)),
            )
            distance = (
                abs(group_center[0] - avatar_center[0]) + abs(group_center[1] - avatar_center[1])
                if avatar_center
                else 0
            )
            touches_boundary = any(part.touches_boundary for part in group)
            # The game has a bottom HUD made from small color-9 components.
            # Keep this heuristic conservative: only suppress small objects in
            # the bottom fifth, never a large interior object.
            hud_like = (
                group_center[1] >= snapshot.height * 0.80
                and group_area <= max(24, snapshot.width // 2)
            )
            if touches_boundary or hud_like:
                continue
            candidates.append(
                {
                    "object_id": item.object_id,
                    "component_ids": [part.object_id for part in group],
                    "color": item.color,
                    "area": group_area,
                    "bbox": (min_x, min_y, max_x, max_y),
                    "center": group_center,
                    "distance": distance,
                    "touches_boundary": touches_boundary,
                    "hud_like": hud_like,
                    "hypothesis": "same-color target candidate",
                }
            )
        # Prefer interior static-sized objects.  Distance is only a tie-breaker;
        # a far-away endpoint is often more informative than a nearby HUD item.
        candidates.sort(key=lambda item: (item["hud_like"], item["touches_boundary"], -item["area"], item["distance"]))
        return candidates

    def interaction_candidates(self, snapshot: GridSnapshot | None = None) -> list[dict[str, Any]]:
        """Return small interior objects that are plausible interaction targets."""
        snapshot = snapshot or self.initial
        if snapshot is None:
            return []
        avatar = self._avatar(snapshot)
        avatar_center = self._center(avatar) if avatar else None
        avatar_parts = self._avatar_parts(snapshot)
        color_area: dict[int, int] = {}
        for item in snapshot.objects:
            color_area[item.color] = color_area.get(item.color, 0) + item.area
        structural_limit = max(64, int(snapshot.width * snapshot.height * 0.08))
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in snapshot.objects:
            if item.object_id in seen or item.object_id in avatar_parts:
                continue
            if item.color == 12 or color_area.get(item.color, 0) > structural_limit:
                continue
            group = self._group_same_color(snapshot, item)
            seen.update(part.object_id for part in group)
            group_area = sum(part.area for part in group)
            min_x = min(part.bbox[0] for part in group)
            min_y = min(part.bbox[1] for part in group)
            max_x = max(part.bbox[2] for part in group)
            max_y = max(part.bbox[3] for part in group)
            center = (
                int(round(sum(part.centroid[0] * part.area for part in group) / group_area)),
                int(round(sum(part.centroid[1] * part.area for part in group) / group_area)),
            )
            if any(part.touches_boundary for part in group):
                continue
            # The lower strip is HUD/status UI even when it is a large
            # connected component, so do not treat it as an interaction goal.
            if center[1] >= snapshot.height * 0.80:
                continue
            distance = (
                abs(center[0] - avatar_center[0]) + abs(center[1] - avatar_center[1])
                if avatar_center
                else 0
            )
            candidates.append(
                {
                    "object_id": item.object_id,
                    "component_ids": [part.object_id for part in group],
                    "color": item.color,
                    "area": group_area,
                    "bbox": (min_x, min_y, max_x, max_y),
                    "center": center,
                    "distance": distance,
                    "hypothesis": "small interior interaction candidate",
                }
            )
        candidates.sort(key=lambda item: (item["distance"], -item["area"]))
        return candidates

    def hypotheses(self, snapshot: GridSnapshot | None = None) -> list[dict[str, Any]]:
        effects = {name: effect.summary() for name, effect in self.action_effects.items()}
        moved = sum(item["moved"] for item in effects.values())
        blocked = sum(item["blocked"] for item in effects.values())
        items: list[dict[str, Any]] = []
        if moved:
            confidence = min(0.98, 0.45 + 0.05 * moved)
            items.append(
                {
                    "key": "keyboard-navigation",
                    "statement": "actions move a controllable avatar on a grid",
                    "confidence": round(confidence, 2),
                    "evidence": f"{moved} moving transitions",
                }
            )
        if blocked:
            items.append(
                {
                    "key": "collision-or-boundary",
                    "statement": "some attempted moves are blocked by a boundary or obstacle",
                    "confidence": round(min(0.95, 0.45 + 0.08 * blocked), 2),
                    "evidence": f"{blocked} blocked transitions",
                }
            )
        targets = self.target_candidates(snapshot)
        if targets:
            items.append(
                {
                    "key": "same-color-target",
                    "statement": "a stationary color-9 object may be a target or endpoint",
                    "confidence": 0.35,
                    "evidence": f"{len(targets)} candidate objects",
                }
            )
        if any(item.contact_objects for item in self.evidence):
            items.append(
                {
                    "key": "overlap-not-collection",
                    "statement": "object disappearance during overlap is not yet proof of collection",
                    "confidence": 0.7,
                    "evidence": "contact object reappeared or level did not advance",
                }
            )
            if not any(item.level_delta > 0 for item in self.evidence):
                items.append(
                    {
                        "key": "contact-not-sufficient",
                        "statement": "candidate contact has not been sufficient to complete a level",
                        "confidence": 0.8,
                        "evidence": f"{sum(bool(item.contact_objects) for item in self.evidence)} contact transitions, 0 level advances",
                    }
                )
        return items

    def target_distance(self, snapshot: GridSnapshot | None = None) -> int | None:
        """Distance from the avatar to the best filtered target candidate."""
        snapshot = snapshot or self.initial
        if snapshot is None:
            return None
        avatar = self._avatar(snapshot)
        targets = self.target_candidates(snapshot)
        if avatar is None or not targets:
            return None
        center = self._center(avatar)
        return min(
            abs(center[0] - int(target["center"][0]))
            + abs(center[1] - int(target["center"][1]))
            for target in targets
        )

    def interaction_distance(self, snapshot: GridSnapshot | None = None) -> int | None:
        snapshot = snapshot or self.initial
        if snapshot is None:
            return None
        avatar = self._avatar(snapshot)
        candidates = self.interaction_candidates(snapshot)
        if avatar is None or not candidates:
            return None
        center = self._center(avatar)
        return min(
            abs(center[0] - int(candidate["center"][0]))
            + abs(center[1] - int(candidate["center"][1]))
            for candidate in candidates
        )

    def summary(self, snapshot: GridSnapshot | None = None) -> dict[str, Any]:
        return {
            "action_effects": {
                name: effect.summary()
                for name, effect in sorted(self.action_effects.items())
            },
            "target_candidates": self.target_candidates(snapshot),
            "target_distance": self.target_distance(snapshot),
            "interaction_candidates": self.interaction_candidates(snapshot),
            "interaction_distance": self.interaction_distance(snapshot),
            "hypotheses": self.hypotheses(snapshot),
            "level_deltas": sum(item.level_delta for item in self.evidence),
        }
