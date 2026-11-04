"""Conservative visual ownership for compact, potentially multicolour sprites.

Connectivity proposes a visual entity; it does not prove persistent identity or
an interaction rule. Real pixel masks, not rounded cell proximity, link parts.
Target contents, player contents and oversized/ambiguous groups are excluded.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .mechanism_state_tracker import BBox, MechanismObject, MechanismState


@dataclass(frozen=True)
class EntityMembership:
    entity_id: str
    component_ids: tuple[str, ...]


@dataclass(frozen=True)
class EntityView:
    objects: tuple[MechanismObject, ...]
    memberships: tuple[EntityMembership, ...]
    exclusions: tuple[tuple[str, str], ...]


def _contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] <= inner[0] <= inner[2] <= outer[2]
        and outer[1] <= inner[1] <= inner[3] <= outer[3]
    )


def _overlaps(first: BBox, second: BBox) -> bool:
    return max(first[0], second[0]) <= min(first[2], second[2]) and max(
        first[1], second[1]
    ) <= min(first[3], second[3])


def extract_entities(state: MechanismState, grid: np.ndarray) -> EntityView:
    layout = state.layout
    if layout is None or layout.confidence < 0.8 or layout.projected_view:
        return EntityView((), (), ())
    targets = [item.bbox for item in state.objects if item.static_target]
    width, height = layout.cell_size
    pieces = []
    exclusions = []
    pixels = {}
    for item in state.objects:
        reason = None
        x1, y1, x2, y2 = item.bbox
        if item.object_id in state.player_component_ids or (
            state.player_bbox and _overlaps(state.player_bbox, item.bbox)
        ):
            reason = "player_owned_or_overlapped"
        elif any(_contains(target, item.bbox) for target in targets):
            reason = "target_owned"
        elif not _contains(layout.bounds, item.bbox):
            reason = "outside_board"
        elif x2 - x1 + 1 > width or y2 - y1 + 1 > height:
            reason = "oversized_component"
        if reason is not None:
            exclusions.append((item.object_id, reason))
            continue
        # The crop can include a disconnected component of the same colour.
        # If its area no longer matches the observed component, abstain.
        ys, xs = np.where(grid[y1 : y2 + 1, x1 : x2 + 1] == item.color)
        if len(xs) != item.area:
            exclusions.append((item.object_id, "ambiguous_pixel_mask"))
            continue
        index = len(pieces)
        pieces.append(item)
        for x, y in zip(xs, ys):
            pixels[(int(x) + x1, int(y) + y1)] = index
    neighbors = {i: set() for i in range(len(pieces))}
    for (x, y), first in pixels.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                second = pixels.get((x + dx, y + dy))
                if second is not None and first != second:
                    neighbors[first].add(second)
    groups, seen = [], set()
    for first in neighbors:
        if first in seen:
            continue
        queue = [first]
        seen.add(first)
        for current in queue:
            for other in neighbors[current] - seen:
                seen.add(other)
                queue.append(other)
        groups.append(set(queue))
    objects, memberships = [], []
    for group in groups:
        members = [pieces[i] for i in sorted(group)]
        bbox = (
            min(o.bbox[0] for o in members),
            min(o.bbox[1] for o in members),
            max(o.bbox[2] for o in members),
            max(o.bbox[3] for o in members),
        )
        x1, y1, x2, y2 = bbox
        if x2 - x1 + 1 > width or y2 - y1 + 1 > height:
            exclusions.extend(
                (o.object_id, "oversized_connected_group") for o in members
            )
            continue
        colours = Counter()
        raster = [["." for _ in range(x2 - x1 + 1)] for _ in range(y2 - y1 + 1)]
        for (x, y), index in pixels.items():
            if index in group:
                colour = pieces[index].color
                raster[y - y1][x - x1] = str(colour)
                colours[colour] += 1
        signature = "entity=" + "/".join(",".join(row) for row in raster)
        ids = tuple(sorted(o.object_id for o in members))
        # Reuse the coordinate convention of the full player sprite, not that
        # of whichever coloured fragment happened to be chosen as an anchor.
        cell = (
            int(round(((x1 + x2) / 2 - layout.origin[0]) / width)),
            int(round(((y1 + y2) / 2 - layout.origin[1]) / height)),
        )
        identifier = "entity:" + ";".join(ids)
        objects.append(
            MechanismObject(
                identifier,
                min(colours, key=lambda c: (-colours[c], c)),
                cell,
                bbox,
                sum(colours.values()),
                signature,
            )
        )
        memberships.append(EntityMembership(identifier, ids))
    return EntityView(tuple(objects), tuple(memberships), tuple(exclusions))
