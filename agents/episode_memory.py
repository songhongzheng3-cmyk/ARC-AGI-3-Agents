"""Small, exact short-term memory for one ARC game episode.

This is deliberately not a semantic/vector store.  ARC decisions depend on
exact frame states and coordinates, so hashes and action outcomes are safer
and cheaper than embedding every observation.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionOutcome:
    action: str
    data: dict[str, Any]
    changed: bool
    score_delta: int
    level_delta: int
    state_hash: str


@dataclass
class _PendingAction:
    action: str
    data: dict[str, Any]
    before_hash: str
    before_score: int
    before_level: int


class EpisodeMemory:
    """Exact state/action memory scoped to a single game instance."""

    def __init__(self, history_limit: int = 8) -> None:
        self.history: deque[ActionOutcome] = deque(maxlen=max(1, history_limit))
        self.actions_by_state: dict[str, set[str]] = defaultdict(set)
        self.tried_clicks_by_state: dict[str, set[tuple[int, int]]] = defaultdict(set)
        self.state_visits: dict[str, int] = defaultdict(int)
        self._pending: _PendingAction | None = None
        self._current_hash = ""
        self._current_level: int | None = None

    @staticmethod
    def state_hash(frame: Any) -> str:
        """Hash the visible grid only; overlays should not create new states."""
        layers = getattr(frame, "frame", []) or []
        # A FrameData can contain sequential animation grids; the last grid is
        # the settled state on which the next action operates.
        visible_grid = layers[-1] if layers else []
        payload = json.dumps(
            visible_grid, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=8).hexdigest()

    @staticmethod
    def action_key(action: str, data: dict[str, Any] | None = None) -> str:
        data = data or {}
        if action.upper() == "ACTION6":
            return f"ACTION6({int(data.get('x', 0))},{int(data.get('y', 0))})"
        return action.upper()

    def observe_frame(self, frame: Any) -> str:
        """Close the previous transition and return the current state hash."""
        current_hash = self.state_hash(frame)
        score = int(getattr(frame, "levels_completed", 0))
        level = score
        if self._pending is not None:
            pending = self._pending
            changed = current_hash != pending.before_hash
            self.history.append(
                ActionOutcome(
                    action=pending.action,
                    data=dict(pending.data),
                    changed=changed,
                    score_delta=score - pending.before_score,
                    level_delta=level - pending.before_level,
                    state_hash=current_hash,
                )
            )
            if not changed and pending.action.upper() == "ACTION6":
                click = (
                    int(pending.data.get("x", 0)),
                    int(pending.data.get("y", 0)),
                )
                self.tried_clicks_by_state[pending.before_hash].add(click)
            self._pending = None

        # Game mechanics can change at a new level; don't leak state/action
        # assumptions across levels, while retaining the compact recent history.
        if self._current_level is not None and level != self._current_level:
            self.actions_by_state.clear()
            self.tried_clicks_by_state.clear()
            self.state_visits.clear()
        self._current_level = level
        self._current_hash = current_hash
        self.state_visits[current_hash] += 1
        return current_hash

    def record_action(
        self,
        action: str,
        data: dict[str, Any] | None,
        state_hash: str,
        score: int,
    ) -> None:
        """Record an action before submitting it to the environment."""
        payload = dict(data or {})
        self.actions_by_state[state_hash].add(self.action_key(action, payload))
        self._pending = _PendingAction(
            action=action.upper(),
            data=payload,
            before_hash=state_hash,
            before_score=score,
            before_level=score,
        )

    def tried_clicks(self, state_hash: str) -> list[tuple[int, int]]:
        return sorted(self.tried_clicks_by_state.get(state_hash, set()))

    def tried_actions(self, state_hash: str) -> set[str]:
        return set(self.actions_by_state.get(state_hash, set()))

    @staticmethod
    def click_candidates(
        frame: Any,
        tried: set[tuple[int, int]] | None = None,
        max_candidates: int = 12,
    ) -> list[dict[str, Any]]:
        """Extract deterministic ACTION6 points from non-background regions."""
        layers = getattr(frame, "frame", []) or []
        grid = layers[-1] if layers else []
        if not grid or not grid[0]:
            return []
        height = len(grid)
        width = len(grid[0])
        tried = tried or set()

        counts: dict[int, int] = defaultdict(int)
        for row in grid:
            for value in row:
                counts[int(value)] += 1
        background = max(counts, key=counts.get)
        visited: set[tuple[int, int]] = set()
        components: list[dict[str, Any]] = []
        for y in range(height):
            for x in range(width):
                if (x, y) in visited or int(grid[y][x]) == background:
                    continue
                queue = [(x, y)]
                visited.add((x, y))
                pixels: list[tuple[int, int]] = []
                color = int(grid[y][x])
                while queue:
                    cx, cy = queue.pop()
                    pixels.append((cx, cy))
                    for nx, ny in (
                        (cx - 1, cy),
                        (cx + 1, cy),
                        (cx, cy - 1),
                        (cx, cy + 1),
                    ):
                        if not (0 <= nx < width and 0 <= ny < height):
                            continue
                        if (nx, ny) in visited or int(grid[ny][nx]) == background:
                            continue
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                if not pixels:
                    continue
                center_x = sum(px for px, _ in pixels) / len(pixels)
                center_y = sum(py for _, py in pixels) / len(pixels)
                available_pixels = [point for point in pixels if point not in tried]
                if not available_pixels:
                    continue
                representative = min(
                    available_pixels,
                    key=lambda p: (p[0] - center_x) ** 2 + (p[1] - center_y) ** 2,
                )
                components.append(
                    {
                        "x": representative[0],
                        "y": representative[1],
                        "area": len(pixels),
                        "color": color,
                        "salient": color >= 6,
                    }
                )

        # Medium salient regions are usually more useful interactive targets;
        # area breaks ties deterministically without adding model calls.
        components.sort(
            key=lambda item: (
                bool(item["salient"]),
                min(int(item["area"]), 2048),
                -int(item["y"]),
                -int(item["x"]),
            ),
            reverse=True,
        )
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for item in components:
            point = (int(item["x"]), int(item["y"]))
            if point in tried or point in seen:
                continue
            seen.add(point)
            candidates.append(item)
            if len(candidates) >= max_candidates:
                break
        return candidates

    def prompt_context(
        self,
        state_hash: str,
        available_actions: list[Any],
        click_candidates: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build a bounded, deterministic summary for the next LLM request."""
        available: list[str] = []
        for value in available_actions or []:
            try:
                action_id = int(getattr(value, "value", value))
                available.append("RESET" if action_id == 0 else f"ACTION{action_id}")
            except (TypeError, ValueError):
                available.append(str(value))
        tried = self.tried_actions(state_hash)
        untried = [name for name in available if name not in tried]
        clicks = self.tried_clicks(state_hash)
        ranked = self.ranked_actions(
            state_hash,
            available_actions,
            click_candidates=click_candidates,
        )
        lines = [
            f"state_hash={state_hash[:8]} visits={self.state_visits.get(state_hash, 0)}",
            f"untried_actions={', '.join(untried[:8]) or 'none'}",
        ]
        if ranked:
            lines.append("ranked_actions=" + ", ".join(ranked[:12]))
        if clicks:
            lines.append(
                "no_change_clicks="
                + ", ".join(f"({x},{y})" for x, y in clicks[:12])
            )
        if click_candidates:
            lines.append(
                "click_candidates="
                + ", ".join(
                    f"({int(item['x'])},{int(item['y'])})"
                    f"/c{int(item['color'])}/a{int(item['area'])}"
                    for item in click_candidates[:12]
                )
            )
            if "ACTION6" in untried:
                first = click_candidates[0]
                lines.append(
                    "recommended=ACTION6("
                    f"{int(first['x'])},{int(first['y'])})"
                )
        elif untried:
            lines.append(f"recommended={untried[0]}")
        if self.history:
            recent = []
            for outcome in list(self.history)[-6:]:
                result = "changed" if outcome.changed else "no-change"
                key = self.action_key(outcome.action, outcome.data)
                recent.append(f"{key}:{result}")
            lines.append("recent=" + "; ".join(recent))
        return "\n".join(lines)

    def ranked_actions(
        self,
        state_hash: str,
        available_actions: list[Any],
        click_candidates: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Return executable actions ordered by unexplored state first.

        ACTION6 is expanded into coordinate-specific actions only when it is
        still untried in this state.  This makes the ranking useful to the
        model without pretending that two clicks at different coordinates are
        the same action.
        """
        available: list[str] = []
        for value in available_actions or []:
            try:
                action_id = int(getattr(value, "value", value))
                name = "RESET" if action_id == 0 else f"ACTION{action_id}"
            except (TypeError, ValueError):
                name = str(value).upper()
            if name not in available:
                available.append(name)

        tried = self.tried_actions(state_hash)
        untried = [name for name in available if name not in tried]
        ranked: list[str] = []

        # Keep ordinary untried actions in the environment's stable order.
        ranked.extend(name for name in untried if name != "ACTION6")
        if "ACTION6" in untried:
            for item in click_candidates or []:
                ranked.append(
                    f"ACTION6({int(item['x'])},{int(item['y'])})"
                )
            # Preserve ACTION6 as a fallback when extraction found no point.
            if not click_candidates:
                ranked.append("ACTION6")

        # Once exploration is exhausted, retain a compact fallback ordering.
        if not ranked:
            ranked.extend(available)
        return ranked
