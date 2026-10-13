"""Backward-compatible public data models for older integrations.

Runtime action and frame types moved to :mod:`arcengine`.  Keep the historical
import path working while retaining the small aggregate scorecard models used
by legacy callers and tests.
"""

from __future__ import annotations

from typing import Any

from arcengine import ActionInput, GameAction, GameState
from arcengine import FrameData as ArcFrameData
from pydantic import BaseModel, Field, computed_field


class Card(BaseModel):
    """Aggregate repeated plays for one game."""

    game_id: str
    total_plays: int = 0
    guids: list[str] = Field(default_factory=list, exclude=True)
    scores: list[float] = Field(default_factory=list)
    states: list[GameState] = Field(default_factory=list)
    actions: list[int] = Field(default_factory=list)
    resets: list[int] = Field(default_factory=list)

    @property
    def idx(self) -> int:
        return self.total_plays - 1

    @property
    def started(self) -> bool:
        return self.total_plays > 0

    @property
    def score(self) -> float | None:
        return self.scores[self.idx] if self.started and self.scores else None

    @property
    def high_score(self) -> float:
        return max(self.scores) if self.scores else 0

    @property
    def state(self) -> GameState:
        return (
            self.states[self.idx]
            if self.started and self.states
            else GameState.NOT_PLAYED
        )

    @property
    def action_count(self) -> int | None:
        return self.actions[self.idx] if self.started and self.actions else None

    @property
    def total_actions(self) -> int:
        return sum(self.actions)


class Scorecard(BaseModel):
    """Legacy multi-game scorecard aggregate."""

    games: list[str] = Field(default_factory=list, exclude=True)
    cards: dict[str, Card] = Field(default_factory=dict)
    source_url: str | None = None
    tags: list[str] | None = None
    opaque: Any | None = None
    card_id: str = ""
    api_key: str = ""

    @computed_field(return_type=int)
    @property
    def won(self) -> int:
        return sum(GameState.WIN in card.states for card in self.cards.values())

    @computed_field(return_type=int)
    @property
    def played(self) -> int:
        return sum(bool(card.states) for card in self.cards.values())

    @computed_field(return_type=int)
    @property
    def total_actions(self) -> int:
        return sum(card.total_actions for card in self.cards.values())

    @computed_field(return_type=float)
    @property
    def score(self) -> float:
        return sum(card.high_score for card in self.cards.values())

    def get(self, game_id: str | None = None) -> dict[str, Any]:
        if game_id is not None:
            card = self.cards.get(game_id)
            return {game_id: card.model_dump()} if card else {}
        return {key: value.model_dump() for key, value in self.cards.items()}

    def get_json_for(self, game_id: str) -> dict[str, Any]:
        card = self.cards.get(game_id)
        return {
            "won": self.won,
            "played": self.played,
            "total_actions": self.total_actions,
            "score": self.score,
            "cards": {game_id: card.model_dump()} if card else {},
        }


class FrameData(ArcFrameData):
    """Historical frame shape with the pre-SDK aggregate score field."""

    score: int = Field(default=0, ge=0, le=254)


__all__ = [
    "ActionInput",
    "Card",
    "FrameData",
    "GameAction",
    "GameState",
    "Scorecard",
]
