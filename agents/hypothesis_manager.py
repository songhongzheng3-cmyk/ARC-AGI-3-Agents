"""Evidence-weighted, episode-scoped rule hypotheses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .transition_model import Transition


@dataclass
class Hypothesis:
    key: str
    statement: str
    confidence: float = 0.2
    evidence: list[str] = field(default_factory=list)
    counter_evidence: list[str] = field(default_factory=list)
    tests: int = 0

    def support(self, reason: str) -> None:
        self.tests += 1
        self.confidence = min(0.98, self.confidence + 0.18)
        self.evidence.append(reason)
        del self.evidence[:-4]

    def refute(self, reason: str) -> None:
        self.tests += 1
        self.confidence = max(0.02, self.confidence - 0.15)
        self.counter_evidence.append(reason)
        del self.counter_evidence[:-4]


class HypothesisManager:
    """Turns observed transitions into auditable action-effect hypotheses."""

    def __init__(self, limit: int = 32) -> None:
        self.limit = max(1, limit)
        self.hypotheses: dict[str, Hypothesis] = {}

    def _get(self, key: str, statement: str) -> Hypothesis:
        if key not in self.hypotheses and len(self.hypotheses) >= self.limit:
            weakest = min(self.hypotheses, key=lambda item: self.hypotheses[item].confidence)
            del self.hypotheses[weakest]
        return self.hypotheses.setdefault(key, Hypothesis(key, statement))

    def observe(self, transition: Transition) -> None:
        key = f"action:{transition.action}"
        hypothesis = self._get(
            key,
            f"{transition.action} changes the visible state in the current context",
        )
        if transition.changed or transition.score_delta > 0:
            hypothesis.support(
                f"state {transition.before_hash[:8]} -> {transition.after_hash[:8]}"
            )
        else:
            hypothesis.refute(f"no visible change at {transition.before_hash[:8]}")

        for effect in transition.object_changes:
            effect_key = f"effect:{transition.action}:{effect}"
            effect_hypothesis = self._get(
                effect_key,
                f"{transition.action} can cause {effect}",
            )
            effect_hypothesis.support(effect)

    def confidence(self, action: str) -> float:
        item = self.hypotheses.get(f"action:{action.upper()}")
        return item.confidence if item else 0.2

    def rank_actions(self, actions: list[str]) -> list[str]:
        """Rank actions by confidence and a small exploration bonus."""
        order = {action: index for index, action in enumerate(actions)}

        def base_action(action: str) -> str:
            return re.sub(r"\([^)]*\)$", "", action).upper()

        return sorted(
            actions,
            key=lambda action: (
                self.confidence(base_action(action))
                + (0.08 if base_action(action) not in self.hypotheses else 0.0),
                -order[action],
            ),
            reverse=True,
        )

    def prompt_context(self, max_items: int = 6) -> str:
        items = sorted(
            self.hypotheses.values(),
            key=lambda item: item.confidence,
            reverse=True,
        )[:max_items]
        return "\n".join(
            f"hypothesis={item.statement}; confidence={item.confidence:.2f}"
            for item in items
        )
