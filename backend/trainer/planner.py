"""Session planning: pick the skills that need work and turn them into drills.

Priority = (1 - mastery) weighted up when FSRS says the skill is due. Low-mastery skills
start with ear training (perception precedes production), then words, then sentences."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import bkt, exercises, fsrs
from .skills import SKILL_LABEL


@dataclass
class SkillState:
    skill: str
    p_known: float = bkt.DEFAULT.p_init
    card: fsrs.Card = field(default_factory=fsrs.Card)
    n_obs: int = 0

    def observe(self, correct: bool) -> None:
        self.p_known = bkt.update(self.p_known, correct)
        self.n_obs += 1

    def review(self, score: float, now: datetime) -> None:
        self.card = fsrs.review(self.card, fsrs.rating_from_score(score), now)


@dataclass(frozen=True)
class Drill:
    kind: str  # "perception" | "words" | "sentences" | "twister" | "stress_pairs"
    skill: str
    label: str
    items: list[Any]
    reason: str


def priority(s: SkillState, now: datetime) -> float:
    if s.n_obs == 0:
        return 0.0  # never observed: no evidence of a problem yet
    due = s.card.due is None or s.card.due <= now
    return (1 - s.p_known) * (1.5 if due else 0.5)


def plan(
    states: list[SkillState], now: datetime, max_skills: int = 3, seed: int | None = None
) -> list[Drill]:
    ranked = sorted((s for s in states if s.p_known < bkt.MASTERED), key=lambda s: -priority(s, now))
    ranked = [s for s in ranked if priority(s, now) > 0][:max_skills]
    out: list[Drill] = []
    for s in ranked:
        lab = SKILL_LABEL.get(s.skill, s.skill)
        why = f"mastery {s.p_known:.0%}" + (
            ", due for review" if s.card.due is None or s.card.due <= now else ""
        )
        if s.skill == "LEXICAL_STRESS":
            out.append(Drill("stress_pairs", s.skill, lab, list(exercises.stress_shift_pairs(10)), why))
            continue
        pairs = exercises.minimal_pairs(s.skill, 12)
        if s.p_known < 0.5 and pairs:
            out.append(Drill("perception", s.skill, lab, pairs[:8], why + " → train the ear first"))
        words = exercises.words_for(s.skill, n=8, seed=seed)
        if words:
            out.append(Drill("words", s.skill, lab, words, why))
        if s.skill in exercises.TONGUE_TWISTERS and s.p_known >= 0.3:
            out.append(Drill("twister", s.skill, lab, exercises.TONGUE_TWISTERS[s.skill], why))
    return out
