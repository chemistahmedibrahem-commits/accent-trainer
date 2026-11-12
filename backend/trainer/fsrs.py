"""FSRS-5 spaced-repetition scheduler (Ye et al.; open-spaced-repetition) – compact port.

A *card* here is a skill (e.g. "TH_SUB"), reviewed whenever the learner practises it. The
rating comes from the engine's score for that skill in the session, not from self-report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import IntEnum

W: tuple[float, ...] = (
    0.40255,
    1.18385,
    3.173,
    15.69105,
    7.1949,
    0.5345,
    1.4604,
    0.0046,
    1.54575,
    0.1192,
    1.01925,
    1.9395,
    0.11,
    0.29605,
    2.2698,
    0.2315,
    2.9898,
    0.51655,
    0.6621,
)
DECAY = -0.5
FACTOR = 19 / 81  # so that R(S) = 0.9
DESIRED_RETENTION = 0.9
MAX_INTERVAL_DAYS = 365.0


class Rating(IntEnum):
    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4


def rating_from_score(score: float) -> Rating:
    """Map a 0–100 engine score for the skill to an FSRS grade."""
    if score < 50:
        return Rating.AGAIN
    if score < 75:
        return Rating.HARD
    if score < 90:
        return Rating.GOOD
    return Rating.EASY


@dataclass(frozen=True)
class Card:
    stability: float = 0.0  # days until R drops to 90 %
    difficulty: float = 0.0  # 1..10
    reps: int = 0
    lapses: int = 0
    last_review: datetime | None = None
    due: datetime | None = None

    @property
    def is_new(self) -> bool:
        return self.reps == 0


def retrievability(elapsed_days: float, stability: float) -> float:
    if stability <= 0:
        return 0.0
    return float((1 + FACTOR * elapsed_days / stability) ** DECAY)


def interval_days(stability: float, retention: float = DESIRED_RETENTION) -> float:
    i = stability / FACTOR * (retention ** (1 / DECAY) - 1)
    return min(max(i, 1 / 24), MAX_INTERVAL_DAYS)  # at least one hour


def _d0(g: Rating) -> float:
    return min(10.0, max(1.0, W[4] - math.exp(W[5] * (g - 1)) + 1))


def _next_d(d: float, g: Rating) -> float:
    delta = -W[6] * (g - 3)
    d2 = d + delta * (10 - d) / 9  # linear damping (FSRS-5)
    d3 = W[7] * _d0(Rating.EASY) + (1 - W[7]) * d2  # mean reversion
    return min(10.0, max(1.0, d3))


def _recall_s(d: float, s: float, r: float, g: Rating) -> float:
    hard = W[15] if g == Rating.HARD else 1.0
    easy = W[16] if g == Rating.EASY else 1.0
    return s * (1 + math.exp(W[8]) * (11 - d) * s ** (-W[9]) * (math.exp(W[10] * (1 - r)) - 1) * hard * easy)


def _forget_s(d: float, s: float, r: float) -> float:
    sf = W[11] * d ** (-W[12]) * ((s + 1) ** W[13] - 1) * math.exp(W[14] * (1 - r))
    return min(sf, s)


def _short_term_s(s: float, g: Rating) -> float:
    return s * math.exp(W[17] * (g - 3 + W[18]))


def review(card: Card, g: Rating, now: datetime) -> Card:
    if card.is_new or card.last_review is None:
        s, d = W[g - 1], _d0(g)
    else:
        elapsed = (now - card.last_review).total_seconds() / 86400
        if elapsed < 1:  # same-day review
            s = _short_term_s(card.stability, g)
        else:
            r = retrievability(elapsed, card.stability)
            s = (
                _forget_s(card.difficulty, card.stability, r)
                if g == Rating.AGAIN
                else _recall_s(card.difficulty, card.stability, r, g)
            )
        d = _next_d(card.difficulty, g)
    s = max(s, 0.01)
    ivl = interval_days(s) if g != Rating.AGAIN else min(interval_days(s), 10 / 1440)  # relearn in 10 min
    return replace(
        card,
        stability=s,
        difficulty=d,
        reps=card.reps + 1,
        lapses=card.lapses + (1 if g == Rating.AGAIN and not card.is_new else 0),
        last_review=now,
        due=now + timedelta(days=ivl),
    )
