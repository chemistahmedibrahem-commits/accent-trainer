"""Bayesian Knowledge Tracing (Corbett & Anderson 1995), one model per skill."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BKTParams:
    p_init: float = 0.25  # prior that the learner already masters the skill
    p_learn: float = 0.12  # P(unmastered → mastered) per practice opportunity
    p_slip: float = 0.10  # P(error | mastered)
    p_guess: float = 0.25  # P(correct | not mastered)
    p_forget: float = 0.03  # P(mastered → unmastered); BKT+forgetting, errors after "mastery" count


DEFAULT = BKTParams()
MASTERED = 0.95


def update(p_known: float, correct: bool, params: BKTParams = DEFAULT) -> float:
    """Posterior after one observation, then the learning transition."""
    s, g = params.p_slip, params.p_guess
    if correct:
        num = p_known * (1 - s)
        den = num + (1 - p_known) * g
    else:
        num = p_known * s
        den = num + (1 - p_known) * (1 - g)
    post = num / den if den > 0 else p_known
    return post * (1 - params.p_forget) + (1 - post) * params.p_learn


def p_correct(p_known: float, params: BKTParams = DEFAULT) -> float:
    return p_known * (1 - params.p_slip) + (1 - p_known) * params.p_guess
