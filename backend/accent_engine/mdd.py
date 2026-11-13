"""Mispronunciation detection (MDD): which phones to *report* as errors.

Scoring and detection are different jobs. The speechocean762 regressor (``scorer.py``) predicts
a graded human score; its threshold was tuned on Mandarin-L1 graded labels and suppressed
categorical Arabic-L1 errors (P→B, TH, rhotics) on L2-ARCTIC. This classifier is trained on
L2-ARCTIC's *categorical* annotations (perceived substitution / deletion) and decides what the
learner is told; the diagnosis (what it sounded like) comes from the alignment, or else from
the most likely competitor in the LPR vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .scorer import CANDIDATES, FEATURE_NAMES

MDD_FEATURES: tuple[str, ...] = (*FEATURE_NAMES, "so762_score")
_LPR0 = FEATURE_NAMES.index(f"lpr_vs_{CANDIDATES[0]}")


@dataclass
class MDDModel:
    model: Any
    threshold: float  # "balanced" (F1-optimal on training-speaker OOF predictions)
    meta: dict[str, Any]
    thresholds: dict[str, float] | None = None  # e.g. {"balanced": .., "gentle": ..}

    def threshold_for(self, strictness: str | None) -> float:
        if strictness and self.thresholds and strictness in self.thresholds:
            return float(self.thresholds[strictness])
        return self.threshold

    @classmethod
    def load(cls, path: str | Path) -> MDDModel:
        import joblib

        b = joblib.load(path)
        if tuple(b["feature_names"]) != MDD_FEATURES:
            raise ValueError("MDD model was trained with a different feature set")
        return cls(b["model"], float(b["threshold"]), b.get("meta", {}), b.get("thresholds"))

    def proba(self, X: NDArray[np.float64], so762_scores: NDArray[np.float64]) -> NDArray[np.float64]:
        from threadpoolctl import threadpool_limits

        Xa = np.column_stack([X, so762_scores])
        with threadpool_limits(1):
            return np.asarray(self.model.predict_proba(Xa)[:, 1], dtype=np.float64)


def likely_realisation(x_row: NDArray[np.float64], own_phone: str) -> str:
    """Best competing explanation from the LPR vector: a phone symbol or "<del>"."""
    lprs = np.asarray(x_row[_LPR0 : _LPR0 + len(CANDIDATES) + 1], dtype=np.float64)
    labels = [*CANDIDATES, "<del>"]
    order = np.argsort(lprs)
    for k in order:
        if labels[k] != own_phone:
            return labels[int(k)]
    return "<del>"
