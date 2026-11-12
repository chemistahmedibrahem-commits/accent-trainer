"""Learned phone scorer on top of GOP features (trained on speechocean762 *train*).

Raw GOP correlates only moderately with human ratings. Following GOPT-style systems, we
give a regressor a rich per-phone vector:

* the full **LPR vector** — log-likelihood ratio of the canonical sequence against
  substituting the phone with *every* other phone class, and against deleting it (GOP-AF);
* posterior statistics over the aligned segment (mean/max canonical posterior, non-blank mass);
* duration, phone identity (categorical), vowel/stress/position/allophone, and the same
  key features for the neighbouring phones (context).

``scripts/eval_speechocean.py train-scorer`` fits a HistGradientBoostingRegressor on these and
saves it with joblib; :class:`LearnedScorer` loads it into the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .acoustic import BLANK_ID, PHONE_TO_ID, Posteriors
from .ctc import Segment, sequence_log_likelihood
from .g2p import TargetPhone
from .phonemes import PHONES, is_vowel

CANDIDATES: tuple[str, ...] = tuple(p for p in PHONES if p != "Q")
CLIP = 30.0
_POS = {"initial": 0, "medial": 1, "final": 2}
_ALLO = {None: 0, "flap": 1, "glottal": 2, "dark_l": 3}


def lpr_matrix(
    post: Posteriors, targets: Sequence[TargetPhone], segments: Sequence[Segment], window: int = 2
) -> NDArray[np.float64]:
    """(L, len(CANDIDATES)+1): log P(canonical) − log P(phone i → candidate | deleted)."""
    lp = post.log_probs
    T = lp.shape[0]
    base = [PHONE_TO_ID[t.phone] for t in targets]
    L = len(base)
    out = np.zeros((L, len(CANDIDATES) + 1))
    for i, t in enumerate(targets):
        lo, hi = max(0, i - window), min(L, i + window + 1)
        local = base[lo:hi]
        j = i - lo
        f_lo = max(0, segments[lo].start - 2)
        f_hi = min(T, segments[hi - 1].end + 2)
        if f_hi - f_lo < 2 * len(local) + 1:
            f_lo, f_hi = max(0, f_lo - 2 * len(local)), min(T, f_hi + 2 * len(local))
        seqs = [local]
        acc = [0]
        for alt in sorted(t.alternatives):
            s = list(local)
            s[j] = PHONE_TO_ID[alt]
            seqs.append(s)
            acc.append(len(seqs) - 1)
        first_cand = len(seqs)
        for c in CANDIDATES:
            s = list(local)
            s[j] = PHONE_TO_ID[c]
            seqs.append(s)
        seqs.append(local[:j] + local[j + 1 :] if len(local) > 1 else local)
        ll = sequence_log_likelihood(lp[f_lo:f_hi], seqs, BLANK_ID)
        canon = max(ll[a] for a in acc)
        out[i] = np.clip(canon - ll[first_cand:], -CLIP, CLIP)
        if len(local) == 1:
            out[i, -1] = CLIP
    return out


def phone_features(
    post: Posteriors,
    targets: Sequence[TargetPhone],
    segments: Sequence[Segment],
    lpr: Sequence[float],
    gop_frame: Sequence[float],
    hard_err: Sequence[bool],
    matched: Sequence[bool],
) -> NDArray[np.float64]:
    lp = post.log_probs
    probs = np.exp(lp)
    nonblank = 1 - probs[:, BLANK_ID]
    M = lpr_matrix(post, targets, segments)
    L = len(targets)
    own_min = np.array(
        [np.min(np.where(np.array(CANDIDATES) == t.phone, CLIP, M[i, :-1])) for i, t in enumerate(targets)]
    )
    rows = []
    for i, t in enumerate(targets):
        sg = segments[i]
        fr = slice(sg.start, max(sg.end, sg.start + 1))
        pid = PHONE_TO_ID[t.phone]
        p_c = probs[fr, pid]
        nb = nonblank[fr]
        base = [
            float(np.clip(lpr[i], -CLIP, CLIP)),
            float(np.clip(gop_frame[i], -CLIP, 0)),
            own_min[i],
            M[i, -1],
            float(p_c.mean()),
            float(p_c.max()),
            float(nb.mean()),
            float(np.log1p(sg.end - sg.start)),
            float(pid),
            float(is_vowel(t.phone)),
            float(-1 if t.stress is None else t.stress),
            float(_POS[t.position]),
            float(_ALLO.get(t.allophone, 0)),
            float(hard_err[i]),
            float(matched[i]),
        ]
        ctx = []
        for k in (i - 1, i + 1):
            if 0 <= k < L:
                ctx += [float(np.clip(lpr[k], -CLIP, CLIP)), own_min[k], float(hard_err[k])]
            else:
                ctx += [CLIP, CLIP, 0.0]
        word = [k for k in range(L) if targets[k].word_index == t.word_index]
        ctx.append(float(np.mean([np.clip(lpr[k], -CLIP, CLIP) for k in word])))
        rows.append(base + ctx + list(M[i]))
    return np.asarray(rows, dtype=np.float64)


FEATURE_NAMES: tuple[str, ...] = (
    "lpr",
    "gop_frame",
    "lpr_min_sub",
    "lpr_del",
    "post_mean",
    "post_max",
    "nonblank",
    "log_dur",
    "phone_id",
    "is_vowel",
    "stress",
    "position",
    "allophone",
    "hard_err",
    "matched",
    "prev_lpr",
    "prev_min_sub",
    "prev_err",
    "next_lpr",
    "next_min_sub",
    "next_err",
    "word_mean_lpr",
    *(f"lpr_vs_{c}" for c in CANDIDATES),
    "lpr_vs_del",
)
CATEGORICAL = (FEATURE_NAMES.index("phone_id"),)


@dataclass
class LearnedScorer:
    model: Any
    meta: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> LearnedScorer:
        import joblib

        bundle = joblib.load(path)
        if tuple(bundle["feature_names"]) != FEATURE_NAMES:
            raise ValueError("scorer was trained with a different feature set")
        return cls(bundle["model"], bundle.get("meta", {}))

    def score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predicted human phone accuracy (0–2) → 0–100."""
        return np.clip(self.model.predict(X), 0, 2) * 50.0

    @property
    def error_gate(self) -> float:
        """Phones scoring below this are reported as errors (tuned on train for best F1)."""
        return float(self.meta.get("error_gate", 60.0))
