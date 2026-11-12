"""Goodness of Pronunciation for CTC phone models.

Two complementary per-phone measures:

* **GOP-LPR** (alignment-free, after Cao et al. 2024 "GOP-AF"): the log-likelihood ratio
  between the canonical sequence and the best competing sequence obtained by
  substituting (or deleting) that single phone, computed exactly with the CTC forward
  algorithm. Robust to CTC's peaky posteriors because it never needs phone boundaries.
* **GOP-frame**: classic posterior GOP over the forced-aligned segment, with frames
  weighted by non-blank mass and posteriors renormalised over non-blank tokens.

``score`` maps LPR → 0..100 with a logistic calibration; defaults are *uncalibrated* and
are meant to be refit on speechocean762 (``scripts/eval_speechocean.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .acoustic import BLANK_ID, OTHER_ID, PHONE_TO_ID, VOCAB, Posteriors
from .ctc import Segment, sequence_log_likelihood
from .g2p import TargetPhone

# L1-Arabic-informed confusion sets: always tried as competitors.
CONFUSIONS: dict[str, tuple[str, ...]] = {
    "P": ("B",),
    "B": ("P",),
    "V": ("F", "W", "B"),
    "F": ("V",),
    "TH": ("T", "S", "F"),
    "DH": ("D", "Z", "V"),
    "IH": ("IY", "EH"),
    "IY": ("IH",),
    "UH": ("UW",),
    "UW": ("UH",),
    "AE": ("EH", "AA"),
    "EH": ("AE", "IH", "EY"),
    "AH": ("AA", "AE", "EH", "OW"),
    "ER": ("AH", "EH", "AA"),
    "R": ("L", "W"),
    "DX": ("T", "D"),
    "OW": ("AO", "UW"),
    "AO": ("AA", "OW"),
    "AA": ("AO", "AH", "AE"),
    "EY": ("EH", "IY"),
    "D": ("T",),
    "G": ("K",),
    "Z": ("S",),
    "ZH": ("SH", "JH"),
    "JH": ("ZH", "G", "CH"),
    "CH": ("SH",),
    "NG": ("N", "G"),
    "L": ("R",),
}

EXCLUDED_COMPETITORS = frozenset({"Q"})  # glottal stop only ever accepted, not competed


@dataclass(frozen=True)
class Calibration:
    slope: float = 1.2
    intercept: float = 0.0

    def score(self, lpr: float) -> float:
        z = self.slope * lpr + self.intercept
        return float(100.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))


@dataclass(frozen=True)
class PhoneGOP:
    index: int
    phone: str
    lpr: float
    gop_frame: float
    score: float
    best_competitor: str | None  # phone symbol, "<del>" for deletion, None if nothing tried
    competitor_margin: float  # log P(best competitor) - log P(canonical)


def _competitors(target: TargetPhone, posteriors: Posteriors, seg: Segment | None, top_k: int) -> list[str]:
    accepted = {target.phone, *target.alternatives}
    cands: list[str] = list(CONFUSIONS.get(target.phone, ()))
    if target.underlying != target.phone:
        cands += list(CONFUSIONS.get(target.underlying, ()))
    if seg is not None:
        lo = max(0, seg.start - 2)
        hi = min(posteriors.num_frames, seg.end + 2)
        mass = np.exp(posteriors.log_probs[lo:hi]).sum(axis=0)
        mass[[BLANK_ID, OTHER_ID]] = 0
        for idx in np.argsort(-mass)[: top_k + 2]:
            cands.append(VOCAB[int(idx)])
    out: list[str] = []
    for c in cands:
        if c not in accepted and c not in EXCLUDED_COMPETITORS and c not in out:
            out.append(c)
    return out


def _position_batch(
    base: list[int], j: int, t: TargetPhone, competitors: list[str]
) -> tuple[list[list[int]], list[int], list[tuple[str, int]]]:
    """Sequences for scoring position ``j`` of ``base``: canonical + accepted alternatives
    (``acc`` indices) and single-phone substitutions / deletion (``comp``)."""
    seqs = [base]
    acc = [0]
    for alt in sorted(t.alternatives):
        s = list(base)
        s[j] = PHONE_TO_ID[alt]
        seqs.append(s)
        acc.append(len(seqs) - 1)
    comp: list[tuple[str, int]] = []
    for c in competitors:
        s = list(base)
        s[j] = PHONE_TO_ID[c]
        seqs.append(s)
        comp.append((c, len(seqs) - 1))
    if len(base) > 1:
        seqs.append(base[:j] + base[j + 1 :])
        comp.append(("<del>", len(seqs) - 1))
    return seqs, acc, comp


def compute_gop(
    posteriors: Posteriors,
    targets: Sequence[TargetPhone],
    segments: Sequence[Segment] | None = None,
    calibration: Calibration | None = None,
    top_k: int = 6,
    window: int | None = 2,
) -> list[PhoneGOP]:
    """``window``: score each phone with a CTC forward restricted to ±``window`` neighbouring
    phones and their aligned frames (needs ``segments``). Substituting one phone barely changes
    the likelihood of distant context, so this approximates the full-utterance ratio at
    O(L) instead of O(L²) cost. ``None`` → exact full-sequence computation."""
    cal = calibration or Calibration()
    lp = posteriors.log_probs
    T = lp.shape[0]
    base = [PHONE_TO_ID[t.phone] for t in targets]
    L = len(base)
    windowed = window is not None and segments is not None and 2 * window + 1 < L

    results: list[tuple[float, str | None, float]] = []  # canonical ll, best competitor, its ll
    if not windowed:
        all_seqs: list[list[int]] = []
        plans = []
        for i, t in enumerate(targets):
            comps = _competitors(t, posteriors, segments[i] if segments else None, top_k)
            seqs, acc, comp = _position_batch(base, i, t, comps)
            off = len(all_seqs)
            all_seqs += seqs
            plans.append(([a + off for a in acc], [(c, k + off) for c, k in comp]))
        ll = sequence_log_likelihood(lp, all_seqs, BLANK_ID)
        for acc, comp in plans:
            canon = float(max(ll[a] for a in acc))
            if comp:
                lab, k = max(comp, key=lambda c: ll[c[1]])
                results.append((canon, lab, float(ll[k])))
            else:
                results.append((canon, None, -1e30))
    else:
        assert segments is not None and window is not None
        for i, t in enumerate(targets):
            lo, hi = max(0, i - window), min(L, i + window + 1)
            local = base[lo:hi]
            f_lo = max(0, segments[lo].start - 2)
            f_hi = min(T, segments[hi - 1].end + 2)
            if f_hi - f_lo < 2 * len(local) + 1:  # degenerate alignment → widen
                f_lo, f_hi = max(0, f_lo - 2 * len(local)), min(T, f_hi + 2 * len(local))
            comps = _competitors(t, posteriors, segments[i], top_k)
            seqs, acc, comp = _position_batch(local, i - lo, t, comps)
            ll = sequence_log_likelihood(lp[f_lo:f_hi], seqs, BLANK_ID)
            canon = float(max(ll[a] for a in acc))
            if comp:
                lab, k = max(comp, key=lambda c: ll[c[1]])
                results.append((canon, lab, float(ll[k])))
            else:
                results.append((canon, None, -1e30))

    # --- frame GOP (needs segments)
    probs = np.exp(lp)
    nonblank = 1.0 - probs[:, BLANK_ID]
    renorm = lp.copy()
    renorm[:, BLANK_ID] = -1e30
    renorm -= np.log(np.maximum(nonblank, 1e-12))[:, None]

    out: list[PhoneGOP] = []
    for i, (t, (canon, best_lab, best)) in enumerate(zip(targets, results, strict=True)):
        lpr = canon - best
        gf = 0.0
        if segments is not None:
            sg = segments[i]
            fr = slice(sg.start, max(sg.end, sg.start + 1))
            w = nonblank[fr]
            ids = [PHONE_TO_ID[t.phone], *[PHONE_TO_ID[a] for a in t.alternatives]]
            own = renorm[fr][:, ids].max(axis=1)
            gf = float(np.sum(w * (own - renorm[fr].max(axis=1))) / max(np.sum(w), 1e-9))
        out.append(PhoneGOP(i, t.phone, float(lpr), gf, cal.score(lpr), best_lab, float(best - canon)))
    return out
