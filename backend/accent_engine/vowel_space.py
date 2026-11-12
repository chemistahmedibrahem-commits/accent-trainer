"""Speaker normalisation and General American vowel reference.

Reference means: Hillenbrand, Getty, Clark & Wheeler (1995), hVd tokens, adult male and
female talkers (Michigan, General-American-like). Values are per-gender means in Hz.

Normalisation:
* **Lobanov**: z-score per formant per speaker. Reference ellipses are built by
  Lobanov-normalising each gender's table with that table's own across-vowel mean/SD,
  then averaging genders. The user's calibration session must therefore cover the same
  balanced vowel set (the hVd calibration list below) for the two spaces to match.
* **Nearey** (log-mean, "Nearey1"): ln(F) − mean ln(F) over the speaker's tokens.

Reference spread: a fixed isotropic SD (``REF_SD``) in Lobanov units is used until it is
replaced by spreads estimated from native recordings — clearly a placeholder.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# (F1, F2) Hz
HILLENBRAND_MALE: dict[str, tuple[float, float]] = {
    "IY": (342, 2322),
    "IH": (427, 2034),
    "EY": (476, 2089),
    "EH": (580, 1799),
    "AE": (588, 1952),
    "AA": (768, 1333),
    "AO": (652, 997),
    "OW": (497, 910),
    "UH": (469, 1122),
    "UW": (378, 997),
    "AH": (623, 1200),
    "ER": (474, 1379),
}
HILLENBRAND_FEMALE: dict[str, tuple[float, float]] = {
    "IY": (437, 2761),
    "IH": (483, 2365),
    "EY": (536, 2530),
    "EH": (731, 2058),
    "AE": (669, 2349),
    "AA": (936, 1551),
    "AO": (781, 1136),
    "OW": (555, 1035),
    "UH": (519, 1225),
    "UW": (459, 1105),
    "AH": (753, 1426),
    "ER": (523, 1588),
}
CALIBRATION_WORDS: dict[str, str] = {
    "IY": "heed",
    "IH": "hid",
    "EY": "hayed",
    "EH": "head",
    "AE": "had",
    "AA": "hod",
    "AO": "hawed",
    "OW": "hoed",
    "UH": "hood",
    "UW": "who'd",
    "AH": "hud",
    "ER": "heard",
}
REF_SD = 0.35  # Lobanov units, placeholder (see module docstring)
MONOPHTHONGS = tuple(HILLENBRAND_MALE)


def _lobanov_table(table: Mapping[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    arr = np.array(list(table.values()), dtype=np.float64)
    mu, sd = arr.mean(axis=0), arr.std(axis=0, ddof=0)
    return {v: (np.array(f) - mu) / sd for v, f in table.items()}


def reference_lobanov() -> dict[str, np.ndarray]:
    m, f = _lobanov_table(HILLENBRAND_MALE), _lobanov_table(HILLENBRAND_FEMALE)
    return {v: (m[v] + f[v]) / 2 for v in MONOPHTHONGS}


REFERENCE = reference_lobanov()


@dataclass(frozen=True)
class SpeakerNorm:
    """Per-speaker normalisation parameters estimated from a calibration session."""

    mean: tuple[float, float]  # F1, F2 Hz
    sd: tuple[float, float]
    log_mean: float  # Nearey grand mean of ln F1, ln F2
    n_tokens: int

    @classmethod
    def from_tokens(cls, tokens: Iterable[tuple[str, float, float]]) -> SpeakerNorm:
        """``tokens``: (vowel, F1, F2). Vowel-balanced: per-vowel means are averaged so that
        an unbalanced calibration doesn't skew the centre."""
        by_v: dict[str, list[tuple[float, float]]] = {}
        for v, f1, f2 in tokens:
            if np.isfinite(f1) and np.isfinite(f2):
                by_v.setdefault(v, []).append((f1, f2))
        if len(by_v) < 3:
            raise ValueError("calibration needs at least 3 distinct vowels")
        per_v = np.array([np.mean(x, axis=0) for x in by_v.values()])
        mu = per_v.mean(axis=0)
        sd = per_v.std(axis=0, ddof=0)
        n = sum(len(x) for x in by_v.values())
        return cls((float(mu[0]), float(mu[1])), (float(sd[0]), float(sd[1])), float(np.log(per_v).mean()), n)

    @classmethod
    def default(cls, gender: str = "male") -> SpeakerNorm:
        table = HILLENBRAND_MALE if gender == "male" else HILLENBRAND_FEMALE
        return cls.from_tokens((v, f1, f2) for v, (f1, f2) in table.items())

    def lobanov(self, f1: float, f2: float) -> np.ndarray:
        return np.array([(f1 - self.mean[0]) / self.sd[0], (f2 - self.mean[1]) / self.sd[1]])

    def nearey(self, f1: float, f2: float) -> np.ndarray:
        return np.log([f1, f2]) - self.log_mean


@dataclass(frozen=True)
class VowelScore:
    vowel: str
    point: tuple[float, float]  # Lobanov (zF1, zF2)
    target: tuple[float, float]
    distance: float  # Mahalanobis distance to the target vowel (in REF_SD units)
    nearest: str  # nearest reference vowel
    score: float  # 0..100


def score_vowel(
    vowel: str, f1: float, f2: float, norm: SpeakerNorm, candidates: Sequence[str] = MONOPHTHONGS
) -> VowelScore | None:
    if vowel not in REFERENCE or not (np.isfinite(f1) and np.isfinite(f2)):
        return None
    z = norm.lobanov(f1, f2)
    dists = {v: float(np.linalg.norm(z - REFERENCE[v]) / REF_SD) for v in candidates}
    d = dists[vowel]
    # within 1 SD → ~100, 3 SD → ~50, 5 SD → ~15
    score = float(100.0 * np.exp(-(max(0.0, d - 1.0) ** 2) / (2 * 2.4**2)))
    tgt = REFERENCE[vowel]
    return VowelScore(
        vowel,
        (float(z[0]), float(z[1])),
        (float(tgt[0]), float(tgt[1])),
        d,
        min(dists, key=lambda v: dists[v]),
        score,
    )
