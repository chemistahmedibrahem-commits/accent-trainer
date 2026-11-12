"""Prosody: F0, energy, intonation similarity (DTW), lexical/sentence stress, rhythm."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import parselmouth
from numpy.typing import NDArray

HOP = 0.01  # seconds


@dataclass
class ProsodyTrack:
    times: NDArray[np.float64]
    f0: NDArray[np.float64]  # Hz, NaN where unvoiced
    energy_db: NDArray[np.float64]

    @property
    def voiced(self) -> NDArray[np.bool_]:
        return np.isfinite(self.f0)

    def semitones(self, ref_hz: float | None = None) -> NDArray[np.float64]:
        ref = ref_hz if ref_hz is not None else float(np.nanmedian(self.f0)) if self.voiced.any() else 100.0
        return 12.0 * np.log2(self.f0 / ref)

    def to_dict(self) -> dict[str, list[float | None]]:
        return {
            "times": [round(float(t), 3) for t in self.times],
            "f0": [None if not np.isfinite(v) else round(float(v), 1) for v in self.f0],
            "energy_db": [round(float(v), 1) for v in self.energy_db],
        }


def extract(
    audio: NDArray[np.float32],
    sr: int = 16000,
    method: Literal["praat", "pyin"] = "praat",
    fmin: float = 65.0,
    fmax: float = 450.0,
) -> ProsodyTrack:
    """F0 + RMS energy on a 10 ms grid. ``praat`` (autocorrelation) is fast enough for the
    real-time path; ``pyin`` is more robust offline (CREPE can be plugged in the same way)."""
    n = int(np.ceil(len(audio) / sr / HOP))
    times = ((np.arange(n) + 0.5) * HOP).astype(np.float64)
    if method == "praat":
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
        pitch = snd.to_pitch_ac(time_step=HOP, pitch_floor=fmin, pitch_ceiling=fmax)
        f0 = np.array([pitch.get_value_at_time(t) for t in times], dtype=np.float64)
    else:
        import librosa

        f0_raw, _, _ = librosa.pyin(
            audio.astype(np.float64),
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=int(HOP * sr),
            frame_length=1024,
            center=True,
        )
        f0 = np.full(n, np.nan)
        m = min(n, len(f0_raw))
        f0[:m] = f0_raw[:m]
    hop = int(HOP * sr)
    win = int(0.025 * sr)
    pad = np.pad(audio.astype(np.float64), (win // 2, win))
    rms = np.array([np.sqrt(np.mean(pad[i * hop : i * hop + win] ** 2) + 1e-12) for i in range(n)])
    f0[~np.isfinite(f0)] = np.nan
    f0[f0 <= 0] = np.nan
    return ProsodyTrack(times, f0, 20 * np.log10(rms))


# ---------------------------------------------------------------------------
# Intonation similarity
# ---------------------------------------------------------------------------


def _interp_voiced(st: NDArray[np.float64]) -> NDArray[np.float64]:
    ok = np.isfinite(st)
    if ok.sum() < 2:
        return np.zeros(0)
    idx = np.arange(len(st))
    first, last = idx[ok][0], idx[ok][-1]
    seg = st[first : last + 1]
    sub = idx[first : last + 1]
    return np.interp(sub, idx[ok], st[ok]) if not np.isfinite(seg).all() else seg


def dtw_path(a: NDArray[np.float64], b: NDArray[np.float64], band: float = 0.25) -> list[tuple[int, int]]:
    """DTW on (value, delta) features with a Sakoe–Chiba band (fraction of length)."""
    fa = np.stack([a, np.gradient(a) * 5], axis=1)
    fb = np.stack([b, np.gradient(b) * 5], axis=1)
    n, m = len(a), len(b)
    w = max(int(band * max(n, m)), abs(n - m) + 1)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jc = round(i * m / n)
        lo, hi = max(1, jc - w), min(m, jc + w)
        cost = np.linalg.norm(fb[lo - 1 : hi] - fa[i - 1], axis=1)
        for k, j in enumerate(range(lo, hi + 1)):
            D[i, j] = cost[k] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return path[::-1]


@dataclass(frozen=True)
class IntonationScore:
    correlation: float  # Pearson along the DTW path
    rmse_st: float  # semitones, after per-speaker median normalisation
    range_ratio: float  # user pitch range / reference range (flat intonation < 0.6)
    score: float  # 0..100


def intonation_similarity(user: ProsodyTrack, ref: ProsodyTrack) -> IntonationScore | None:
    u = _interp_voiced(user.semitones())
    r = _interp_voiced(ref.semitones())
    if len(u) < 5 or len(r) < 5:
        return None
    # downsample to 20 ms for DTW speed
    u, r = u[::2], r[::2]
    path = dtw_path(u, r)
    ua = np.array([u[i] for i, _ in path])
    ra = np.array([r[j] for _, j in path])
    corr = float(np.corrcoef(ua, ra)[0, 1]) if ua.std() > 1e-6 and ra.std() > 1e-6 else 0.0
    rmse = float(np.sqrt(np.mean((ua - ra) ** 2)))
    rng_u = np.percentile(u, 95) - np.percentile(u, 5)
    rng_r = np.percentile(r, 95) - np.percentile(r, 5)
    ratio = float(rng_u / rng_r) if rng_r > 1e-6 else 1.0
    s_corr = max(0.0, corr) * 100
    s_rmse = 100 * np.exp(-max(0.0, rmse - 1.0) / 3.0)
    s_range = 100 * np.exp(-abs(np.log(max(ratio, 1e-3))) / 0.7)
    return IntonationScore(corr, rmse, ratio, float(0.5 * s_corr + 0.3 * s_rmse + 0.2 * s_range))


# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Syllable:
    word_index: int
    syllable_index: int
    start: float  # vowel nucleus boundaries (s)
    end: float
    stress: int  # lexical stress from the dictionary (0, 1, 2)


@dataclass(frozen=True)
class SyllableProminence:
    syllable: Syllable
    f0_st: float  # mean semitones re speaker median (NaN if unvoiced)
    energy_db: float
    duration: float
    prominence: float  # sum of z-scores across the utterance


def syllable_prominence(track: ProsodyTrack, syllables: Sequence[Syllable]) -> list[SyllableProminence]:
    st = track.semitones()
    raw = []
    for s in syllables:
        m = (track.times >= s.start) & (track.times < s.end)
        if not m.any():
            m = np.abs(track.times - (s.start + s.end) / 2) == np.min(
                np.abs(track.times - (s.start + s.end) / 2)
            )
        f = st[m]
        f0 = float(np.nanmean(f)) if np.isfinite(f).any() else np.nan
        raw.append((f0, float(np.max(track.energy_db[m])), s.end - s.start))
    arr = np.array(raw, dtype=np.float64)

    def z(col: NDArray[np.float64]) -> NDArray[np.float64]:
        ok = np.isfinite(col)
        out = np.zeros_like(col)
        if ok.sum() >= 2 and col[ok].std() > 1e-9:
            out[ok] = (col[ok] - col[ok].mean()) / col[ok].std()
        return out

    prom = z(arr[:, 0]) + z(arr[:, 1]) + z(np.log(arr[:, 2]))
    return [SyllableProminence(s, *raw[k], float(prom[k])) for k, s in enumerate(syllables)]


@dataclass(frozen=True)
class WordStress:
    word_index: int
    expected_syllable: int
    produced_syllable: int
    correct: bool
    margin: float  # prominence(expected) − max prominence(others)


def lexical_stress(prom: Sequence[SyllableProminence]) -> list[WordStress]:
    by_word: dict[int, list[SyllableProminence]] = {}
    for p in prom:
        by_word.setdefault(p.syllable.word_index, []).append(p)
    out = []
    for w, sylls in sorted(by_word.items()):
        if len(sylls) < 2 or not any(p.syllable.stress == 1 for p in sylls):
            continue
        exp = next(p for p in sylls if p.syllable.stress == 1)
        prod = max(sylls, key=lambda p: p.prominence)
        others = [p.prominence for p in sylls if p is not exp]
        out.append(
            WordStress(
                w,
                exp.syllable.syllable_index,
                prod.syllable.syllable_index,
                prod is exp,
                exp.prominence - max(others),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Rhythm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RhythmMetrics:
    npvi_v: float  # normalised PVI of vocalic intervals (Grabe & Low 2002)
    rpvi_c: float  # raw PVI of consonantal intervals (ms)
    percent_v: float  # %V (Ramus et al. 1999)
    delta_c: float  # ΔC (ms)
    varco_v: float  # Dellwo
    score: float  # 0..100 closeness to stress-timed GA targets


GA_NPVI_TARGET = 60.0  # stress-timed English ≈ 55–65; syllable-timed langs ≈ 30–45


def npvi(d: Sequence[float]) -> float:
    if len(d) < 2:
        return float("nan")
    return float(100 * np.mean([abs(a - b) / ((a + b) / 2) for a, b in itertools.pairwise(d)]))


def rhythm(intervals: Sequence[tuple[bool, float]]) -> RhythmMetrics:
    """``intervals``: consecutive (is_vocalic, duration_s) runs — merge adjacent phones of the
    same class before calling (pauses excluded)."""
    v = [d for is_v, d in intervals if is_v]
    c = [d for is_v, d in intervals if not is_v]
    nv = npvi(v)
    rc = float(1000 * np.mean([abs(a - b) for a, b in itertools.pairwise(c)])) if len(c) > 1 else float("nan")
    tot = sum(v) + sum(c)
    pv = 100 * sum(v) / tot if tot > 0 else float("nan")
    dc = float(1000 * np.std(c)) if c else float("nan")
    vv = float(100 * np.std(v) / np.mean(v)) if v else float("nan")
    score = float(100 * np.exp(-(((nv - GA_NPVI_TARGET) / 20.0) ** 2))) if np.isfinite(nv) else float("nan")
    return RhythmMetrics(nv, rc, pv, dc, vv, score)
