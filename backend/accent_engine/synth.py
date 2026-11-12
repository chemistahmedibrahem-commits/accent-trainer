"""Tiny Klatt-style source–filter synthesizer.

Used to generate signals with *known* formants / F0 / durations, so the formant, prosody
and VAD modules can be tested against ground truth without recorded speech.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

SR = 16000


def _resonator(x: NDArray[np.float64], f: float, bw: float, sr: int) -> NDArray[np.float64]:
    r = np.exp(-np.pi * bw / sr)
    c = -(r**2)
    b = 2 * r * np.cos(2 * np.pi * f / sr)
    a = 1 - b - c
    return np.asarray(lfilter([a], [1, -b, -c], x), dtype=np.float64)


def glottal_source(
    duration: float, f0: float | Callable[[NDArray[np.float64]], NDArray[np.float64]], sr: int = SR
) -> NDArray[np.float64]:
    """Impulse train (with -12 dB/oct spectral tilt) following an F0 value or contour fn(t)."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    f = np.full(n, f0, dtype=np.float64) if not callable(f0) else np.asarray(f0(t), dtype=np.float64)
    phase = np.cumsum(f / sr)
    pulses = np.zeros(n)
    idx = np.nonzero(np.diff(np.floor(phase)) > 0)[0] + 1
    pulses[idx] = 1.0
    # two first-order lowpasses ≈ -12 dB/oct glottal tilt
    src = lfilter([1.0], [1.0, -0.97], pulses)
    return np.asarray(lfilter([1.0], [1.0, -0.97], src), dtype=np.float64)


def vowel(
    formants: Sequence[float],
    duration: float = 0.3,
    f0: float | Callable[[NDArray[np.float64]], NDArray[np.float64]] = 120.0,
    bandwidths: Sequence[float] | None = None,
    amplitude: float = 0.5,
    sr: int = SR,
) -> NDArray[np.float32]:
    # pad to 5 formants: LPC analysis expects ~5 poles below 5–5.5 kHz in real speech
    fs = (
        list(formants)
        + [f for f in (2600.0, 3300.0, 3700.0, 4500.0, 4900.0) if f > max(formants) + 450][
            : max(0, 5 - len(formants))
        ]
    )
    formants = fs
    bws = list(bandwidths) if bandwidths else [60, 90, 120, 150, 200][: len(formants)]
    y = glottal_source(duration, f0, sr)
    for f, bw in zip(formants, bws, strict=False):
        y = _resonator(y, f, bw, sr)
    y = np.diff(y, prepend=0.0)  # lip radiation (+6 dB/oct)
    ramp = min(int(0.02 * sr), len(y) // 4)
    if ramp > 0:
        env = np.ones(len(y))
        env[:ramp] = np.linspace(0, 1, ramp)
        env[-ramp:] = np.linspace(1, 0, ramp)
        y *= env
    y = y / (np.max(np.abs(y)) + 1e-12) * amplitude
    return y.astype(np.float32)


def silence(duration: float, noise_db: float = -70.0, seed: int = 0, sr: int = SR) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(duration * sr)) * 10 ** (noise_db / 20)).astype(np.float32)


def concat(*parts: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.concatenate(parts).astype(np.float32)
