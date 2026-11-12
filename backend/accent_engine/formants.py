"""Formant measurement with Praat (parselmouth) over vowel midpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import parselmouth
from numpy.typing import NDArray


@dataclass(frozen=True)
class FormantMeasure:
    f1: float
    f2: float
    f3: float
    start: float
    end: float
    n_samples: int  # number of formant frames used

    @property
    def valid(self) -> bool:
        return self.n_samples > 0 and bool(np.isfinite([self.f1, self.f2]).all())


class FormantTracker:
    """Runs Burg LPC once per utterance; individual vowels are then sampled cheaply.

    ``max_formant``: Praat ceiling — ~5000 Hz for adult male voices, ~5500 for female.
    The calibration session can pick the per-speaker optimum (Escudero et al. 2009).
    """

    def __init__(
        self,
        audio: NDArray[np.float32],
        sr: int = 16000,
        max_formant: float = 5500.0,
        time_step: float = 0.005,
    ) -> None:
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
        self._formant = snd.to_formant_burg(
            time_step=time_step,
            max_number_of_formants=5,
            maximum_formant=max_formant,
            window_length=0.025,
            pre_emphasis_from=50.0,
        )
        self.time_step = time_step

    def measure(self, start: float, end: float, portion: float = 0.4) -> FormantMeasure:
        """Median F1–F3 over the central ``portion`` of [start, end] (the steady state)."""
        dur = end - start
        mid = (start + end) / 2
        lo, hi = mid - dur * portion / 2, mid + dur * portion / 2
        times = np.arange(lo, hi + 1e-9, self.time_step)
        if len(times) == 0:
            times = np.array([mid])
        vals = np.array([[self._formant.get_value_at_time(k, t) for k in (1, 2, 3)] for t in times])
        ok = np.isfinite(vals[:, 0]) & np.isfinite(vals[:, 1])
        if not ok.any():
            return FormantMeasure(np.nan, np.nan, np.nan, start, end, 0)
        med = np.nanmedian(vals[ok], axis=0)
        return FormantMeasure(float(med[0]), float(med[1]), float(med[2]), start, end, int(ok.sum()))
