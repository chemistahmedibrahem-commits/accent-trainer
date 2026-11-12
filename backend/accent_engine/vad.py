"""Voice activity detection and streaming utterance segmentation.

``EnergyVAD`` (no dependencies, adaptive noise floor) is the default; ``SileroVAD`` is used
when ``torch`` and the Silero model are available. Both expose ``is_speech(frame) -> prob``
on 32 ms (512-sample) frames at 16 kHz.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FRAME = 512  # samples @16 kHz = 32 ms (Silero's native frame size)
SR = 16000


class VAD(Protocol):
    def speech_prob(self, frame: NDArray[np.float32]) -> float: ...
    def reset(self) -> None: ...


class EnergyVAD:
    def __init__(self, margin_db: float = 12.0, floor_db: float = -60.0) -> None:
        self.margin_db = margin_db
        self.init_floor = floor_db
        self.noise_db = floor_db

    def reset(self) -> None:
        self.noise_db = self.init_floor

    def speech_prob(self, frame: NDArray[np.float32]) -> float:
        db = 10 * np.log10(np.mean(frame.astype(np.float64) ** 2) + 1e-12)
        # track noise floor: fast down, slow up
        self.noise_db = db if db < self.noise_db else self.noise_db + 0.02 * (db - self.noise_db)
        self.noise_db = max(self.noise_db, -90.0)
        x = db - (self.noise_db + self.margin_db)
        return float(1 / (1 + np.exp(-x / 2.0)))


class SileroVAD:  # pragma: no cover - needs torch + model download
    def __init__(self) -> None:
        import torch

        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self._torch = torch

    def reset(self) -> None:
        self.model.reset_states()

    def speech_prob(self, frame: NDArray[np.float32]) -> float:
        return float(self.model(self._torch.from_numpy(frame.astype(np.float32)), SR).item())


def make_vad(prefer_silero: bool = True) -> VAD:
    if prefer_silero:
        try:
            return SileroVAD()
        except Exception:
            pass
    return EnergyVAD()


@dataclass
class VadEvent:
    kind: str  # "speech_start" | "speech_end" | "level"
    time: float
    audio: NDArray[np.float32] | None = None
    level_db: float | None = None
    speech_prob: float | None = None


@dataclass
class UtteranceSegmenter:
    """Feed arbitrary-size PCM chunks; emits utterances bounded by VAD + hangover.

    * ``start_frames`` consecutive speech frames open an utterance (with ``preroll_s``
      of audio kept from before onset),
    * ``end_silence_s`` of non-speech closes it (that silence is trimmed to 150 ms),
    * utterances shorter than ``min_speech_s`` are dropped, longer than ``max_s`` are cut.
    """

    vad: VAD = field(default_factory=EnergyVAD)
    threshold: float = 0.5
    start_frames: int = 3
    end_silence_s: float = 0.5
    preroll_s: float = 0.3
    min_speech_s: float = 0.25
    max_s: float = 30.0

    def __post_init__(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._pre: deque[NDArray[np.float32]] = deque(maxlen=max(1, int(self.preroll_s * SR / FRAME)))
        self._buf: list[NDArray[np.float32]] = []
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._speech_frames = 0
        self._t = 0  # samples consumed

    def push(self, pcm: NDArray[np.float32]) -> list[VadEvent]:
        events: list[VadEvent] = []
        self._pending = np.concatenate([self._pending, pcm.astype(np.float32)])
        while len(self._pending) >= FRAME:
            frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
            self._t += FRAME
            p = self.vad.speech_prob(frame)
            now = self._t / SR
            is_sp = p >= self.threshold
            if not self._in_speech:
                self._pre.append(frame)
                self._speech_run = self._speech_run + 1 if is_sp else 0
                if self._speech_run >= self.start_frames:
                    self._in_speech = True
                    self._buf = list(self._pre)
                    self._speech_frames = self._speech_run
                    self._silence_run = 0
                    events.append(VadEvent("speech_start", now - len(self._buf) * FRAME / SR))
            else:
                self._buf.append(frame)
                if is_sp:
                    self._silence_run = 0
                    self._speech_frames += 1
                else:
                    self._silence_run += 1
                ended = self._silence_run * FRAME / SR >= self.end_silence_s
                if ended or len(self._buf) * FRAME / SR >= self.max_s:
                    keep = len(self._buf) - max(0, self._silence_run - int(0.15 * SR / FRAME))
                    audio = np.concatenate(self._buf[:keep])
                    if self._speech_frames * FRAME / SR >= self.min_speech_s:
                        events.append(VadEvent("speech_end", now, audio=audio))
                    self._in_speech = False
                    self._speech_run = 0
                    self._buf = []
                    self._pre.clear()
        return events

    def flush(self) -> list[VadEvent]:
        """Force-close an open utterance (e.g. client pressed stop)."""
        if self._in_speech and self._buf and self._speech_frames * FRAME / SR >= self.min_speech_s:
            audio = np.concatenate(self._buf)
            self._in_speech = False
            self._buf = []
            return [VadEvent("speech_end", self._t / SR, audio=audio)]
        self._in_speech = False
        self._buf = []
        return []
