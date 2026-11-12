"""Speech I/O services: reference TTS (Piper, General American voice) and ASR (Whisper)."""

from __future__ import annotations

import io
import os
from functools import lru_cache

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

TTS_VOICE = os.environ.get("AT_TTS_VOICE", "en/en_US/lessac/medium/en_US-lessac-medium")
ASR_MODEL = os.environ.get("AT_ASR_MODEL", "openai/whisper-base.en")
SR = 16000


@lru_cache(maxsize=1)
def _voice():  # type: ignore[no-untyped-def]
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice

    onnx = hf_hub_download("rhasspy/piper-voices", f"{TTS_VOICE}.onnx")
    hf_hub_download("rhasspy/piper-voices", f"{TTS_VOICE}.onnx.json")
    return PiperVoice.load(onnx)


@lru_cache(maxsize=256)
def tts(text: str, rate: float = 1.0) -> tuple[bytes, int]:
    """Synthesize ``text``; returns (wav bytes, sample rate). Cached: reference sentences repeat."""
    from piper import SynthesisConfig

    v = _voice()
    chunks = list(v.synthesize(text, syn_config=SynthesisConfig(length_scale=1.0 / rate)))
    audio = np.concatenate([c.audio_float_array for c in chunks]) if chunks else np.zeros(1, np.float32)
    sr = chunks[0].sample_rate if chunks else 22050
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue(), sr


def tts_16k(text: str) -> NDArray[np.float32]:
    import librosa

    wav, sr = tts(text)
    a, _ = sf.read(io.BytesIO(wav), dtype="float32")
    return np.asarray(librosa.resample(a, orig_sr=sr, target_sr=SR), dtype=np.float32)


@lru_cache(maxsize=1)
def _asr():  # type: ignore[no-untyped-def]
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model=ASR_MODEL, device=-1)


def transcribe(audio: NDArray[np.float32]) -> str:
    """Whisper transcription of 16 kHz mono audio (what the learner *meant* to say)."""
    out = _asr()({"raw": audio.astype(np.float32), "sampling_rate": SR})
    return str(out["text"]).strip()
