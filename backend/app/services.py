"""Glue between the API and the (UI-independent) accent engine."""

from __future__ import annotations

import io
import logging
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from accent_engine import PronunciationEngine
from accent_engine.acoustic import SAMPLE_RATE, AcousticModel, SyntheticAcousticModel
from accent_engine.engine import EngineConfig
from accent_engine.g2p import g2p
from accent_engine.vowel_space import SpeakerNorm

from .config import settings
from .db import Attempt, PhoneStat, User

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _wav2vec2() -> AcousticModel:
    from accent_engine.acoustic import Wav2Vec2PhonemeModel

    log.info("loading %s", settings.acoustic_model_id)
    return Wav2Vec2PhonemeModel(settings.acoustic_model_id, settings.device)


def warmup() -> str:
    if settings.acoustic_backend == "wav2vec2":
        m = _wav2vec2()
        m.posteriors(np.zeros(SAMPLE_RATE, dtype=np.float32))
        return m.name
    return "synthetic"


def engine_for(text: str, gender: str = "male") -> PronunciationEngine:
    cfg = EngineConfig(default_gender="female" if gender == "female" else "male")
    if settings.acoustic_backend == "synthetic":
        # Dev/test double: "hears" the canonical target, so every attempt scores ~perfect.
        return PronunciationEngine(SyntheticAcousticModel([p.phone for p in g2p(text).phones]), cfg)
    return PronunciationEngine(_wav2vec2(), cfg)


def decode_audio(data: bytes) -> NDArray[np.float32]:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.asarray(mono, dtype=np.float32)


def assess(
    audio: NDArray[np.float32],
    text: str,
    norm: dict[str, Any] | None,
    gender: str,
    reference: NDArray[np.float32] | None = None,
) -> dict[str, Any]:
    if len(audio) > settings.max_utterance_s * SAMPLE_RATE:
        raise ValueError("utterance too long")
    sn = (
        SpeakerNorm(
            mean=(float(norm["mean"][0]), float(norm["mean"][1])),
            sd=(float(norm["sd"][0]), float(norm["sd"][1])),
            log_mean=float(norm["log_mean"]),
            n_tokens=int(norm["n_tokens"]),
        )
        if norm
        else None
    )
    return engine_for(text, gender).assess(audio, text, speaker_norm=sn, reference_audio=reference).to_dict()


async def get_or_create_user(db: AsyncSession, name: str) -> User:
    user = (await db.execute(select(User).where(User.name == name))).scalar_one_or_none()
    if user is None:
        user = User(name=name)
        db.add(user)
        await db.flush()
    return user


def save_audio(audio: NDArray[np.float32]) -> str | None:
    if not settings.store_audio:
        return None
    d = Path(settings.audio_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{uuid.uuid4().hex}.flac"
    sf.write(p, audio, SAMPLE_RATE)
    return str(p)


async def record_attempt(
    db: AsyncSession, user: User, text: str, mode: str, result: dict[str, Any], audio: NDArray[np.float32]
) -> Attempt:
    s = result["scores"]
    att = Attempt(
        user_id=user.id,
        text=text,
        mode=mode,
        total=s.get("total"),
        accuracy=s.get("accuracy"),
        fluency=s.get("fluency"),
        prosody=s.get("prosody"),
        result=result,
        audio_path=save_audio(audio),
        model=result.get("model", ""),
    )
    db.add(att)
    error_idx = {e["target_index"] for e in result["errors"] if e["target_index"] is not None}
    tp = g2p(text).phones
    for w in result["words"]:
        for ph in w["phones"]:
            pos = tp[ph["index"]].position if ph["index"] < len(tp) else "medial"
            stat = (
                await db.execute(
                    select(PhoneStat).where(
                        PhoneStat.user_id == user.id,
                        PhoneStat.phone == ph["phone"],
                        PhoneStat.position == pos,
                    )
                )
            ).scalar_one_or_none()
            if stat is None:
                stat = PhoneStat(
                    user_id=user.id, phone=ph["phone"], position=pos, n=0, n_errors=0, mean_score=0.0
                )
                db.add(stat)
            score = ph["score"] if ph["score"] is not None else 0.0
            stat.n += 1
            stat.n_errors += int(ph["index"] in error_idx)
            stat.mean_score += (score - stat.mean_score) / stat.n
    await db.commit()
    return att
