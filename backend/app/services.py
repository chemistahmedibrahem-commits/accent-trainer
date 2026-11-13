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
from accent_engine.g2p import build_target, g2p, lexicon_for
from accent_engine.vowel_space import SpeakerNorm
from trainer import bkt, fsrs
from trainer.planner import SkillState
from trainer.skills import observations

from .config import settings
from .db import Attempt, PhoneStat, SkillObservation, SkillStat, User

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


def _config(gender: str) -> EngineConfig:
    # the scorer is trained for one acoustic model's features; never pair it with the synthetic double
    use_scorer = (
        settings.acoustic_backend != "synthetic" and bool(settings.scorer_path) and Path(settings.scorer_path).exists()
    )
    return EngineConfig(
        default_gender="female" if gender == "female" else "male",
        scorer_path=settings.scorer_path if use_scorer else None,
    )


@lru_cache(maxsize=2)
def _real_engine(gender: str) -> PronunciationEngine:
    return PronunciationEngine(_wav2vec2(), _config(gender))


def engine_for(text: str, gender: str = "male") -> PronunciationEngine:
    if settings.acoustic_backend == "synthetic":
        # Dev/test double: "hears" the canonical target, so every attempt scores ~perfect.
        return PronunciationEngine(
            SyntheticAcousticModel([p.phone for p in g2p(text).phones]), _config(gender)
        )
    return _real_engine("female" if gender == "female" else "male")


def decode_audio(data: bytes) -> NDArray[np.float32]:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.asarray(mono, dtype=np.float32)


def reference_audio(text: str) -> NDArray[np.float32] | None:
    """Native-like GA reference (TTS) for intonation comparison; None if TTS is unavailable."""
    if not settings.reference_tts:
        return None
    try:
        from .speech import tts_16k

        return tts_16k(text)
    except Exception as e:
        log.warning("reference TTS unavailable: %s", e)
        return None


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
    if reference is None:
        reference = reference_audio(text)
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
    await db.flush()
    error_idx = {e["target_index"] for e in result["errors"] if e["target_index"] is not None}
    # same variant choice the engine made, so indices line up
    tp = build_target(lexicon_for(text), [w.get("variant", 0) for w in result["words"]]).phones
    await update_skills(db, user, att, result, tp)
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


def _aware(d: Any) -> Any:
    from datetime import UTC

    return d.replace(tzinfo=UTC) if d is not None and d.tzinfo is None else d  # SQLite drops tz


def _to_state(row: SkillStat) -> SkillState:
    return SkillState(
        row.skill,
        row.p_known,
        fsrs.Card(
            row.stability, row.difficulty, row.reps, row.lapses, _aware(row.last_review), _aware(row.due)
        ),
        row.n_obs,
    )


async def skill_states(db: AsyncSession, user: User) -> list[SkillState]:
    rows = (await db.execute(select(SkillStat).where(SkillStat.user_id == user.id))).scalars().all()
    return [_to_state(r) for r in rows]


async def update_skills(db: AsyncSession, user: User, att: Attempt, result: dict[str, Any], tp: Any) -> None:
    """BKT update per opportunity; one FSRS review per skill per attempt (graded by mean score)."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    obs = observations(result, tp)
    by_skill: dict[str, list[Any]] = {}
    for o in obs:
        by_skill.setdefault(o.skill, []).append(o)
        db.add(
            SkillObservation(
                user_id=user.id,
                attempt_id=att.id,
                skill=o.skill,
                word=o.word[:64],
                correct=o.correct,
                score=o.score,
            )
        )
    for skill, items in by_skill.items():
        row = (
            await db.execute(select(SkillStat).where(SkillStat.user_id == user.id, SkillStat.skill == skill))
        ).scalar_one_or_none()
        if row is None:
            row = SkillStat(
                user_id=user.id,
                skill=skill,
                p_known=bkt.DEFAULT.p_init,
                n_obs=0,
                n_correct=0,
                stability=0.0,
                difficulty=0.0,
                reps=0,
                lapses=0,
            )
            db.add(row)
        st = _to_state(row)
        for o in items:
            st.observe(o.correct)
        st.review(sum(o.score for o in items) / len(items), now)
        row.p_known, row.n_obs = st.p_known, st.n_obs
        row.n_correct = (row.n_correct or 0) + sum(o.correct for o in items)
        c = st.card
        row.stability, row.difficulty, row.reps, row.lapses = c.stability, c.difficulty, c.reps, c.lapses
        row.last_review, row.due = c.last_review, c.due
