"""Persistence: users, attempts and per-phone statistics (PostgreSQL in prod, SQLite in dev)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .config import settings


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    l1: Mapped[str] = mapped_column(String(32), default="arz")  # Egyptian Arabic
    gender: Mapped[str] = mapped_column(String(8), default="male")
    # Lobanov/Nearey parameters from the vowel calibration session
    speaker_norm: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # type: ignore[type-arg]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Attempt(Base):
    __tablename__ = "attempts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    text: Mapped[str] = mapped_column(String(1000))
    mode: Mapped[str] = mapped_column(String(32), default="repeat")
    total: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    fluency: Mapped[float | None] = mapped_column(Float, nullable=True)
    prosody: Mapped[float | None] = mapped_column(Float, nullable=True)
    result: Mapped[dict] = mapped_column(JSON)  # type: ignore[type-arg]
    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    model: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class PhoneStat(Base):
    """Running per-phone/context statistics; the adaptive engine reads these."""

    __tablename__ = "phone_stats"
    __table_args__ = (UniqueConstraint("user_id", "phone", "position"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    phone: Mapped[str] = mapped_column(String(4))
    position: Mapped[str] = mapped_column(String(8))  # initial / medial / final
    n: Mapped[int] = mapped_column(Integer, default=0)
    n_errors: Mapped[int] = mapped_column(Integer, default=0)
    mean_score: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class SkillStat(Base):
    """Adaptive-training state per user × skill (error-taxonomy skill): BKT + FSRS card."""

    __tablename__ = "skill_stats"
    __table_args__ = (UniqueConstraint("user_id", "skill"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    skill: Mapped[str] = mapped_column(String(32))
    p_known: Mapped[float] = mapped_column(Float)
    n_obs: Mapped[int] = mapped_column(Integer, default=0)
    n_correct: Mapped[int] = mapped_column(Integer, default=0)
    stability: Mapped[float] = mapped_column(Float, default=0.0)
    difficulty: Mapped[float] = mapped_column(Float, default=0.0)
    reps: Mapped[int] = mapped_column(Integer, default=0)
    lapses: Mapped[int] = mapped_column(Integer, default=0)
    last_review: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    due: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SkillObservation(Base):
    """One opportunity to show a skill (for trends / heatmaps over time)."""

    __tablename__ = "skill_observations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    attempt_id: Mapped[int] = mapped_column(ForeignKey("attempts.id"), index=True)
    skill: Mapped[str] = mapped_column(String(32), index=True)
    word: Mapped[str] = mapped_column(String(64))
    correct: Mapped[bool] = mapped_column()
    score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


engine = create_async_engine(settings.database_url, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
