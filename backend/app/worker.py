"""arq worker for heavy offline analysis (pYIN re-analysis, batch re-scoring, reports).

arq app.worker.WorkerSettings
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import soundfile as sf
from arq.connections import RedisSettings  # type: ignore[import-not-found]

from accent_engine.engine import EngineConfig, PronunciationEngine

from .services import _wav2vec2


async def rescore_offline(ctx: dict[str, Any], audio_path: str, text: str) -> dict[str, Any]:
    """Slower, more robust analysis than the real-time path (pYIN pitch)."""
    audio, _ = sf.read(audio_path, dtype="float32")
    eng = PronunciationEngine(_wav2vec2(), EngineConfig(pitch_method="pyin"))
    return eng.assess(audio, text).to_dict()


class WorkerSettings:
    functions: ClassVar[list[Any]] = [rescore_offline]
    redis_settings = RedisSettings.from_dsn(os.environ.get("AT_REDIS_URL", "redis://redis:6379"))
