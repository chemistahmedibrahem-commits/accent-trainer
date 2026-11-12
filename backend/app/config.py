from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AT_", env_file=".env", extra="ignore")

    # "wav2vec2" loads the HF phoneme model (needs the `ml` extra + network/cache);
    # "synthetic" is a test double that echoes the canonical target (no real scoring).
    acoustic_backend: Literal["wav2vec2", "synthetic"] = "wav2vec2"
    acoustic_model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    device: str | None = None
    database_url: str = "sqlite+aiosqlite:///./accent_trainer.db"
    audio_dir: str = "./audio_store"
    store_audio: bool = True
    cors_origins: list[str] = ["http://localhost:5173"]
    max_utterance_s: float = 20.0
    reference_tts: bool = True  # synthesize a GA reference for intonation scoring


settings = Settings()
