from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AT_", env_file=".env", extra="ignore")

    # "wav2vec2" loads the HF phoneme model (needs the `ml` extra + network/cache);
    # "synthetic" is a test double that echoes the canonical target (no real scoring).
    acoustic_backend: Literal["wav2vec2", "synthetic"] = "wav2vec2"
    # Chosen by measurement (README): hears post-vocalic /r/ (95 % vs 3.6 % for the espeak model)
    acoustic_model_id: str = "slplab/wav2vec2-large-robust-L2-english-phoneme-recognition"
    device: str | None = None
    database_url: str = "sqlite+aiosqlite:///./accent_trainer.db"
    audio_dir: str = "./audio_store"
    store_audio: bool = True
    cors_origins: list[str] = ["http://localhost:5173"]
    max_utterance_s: float = 20.0
    # learned phone/utterance scorer trained for the model above ("" → calibrated GOP)
    scorer_path: str = str(Path(__file__).resolve().parents[1] / "models" / "scorer_slplab.joblib")
    # mispronunciation detector trained on L2-ARCTIC (decides which errors are reported)
    mdd_path: str = str(Path(__file__).resolve().parents[1] / "models" / "mdd_slplab.joblib")
    reference_tts: bool = True  # synthesize a GA reference for intonation scoring


settings = Settings()
