"""FastAPI app: REST for content/assessment/progress, WebSocket for live streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from accent_engine.g2p import g2p
from accent_engine.phonemes import to_ipa
from accent_engine.vad import UtteranceSegmenter
from accent_engine.vowel_space import CALIBRATION_WORDS, REF_SD, REFERENCE, SpeakerNorm

from . import services
from .config import settings
from .content import MINIMAL_PAIRS, SENTENCES
from .db import Attempt, PhoneStat, SessionLocal, User, init_db

log = logging.getLogger("accent_trainer")


@asynccontextmanager
async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
    await init_db()
    try:
        app.state.model = await run_in_threadpool(services.warmup)
    except Exception as e:
        log.exception("acoustic model failed to load")
        app.state.model = f"unavailable: {e}"
    yield


app = FastAPI(title="American Accent Trainer", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
)


async def get_db():  # type: ignore[no-untyped-def]
    async with SessionLocal() as s:
        yield s


# --------------------------------------------------------------------------- REST


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "backend": settings.acoustic_backend, "model": getattr(app.state, "model", None)}


@app.get("/api/g2p")
async def api_g2p(text: str = Query(..., max_length=500)) -> dict[str, Any]:
    t = g2p(text)
    return {
        "words": [
            {
                "text": w,
                "phones": [
                    {
                        "phone": p.phone,
                        "ipa": to_ipa(p.phone, p.stress),
                        "stress": p.stress,
                        "allophone": p.allophone,
                        "position": p.position,
                    }
                    for p in t.phones
                    if p.word_index == wi
                ],
            }
            for wi, w in enumerate(t.words)
        ]
    }


@app.get("/api/content")
async def content() -> dict[str, Any]:
    return {"minimal_pairs": MINIMAL_PAIRS, "sentences": SENTENCES, "calibration": CALIBRATION_WORDS}


@app.get("/api/vowel-reference")
async def vowel_reference() -> dict[str, Any]:
    return {
        "space": "lobanov",
        "sd": REF_SD,
        "vowels": {
            v: {"z_f1": float(z[0]), "z_f2": float(z[1]), "ipa": to_ipa(v)} for v, z in REFERENCE.items()
        },
    }


@app.post("/api/assess")
async def api_assess(
    text: str = Form(...),
    user: str = Form("default"),
    mode: str = Form("repeat"),
    audio: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    pcm = services.decode_audio(await audio.read())
    u = await services.get_or_create_user(db, user)
    try:
        result = await run_in_threadpool(services.assess, pcm, text, u.speaker_norm, u.gender)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    att = await services.record_attempt(db, u, text, mode, result, pcm)
    return {"attempt_id": att.id, **result}


class CalibrationIn(BaseModel):
    user: str = "default"
    tokens: list[tuple[str, float, float]]  # (vowel, F1, F2) measured from hVd words
    gender: str | None = None


@app.post("/api/calibration")
async def calibration(body: CalibrationIn, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    try:
        norm = SpeakerNorm.from_tokens(body.tokens)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    u = await services.get_or_create_user(db, body.user)
    u.speaker_norm = {
        "mean": list(norm.mean),
        "sd": list(norm.sd),
        "log_mean": norm.log_mean,
        "n_tokens": norm.n_tokens,
    }
    if body.gender:
        u.gender = body.gender
    await db.commit()
    return {"user": u.name, "speaker_norm": u.speaker_norm}


@app.get("/api/users/{name}/progress")
async def progress(name: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    u = (await db.execute(select(User).where(User.name == name))).scalar_one_or_none()
    if u is None:
        raise HTTPException(404, "unknown user")
    stats = (await db.execute(select(PhoneStat).where(PhoneStat.user_id == u.id))).scalars().all()
    attempts = (
        (
            await db.execute(
                select(Attempt).where(Attempt.user_id == u.id).order_by(Attempt.created_at.desc()).limit(50)
            )
        )
        .scalars()
        .all()
    )
    weak = sorted((s for s in stats if s.n >= 3), key=lambda s: s.mean_score)[:5]
    return {
        "user": u.name,
        "calibrated": u.speaker_norm is not None,
        "phones": [
            {
                "phone": s.phone,
                "position": s.position,
                "n": s.n,
                "error_rate": s.n_errors / s.n,
                "mean_score": s.mean_score,
            }
            for s in stats
        ],
        "weaknesses": [{"phone": s.phone, "position": s.position, "mean_score": s.mean_score} for s in weak],
        "attempts": [
            {
                "id": a.id,
                "text": a.text,
                "mode": a.mode,
                "total": a.total,
                "accuracy": a.accuracy,
                "fluency": a.fluency,
                "prosody": a.prosody,
                "created_at": a.created_at.isoformat(),
            }
            for a in attempts
        ],
    }


# --------------------------------------------------------------------------- WebSocket
#
# Protocol (client → server):
#   {"type": "start", "text": "...", "user": "name", "mode": "repeat"}   (text optional until speech ends)
#   binary frames: little-endian float32 or int16 PCM, mono, 16 kHz, any chunk size (20–40 ms typical)
#   {"type": "format", "dtype": "int16" | "float32"}
#   {"type": "stop"}  → force end of utterance
# Server → client:
#   {"type": "ready"} · {"type": "level", "db": .., "speech_prob": ..} (~every 100 ms)
#   {"type": "speech_start", "t": ..} · {"type": "speech_end", "t": .., "duration": ..}
#   {"type": "result", "latency_ms": .., "attempt_id": .., ...assessment} · {"type": "error", "message": ..}


@app.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    seg = UtteranceSegmenter()
    text: str | None = None
    user_name = "default"
    mode = "repeat"
    dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32
    level_every = 3  # frames of 32 ms → ~100 ms
    n_frames = 0
    pending: set[asyncio.Task[None]] = set()

    async def process(audio: np.ndarray, t_end: float) -> None:  # type: ignore[type-arg]
        if not text:
            await ws.send_json({"type": "error", "message": "no target text set"})
            return
        t0 = time.perf_counter()
        try:
            async with SessionLocal() as db:
                u = await services.get_or_create_user(db, user_name)
                result = await run_in_threadpool(services.assess, audio, text, u.speaker_norm, u.gender)
                att = await services.record_attempt(db, u, text, mode, result, audio)
            await ws.send_json(
                {
                    "type": "result",
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                    "attempt_id": att.id,
                    **result,
                }
            )
        except Exception as e:
            log.exception("assessment failed")
            await ws.send_json({"type": "error", "message": str(e)})

    def schedule(audio: np.ndarray, t: float) -> None:  # type: ignore[type-arg]
        task = asyncio.create_task(process(audio, t))
        pending.add(task)
        task.add_done_callback(pending.discard)

    await ws.send_json({"type": "ready", "sample_rate": 16000})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                raw = msg["bytes"]
                pcm = np.frombuffer(raw, dtype=dtype)
                if dtype is np.int16:
                    pcm = pcm.astype(np.float32) / 32768.0
                for ev in seg.push(pcm.astype(np.float32)):
                    if ev.kind == "speech_start":
                        await ws.send_json({"type": "speech_start", "t": ev.time})
                    elif ev.kind == "speech_end" and ev.audio is not None:
                        await ws.send_json(
                            {"type": "speech_end", "t": ev.time, "duration": len(ev.audio) / 16000}
                        )
                        schedule(ev.audio, ev.time)
                n_frames += 1
                if n_frames % level_every == 0 and len(pcm):
                    db_ = float(10 * np.log10(np.mean(pcm.astype(np.float64) ** 2) + 1e-12))
                    await ws.send_json({"type": "level", "db": round(db_, 1)})
            elif msg.get("text") is not None:
                data = json.loads(msg["text"])
                kind = data.get("type")
                if kind == "start":
                    text = data.get("text") or text
                    user_name = data.get("user") or user_name
                    mode = data.get("mode") or mode
                    seg = UtteranceSegmenter()
                elif kind == "format":
                    dtype = np.int16 if data.get("dtype") == "int16" else np.float32
                elif kind == "text":
                    text = data.get("text")
                elif kind == "stop":
                    for ev in seg.flush():
                        if ev.audio is not None:
                            await ws.send_json(
                                {"type": "speech_end", "t": ev.time, "duration": len(ev.audio) / 16000}
                            )
                            schedule(ev.audio, ev.time)
    except WebSocketDisconnect:
        pass
    finally:
        for t in list(pending):
            t.cancel()
