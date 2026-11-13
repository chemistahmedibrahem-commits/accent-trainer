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
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from accent_engine.g2p import g2p
from accent_engine.phonemes import to_ipa
from accent_engine.vad import UtteranceSegmenter
from accent_engine.vowel_space import CALIBRATION_WORDS, REF_SD, REFERENCE, SpeakerNorm
from trainer import exercises
from trainer.planner import plan as make_plan
from trainer.skills import SKILL_LABEL

from . import coach, conversation, reports, services
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
    strictness: str = Form("focused"),
    audio: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    pcm = services.decode_audio(await audio.read())
    u = await services.get_or_create_user(db, user)
    try:
        result = await run_in_threadpool(
            services.assess, pcm, text, u.speaker_norm, u.gender, None, strictness
        )
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


async def _user(db: AsyncSession, name: str) -> User:
    u = (await db.execute(select(User).where(User.name == name))).scalar_one_or_none()
    if u is None:
        raise HTTPException(404, "unknown user")
    return u


@app.get("/api/users/{name}/dashboard")
async def dashboard(name: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    return await reports.dashboard(db, await _user(db, name))


@app.get("/api/users/{name}/plan")
async def study_plan(name: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    from datetime import UTC, datetime

    u = await _user(db, name)
    drills = make_plan(await services.skill_states(db, u), datetime.now(UTC))
    return {
        "drills": [
            {"kind": d.kind, "skill": d.skill, "label": d.label, "items": d.items, "reason": d.reason}
            for d in drills
        ]
    }


@app.get("/api/drills/{skill}")
async def drills(skill: str, position: str | None = None) -> dict[str, Any]:
    if skill not in SKILL_LABEL:
        raise HTTPException(404, "unknown skill")
    return {
        "skill": skill,
        "label": SKILL_LABEL[skill],
        "words": await run_in_threadpool(exercises.words_for, skill, position, 12),
        "minimal_pairs": await run_in_threadpool(exercises.minimal_pairs, skill, 12),
        "tongue_twisters": exercises.TONGUE_TWISTERS.get(skill, []),
        "shadowing": exercises.SHADOWING,
    }


@app.get("/api/tts")
async def tts(text: str = Query(..., max_length=500), rate: float = Query(1.0, ge=0.5, le=1.5)) -> Response:
    from .speech import tts as synth

    try:
        wav, _ = await run_in_threadpool(synth, text, rate)
    except Exception as e:
        raise HTTPException(503, f"TTS unavailable: {e}") from e
    return Response(wav, media_type="audio/wav", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/attempts/{attempt_id}/coach")
async def attempt_coach(attempt_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    att = await db.get(Attempt, attempt_id)
    if att is None:
        raise HTTPException(404, "unknown attempt")
    u = await db.get(User, att.user_id)
    assert u is not None
    dash = await reports.dashboard(db, u)
    return await run_in_threadpool(coach.coach, att.result, dash["weaknesses"])


@app.get("/api/conversation/scenarios")
async def scenarios() -> dict[str, Any]:
    return {k: {"title": v["title"], "opener": v["script"][0]} for k, v in conversation.SCENARIOS.items()}


@app.post("/api/conversation/turn")
async def conversation_turn(
    scenario: str = Form("small_talk"),
    history: str = Form("[]"),
    user: str = Form("default"),
    audio: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    import json as _json

    from .speech import transcribe

    pcm = services.decode_audio(await audio.read())
    transcript = await run_in_threadpool(transcribe, pcm)
    hist = _json.loads(history)
    if not transcript:
        return {
            "transcript": "",
            "reply": {"text": "Sorry, I didn't catch that. Could you say it again?", "source": "script"},
        }
    u = await services.get_or_create_user(db, user)
    try:
        result = await run_in_threadpool(services.assess, pcm, transcript, u.speaker_norm, u.gender)
        att = await services.record_attempt(db, u, transcript, "conversation", result, pcm)
        feedback = await run_in_threadpool(coach.coach, result, None)
    except ValueError:
        result, att, feedback = None, None, None
    rep = await run_in_threadpool(conversation.reply, scenario, hist, transcript)
    return {
        "transcript": transcript,
        "reply": rep,
        "attempt_id": att.id if att else None,
        "assessment": result,
        "feedback": feedback,
    }


@app.get("/api/users/{name}/report.{fmt}")
async def report(name: str, fmt: str, db: AsyncSession = Depends(get_db)) -> Response:
    u = await _user(db, name)
    dash = await reports.dashboard(db, u, weeks=52)
    atts = list(
        (
            await db.execute(select(Attempt).where(Attempt.user_id == u.id).order_by(Attempt.created_at))
        ).scalars()
    )
    if fmt == "csv":
        return Response(
            reports.to_csv(dash, atts),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="accent-report-{name}.csv"'},
        )
    if fmt == "pdf":
        pdf = await run_in_threadpool(reports.to_pdf, dash, atts)
        return Response(
            pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="accent-report-{name}.pdf"'},
        )
    raise HTTPException(404, "format must be csv or pdf")


# --------------------------------------------------------------------------- WebSocket
#
# Protocol (client → server):
#   {"type": "start", "text": "...", "user": "name", "mode": "repeat", "strictness": "focused"|"balanced"|"gentle"}   (text optional until speech ends)
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
    strictness = "focused"
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
                result = await run_in_threadpool(
                    services.assess, audio, text, u.speaker_norm, u.gender, None, strictness
                )
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
                    strictness = data.get("strictness") or strictness
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
