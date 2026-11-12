import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from accent_engine.synth import concat, silence, vowel
from app.main import app


def speech() -> np.ndarray:
    return concat(
        silence(0.4, -60),
        vowel((700, 1220), 0.25, 140),
        silence(0.05, -60),
        vowel((350, 2300), 0.3, 120),
        silence(1.0, -60),
    )


def wav_bytes(a: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, a, sr, format="WAV")
    return buf.getvalue()


def test_health_g2p_content() -> None:
    with TestClient(app) as c:
        assert c.get("/api/health").json()["backend"] == "synthetic"
        words = c.get("/api/g2p", params={"text": "better water"}).json()["words"]
        assert [p["ipa"] for p in words[0]["phones"]] == ["b", "ɛ", "ɾ", "ɚ"]
        assert "P_B" in c.get("/api/content").json()["minimal_pairs"]
        assert "IY" in c.get("/api/vowel-reference").json()["vowels"]


def test_assess_resamples_and_records_progress() -> None:
    import librosa

    a44 = librosa.resample(speech(), orig_sr=16000, target_sr=44100)
    with TestClient(app) as c:
        r = c.post(
            "/api/assess",
            data={"text": "a key", "user": "ahmed"},
            files={"audio": ("x.wav", wav_bytes(a44, 44100), "audio/wav")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["attempt_id"] > 0 and body["scores"]["total"] is not None
        prog = c.get("/api/users/ahmed/progress").json()
        assert prog["attempts"][0]["text"] == "a key"
        assert {p["phone"] for p in prog["phones"]} >= {"K", "IY"}


def test_calibration_validates() -> None:
    with TestClient(app) as c:
        assert c.post("/api/calibration", json={"tokens": [["IY", 300, 2300]]}).status_code == 422
        ok = c.post(
            "/api/calibration",
            json={"user": "cal", "tokens": [["IY", 300, 2300], ["AA", 750, 1200], ["UW", 330, 950]]},
        )
        assert ok.status_code == 200 and ok.json()["speaker_norm"]["n_tokens"] == 3


def test_websocket_stream_end_to_end() -> None:
    audio = speech()
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with TestClient(app) as c, c.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "format", "dtype": "int16"})
        ws.send_json({"type": "start", "text": "a key", "user": "ws-user"})
        chunk = 640  # 40 ms
        for i in range(0, len(pcm16), chunk):
            ws.send_bytes(pcm16[i : i + chunk].tobytes())
        seen = []
        while True:
            m = ws.receive_json()
            seen.append(m["type"])
            if m["type"] in ("result", "error"):
                break
        assert m["type"] == "result", m
        assert "speech_start" in seen and "speech_end" in seen
        assert m["latency_ms"] < 2000
        assert [w["text"] for w in m["words"]] == ["a", "key"]


def test_assess_uses_stored_calibration() -> None:
    with TestClient(app) as c:
        toks = [["IY", 300, 2300], ["AA", 750, 1200], ["UW", 330, 950], ["AE", 650, 1800]]
        assert c.post("/api/calibration", json={"user": "calib2", "tokens": toks}).status_code == 200
        r = c.post(
            "/api/assess",
            data={"text": "a key", "user": "calib2"},
            files={"audio": ("x.wav", wav_bytes(speech()), "audio/wav")},
        )
        assert r.status_code == 200, r.text
        assert not any("default speaker normalisation" in w for w in r.json()["warnings"])
