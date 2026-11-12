import json
from types import SimpleNamespace

import numpy as np
import pytest

from accent_engine import EngineConfig, PronunciationEngine
from accent_engine.acoustic import SyntheticAcousticModel
from accent_engine.g2p import g2p
from app import coach

TEXT = "think about the street"


def result(spoken_override: dict[int, str] | None = None) -> dict:  # type: ignore[type-arg]
    tp = [p.phone for p in g2p(TEXT).phones]
    for i, ph in (spoken_override or {}).items():
        tp[i] = ph
    eng = PronunciationEngine(SyntheticAcousticModel(tp), EngineConfig(formants=False, prosody=False))
    return eng.assess(np.zeros(32000, np.float32), TEXT).to_dict()


def test_digest_is_compact_and_integer_valued() -> None:
    dg = coach.digest(result({0: "S"}))
    assert dg["errors"][0]["code"] == "TH_SUB" and dg["errors"][0]["word"] == "think"
    assert all(isinstance(v, int) for v in dg["scores"].values())
    json.dumps(dg)


def test_rule_based_reply_uses_taxonomy_and_drills() -> None:
    r = coach.rule_based(coach.digest(result({0: "S"})))
    assert r["focus"][0]["code"] == "TH_SUB" and "tongue" in r["focus"][0]["articulation"].lower()
    assert 1 <= len(r["drills"]) <= 3
    assert coach.validate(r, coach.digest(result({0: "S"}))) == []


def test_clean_attempt() -> None:
    r = coach.rule_based(coach.digest(result()))
    assert r["focus"] == [] and "no pronunciation errors" in r["summary"].lower()


def test_validator_rejects_invented_scores_codes_and_words() -> None:
    dg = coach.digest(result({0: "S"}))
    ok = {
        "summary": "Good effort.",
        "drills": ["Think of three things."],
        "focus": [{"code": "TH_SUB", "word": "think", "explanation": "x", "articulation": "y"}],
    }
    assert coach.validate(ok, dg) == []
    bad = {**ok, "summary": "Your accuracy was 87 percent."}
    assert any("87" in p for p in coach.validate(bad, dg))
    bad2 = {**ok, "focus": [{"code": "V_F", "word": "very", "explanation": "x", "articulation": "y"}]}
    assert len(coach.validate(bad2, dg)) == 2


class _FakeClient:
    def __init__(self, payload: dict) -> None:  # type: ignore[type-arg]
        self.payload = payload
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))
        self.kwargs: dict = {}  # type: ignore[type-arg]

    def create(self, **kw):  # type: ignore[no-untyped-def]
        self.kwargs = kw
        return SimpleNamespace(
            stop_reason="end_turn",
            model="claude-opus-5-5",
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
        )


@pytest.mark.parametrize("invent", [False, True])
def test_llm_path_validated(invent: bool) -> None:
    import anthropic

    dg = coach.digest(result({0: "S"}))
    payload = {
        "summary": "Overall 42!" if invent else "Nice try.",
        "drills": ["Thank them for the three things."],
        "focus": [
            {"code": "TH_SUB", "word": "think", "explanation": "You said s.", "articulation": "Tongue out."}
        ],
    }
    c = coach.LLMCoach.__new__(coach.LLMCoach)
    c._anthropic = anthropic
    c.client = _FakeClient(payload)  # type: ignore[assignment]
    out = c.coach(dg)
    assert c.client.kwargs["output_config"]["format"]["type"] == "json_schema"  # type: ignore[attr-defined]
    assert c.client.kwargs["model"] == "claude-opus-5-5"  # type: ignore[attr-defined]
    if invent:
        assert out["source"] == "rules" and out["fallback_reason"] == "validation"
    else:
        assert out["source"] == "llm" and out["summary"] == "Nice try."
