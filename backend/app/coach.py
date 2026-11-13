"""AI coach: explains engine output like a speech coach. It never produces scores.

* ``digest()`` compresses an assessment into the small structured JSON the LLM sees.
* ``LLMCoach`` asks Claude for {summary, focus[], drills[]} with structured outputs.
* ``validate()`` enforces the hard rule: every number in the reply must already exist in the
  digest, and every error code / word must come from the digest. Violations → rule-based reply.
* ``rule_based()`` is the deterministic coach used without an API key or on any failure.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from trainer import exercises

from .content import SENTENCES

log = logging.getLogger(__name__)

MODEL = os.environ.get("AT_COACH_MODEL", "claude-opus-5-5")

SYSTEM = """You are an American English (General American) pronunciation coach for a learner whose \
first language is Egyptian Arabic. You receive the output of an automatic pronunciation engine as JSON.

Rules:
- Explain ONLY what the engine reported. Never invent, estimate or change a score, and never write \
a number that does not appear in the input JSON.
- Pick at most 3 focus items, most severe first. Use the error codes and words exactly as given.
- For each, give a one-sentence explanation of what went wrong and a concrete articulatory tip \
(tongue, lips, jaw, voicing, airflow). Relate it to Egyptian Arabic when that genuinely helps \
(e.g. Arabic has /b/ but no /p/, no /v/, no /θ ð/ in Egyptian colloquial, a tapped or trilled r).
- Write exactly 3 short, natural drill sentences that contain the focus sounds many times.
- Be warm and brief: plain language, no jargon without explanation, no markdown."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "focus": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "word": {"type": "string"},
                    "explanation": {"type": "string"},
                    "articulation": {"type": "string"},
                },
                "required": ["code", "word", "explanation", "articulation"],
                "additionalProperties": False,
            },
        },
        "drills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "focus", "drills"],
    "additionalProperties": False,
}

_SEV = {"major": 0, "moderate": 1, "minor": 2}


def _r(x: Any) -> int | None:
    return None if x is None else round(float(x))


def digest(result: dict[str, Any], weaknesses: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Compact, LLM-facing view of one assessment. All numbers are rounded integers."""
    errors = sorted(result["errors"], key=lambda e: (_SEV.get(e["severity"], 3), -e["confidence"]))
    phones = [(w["text"], p) for w in result["words"] for p in w["phones"]]
    weak = sorted((x for x in phones if x[1]["score"] is not None), key=lambda x: x[1]["score"])[:5]
    vowels = [
        {
            "word": w,
            "vowel": p["ipa"],
            "sounds_like": p["vowel_score"]["nearest"],
            "score": _r(p["vowel_score"]["score"]),
        }
        for w, p in phones
        if p.get("vowel_score") and p["vowel_score"]["score"] < 60
    ]
    stress = [
        {
            "word": w["text"],
            "expected_syllable": w["stress"]["expected_syllable"] + 1,
            "stressed_syllable": w["stress"]["produced_syllable"] + 1,
        }
        for w in result["words"]
        if w.get("stress") and not w["stress"]["correct"]
    ]
    return {
        "target_text": result["text"],
        "scores": {k: _r(v) for k, v in result["scores"].items() if v is not None},
        "errors": [
            {
                "code": e["code"],
                "name": e["name"],
                "word": e["word"],
                "expected": e["expected"],
                "heard": e["realized"],
                "severity": e["severity"],
                "reference_tip": e["tip"],
            }
            for e in errors[:8]
        ],
        "weakest_sounds": [
            {"word": w, "sound": p["ipa"], "score": _r(p["score"])} for w, p in weak if p["score"] < 75
        ],
        "vowels_off_target": vowels[:4],
        "stress_errors": stress,
        "learner_top_weaknesses": [w["label"] for w in (weaknesses or [])][:3],
    }


def _numbers(obj: Any) -> set[int]:
    return {int(n) for n in re.findall(r"(?<![\w.])\d+(?![\w.])", json.dumps(obj, ensure_ascii=False))}


def validate(reply: dict[str, Any], dg: dict[str, Any]) -> list[str]:
    """Return a list of violations (empty = OK)."""
    problems: list[str] = []
    allowed = _numbers(dg) | {1, 2, 3}  # "3 drills", "syllable 2" are not scores
    text = " ".join(
        [
            reply["summary"],
            *reply["drills"],
            *[f"{f['explanation']} {f['articulation']}" for f in reply["focus"]],
        ]
    )
    invented = {int(n) for n in re.findall(r"(?<![\w.])\d+(?![\w.])", text)} - allowed
    if invented:
        problems.append(f"numbers not in engine output: {sorted(invented)}")
    codes = {e["code"] for e in dg["errors"]} | {"LEXICAL_STRESS", "VOWEL", "GENERAL"}
    words = (
        {e["word"].lower() for e in dg["errors"]}
        | {w["word"].lower() for w in dg["weakest_sounds"]}
        | {v["word"].lower() for v in dg["vowels_off_target"]}
        | {s["word"].lower() for s in dg["stress_errors"]}
    )
    for f in reply["focus"]:
        if f["code"] not in codes:
            problems.append(f"unknown error code {f['code']}")
        if f["word"].lower() not in words:
            problems.append(f"word not in engine output: {f['word']}")
    if len(reply["drills"]) < 1:
        problems.append("no drills")
    return problems


def rule_based(dg: dict[str, Any]) -> dict[str, Any]:
    s = dg["scores"]
    total = s.get("total")
    major = [e for e in dg["errors"] if e["severity"] != "minor"]
    minor = [e for e in dg["errors"] if e["severity"] == "minor"]
    n_stress = len(dg["stress_errors"])
    summary = f"Overall {total}. " if total is not None else ""
    if not major and not n_stress:
        summary += "Clear and accurate — no pronunciation errors were detected in this sentence."
        if minor:
            summary += " A small style note is below."
    else:
        parts = []
        if major:
            parts.append(f"{len(major)} sound{'s' if len(major) > 1 else ''} to fix")
        if n_stress:
            parts.append(f"{n_stress} word-stress issue{'s' if n_stress > 1 else ''}")
        summary += f"Found {' and '.join(parts)}; start with the first one below."
    focus, seen = [], set()
    for e in dg["errors"]:
        if e["code"] in seen:
            continue
        seen.add(e["code"])
        heard = "it was missing" if e["heard"] == "<del>" else f"it sounded like {e['heard']}"
        exp = f"the {e['expected']} sound" if e["expected"] else "this word"
        focus.append(
            {
                "code": e["code"],
                "word": e["word"],
                "explanation": f"In “{e['word']}”, {exp}: {heard} ({e['name']}).",
                "articulation": e["reference_tip"] or "Listen to the model and repeat slowly.",
            }
        )
        if len(focus) == 3:
            break
    for st in dg["stress_errors"]:
        if len(focus) < 3:
            focus.append(
                {
                    "code": "LEXICAL_STRESS",
                    "word": st["word"],
                    "explanation": f"“{st['word']}” was stressed on syllable {st['stressed_syllable']}"
                    f" instead of syllable {st['expected_syllable']}.",
                    "articulation": "Make the stressed syllable longer, louder and higher in pitch; "
                    "reduce the others.",
                }
            )
    drills: list[str] = []
    for f in focus:
        drills += exercises.TONGUE_TWISTERS.get(f["code"], [])[:1] + SENTENCES.get(f["code"], [])[:1]
        if len(drills) < 3 and f["code"] in exercises.CONTRASTS:
            words = exercises.words_for(f["code"], n=5, seed=0)
            if words:
                drills.append("Say each word twice: " + ", ".join(words) + ".")
    if not drills:
        drills = [x["text"] for x in exercises.SHADOWING[:3]]
    return {"summary": summary, "focus": focus, "drills": drills[:3], "source": "rules"}


class LLMCoach:
    def __init__(self) -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()

    def coach(self, dg: dict[str, Any]) -> dict[str, Any]:
        anthropic = self._anthropic
        try:
            resp = self.client.beta.messages.create(
                model=MODEL,
                max_tokens=4000,
                system=SYSTEM,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": json.dumps(dg, ensure_ascii=False)}],
            )
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError) as e:
            log.warning("coach LLM call failed: %s", e)
            return {**rule_based(dg), "fallback_reason": "llm_unavailable"}
        if resp.stop_reason in ("refusal", "max_tokens"):
            return {**rule_based(dg), "fallback_reason": str(resp.stop_reason)}
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            reply = json.loads(text)
        except json.JSONDecodeError:
            return {**rule_based(dg), "fallback_reason": "bad_json"}
        reply["focus"], reply["drills"] = reply["focus"][:3], reply["drills"][:3]
        problems = validate(reply, dg)
        if problems:
            log.warning("coach reply rejected: %s", problems)
            return {**rule_based(dg), "fallback_reason": "validation", "violations": problems}
        return {**reply, "source": "llm", "model": resp.model}


_llm: LLMCoach | None = None


def coach(result: dict[str, Any], weaknesses: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    global _llm
    dg = digest(result, weaknesses)
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return rule_based(dg)
    if _llm is None:
        _llm = LLMCoach()
    return _llm.coach(dg)
