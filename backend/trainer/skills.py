"""Skills = the error taxonomy. Every target phone is an *opportunity* for the skills it
exercises; it is answered correctly unless the engine reported that error at that site.
This is what BKT needs: per-opportunity binary evidence, not just error counts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from accent_engine.g2p import TargetPhone
from accent_engine.phonemes import VOICED_OBSTRUENTS, is_vowel

SKILLS: tuple[str, ...] = (
    "P_B",
    "V_F",
    "TH_SUB",
    "DH_SUB",
    "IH_IY",
    "UH_UW",
    "AE_EH",
    "SCHWA_REDUCTION",
    "RHOTIC",
    "FLAP_T",
    "DARK_L",
    "EPENTHESIS",
    "FINAL_DEVOICING",
    "LEXICAL_STRESS",
)

SKILL_LABEL: dict[str, str] = {
    "P_B": "/p/ vs /b/",
    "V_F": "/v/ vs /f/",
    "TH_SUB": "voiceless th /θ/",
    "DH_SUB": "voiced th /ð/",
    "IH_IY": "ship vs sheep /ɪ i/",
    "UH_UW": "full vs fool /ʊ u/",
    "AE_EH": "bad vs bed /æ ɛ/",
    "SCHWA_REDUCTION": "reduced vowels /ə/",
    "RHOTIC": "American R /ɹ ɚ/",
    "FLAP_T": "flap T (water)",
    "DARK_L": "dark L (feel)",
    "EPENTHESIS": "consonant clusters (street)",
    "FINAL_DEVOICING": "final voiced consonants (bag)",
    "LEXICAL_STRESS": "word stress",
}


def site_skills(phones: Sequence[TargetPhone], i: int) -> list[str]:
    """Skills exercised by target phone ``i``."""
    p = phones[i]
    ph, out = p.phone, []
    stressed = p.stress is not None and p.stress > 0
    simple = {"P": "P_B", "V": "V_F", "TH": "TH_SUB", "DH": "DH_SUB"}
    if ph in simple:
        out.append(simple[ph])
    if stressed and ph in ("IH", "IY"):
        out.append("IH_IY")
    if stressed and ph in ("UH", "UW"):
        out.append("UH_UW")
    if stressed and ph in ("AE", "EH"):
        out.append("AE_EH")
    if ph == "AH" and p.stress == 0:
        out.append("SCHWA_REDUCTION")
    if ph in ("R", "ER"):
        out.append("RHOTIC")
    if p.allophone == "flap":
        out.append("FLAP_T")
    if p.allophone == "dark_l":
        out.append("DARK_L")
    if p.position == "final" and p.underlying in VOICED_OBSTRUENTS:
        out.append("FINAL_DEVOICING")
    nxt = phones[i + 1] if i + 1 < len(phones) else None
    if (
        p.position == "initial"
        and not is_vowel(ph)
        and nxt is not None
        and nxt.word_index == p.word_index
        and not is_vowel(nxt.phone)
    ):
        out.append("EPENTHESIS")
    return out


@dataclass(frozen=True)
class Observation:
    skill: str
    correct: bool
    score: float  # 0..100 evidence strength for FSRS grading
    word: str


def observations(assessment: dict[str, Any], phones: Sequence[TargetPhone]) -> list[Observation]:
    """Turn one engine assessment (``Assessment.to_dict()``) into skill observations."""
    words = assessment["words"]
    flat = {p["index"]: p for w in words for p in w["phones"]}
    errs: dict[tuple[str, int], bool] = {}
    for e in assessment["errors"]:
        if e["target_index"] is not None:
            errs[(e["code"], e["target_index"])] = True
    out: list[Observation] = []
    for i, _ in enumerate(phones):
        if i not in flat:
            continue
        score = float(flat[i]["score"] if flat[i]["score"] is not None else 0.0)
        word = words[phones[i].word_index]["text"]
        for sk in site_skills(phones, i):
            wrong = errs.get((sk, i), False)
            out.append(Observation(sk, not wrong, min(score, 40.0) if wrong else score, word))
    for w in words:
        st = w.get("stress")
        if st is not None:
            out.append(
                Observation("LEXICAL_STRESS", bool(st["correct"]), 90.0 if st["correct"] else 30.0, w["text"])
            )
    return out
