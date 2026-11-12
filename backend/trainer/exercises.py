"""Drill content generated from CMUdict + word frequencies (wordfreq), so it scales to any
skill without hand-written lists: target words by skill/position, mined minimal pairs and
stress-shift pairs. Sentence-level drills come from the curated bank (and the LLM coach)."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from functools import cache, lru_cache

from accent_engine.g2p import _cmudict, _parse_arpabet, build_target, lookup
from accent_engine.phonemes import VOICING_PAIRS

from .skills import site_skills

N_COMMON = 20000

# Contrasts mined as minimal pairs: (target phone, confusable phone) per skill
CONTRASTS: dict[str, list[tuple[str, str]]] = {
    "P_B": [("P", "B")],
    "V_F": [("V", "F"), ("V", "B"), ("V", "W")],
    "TH_SUB": [("TH", "S"), ("TH", "T"), ("TH", "F")],
    "DH_SUB": [("DH", "D"), ("DH", "Z")],
    "IH_IY": [("IH", "IY")],
    "UH_UW": [("UH", "UW")],
    "AE_EH": [("AE", "EH")],
    "FINAL_DEVOICING": [(v, u) for v, u in VOICING_PAIRS.items()],
    "RHOTIC": [("R", "L"), ("R", "W")],
}

TONGUE_TWISTERS: dict[str, list[str]] = {
    "P_B": ["Peter Piper picked a peck of pickled peppers.", "Big black bugs bleed blue black blood."],
    "V_F": ["Vincent vowed vengeance very vehemently.", "Five valiant vets fixed vivid vans."],
    "TH_SUB": [
        "Thirty-three thousand thoughtful thinkers thought thoroughly.",
        "Thank them for the three things.",
    ],
    "DH_SUB": ["This and that and these and those.", "The other brother breathes there."],
    "RHOTIC": ["Red lorry, yellow lorry.", "Rory's ruler rarely rests right."],
    "IH_IY": ["Six sick sheep sit and sleep.", "Will these sheep fit in this ship?"],
    "EPENTHESIS": ["Strong students stream through the street.", "Spring splits the spectacular sky."],
    "FLAP_T": ["A bitter butter made the batter better.", "Get a little bit of water."],
}

SHADOWING: list[dict[str, str]] = [
    {
        "title": "Lab meeting",
        "text": "So, the main result this week is that the reaction yield went up to about eighty percent. "
        "I think the temperature was the key factor, but we should run it again to be sure.",
    },
    {
        "title": "Small talk",
        "text": "Hey, how's it going? I haven't seen you in a while. Did you get a chance to try that "
        "new coffee place downtown? It's actually pretty good.",
    },
    {
        "title": "Job interview",
        "text": "I'd say my biggest strength is that I'm really organized. In my last position I "
        "managed three projects at the same time, and we delivered all of them on schedule.",
    },
]


@dataclass(frozen=True)
class WordItem:
    word: str
    zipf: float
    position: str


@lru_cache(maxsize=1)
def _common_words() -> list[tuple[str, float]]:
    from wordfreq import top_n_list, zipf_frequency

    cmu = _cmudict()
    return [
        (w, zipf_frequency(w, "en"))
        for w in top_n_list("en", N_COMMON)
        if w.isalpha() and w in cmu and len(w) > 1
    ]


@lru_cache(maxsize=1)
def _skill_index() -> dict[str, list[WordItem]]:
    idx: dict[str, list[WordItem]] = defaultdict(list)
    for w, z in _common_words():
        t = build_target([lookup(w)])
        seen: set[str] = set()
        for i, p in enumerate(t.phones):
            for sk in site_skills(t.phones, i):
                key = f"{sk}:{p.position}"
                if key not in seen:
                    idx[key].append(WordItem(w, z, p.position))
                    seen.add(key)
                if sk not in seen:
                    idx[sk].append(WordItem(w, z, p.position))
                    seen.add(sk)
    return idx


def words_for(skill: str, position: str | None = None, n: int = 10, seed: int | None = None) -> list[str]:
    """Common words exercising ``skill`` (optionally at word position), frequency-weighted sample."""
    items = _skill_index().get(f"{skill}:{position}" if position else skill, [])
    pool = items[:400]  # list is frequency ordered: keep it practical
    rng = random.Random(seed)
    if len(pool) <= n:
        return [i.word for i in pool]
    return [i.word for i in rng.sample(pool, n)]


@cache
def minimal_pairs(skill: str, limit: int = 40) -> list[tuple[str, str]]:
    """Mine common-word minimal pairs differing in exactly one phone of the skill's contrast."""
    contrasts = CONTRASTS.get(skill, [])
    if not contrasts:
        return []
    by_pron: dict[tuple[str, ...], str] = {}
    for w, _ in _common_words():
        pron = tuple(p for p, _ in lookup(w).default)
        by_pron.setdefault(pron, w)
    pairs: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for pron, w in by_pron.items():
        for a, b in contrasts:
            for k, ph in enumerate(pron):
                if ph != a:
                    continue
                if skill == "FINAL_DEVOICING" and k != len(pron) - 1:
                    continue
                other = (*pron[:k], b, *pron[k + 1 :])
                w2 = by_pron.get(other)
                if w2 and w2 != w and frozenset((w, w2)) not in seen:
                    seen.add(frozenset((w, w2)))
                    pairs.append((w, w2))
        if len(pairs) >= limit:
            break
    return pairs


@lru_cache(maxsize=1)
def stress_shift_pairs(limit: int = 30) -> list[tuple[str, str, str]]:
    """Words whose dictionary variants differ in primary stress (REcord/reCORD): (word, noun, verb)."""
    out = []
    for w, _ in _common_words():
        variants = _cmudict()[w]
        pats = {}
        for v in variants:
            stresses = [s for _, s in _parse_arpabet(v) if s is not None]
            if len(stresses) >= 2 and 1 in stresses:
                pats[stresses.index(1)] = v
        if 0 in pats and 1 in pats:
            out.append((w, " ".join(pats[0]), " ".join(pats[1])))
        if len(out) >= limit:
            break
    return out
