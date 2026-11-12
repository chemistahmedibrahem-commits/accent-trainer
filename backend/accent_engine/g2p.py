"""Text → General American target phone sequence.

Pipeline:
  1. tokenize text into words
  2. lexical lookup (CMUdict, all listed variants) → weak forms for function words
     → context reductions (gonna / wanna / gotta) → OOV fallback (g2p_en if installed,
     else a small letter-to-sound rule set)
  3. :func:`build_target` concatenates one variant per word and applies General
     American post-lexical rules (flap T/D, t-glottalization before syllabic n,
     dark-L marking).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from .phonemes import VOWELS, is_vowel

Position = Literal["initial", "medial", "final"]
Pron = tuple[tuple[str, int | None], ...]  # ((phone, stress|None), ...)


@dataclass(frozen=True)
class TargetPhone:
    phone: str
    stress: int | None
    word_index: int
    syllable_index: int
    position: Position
    underlying: str  # phonemic phone before post-lexical rules (e.g. T for a flap)
    alternatives: frozenset[str] = frozenset()  # other realizations counted as correct
    allophone: str | None = None  # "flap" | "glottal" | "dark_l"


@dataclass
class WordPron:
    text: str
    variants: list[Pron]
    source: Literal["cmudict", "weak_form", "reduction", "g2p_model", "rules"]

    @property
    def default(self) -> Pron:
        return self.variants[0]


@dataclass
class Target:
    words: list[str]
    phones: list[TargetPhone]
    variant_choice: list[int] = field(default_factory=list)

    def word_phone_indices(self, word_index: int) -> list[int]:
        return [i for i, p in enumerate(self.phones) if p.word_index == word_index]


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------


def _parse_arpabet(tokens: Sequence[str]) -> Pron:
    out: list[tuple[str, int | None]] = []
    for t in tokens:
        m = re.fullmatch(r"([A-Z]+)([012])?", t)
        if not m:
            raise ValueError(f"bad ARPAbet token {t!r}")
        phone, stress = m.group(1), m.group(2)
        out.append((phone, int(stress) if stress is not None else (1 if phone in VOWELS else None)))
    return tuple(out)


def parse_pron(s: str) -> Pron:
    return _parse_arpabet(s.split())


@lru_cache(maxsize=1)
def _cmudict() -> dict[str, list[list[str]]]:
    import cmudict  # local import: ~1 s load, keep module import cheap

    return cmudict.dict()  # type: ignore[no-any-return]


# Weak (reduced) forms are listed FIRST only where the reduced form is the normal
# conversational GA target; strong forms stay as accepted alternatives.
WEAK_FORMS: dict[str, list[str]] = {
    "a": ["AH0", "EY1"],
    "an": ["AH0 N", "AE1 N"],
    "the": ["DH AH0", "DH IY0", "DH IY1"],
    "to": ["T AH0", "T UW1"],
    "for": ["F ER0", "F AO1 R"],
    "and": ["AH0 N D", "AH0 N", "AE1 N D"],
    "of": ["AH0 V", "AH0", "AH1 V"],
    "can": ["K AH0 N", "K AE1 N"],
    "was": ["W AH0 Z", "W AA1 Z", "W AH1 Z"],
    "at": ["AH0 T", "AE1 T"],
    "from": ["F R AH0 M", "F R AH1 M"],
    "you": ["Y UW1", "Y AH0"],
    "your": ["Y ER0", "Y AO1 R", "Y UH1 R"],
    "them": ["DH AH0 M", "DH EH1 M"],
    "him": ["IH0 M", "HH IH1 M"],
    "her": ["ER0", "HH ER1"],
    "some": ["S AH0 M", "S AH1 M"],
    "but": ["B AH0 T", "B AH1 T"],
    "as": ["AH0 Z", "AE1 Z"],
    "are": ["ER0", "AA1 R"],
    "or": ["ER0", "AO1 R"],
}

# Context reductions: (word, next word) → extra variants for `word`,
# (word, previous word) → extra variants for `word`.
_REDUCE_BEFORE: dict[tuple[str, str], list[str]] = {
    ("going", "to"): ["G AH1 N"],  # gonna
    ("want", "to"): ["W AA1 N", "W AH1 N"],  # wanna
    ("got", "to"): ["G AA1"],  # gotta → [ɡɑɾə]
    ("have", "to"): ["HH AE1 F"],  # hafta
    ("kind", "of"): ["K AY1 N"],  # kinda
}
_REDUCE_AFTER: dict[tuple[str, str], list[str]] = {
    ("to", "going"): ["AH0"],
    ("to", "want"): ["AH0"],
    ("to", "got"): ["DX AH0"],
    ("to", "have"): ["T AH0"],
    ("of", "kind"): ["D AH0"],
}


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _num_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + _num_to_words(rest))
    if n < 10000:
        rest = n % 1000
        return _ONES[n // 1000] + " thousand" + ("" if rest == 0 else " " + _num_to_words(rest))
    return " ".join(_ONES[int(d)] for d in str(n))


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    text = re.sub(r"\d+", lambda m: " " + _num_to_words(int(m.group())) + " ", text)
    return [w.strip("'") for w in re.findall(r"[a-z]+(?:'[a-z]+)*", text) if w.strip("'")]


# ---------------------------------------------------------------------------
# OOV fallback
# ---------------------------------------------------------------------------

_LTS_RULES: list[tuple[str, str]] = [
    ("tch", "CH"),
    ("sch", "S K"),
    ("ph", "F"),
    ("sh", "SH"),
    ("ch", "CH"),
    ("th", "TH"),
    ("ng", "NG"),
    ("ck", "K"),
    ("qu", "K W"),
    ("wh", "W"),
    ("gh", ""),
    ("kn", "N"),
    ("ee", "IY1"),
    ("ea", "IY1"),
    ("oo", "UW1"),
    ("ou", "AW1"),
    ("ow", "OW1"),
    ("oi", "OY1"),
    ("oy", "OY1"),
    ("ai", "EY1"),
    ("ay", "EY1"),
    ("au", "AO1"),
    ("aw", "AO1"),
    ("ie", "IY1"),
    ("er", "ER0"),
    ("ir", "ER1"),
    ("ur", "ER1"),
    ("ar", "AA1 R"),
    ("or", "AO1 R"),
    ("a", "AE1"),
    ("e", "EH1"),
    ("i", "IH1"),
    ("o", "AA1"),
    ("u", "AH1"),
    ("y", "IY0"),
    ("b", "B"),
    ("c", "K"),
    ("d", "D"),
    ("f", "F"),
    ("g", "G"),
    ("h", "HH"),
    ("j", "JH"),
    ("k", "K"),
    ("l", "L"),
    ("m", "M"),
    ("n", "N"),
    ("p", "P"),
    ("r", "R"),
    ("s", "S"),
    ("t", "T"),
    ("v", "V"),
    ("w", "W"),
    ("x", "K S"),
    ("z", "Z"),
]


def _letter_to_sound(word: str) -> Pron:
    """Very small greedy rule set; only a last resort for OOV words."""
    w = word.lower()
    if len(w) > 2 and w.endswith("e") and w[-2] not in "aeiouy":
        w = w[:-1]  # silent final e
    out: list[str] = []
    i = 0
    while i < len(w):
        for graph, phones in _LTS_RULES:
            if w.startswith(graph, i):
                if phones:
                    out.extend(phones.split())
                i += len(graph)
                break
        else:
            i += 1
        # collapse doubled consonants (ll, ss, tt)
        if len(out) >= 2 and out[-1] == out[-2] and not is_vowel(out[-1].rstrip("012")):
            out.pop()
    pron = list(_parse_arpabet(out))
    # keep primary stress only on the first vowel
    seen = False
    for k, (p, s) in enumerate(pron):
        if s is not None:
            pron[k] = (p, 0 if seen and s == 1 else s)
            seen = seen or s == 1
    return tuple(pron)


@lru_cache(maxsize=1)
def _neural_g2p():  # type: ignore[no-untyped-def]
    try:
        from g2p_en import G2p  # type: ignore[import-not-found]

        return G2p()
    except Exception:
        return None


def _oov(word: str) -> WordPron:
    g = _neural_g2p()
    if g is not None:
        toks = [t for t in g(word) if re.fullmatch(r"[A-Z]+[012]?", t)]
        if toks:
            return WordPron(word, [_parse_arpabet(toks)], "g2p_model")
    return WordPron(word, [_letter_to_sound(word)], "rules")


def lookup(word: str, prev: str | None = None, nxt: str | None = None) -> WordPron:
    word = word.lower()
    variants: list[Pron] = []
    source: Literal["cmudict", "weak_form", "reduction", "g2p_model", "rules"] = "cmudict"
    if word in WEAK_FORMS:
        variants = [parse_pron(s) for s in WEAK_FORMS[word]]
        source = "weak_form"
    entries = _cmudict().get(word)
    if entries:
        for e in entries:
            p = _parse_arpabet(e)
            if p not in variants:
                variants.append(p)
    if not variants:
        return _oov(word)
    extra = _REDUCE_BEFORE.get((word, nxt or "")) or _REDUCE_AFTER.get((word, prev or ""))
    if extra:
        for s in extra:
            p = parse_pron(s)
            if p not in variants:
                variants.append(p)  # accepted alternatives, not the default target
    return WordPron(word, variants, source)


def lexicon_for(text: str) -> list[WordPron]:
    words = tokenize(text)
    return [
        lookup(w, words[i - 1] if i > 0 else None, words[i + 1] if i + 1 < len(words) else None)
        for i, w in enumerate(words)
    ]


# ---------------------------------------------------------------------------
# Post-lexical rules
# ---------------------------------------------------------------------------


def build_target(lexicon: Sequence[WordPron], choice: Sequence[int] | None = None) -> Target:
    """Concatenate one variant per word and apply General American post-lexical rules."""
    choice = list(choice) if choice is not None else [0] * len(lexicon)
    flat: list[tuple[str, int | None, int, int, Position]] = []
    for wi, (wp, c) in enumerate(zip(lexicon, choice, strict=True)):
        pron = wp.variants[c]
        syl = -1
        for k, (ph, st) in enumerate(pron):
            if is_vowel(ph):
                syl += 1
            pos: Position = "initial" if k == 0 else ("final" if k == len(pron) - 1 else "medial")
            flat.append((ph, st, wi, max(syl, 0), pos))

    phones: list[TargetPhone] = []
    for i, (ph, st, wi, syl, pos) in enumerate(flat):
        prev = flat[i - 1] if i > 0 else None
        nxt = flat[i + 1] if i + 1 < len(flat) else None
        nxt2 = flat[i + 2] if i + 2 < len(flat) else None
        surface, alts, allo = ph, frozenset[str](), None

        glottal_site = (
            ph == "T"
            and nxt is not None
            and nxt2 is not None
            and prev is not None
            and nxt[0] == "AH"
            and nxt[1] == 0
            and nxt2[0] == "N"
            and nxt2[4] == "final"
            and nxt2[2] == wi
            and (is_vowel(prev[0]) or prev[0] == "N")
        )
        if glottal_site:
            alts, allo = frozenset({"Q"}), "glottal"  # button, mountain
        elif ph in ("T", "D") and prev is not None and nxt is not None:
            prev_ok = is_vowel(prev[0]) or prev[0] == "R"
            same_word = nxt[2] == wi
            if (
                prev_ok
                and is_vowel(nxt[0])
                and pos != "initial"
                and ((same_word and nxt[1] == 0) or (not same_word and pos == "final"))
            ):
                surface, alts, allo = "DX", frozenset({ph}), "flap"
        if ph == "L" and (nxt is None or not is_vowel(nxt[0]) or nxt[2] != wi):
            allo = "dark_l"

        phones.append(TargetPhone(surface, st, wi, syl, pos, ph, alts, allo))
    return Target([wp.text for wp in lexicon], phones, choice)


def g2p(text: str) -> Target:
    return build_target(lexicon_for(text))
