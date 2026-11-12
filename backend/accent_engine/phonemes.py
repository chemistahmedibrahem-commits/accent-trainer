"""General American phone inventory (ARPAbet), IPA mapping and articulatory features.

All engine-internal phone symbols are ARPAbet *without* stress digits. Stress is
carried separately on :class:`accent_engine.g2p.TargetPhone`.

Surface-only phones (never in CMUdict, produced by post-lexical rules or the
acoustic model):

* ``DX`` – alveolar flap [ɾ] (water, better)
* ``Q``  – glottal stop [ʔ] (t-glottalization: button, "ge'it")
"""

from __future__ import annotations

from dataclasses import dataclass

VOWELS: frozenset[str] = frozenset(
    ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
)
CONSONANTS: frozenset[str] = frozenset(
    [
        "B",
        "CH",
        "D",
        "DH",
        "F",
        "G",
        "HH",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
        "DX",
        "Q",
    ]
)
PHONES: tuple[str, ...] = tuple(sorted(VOWELS | CONSONANTS))

IPA: dict[str, str] = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "IH": "ɪ",
    "IY": "i",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "u",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
    "DX": "ɾ",
    "Q": "ʔ",
}


def to_ipa(phone: str, stress: int | None = None) -> str:
    """ARPAbet → IPA, rendering unstressed AH/ER as schwa / schwar."""
    if phone == "AH" and stress == 0:
        return "ə"
    if phone == "ER" and stress == 0:
        return "ɚ"
    return IPA[phone]


# ---------------------------------------------------------------------------
# Articulatory features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsonantFeatures:
    place: int  # 0 bilabial … 7 glottal (ordinal, so distance is meaningful)
    manner: str
    voiced: bool


@dataclass(frozen=True)
class VowelFeatures:
    height: float  # 0 low … 3 high
    backness: float  # 0 front … 2 back
    rounded: bool
    tense: bool
    rhotic: bool
    diphthong: bool


_PLACE = {
    "bilabial": 0,
    "labiodental": 1,
    "dental": 2,
    "alveolar": 3,
    "postalveolar": 4,
    "palatal": 5,
    "velar": 6,
    "glottal": 7,
}


def _c(place: str, manner: str, voiced: bool) -> ConsonantFeatures:
    return ConsonantFeatures(_PLACE[place], manner, voiced)


CONSONANT_FEATURES: dict[str, ConsonantFeatures] = {
    "P": _c("bilabial", "stop", False),
    "B": _c("bilabial", "stop", True),
    "T": _c("alveolar", "stop", False),
    "D": _c("alveolar", "stop", True),
    "K": _c("velar", "stop", False),
    "G": _c("velar", "stop", True),
    "F": _c("labiodental", "fricative", False),
    "V": _c("labiodental", "fricative", True),
    "TH": _c("dental", "fricative", False),
    "DH": _c("dental", "fricative", True),
    "S": _c("alveolar", "fricative", False),
    "Z": _c("alveolar", "fricative", True),
    "SH": _c("postalveolar", "fricative", False),
    "ZH": _c("postalveolar", "fricative", True),
    "HH": _c("glottal", "fricative", False),
    "CH": _c("postalveolar", "affricate", False),
    "JH": _c("postalveolar", "affricate", True),
    "M": _c("bilabial", "nasal", True),
    "N": _c("alveolar", "nasal", True),
    "NG": _c("velar", "nasal", True),
    "L": _c("alveolar", "lateral", True),
    "R": _c("postalveolar", "approximant", True),
    "W": _c("bilabial", "approximant", True),
    "Y": _c("palatal", "approximant", True),
    "DX": _c("alveolar", "flap", True),
    "Q": _c("glottal", "stop", False),
}

VOWEL_FEATURES: dict[str, VowelFeatures] = {
    "IY": VowelFeatures(3.0, 0.0, False, True, False, False),
    "IH": VowelFeatures(2.5, 0.2, False, False, False, False),
    "EY": VowelFeatures(2.0, 0.0, False, True, False, True),
    "EH": VowelFeatures(1.3, 0.2, False, False, False, False),
    "AE": VowelFeatures(0.5, 0.2, False, False, False, False),
    "AA": VowelFeatures(0.0, 1.8, False, False, False, False),
    "AO": VowelFeatures(0.8, 2.0, True, False, False, False),
    "OW": VowelFeatures(2.0, 2.0, True, True, False, True),
    "UH": VowelFeatures(2.5, 1.8, True, False, False, False),
    "UW": VowelFeatures(3.0, 2.0, True, True, False, False),
    "AH": VowelFeatures(1.2, 1.2, False, False, False, False),
    "ER": VowelFeatures(1.5, 1.2, False, False, True, False),
    "AY": VowelFeatures(0.5, 1.0, False, False, False, True),
    "AW": VowelFeatures(0.5, 1.2, False, False, False, True),
    "OY": VowelFeatures(1.0, 2.0, True, False, False, True),
}

_MANNER_GROUP = {
    "stop": 0,
    "affricate": 0.5,
    "fricative": 1,
    "flap": 0.3,
    "nasal": 2,
    "lateral": 3,
    "approximant": 3,
}


def is_vowel(phone: str) -> bool:
    return phone in VOWELS


def phone_distance(a: str, b: str) -> float:
    """Articulatory distance in [0, 1]. 0 = identical, 1 = maximally different."""
    if a == b:
        return 0.0
    if is_vowel(a) != is_vowel(b):
        # R and ER are close despite crossing the C/V boundary.
        return 0.5 if {a, b} == {"R", "ER"} else 1.0
    if is_vowel(a):
        fa, fb = VOWEL_FEATURES[a], VOWEL_FEATURES[b]
        d = (abs(fa.height - fb.height) / 3.0 + abs(fa.backness - fb.backness) / 2.0) / 2.0
        d += 0.1 * (fa.rounded != fb.rounded) + 0.1 * (fa.tense != fb.tense)
        d += 0.25 * (fa.rhotic != fb.rhotic) + 0.1 * (fa.diphthong != fb.diphthong)
        return min(1.0, 0.15 + d)
    ca, cb = CONSONANT_FEATURES[a], CONSONANT_FEATURES[b]
    d = abs(ca.place - cb.place) / 7.0 * 0.4
    d += min(1.0, abs(_MANNER_GROUP[ca.manner] - _MANNER_GROUP[cb.manner]) / 2.0) * 0.4
    d += 0.2 * (ca.voiced != cb.voiced)
    return min(1.0, 0.1 + d)


VOICING_PAIRS: dict[str, str] = {
    "B": "P",
    "D": "T",
    "G": "K",
    "V": "F",
    "Z": "S",
    "DH": "TH",
    "ZH": "SH",
    "JH": "CH",
}
VOICED_OBSTRUENTS: frozenset[str] = frozenset(VOICING_PAIRS)
