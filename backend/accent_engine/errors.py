"""Error detection and classification with an Egyptian-Arabic-L1 taxonomy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .alignment import AlignOp
from .g2p import TargetPhone
from .gop import PhoneGOP
from .phonemes import VOICING_PAIRS, is_vowel

Severity = Literal["minor", "moderate", "major"]
Source = Literal["alignment", "gop", "acoustic"]


@dataclass(frozen=True)
class ErrorType:
    code: str
    name: str
    severity: Severity
    tip: str  # articulatory tip; the coach paraphrases it, it never invents new facts


TAXONOMY: dict[str, ErrorType] = {
    e.code: e
    for e in [
        ErrorType(
            "P_B",
            "/p/ produced as /b/",
            "major",
            "Close both lips, build air pressure, and release with a puff of air "
            "(aspiration) before the vowel. Keep the vocal folds silent during the closure.",
        ),
        ErrorType(
            "V_F",
            "/v/ produced as /f/ (or /w/, /b/)",
            "major",
            "Rest the upper teeth lightly on the inner lower lip and keep your voice "
            "buzzing through the friction — a hand on your throat should feel vibration.",
        ),
        ErrorType(
            "TH_SUB",
            "voiceless /θ/ replaced (t, s or f)",
            "major",
            "Place the tongue tip lightly between or just behind the upper front teeth "
            "and blow air through the gap; do not stop the airflow.",
        ),
        ErrorType(
            "DH_SUB",
            "voiced /ð/ replaced (d, z or v)",
            "major",
            "Tongue tip lightly touching the upper teeth, voice on, continuous airflow — "
            "a buzzing 'th', never a stop.",
        ),
        ErrorType(
            "IH_IY",
            "/ɪ/ vs /i/ confusion (ship/sheep)",
            "major",
            "/i/ is tense with spread lips and a high tongue; /ɪ/ is shorter, lax, the "
            "tongue a little lower and further back, lips relaxed.",
        ),
        ErrorType(
            "UH_UW",
            "/ʊ/ vs /u/ confusion (full/fool)",
            "major",
            "/u/ is tense with tightly rounded lips; /ʊ/ is short and lax with only "
            "slight rounding and a lower tongue.",
        ),
        ErrorType(
            "AE_EH",
            "/æ/ vs /ɛ/ confusion (bad/bed)",
            "major",
            "For /æ/ drop the jaw further and spread the lips — tongue low and front; /ɛ/ "
            "is mid with a smaller jaw opening.",
        ),
        ErrorType(
            "SCHWA_REDUCTION",
            "unstressed vowel not reduced to schwa",
            "minor",
            "Unstressed syllables in American English are short and relaxed: let the "
            "vowel collapse to a neutral 'uh' [ə] with the jaw and lips at rest.",
        ),
        ErrorType(
            "RHOTIC",
            "American /ɹ/ or r-colored vowel missing",
            "moderate",
            "Bunch or curl the tongue back without touching the roof of the mouth, round "
            "the lips slightly; for -er words hold that tongue shape through the vowel. "
            "Never tap or trill it.",
        ),
        ErrorType(
            "FLAP_T",
            "flap T not used (water, better)",
            "minor",
            "Between vowels, when the next syllable is unstressed, tap the tongue tip "
            "once very quickly on the ridge behind the teeth — a quick, light 'd'.",
        ),
        ErrorType(
            "DARK_L",
            "clear /l/ where American dark [ɫ] is expected",
            "minor",
            "At the end of a syllable (feel, milk) keep the tongue tip on the ridge but "
            "raise the back of the tongue toward the soft palate, like an 'oo' shape.",
        ),
        ErrorType(
            "EPENTHESIS",
            "extra vowel inserted in a consonant cluster",
            "moderate",
            "Glide straight from one consonant to the next with no vowel in between: "
            "start 'street' directly on the hiss of /s/.",
        ),
        ErrorType(
            "FINAL_DEVOICING",
            "final voiced consonant devoiced",
            "moderate",
            "Keep voicing through the final consonant and lengthen the vowel before it — "
            "'bag' has a longer vowel than 'back'.",
        ),
        ErrorType("SUBSTITUTION", "phone substitution", "major", ""),
        ErrorType("DELETION", "phone omitted", "major", ""),
        ErrorType("INSERTION", "extra phone inserted", "moderate", ""),
    ]
}


@dataclass(frozen=True)
class PronunciationError:
    code: str
    target_index: int | None  # index into Target.phones (None for some insertions)
    word_index: int
    expected: str | None
    realized: str | None  # phone symbol, "<del>" or None
    source: Source
    confidence: float  # 0..1

    @property
    def severity(self) -> Severity:
        return TAXONOMY[self.code].severity


_PAIRS = [
    ({"P"}, {"B"}, "P_B"),
    ({"V"}, {"F", "W", "B"}, "V_F"),
    ({"TH"}, {"T", "S", "F", "DH"}, "TH_SUB"),
    ({"DH"}, {"D", "Z", "V", "TH"}, "DH_SUB"),
    ({"IH", "IY"}, {"IH", "IY"}, "IH_IY"),
    ({"UH", "UW"}, {"UH", "UW"}, "UH_UW"),
    ({"AE", "EH"}, {"AE", "EH"}, "AE_EH"),
]


def classify_substitution(t: TargetPhone, realized: str) -> str:
    exp = t.phone
    if t.allophone == "flap" and realized == "T":
        return "FLAP_T"
    if t.position == "final" and VOICING_PAIRS.get(exp) == realized:
        return "FINAL_DEVOICING"
    for exp_set, real_set, code in _PAIRS:
        if exp in exp_set and realized in real_set and realized != exp:
            return code
    if exp in ("R", "ER") and realized not in ("R", "ER"):
        return "RHOTIC"
    if exp == "AH" and t.stress == 0 and is_vowel(realized):
        return "SCHWA_REDUCTION"
    return "SUBSTITUTION"


def detect_errors(
    targets: Sequence[TargetPhone],
    ops: Sequence[AlignOp],
    gop: Sequence[PhoneGOP],
    gop_threshold: float = 50.0,
) -> list[PronunciationError]:
    errs: list[PronunciationError] = []
    flagged: set[int] = set()

    for k, op in enumerate(ops):
        if op.kind == "match" and op.ref_index is not None and op.hyp is not None:
            t = targets[op.ref_index]
            if t.allophone == "flap" and op.hyp == "T":
                # accepted (intelligible) but non-native: report as a minor note, no score cap
                errs.append(
                    PronunciationError(
                        "FLAP_T", op.ref_index, t.word_index, t.phone, op.hyp, "alignment", 0.7
                    )
                )
                flagged.add(op.ref_index)
        elif op.kind == "sub":
            assert op.ref_index is not None and op.hyp is not None
            t = targets[op.ref_index]
            conf = 1.0 - gop[op.ref_index].score / 100.0 if gop else 0.8
            errs.append(
                PronunciationError(
                    classify_substitution(t, op.hyp),
                    op.ref_index,
                    t.word_index,
                    t.phone,
                    op.hyp,
                    "alignment",
                    max(0.5, conf),
                )
            )
            flagged.add(op.ref_index)
        elif op.kind == "del":
            assert op.ref_index is not None
            t = targets[op.ref_index]
            code = "RHOTIC" if t.phone in ("R", "ER") else "DELETION"
            errs.append(
                PronunciationError(code, op.ref_index, t.word_index, t.phone, "<del>", "alignment", 0.8)
            )
            flagged.add(op.ref_index)
        elif op.kind == "ins":
            assert op.hyp is not None
            prev_ref = next((o.ref_index for o in reversed(ops[:k]) if o.ref_index is not None), None)
            next_ref = next((o.ref_index for o in ops[k + 1 :] if o.ref_index is not None), None)
            prev_t = targets[prev_ref] if prev_ref is not None else None
            next_t = targets[next_ref] if next_ref is not None else None
            word = next_t.word_index if next_t else (prev_t.word_index if prev_t else 0)
            code = "INSERTION"
            if is_vowel(op.hyp) and next_t is not None and not is_vowel(next_t.phone):
                nxt2 = targets[next_ref + 1] if next_ref is not None and next_ref + 1 < len(targets) else None
                word_initial_cluster = (
                    next_t.position == "initial"
                    and nxt2 is not None
                    and nxt2.word_index == next_t.word_index
                    and not is_vowel(nxt2.phone)
                )
                medial_cluster = (
                    prev_t is not None
                    and not is_vowel(prev_t.phone)
                    and prev_t.word_index == next_t.word_index
                )
                if word_initial_cluster or medial_cluster:
                    code = "EPENTHESIS"
            errs.append(PronunciationError(code, next_ref, word, None, op.hyp, "alignment", 0.8))

    # GOP-only detections: aligned as a match, but the acoustic evidence prefers another phone
    for g in gop:
        if g.index in flagged or g.score >= gop_threshold or g.best_competitor is None:
            continue
        t = targets[g.index]
        if g.best_competitor == "<del>":
            code = "RHOTIC" if t.phone in ("R", "ER") else "DELETION"
        else:
            code = classify_substitution(t, g.best_competitor)
        errs.append(
            PronunciationError(
                code, g.index, t.word_index, t.phone, g.best_competitor, "gop", 1.0 - g.score / 100.0
            )
        )
    errs.sort(key=lambda e: (e.word_index, e.target_index if e.target_index is not None else -1))
    return errs


def is_clear_l(l_f2_hz: float, speaker_mean_f2_hz: float) -> bool:
    """Heuristic dark-L check: a velarised [ɫ] has F2 well below the speaker's mean F2
    (≈ 800–1100 Hz for men vs ≈ 1500+ for clear [l]). Ratio threshold to be tuned on data."""
    return l_f2_hz > 0.85 * speaker_mean_f2_hz
