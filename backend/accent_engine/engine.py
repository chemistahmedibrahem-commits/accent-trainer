"""PronunciationEngine: audio + target text → structured assessment.

UI-independent; the API, the benchmark scripts and the tests all call :meth:`assess`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from . import prosody as pros
from .acoustic import BLANK_ID, OTHER_ID, PHONE_TO_ID, SAMPLE_RATE, VOCAB, AcousticModel, Posteriors
from .alignment import AlignOp, align, alignment_cost
from .ctc import Segment, forced_align, greedy_decode
from .errors import TAXONOMY, PronunciationError, detect_errors, is_clear_l
from .formants import FormantTracker
from .g2p import Target, TargetPhone, WordPron, build_target, lexicon_for
from .gop import Calibration, PhoneGOP, compute_gop
from .phonemes import is_vowel, to_ipa
from .vowel_space import REFERENCE, SpeakerNorm, score_vowel


@dataclass
class EngineConfig:
    gop_threshold: float = 50.0
    max_formant: float = 5500.0
    pitch_method: Literal["praat", "pyin"] = "praat"
    formants: bool = True
    prosody: bool = True
    default_gender: Literal["male", "female"] = "male"
    pause_min_s: float = 0.25
    scorer_path: str | None = None  # learned phone scorer (joblib); None → calibrated GOP
    dark_l_check: bool = False  # F2 heuristic: 13 % false alarms on native speakers → opt-in
    mdd_path: str | None = None  # mispronunciation detector (needs scorer_path); decides reported errors
    export_features: bool = False  # attach per-phone scorer features (for training/eval)


@dataclass
class PhoneResult:
    index: int
    phone: str
    ipa: str
    stress: int | None
    allophone: str | None
    realized: str | None  # recognized phone aligned to this target, "<del>" if deleted
    score: float
    lpr: float
    gop_frame: float
    start: float
    end: float
    formants: dict[str, float] | None = None
    vowel_score: dict[str, Any] | None = None
    features: list[float] | None = None


@dataclass
class WordResult:
    index: int
    text: str
    accuracy: float
    phones: list[PhoneResult]
    stress: dict[str, Any] | None = None
    variant: int = 0


@dataclass
class Assessment:
    text: str
    words: list[WordResult]
    errors: list[dict[str, Any]]
    recognized: list[str]
    scores: dict[str, float | None]
    prosody: dict[str, Any] = field(default_factory=dict)
    timing_ms: dict[str, float] = field(default_factory=dict)
    model: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _round(asdict(self))


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return None if not np.isfinite(x) else round(x, 3)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_round(v) for v in x]
    if isinstance(x, np.floating | np.integer):
        return _round(x.item())
    return x


def choose_variants(lexicon: Sequence[WordPron], hyp: Sequence[str]) -> list[int]:
    """Pick, per word, the dictionary variant closest to what was recognized."""
    default = build_target(lexicon)
    if all(len(w.variants) == 1 for w in lexicon):
        return [0] * len(lexicon)
    ops = align([p.phone for p in default.phones], hyp, [p.alternatives for p in default.phones])
    # hyp span per word: hyp phones aligned to that word + insertions before its next phone
    spans: dict[int, list[str]] = {i: [] for i in range(len(lexicon))}
    cur_word = 0
    for op in ops:
        if op.ref_index is not None:
            cur_word = default.phones[op.ref_index].word_index
        if op.hyp is not None:
            spans[cur_word].append(op.hyp)
    choice = []
    for wi, wp in enumerate(lexicon):
        if len(wp.variants) == 1:
            choice.append(0)
            continue
        costs = []
        for v in wp.variants:
            single = build_target([wp], [wp.variants.index(v)])
            c = alignment_cost(
                align([p.phone for p in single.phones], spans[wi], [p.alternatives for p in single.phones])
            )
            costs.append(c)
        best = int(np.argmin(costs))
        choice.append(best if costs[best] < costs[0] - 1e-9 else 0)
    return choice


def _pauses(track: pros.ProsodyTrack, start: float, end: float, min_s: float) -> list[tuple[float, float]]:
    m = (track.times >= start) & (track.times <= end)
    if not m.any():
        return []
    e = track.energy_db
    thr = np.percentile(e[m], 95) - 30.0
    silent = (e < thr) & m
    out = []
    t = 0
    n = len(silent)
    while t < n:
        if silent[t]:
            s = t
            while t < n and silent[t]:
                t += 1
            if (t - s) * pros.HOP >= min_s:
                out.append((float(track.times[s]), float(track.times[t - 1] + pros.HOP)))
        t += 1
    return out


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


class PronunciationEngine:
    def __init__(
        self,
        acoustic: AcousticModel,
        config: EngineConfig | None = None,
        calibration: Calibration | None = None,
    ) -> None:
        self.acoustic = acoustic
        self.config = config or EngineConfig()
        self.calibration = calibration or Calibration()
        self.scorer = None
        if self.config.scorer_path:
            from .scorer import LearnedScorer

            self.scorer = LearnedScorer.load(self.config.scorer_path)
            trained_for = self.scorer.meta.get("acoustic_model")
            name = getattr(acoustic, "name", "")
            if trained_for and name not in (trained_for, "synthetic"):
                raise ValueError(
                    f"scorer was trained on features of {trained_for!r}, not {name!r}: "
                    "retrain it (scripts/eval_speechocean.py) or unset the scorer path"
                )
        self.mdd = None
        if self.config.mdd_path and self.scorer is not None:
            from .mdd import MDDModel

            self.mdd = MDDModel.load(self.config.mdd_path)

    def assess(
        self,
        audio: NDArray[np.float32],
        text: str,
        speaker_norm: SpeakerNorm | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        strictness: str | None = None,
    ) -> Assessment:
        """``strictness``: feedback level for the mispronunciation detector ("balanced" default,
        "gentle" = fewer, higher-precision corrections)."""
        cfg = self.config
        timing: dict[str, float] = {}
        warnings: list[str] = []
        t0 = time.perf_counter()

        lexicon = lexicon_for(text)
        if not lexicon:
            raise ValueError("target text contains no words")
        for wp in lexicon:
            if wp.source == "rules":
                warnings.append(f"'{wp.text}' not in dictionary; rule-based pronunciation used")

        post = self.acoustic.posteriors(audio)
        timing["acoustic"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        dec = [s for s in greedy_decode(post.log_probs, BLANK_ID) if s.token != OTHER_ID]
        hyp = [VOCAB[s.token] for s in dec]
        choice = choose_variants(lexicon, hyp)
        target = build_target(lexicon, choice)
        tp = target.phones
        try:
            segs: list[Segment] | None = forced_align(
                post.log_probs, [PHONE_TO_ID[p.phone] for p in tp], BLANK_ID
            )
        except ValueError as e:
            segs = None
            warnings.append(str(e))
        gop = compute_gop(post, tp, segs, self.calibration)
        ops = align([p.phone for p in tp], hyp, [p.alternatives for p in tp])
        errors = detect_errors(tp, ops, gop, cfg.gop_threshold)
        timing["scoring"] = (time.perf_counter() - t1) * 1000

        realized: dict[int, str] = {}
        for op in ops:
            if op.ref_index is not None:
                realized[op.ref_index] = op.hyp if op.hyp is not None else "<del>"

        phones = self._phone_results(tp, gop, segs, post, realized, errors)

        # --- formants / vowel space / dark L
        t2 = time.perf_counter()
        norm = speaker_norm
        if cfg.formants and segs is not None:
            if norm is None:
                norm = SpeakerNorm.default(cfg.default_gender)
                warnings.append("vowel scores use a default speaker normalisation; run calibration")
            errors += self._acoustic_checks(audio, tp, phones, norm)
        timing["formants"] = (time.perf_counter() - t2) * 1000

        raw = None
        if segs is not None and (self.scorer is not None or cfg.export_features):
            t_s = time.perf_counter()
            raw = ([replace(p) for p in phones], list(errors))  # GOP-based, ungated: utterance-model inputs
            errors = self._learned(post, tp, segs, gop, ops, errors, phones, norm, strictness)
            timing["scorer"] = (time.perf_counter() - t_s) * 1000

        # --- prosody
        t3 = time.perf_counter()
        prosody_out: dict[str, Any] = {}
        stress_by_word: dict[int, dict[str, Any]] = {}
        fluency = None
        if cfg.prosody and segs is not None:
            prosody_out, stress_by_word, fluency = self._prosody(audio, tp, phones, reference_audio)
        timing["prosody"] = (time.perf_counter() - t3) * 1000

        words = self._word_results(target, phones, errors, stress_by_word, choice)
        scores = self._scores(target, words, ops, prosody_out, fluency)
        if self.scorer is not None and raw is not None and self.scorer.meta.get("utt_models"):
            from .scorer import utterance_features

            raw_words = self._word_results(target, raw[0], raw[1], stress_by_word, choice)
            raw_scores = self._scores(target, raw_words, ops, prosody_out, fluency)
            learned = self.scorer.utterance(
                utterance_features(
                    raw_scores, [p.score for p in phones], self.scorer.error_gate, len(audio) / SAMPLE_RATE
                )
            )
            for ours, theirs in (
                ("accuracy", "accuracy"),
                ("fluency", "fluency"),
                ("prosody", "prosodic"),
                ("total", "total"),
            ):
                if theirs in learned:
                    scores[ours] = learned[theirs]
        timing["total"] = (time.perf_counter() - t0) * 1000

        return Assessment(
            text=text,
            words=words,
            errors=[self._error_dict(e, target) for e in errors],
            recognized=hyp,
            scores=scores,
            prosody=prosody_out,
            timing_ms=timing,
            model=self.acoustic.name,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ helpers

    def _learned(
        self,
        post: Posteriors,
        tp: Sequence[TargetPhone],
        segs: Sequence[Segment],
        gop: Sequence[PhoneGOP],
        ops: Sequence[AlignOp],
        errors: list[PronunciationError],
        phones: list[PhoneResult],
        norm: SpeakerNorm | None,
        strictness: str | None = None,
    ) -> list[PronunciationError]:
        from .scorer import phone_features

        hard = {
            e.target_index
            for e in errors
            if e.source == "alignment" and e.expected is not None and e.severity != "minor"
        }
        matched = {op.ref_index for op in ops if op.kind == "match"}
        X = phone_features(
            post,
            tp,
            segs,
            [g.lpr for g in gop],
            [g.gop_frame for g in gop],
            [i in hard for i in range(len(tp))],
            [i in matched for i in range(len(tp))],
            [
                (pr.formants["f1"], pr.formants["f2"]) if pr.formants and is_vowel(t.phone) else None
                for t, pr in zip(tp, phones, strict=True)
            ],
            norm,
        )
        if self.config.export_features:
            for pr, row in zip(phones, X, strict=True):
                pr.features = [float(v) for v in row]
        if self.scorer is None:
            return errors
        learned = self.scorer.score(X)
        for pr, sc in zip(phones, learned, strict=True):
            pr.score = float(sc)
        keep_always = [
            e for e in errors if e.target_index is None or e.expected is None or e.severity == "minor"
        ]
        phone_errs = [e for e in errors if e not in keep_always]
        if self.mdd is None:
            # keep a phone-level error only if the learned scorer agrees the phone is off
            out = keep_always + [e for e in phone_errs if learned[e.target_index] < self.scorer.error_gate]
            return self._cap_reported(out, phones)
        from .errors import classify_substitution
        from .mdd import likely_realisation

        p_err = self.mdd.proba(X, learned)

        def over(i: int, code: str) -> bool:
            return bool(p_err[i] >= self.mdd.threshold_for(strictness, code))  # type: ignore[union-attr]

        out = keep_always + [
            e for e in phone_errs if e.target_index is not None and over(e.target_index, e.code)
        ]
        have = {e.target_index for e in out if e.expected is not None and e.severity != "minor"}
        candidates = np.nonzero(
            p_err >= min(self.mdd.threshold_for(strictness, "P_B"), self.mdd.threshold_for(strictness, None))
        )[0]
        for i in (int(k) for k in candidates):
            if i in have or i in {e.target_index for e in phone_errs}:
                continue
            t = tp[i]
            real = likely_realisation(X[i], t.phone)
            code = (
                ("RHOTIC" if t.phone in ("R", "ER") else "DELETION")
                if real == "<del>"
                else classify_substitution(t, real)
            )
            if over(i, code):
                out.append(PronunciationError(code, i, t.word_index, t.phone, real, "gop", float(p_err[i])))
        out.sort(key=lambda e: (e.word_index, e.target_index if e.target_index is not None else -1))
        return self._cap_reported(out, phones)

    @staticmethod
    def _cap_reported(
        errors: list[PronunciationError], phones: list[PhoneResult]
    ) -> list[PronunciationError]:
        """What the learner sees must agree: a reported (non-minor) error never shows a "good" score."""
        for e in errors:
            if e.target_index is not None and e.expected is not None and e.severity != "minor":
                phones[e.target_index].score = min(phones[e.target_index].score, 45.0)
        return errors

    def _phone_results(
        self,
        tp: Sequence[TargetPhone],
        gop: Sequence[PhoneGOP],
        segs: Sequence[Segment] | None,
        post: Posteriors,
        realized: dict[int, str],
        errors: Sequence[PronunciationError],
    ) -> list[PhoneResult]:
        hard = {
            e.target_index
            for e in errors
            if e.source == "alignment" and e.expected is not None and e.severity != "minor"
        }
        out = []
        for i, (t, g) in enumerate(zip(tp, gop, strict=True)):
            score = g.score
            if i in hard:
                score = min(score, 40.0)
            start = post.frame_to_time(segs[i].start) if segs else float("nan")
            end = post.frame_to_time(segs[i].end) if segs else float("nan")
            out.append(
                PhoneResult(
                    i,
                    t.phone,
                    to_ipa(t.phone, t.stress),
                    t.stress,
                    t.allophone,
                    realized.get(i),
                    score,
                    g.lpr,
                    g.gop_frame,
                    start,
                    end,
                )
            )
        return out

    def _acoustic_checks(
        self,
        audio: NDArray[np.float32],
        tp: Sequence[TargetPhone],
        phones: list[PhoneResult],
        norm: SpeakerNorm,
    ) -> list[PronunciationError]:
        tracker = FormantTracker(audio, SAMPLE_RATE, self.config.max_formant)
        errs: list[PronunciationError] = []
        for t, pr in zip(tp, phones, strict=True):
            if pr.end - pr.start < 0.03:
                continue
            if is_vowel(t.phone):
                fm = tracker.measure(pr.start, pr.end)
                if fm.valid:
                    pr.formants = {"f1": fm.f1, "f2": fm.f2, "f3": fm.f3}
                    if t.phone in REFERENCE and t.stress != 0:
                        vs = score_vowel(t.phone, fm.f1, fm.f2, norm)
                        if vs is not None:
                            pr.vowel_score = asdict(vs)
            elif t.allophone == "dark_l" and self.config.dark_l_check:
                fm = tracker.measure(pr.start, pr.end, portion=0.6)
                if fm.valid:
                    pr.formants = {"f1": fm.f1, "f2": fm.f2, "f3": fm.f3}
                    if is_clear_l(fm.f2, norm.mean[1]):
                        errs.append(
                            PronunciationError(
                                "DARK_L", pr.index, t.word_index, "L", "L(clear)", "acoustic", 0.6
                            )
                        )
        return errs

    def _prosody(
        self,
        audio: NDArray[np.float32],
        tp: Sequence[TargetPhone],
        phones: list[PhoneResult],
        reference_audio: NDArray[np.float32] | None,
    ) -> tuple[dict[str, Any], dict[int, dict[str, Any]], float | None]:
        cfg = self.config
        track = pros.extract(audio, SAMPLE_RATE, cfg.pitch_method)
        utt = (phones[0].start, phones[-1].end)
        pauses = _pauses(track, utt[0], utt[1], cfg.pause_min_s)
        sylls = [
            pros.Syllable(t.word_index, t.syllable_index, p.start, p.end, t.stress or 0)
            for t, p in zip(tp, phones, strict=True)
            if is_vowel(t.phone)
        ]
        prom = pros.syllable_prominence(track, sylls) if len(sylls) >= 2 else []
        lex = pros.lexical_stress(prom)
        stress_by_word = {w.word_index: asdict(w) for w in lex}

        intervals: list[tuple[bool, float]] = []
        for t, p in zip(tp, phones, strict=True):
            d = (p.end - p.start) - sum(_overlap((p.start, p.end), q) for q in pauses)
            if d <= 0:
                continue
            v = is_vowel(t.phone)
            if intervals and intervals[-1][0] == v:
                intervals[-1] = (v, intervals[-1][1] + d)
            else:
                intervals.append((v, d))
        rh = pros.rhythm(intervals)

        speech_time = (utt[1] - utt[0]) - sum(b - a for a, b in pauses)
        rate = len(tp) / speech_time if speech_time > 0 else float("nan")
        fluency = 100.0 - 12.0 * len(pauses)
        if np.isfinite(rate) and rate < 7.0:
            fluency -= (7.0 - rate) * 8.0
        fluency = float(np.clip(fluency, 0, 100))

        out: dict[str, Any] = {
            "track": track.to_dict(),
            "pauses": pauses,
            "articulation_rate_phones_per_s": rate,
            "rhythm": asdict(rh),
            "lexical_stress": [asdict(w) for w in lex],
            "syllables": [
                {
                    "word_index": p.syllable.word_index,
                    "syllable_index": p.syllable.syllable_index,
                    "stress": p.syllable.stress,
                    "prominence": p.prominence,
                    "f0_st": p.f0_st,
                    "energy_db": p.energy_db,
                    "duration": p.duration,
                }
                for p in prom
            ],
        }
        if reference_audio is not None:
            ref_track = pros.extract(reference_audio, SAMPLE_RATE, cfg.pitch_method)
            it = pros.intonation_similarity(track, ref_track)
            out["intonation"] = asdict(it) if it else None
            out["reference_track"] = ref_track.to_dict()
        return out, stress_by_word, fluency

    def _word_results(
        self,
        target: Target,
        phones: list[PhoneResult],
        errors: list[PronunciationError],
        stress: dict[int, dict[str, Any]],
        choice: list[int],
    ) -> list[WordResult]:
        ins_per_word: dict[int, int] = {}
        for e in errors:
            if e.expected is None:
                ins_per_word[e.word_index] = ins_per_word.get(e.word_index, 0) + 1
        out = []
        for wi, w in enumerate(target.words):
            ph = [phones[i] for i in target.word_phone_indices(wi)]
            acc = float(np.mean([p.score for p in ph])) if ph else 0.0
            acc = max(0.0, acc - 10.0 * ins_per_word.get(wi, 0))
            out.append(WordResult(wi, w, acc, ph, stress.get(wi), choice[wi]))
        return out

    def _scores(
        self,
        target: Target,
        words: list[WordResult],
        ops: Sequence[AlignOp],
        prosody_out: dict[str, Any],
        fluency: float | None,
    ) -> dict[str, float | None]:
        n_ph = sum(len(w.phones) for w in words)
        accuracy = sum(w.accuracy * len(w.phones) for w in words) / max(n_ph, 1)
        deleted = {op.ref_index for op in ops if op.kind == "del"}
        complete_words = 0
        for wi in range(len(target.words)):
            idx = target.word_phone_indices(wi)
            if idx and sum(i not in deleted for i in idx) / len(idx) >= 0.5:
                complete_words += 1
        completeness = 100.0 * complete_words / max(len(target.words), 1)

        parts = []
        lex = prosody_out.get("lexical_stress") or []
        stress_score = 100.0 * float(np.mean([w["correct"] for w in lex])) if lex else None
        if stress_score is not None:
            parts.append(stress_score)
        rh = prosody_out.get("rhythm", {}).get("score")
        rhythm_score = rh if rh is not None and np.isfinite(rh) else None
        if rhythm_score is not None and len(target.phones) >= 12:
            parts.append(rhythm_score)
        it = prosody_out.get("intonation")
        intonation_score = it["score"] if it else None
        if intonation_score is not None:
            parts.append(intonation_score)
        prosody_score = float(np.mean(parts)) if parts else None

        weights = {"accuracy": 0.5, "completeness": 0.15, "fluency": 0.15, "prosody": 0.2}
        vals = {
            "accuracy": accuracy,
            "completeness": completeness,
            "fluency": fluency,
            "prosody": prosody_score,
        }
        tot_w = sum(weights[k] for k, v in vals.items() if v is not None)
        total = sum(weights[k] * v for k, v in vals.items() if v is not None) / tot_w
        return {
            **vals,
            "stress": stress_score,
            "rhythm": rhythm_score,
            "intonation": intonation_score,
            "total": total,
        }

    @staticmethod
    def _error_dict(e: PronunciationError, target: Target) -> dict[str, Any]:
        et = TAXONOMY[e.code]
        return {
            "code": e.code,
            "name": et.name,
            "severity": et.severity,
            "tip": et.tip,
            "word_index": e.word_index,
            "word": target.words[e.word_index],
            "target_index": e.target_index,
            "expected": e.expected,
            "expected_ipa": to_ipa(e.expected) if e.expected and e.expected in TAXONOMY_SAFE else None,
            "realized": e.realized,
            "source": e.source,
            "confidence": e.confidence,
        }


TAXONOMY_SAFE = set(VOCAB) - {"<blank>", "<other>"}
