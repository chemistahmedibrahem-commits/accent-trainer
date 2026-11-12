import json

import numpy as np
import pytest

from accent_engine import EngineConfig, PronunciationEngine
from accent_engine.acoustic import SyntheticAcousticModel
from accent_engine.g2p import g2p
from accent_engine.gop import compute_gop

AUDIO = np.zeros(16000 * 2, dtype=np.float32)
FAST = EngineConfig(formants=False, prosody=False)


def run(text: str, spoken: list) -> dict:  # type: ignore[type-arg]
    return PronunciationEngine(SyntheticAcousticModel(spoken), FAST).assess(AUDIO, text).to_dict()


def codes(r: dict) -> list[tuple[str, str | None, str | None]]:  # type: ignore[type-arg]
    return [(e["code"], e["expected"], e["realized"]) for e in r["errors"]]


@pytest.mark.parametrize(
    "text,spoken,expected",
    [
        ("people", "B IY P AH L", ("P_B", "P", "B")),
        ("very", "F EH R IY", ("V_F", "V", "F")),
        ("think", "S IH NG K", ("TH_SUB", "TH", "S")),
        ("think", "T IH NG K", ("TH_SUB", "TH", "T")),
        ("this", "Z IH S", ("DH_SUB", "DH", "Z")),
        ("ship", "SH IY P", ("IH_IY", "IH", "IY")),
        ("full", "F UW L", ("UH_UW", "UH", "UW")),
        ("bad", "B EH D", ("AE_EH", "AE", "EH")),
        ("car", "K AA", ("RHOTIC", "R", "<del>")),
        ("water", "W AO T ER", ("FLAP_T", "DX", "T")),
        ("street", "IH S T R IY T", ("EPENTHESIS", None, "IH")),
        ("bag", "B AE K", ("FINAL_DEVOICING", "G", "K")),
        ("problem", "P R AA B L AE M", ("SCHWA_REDUCTION", "AH", "AE")),
    ],
)
def test_taxonomy(text: str, spoken: str, expected: tuple) -> None:  # type: ignore[type-arg]
    r = run(text, spoken.split())
    assert expected in codes(r), codes(r)


def test_correct_pronunciation_has_no_errors_and_high_scores() -> None:
    r = run("think about it", ["TH", "IH", "NG", "K", "AH", "B", "AW", "DX", "IH", "T"])
    assert r["errors"] == []
    assert r["scores"]["accuracy"] > 90
    assert r["scores"]["completeness"] == 100


def test_weak_and_strong_forms_both_accepted() -> None:
    weak = run("to go", ["T", "AH", "G", "OW"])
    strong = run("to go", ["T", "UW", "G", "OW"])
    assert weak["errors"] == [] and strong["errors"] == []
    assert strong["words"][0]["variant"] == 1


def test_glottal_t_accepted() -> None:
    assert run("button", ["B", "AH", "Q", "AH", "N"])["errors"] == []


def test_gop_only_detection_on_ambiguous_phone() -> None:
    # aligned as P but the acoustics are mostly B → GOP flags it even though greedy says P
    spoken = [("P", 0.45, "B"), "IY", "P", "AH", "L"]
    r = run("people", spoken)
    assert r["words"][0]["phones"][0]["score"] < 50
    assert ("P_B", "P", "B") in codes(r)


def test_gop_monotonic_in_confidence() -> None:
    t = g2p("pat").phones
    scores = []
    for conf in (0.95, 0.6, 0.3):
        post = SyntheticAcousticModel([("P", conf, "B"), "AE", "T"]).posteriors(AUDIO)
        scores.append(compute_gop(post, t)[0].score)
    assert scores[0] > scores[1] > scores[2]


def test_deletion_lowers_completeness() -> None:
    r = run("big dog", ["B", "IH", "G"])
    assert r["scores"]["completeness"] == 50


def test_output_is_json_serialisable_with_full_pipeline() -> None:
    from accent_engine.synth import concat, silence, vowel

    audio = concat(
        silence(0.2), vowel((700, 1220), 0.25, 130), silence(0.05), vowel((500, 1500), 0.2, 110), silence(0.2)
    )
    eng = PronunciationEngine(SyntheticAcousticModel(["AH", "B", "AW", "T"]))
    r = eng.assess(audio, "about", reference_audio=audio).to_dict()
    json.dumps(r)
    assert r["prosody"]["intonation"]["correlation"] > 0.95
    assert "track" in r["prosody"]


def test_windowed_gop_agrees_with_full_computation() -> None:
    from accent_engine.acoustic import BLANK_ID, PHONE_TO_ID
    from accent_engine.ctc import forced_align

    text = "the early bird gets the worm"
    tp = g2p(text).phones
    spoken: list = [p.phone for p in tp]
    spoken[3] = ("ER", 0.5, "AH")  # one ambiguous phone
    spoken[7] = "D"  # one wrong phone
    post = SyntheticAcousticModel(spoken).posteriors(np.zeros(16000 * 3, dtype=np.float32))
    segs = forced_align(post.log_probs, [PHONE_TO_ID[p.phone] for p in tp], BLANK_ID)
    full = compute_gop(post, tp, segs, window=None)
    win = compute_gop(post, tp, segs, window=2)
    lpr_full = np.array([g.lpr for g in full])
    lpr_win = np.array([g.lpr for g in win])
    assert np.corrcoef(lpr_full, lpr_win)[0, 1] > 0.99
    assert [g.best_competitor for g in full] == [g.best_competitor for g in win]
    assert np.max(np.abs(lpr_full - lpr_win)) < 0.5
