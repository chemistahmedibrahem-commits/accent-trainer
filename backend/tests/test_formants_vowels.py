import numpy as np
import pytest

from accent_engine.formants import FormantTracker
from accent_engine.synth import concat, silence, vowel
from accent_engine.vowel_space import HILLENBRAND_MALE, SpeakerNorm, score_vowel


@pytest.mark.parametrize("v", ["IY", "IH", "EH", "AE", "AA", "AO", "UW", "ER"])
def test_formants_recovered_from_synthetic_vowels(v: str) -> None:
    f1, f2 = HILLENBRAND_MALE[v]
    audio = concat(silence(0.1), vowel((f1, f2), 0.3, 120), silence(0.1))
    m = FormantTracker(audio, max_formant=5000).measure(0.1, 0.4)
    assert m.valid
    assert abs(m.f1 - f1) / f1 < 0.07
    assert abs(m.f2 - f2) / f2 < 0.05


def test_speaker_norm_self_consistency() -> None:
    norm = SpeakerNorm.default("male")
    for v, (f1, f2) in HILLENBRAND_MALE.items():
        s = score_vowel(v, f1, f2, norm)
        assert s is not None and s.score > 85


def test_female_vowels_match_after_normalisation() -> None:
    from accent_engine.vowel_space import HILLENBRAND_FEMALE

    norm = SpeakerNorm.default("female")
    for v, (f1, f2) in HILLENBRAND_FEMALE.items():
        s = score_vowel(v, f1, f2, norm)
        assert s is not None and s.nearest == v


def test_confusable_vowel_scores_low() -> None:
    norm = SpeakerNorm.default("male")
    f1, f2 = HILLENBRAND_MALE["IY"]
    s = score_vowel("IH", f1, f2, norm)  # said "sheep" for "ship"
    assert s is not None and s.nearest == "IY" and s.score < 85


def test_calibration_requires_three_vowels() -> None:
    with pytest.raises(ValueError):
        SpeakerNorm.from_tokens([("IY", 300, 2300), ("AA", 700, 1100)])
    n = SpeakerNorm.from_tokens([("IY", 300, 2300), ("AA", 700, 1100), ("UW", 320, 900)])
    assert np.isclose(n.mean[0], np.mean([300, 700, 320]))
