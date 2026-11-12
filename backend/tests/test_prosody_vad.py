import numpy as np

from accent_engine import prosody as P
from accent_engine.synth import SR, concat, silence, vowel
from accent_engine.vad import UtteranceSegmenter


def rising(t: np.ndarray) -> np.ndarray:
    return 100 + 150 * t / t[-1]


def falling(t: np.ndarray) -> np.ndarray:
    return 250 - 150 * t / t[-1]


def test_f0_tracks_known_contour() -> None:
    a = vowel((600, 1200), 0.6, 150.0)
    tr = P.extract(a)
    v = tr.f0[np.isfinite(tr.f0)]
    assert len(v) > 30 and abs(np.median(v) - 150) < 3


def test_intonation_similarity_discriminates() -> None:
    up = P.extract(vowel((600, 1200), 0.8, rising))
    up_slow = P.extract(vowel((600, 1200), 1.1, rising))  # same shape, different tempo
    down = P.extract(vowel((600, 1200), 0.8, falling))
    same = P.intonation_similarity(up_slow, up)
    diff = P.intonation_similarity(down, up)
    assert same is not None and diff is not None
    assert same.correlation > 0.9 and same.score > 80
    assert diff.correlation < 0 and diff.score < same.score - 30


def test_flat_intonation_has_low_range_ratio() -> None:
    ref = P.extract(vowel((600, 1200), 0.8, rising))
    flat = P.extract(vowel((600, 1200), 0.8, 130.0))
    s = P.intonation_similarity(flat, ref)
    assert s is not None and s.range_ratio < 0.3


def test_lexical_stress_detection() -> None:
    # REcord: first syllable louder, longer, higher
    strong = vowel((700, 1200), 0.25, 180.0, amplitude=0.8)
    weak = vowel((500, 1500), 0.12, 120.0, amplitude=0.25)
    gap = silence(0.06)
    audio = concat(silence(0.1), strong, gap, weak, silence(0.1))
    tr = P.extract(audio)
    s1 = P.Syllable(0, 0, 0.1, 0.35, 1)
    s2 = P.Syllable(0, 1, 0.41, 0.53, 0)
    ok = P.lexical_stress(P.syllable_prominence(tr, [s1, s2]))
    assert ok[0].correct and ok[0].margin > 0
    # reCORD expected → same audio is wrong
    bad = P.lexical_stress(
        P.syllable_prominence(tr, [P.Syllable(0, 0, 0.1, 0.35, 0), P.Syllable(0, 1, 0.41, 0.53, 1)])
    )
    assert not bad[0].correct


def test_npvi_and_rhythm() -> None:
    assert P.npvi([0.1, 0.1, 0.1]) == 0
    assert np.isclose(P.npvi([0.1, 0.3]), 100.0)

    def runs(vs: list[float]) -> list[tuple[bool, float]]:
        out: list[tuple[bool, float]] = []
        for v in vs:
            out += [(True, v), (False, 0.09)]
        return out[:-1]

    stress_timed = P.rhythm(runs([0.15, 0.08, 0.16, 0.09]))  # nPVI ≈ 61, like GA
    syllable_timed = P.rhythm(runs([0.12, 0.11, 0.12, 0.11]))
    exaggerated = P.rhythm(runs([0.2, 0.04, 0.22, 0.03]))
    assert 55 < stress_timed.npvi_v < 65 and syllable_timed.npvi_v < 15
    assert stress_timed.score > 95
    assert stress_timed.score > syllable_timed.score
    assert stress_timed.score > exaggerated.score
    assert syllable_timed.npvi_v < 15


def test_vad_segments_one_utterance() -> None:
    audio = concat(
        silence(0.5, -60),
        vowel((600, 1200), 0.4),
        silence(0.08, -60),
        vowel((400, 2000), 0.4),
        silence(1.0, -60),
    )
    seg = UtteranceSegmenter()
    events = []
    for i in range(0, len(audio), 640):  # 40 ms client chunks
        events += seg.push(audio[i : i + 640])
    kinds = [e.kind for e in events]
    assert kinds == ["speech_start", "speech_end"]
    utt = events[1].audio
    assert utt is not None
    dur = len(utt) / SR
    assert 0.85 < dur < 1.4  # 0.88 s speech + ≤0.3 s preroll + ≤0.15 s tail


def test_vad_ignores_clicks() -> None:
    audio = concat(silence(0.5, -60), vowel((600, 1200), 0.05), silence(1.0, -60))
    seg = UtteranceSegmenter()
    events = seg.push(audio) + seg.flush()
    assert not any(e.kind == "speech_end" for e in events)
