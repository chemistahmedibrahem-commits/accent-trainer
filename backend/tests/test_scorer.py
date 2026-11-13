import numpy as np

from accent_engine.acoustic import BLANK_ID, PHONE_TO_ID, SyntheticAcousticModel
from accent_engine.ctc import forced_align
from accent_engine.g2p import g2p
from accent_engine.gop import compute_gop
from accent_engine.scorer import CANDIDATES, FEATURE_NAMES, lpr_matrix, phone_features

AUDIO = np.zeros(16000 * 2, dtype=np.float32)


def _setup(spoken: list[str], text: str):  # type: ignore[no-untyped-def]
    tp = g2p(text).phones
    post = SyntheticAcousticModel(spoken).posteriors(AUDIO)
    segs = forced_align(post.log_probs, [PHONE_TO_ID[p.phone] for p in tp], BLANK_ID)
    return tp, post, segs


def test_lpr_matrix_identifies_the_substituted_phone() -> None:
    tp, post, segs = _setup(["S", "IH", "NG", "K"], "think")
    M = lpr_matrix(post, tp, segs)
    assert M.shape == (4, len(CANDIDATES) + 1)
    row = M[0]
    assert row[CANDIDATES.index("S")] < 0  # "sink" explains the audio better than "think"
    assert int(np.argmin(row[:-1])) == CANDIDATES.index("S")
    for i in range(1, 4):  # other phones were said correctly: every substitution is worse
        own = CANDIDATES.index(tp[i].phone)
        assert np.delete(M[i, :-1], own).min() > 0


def test_lpr_matrix_bounds_compute_gop() -> None:
    tp, post, segs = _setup(["TH", "IH", "NG", "K"], "think")
    M = lpr_matrix(post, tp, segs)
    gop = compute_gop(post, tp, segs)
    for i, g in enumerate(gop):
        own = CANDIDATES.index(tp[i].phone)
        best_competitor = np.delete(M[i], own).min()
        assert g.lpr >= best_competitor - 1e-6  # compute_gop uses a subset of competitors


def test_phone_features_shape_and_names() -> None:
    tp, post, segs = _setup(["TH", "IH", "NG", "K"], "think")
    gop = compute_gop(post, tp, segs)
    X = phone_features(
        post, tp, segs, [g.lpr for g in gop], [g.gop_frame for g in gop], [False] * 4, [True] * 4
    )
    assert X.shape == (4, len(FEATURE_NAMES))
    core = FEATURE_NAMES.index("err_mean")
    assert np.isfinite(X[:, :core]).all()
    assert np.isnan(X[:, core:]).all()  # no _err channel, no formants → missing (HistGBR handles NaN)


def test_vowel_features_rank_the_target_vowel() -> None:
    from accent_engine.scorer import vowel_feats
    from accent_engine.vowel_space import HILLENBRAND_MALE, SpeakerNorm

    norm = SpeakerNorm.default("male")
    good = vowel_feats("IH", *HILLENBRAND_MALE["IH"], norm)
    bad = vowel_feats("IH", *HILLENBRAND_MALE["IY"], norm)  # "sheep" for "ship"
    assert good[2] < bad[2]  # distance to target
    assert good[3] < 0 < bad[3]  # closer to target than to any other vowel only when correct
