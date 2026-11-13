from pathlib import Path

import numpy as np
import pytest

from accent_engine import EngineConfig, PronunciationEngine
from accent_engine.acoustic import SyntheticAcousticModel
from accent_engine.g2p import g2p
from accent_engine.mdd import MDD_FEATURES, MDDModel, likely_realisation
from accent_engine.scorer import CANDIDATES, FEATURE_NAMES

AUDIO = np.zeros(32000, np.float32)
SCORER = Path(__file__).resolve().parents[1] / "models" / "scorer_slplab.joblib"


def test_likely_realisation_picks_best_competitor() -> None:
    row = np.full(len(FEATURE_NAMES), 10.0)
    lpr0 = FEATURE_NAMES.index(f"lpr_vs_{CANDIDATES[0]}")
    row[lpr0 + CANDIDATES.index("S")] = -3.0  # "S" explains the audio best
    assert likely_realisation(row, "TH") == "S"
    row[lpr0 + len(CANDIDATES)] = -5.0  # deletion even better
    assert likely_realisation(row, "TH") == "<del>"
    assert MDD_FEATURES[-1] == "so762_score"


class _Stub:
    """predict_proba: flag exactly the phone with the lowest so762 score (last column)."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = (X[:, -1] == X[:, -1].min()).astype(float)
        return np.column_stack([1 - p, p])


@pytest.mark.skipif(not SCORER.exists(), reason="scorer not shipped")
def test_engine_reports_mdd_flags_with_diagnosis() -> None:
    text = "think about it"
    tp = [p.phone for p in g2p(text).phones]
    tp[0] = "S"
    eng = PronunciationEngine(SyntheticAcousticModel(tp), EngineConfig(scorer_path=str(SCORER)))
    eng.mdd = MDDModel(_Stub(), 0.5, {})
    r = eng.assess(AUDIO, text).to_dict()
    th = [e for e in r["errors"] if e["word"] == "think" and e["expected"] == "TH"]
    assert th and th[0]["code"] == "TH_SUB" and th[0]["realized"] == "S"
    # only the flagged phone is reported, and its displayed score agrees with the report
    phone_errs = [e for e in r["errors"] if e["expected"] is not None and e["severity"] != "minor"]
    assert [e["target_index"] for e in phone_errs] == [0]
    assert r["words"][0]["phones"][0]["score"] <= 45


def test_strictness_thresholds() -> None:
    m = MDDModel(_Stub(), 0.6, {}, {"balanced": 0.6, "gentle": 0.8})
    assert m.threshold_for(None) == 0.6
    assert m.threshold_for("gentle") == 0.8
    assert m.threshold_for("unknown") == 0.6
    assert MDDModel(_Stub(), 0.5, {}).threshold_for("gentle") == 0.5  # older bundles: one threshold
