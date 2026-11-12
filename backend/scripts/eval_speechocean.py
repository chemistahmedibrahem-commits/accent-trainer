"""Evaluate (and calibrate) the scorer against speechocean762 human ratings.

Usage (needs the `ml` extra and the dataset from https://www.openslr.org/101/):
    python scripts/eval_speechocean.py --root /data/speechocean762 [--limit 500] [--out report.json]

Reports Pearson correlation between engine phone measures and the human phone accuracy
(0–2), fits the GOP logistic calibration on *train* and evaluates it on *test*, and
correlates utterance-level scores (accuracy / fluency / prosodic / total).
Target from the spec: phone-level PCC ≥ 0.6.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accent_engine.acoustic import Wav2Vec2PhonemeModel
from accent_engine.alignment import align
from accent_engine.engine import EngineConfig, PronunciationEngine
from accent_engine.g2p import g2p


def load_split(root: Path, split: str) -> list[tuple[str, Path, str]]:
    wavs = dict(
        line.split(maxsplit=1) for line in (root / split / "wav.scp").read_text().splitlines() if line
    )
    texts = dict(line.split(maxsplit=1) for line in (root / split / "text").read_text().splitlines() if line)
    return [(u, root / wavs[u].strip(), texts[u].strip()) for u in sorted(wavs) if u in texts]


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def collect(engine: PronunciationEngine, items, scores: dict, limit: int | None):  # type: ignore[no-untyped-def]
    phone_rows: list[tuple[float, float, float, int, float]] = []  # lpr, gop_frame, score, hard_err, human
    utt_rows: dict[str, list[tuple[float, float]]] = {
        k: [] for k in ("accuracy", "fluency", "prosodic", "total")
    }
    for n, (uid, wav, text) in enumerate(items):
        if limit and n >= limit:
            break
        audio, sr = sf.read(wav, dtype="float32")
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        try:
            r = engine.assess(audio, text).to_dict()
        except Exception as e:
            print(f"skip {uid}: {e}", file=sys.stderr)
            continue
        human = scores[uid]
        target = g2p(text)
        errs = {e["target_index"] for e in r["errors"] if e["source"] == "alignment"}
        ours = [p for w in r["words"] for p in w["phones"]]
        for wi, hw in enumerate(human["words"]):
            if wi >= len(r["words"]):
                break
            h_ph = [re.sub(r"\d", "", p).upper() for p in hw["phones"]]
            h_acc = hw["phones-accuracy"]
            o_idx = [p["index"] for p in r["words"][wi]["phones"]]
            o_under = [
                target.phones[i].underlying if i < len(target.phones) else ours[i]["phone"] for i in o_idx
            ]
            for op in align(h_ph, o_under):
                if op.kind in ("match", "sub") and op.ref_index is not None and op.hyp_index is not None:
                    p = ours[o_idx[op.hyp_index]]
                    phone_rows.append(
                        (
                            p["lpr"],
                            p["gop_frame"],
                            p["score"],
                            int(p["index"] in errs),
                            float(h_acc[op.ref_index]),
                        )
                    )
        for k, ours_k in (
            ("accuracy", "accuracy"),
            ("fluency", "fluency"),
            ("prosodic", "prosody"),
            ("total", "total"),
        ):
            v = r["scores"].get(ours_k)
            if v is not None:
                utt_rows[k].append((v, float(human[k])))
        if n % 100 == 0:
            print(f"{n} utterances", file=sys.stderr)
    return phone_rows, utt_rows


def fit_logistic(lpr: np.ndarray, human: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of 2*sigmoid(a*lpr+b) to human accuracy (0..2)."""
    from scipy.optimize import curve_fit

    (a, b), _ = curve_fit(
        lambda x, a, b: 2 / (1 + np.exp(-(a * x + b))), lpr, human, p0=[1.0, 0.0], maxfev=10000
    )
    return float(a), float(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--model", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("speechocean_report.json"))
    args = ap.parse_args()

    scores = json.loads((args.root / "resource" / "scores.json").read_text())
    eng = PronunciationEngine(Wav2Vec2PhonemeModel(args.model), EngineConfig(formants=False, prosody=True))

    train_ph, _ = collect(eng, load_split(args.root, "train"), scores, args.limit)
    test_ph, test_utt = collect(eng, load_split(args.root, "test"), scores, args.limit)
    tr, te = np.array(train_ph), np.array(test_ph)
    a, b = fit_logistic(np.clip(tr[:, 0], -20, 20), tr[:, 4])
    cal_test = 2 / (1 + np.exp(-(a * np.clip(te[:, 0], -20, 20) + b)))

    report = {
        "n_phones_test": len(te),
        "phone_pcc": {
            "lpr": pearson(list(te[:, 0]), list(te[:, 4])),
            "gop_frame": pearson(list(te[:, 1]), list(te[:, 4])),
            "score_uncalibrated": pearson(list(te[:, 2]), list(te[:, 4])),
            "score_calibrated": pearson(list(cal_test), list(te[:, 4])),
            "alignment_error_flag": pearson(list(-te[:, 3]), list(te[:, 4])),
        },
        "calibration": {"slope": a, "intercept": b},
        "utterance_pcc": {k: pearson([x for x, _ in v], [y for _, y in v]) for k, v in test_utt.items()},
        "target_phone_pcc": 0.6,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
