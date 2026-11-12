"""Evaluate and calibrate the scorer against speechocean762 human ratings.

    # 1) extract engine features for every phone (slow: runs the acoustic model)
    python scripts/eval_speechocean.py extract --root /data/speechocean762 --model MODEL --out feats.jsonl
    # 2) fit calibration on train, report Pearson on test (fast, re-runnable)
    python scripts/eval_speechocean.py fit --feats feats.jsonl --calibration-out calibration.json

Phone level: human accuracy 0–2 per phone. Utterance level: accuracy / fluency / prosodic /
total 0–10. Target from the spec: phone-level PCC ≥ 0.6.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accent_engine.alignment import align
from accent_engine.engine import EngineConfig, PronunciationEngine
from accent_engine.g2p import build_target, lexicon_for

FEATURES = ("lpr", "gop_frame", "hard_err", "margin_clip")


def load_split(root: Path, split: str) -> list[tuple[str, Path, str]]:
    def kv(p: Path) -> dict[str, str]:
        return dict(line.split(maxsplit=1) for line in p.read_text().splitlines() if line.strip())

    wavs, texts = kv(root / split / "wav.scp"), kv(root / split / "text")
    return [(u, root / wavs[u].strip(), texts[u].strip()) for u in sorted(wavs) if u in texts]


class CachedModel:
    """Wraps an acoustic model with an on-disk posterior cache keyed by utterance id."""

    def __init__(self, model_id: str, cache_dir: Path | None) -> None:
        self.model_id, self.cache_dir = model_id, cache_dir
        self.name = model_id
        self.uid = ""
        self._model: Any = None

    def posteriors(self, audio: np.ndarray) -> Any:
        from accent_engine.acoustic import Posteriors, Wav2Vec2PhonemeModel

        f = self.cache_dir / f"{self.uid}.npz" if self.cache_dir else None
        if f is not None and f.exists():
            z = np.load(f)
            return Posteriors(z["lp"].astype(np.float64), float(z["shift"]))
        if self._model is None:
            self._model = Wav2Vec2PhonemeModel(self.model_id)
        post = self._model.posteriors(audio)
        if f is not None:
            np.savez_compressed(f, lp=post.log_probs.astype(np.float16), shift=post.frame_shift)
        return post


def extract(args: argparse.Namespace) -> None:
    scores = json.loads((args.root / "resource" / "scores.json").read_text())
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    model = CachedModel(args.model, args.cache_dir)
    cfg = EngineConfig(formants=False, prosody=True, export_features=True, scorer_path=args.scorer)
    eng = PronunciationEngine(model, cfg)
    t0 = time.time()
    with open(args.out, "w") as out:
        for split in ("train", "test"):
            items = load_split(args.root, split)[: args.limit or None]
            for n, (uid, wav, text) in enumerate(items):
                audio, sr = sf.read(wav, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                model.uid = uid
                try:
                    r = eng.assess(audio, text).to_dict()
                except Exception as e:
                    print(f"skip {uid}: {e}", file=sys.stderr)
                    continue
                human = scores[uid]
                target = build_target(lexicon_for(text), [w["variant"] for w in r["words"]])
                flagged = {e["target_index"] for e in r["errors"] if e["expected"] is not None}
                hard = {
                    e["target_index"]
                    for e in r["errors"]
                    if e["source"] == "alignment" and e["severity"] != "minor"
                }
                phones = []
                for wi, hw in enumerate(human["words"]):
                    if wi >= len(r["words"]):
                        break
                    h_ph = [re.sub(r"\d", "", p).upper() for p in hw["phones"]]
                    ours = r["words"][wi]["phones"]
                    under = [target.phones[p["index"]].underlying for p in ours]
                    for op in align(h_ph, under):
                        if (
                            op.kind in ("match", "sub")
                            and op.ref_index is not None
                            and op.hyp_index is not None
                        ):
                            p = ours[op.hyp_index]
                            phones.append(
                                {
                                    "phone": h_ph[op.ref_index],
                                    "human": float(hw["phones-accuracy"][op.ref_index]),
                                    "lpr": p["lpr"],
                                    "gop_frame": p["gop_frame"],
                                    "score": p["score"],
                                    "hard_err": int(p["index"] in hard),
                                    "realized": p["realized"],
                                    "features": p.get("features"),
                                    "flagged": int(p["index"] in flagged),
                                }
                            )
                out.write(
                    json.dumps(
                        {
                            "uid": uid,
                            "split": split,
                            "phones": phones,
                            "ours": {
                                k: r["scores"].get(k)
                                for k in ("accuracy", "completeness", "fluency", "prosody", "total")
                            },
                            "human": {
                                k: human[k]
                                for k in ("accuracy", "completeness", "fluency", "prosodic", "total")
                            },
                            "acoustic_ms": r["timing_ms"]["acoustic"],
                            "audio_s": len(audio) / 16000,
                        }
                    )
                    + "\n"
                )
                if n % 250 == 0:
                    print(f"{split} {n}/{len(items)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


def _feat(p: dict[str, Any]) -> list[float]:
    lpr = float(np.clip(p["lpr"], -30, 30))
    return [lpr, float(np.clip(p["gop_frame"], -30, 0)), float(p["hard_err"]), float(np.clip(lpr, -5, 5))]


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def fit(args: argparse.Namespace) -> None:
    from scipy.optimize import minimize

    with open(args.feats) as fh:
        rows = [json.loads(line) for line in fh]
    tr = [p for r in rows if r["split"] == "train" for p in r["phones"]]
    te = [p for r in rows if r["split"] == "test" for p in r["phones"]]
    Xtr, Xte = np.array([_feat(p) for p in tr]), np.array([_feat(p) for p in te])
    ytr, yte = np.array([p["human"] for p in tr]) / 2, np.array([p["human"] for p in te]) / 2

    def pred(w: np.ndarray, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(X @ w[:-1] + w[-1])))

    res = minimize(
        lambda w: float(np.mean((pred(w, Xtr) - ytr) ** 2)), np.zeros(Xtr.shape[1] + 1), method="L-BFGS-B"
    )
    w = res.x
    report: dict[str, Any] = {
        "n_phones": {"train": len(tr), "test": len(te)},
        "phone_pcc_test": {
            "calibrated": pcc(pred(w, Xte), yte),
            "lpr_raw": pcc(Xte[:, 0], yte),
            "gop_frame_raw": pcc(Xte[:, 1], yte),
            "engine_score_uncalibrated": pcc(np.array([p["score"] for p in te]), yte),
            "hard_err_flag": pcc(-Xte[:, 2], yte),
        },
        "target_phone_pcc": 0.6,
        "calibration": {
            "features": list(FEATURES),
            "weights": [float(x) for x in w[:-1]],
            "intercept": float(w[-1]),
        },
    }
    # mispronunciation detection (human score < 1.5 ≈ "noticeably wrong")
    bad = yte < 0.75
    flagged = pred(w, Xte) < 0.75
    tp = int((bad & flagged).sum())
    report["mdd_test"] = {
        "precision": tp / max(int(flagged.sum()), 1),
        "recall": tp / max(int(bad.sum()), 1),
        "human_bad_rate": float(bad.mean()),
    }
    utt = [r for r in rows if r["split"] == "test"]
    report["utterance_pcc_test"] = {
        k: pcc(
            np.array([r["ours"][o] for r in utt if r["ours"][o] is not None], dtype=float),
            np.array([r["human"][k] for r in utt if r["ours"][o] is not None], dtype=float),
        )
        for k, o in (
            ("accuracy", "accuracy"),
            ("fluency", "fluency"),
            ("prosodic", "prosody"),
            ("total", "total"),
        )
    }
    a_ms = sum(r["acoustic_ms"] for r in rows) / 1000
    report["acoustic_rtf"] = a_ms / sum(r["audio_s"] for r in rows)
    print(json.dumps(report, indent=2))
    if args.calibration_out:
        Path(args.calibration_out).write_text(json.dumps(report["calibration"], indent=2))


def train_scorer(args: argparse.Namespace) -> None:
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, cross_val_predict

    from accent_engine.scorer import CATEGORICAL, FEATURE_NAMES

    with open(args.feats) as fh:
        rows = [json.loads(line) for line in fh]

    def xy(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ph = [(r["uid"], p) for r in rows if r["split"] == split for p in r["phones"] if p.get("features")]
        X = np.array([p["features"] for _, p in ph], dtype=np.float64)
        y = np.array([p["human"] for _, p in ph], dtype=np.float64)
        g = np.array([int(u[:5]) for u, _ in ph])  # speaker id → grouped CV (no speaker leakage)
        return X, y, g

    Xtr, ytr, gtr = xy("train")
    Xte, yte, _ = xy("test")
    assert Xtr.shape[1] == len(FEATURE_NAMES), (Xtr.shape, len(FEATURE_NAMES))
    cat = np.zeros(Xtr.shape[1], dtype=bool)
    cat[list(CATEGORICAL)] = True

    def make() -> Any:
        return HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            categorical_features=cat,
            random_state=0,
        )

    oof = cross_val_predict(make(), Xtr, ytr, groups=gtr, cv=GroupKFold(min(5, len(set(gtr.tolist())))))
    bad_tr = ytr < 1.5
    best = (0.0, 60.0)
    for gate in np.arange(30, 90, 2.5):
        flag = np.clip(oof, 0, 2) * 50 < gate
        tp = (flag & bad_tr).sum()
        prec, rec = tp / max(flag.sum(), 1), tp / max(bad_tr.sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        best = max(best, (float(f1), float(gate)))
    model = make().fit(Xtr, ytr)
    pred = np.clip(model.predict(Xte), 0, 2)
    gate = best[1]
    flag, bad = pred * 50 < gate, yte < 1.5
    tp = int((flag & bad).sum())
    meta = {
        "acoustic_model": args.model_id,
        "n_train": len(ytr),
        "n_test": len(yte),
        "pcc_test": pcc(pred, yte),
        "pcc_oof_train": pcc(oof, ytr),
        "error_gate": gate,
        "mdd_test": {
            "precision": tp / max(int(flag.sum()), 1),
            "recall": tp / max(int(bad.sum()), 1),
            "flag_rate": float(flag.mean()),
            "human_bad_rate": float(bad.mean()),
        },
        "baseline_lpr_pcc_test": pcc(Xte[:, 0], yte),
    }
    print(json.dumps(meta, indent=2))
    if args.out:
        joblib.dump({"model": model, "feature_names": list(FEATURE_NAMES), "meta": meta}, args.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--model", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    e.add_argument("--limit", type=int, default=0, help="max utterances per split (0 = all)")
    e.add_argument("--out", type=Path, default=Path("so762_feats.jsonl"))
    e.add_argument("--cache-dir", type=Path, default=None, help="posterior cache (npz per utterance)")
    e.add_argument("--scorer", default=None, help="evaluate with a trained scorer")
    f = sub.add_parser("fit")
    f.add_argument("--feats", type=Path, required=True)
    f.add_argument("--calibration-out", type=Path, default=None)
    t = sub.add_parser("train-scorer")
    t.add_argument("--feats", type=Path, required=True)
    t.add_argument("--model-id", default="")
    t.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    {"extract": extract, "fit": fit, "train-scorer": train_scorer}[args.cmd](args)


if __name__ == "__main__":
    main()
