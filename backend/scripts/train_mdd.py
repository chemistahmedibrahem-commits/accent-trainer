"""Train / evaluate the mispronunciation detector on L2-ARCTIC categorical annotations.

    # 1) features for every annotated utterance of every speaker (posteriors cached)
    python scripts/train_mdd.py extract --root DATA/l2arctic --model MODEL --cache-dir cache --out mdd.jsonl
    # 2) leave-one-Arabic-speaker-out evaluation + final model on all speakers
    python scripts/train_mdd.py train --feats mdd.jsonl --scorer models/scorer_slplab.joblib \\
        --out models/mdd_slplab.joblib

Arabic-L1 speakers in L2-ARCTIC: ABA, SKA, YBAA, ZHAA. Each is evaluated by a model that never
saw that speaker (trained on the other 23 speakers), against the current detector as baseline.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_l2arctic import category, parse, read_phone_tier
from eval_speechocean import CachedModel

from accent_engine import EngineConfig, PronunciationEngine
from accent_engine.alignment import align
from accent_engine.g2p import build_target, lexicon_for

ARABIC = ("ABA", "SKA", "YBAA", "ZHAA")
CORE_CATEGORIES = {"P_B", "TH", "DH", "V_F", "rhotic"}  # mirrors accent_engine.mdd.CORE_CODES
GENTLE_PRECISION = 0.60  # "gentle" feedback: fewer, surer corrections


def extract(args: argparse.Namespace) -> None:
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    model = CachedModel(args.model, args.cache_dir)
    eng = PronunciationEngine(model, EngineConfig(formants=True, prosody=False, export_features=True))
    if args.native_index:
        _extract_native(args, eng, model)
        return
    speakers = sorted(p.name for p in args.root.iterdir() if (p / "annotation").is_dir())
    with open(args.out, "w") as out:
        for spk in speakers:
            d = args.root / spk
            n = 0
            for g in sorted((d / "annotation").glob("*.TextGrid")):
                ann = parse(read_phone_tier(g))
                wav, txt = d / "wav" / f"{g.stem}.wav", d / "transcript" / f"{g.stem}.txt"
                if not ann or not wav.exists() or not txt.exists():
                    continue
                text = txt.read_text().strip()
                audio, sr = sf.read(wav, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                model.uid = f"{spk}_{g.stem}"
                try:
                    r = eng.assess(audio, text).to_dict()
                except Exception as e:
                    print(f"skip {spk}/{g.stem}: {e}", file=sys.stderr)
                    continue
                target = build_target(lexicon_for(text), [w["variant"] for w in r["words"]])
                flagged = {e["target_index"]: e["realized"] for e in r["errors"] if e["expected"] is not None}
                ours = [p for w in r["words"] for p in w["phones"]]
                rows = []
                for op in align([a[0] for a in ann], [p.underlying for p in target.phones]):
                    if op.ref_index is None or op.hyp_index is None:
                        continue
                    canon, perceived, kind = ann[op.ref_index]
                    p = ours[op.hyp_index]
                    rows.append(
                        {
                            "features": p.get("features"),
                            "our_phone": p["phone"],
                            "stress": p["stress"],
                            "f1f2": [p["formants"]["f1"], p["formants"]["f2"]] if p.get("formants") else None,
                            "wrong": int(kind in ("s", "d")),
                            "canon": canon,
                            "perceived": perceived,
                            "kind": kind,
                            "baseline_flag": int(op.hyp_index in flagged),
                            "baseline_diag": flagged.get(op.hyp_index),
                        }
                    )
                out.write(json.dumps({"spk": spk, "utt": g.stem, "phones": rows}) + "\n")
                n += 1
            print(f"{spk}: {n} utterances", file=sys.stderr, flush=True)


def _extract_native(args: argparse.Namespace, eng: PronunciationEngine, model: CachedModel) -> None:
    """Native American speakers as negatives: every phone labelled correct (appends to --out)."""
    n = 0
    with open(args.native_index) as fh, open(args.out, "a") as out:
        for line in fh:
            wav, text, spk = line.rstrip("\n").split("\t")
            audio, _ = sf.read(wav, dtype="float32")
            model.uid = f"NAT_{Path(wav).stem}"
            try:
                r = eng.assess(audio, text).to_dict()
            except Exception as e:
                print(f"skip {wav}: {e}", file=sys.stderr)
                continue
            flagged = {e["target_index"]: e["realized"] for e in r["errors"] if e["expected"] is not None}
            rows = []
            for p in (p for w in r["words"] for p in w["phones"]):
                rows.append(
                    {
                        "features": p.get("features"),
                        "our_phone": p["phone"],
                        "stress": p["stress"],
                        "f1f2": [p["formants"]["f1"], p["formants"]["f2"]] if p.get("formants") else None,
                        "wrong": 0,
                        "canon": p["phone"],
                        "perceived": p["phone"],
                        "kind": "",
                        "baseline_flag": int(p["index"] in flagged),
                        "baseline_diag": flagged.get(p["index"]),
                    }
                )
            out.write(json.dumps({"spk": f"NAT_{spk}", "utt": Path(wav).stem, "phones": rows}) + "\n")
            n += 1
    print(f"native: {n} utterances", file=sys.stderr)


def _prf(pred: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    tp = int((pred & gold).sum())
    p, r = tp / max(int(pred.sum()), 1), tp / max(int(gold.sum()), 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "flag_rate": float(pred.mean())}


def train(args: argparse.Namespace) -> None:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict

    from accent_engine.mdd import MDD_FEATURES, likely_realisation
    from accent_engine.scorer import CATEGORICAL, FEATURE_NAMES, VOWEL_FEATURES, LearnedScorer, vowel_feats
    from accent_engine.vowel_space import REFERENCE, SpeakerNorm

    with open(args.feats) as fh:
        utts = [json.loads(line) for line in fh]
    # per-speaker vowel normalisation (as in production calibration)
    toks: dict[str, list[tuple[str, float, float]]] = collections.defaultdict(list)
    for u in utts:
        for p in u["phones"]:
            if p["f1f2"] and p["our_phone"] in REFERENCE and p["stress"]:
                toks[u["spk"]].append((p["our_phone"], *p["f1f2"]))
    norms: dict[str, SpeakerNorm] = {}
    for spk, t in toks.items():
        with contextlib.suppress(ValueError):
            norms[spk] = SpeakerNorm.from_tokens(t)
    v0 = len(FEATURE_NAMES) - len(VOWEL_FEATURES)
    rows: list[tuple[str, dict[str, Any]]] = []
    for u in utts:
        for p in u["phones"]:
            if not p["features"] or len(p["features"]) != len(FEATURE_NAMES):
                continue
            f = [np.nan if v is None else v for v in p["features"]]
            nm = norms.get(u["spk"])
            f[v0:] = (
                vowel_feats(p["our_phone"], *p["f1f2"], nm)
                if nm and p["f1f2"]
                else [np.nan] * len(VOWEL_FEATURES)
            )
            p["features"] = f
            rows.append((u["spk"], p))
    X = np.array([p["features"] for _, p in rows], dtype=np.float64)
    so = LearnedScorer.load(args.scorer).score(X)
    Xa = np.column_stack([X, so])
    y = np.array([p["wrong"] for _, p in rows], dtype=bool)
    spk = np.array([s for s, _ in rows])
    base = np.array([p["baseline_flag"] for _, p in rows], dtype=bool)
    cat_mask = np.zeros(Xa.shape[1], dtype=bool)
    cat_mask[list(CATEGORICAL)] = True

    def make() -> Any:
        return HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            categorical_features=cat_mask,
            class_weight="balanced",
            random_state=0,
        )

    def pick_thresholds(Xt: np.ndarray, yt: np.ndarray, gt: np.ndarray) -> dict[str, float]:
        """Chosen on speaker-grouped OOF predictions of the *training* speakers only."""
        oof = cross_val_predict(make(), Xt, yt, groups=gt, cv=GroupKFold(5), method="predict_proba")[:, 1]
        grid = np.arange(0.2, 0.975, 0.025)
        stats = [(float(t), _prf(oof >= t, yt)) for t in grid]
        balanced = max(stats, key=lambda x: x[1]["f1"])[0]
        ok = [(t, m) for t, m in stats if m["precision"] >= GENTLE_PRECISION]
        gentle = max(ok, key=lambda x: x[1]["recall"])[0] if ok else float(grid[-1])
        return {"balanced": balanced, "gentle": max(gentle, balanced)}

    report: dict[str, Any] = {
        "n_phones": len(y),
        "speakers": sorted(set(spk.tolist())),
        "error_rate": float(y.mean()),
        "loso_arabic": {},
    }
    agg_pred, agg_base, agg_gold, agg_cat, agg_gentle, agg_proba, agg_focused = [], [], [], [], [], [], []
    diag_ok = diag_n = 0
    for held in ARABIC:
        tr, te = spk != held, spk == held
        if not te.any():
            continue
        thrs = pick_thresholds(Xa[tr], y[tr], spk[tr])
        m = make().fit(Xa[tr], y[tr])
        proba = m.predict_proba(Xa[te])[:, 1]
        pred = proba >= thrs["balanced"]
        report["loso_arabic"][held] = {
            "model": _prf(pred, y[te]),
            "gentle": _prf(proba >= thrs["gentle"], y[te]),
            "baseline": _prf(base[te], y[te]),
            "thresholds": thrs,
        }
        agg_gentle.append(proba >= thrs["gentle"])
        agg_proba.append(proba)
        # "focused": balanced threshold where the diagnosed error is a core Arabic-L1 pattern
        held_rows_ = [p for s_, p in rows if s_ == held]
        core = np.array(
            [
                category(
                    p["our_phone"],
                    p["baseline_diag"] or likely_realisation(np.asarray(f[:-1]), p["our_phone"]),
                    "s",
                )
                in CORE_CATEGORIES
                for p, f in zip(held_rows_, Xa[te], strict=True)
            ]
        )
        focused = np.where(core, proba >= thrs["balanced"], proba >= thrs["gentle"])
        report["loso_arabic"][held]["focused"] = _prf(focused, y[te])
        agg_focused.append(focused)
        agg_pred.append(pred)
        agg_base.append(base[te])
        agg_gold.append(y[te])
        held_rows = [p for s, p in rows if s == held]
        agg_cat += [category(p["canon"], p["perceived"], p["kind"]) if p["wrong"] else "" for p in held_rows]
        for p, f, hit in zip(held_rows, Xa[te], pred, strict=True):
            if hit and p["wrong"] and p["kind"] == "s":
                diag = p["baseline_diag"] or likely_realisation(np.asarray(f[:-1]), p["our_phone"])
                diag_n += 1
                diag_ok += diag == p["perceived"]
        print(held, json.dumps(report["loso_arabic"][held]), file=sys.stderr, flush=True)
    P, B, G = np.concatenate(agg_pred), np.concatenate(agg_base), np.concatenate(agg_gold)
    cats = np.array(agg_cat)
    Pr = np.concatenate(agg_proba)
    report["arabic_pr_curve"] = [
        {"threshold": float(t), **{k: round(v, 3) for k, v in _prf(Pr >= t, G).items()}}
        for t in np.arange(0.5, 0.975, 0.05)
    ]
    report["arabic_overall"] = {
        "model": _prf(P, G),
        "gentle": _prf(np.concatenate(agg_gentle), G),
        "focused": _prf(np.concatenate(agg_focused), G),
        "baseline": _prf(B, G),
        "diagnosis_accuracy": diag_ok / max(diag_n, 1),
    }
    report["arabic_recall_by_category"] = {
        c: {
            "n": int((cats == c).sum()),
            "model": float(P[cats == c].mean()),
            "focused": float(np.concatenate(agg_focused)[cats == c].mean()),
            "baseline": float(B[cats == c].mean()),
        }
        for c in sorted(set(agg_cat) - {""})
    }
    thrs = pick_thresholds(Xa, y, spk)
    final = make().fit(Xa, y)
    thr = thrs["balanced"]
    report["final_threshold"] = thr
    report["final_thresholds"] = thrs
    print(json.dumps(report, indent=2))
    if args.out:
        joblib.dump(
            {
                "model": final,
                "threshold": thr,
                "thresholds": thrs,
                "feature_names": list(MDD_FEATURES),
                "meta": {k: v for k, v in report.items() if k != "speakers"},
            },
            args.out,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--cache-dir", type=Path, default=None)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--native-index", type=Path, default=None, help="wav<TAB>text<TAB>speaker (appends)")
    t = sub.add_parser("train")
    t.add_argument("--feats", type=Path, required=True)
    t.add_argument("--scorer", required=True)
    t.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    {"extract": extract, "train": train}[args.cmd](args)


if __name__ == "__main__":
    main()
