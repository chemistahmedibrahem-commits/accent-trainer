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


def extract(args: argparse.Namespace) -> None:
    from accent_engine.acoustic import Wav2Vec2PhonemeModel

    scores = json.loads((args.root / "resource" / "scores.json").read_text())
    eng = PronunciationEngine(Wav2Vec2PhonemeModel(args.model), EngineConfig(formants=False, prosody=True))
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
                try:
                    r = eng.assess(audio, text).to_dict()
                except Exception as e:
                    print(f"skip {uid}: {e}", file=sys.stderr)
                    continue
                human = scores[uid]
                target = build_target(lexicon_for(text), [w["variant"] for w in r["words"]])
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


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--model", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    e.add_argument("--limit", type=int, default=0, help="max utterances per split (0 = all)")
    e.add_argument("--out", type=Path, default=Path("so762_feats.jsonl"))
    f = sub.add_parser("fit")
    f.add_argument("--feats", type=Path, required=True)
    f.add_argument("--calibration-out", type=Path, default=None)
    args = ap.parse_args()
    extract(args) if args.cmd == "extract" else fit(args)


if __name__ == "__main__":
    main()
