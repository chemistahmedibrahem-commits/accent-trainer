"""Mispronunciation detection & diagnosis (MDD) on L2-ARCTIC annotated utterances.

    python scripts/eval_l2arctic.py --root /data/l2arctic --speakers ABA SKA YBAA ZHAA \\
        --model MODEL [--scorer scorer.joblib] [--limit 150]

L2-ARCTIC `annotation/*.TextGrid` phone tier entries are either a plain canonical phone
("AH0") or "canonical,perceived,type" with type s (substitution), d (deletion), a (addition).
We align the engine's target phones to the annotated canonical sequence and report phone-level
detection precision / recall / F1, diagnosis accuracy (for detected substitutions, did we name
the perceived phone?), and per-category recall for the Arabic-L1 taxonomy.

Check that the acoustic model was not fine-tuned on L2-ARCTIC before trusting these numbers
(the default slplab model was fine-tuned on Korean-English ETRI data, so it is clean here).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accent_engine import EngineConfig, PronunciationEngine
from accent_engine.alignment import align
from accent_engine.g2p import build_target, lexicon_for
from accent_engine.phonemes import PHONES

_CLEAN = re.compile(r"[^A-Z]")


def read_phone_tier(path: Path) -> list[str]:
    """Minimal long-format TextGrid reader → phone tier labels (non-empty)."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    tiers = re.split(r"item \[\d+\]:", txt)[1:]
    for t in tiers:
        name = re.search(r'name = "([^"]*)"', t)
        if name and name.group(1).lower() == "phones":
            return [m.strip() for m in re.findall(r'text = "([^"]*)"', t) if m.strip()]
    return []


def parse(labels: list[str]) -> list[tuple[str, str, str]]:
    """→ list of (canonical, perceived, type) for canonical phones; type '' = correct."""
    out = []
    for lab in labels:
        parts = [p.strip() for p in lab.split(",")]
        canon = _CLEAN.sub("", parts[0].upper())  # the corpus has a few typos like "ER)"
        if len(parts) >= 3:
            perceived = _CLEAN.sub("", parts[1].upper())
            kind = parts[2].lower()
            if kind == "a":  # insertion: no canonical phone
                continue
            out.append((canon, perceived, kind))
        elif canon not in ("SIL", "SP", "SPN", ""):
            out.append((canon, canon, ""))
    return [o for o in out if o[0] in PHONES]


def category(canon: str, perceived: str, kind: str) -> str:
    pair = (canon, perceived)
    if kind == "d":
        return "deletion"
    table = {
        ("P", "B"): "P_B",
        ("V", "F"): "V_F",
        ("TH", "S"): "TH",
        ("TH", "T"): "TH",
        ("DH", "D"): "DH",
        ("DH", "Z"): "DH",
        ("IH", "IY"): "IH_IY",
        ("IY", "IH"): "IH_IY",
        ("AE", "EH"): "AE_EH",
        ("EH", "AE"): "AE_EH",
        ("UH", "UW"): "UH_UW",
        ("UW", "UH"): "UH_UW",
    }
    if pair in table:
        return table[pair]
    if canon in ("R", "ER"):
        return "rhotic"
    if canon in ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"):
        return "other_vowel"
    return "other_consonant"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--speakers", nargs="+", default=["ABA", "SKA", "YBAA", "ZHAA"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--limit", type=int, default=150, help="annotated utterances per speaker")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from accent_engine.acoustic import Wav2Vec2PhonemeModel

    eng = PronunciationEngine(
        Wav2Vec2PhonemeModel(args.model), EngineConfig(formants=False, prosody=False, scorer_path=args.scorer)
    )
    tp = fp = fn = tn = diag_ok = diag_n = 0
    cat_tot: collections.Counter[str] = collections.Counter()
    cat_hit: collections.Counter[str] = collections.Counter()
    per_spk: dict[str, Any] = {}
    for spk in args.speakers:
        d = args.root / spk
        grids = sorted((d / "annotation").glob("*.TextGrid"))[: args.limit]
        s_tp = s_fp = s_fn = 0
        for g in grids:
            ann = parse(read_phone_tier(g))
            wav = d / "wav" / f"{g.stem}.wav"
            txt = d / "transcript" / f"{g.stem}.txt"
            if not ann or not wav.exists() or not txt.exists():
                continue
            text = txt.read_text().strip()
            audio, sr = sf.read(wav, dtype="float32")
            if sr != 16000:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            try:
                r = eng.assess(audio, text).to_dict()
            except Exception as e:
                print(f"skip {g.stem}: {e}", file=sys.stderr)
                continue
            target = build_target(lexicon_for(text), [w["variant"] for w in r["words"]])
            flagged: dict[int, str | None] = {}
            for e in r["errors"]:
                if e["target_index"] is not None and e["expected"] is not None:
                    flagged[e["target_index"]] = e["realized"]
            ours = [p.underlying for p in target.phones]
            for op in align([a[0] for a in ann], ours):
                if op.ref_index is None or op.hyp_index is None:
                    continue
                canon, perceived, kind = ann[op.ref_index]
                wrong = kind in ("s", "d")
                det = op.hyp_index in flagged
                if wrong:
                    c = category(canon, perceived, kind)
                    cat_tot[c] += 1
                    cat_hit[c] += det
                if wrong and det:
                    tp += 1
                    s_tp += 1
                    if kind == "s":
                        diag_n += 1
                        diag_ok += flagged[op.hyp_index] == perceived
                elif wrong:
                    fn += 1
                    s_fn += 1
                elif det:
                    fp += 1
                    s_fp += 1
                else:
                    tn += 1
        per_spk[spk] = {
            "precision": s_tp / max(s_tp + s_fp, 1),
            "recall": s_tp / max(s_tp + s_fn, 1),
            "n_utts": len(grids),
        }
        print(spk, json.dumps(per_spk[spk]), file=sys.stderr, flush=True)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    report = {
        "model": args.model,
        "scorer": args.scorer,
        "detection": {
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "false_rejection_rate": fp / max(fp + tn, 1),
            "annotated_error_rate": (tp + fn) / max(tp + fn + fp + tn, 1),
        },
        "diagnosis_accuracy": diag_ok / max(diag_n, 1),
        "recall_by_category": {
            c: {"n": cat_tot[c], "recall": cat_hit[c] / cat_tot[c]} for c in sorted(cat_tot)
        },
        "per_speaker": per_spk,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
