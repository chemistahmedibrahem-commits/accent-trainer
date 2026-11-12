"""Compare acoustic models on *native* American speech (LibriSpeech-style index.tsv).

A good GA scoring model should (1) flag almost nothing on native speech and (2) actually
hear post-vocalic /r/ (car, sort, near), which espeak-labelled models tend to miss.

    python scripts/compare_models.py /data/libri/index.tsv model_id [model_id ...]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accent_engine import EngineConfig, PronunciationEngine
from accent_engine.acoustic import Wav2Vec2PhonemeModel
from accent_engine.phonemes import is_vowel


def main() -> None:
    index, models = sys.argv[1], sys.argv[2:]
    with open(index) as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    report = {}
    for mid in models:
        eng = PronunciationEngine(Wav2Vec2PhonemeModel(mid), EngineConfig(formants=False, prosody=False))
        n_ph = n_err = coda_r = coda_r_ok = 0
        codes: collections.Counter[str] = collections.Counter()
        t_model = audio_s = 0.0
        for path, text in rows:
            a, _ = sf.read(path, dtype="float32")
            audio_s += len(a) / 16000
            r = eng.assess(a, text).to_dict()
            t_model += r["timing_ms"]["acoustic"] / 1000
            ph = [p for w in r["words"] for p in w["phones"]]
            n_ph += len(ph)
            n_err += len(r["errors"])
            codes.update(e["code"] for e in r["errors"])
            for i, p in enumerate(ph):
                nxt = ph[i + 1] if i + 1 < len(ph) else None
                if (
                    p["phone"] == "R"
                    and i > 0
                    and is_vowel(ph[i - 1]["phone"])
                    and (nxt is None or not is_vowel(nxt["phone"]))
                ):
                    coda_r += 1
                    coda_r_ok += p["realized"] == "R"
        report[mid] = {
            "errors_per_100_phones": round(100 * n_err / n_ph, 2),
            "coda_r_recall": round(coda_r_ok / max(coda_r, 1), 3),
            "coda_r_n": coda_r,
            "rtf_cpu": round(t_model / audio_s, 3),
            "top_errors": codes.most_common(8),
        }
        print(mid, json.dumps(report[mid]), flush=True)


if __name__ == "__main__":
    main()
