"""Latency of the scoring path *after* the acoustic model (synthetic posteriors).

    python scripts/bench_latency.py

Add the acoustic model's forward time on your hardware (wav2vec2-large, ~3 s audio:
≈ 30–60 ms on a modern GPU, several hundred ms on CPU) to get end-to-end latency.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accent_engine import EngineConfig, PronunciationEngine
from accent_engine.acoustic import SyntheticAcousticModel
from accent_engine.g2p import g2p
from accent_engine.synth import concat, silence, vowel

SENTENCES = [
    "think",
    "I would like a bottle of water",
    "The early bird gets the worm but the second mouse gets the cheese",
    "We ran the titration three times and the reaction mixture turned yellow before we could label the beakers",
]


def audio_for(n_phones: int):  # type: ignore[no-untyped-def]
    dur = max(0.6, n_phones * 0.08)
    parts = [silence(0.2)]
    t = 0.0
    while t < dur:
        parts += [vowel((650, 1300), 0.12, 120 + 30 * (t / dur)), silence(0.06)]
        t += 0.18
    return concat(*parts, silence(0.2))


def main() -> None:
    g2p("warmup")
    print(
        f"{'phones':>6} {'audio_s':>7} | {'scoring':>8} {'formants':>8} {'prosody':>8} {'total':>8}  (ms, median of 10)"
    )
    for s in SENTENCES:
        tp = [p.phone for p in g2p(s).phones]
        audio = audio_for(len(tp))
        eng = PronunciationEngine(SyntheticAcousticModel(tp), EngineConfig())
        runs = [eng.assess(audio, s).timing_ms for _ in range(10)]
        med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
        print(
            f"{len(tp):>6} {len(audio) / 16000:>7.2f} | {med['scoring']:>8.1f} {med['formants']:>8.1f} "
            f"{med['prosody']:>8.1f} {med['total']:>8.1f}"
        )


if __name__ == "__main__":
    main()
