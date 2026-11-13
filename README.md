# American Accent Trainer

A real-time General American (GA) pronunciation and accent trainer. It gives feedback on phones, vowel acoustics and prosody, explains errors like a speech coach, and adapts training to each learner. It is tuned for learners whose first language is Egyptian Arabic.

```
frontend/   React + TS (Vite). AudioWorklet capture → anti-aliased 16 kHz → WebSocket; D3 charts
backend/
  accent_engine/  UI-independent scoring engine (benchmarkable on its own)
  trainer/        adaptive training: BKT skill tracing, FSRS scheduling, drill generation
  app/            FastAPI (REST + /ws/stream), SQLAlchemy, Claude coach, Whisper, Piper TTS, reports
  scripts/        evaluation & benchmarking (speechocean762, L2-ARCTIC, native speech, latency)
docker-compose.yml (+ docker-compose.cpu.yml)   frontend · api · worker · postgres · redis
```

## What it does

| Mode | What the learner gets |
|---|---|
| **Practice** (listen & repeat) | American reference audio (normal/slow); live streaming with VAD; overall, accuracy, completeness, fluency and prosody scores; per-phone IPA chips; errors with articulatory tips; pitch contour vs the native reference; vowel chart; coach explanation and three drills |
| **Today** | Adaptive plan: the weakest *due* skills, starting with ear training (minimal pairs) when mastery is low, then word drills, tongue twisters, and stress-shift pairs |
| **Conversation** | Role-play (job interview, lab meeting, small talk, doctor visit). Whisper transcribes each turn, the pronunciation engine scores it, and you get a short tip |
| **Ear training** | Minimal-pair perception tests per contrast |
| **Progress** | Skill mastery, a weekly heatmap, a daily score trend, top-5 weaknesses, and CSV/PDF export |

## Scoring engine (`backend/accent_engine`)

| Stage | Module | Notes |
|---|---|---|
| Text → GA phones | `g2p.py` | CMUdict (all variants, e.g. REcord/reCORD), weak forms, *gonna/wanna/gotta*. OOV words go through a morphological fallback (possessive/plural/-ed/-ing on a known stem), then letter-to-sound rules. GA post-lexical rules: flap T/D, glottal T before syllabic n, dark-L marking. Accepted GA variation: pre-/r/ mergers, reduced vowels [ə]/[ɪ], the cot–caught merger, [d]-like flaps. |
| Acoustic model | `acoustic.py` | One ARPAbet posterior interface over any HF CTC phone model. It handles espeak-IPA, TIMIT/ARPAbet, and L2 vocabularies with `_err` tokens. Two-phone tokens (`ɑːɹ`, `el`, `ts`, …) are split exactly by frame doubling. Classes a model lacks borrow an alias's posterior. |
| Recognition & alignment | `ctc.py`, `alignment.py` | NumPy CTC (greedy, Viterbi forced alignment, batched forward). Needleman–Wunsch with articulatory costs. |
| GOP | `gop.py` | GOP-LPR (alignment-free, GOP-AF style) in a ±2-phone window (linear time), plus frame GOP |
| **Learned scorer** | `scorer.py` | Gradient-boosted regressor trained on speechocean762 **train**. Features: the LPR vector against every phone and against deletion, posterior statistics, the `_err`-token mass, duration, identity/context, and speaker-normalised vowel quality. It replaces calibrated GOP and gates error reports. |
| Errors | `errors.py` | Arabic-L1 taxonomy: P_B, V_F, TH_SUB, DH_SUB, IH_IY, UH_UW, AE_EH, SCHWA_REDUCTION, RHOTIC, FLAP_T, DARK_L, EPENTHESIS, FINAL_DEVOICING (+ generic sub/del/ins) |
| Vowels | `formants.py`, `vowel_space.py` | Praat Burg formants; Lobanov/Nearey normalisation from an hVd calibration; Hillenbrand (1995) GA targets |
| Prosody | `prosody.py` | F0 (Praat; pYIN offline), DTW intonation vs. a Piper GA reference, lexical stress from syllable prominence, rhythm (nPVI, %V, ΔC) |

### Choosing the acoustic model: measured, not assumed

The spec suggested `facebook/wav2vec2-xlsr-53-espeak-cv-ft`. Measured on native American read speech (LibriSpeech speaker 1272, `scripts/compare_models.py`), it **detected only 3.6% of post-vocalic /r/** ("car", "sort", "near"). Its espeak-derived labels are effectively non-rhotic, so it would flag every American speaker on the most important Arabic-L1 contrast.

| Model | Coda-/r/ recall (native) | CPU RTF |
|---|---|---|
| espeak-cv-ft (spec default) | 3.6% | 0.40 |
| TIMIT base, ARPAbet-39 | 76% | **0.05** |
| vitouphy TIMIT (IPA) | 81% | 0.37 |
| **slplab L2-English (default)** | **95%** | 0.29 |
| speech31 WavLM | 92% | 0.15 |

### Accuracy on speechocean762 (held-out **test** split, human phone ratings 0–2)

RESULTS_TABLE

## The coach never invents scores

`app/coach.py` sends Claude (`claude-opus-5-5`, structured outputs, low effort, server-side refusal fallback) a compact digest of the engine output. The reply must match a JSON schema (summary, ≤3 focus items, 3 drills). A validator then rejects any reply that contains a number not present in the digest, or an error code or word the engine didn't report. Rejected or failed calls fall back to a deterministic rule-based coach built from the taxonomy tips and generated drills. Without `ANTHROPIC_API_KEY` the rule-based coach is used.

## Adaptive training (`backend/trainer`)

* **Skills = the error taxonomy (+ word stress).** Every target phone is an *opportunity* for the skills it exercises. It counts as correct unless the engine reported that error at that site, which is the per-opportunity evidence BKT needs.
* **BKT with forgetting**, because a "mastered" skill must be able to drop again when errors return (fossilisation).
* **FSRS-5** scheduling per skill, graded from the session's scores.
* **Drill generation** from CMUdict + word frequencies: target words by skill and position, **mined minimal pairs** (ship/sheep, bag/back, think/sink…), stress-shift pairs, tongue twisters, and shadowing passages.

## Run

```bash
cd backend && python -m venv ../.venv && . ../.venv/bin/activate
pip install -e ".[dev,ml,speech]"
AT_ACOUSTIC_MODEL_ID=slplab/wav2vec2-large-robust-L2-english-phoneme-recognition \
AT_SCORER_PATH=models/scorer_slplab.joblib uvicorn app.main:app
```

```bash
cd frontend && npm install && npm run dev        # http://localhost:5173 (proxies /api, /ws)
```

Optional: `ANTHROPIC_API_KEY` turns on the Claude coach and conversation partner. `AT_ACOUSTIC_BACKEND=synthetic` gives UI development without a model.
Full stack: `docker compose up --build` (GPU) or `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build`.

## Evaluate

```bash
# speechocean762 (openslr.org/101): extract features (posteriors cached), then fit
python scripts/eval_speechocean.py extract --root DATA/speechocean762 --model MODEL --cache-dir cache --out feats.jsonl
python scripts/eval_speechocean.py train-scorer --feats feats.jsonl --model-id MODEL --out models/scorer.joblib
# L2-ARCTIC Arabic-L1 speakers: mispronunciation detection & diagnosis
python scripts/eval_l2arctic.py --root DATA/l2arctic --model MODEL --scorer models/scorer.joblib
# native speech false-positive rate + coda-/r/ recall
python scripts/compare_models.py DATA/native/index.tsv MODEL [MODEL...]
python scripts/bench_latency.py
```

## Checks

```bash
cd backend && ruff check . && ruff format --check . && mypy accent_engine app trainer && pytest -q
cd frontend && npm run lint && npm run typecheck && npm test && npm run build
```

STATUS_SECTION
