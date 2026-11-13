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
| **Mispronunciation detector** | `mdd.py` | Decides which errors are *reported* (see below); trained on L2-ARCTIC with native negatives |
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

### Accuracy on speechocean762 (held-out **test** split: 2,500 utterances, 46,781 phones, unseen speakers)

The scorer was trained on the train split only. Its error threshold was chosen from speaker-grouped out-of-fold predictions on train. The acoustic model was fine-tuned on Korean-English ETRI data, so it has never seen this corpus.

| Pearson r with human ratings | Phone | Utt. accuracy | Utt. fluency | Utt. prosodic | Utt. total |
|---|---|---|---|---|---|
| Raw GOP-LPR | 0.422 | – | – | – | – |
| Logistic-calibrated GOP | 0.474 | 0.63 | 0.61 | −0.05 | 0.60 |
| **Learned scorer (shipped)** | **0.639** | **0.74** | **0.80** | **0.78** | **0.76** |
| *GOPT (Gong et al. 2022), for reference* | *0.612* | *0.714* | *0.753* | *0.760* | *0.742* |

The spec's target of phone-level Pearson ≥ 0.6 is met. The hand-built prosody score did not track human prosody ratings (r ≈ 0), so the shipped utterance scores come from the learned regressors.

### What gets reported: the mispronunciation detector (`accent_engine/mdd.py`)

Scoring and *telling the learner what is wrong* are separate jobs. A threshold on the speechocean762 scorer (graded, mostly Mandarin-L1 labels) caught only 14% of the annotated errors of Arabic speakers. So which errors get reported is decided by a classifier trained on **L2-ARCTIC's categorical annotations** (24 speakers, 6 L1s). It also uses **910 utterances from 7 native US speakers as negatives**, so that native-like variation isn't "corrected". It has three strictness levels:

* **focused (default):** sensitive for the documented Egyptian-Arabic transfer errors (P→B, TH, DH, V→F, rhotics), conservative for everything else
* **thorough:** the F1-optimal threshold for all errors
* **gentle:** only the surest corrections

**Arabic L1, leave-one-speaker-out.** Each of ABA, SKA, YBAA and ZHAA is scored by a model that never heard them. Thresholds were picked on the training speakers only. 19,558 phones; 1,861 (9.5%) were annotated as wrong.

| Detector | Precision | Recall | Flag rate |
|---|---|---|---|
| Alignment/GOP baseline | 0.30 | 0.39 | 12.4% |
| **focused (default)** | **0.40** | **0.39** | **9.4%** |
| thorough | 0.31 | 0.61 | 18.5% |
| gentle | 0.44 | 0.23 | 5.0% |

Recall in focused mode for the core Arabic-L1 errors: **DH 96%** (baseline 32%), **TH 95%**, **P→B 78%**, **rhotics 69%**, V→F 29%. Diagnosis accuracy (correct "it sounded like …" phone) is 53%.

**False alarms on native American speech** (held-out speakers, `scripts/compare_models.py`). Counts are errors per 100 phones and include minor "flap T not used" style notes; careful read speech often doesn't flap.

| | focused | thorough | gentle |
|---|---|---|---|
| CMU ARCTIC `bdl` + `clb` (US male + female, 120 utts) | 3.5 | 6.7 | 2.4 |
| LibriSpeech speaker 1272 (73 utts) | 6.3 | 13.3 | 4.7 |

Coda-/r/ recall on native speech is 98.4% and 95.1%.

Two measured artefacts are handled explicitly. (1) The phone model labels native unstressed [ɚ] as AH 34% of the time, so unstressed ER accepts AH; stressed ER ("bird", "heard") is still checked. (2) The dark-L F2 heuristic fired on 13% of native L's, so it is off by default (`EngineConfig.dark_l_check`).

### Latency (CPU, 4 cores, no GPU)

| Path | Time |
|---|---|
| Acoustic model (wav2vec2-large) | ~0.43 s per 3.5 s utterance (RTF ≈ 0.11) |
| Everything after the model (alignment, GOP, scorer, detector, formants, prosody, TTS reference) | ~0.25 s |
| WebSocket: end of speech → result (real Arabic-L1 utterance) | **0.80 s** |

The spec's < 300 ms needs a GPU for the forward pass: wav2vec2-large is typically 30–60 ms on a modern GPU, and fp16 is enabled automatically on CUDA. The rest of the pipeline already fits inside that budget.

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
uvicorn app.main:app        # defaults: slplab model + shipped scorer + detector + Piper reference voice
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
# mispronunciation detector: L2-ARCTIC (+ native negatives), leave-one-Arabic-speaker-out
python scripts/train_mdd.py extract --root DATA/l2arctic --model MODEL --cache-dir cache --out mdd.jsonl
python scripts/train_mdd.py extract --root DATA/l2arctic --model MODEL --cache-dir cache --out mdd.jsonl --native-index native.tsv
python scripts/train_mdd.py train --feats mdd.jsonl --scorer models/scorer.joblib --out models/mdd.joblib
# native speech false-positive rate + coda-/r/ recall
python scripts/compare_models.py DATA/native/index.tsv MODEL [MODEL...]
python scripts/bench_latency.py
```

## Checks

```bash
cd backend && ruff check . && ruff format --check . && mypy accent_engine app trainer && pytest -q
cd frontend && npm run lint && npm run typecheck && npm test && npm run build
```

## Status

| Build-plan step | State |
|---|---|
| 1. Monorepo, Compose, CI | ✅ Compose files validated. Images not built here (no Docker daemon in the dev sandbox). |
| 2. Audio capture + streaming | ✅ Verified in headless Chromium (fake mic) and with real speech over the WebSocket |
| 3. G2P, alignment, GOP, tests | ✅ Real model, 93 backend tests |
| 4. Formants + prosody | ✅ |
| 5. Visualisations | ✅ Pitch vs native reference, vowel chart, phone chips, heatmap, trends |
| 6. Coach + conversation | ✅ Rule-based coach verified end to end. Claude path tested with a fake client (no API key here). Whisper + Piper verified. |
| 7. Adaptive engine, dashboard, evaluation | ✅ BKT + FSRS + drill generation, CSV/PDF, speechocean762 (target met), L2-ARCTIC, native checks |

### Known limitations

* **Precision for Arabic speakers is ~0.4** at the default setting. L2-ARCTIC annotators mark only clear categorical errors, so many "false" flags are accented-but-acceptable phones. Some are real misses by the annotators, and some are model errors. Learners can switch to *gentle*.
* **Vowel-quality errors are the weakest category** (AE/EH, IH/IY, UH/UW recall 0–30%). Categorical phone models hear the nearest phoneme. The speaker-normalised formant features help the *score* more than the detector. A per-user vowel calibration (the hVd session) improves both.
* **The intonation score** compares against a single Piper TTS rendition. The utterance-level prosody score shipped is the learned one (r = 0.78 with human prosody ratings); the raw contour similarity is shown for visualisation only.
* Both trained models are specific to the slplab acoustic model's features. Changing `AT_ACOUSTIC_MODEL_ID` requires re-running `eval_speechocean.py` and `train_mdd.py`. The engine raises an error if the scorer was trained for a different acoustic model.
* Model and dataset licences: speechocean762 is Apache-2.0 and L2-ARCTIC is CC BY-NC 4.0, so the shipped detector is **for non-commercial use** unless it is retrained on other data. CMU ARCTIC is free to use. Check the slplab model card before commercial use.
