# American Accent Trainer

Real-time General American pronunciation and accent trainer with phone-level, vowel-acoustic and prosody feedback, tuned for Egyptian Arabic L1 speakers.

```
frontend/  React + TS (Vite). AudioWorklet capture → 16 kHz int16 → WebSocket; D3 pitch & vowel charts
backend/
  accent_engine/   UI-independent scoring engine (pure Python + NumPy, benchmarkable on its own)
  app/             FastAPI: REST + /ws/stream, SQLAlchemy models, arq worker
  scripts/         bench_latency.py, eval_speechocean.py
docker-compose.yml (+ docker-compose.cpu.yml)   frontend · api · worker · postgres · redis
```

## Engine pipeline (`accent_engine`)

| Stage | Module | Notes |
|---|---|---|
| Text → GA phones | `g2p.py` | CMUdict (all variants, e.g. REcord/reCORD), weak forms (*to, for, and, the…*), context reductions (*gonna, wanna, gotta, hafta, kinda*), OOV fallback: `g2p_en` if installed, else letter-to-sound rules. Post-lexical GA rules: **flap T/D**, **glottal T** before syllabic n (*button*), **dark-L** marking. |
| Acoustic model | `acoustic.py` | `AcousticModel` protocol producing posteriors over one ARPAbet vocabulary. `Wav2Vec2PhonemeModel` (default `facebook/wav2vec2-xlsr-53-espeak-cv-ft`) pools its IPA tokens into ARPAbet classes. `SyntheticAcousticModel` is a test double. |
| Recognition + alignment | `ctc.py`, `alignment.py` | NumPy CTC: greedy decoding, Viterbi forced alignment, batched forward pass. Needleman–Wunsch alignment with articulatory substitution costs, so TH→S counts as a substitution and an epenthetic vowel counts as an insertion. |
| GOP | `gop.py` | **GOP-LPR** (alignment-free, after GOP-AF): log-likelihood ratio of the canonical sequence against the best single-phone substitution or deletion, computed in a ±2-phone window, so cost grows linearly with length. **GOP-frame**: blank-weighted posterior GOP. Uses a logistic calibration to 0–100. |
| Errors | `errors.py` | Arabic-L1 taxonomy: P_B, V_F, TH_SUB, DH_SUB, IH_IY, UH_UW, AE_EH, SCHWA_REDUCTION, RHOTIC, FLAP_T, DARK_L, EPENTHESIS, FINAL_DEVOICING (+ generic sub/del/ins). Errors come from the alignment *or* from GOP alone (aligned as correct but acoustically closer to a competitor). Each entry carries an articulatory tip. |
| Vowels | `formants.py`, `vowel_space.py` | Praat Burg formants over the middle 40 % of each aligned vowel. Lobanov/Nearey speaker normalisation from a 12-word hVd calibration session; Hillenbrand et al. (1995) GA reference. |
| Prosody | `prosody.py` | F0 (Praat AC for real time, pYIN offline), energy, DTW intonation similarity in semitones, syllable prominence → lexical stress check, rhythm (nPVI-V, %V, ΔC, VarcoV). |
| Streaming | `vad.py` | Energy VAD by default (Silero when torch is present) with an utterance segmenter that handles pre-roll and hangover. |
| Orchestration | `engine.py` | `PronunciationEngine.assess(audio, text, speaker_norm=None, reference_audio=None)` → JSON-serialisable `Assessment` (words, phones, errors, scores, prosody tracks, timings). |

Scores: accuracy (phone GOP, capped on hard errors), completeness (words ≥50 % produced), fluency (pauses, articulation rate) and prosody (stress, rhythm, intonation when a reference exists). Total = 0.5/0.15/0.15/0.2 weighted.

## Run

```bash
cd backend && python -m venv ../.venv && . ../.venv/bin/activate
pip install -e ".[dev,ml]"          # drop `ml` to run without torch (synthetic backend only)
uvicorn app.main:app --reload       # AT_ACOUSTIC_BACKEND=synthetic for UI work without a model
```

```bash
cd frontend && npm install && npm run dev    # http://localhost:5173, proxies /api and /ws to :8000
```

Full stack (GPU): `docker compose up --build`. On CPU only: `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build`. The UI is then at http://localhost:8080.

## Checks

```bash
cd backend && ruff check . && ruff format --check . && mypy accent_engine app && pytest -q
cd frontend && npm run lint && npm run typecheck && npm test && npm run build
python backend/scripts/bench_latency.py
python backend/scripts/eval_speechocean.py --root /data/speechocean762   # needs the dataset + `ml`
```

### Latency budget (post-model, CPU, measured by `bench_latency.py`)

| phones | audio | scoring | formants | prosody | total |
|---:|---:|---:|---:|---:|---:|
| 4 | 1.1 s | 2 ms | 12 ms | 4 ms | 18 ms |
| 19 | 2.0 s | 11 ms | 22 ms | 6 ms | 39 ms |
| 40 | 3.6 s | 24 ms | 37 ms | 10 ms | 72 ms |
| 71 | 6.2 s | 44 ms | 65 ms | 16 ms | 129 ms |

Add the wav2vec2-large forward pass: roughly 30–60 ms for 3 s of audio on a modern GPU, several hundred ms on CPU. VAD end-of-speech adds a 0.5 s hangover before processing starts. Not yet implemented: batching concurrent utterances across WebSocket sessions on the GPU.

## Status against the build plan

| Step | State |
|---|---|
| 1. Monorepo, Compose, CI | ✅ (compose files validated; images not built in the dev sandbox) |
| 2. Audio capture + WS streaming | ✅ verified in headless Chromium with a fake microphone |
| 3. G2P + alignment + GOP + tests | ✅ (69 backend tests; real-model accuracy not yet measured, see below) |
| 4. Formants + prosody | ✅ verified against synthesised signals with known F1/F2/F0 |
| 5. Frontend visualisations | ✅ scores, phone chips, error tips, pitch contour, vowel chart, ear training, progress |
| 6. LLM coach, conversation mode (Whisper) | ⏳ next. Engine output is already structured JSON with taxonomy tips, so the coach only has to explain it. |
| 7. Adaptive engine (BKT/FSRS), dashboard, speechocean762 eval | ⏳ `PhoneStat` table and eval script exist; BKT/FSRS and PDF/CSV export are still to do |

### Known limitations / honest caveats

* **The real acoustic model has not been run yet.** The dev sandbox could not reach Hugging Face, OpenSLR or the PyTorch wheel index. Everything downstream of posteriors is tested with synthetic posteriors and synthesised audio. The first job on a machine with access is `eval_speechocean.py`, which also fits the GOP calibration. Phone-level PCC ≥ 0.6 is the acceptance target.
* `Calibration` defaults (slope 1.2, intercept 0) and the fluency heuristics are uncalibrated placeholders.
* `REF_SD` (vowel target spread) is a fixed placeholder until spreads are estimated from native recordings. The dark-L check is a heuristic F2 ratio.
* CTC phone boundaries come from peaky posteriors split at midpoints. They are good enough for formant midpoints and durations; use MFA for research-grade boundaries.
* The intonation reference needs reference audio. The browser's TTS *Listen* button can't be captured, so server-side TTS is planned with the coach work.
