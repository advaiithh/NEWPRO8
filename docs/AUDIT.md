# Self-audit (strict)

Machine: AMD Ryzen 5 5600H (6 cores / 12 threads), 15.7 GB RAM, Windows 11. Declared limit for all benchmarks: **4 cores / 4096 MB**, `qwen2.5:1.5b`.
Numbers come from `bench/out/*.json`, `reports/proofs.json` or the pytest/CLI output quoted below.
PASS = verified by a command that was run; PARTIAL = works but misses part of the requirement or evidence is reduced; FAIL = not met.

## Requirements

| # | Requirement | Status | Command run | Evidence |
|---|---|---|---|---|
| R1 | Full ASR -> LLM -> TTS loop on-device | **PASS** | `pytest tests/test_e2e.py` | Baseline loop; streaming pipeline through skill / cache / llm routes with all timeline stamps populated and a reply WAV; wake word ("Hey Jarvis" + question) -> "That is 391."; plus 20+ real-microphone utterances in a live session. |
| R2 | Fully offline | **PASS** (software proof) | `pytest tests/test_offline.py`; `python scripts/proofs.py` | External connect/DNS/HTTP raise `OfflineViolation` and are counted; real run = 0 attempts; a deliberate probe to 8.8.8.8 was blocked. **Not done by me:** a physical Wi-Fi-off run; optional `offline_fw.ps1` (needs admin). |
| R3 | CPU-only (gate) | **PASS** | `pytest tests/test_gpu_free.py`; `proofs.py`; `bench.llm_compare` | All ORT sessions CPU only; no torch/tensorflow/onnxruntime-gpu; Ollama `size_vram: 0` for every model tested (0.5B, 1.5B, 3B, `qwen3.5:2b`, `qwen3.5:4b`); Ollama log `inference compute id=cpu`. **Not done:** Task Manager GPU-graph screenshot. |
| R4 | Simulated edge device enforced by OS | **PASS** | `pytest tests/test_limits.py`; `proofs.py` | Affinity applied; Job Object memory cap: child allocating 100 MB under a 400 MB cap succeeds, 900 MB raises `MemoryError`; live `tighten()` works (the OS cap is eased down, see D-13). |
| R5 | Wake word / VAD | **PASS** | `pytest tests/test_wake_vad.py tests/test_always_listening.py` | VAD 0.16 ms/frame; speech vs silence vs noise separated; wake word fires on its phrase only; hands-free mode answers once when spoken to and stays silent on silence and noise. |
| R6 | Beat baseline latency | **PASS** | `bench.run_ablation`; A7 re-run | p50 **4,180 -> 1,205 ms (-71%)**, p95 **13,976 -> 2,121 ms (-85%)** (baseline 180 turns, ours 120). Synthetic speech, one machine. |
| R7 | Lower footprint (CPU, RAM, energy) | **PARTIAL** | same | Peak RSS 1,958 -> **1,733 MB (-11.5%)**; CPU-s/turn 17.1 -> **5.0 (-71%)**; RAPL gross J/turn 121.9 -> 79.5 (-35%) but **net-of-idle 32.2 -> 35.1 J (no improvement; RAPL is whole-machine and noisy)**; CPU-seconds proxy 102.6 -> 29.9 J (assumption). **Idle footprint is not lower** (idle CPU 1.95% -> 2.73%, idle RSS 1,502 -> 1,589 MB). |
| R8 | Report CPU, RAM, energy | **PASS** | `python -m bench.make_report` | `reports/report.md` with CPU-s/turn, RSS and commit, J/turn (RAPL measured and proxy, labelled), idle CPU/RSS. |
| R9 | Short write-up | **PASS** | `bench/make_report.py` | `reports/WRITEUP.md`, ~700 words (under 2 pages), generated from the JSON. |
| S1 | Multilingual | **PARTIAL** | `pytest tests/test_multilingual.py` | Language detection, Hindi voice and prompt work; Whisper-base Hindi transcripts are poor so the Hindi cache rarely matches (D-12). Not claimed as reliable. |
| S2 | Smaller device (pi_like) | **PARTIAL** | `run_bench --system FULL --profile pi_like` | 20/20 completed within the cap (peak commit 1,223 MB), p50 578 ms; governor chose T4 Survival (0.5B does not fit under 1.5 GB). 20 skill-type requests only; no real device. |
| S3 | Cached TTS | **PASS** | `bench.run_bench` A7; `bench.cache_eval` | Cache route p50 **534 ms** (pre-rendered audio, no synthesis); 22 of 120 turns. |
| S4 | Quality as limit tightens | **PASS** | `bench.run_degrade --n 40` (5 limits x 40) + `bench.quality_eval` | Governor: completion 100/97/100/100/100% at 6c6G/4c4G/3c3G/2c2G/2c1.5G with quality 0.95/0.90/0.90/0.80/0.35 and p50 1,467/1,858/1,460/1,309/826 ms. Fixed Full config: 100% down to 3c/3G (quality 0.97/0.95/0.95), **fails to start (HTTP 500) at 2c/2G and 2c/1.5G**. Per tier (fixed, 4c/4G): T0 0.95, T1 0.95, T2 0.72, T3 0.72, T4 0.35. |

## Rubric

| Item | Status | Evidence / caveat |
|---|---|---|
| Latency vs baseline (25%) | **PASS** | R6. Per-route p50 (A7): skill 741 ms, cache 534 ms, LLM 1,509 ms (p95 2,228). Spec targets skill <=400, cache <=450, LLM <=1,500/2,500: **skill and cache targets missed** (the span starts at the last voiced frame, so ~340 ms mean hangover is inside every number); LLM met. |
| Footprint vs baseline (25%) | **PARTIAL** | R7: CPU and peak RAM lower; energy not demonstrated; idle not lower. |
| Quantization + offload (15%) | **PASS** (caveat) | `bench.run_quant_sweep`: 1.5B Q8/Q5/Q4/Q3 and 0.5B: 24.7/35.6/39.0/42.6/80.1 tok/s, RSS 1,657/1,156/1,028/874/451 MB; KV f16 vs q8 and flash attention (RSS within 1.5%); ASR int8 vs fp32, Moonshine tiny vs small; TTS medium vs low (RTF 0.07/0.05); load 1.67 s / unload 0.01 s. Caveat: the keyword rubric scores every 1.5B variant 1.00, so it cannot separate them; TTS subjective rating needs humans. |
| Graceful degradation (15%) | **PASS** | S4 plus 14 governor tests and the live-tighten integration test. |
| Research contribution + ablation (20%) | **PASS** (caveats) | Ladder A0-A7 below. A1-A7 used **2 repeats** (baseline 3, D-09); A1-A6 predate the Piper-leak fix (D-15). |
| Bonus: multilingual / smaller device / cached TTS | **PARTIAL** | S1, S2 partial; S3 pass. |

## Ablation ladder (60 requests; quality re-scored with the corrected rubric, see "Measurement bug")

| Step | p50 ms | p95 ms | RSS MB | CPU-s | quality |
|---|---|---|---|---|---|
| A0 baseline | 4,180 | 13,976 | 1,958 | 17.1 | 0.85 |
| A1 +Moonshine streaming | 5,033 | 11,859 | 2,759 | 11.0 | 0.71 |
| A2 +sentence-level TTS | 4,144 | 6,634 | 2,295 | 11.3 | 0.78 |
| A3 +voice prompt/caps | 2,784 | 3,765 | 1,978 | 6.6 | 0.72 |
| A4 +prefix warming | 1,608 | 2,492 | 1,937 | 6.8 | 0.74 |
| A5 +adaptive endpointing | 1,538 | 2,341 | 1,954 | 6.7 | 0.76 |
| A6 +router (skills+cache) | 1,154 | 1,956 | 2,028 | 4.9 | 0.93 |
| A7 +governor (post-fix re-run) | 1,205 | 2,121 | 1,733 | 5.0 | 0.93 |

A1 is *slower* at p50 than the baseline; the gains start at A2. The large A3->A4 drop (-42%) was not decomposed, so I do not claim its mechanism. Quality rises at A6 because skills answer time/date/maths/units deterministically. The keyword rubric is lenient (a loose fallback can match), so treat quality as indicative.

## Measurement bug found and fixed (honest note)

The benchmark manifest held an older, over-strict scoring pattern for 15 skill items, so quality for LLM-answered skill questions was understated (baseline **0.75 -> 0.85** once corrected; ours unchanged at 0.93). `bench/rescore.py` re-applied the correct rubric to every stored result; the table above uses the corrected values. The quality gap to the baseline is therefore **0.85 vs 0.93**, not 0.75 vs 0.93.

## Cache, router and knowledge honesty

- Cache (tau 0.75, margin 0.03 chosen on dev): dev hit 60%, wrong 8% of hits; **held-out (n=10, written by me): hit 50%, wrong 20% of hits**; spec default 0.80/0.05 hits only 20% held-out. The live app uses a stricter tau 0.85. Known failure: topic questions match assistant-directed intents ("why do we sleep" -> "do you sleep").
- Router answered 45% of ladder turns without the LLM.
- **Knowledge base** (160k Simple English Wikipedia articles, offline): `bench.knowledge_eval` on 105 questions: LLM alone 98%, with KB 97% (no accuracy gain on well-known facts); KB answers 53% of questions in 0.42 s vs 1.11 s and works in the Survival tier (D-16).
- **LLM comparison** (`bench.llm_compare`, 40 reasoning/fact questions, CPU only): 1.5B 93%, 3B 93-95%, `qwen3.5:2b` 97% (3.2 GB), **`qwen3.5:4b` 100%** (5.0 GB RSS, 12 tok/s). The `quality` profile (6 cores / 8 GB) uses 4B; it does not fit the 4 GB edge limit (D-19). On this laptop with other apps open, only ~4 GB of memory is free, so the governor drops from 4B to lighter models; a measured `quality`-profile run completed 40/40 but sat mostly in lower tiers for that reason (keyword quality 0.88, also hurt by speech-recognition slips from the small benchmark recogniser).

## Live-session findings (real microphone) and fixes

Hands-free mode, noise gate, half-duplex muting, junk-transcript filter, cut-off-sentence trimming (D-18); a governor cascade (one slow turn dropped T0->T4 in 2 s) fixed with cooldown + 3-sample rule; a console UnicodeEncodeError; a crash when Whisper failed under memory pressure (turn failures are now contained and the governor reacts to low system memory); accurate Whisper final pass and mic-gain normalisation (D-17).

## Final results

```
pytest tests -q                         -> 79 passed (5 min 40 s)
python scripts/proofs.py                -> PROOFS PASS
bench.run_degrade --n 40 (5 limits)     -> table above
```

## Definition of done

- [x] Voice in -> voice out, offline, CPU-only, under the limit, with wake word/VAD/hands-free (synthetic + real mic).
- [x] GPU proof and network proof on the dashboard (via `/events`).
- [~] Baseline vs ours: latency, CPU and RAM lower; energy and idle footprint not shown lower (R7).
- [x] Ablation ladder, quantization sweep, full 5-limit degradation test, per-tier quality generated (ladder at 2 repeats).
- [x] Router/cache savings and error rate reported.
- [x] Live limit tightening without a crash (test + fix D-13).
- [ ] **Backup demo video: not made** (needs a person). Short write-up exists.
- [x] `docs/DEVIATIONS.md` lists every substitution (D-01 .. D-19).

## Not verified by me (needs a person)

Wi-Fi-off run; Task Manager GPU graph; human TTS/answer ratings; teammate-written held-out questions; `offline_fw.ps1`; backup video; whether the larger `quality` model feels right on the user's voice for every question.

## Known issues

- Skill and cache routes miss the spec's 400/450 ms targets (741/534 ms p50).
- Hindi transcription weak (D-12); cache covers about half of unseen paraphrases.
- The `quality` profile needs ~8 GB free; with many apps open it falls back automatically (message printed at start-up).
- Ladder uses 2 repeats (A1-A7); a strict 3-repeat re-run would take about 2 more hours.
- Windows quirk: Ollama joined to the limiter's Job Object exits with the app that created it (by design in `run.ps1`).
