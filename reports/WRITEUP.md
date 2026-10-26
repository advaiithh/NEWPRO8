# PocketVoice: write-up

## 1. Problem and device limit

A voice assistant (ASR, LLM, TTS) that runs fully offline on CPU only inside a simulated edge device: DEVICE LIMIT: 4 cores (LLM cpus [4, 6], audio cpus [0, 2]), 4096 MB RAM (Job Object cap 4096 MB). The limit is enforced by the OS (CPU affinity plus a Windows Job Object memory cap over the app and the Ollama process tree).

## 2. Baseline (exact)

Fixed 800 ms silence, then faster-whisper `base.en` int8 (default beam 5) on the whole utterance, then Ollama `qwen2.5:1.5b-instruct` Q4_K_M with default options (default context, no prefix warm, full uncapped reply), then Piper `lessac-medium` on the full reply, then play. Sequential; no router, governor or barge-in; same limit, same machine.

## 3. System and what is new

```
mic -> Silero VAD / openWakeWord -> Moonshine streaming ASR -> router: skills -> semantic cache -> LLM (Ollama, CPU)
    -> sentence chunker -> Piper (two-slot) -> speaker;   governor: tiers T0..T4 from RSS, CPU, tok/s, latency
```
New: **D1** compute-avoidance router (skills, then a semantic cache that plays pre-rendered audio, then the LLM); **D2** adaptive endpointing (hangover follows how complete the partial transcript is); **D3** prefix warming at wake; **D4** tier governor with hysteresis and a live tighten-the-limit control; **D5** barge-in; **D6** sentence-level streaming into a two-slot TTS pipeline; **D7** speakable-text normaliser.

## 4. Results (60 spoken requests x 3 repeats, warm-up discarded)

| | p50 ms | p95 ms | peak RSS MB | CPU-s/turn | J/turn RAPL (net of idle) | J/turn proxy | quality |
|---|---|---|---|---|---|---|---|
| Baseline (A0) | 4,180 | 13,976 | 1,958 | 17.1 | 32.2 | 102.6 | 0.85 |
| Ours (A7) | 1,205 | 2,121 | 1,733 | 5.0 | 35.1 | 29.9 | 0.93 |

Change vs baseline: p50 -71%, p95 -85%, peak RSS -11%, CPU-seconds per turn -71%, measured energy +9%. Energy source: Windows Energy Meter RAPL package power (whole machine), measured. The proxy column is cpu-seconds x 6.0 W/core, an assumption and not a measurement.

## 5. Ablation (each row adds one change)

- **A0 baseline**: p50 4,180 ms, p95 13,976 ms
- **A1 +Moonshine streaming**: p50 5,033 ms, p95 11,859 ms (+20% p50 vs previous step)
- **A2 +sentence-level TTS**: p50 4,144 ms, p95 6,634 ms (-18% p50 vs previous step)
- **A3 +voice prompt/caps**: p50 2,784 ms, p95 3,765 ms (-33% p50 vs previous step)
- **A4 +prefix warming**: p50 1,608 ms, p95 2,492 ms (-42% p50 vs previous step)
- **A5 +adaptive endpointing**: p50 1,538 ms, p95 2,341 ms (-4% p50 vs previous step)
- **A6 +router (skills+cache)**: p50 1,154 ms, p95 1,956 ms (-25% p50 vs previous step)
- **A7 +governor**: p50 1,205 ms, p95 2,121 ms (+4% p50 vs previous step)

## 6. Quantization sweep and chosen configuration

1.5b-Q8_0: 25 tok/s, 1657 MB RSS, quality 1.00; 1.5b-Q5_K_M: 36 tok/s, 1156 MB RSS, quality 1.00; 1.5b-Q4_K_M: 39 tok/s, 1028 MB RSS, quality 1.00; 1.5b-Q3_K_M: 43 tok/s, 874 MB RSS, quality 1.00; 0.5b-Q4: 80 tok/s, 451 MB RSS, quality 0.90. Per tier: T0/T1 use 1.5B Q4_K_M, T2/T3 use 0.5B Q4, with the KV-cache, ASR and TTS comparisons in `report.md`.

## 7. Graceful degradation (same 40 requests, limit stepped down)

- 6c/6144MB: fixed Full config completes 100% (p50 1,465 ms); with governor 100% (p50 1,467 ms)
- 4c/4096MB: fixed Full config completes 100% (p50 2,218 ms); with governor 98% (p50 1,858 ms)
- 3c/3072MB: fixed Full config completes 100% (p50 1,445 ms); with governor 100% (p50 1,460 ms)
- 2c/2048MB: fixed Full config completes 0% (p50 n/a ms); with governor 100% (p50 1,309 ms)
- 2c/1536MB: fixed Full config completes 0% (p50 n/a ms); with governor 100% (p50 826 ms)

## 8. Router and cache: savings and honest errors

The router answered 45% of turns without the LLM in the full ladder run. Cache threshold tau=0.75, margin 0.03, chosen on the dev paraphrases (hit 60%, wrong 8% of hits). Held-out (n=10): hit 50%, wrong 20% of hits, false hits on out-of-domain 0%. Static embeddings confuse questions about a topic with questions to the assistant ("why do we sleep" matches the "do you sleep" intent); the failure list is in `report.md`.

## 9. Offline and CPU-only proof

See `docs/AUDIT.md` (commands and outputs).

## 10. Limitations

Test speech is synthetic (Piper voices, clean audio); no real-microphone recordings or human ratings were available to the build (see `docs/LIVE_TEST.md`). The held-out set was written by the same assistant that wrote the cache, so it is less independent than a teammate-written set. RAPL is whole-package power including background load. Hindi was tested end to end on a few synthetic phrases only; no other language is claimed. Barge-in was tested with synthetic interruptions, not through a real speaker-to-microphone path.