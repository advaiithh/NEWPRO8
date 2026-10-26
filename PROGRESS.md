# PROGRESS

Declared device for all benchmarks: **4 cores / 4096 MB**, `qwen2.5:1.5b` (Python 3.12.10, CPU only). Last updated: see git log.

## C0 Setup: DONE
- Python 3.12.10 venv; all SPEC 3.3 packages installed (no torch / CUDA: `pip list` shows none).
- Ollama pulled: `qwen2.5:1.5b` / `0.5b` (+ Q8/Q5/Q4/Q3 variants, `3b`, `qwen3.5:2b`; `qwen3.5:4b` was already installed). Own CPU-only server on 127.0.0.1:11435, Vulkan/CUDA/HIP hidden, `size_vram 0` verified for every model used.
- `scripts/setup_models.py`, `download_extra.py` (Whisper small.en, Simple English Wikipedia), `download_tamil.py` (Tamil voices, Whisper small), `build_kb.py`, `render_cache.py`, `verify_assets.py` (158 files / 2.2 GB in `models/MANIFEST.json`).

## C1 Baseline loop: DONE
- Fixed 800 ms silence -> Whisper base.en -> Ollama defaults -> Piper full reply. `--wav` and `--no-audio-out` modes; timeline logged.
- Baseline (4c/4GB, 60 requests x 3 repeats = 180 turns): p50 4,180 ms, p95 13,976 ms, 180/180 completed, peak RSS 1,958 MB, 17.1 CPU-s/turn.

## C2 Guards + metrics + harness: DONE
- `gpucheck` (CPU-only gate), `netguard` (offline tripwire), `limiter` (affinity + Job Object memory cap over app and Ollama tree, live `tighten()` that eases the OS cap, see D-13), RAPL energy meter, resource sampler, 90-file synthetic test set, `bench/run_bench.py` with a free-memory pre-flight guard.
- 14 guard tests; cap enforcement proven on a child process (`MemoryError` above the Job Object cap).

## C3 Streaming core: DONE
- Silero VAD (0.16 ms/frame), openWakeWord ("Hey Jarvis"), Moonshine streaming ASR, sentence chunker + two-slot Piper, voice prompt, caps, normaliser.
- Found/fixed by measurement: Moonshine 25x slower when pinned to 2 CPUs (single-thread env var), Piper memory leak (ORT arena), `run_wav` race that made runs 10x slower.

## C4 Novelty: DONE
- Prefix warming, adaptive endpointing, router (skills -> semantic cache with pre-rendered audio -> offline Wikipedia knowledge base -> LLM).

## C5 Governor, degradation, barge-in: DONE
- Tiers T0-T4, hysteresis, cooldown + 3-sample rule (fixes a T0->T4 cascade found in a live test), low-system-memory rule, per-profile model lists and footprints.
- Barge-in tested with synthetic interruptions (opt-in with `--barge-in`; default half-duplex so the assistant never answers itself).
- Full degradation test (5 limits x 40 requests): fixed config fails to start at 2c/2G and 2c/1.5G; governor completes 97-100% at every limit.

## C6 Dashboard, eco, notes/timers, languages, pi_like: DONE
- Dashboard with all spec panels; notes/timers persisted; hands-free default mode (VAD + energy gate; answers only when spoken to).
- Languages: English default; **Tamil and Hindi** via sticky language mode ("speak Tamil" / L key / `--language`), Whisper-small recognition, Tamil Piper voice, Tamil cache (D-22, D-23). Hindi remains tested-partial.
- pi_like: 20/20 completed under 2c/1.5G (governor chose Survival tier).

## C7 Evaluation: DONE
- Ladder A0-A7 (A1-A7 re-run with **3 repeats** after the Tamil work), quantization sweep, KV-cache/ASR/TTS comparison, cache eval, knowledge-base eval (105 questions), LLM comparison (`qwen3.5:4b` 100% vs 1.5B 93%), per-tier quality, Tamil eval.
- Scoring bug found and fixed (stale rubric in the manifest; `bench/rescore.py`, D-20).

## C8 Polish: DONE except human-only items
- README (profiles, languages, hands-free), WRITEUP, AUDIT, LIVE_TEST, DEVIATIONS (D-01 .. D-23).
- Tests and proofs: see the end of `docs/AUDIT.md` for the final counts.
- Open (need a person): Wi-Fi-off run, Task Manager GPU graph, backup demo video, human ratings, Tamil/Hindi checks with real speakers.
