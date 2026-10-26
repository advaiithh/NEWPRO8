# PLAN: checkpoints and verification commands

All commands run from the repo root with the venv: `.\.venv\Scripts\python.exe` (Python 3.12.10).

| CP | Scope | Verification commands |
|---|---|---|
| C0 | Py3.12 venv, probe, packages, Ollama pulls, model downloads, verify_assets, own CPU-only Ollama on :11435 | `python scripts\probe.py`; `python scripts\setup_models.py`; `python scripts\verify_assets.py`; `powershell scripts\launch_ollama.ps1`; `ollama ps` with `OLLAMA_HOST=127.0.0.1:11435` -> VRAM 0 |
| C1 | Sequential baseline loop (faster-whisper base.en -> Ollama -> Piper), timeline log, `--wav`, `--no-audio-out` | `python -m pv.baseline --wav bench\testset\x.wav --no-audio-out`; `pytest tests\test_e2e.py` |
| C2 | gpucheck, netguard, limiter, resource/energy metrics, testset, bench harness | `pytest tests\test_gpu_free.py tests\test_offline.py tests\test_limits.py`; `python bench\make_testset.py`; `python bench\run_bench.py --system baseline` |
| C3 | VAD, wake word, Moonshine streaming, chunker, 2-slot TTS, prompt, caps | `pytest tests\test_vad_endpoint.py tests\test_chunker.py tests\test_normalizer.py tests\test_wake_vad.py`; `python bench\run_ablation.py --steps A1,A2,A3` |
| C4 | Prefix warm, adaptive endpointing, router (skills, cache with audio) | `pytest tests\test_skills.py tests\test_router_cache.py`; ablation A4..A6 |
| C5 | Governor, degradation, barge-in | `pytest tests\test_governor.py`; `python bench\run_degrade.py` |
| C6 | Dashboard, eco, notes/timers, Hindi, pi_like | `pytest tests\test_multilingual.py`; open `http://127.0.0.1:8765`; bench under pi_like |
| C7 | Full ablation, quant sweep, quality eval, report | `scripts\bench.ps1`; `reports\report.md` exists |
| C8 | README, WRITEUP, demo, AUDIT | `pytest`; offline + GPU + limits proofs; `docs\AUDIT.md` |

Fallbacks follow SPEC section 16. Every deviation goes to docs/DEVIATIONS.md.
