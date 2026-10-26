# PocketVoice: prompts to paste into Claude Code

Two things below:
- **PART A: `CLAUDE.md`** (save as a file in the project folder; Claude Code reads it automatically every session)
- **PART B: MASTER PROMPT** (paste into Claude Code once; then use the short follow-up prompts)

---

## PART A: save this as `CLAUDE.md` in your project folder

```
# Project rules (always follow)

Project: PocketVoice, CPU-only, fully offline voice assistant for hackathon PS HNX26EPS08.
The full specification is in SPEC.md. It is the source of truth. Read it fully before coding.

Non-negotiable:
1. Runtime must be 100% offline and 100% CPU. No GPU libraries, no torch, no cloud calls.
2. Setup phase may use the internet to install packages and download models. Runtime never may.
3. Windows + PowerShell. Use pathlib. Python venv in .venv.
4. Never invent package names, model names, APIs or numbers. Verify by running commands and reading installed code. If something differs from SPEC.md, use the fallback and record it in docs/DEVIATIONS.md.
5. Never say something works unless you ran it and saw it work. Show the command and its real output.
6. After each checkpoint in SPEC.md section 15: run its tests, fix failures, update PROGRESS.md, then git commit.
7. Never write numbers into reports that did not come from bench/ output.
8. Keep code simple, typed, and short. No dead code. No placeholders or TODOs left in finished modules.
9. If blocked for more than 15 minutes on one item, apply the fallback from SPEC.md section 16, log it, and continue.
10. Ask the user only when a decision truly cannot be made from the spec (for example: they must speak into the microphone, or approve an admin action).
```

---

## PART B: MASTER PROMPT (paste this into Claude Code)

```
You are building the project described in SPEC.md in this folder. Read SPEC.md and CLAUDE.md completely first, then read them again section by section as you implement. Your job is to deliver a fully working, tested, demo-ready project that satisfies every requirement in SPEC.md section 1 and the "Definition of done" in section 18. Do the whole build yourself, end to end, without stopping to ask me unless truly necessary.

ENVIRONMENT FACTS
- Windows laptop, PowerShell. Only Python and Ollama are installed. You must install everything else (packages, models, voices) during the setup phase, using the lists in SPEC.md section 3.
- PYTHON 3.12 IS REQUIRED. The machine may have a newer Python (for example 3.14) that lacks wheels for onnxruntime, ctranslate2/faster-whisper, sounddevice and piper-tts. Do this as the very first action of C0, before any package install:
  1. Run `py -0` and `python --version` to see what is installed.
  2. If Python 3.12 is not listed, download and install it side by side (do not remove other versions):
     a) `winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements`
     b) If winget is unavailable or fails: download the latest Python 3.12.x Windows 64-bit installer from https://www.python.org/downloads/ (the 3.12 release page) with PowerShell `Invoke-WebRequest`, then run it silently: `.\python-3.12.x-amd64.exe /quiet InstallAllUsers=0 Include_launcher=1 PrependPath=0 Include_pip=1`. Delete the installer afterwards.
  3. Verify with `py -3.12 --version` (must print 3.12.x).
  4. Create the project venv ONLY with `py -3.12 -m venv .venv` and activate it. Confirm `python --version` inside the venv prints 3.12.x. Every later command, script and test must use this venv.
  5. Record the exact Python version in docs/DEVIATIONS.md and requirements.lock.
  If any required package still has no wheel on 3.12, say so, use the fallback from SPEC.md section 16, and log it.
- No GPU use, ever. No torch. Run `python -m pip list` at the end to prove neither torch nor any GPU package is present.

HOW TO WORK
1. First, make a short written plan in PLAN.md: the checkpoints C0..C8 from SPEC.md with the exact commands you will run to verify each. Then start.
2. Create PROGRESS.md and update it after every checkpoint (what works, what was verified, what deviated).
3. Work checkpoint by checkpoint (C0 to C8). For each checkpoint: implement, write tests, RUN them, fix failures, and only then move on. Commit to git after each checkpoint (git init on the first step).
4. Verify each dependency's real API after installing it (read the installed package source or run its help) before writing code that uses it. Do not rely on memory for APIs.
5. Because you cannot speak into my microphone, build and use these so you can test fully without me:
   - a `--wav <file>` input mode and a `--no-audio-out` mode that writes the reply WAV to disk,
   - the synthetic test set generator (bench/make_testset.py) using Piper,
   - a loopback test that feeds test WAVs through the whole pipeline and logs the timeline.
   Only the last-mile live-microphone and speaker test requires me; list exactly what I need to do for that in docs/LIVE_TEST.md (steps, expected result).
6. Everything must run offline at runtime. Prove it: run the end-to-end test with the network guard enabled and show the external-network-attempt counter is 0. Also show `ollama` reports zero VRAM for the model in use.
7. Be honest: if a feature cannot be made to work reliably within reason, implement the fallback from SPEC.md section 16, mark it in docs/DEVIATIONS.md, and tell me. Do not hide failures and do not fake results.

BUILD ORDER (follow SPEC.md section 15 exactly)
- C0 Setup: probe machine, venv, install packages, pull Ollama models, download all model files, verify_assets.py, launch our own CPU-only Ollama server on 127.0.0.1:11435 and confirm zero VRAM.
- C1 Baseline sequential loop and timeline logging, working with the network off.
- C2 Guards (gpucheck, netguard, limiter) + metrics + benchmark harness + test set. Choose the declared limit from the machine probe.
- C3 Streaming core: VAD, wake word (with push-to-talk fallback), Moonshine streaming ASR (with fallback), sentence chunker, two-slot TTS pipeline, voice prompt, caps.
- C4 Novelty: prefix warming, adaptive endpointing, router (skills, semantic cache with rendered audio, LLM).
- C5 Governor, degradation test, barge-in.
- C6 Dashboard, eco mode, notes/timers, multilingual (Hindi) and pi_like profile if verified working.
- C7 Full ablation, quantization sweep, quality evaluation, auto-generated report in reports/.
- C8 Polish: README with one-command run, WRITEUP.md (max 2 pages) filled with real measured numbers, demo script, final self-audit.

FINAL SELF-AUDIT (mandatory before you say you are done)
Create docs/AUDIT.md containing a table with one row per requirement R1..R9, S1..S4 and each rubric item from SPEC.md section 1: status (PASS / PARTIAL / FAIL), the exact command you ran, and the real output or file proving it. Re-run the full pytest suite, the end-to-end test, the offline proof, the GPU-free proof and the limit-enforcement test one last time and paste results. Fix any FAIL. Be strict and honest.

ONE-COMMAND OPERATION
Provide these PowerShell scripts, tested:
- `scripts\setup.ps1` (does the whole setup phase, safe to re-run),
- `scripts\run.ps1` (starts CPU-only Ollama, the limiter, the app and the dashboard; offline),
- `scripts\bench.ps1` (runs the baseline, ablation, quantization sweep and degradation test and generates reports),
- `scripts\demo.ps1` (starts the demo mode with the dashboard open).

When you finish, give me: (1) what works, verified; (2) what deviated and why; (3) exactly what I must do for the live microphone test; (4) the commands to start the demo. Begin now with PLAN.md and C0.
```

---

## PART C: short follow-up prompts (use as needed)

**If it stops or you want it to continue:**
```
Continue from where you stopped. Read PROGRESS.md and PLAN.md, state which checkpoint you are on, and continue until the next checkpoint's tests pass. Show real command output.
```

**If something fails:**
```
That failed. Do not guess. Reproduce it, show me the exact error, find the root cause, fix it, re-run the failing test and the full test suite, and update DEVIATIONS.md if you changed approach.
```

**Before the demo (live mic test):**
```
Walk me through the live microphone test in docs/LIVE_TEST.md step by step, one step at a time. After each step, wait for my result. If something fails, diagnose it using the logs.
```

**To check your work honestly:**
```
Act as a strict hackathon judge. Re-read the original requirements in SPEC.md section 1. Try to break the system: no network, tightened limit, unseen questions, interruptions, long silence, noisy input. Report every failure with evidence, then fix them.
```

**To tune performance:**
```
Latency is too high. Using the timeline logs, find the slowest stage, propose one change, implement it, re-run the benchmark on the same test set, and report before/after numbers. Repeat for the top three bottlenecks.
```

**Final freeze:**
```
Freeze the project. Run the full test suite, scripts\bench.ps1, and the offline/GPU/limit proofs. Regenerate reports and WRITEUP.md from real results. Make sure README.md has one-command setup and run instructions. Commit and tag v1.0-demo.
```
