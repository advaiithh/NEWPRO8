# SPEC: "PocketVoice", a CPU-only, fully offline voice assistant (HNX26EPS08)

**Audience:** Claude Code, building from scratch on a Windows laptop that has only **Python** and **Ollama** installed.
**Goal:** win the Edge AI hackathon problem statement 8 (On-Device Conversational Stack) in 24 hours.

---

## 0. Rules for Claude Code (read first)

1. **Two phases.** *Setup phase* (online allowed): install packages and download models. *Runtime phase* (strictly offline): nothing may touch a non-loopback network address. After setup finishes, never download anything again.
2. **No GPU, ever.** No CUDA/ROCm/DirectML/Vulkan/Metal builds, no `onnxruntime-gpu`, no CUDA torch. **Do not install torch at all** (it adds hundreds of MB of RAM and is not needed).
3. **Verify, don't assume.** Package APIs and model tags change. After installing each dependency, read its installed source or `--help` and adapt the code. If a named model or voice does not exist, use the listed fallback and record it in `docs/DEVIATIONS.md`.
4. **Measure, don't claim.** Every latency/RAM/energy number in the README, dashboard and report must come from `bench/` output. Numbers in this spec are *targets*, not results.
5. **Work in checkpoints (section 15).** After each checkpoint, run its acceptance test, fix failures, commit. Do not start the next checkpoint with a red test.
6. **Keep the runtime lean.** Stdlib wherever possible. Every extra dependency must justify itself in RAM or latency.
7. **Windows first.** Use PowerShell commands and `pathlib`. Paths in this spec use forward slashes.
8. Write short, readable code. One class per module. Type hints. No global mutable state except `config`.

---

## 1. Problem statement traceability (every line of PS8 → where it is met)

| # | PS8 requirement (verbatim intent) | How this project satisfies it | Acceptance test |
|---|---|---|---|
| R1 | Full ASR → LLM → TTS conversational loop, on-device | `pipeline.py` wires VAD → ASR → router → LLM → TTS | `tests/test_e2e.py` speaks a question, gets spoken audio |
| R2 | Fully offline, no cloud | Network tripwire (`guard/netguard.py`) blocks non-loopback sockets at runtime; Ollama bound to 127.0.0.1; models pre-downloaded | `tests/test_offline.py` (attempted external connect must raise and be counted) + demo with Wi-Fi off |
| R3 | CPU-only, no GPU (pass/fail gate) | CPU-only env vars, `num_gpu=0`, ORT CPU provider only, no torch; **GPU proof panel** | `guard/gpucheck.py` asserts and logs; `tests/test_gpu_free.py` |
| R4 | Simulated edge device: restricted cores + RAM enforced by OS/container | `guard/limiter.py` (Windows affinity + Job Object memory cap, applied to the app **and** the Ollama process tree) | `tests/test_limits.py` proves cap is enforced and RSS total ≤ cap |
| R5 | Wake-word / VAD handling | openWakeWord + Silero VAD (ONNX) | `tests/test_wake_vad.py` |
| R6 | Beat the baseline on end-to-end latency (end of speech → first audio of reply) | Streaming + prefix warm + adaptive endpointing + router/cache; measured by `bench/run_bench.py` | Bench table shows lower p50 and p95 than baseline |
| R7 | Lower resource footprint than baseline (CPU, RAM, energy) | Smaller/quantized models, no torch, mmap, small ctx, thread pinning, idle sleep | Bench table: RSS, CPU-seconds, joules/turn vs baseline |
| R8 | Report CPU, RAM, energy | `metrics/` collectors + `report/` generator | `reports/report.md` auto-generated |
| R9 | Short write-up | `reports/WRITEUP.md` template filled from bench output | Exists, ≤ 2 pages |
| S1 | Stretch: multilingual | Language-ID + per-language ASR/TTS (Hindi + English minimum; others if voices exist) | `tests/test_multilingual.py` |
| S2 | Stretch: smaller device | `profiles/pi_like.yaml` (2 cores / 2 GB simulated) and optional real-device notes | Bench run under that profile |
| S3 | Stretch: cached / pre-synthesized TTS | Semantic response cache with pre-rendered audio | Bench shows cache-hit latency and hit rate |
| S4 | Stretch: keep quality as limit is tightened | Governor tiers + quality eval per tier | Quality-vs-limit plot |
| Rubric 25% | Latency vs baseline | See §9, §10 | Bench |
| Rubric 25% | Footprint vs baseline | See §9, §10 | Bench |
| Rubric 15% | Quantization + offload strategy | Quantization sweep (§10.2); "offload" = on-demand model load/unload, mmap, core pinning, compute avoidance (skills/cache) | `bench/run_quant_sweep.py` |
| Rubric 15% | Graceful degradation | Governor + live "tighten the limit" control (§7) | `bench/run_degrade.py` plot |
| Rubric 20% | Research contribution, with ablation | Ablation ladder + 2 novel ideas (§8 D1, D2) | `bench/run_ablation.py` |
| Bonus | Multilingual, smaller-device, cached TTS | S1, S2, S3 | as above |

**Held-out rule:** judges speak unseen requests. Never hard-code answers to demo phrases. The cache and skills must *generalize* (semantic match, not exact string).

---

## 2. Hard constraints and design principles

- **Declared limit (default):** `4 cores, 4096 MB RAM` total for the *entire* stack including `ollama` processes. Alternate profiles: `eco` (2 cores / 2048 MB), `pi_like` (2 cores / 1536 MB), `roomy` (6 cores / 6144 MB). Probe the machine first (§3.1) and never declare a limit the machine cannot give.
- **Latency definition (used everywhere):** `T_first_audio − T_end_of_speech`, where `T_end_of_speech` is the timestamp of the last voiced audio frame (from the VAD) and `T_first_audio` is when the first sample of the reply is handed to the audio device. Log it per turn.
- **Honesty switch:** an optional "filler" cue (a short "mm") must be **off** in all scored runs. If shown in the demo, label it.
- **Idle must be cheap:** while waiting for the wake word, only the VAD and wake-word model run. ASR/LLM/TTS stay loaded but idle (or unloaded in `eco`).

---

## 3. Environment setup (setup phase, online allowed)

### 3.1 Machine probe (do this first)
Write `scripts/probe.py` that prints: Windows version, CPU model, physical/logical cores, total/free RAM, battery presence, whether a discrete GPU exists (informational only; we will not use it), Python versions available (`py -0`), and Ollama version (`ollama --version`). Save to `reports/machine.json`. Pick the declared limit from this.

### 3.2 Python version
**Python 3.12 is required for this project.** The machine may have a newer Python (for example 3.14) for which wheels of onnxruntime, ctranslate2, sounddevice and piper may not exist. Install 3.12 side by side (never remove other versions), then build the venv from it:

```powershell
py -0                                     # list installed versions
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
# If winget fails: download the Python 3.12.x Windows 64-bit installer from python.org and run:
#   .\python-3.12.x-amd64.exe /quiet InstallAllUsers=0 Include_launcher=1 PrependPath=0 Include_pip=1
py -3.12 --version                        # must print 3.12.x
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version                          # must print 3.12.x inside the venv
python -m pip install --upgrade pip wheel
```
Record the exact version in `docs/DEVIATIONS.md` and `requirements.lock`. If a package still has no 3.12 wheel, use the fallback in §16.

### 3.3 Python packages (CPU-only; pin exact versions after first successful install into `requirements.lock`)

| Package | Purpose | Notes |
|---|---|---|
| `numpy` | audio math | |
| `sounddevice` | mic + speaker I/O | needs PortAudio (bundled in the Windows wheel) |
| `onnxruntime` | Silero VAD, Piper, wake word, embeddings | **CPU package only** |
| `faster-whisper` | baseline ASR + multilingual ASR | CTranslate2 int8 on CPU; pulls `ctranslate2`, `tokenizers`, `huggingface-hub` |
| `moonshine-voice` (or `useful-moonshine-onnx` if that is the maintained name; check PyPI) | streaming low-latency English ASR | verify package name and API after install |
| `openwakeword` | wake word | use ONNX inference; run its model download helper |
| `piper-tts` | TTS | voices downloaded separately (§3.5); if wheel is missing, use the Piper release binary and call it via `subprocess` |
| `model2vec` | tiny static embeddings for the semantic cache/router | fallback: `rapidfuzz` |
| `rapidfuzz` | fuzzy matching fallback + intent matching | |
| `psutil` | CPU/RAM metering, affinity | |
| `pywin32` | Windows Job Object memory/CPU limits | |
| `httpx` | talk to Ollama over loopback (streaming) | |
| `pyyaml` | config | |
| `jiwer` | WER measurement | |
| `matplotlib` | report charts | |
| `pytest` | tests | |
| `soundfile` | read/write WAV | |
| `pyttsx3` | "Starved tier" offline TTS fallback (Windows SAPI) | |

Do **not** install: `torch`, `torchaudio`, `tensorflow`, `onnxruntime-gpu`, `llama-cpp-python` with GPU flags, any cloud SDK.

### 3.4 Ollama models (pull once, then run offline)
Ollama is already installed. Pull these (verify tag names with `ollama list` / the library; use the closest available quantization):

| Role | Preferred tag | Fallback | Why |
|---|---|---|---|
| **Tier Full LLM** | `qwen2.5:1.5b-instruct` (Q4_K_M) | `llama3.2:1b` | Strong instruction-following at small size; no "thinking" mode to disable |
| **Tier Medium/Low LLM** | `qwen2.5:0.5b-instruct` | `smollm2:360m` / `gemma3:1b` | Very fast, lower RAM |
| **Quality option** (if hardware allows ≥6 GB) | `qwen2.5:3b-instruct` | `llama3.2:3b` | Used only in the `roomy` profile and the quality benchmark |
| **Baseline LLM** | same family as Full tier but **default Ollama settings** (default ctx, no streaming into TTS) | | Fair, standard baseline |

If a Qwen3 model is used, thinking **must** be disabled (`think: false` in the API); reasoning tokens destroy voice latency.

### 3.5 Other model downloads (setup phase)

| Asset | Source (verify repo/path) | Destination |
|---|---|---|
| Silero VAD ONNX (`silero_vad.onnx`) | GitHub `snakers4/silero-vad` repo (`src/silero_vad/data/`) | `models/vad/` |
| openWakeWord models | `openwakeword.utils.download_models()` | `models/wakeword/` |
| faster-whisper `tiny.en`, `base.en` (CT2 int8) | Hugging Face `Systran/faster-whisper-tiny.en`, `-base.en` | `models/asr/` |
| faster-whisper multilingual `base` or `small` (only if multilingual enabled) | `Systran/faster-whisper-base` / `-small` | `models/asr/` |
| Moonshine streaming models (tiny, small) | via the Moonshine package's own downloader | `models/asr/moonshine/` |
| Piper voices: `en_US-lessac-medium`, `en_US-lessac-low` (or `en_US-amy-low`), `en_GB-alba-medium`, `hi_IN-*` (check which Hindi voices exist) | Hugging Face `rhasspy/piper-voices` (`.onnx` + `.onnx.json`) | `models/tts/` |
| model2vec static embedding (`minishlab/potion-base-8M` or smaller) | Hugging Face | `models/embed/` |

After downloading, set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and load every model **from local paths only**. Write `scripts/verify_assets.py` that checks every file exists and records size + SHA-256 in `models/MANIFEST.json`.

---

## 4. Repository layout

```
pocketvoice/
  SPEC.md                    (this file)
  config/
    default.yaml             declared limit, model paths, thresholds
    profiles/{eco,pi_like,roomy}.yaml
  scripts/                   probe.py, setup_models.py, verify_assets.py, launch_ollama.ps1, run.ps1
  pv/
    audio/                   io.py (mic/speaker, ring buffer), playback.py (interruptible)
    vad/                     silero_onnx.py, endpoint.py (adaptive endpointing)
    wake/                    oww.py
    asr/                     base.py, moonshine_asr.py, whisper_asr.py, langid.py
    router/                  router.py, skills/ (time, date, calc, timer, units, notes), cache.py, intents.yaml
    llm/                     ollama_client.py (streaming), prompts.py, warm.py
    tts/                     piper_tts.py, sapi_fallback.py, chunker.py, cache_audio.py
    governor/                governor.py, tiers.py, pressure.py
    guard/                   netguard.py, gpucheck.py, limiter.py
    metrics/                 timeline.py, resources.py, energy.py
    pipeline.py              orchestrates everything
    app.py                   CLI entry (live mic)
  dash/                      server.py (stdlib http + SSE), index.html (self-contained, no CDN)
  bench/                     testset/, make_testset.py, run_bench.py, run_ablation.py, run_quant_sweep.py, run_degrade.py, quality_eval.py
  reports/                   auto-generated tables, charts, WRITEUP.md
  tests/
  docs/DEVIATIONS.md
```

---

## 5. Module contracts

### 5.1 `guard/gpucheck.py` (pass/fail gate)
- At import time set: `CUDA_VISIBLE_DEVICES=-1`, `HIP_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=<cores>`.
- Assert every `onnxruntime` session reports only `CPUExecutionProvider`; log it.
- Assert `ctranslate2` devices are CPU (`device="cpu"` explicitly; never `"auto"`).
- Query Ollama's running-model info (`/api/ps`) and assert `size_vram == 0`.
- Publish a **GPU PROOF** record: providers, devices, Ollama `size_vram`, and (if a GPU exists) a note to show Task Manager's GPU graph flat. Display on the dashboard.

### 5.2 `guard/netguard.py` (offline proof)
- Monkeypatch `socket.socket.connect`/`connect_ex`/`getaddrinfo` so any non-loopback destination raises `OfflineViolation` and increments a counter. Loopback (`127.0.0.1`, `::1`) is allowed (Ollama, dashboard).
- Counter and last-violation shown on the dashboard ("External network attempts: 0").
- Optional Windows firewall rule script `scripts/offline_fw.ps1` that blocks outbound traffic for the venv's `python.exe` and the dedicated `ollama.exe` (document it; run as admin only if the user agrees).

### 5.3 `guard/limiter.py` (the simulated edge device)
- Apply to the app process **and** every child/related process, including the Ollama server and its runner processes.
- **CPU:** `psutil.Process.cpu_affinity([...])` to pin to N logical cores. Prefer physical cores; keep ASR/TTS on a different core subset than the LLM where N ≥ 4 (e.g. LLM: 2 cores, ASR+TTS+VAD: 1–2 cores).
- **RAM:** create a Windows **Job Object** (`pywin32`: `win32job`) with `JOB_OBJECT_LIMIT_PROCESS_MEMORY`/`JOB_OBJECT_LIMIT_JOB_MEMORY` set to the declared MB and assign all processes to it. Also run a watchdog that reads total RSS of the process tree and triggers the governor before the cap is hit.
- Provide `Limiter.tighten(cores, mb)` for live changes during the demo.
- Print the active limit prominently at startup and on the dashboard.

### 5.4 `scripts/launch_ollama.ps1`
Start **our own** Ollama server (do not rely on the tray instance; stop it first or use a different port) bound to loopback with CPU-only settings:
```
OLLAMA_HOST=127.0.0.1:11435
CUDA_VISIBLE_DEVICES=-1
OLLAMA_LLM_LIBRARY=cpu          (if supported in the installed version; otherwise rely on num_gpu=0)
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1      (2 only in roomy profile)
OLLAMA_KEEP_ALIVE=-1            (keep warm; governor can unload in eco)
OLLAMA_FLASH_ATTENTION=1        (verify support on CPU build; ablate)
OLLAMA_KV_CACHE_TYPE=q8_0       (needs flash attention; ablate vs f16)
```
Then the limiter pins and caps the server process tree.

### 5.5 `vad/silero_onnx.py` + `vad/endpoint.py`
- Silero VAD via `onnxruntime` (no torch), 16 kHz, 32 ms frames, 1 intra-op thread. Target <1 ms/frame.
- **Adaptive endpointing (novel, D2):** the silence hangover is not fixed. Start at `hangover_ms = 350`. Extend (up to 700) if the partial transcript looks incomplete (ends with "and", "but", "the", a comma, or is < 2 words); shorten (down to 200) if it ends with a complete clause/question mark pattern or matches a known skill/cache intent. Log the chosen hangover per turn so the ablation can show the effect.

### 5.6 `wake/oww.py`
- openWakeWord ONNX, runs on the same audio ring buffer. Default wake phrase: one of the built-in models (e.g. "hey jarvis"); make the phrase configurable. Also support **push-to-talk** (key) and **always-listening mode** (VAD only) for judges' convenience.
- **On wake detection:** immediately trigger prefix warming (§5.9) and ASR pre-activation.

### 5.7 `asr/`
Common interface:
```python
class ASR:
    def start_stream(self, lang_hint=None): ...
    def feed(self, pcm16: np.ndarray) -> Partial   # returns partial text (may be empty)
    def finalize(self) -> Final                    # text, lang, confidence, ms
```
- `moonshine_asr.py`: streaming English ASR (tiny for Low tier, small for Full tier). Partials flow to the endpointer and to the router for early matching.
- `whisper_asr.py`: faster-whisper, `device="cpu"`, `compute_type="int8"`, `beam_size=1`, `vad_filter=False` (we have our own), `without_timestamps=True`, `condition_on_previous_text=False`. Used for (a) the **baseline** (base.en, whole utterance) and (b) **non-English** languages (multilingual model).
- `langid.py`: cheap language ID. First choice: script/word heuristics on a first-pass transcript; second: Whisper's built-in language detection on the first ~2 s only (multilingual model, only when enabled).
- Warm up every model at startup with 1 s of silence + a short synthetic phrase.

### 5.8 `router/` (compute-avoidance router, novel D1)
Order of attempts per final transcript, cheapest first:
1. **Skills** (deterministic, no LLM, no TTS synthesis cost beyond a short string): time, date, day, timer/alarm set and list, simple arithmetic ("what is 17 times 23", "20 percent of 450"), unit conversion, coin flip/dice, repeat last answer, volume/speed commands, "what can you do", note-taking ("remember that…", "what did I ask you to remember") stored locally in JSON. Use regex + `rapidfuzz` intent matching; arithmetic via a safe expression evaluator (no `eval`).
2. **Semantic cache:** embed the query (model2vec, <5 ms) and compare to `router/intents.yaml` entries (greetings, thanks, identity, jokes, capabilities, "how are you", small talk, 60–120 intents with 3–8 paraphrases each, each with 1–3 pre-written answers). If cosine ≥ τ_hit (default 0.80) and the margin over the 2nd best ≥ 0.05 → **play pre-synthesized audio** from `cache/audio/<lang>/<id>_<k>.wav` (rendered at setup by `scripts/render_cache.py`). Log hit/miss and the similarity. Tune τ on a held-out paraphrase set; report precision of hits (wrong-answer rate).
3. **LLM** (fallback for everything else).
Report: % of turns avoided, latency and joules saved by tier. This router is a primary research contribution.

### 5.9 `llm/ollama_client.py` + `llm/warm.py`
- `httpx` streaming client to `http://127.0.0.1:11435/api/chat`, `stream=true`, options: `num_gpu=0`, `num_thread=<llm cores>`, `num_ctx=1024` (Full) / `768` (Low), `num_predict` capped by tier (e.g. 80 / 48 / 32 tokens), `temperature=0.4`, `repeat_penalty=1.1`, `keep_alive=-1`, `think=false` for models that support it.
- **Voice system prompt** (`prompts.py`): "You are a voice assistant. Reply in one to three short spoken sentences. No lists, markdown, emoji or code. Spell out numbers naturally. If unsure, say so briefly." Keep it **fixed and short** so its KV prefix is cacheable.
- **Prefix warming (D3):** at wake-word detection (or push-to-talk key-down), send a no-op request containing system prompt + conversation history prefix with `num_predict=1` so Ollama's prompt cache is hot when the real question arrives. Measure the first-token gain in the ablation.
- Keep a rolling history of the last 2–3 turns (token-capped) to stay inside `num_ctx`.
- Support **cancellation** (barge-in): close the stream immediately.
- Backend interface `LLMBackend` so an alternative runtime can be dropped in, but Ollama is the shipped one.

### 5.10 `tts/`
- `piper_tts.py`: Piper via ONNX Runtime CPU (`piper-tts`) with `length_scale` tuned (~0.9 faster speech), streaming per chunk. Keep the voice loaded. Voices: `lessac-medium` (Full), `lessac-low`/`amy-low` (Low), Hindi voice for `hi`.
- `chunker.py` (**sentence-level streaming, the biggest latency win**): read LLM tokens; cut a chunk at the first sentence end (`. ? !` followed by space), or at a clause break (`, ; :`) once ≥ 6 words, or after 12 words with no punctuation. Send each chunk to Piper immediately; play chunk *k* while synthesizing chunk *k+1* (two-slot pipeline). The **first chunk** should be short (target 4–10 words).
- `playback.py`: non-blocking, interruptible, with a small jitter buffer; exposes `T_first_audio` timestamp when the first sample is written to the device.
- `sapi_fallback.py`: `pyttsx3`/Windows SAPI as the Starved tier voice.
- **Barge-in (feature D5):** while playing, keep the VAD running on the mic; if speech is detected for > 250 ms, stop playback, cancel the LLM stream, and start a new turn. Use the playback reference to avoid self-triggering (reduce sensitivity during playback; recommend headphones for the demo, document it).

### 5.11 `governor/` (graceful degradation, D4)
Input signals every 500 ms: declared limit, total RSS vs cap, process-tree CPU %, rolling LLM tokens/s, rolling turn latency (p50 of the last 5), battery state (eco). Output: a **tier**.

| Tier | ASR | LLM | TTS | Reply cap | Extras |
|---|---|---|---|---|---|
| **T0 Full** | Moonshine small | qwen2.5:1.5b, ctx 1024 | Piper medium | 80 tokens | all features |
| **T1 Medium** | Moonshine tiny | qwen2.5:1.5b, ctx 768 | Piper medium | 56 tokens | |
| **T2 Low** | Moonshine tiny | qwen2.5:0.5b, ctx 768 | Piper low | 40 tokens | shorter system prompt |
| **T3 Starved** | Moonshine tiny | skills + cache first; qwen2.5:0.5b only if idle CPU; ctx 512 | Piper low / SAPI | 24 tokens | answers prefer cache |
| **T4 Survival** | whichever is loaded | cache + skills only; LLM replaced by "I'm low on resources, here is a short answer" template | pre-rendered audio | n/a | never crashes |

Rules: **hysteresis** (downgrade fast, upgrade only after 20 s of headroom); never swap models mid-sentence; unload the larger LLM when downgrading to free RAM; log every transition with the reason. Provide `governor.force_tier()` and the dashboard control "Tighten limit" which calls `Limiter.tighten()` and lets the governor react live.

---

## 6. Pipeline timing and threading

- Threads: `audio_in` (callback → ring buffer), `vad_wake` (frames → events), `asr` (streaming), `route_llm` (router + Ollama stream), `tts` (synthesis), `audio_out` (playback), `metrics`, `dash`. Queues are bounded; use `threading` (the heavy work is in C/ONNX, which releases the GIL).
- Pin threads to the core groups chosen by the limiter.
- Every turn produces a `Timeline` record with timestamps: `wake`, `speech_start`, `speech_end`, `asr_final`, `route_decision`, `llm_first_token`, `tts_first_chunk_ready`, `first_audio`, `turn_end`, plus tier, hangover used, cache/skill/LLM path, tokens, and resource samples.

**Latency targets (to verify, not promise):** end-of-speech → first audio: skill ≤ 400 ms, cache hit ≤ 450 ms, LLM path ≤ 1500 ms p50 / ≤ 2500 ms p95 on 4 cores. Baseline expected to be several seconds.

---

## 7. Features (what makes this project unique)

**Core (required):** F1 offline voice loop · F2 wake word + VAD · F3 CPU-only + proof · F4 enforced resource limit · F5 metrics + report.

**Differentiators (implement in this priority order):**
- **D1 Compute-avoidance router** (skills → semantic cache → LLM) with measured savings. *Primary research contribution.*
- **D2 Adaptive semantic endpointing** (hangover depends on how complete the partial transcript is).
- **D3 Wake-time prefix warming** (LLM KV prefix and ASR hot before the user finishes asking).
- **D4 Pressure-driven tier governor** with live "tighten the limit" demo; quality and latency degrade smoothly instead of failing.
- **D5 Barge-in** (interrupt the assistant mid-sentence).
- **D6 Sentence-level streaming** with a two-slot synth/playback pipeline and short first chunk.
- **D7 Voice-optimized text normalizer:** converts digits, symbols, units and dates to speakable text before TTS ("₹450" → "four hundred fifty rupees"); strips markdown. Prevents Piper mispronunciations.
- **D8 Local notes + timers** persistent across restarts (JSON), all offline.
- **D9 Eco/battery mode:** on battery, cap threads, lengthen idle polling, prefer cache/skills, report estimated energy per turn.
- **D10 Live dashboard** (§11) with per-turn latency waterfall, resource gauges, tier, GPU proof, network-attempt counter.
- **D11 Self-generated test set:** the bench synthesizes spoken test questions with Piper (different voices/speeds, optional noise) so evaluation is repeatable and offline, plus a live-mic mode for judges.
- **D12 Multilingual routing** (Bonus): detect language; use Whisper multilingual for non-English ASR; Piper voice for that language; if the LLM is weak in that language, prefer skills/cache templates in that language. Ship Hindi + English minimum; add others only after verifying a real voice exists and sounds acceptable. Never claim a language that was not tested.
- **D13 Smaller-device profile** (`pi_like`): 2 cores / 1.5 GB, Low tier default; documented results.

Optional stretch if time remains: a **quality-aware cache learner** (promote frequently repeated LLM answers into the cache at idle time, with user-visible opt-in), and a **thermal/CPU-frequency log** to show sustained performance.

---

## 8. Baseline (define exactly; the judges choose nothing, we do)

> **Baseline = standard quantized stack, default settings, same limit, same machine:** faster-whisper `base.en` int8 (default beam) on the whole utterance after a fixed 800 ms silence → Ollama `qwen2.5:1.5b` Q4_K_M with default options (default context, no prefix warm, full reply generated) → Piper `lessac-medium` synthesizing the **full** reply, then play. No router, no governor, no barge-in.

Run the baseline and our system with identical inputs under identical limits. Keep the baseline honest; do not cripple it.

---

## 9. Metrics

Per turn: latency (definition in §2), stage breakdown, tier, path (skill/cache/LLM), tokens/s, hangover used.
Per run: p50/p95/mean/worst latency; **peak total RSS (process tree)**; **CPU-seconds per turn** and average CPU % (via `psutil`, whole tree); **idle CPU %** and idle RSS; **energy per turn (joules)**.
**Energy on Windows (no RAPL):** try, in order: (1) vendor power counters via `typeperf`/Windows "Energy Meter" performance counters if present; (2) battery discharge rate (`psutil.sensors_battery`, WMI `BatteryStatus`) measured over a long loop of turns on battery; (3) otherwise a **clearly labeled proxy**: `energy_proxy_J = cpu_seconds × assumed_W_per_core` with the assumption printed next to every energy number. Never present the proxy as a measurement.
Quality: ASR WER (`jiwer`) on the test set; answer-quality score on 40 fixed prompts (keyword/rubric checks, plus a blind 1–5 human rating by two teammates); TTS intelligibility = WER of Whisper-transcribing our TTS output (sanity check).

---

## 10. Benchmarks and ablation (the 20% research score + evidence for everything else)

### 10.1 Test set (`bench/make_testset.py`)
60 spoken requests in WAV: 20 skills-type, 20 small-talk/cache-type (paraphrased, **not** copies of cache entries), 20 open-ended factual/chat. Synthesized with 3 Piper voices/2 speeds, plus 10 real-mic recordings from teammates. Keep 20 requests **unseen** (written by someone who has not seen the cache file) as the held-out set. Include 10 Hindi requests for the multilingual run.

### 10.2 Quantization sweep (`bench/run_quant_sweep.py`) → rubric "Quantization + offload" 15%
| Component | Variants | Report |
|---|---|---|
| LLM | `qwen2.5:1.5b` at Q8_0, Q5_K_M, Q4_K_M, Q3_K_M (via Ollama tags; if a tag is missing, create with `ollama create` from a local GGUF only if available offline) and 0.5b Q4 | tokens/s, first-token ms, RSS, answer-quality score |
| LLM KV cache | f16 vs q8_0 (and flash attention on/off) | RAM, latency |
| ASR | whisper base.en fp32 vs int8; moonshine tiny vs small | WER, RTF, RSS |
| TTS | Piper medium vs low | RTF, subjective rating |
Output: a table + Pareto plot (quality vs latency vs RAM) and the chosen configuration per tier.
"Offload" evidence: model load/unload timing, mmap RSS vs working set, core pinning on/off, and fraction of turns served without the LLM.

### 10.3 Ablation ladder (`bench/run_ablation.py`)
Each row adds **one** change on top of the previous. Same test set, same limit, 3 repeats, warm-up discarded.
| Step | Change |
|---|---|
| A0 | Baseline (§8) |
| A1 | + Moonshine streaming ASR with VAD endpointing (fixed 500 ms) |
| A2 | + Sentence-level streaming into TTS (D6) |
| A3 | + Short voice prompt, ctx 1024, reply cap |
| A4 | + Prefix warming at wake (D3) |
| A5 | + Adaptive endpointing (D2) |
| A6 | + Compute-avoidance router: skills + cache (D1) |
| A7 | + Governor (D4) |
Report per step: latency p50/p95, RSS peak, CPU-s/turn, J/turn, quality score. Show a waterfall chart of per-stage savings.

### 10.4 Degradation test (`bench/run_degrade.py`)
Run the same 40 requests while stepping the limit: `6c/6GB → 4c/4GB → 3c/3GB → 2c/2GB → 2c/1.5GB`, for **fixed Full config vs governor**. Plot latency, completion rate (no crash/timeouts), and quality vs limit. The fixed config should degrade sharply or fail; the governor should degrade smoothly. This is the demo's centerpiece.

### 10.5 Cache honesty
On the held-out paraphrase set report cache hit rate, wrong-answer rate, and the effect of τ_hit. Include failure cases in the write-up.

---

## 11. Dashboard (`dash/`)
Self-contained single HTML file (no CDN, no external fonts), served by a stdlib HTTP server on `127.0.0.1:8765` with Server-Sent Events. Panels:
1. **Device limit** (cores, MB) with a "Tighten limit" control.
2. **Live gauges:** total RSS vs cap, CPU %, tokens/s.
3. **Current tier** with transition log and reasons.
4. **Last-turn latency waterfall** (stages) and a rolling p50/p95.
5. **Proof panel:** GPU = none (providers, Ollama VRAM 0), external network attempts = 0, models loaded from local paths.
6. **Route badge** per turn: skill / cache / LLM, with similarity score.
7. **Baseline vs ours** results table loaded from `reports/`.
Keep the dashboard cheap (update ≤ 2 Hz) so it does not distort measurements; provide `--no-dash` for benchmark runs.

---

## 12. Tests (`pytest`)
- `test_gpu_free.py`: providers CPU-only, `size_vram == 0`.
- `test_offline.py`: external connect raises and increments counter; loopback allowed.
- `test_limits.py`: affinity set; job-object memory cap present; total RSS < cap during a 20-turn run.
- `test_vad_endpoint.py`: hangover extends for incomplete phrases, shrinks for complete ones.
- `test_chunker.py`: first chunk length and boundaries.
- `test_normalizer.py`: numbers, currency, dates, units.
- `test_skills.py`: arithmetic, timer, notes persistence, no `eval` injection.
- `test_router_cache.py`: paraphrases hit, out-of-domain misses, margin rule.
- `test_governor.py`: hysteresis and tier transitions with simulated pressure.
- `test_e2e.py`: WAV in → audio out, with timeline fields populated.
- `test_multilingual.py`: Hindi sample routes to the Hindi voice.

---

## 13. Offline and CPU-only proof checklist (run before the demo)
1. `scripts/verify_assets.py` passes (all models local, manifest hashes match).
2. Turn Wi-Fi/Ethernet **off**. Start `scripts/run.ps1`. Dashboard shows network attempts = 0.
3. Dashboard proof panel: ORT providers = CPU only; Ollama VRAM = 0; Task Manager GPU graph flat during a conversation (screenshot it).
4. Limiter banner shows the declared cores/RAM; Task Manager shows the process tree under the cap.
5. 5-minute conversation without a crash.

---

## 14. Demo script (5 minutes)
1. Show proof panel (offline, CPU-only, limit enforced).
2. Say "Hey Jarvis" → ask a **skill** question (instant), a **cache** question (instant), an **open** question (streamed answer begins quickly).
3. **Barge in** mid-answer.
4. Ask in Hindi (if enabled).
5. **Tighten the limit live** (4c/4GB → 2c/1.5GB): show governor tier drop, latency stays usable, assistant never crashes.
6. Show the baseline vs ours table and the ablation waterfall.
7. Let a judge ask unseen questions live.
Prepare a **backup recording** of the whole demo.

---

## 15. Build order with checkpoints (24 hours)

| Hours | Checkpoint | Acceptance |
|---|---|---|
| 0–1.5 | **C0 Setup:** probe, venv, packages, model downloads, `verify_assets.py`, launch our Ollama (CPU-only) | Each model runs standalone; `ollama` reports 0 VRAM |
| 1.5–5 | **C1 Baseline loop:** record → faster-whisper → Ollama → Piper → play, sequential, with timeline logging | Works with network off; baseline numbers recorded |
| 5–7 | **C2 Guards + metrics:** limiter, gpucheck, netguard, resource/energy collectors, bench harness + test set | Limit enforced; first baseline table generated |
| 7–11 | **C3 Streaming core:** VAD+wake, Moonshine streaming, sentence chunker, two-slot TTS, voice prompt, ctx/reply caps | A1–A3 measured; latency clearly below baseline |
| 11–14 | **C4 Novelty:** prefix warming, adaptive endpointing, router (skills + cache + rendered audio) | A4–A6 measured |
| 14–17 | **C5 Governor + degradation + barge-in** | `run_degrade.py` plot; live tighten works |
| 17–19 | **C6 Dashboard + eco mode + notes/timers**; multilingual + `pi_like` if time | Dashboard shows all panels |
| 19–22 | **C7 Full ablation + quant sweep + quality eval** on held-out set; generate report | `reports/` complete |
| 22–24 | **C8 Polish:** write-up (≤2 pages), README, demo rehearsal with unseen sentences, backup video, freeze | Definition of done met |

If behind schedule, **cut in this order:** D12 multilingual → D13 → D9 → D8 → D5. Never cut: C1–C5, R1–R9, the ablation, the proof panel, the write-up.

---

## 16. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Python version lacks wheels | Install Python 3.12 side by side |
| Moonshine package/API differs or fails | Use faster-whisper `tiny.en` int8 with a streaming-by-chunks wrapper (re-run on growing window, VAD-gated); note it in DEVIATIONS |
| `piper-tts` has no wheel | Use the Piper release executable via `subprocess` with raw PCM stdout |
| openWakeWord fails | Push-to-talk plus VAD-only always-listening mode |
| Ollama attempts GPU | Force `num_gpu=0`, `CUDA_VISIBLE_DEVICES=-1`, own server env; verify with `/api/ps` |
| Job-object memory cap kills Ollama | Raise cap by 10%, or enforce via watchdog + governor instead and document |
| 1.5B too slow on 2 cores | Governor drops to 0.5B/cache; this is the demo of degradation |
| Hindi voice missing or poor | Do not claim it; ship English only and list multilingual as tested-partial |
| Energy cannot be measured | Use the labeled proxy and battery-drain test; never call a proxy a measurement |
| Echo/self-trigger during barge-in | Headphones in demo; raise barge-in threshold during playback |

---

## 17. Write-up template (`reports/WRITEUP.md`, ≤ 2 pages, auto-filled)
1. Problem and declared device limit.
2. Baseline definition (exact).
3. System overview (diagram) and what is new.
4. Results table: latency (p50/p95), RSS peak, CPU-s/turn, J/turn (labeled measurement or proxy), quality.
5. Ablation ladder chart and what each step contributed.
6. Quantization sweep and chosen configuration per tier.
7. Degradation plot (fixed vs governor).
8. Cache/router: savings and error rate (honest failures).
9. Offline and CPU-only proof.
10. Limitations (synthetic test voices, proxy energy if used, languages not tested).

---

## 18. Definition of done
- [ ] Voice in → voice out works **offline**, **CPU-only**, under the declared limit, with wake word/VAD.
- [ ] GPU proof and network proof visible on the dashboard.
- [ ] Baseline vs ours: lower latency **and** lower footprint, with numbers from `bench/`.
- [ ] Ablation ladder, quantization sweep and degradation plot generated.
- [ ] Router/cache savings and error rate reported.
- [ ] Live limit-tightening demo works without a crash.
- [ ] Short write-up and backup demo video ready.
- [ ] `docs/DEVIATIONS.md` lists every substitution made during the build.
