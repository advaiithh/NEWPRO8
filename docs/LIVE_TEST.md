# Live microphone + speaker test (needs you)

Everything else was verified with synthetic speech (Piper) fed through the full pipeline, because an automated
build cannot speak into your microphone. This checklist covers the last mile. Use **headphones** (barge-in listens
to the mic while the assistant is talking; speakers can retrigger it).

## 0. Before you start
1. `scripts\setup.ps1` finished (or the models are already in `models\`).
2. Plug in the laptop (energy numbers are cleaner), close heavy apps.
3. In Windows Sound settings pick the microphone and headphones you will use. List device numbers if needed:
   `.\.venv\Scripts\python.exe -c "import sounddevice as sd; print(sd.query_devices())"`

## 1. Start
```powershell
scripts\demo.ps1            # starts our CPU-only Ollama, the app, and opens http://127.0.0.1:8765
```
Expected: console prints `DEVICE LIMIT: 4 cores ... 4096 MB`, `[proof] {... "pass": true ...}`, then
`Listening (always mode). Just speak - it answers only when you talk...`. The dashboard shows GPU = none, Ollama VRAM 0, network attempts 0.

## 2. Test cases (just say each one; no key and no wake word in the default hands-free mode)
| # | Say | Expected |
|---|---|---|
| 1 | "What time is it?" | Spoken time within ~0.5 s; console route `skill` |
| 2 | "Thank you so much" / "Tell me a joke" | Instant spoken reply; route `cache` with a similarity score |
| 3 | "Why is the sky blue?" | Reply starts quickly and streams sentence by sentence; route `llm` |
| 4 | Stay silent for 20 s, then cough / type / play music at low volume | Nothing is answered, nothing is printed |
| 4b | (optional, headphones, started with `-Extra "--barge-in"`) interrupt a long reply | Speech stops within ~0.3 s and your new question is answered |
| 5 | "Set a timer for ten seconds" | Confirmation, then a spoken "Your 10 seconds timer is done" |
| 6 | "Remember that my locker code is 427" then, later, "What did I ask you to remember?" | Note spoken back; survives restart (`data\notes.json`) |
| 7 | Say "speak Tamil", then "வணக்கம்", "உங்கள் பெயர் என்ன", "ஒரு நகைச்சுவை சொல்லுங்கள்", then "ஆங்கிலத்தில் பேசு" | Spoken confirmation in Tamil; canned Tamil answers in the Tamil voice; back to English. Report any phrase it mis-hears: Tamil has only been tested on synthetic speech |
| 7b | (optional) "speak Hindi", then "नमस्ते" / "आप कैसे हैं" | Hindi voice answers; recognition is weaker (D-12) |
| 8 | Press **T** (tighten to 2 cores / 1536 MB), ask 3 questions, press **R** | Dashboard tier drops (T0 -> lower) with a reason, assistant keeps answering, no crash |

## 3. Offline / GPU proof for the demo
1. Turn Wi-Fi/Ethernet **off** (or optionally run `scripts\offline_fw.ps1` as administrator; undo with `-Remove`).
2. Restart `scripts\demo.ps1`, hold a 5-minute conversation. Dashboard: **external network attempts = 0**.
3. Open Task Manager -> Performance -> GPU: the graph must stay flat while you talk. Screenshot it.
4. Task Manager -> Details: `python.exe`, `ollama.exe`, `llama-server.exe` are pinned to the listed cores.

## 4. Things only people can measure (please report back)
- Subjective TTS rating, 1-5, for `lessac-medium` vs `lessac-low` (two listeners).
- Blind 1-5 rating of 10 answers from the baseline vs ours (two teammates).
- Optional: 10 real recordings of your own voice into `bench\testset\real\` and 20 unseen questions appended
  to `bench\heldout_team.yaml` (same format as `bench\heldout.yaml`), then `python -m bench.cache_eval`.

If something fails, copy the console output and `data\timelines\*.jsonl` and tell me which row of the table failed.
