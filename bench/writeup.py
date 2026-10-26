"""reports/WRITEUP.md (SPEC 17, <= 2 pages). Every number is read from bench/out; wording follows the comparison."""
from __future__ import annotations

from typing import Any

LABELS = {"A0": "A0 baseline", "A1": "A1 +Moonshine streaming", "A2": "A2 +sentence-level TTS", "A3": "A3 +voice prompt/caps",
          "A4": "A4 +prefix warming", "A5": "A5 +adaptive endpointing", "A6": "A6 +router (skills+cache)", "A7": "A7 +governor"}


def f(x: Any, d: int = 0) -> str:
    return "n/a" if x is None or x != x else f"{x:,.{d}f}"


def delta(a: float, b: float) -> str:
    return f"{(b - a) / a * 100:+.0f}%" if a else "n/a"


def build(abl: dict, deg: Any, quant: Any, cache: Any, qual: Any, proofs: Any) -> str:
    a0 = abl["A0"]
    top = max(k for k in abl if k != "A0")
    last = abl[top]
    avoided = sum(last["paths"].get(p, 0) for p in ("skill", "cache")) / max(last["n_turns"], 1)
    L = ["# PocketVoice: write-up", "",
         "## 1. Problem and device limit", "",
         "A voice assistant (ASR, LLM, TTS) that runs fully offline on CPU only inside a simulated edge device: "
         f"{last['limit']}. The limit is enforced by the OS (CPU affinity plus a Windows Job Object memory cap over the app and the "
         "Ollama process tree).", "",
         "## 2. Baseline (exact)", "",
         "Fixed 800 ms silence, then faster-whisper `base.en` int8 (default beam 5) on the whole utterance, then Ollama "
         "`qwen2.5:1.5b-instruct` Q4_K_M with default options (default context, no prefix warm, full uncapped reply), then Piper "
         "`lessac-medium` on the full reply, then play. Sequential; no router, governor or barge-in; same limit, same machine.", "",
         "## 3. System and what is new", "",
         "```", "mic -> Silero VAD / openWakeWord -> Moonshine streaming ASR -> router: skills -> semantic cache -> LLM (Ollama, CPU)",
         "    -> sentence chunker -> Piper (two-slot) -> speaker;   governor: tiers T0..T4 from RSS, CPU, tok/s, latency", "```",
         "New: **D1** compute-avoidance router (skills, then a semantic cache that plays pre-rendered audio, then the LLM); "
         "**D2** adaptive endpointing (hangover follows how complete the partial transcript is); **D3** prefix warming at wake; "
         "**D4** tier governor with hysteresis and a live tighten-the-limit control; **D5** barge-in; **D6** sentence-level streaming "
         "into a two-slot TTS pipeline; **D7** speakable-text normaliser.", "",
         "## 4. Results (60 spoken requests x 3 repeats, warm-up discarded)", "",
         "| | p50 ms | p95 ms | peak RSS MB | CPU-s/turn | J/turn RAPL (net of idle) | J/turn proxy | quality |",
         "|---|---|---|---|---|---|---|---|"]
    for name, v in (("Baseline (A0)", a0), (f"Ours ({top})", last)):
        L.append(f"| {name} | {f(v['latency_ms']['p50'])} | {f(v['latency_ms']['p95'])} | {f(v['peak_rss_mb'])} | "
                 f"{f(v['cpu_s_per_turn'], 1)} | {f(v['energy_j_per_turn_net_of_idle'], 1)} | {f(v['energy_proxy_j_per_turn'], 1)} | "
                 f"{f(v['quality_pass_rate'], 2)} |")
    L += ["", f"Change vs baseline: p50 {delta(a0['latency_ms']['p50'], last['latency_ms']['p50'])}, p95 "
          f"{delta(a0['latency_ms']['p95'], last['latency_ms']['p95'])}, peak RSS {delta(a0['peak_rss_mb'], last['peak_rss_mb'])}, "
          f"CPU-seconds per turn {delta(a0['cpu_s_per_turn'], last['cpu_s_per_turn'])}, measured energy "
          f"{delta(a0['energy_j_per_turn_net_of_idle'], last['energy_j_per_turn_net_of_idle'])}. Energy source: {last['energy_source']}. "
          f"The proxy column is cpu-seconds x {a0['assumed_w_per_core_for_proxy']} W/core, an assumption and not a measurement.", "",
          "## 5. Ablation (each row adds one change)", ""]
    prev = None
    for k in sorted(abl):
        v = abl[k]
        L.append(f"- **{LABELS[k]}**: p50 {f(v['latency_ms']['p50'])} ms, p95 {f(v['latency_ms']['p95'])} ms"
                 + (f" ({delta(prev, v['latency_ms']['p50'])} p50 vs previous step)" if prev else ""))
        prev = v["latency_ms"]["p50"]
    if quant:
        ok = {k: v for k, v in quant["llm"].items() if "tok_s_mean" in v}
        L += ["", "## 6. Quantization sweep and chosen configuration", "",
              "; ".join(f"{k}: {v['tok_s_mean']:.0f} tok/s, {v['rss_mb']:.0f} MB RSS, quality {v['quality']:.2f}" for k, v in ok.items())
              + ". Per tier: T0/T1 use 1.5B Q4_K_M, T2/T3 use 0.5B Q4, with the KV-cache, ASR and TTS comparisons in `report.md`."]
    else:
        L += ["", "## 6. Quantization sweep", "", "Not run in the shortened benchmark (agreed time budget); `bench/run_quant_sweep.py` implements it and the "
              "Q4_K_M 1.5B / 0.5B Q4 choices are the Ollama defaults, not a measured Pareto optimum."]
    if deg:
        L += ["", "## 7. Graceful degradation (same 40 requests, limit stepped down)", ""]
        for step in next(iter(deg.values())):
            fx, gv = deg["FIXED"].get(step, {}), deg["FULL"].get(step, {})
            L.append(f"- {step}: fixed Full config completes {f(fx.get('completion_rate', 0) * 100)}% (p50 "
                     f"{f(fx.get('latency_ms', {}).get('p50'))} ms); with governor {f(gv.get('completion_rate', 0) * 100)}% (p50 "
                     f"{f(gv.get('latency_ms', {}).get('p50'))} ms)")
    if cache:
        c, h = cache["chosen_on_dev"], cache["heldout_at_chosen"]
        L += ["", "## 8. Router and cache: savings and honest errors", "",
              f"The router answered {avoided:.0%} of turns without the LLM in the full ladder run. Cache threshold tau={c['tau']:.2f}, "
              f"margin {c['margin']}, chosen on the dev paraphrases (hit {c['hit_rate']:.0%}, wrong {c['wrong_answer_rate_of_hits']:.0%} of hits). "
              f"Held-out (n={h['n_pos']}): hit {h['hit_rate']:.0%}, wrong {h['wrong_answer_rate_of_hits']:.0%} of hits, false hits on "
              f"out-of-domain {h['false_hit_rate_on_negatives']:.0%}. Static embeddings confuse questions about a topic with questions to the "
              "assistant (\"why do we sleep\" matches the \"do you sleep\" intent); the failure list is in `report.md`."]
    L += ["", "## 9. Offline and CPU-only proof", ""]
    L.append(proofs["summary"] if proofs else "See `docs/AUDIT.md` (commands and outputs).")
    L += ["", "## 10. Limitations", "",
          "Test speech is synthetic (Piper voices, clean audio); no real-microphone recordings or human ratings were available to the "
          "build (see `docs/LIVE_TEST.md`). The held-out set was written by the same assistant that wrote the cache, so it is less "
          "independent than a teammate-written set. RAPL is whole-package power including background load. Hindi was tested end to end "
          "on a few synthetic phrases only; no other language is claimed. Barge-in was tested with synthetic interruptions, not through a "
          "real speaker-to-microphone path."]
    return "\n".join(L)
