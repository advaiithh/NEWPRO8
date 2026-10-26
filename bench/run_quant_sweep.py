"""Quantization + offload sweep (SPEC 10.2). Writes bench/out/quant_sweep.json.

LLM: qwen2.5:1.5b at Q8_0 / Q5_K_M / Q4_K_M / Q3_K_M and 0.5b Q4 (tokens/s, first-token ms, RSS, answer quality).
KV cache: f16 vs q8_0, flash attention on/off (RAM, latency).   ASR: whisper int8 vs fp32, Moonshine tiny vs small.
TTS: Piper medium vs low (RTF, Whisper-WER intelligibility).   Offload: load/unload timing, RSS vs private memory,
core pinning on/off.   Everything runs under the declared 4-core limit.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import yaml

from bench.common import HERE, OUT, TESTSET, load_manifest, norm_text, wer
from pv import config as cfgmod
from pv.audio.io import load_wav16k
from pv.guard import netguard
from pv.guard.limiter import Limiter, ollama_pid
from pv.llm.ollama_client import OllamaClient, OllamaOptions

ROOT = HERE.parent
LLMS = {"1.5b-Q8_0": "qwen2.5:1.5b-instruct-q8_0", "1.5b-Q5_K_M": "qwen2.5:1.5b-instruct-q5_K_M",
        "1.5b-Q4_K_M": "qwen2.5:1.5b-instruct-q4_K_M", "1.5b-Q3_K_M": "qwen2.5:1.5b-instruct-q3_K_M",
        "0.5b-Q4": "qwen2.5:0.5b-instruct"}
SYSTEM = "You are a voice assistant. Reply in one to three short spoken sentences. No lists, markdown, emoji or code."


def restart_ollama(kv: str = "q8_0", fa: str = "1") -> None:
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "launch_ollama.ps1"),
                    "-KvCache", kv, "-FlashAttention", fa], check=True, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def runner_rss_mb() -> dict[str, float]:
    o = psutil.Process(ollama_pid())
    kids = [k for k in o.children(recursive=True) if "llama" in k.name().lower() or "runner" in k.name().lower()]
    if not kids:
        return {"rss_mb": 0.0, "uss_mb": 0.0}
    k = kids[0]
    return {"rss_mb": k.memory_info().rss / 2**20, "uss_mb": k.memory_full_info().uss / 2**20}


def llm_probe(model: str, prompts: list[dict], limiter: Limiter, ctx: int = 1024, predict: int = 80) -> dict[str, Any]:
    c = OllamaClient()
    opts = OllamaOptions(model, ctx, predict, len(limiter.llm_cpus))
    t0 = time.perf_counter()
    c.warm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Hello"}], opts)
    load_s = time.perf_counter() - t0
    ft, tps, ok = [], [], []
    for p in prompts:
        reply = "".join(c.stream([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["text"]}], options=opts))
        ft.append(c.stats.first_token_s * 1000)
        tps.append(c.stats.tokens_per_s)
        ok.append(any(k.lower() in reply.lower() for k in p["any"]))
    limiter.enforce()
    return {"model": model, "load_plus_first_s": load_s, "first_token_ms_p50": float(np.median(ft)), "tok_s_mean": float(np.mean(tps)),
            "quality": float(np.mean(ok)), "n": len(prompts), **runner_rss_mb()}


def asr_probe() -> dict[str, Any]:
    from pv.asr.moonshine_asr import MoonshineASR
    from pv.asr.whisper_asr import WhisperASR
    items = [m for m in load_manifest("main", "en")][:30]
    audio = {m["id"]: load_wav16k(TESTSET / f"{m['id']}.wav") for m in items}
    dur = sum(len(a) for a in audio.values()) / 16000
    out: dict[str, Any] = {}
    for name, factory in {"whisper-base.en-int8": lambda: WhisperASR("base.en", 2, "int8"),
                          "whisper-base.en-fp32": lambda: WhisperASR("base.en", 2, "float32"),
                          "moonshine-tiny": lambda: MoonshineASR("tiny"), "moonshine-small": lambda: MoonshineASR("small")}.items():
        a = factory()
        a.warmup()
        t0, w = time.perf_counter(), []
        for m in items:
            a.start_stream()
            x = audio[m["id"]]
            for i in range(0, len(x) - 512, 512):
                a.feed(x[i:i + 512])
            w.append(wer(m["text"], a.finalize().text))
        proc = time.perf_counter() - t0
        out[name] = {"wer": float(np.mean(w)), "rtf": proc / dur, "rss_mb": psutil.Process().memory_info().rss / 2**20,
                     "note": "RTF = compute time / audio duration, audio fed without real-time pacing"}
        del a
    return out


def tts_probe() -> dict[str, Any]:
    from pv.audio.playback import resample
    from pv.asr.whisper_asr import WhisperASR
    from pv.tts.piper_tts import PiperTTS
    sentences = [m["text"] for m in load_manifest("main", "en") if m["kind"] == "open"][:10]
    w = WhisperASR("base.en", 2, "int8")
    out = {}
    for voice in ("en_US-lessac-medium", "en_US-lessac-low"):
        t = PiperTTS(voice, threads=2)
        t.warmup()
        synth_s, secs, errs = 0.0, 0.0, []
        for s_ in sentences:
            t0 = time.perf_counter()
            pcm = t.synth(s_, do_normalize=False)
            synth_s += time.perf_counter() - t0                    # synthesis time only (Whisper scoring excluded)
            secs += len(pcm) / t.sample_rate
            errs.append(wer(s_, w.transcribe(resample(pcm, t.sample_rate, 16000))[0]))
        out[voice] = {"rtf": synth_s / secs, "whisper_wer_intelligibility": float(np.mean(errs)),
                      "subjective_rating": "not measured: needs human listeners (docs/LIVE_TEST.md)"}
    return out


def main() -> None:
    cfgmod.config["paths"]["data"] = "bench/out/tmpdata"
    netguard.install()
    open_prompts = [m for m in load_manifest("main", "en") if m["kind"] == "open"]
    res: dict[str, Any] = {"llm": {}, "kv": {}, "offload": {}}

    restart_ollama()
    lim = Limiter(4, 4096, split=False)
    have = {l.split()[0] for l in subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout.splitlines()[1:]}
    for label, tag in LLMS.items():
        if tag not in have:
            res["llm"][label] = {"missing": f"{tag} not pulled"}
            continue
        res["llm"][label] = llm_probe(tag, open_prompts, lim)
        print(label, res["llm"][label], flush=True)
        OllamaClient().unload(tag)

    for kv, fa in (("q8_0", "1"), ("f16", "1"), ("f16", "0")):
        restart_ollama(kv, fa)
        lim.enforce()
        key = f"kv={kv},flash_attn={fa}"
        res["kv"][key] = llm_probe("qwen2.5:1.5b-instruct", open_prompts[:10], lim)
        print(key, res["kv"][key], flush=True)
    restart_ollama()
    lim.enforce()

    # offload evidence: load/unload timing, mapped-weights RSS vs private memory, pinning on/off
    c = OllamaClient()
    c.unload("qwen2.5:1.5b-instruct")
    t0 = time.perf_counter()
    c.warm([{"role": "user", "content": "hi"}], OllamaOptions("qwen2.5:1.5b-instruct", 1024, 1, 2))
    res["offload"]["cold_load_s"] = time.perf_counter() - t0
    res["offload"]["resident"] = runner_rss_mb()
    t0 = time.perf_counter()
    c.unload("qwen2.5:1.5b-instruct")
    res["offload"]["unload_s"] = time.perf_counter() - t0
    time.sleep(1)
    res["offload"]["models_loaded_after_unload"] = len(c.ps())
    pin = llm_probe("qwen2.5:1.5b-instruct", open_prompts[:6], lim)
    runner = [k for k in psutil.Process(ollama_pid()).children(recursive=True) if "llama" in k.name().lower()][0]
    runner.cpu_affinity(list(range(psutil.cpu_count())))
    unpinned = llm_probe("qwen2.5:1.5b-instruct", open_prompts[:6], Limiter(12, 4096, split=False))
    res["offload"]["pinning"] = {"pinned_4_cores_tok_s": pin["tok_s_mean"], "unpinned_all_12_threads_tok_s": unpinned["tok_s_mean"],
                                 "note": "pinning is for the simulated-device limit, not a speed-up"}

    res["asr"] = asr_probe()
    res["tts"] = tts_probe()
    (OUT / "quant_sweep.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["asr"], indent=1))
    print(json.dumps(res["tts"], indent=1))


if __name__ == "__main__":
    main()
