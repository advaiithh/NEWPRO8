"""Run one system configuration over the spoken test set and write bench/out/<name>.json.

  python -m bench.run_bench --system baseline --reps 3
  python -m bench.run_bench --system A3 --reps 3          (ablation ladder: A1..A7, FULL)
One configuration per process so peak RSS / CPU numbers are not contaminated by other systems.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from bench.common import OUT, TESTSET, load_manifest, quality_ok, stats, wer
from pv import config as cfgmod
from pv.audio.playback import FilePlayer
from pv.metrics.timeline import Timeline


def commit_headroom_mb() -> float:
    """System-wide free commit (physical RAM + page file). Allocation failures below ~2 GB corrupt measurements."""
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                    ("tp", "ap", "tpf", "apf", "tv", "av", "ae")]
    m = MS()
    m.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.apf / 2**20


def ollama_ready(url: str) -> None:
    """Our CPU-only server dies with the process whose Job Object it joined, so every run makes sure it is up."""
    import subprocess
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/launch_ollama.ps1", "-IfDown"],
                   check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    httpx.get(f"{url}/api/version", timeout=5).raise_for_status()


def idle_probe(system: Any, rt, seconds: float, settle_s: float = 6.0) -> dict[str, float]:
    """Feed silence through the always-on front end (VAD, and wake word when present); measure CPU/RSS/power."""
    from pv.audio.io import FRAME
    silence = np.zeros(FRAME, dtype=np.int16)
    for _ in range(int(settle_s * 16000 / FRAME)):          # thread pools spin-wait briefly after warm-up
        system.process_frame(time.perf_counter(), silence) if hasattr(system, "process_frame") else system.vad.prob(silence)
        time.sleep(FRAME / 16000)
    cpu0, j0, t0 = rt.sampler.cpu_seconds(), rt.energy.joules(), time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        if hasattr(system, "process_frame"):
            system.process_frame(time.perf_counter(), silence)
        else:
            system.vad.prob(silence)
        n += 1
        time.sleep(max(0.0, t0 + n * FRAME / 16000 - time.perf_counter()))
    wall = time.perf_counter() - t0
    return {"idle_cpu_pct": 100 * (rt.sampler.cpu_seconds() - cpu0) / wall, "idle_rss_mb": rt.sampler.rss_mb,
            "idle_watts": (rt.energy.joules() - j0) / wall if rt.energy.available else float("nan")}


def build(name: str, rt, args):
    from dataclasses import replace
    from pv.baseline import Baseline
    from pv.governor.governor import Governor
    from pv.governor.tiers import build as build_tiers
    from pv.pipeline import Features, Pipeline
    if name == "baseline":
        return Baseline(rt, FilePlayer())
    if name == "FIXED":                                   # full feature set, governor off, fixed tier (default T0)
        return Pipeline(rt, FilePlayer(), replace(Features.full(), governor=False), tier=args.tier, mode="ptt")
    feats = Features.full() if name == "FULL" else Features.ladder(name)
    if not feats.governor:
        return Pipeline(rt, FilePlayer(), feats, tier=args.tier, mode="ptt")
    holder: dict = {}
    gov = Governor(build_tiers(), rt.limiter, rt.sampler, lambda i: holder["p"].set_tier(i))
    pipe = Pipeline(rt, FilePlayer(), feats, tier=gov.tier_id, mode="ptt")
    holder["p"], pipe.governor = pipe, gov
    gov.start()
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--subset", default="main", choices=["main", "heldout", "all", "q40"])
    ap.add_argument("--restart-ollama", action="store_true", help="restart our Ollama before applying the limit")
    ap.add_argument("--llm-timeout", type=float, default=None)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--ram", type=int, default=None)
    ap.add_argument("--tier", type=int, default=0)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--limit-n", type=int, default=None, help="only the first N requests (smoke runs)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true", help="ignore the commit-headroom pre-flight check")
    ap.add_argument("--warmup-turns", type=int, default=2)
    ap.add_argument("--idle-s", type=float, default=8.0)
    args = ap.parse_args()

    if args.profile:
        cfgmod.reload(args.profile)
    cfgmod.config["paths"]["data"] = "bench/out/tmpdata"            # isolated notes/timers for the run
    shutil.rmtree(cfgmod.path("bench", "out", "tmpdata"), ignore_errors=True)
    ollama_ready(cfgmod.config["ollama"]["url"])
    head = commit_headroom_mb()
    print(f"system commit headroom: {head:.0f} MB")
    if head < 3000 and not args.force:
        raise SystemExit(f"only {head:.0f} MB of system commit is free: close other apps (or pass --force) so results are not corrupted")

    if args.restart_ollama:
        import subprocess
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/launch_ollama.ps1"],
                       check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from pv.runtime import Runtime
    rt = Runtime(args.cores, args.ram, split=args.system != "baseline")
    name = f"{args.system}{('_' + args.tag) if args.tag else ''}"
    try:
        system = build(args.system, rt, args)
    except Exception as e:                                  # the configuration cannot even start under this limit
        OUT.mkdir(exist_ok=True)
        (OUT / f"{name}.json").write_text(json.dumps({"summary": {"system": args.system, "startup_failed": repr(e),
                                                       "limit": rt.limiter.banner(), "completed": 0, "n_turns": 0}, "turns": []}))
        print("STARTUP FAILED:", repr(e))
        rt.stop()
        return
    if args.llm_timeout and hasattr(system, "llm"):
        system.llm.http.timeout = httpx.Timeout(args.llm_timeout, connect=5.0)
    reqs = load_manifest(args.subset, args.lang)[: args.limit_n]
    idle = idle_probe(system, rt, args.idle_s)

    rng = random.Random(0)
    turns: list[dict[str, Any]] = []
    order = [list(reqs) for _ in range(args.reps + (1 if args.warmup_turns else 0))]
    for o in order:
        rng.shuffle(o)
    seq = ([("warmup", r) for r in order[0][: args.warmup_turns]] if args.warmup_turns else []) + \
          [(f"rep{i}", r) for i in range(args.reps) for r in order[i + (1 if args.warmup_turns else 0)]]
    rt.sampler.reset_peak()
    t_run = time.perf_counter()
    for tag, r in seq:
        try:
            tl: Timeline = system.run_wav(TESTSET / f"{r['id']}.wav")
        except Exception as e:
            tl = Timeline(system=args.system)
            tl.extra["error"] = repr(e)
            tl.utterance = r["id"]
        if tag == "warmup":
            continue
        row = tl.to_dict()
        row['commit_mb'] = rt.sampler.commit_mb
        row['error'] = tl.extra.get('error')
        row.update(req_id=r["id"], kind=r["kind"], rep=tag, ref=r["text"], heldout=r.get("heldout", False),
                   wer=wer(r["text"], tl.transcript) if tl.transcript else 1.0,
                   ok=quality_ok(r, tl.reply, tl.extra.get("intent", "")))
        turns.append(row)
        print(f"{tag} {r['id']:4} commit={rt.sampler.commit_mb:5.0f}MB {row['latency_ms'] or float('nan'):7.0f} ms  {tl.path:8} {tl.transcript[:40]!r}", flush=True)
    wall = time.perf_counter() - t_run

    ok_lat = [t["latency_ms"] for t in turns if t["latency_ms"] is not None]
    scored = [t["ok"] for t in turns if t["kind"] in ("skill", "open") and t["ok"] is not None]
    net_j = [t["energy_j"] - idle["idle_watts"] * (t["stamps"]["turn_end"] - min(t["stamps"].values()))
             for t in turns if t["energy_j"] is not None and t["stamps"]]
    summary = {
        "system": args.system, "limit": rt.limiter.banner() if rt.limiter else "none", "reps": args.reps,
        "n_turns": len(turns), "completed": sum(t["latency_ms"] is not None and not t["error"] for t in turns), "completion_rate": float(np.mean([t["latency_ms"] is not None and not t["error"] for t in turns])), "latency_ms": stats(ok_lat),
        "peak_rss_mb": rt.sampler.peak_mb, "peak_commit_mb": rt.sampler.peak_commit_mb, "cpu_s_per_turn": float(np.mean([t["cpu_s"] for t in turns])),
        "energy_j_per_turn_gross": float(np.mean([t["energy_j"] for t in turns if t["energy_j"] is not None] or [np.nan])),
        "energy_j_per_turn_net_of_idle": float(np.mean(net_j)) if net_j else float("nan"),
        "energy_source": "Windows Energy Meter RAPL package power (whole machine), measured" if rt.energy.available
        else "proxy: cpu_s x assumed W/core",
        "energy_proxy_j_per_turn": float(np.mean([t["extra"].get("energy_proxy_j", 0) for t in turns])),
        "assumed_w_per_core_for_proxy": cfgmod.config["energy"]["assumed_w_per_core"],
        "wer_mean": float(np.mean([t["wer"] for t in turns])),
        "quality_pass_rate": float(np.mean(scored)) if scored else float("nan"), "quality_n": len(scored),
        "paths": {k: sum(t["path"] == k for t in turns) for k in {t["path"] for t in turns}},
        "wall_s": wall, **idle,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps({"summary": summary, "turns": turns}, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    rt.stop()


if __name__ == "__main__":
    main()
