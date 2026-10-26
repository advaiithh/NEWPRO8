"""PocketVoice CLI. Live microphone by default; --wav for loopback tests; --no-audio-out writes the reply WAV.

  python -m pv.app                         hands-free: just talk, it answers only when you speak (no key, no wake word)
  python -m pv.app --mode wake             wake word "hey jarvis" (SPACE = push-to-talk also works)
  python -m pv.app --mode ptt              only SPACE starts a turn
  python -m pv.app --wav q.wav --no-audio-out --out reply.wav
Keys while running:  SPACE push-to-talk (wake/ptt modes)   L switch language   T tighten to 2 cores / 1536 MB   R restore declared limit   Q quit
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
import sys
import threading
import time
import webbrowser
from pathlib import Path

from pv import config as cfgmod
from pv.config import config, path


def key_loop(pipe, rt, declared: tuple[int, int], stop: threading.Event) -> None:
    import msvcrt
    while not stop.is_set():
        if not msvcrt.kbhit():
            time.sleep(0.05)
            continue
        k = msvcrt.getwch().lower()
        if k == " ":
            pipe.trigger_wake()
            print("[push-to-talk] listening...", flush=True)
        elif k == "l":
            order = ["en", "ta", "hi"]
            pipe.set_language(order[(order.index(pipe.language) + 1) % 3])
            print(f"[language] now {pipe.language}", flush=True)
        elif k == "t":
            rt.limiter.tighten(2, 1536)
            print("[limit] tightened to 2 cores / 1536 MB", flush=True)
        elif k == "r":
            rt.limiter.tighten(*declared)
            print(f"[limit] restored to {declared[0]} cores / {declared[1]} MB", flush=True)
        elif k in ("q", "\x1b"):
            stop.set()


def on_turn_printer(tl) -> None:
    if tl.path in ("ignored", "empty"):
        return                                    # noise or a junk transcript: stay silent, print nothing
    lat = f"{tl.latency_ms:.0f} ms" if tl.latency_ms is not None else "n/a"
    names = {"speech_end->asr_final": "endpoint+ASR", "asr_final->route_decision": "route", "route_decision->llm_first_token": "LLM-1st-token",
             "llm_first_token->tts_first_chunk_ready": "TTS", "route_decision->tts_first_chunk_ready": "TTS"}
    parts = " ".join(f"{names[k]} {v:.0f}" for k, v in tl.stages_ms().items() if k in names and v >= 1)
    print(f"[{tl.path:8}] T{tl.tier} {lat:>8}  you: {tl.transcript!r}  ->  {tl.reply[:90]!r}\n           ({parts}; hangover {tl.hangover_ms:.0f} ms)", flush=True)
    tl.log(path("data", "timelines", "live.jsonl"))


def main() -> None:
    ap = argparse.ArgumentParser(description="PocketVoice: offline, CPU-only voice assistant")
    ap.add_argument("--profile", choices=["eco", "pi_like", "roomy", "quality", "balanced", "default"], default=None)
    ap.add_argument("--mode", choices=["wake", "ptt", "always"], default="always",
                    help="always = hands-free (default); wake = say hey jarvis; ptt = SPACE only")
    ap.add_argument("--barge-in", action="store_true", help="let you interrupt while it speaks (use headphones)")
    ap.add_argument("--cache-tau", type=float, default=0.85, help="cache strictness for live use (benchmarks used 0.75)")
    ap.add_argument("--wav", type=Path, help="loopback: feed this WAV instead of the microphone")
    ap.add_argument("--no-audio-out", action="store_true", help="render the reply to --out instead of the speakers")
    ap.add_argument("--out", type=Path, default=path("data", "reply.wav"))
    ap.add_argument("--no-dash", action="store_true")
    ap.add_argument("--open-dash", action="store_true", help="open the dashboard in the browser")
    ap.add_argument("--no-governor", action="store_true")
    ap.add_argument("--language", choices=["en", "ta", "hi"], default="en", help="start in this language (say 'speak Tamil' or press L to switch)")
    ap.add_argument("--asr-final", choices=["none", "base.en", "small.en"], default=None,
                    help="Whisper re-transcribes each finished utterance (more accurate, slower); default follows the profile")
    ap.add_argument("--no-save-audio", action="store_true", help="do not keep your utterances in data/live_wavs")
    ap.add_argument("--cores", type=int)
    ap.add_argument("--ram", type=int)
    ap.add_argument("--input-device", type=int)
    ap.add_argument("--output-device", type=int)
    args = ap.parse_args()
    if args.profile:
        cfgmod.reload(args.profile)

    from pv.audio.io import MicSource
    from pv.audio.playback import DevicePlayer, FilePlayer
    from pv.governor.governor import Governor
    from pv.governor.tiers import build as build_tiers
    from pv.guard import gpucheck, netguard
    from pv.pipeline import Features, Pipeline
    from pv.runtime import Runtime

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # replies may contain Devanagari / symbols
    from pv.guard.limiter import commit_free_mb
    order = ["quality", "balanced", "default"]                     # most to least demanding
    while args.profile in order[:-1]:
        need = (config.get("governor", {}).get("footprint_mb") or [3000])[0]
        if commit_free_mb() >= need + 1200:
            break
        lower = order[order.index(args.profile) + 1]
        print(f"[memory] only {commit_free_mb():.0f} MB free, profile '{args.profile}' needs ~{need} MB: "
              f"switching to '{lower}'. Close Chrome/Edge to use the better model.", flush=True)
        args.profile = lower
        cfgmod.reload(args.profile)
    config["router"]["tau_hit"], config["router"]["margin"] = args.cache_tau, 0.05
    asr_final = args.asr_final or config["asr"].get("final", "none")
    rt = Runtime(args.cores, args.ram, split=False)          # speech recognition and the LLM take turns, so they share all cores
    declared = (rt.limiter.cores, rt.limiter.ram_mb)
    player = FilePlayer(args.out) if args.no_audio_out or args.wav else DevicePlayer(args.output_device)
    holder: dict[str, Pipeline] = {}
    gov = None if args.no_governor else Governor(build_tiers(), rt.limiter, rt.sampler,
                                                 lambda i: holder["p"].set_tier(i),
                                                 on_transition=lambda e: print(f"[governor] T{e['from']} -> T{e['to']} ({e['reason']})", flush=True))
    pipe = Pipeline(rt, player, replace(Features.full(), barge_in=args.barge_in, accurate_asr="" if asr_final == "none" else asr_final), tier=gov.tier_id if gov else 0, mode=args.mode,
                    on_turn=on_turn_printer)
    holder["p"] = pipe
    pipe.strict = False
    if args.language != "en":
        pipe.set_language(args.language)
    if not args.no_save_audio and not args.wav:
        pipe.save_dir = path("data", "live_wavs")
    if gov:
        pipe.governor = gov
        gov.start()
        print(f"[governor] starting at tier T{gov.tier_id} ({pipe.tiers[gov.tier_id].name}) for this limit")
    print("[proof]", json.dumps(gpucheck.proof(config["ollama"]["url"])))

    if not args.no_dash:
        from dash.server import start
        start(pipe, rt, gov)
        if args.open_dash:
            webbrowser.open(f"http://{config['dash']['host']}:{config['dash']['port']}")

    if args.wav:
        tl = pipe.run_wav(args.wav)
        on_turn_printer(tl)
        player.join()
        print("reply wav:", args.out, "| external network attempts:", netguard.status()["attempts"])
        rt.stop()
        return

    stop = threading.Event()
    threading.Thread(target=key_loop, args=(pipe, rt, declared, stop), daemon=True).start()
    src = MicSource(args.input_device)
    hint = {"always": "Just speak - it answers only when you talk and stays silent otherwise.",
            "wake": "Say 'hey jarvis' (or press SPACE), then ask.", "ptt": "Press SPACE, then ask."}[args.mode]
    print(f"Listening ({args.mode} mode). {hint}  T = tighten limit, R = restore, Q = quit.")
    try:
        for t, frame in src.frames():
            if stop.is_set():
                break
            pipe.process_frame(t, frame)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.close()
        rt.stop()
        print("bye. external network attempts:", netguard.status()["attempts"])


if __name__ == "__main__":
    sys.exit(main())
