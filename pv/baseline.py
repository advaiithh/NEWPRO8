"""The baseline (SPEC section 8): fixed 800 ms silence -> faster-whisper base.en (default beam) on the whole
utterance -> Ollama qwen2.5:1.5b with default options, full reply -> Piper lessac-medium on the full reply -> play.
Sequential. No router, no governor, no barge-in, no warm-up tricks.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pv.asr.whisper_asr import WhisperASR
from pv.audio.io import MicSource, WavSource, load_wav16k
from pv.audio.playback import DevicePlayer, FilePlayer, Player
from pv.config import config, path
from pv.llm.ollama_client import OllamaClient, OllamaOptions
from pv.llm.prompts import SYSTEM_BASELINE
from pv.metrics.timeline import Timeline, now
from pv.runtime import Runtime
from pv.tts.piper_tts import PiperTTS
from pv.vad.endpoint import Endpointer
from pv.vad.silero_onnx import SileroVAD

HANGOVER_MS = 800


class Baseline:
    name = "baseline"

    def __init__(self, rt: Runtime, player: Player) -> None:
        self.rt, self.player = rt, player
        self.vad = SileroVAD()
        self.asr = WhisperASR(config["asr"]["baseline"], threads=0, beam_size=5)
        self.tts = PiperTTS(config["tts"]["voice_full"], threads=0, length_scale=1.0)
        self.llm = OllamaClient()
        self.opts = OllamaOptions(config["ollama"]["full_model"], default_options=True)
        self.asr.warmup()
        self.tts.synth("Ready.", do_normalize=False)
        self.llm.warm([{"role": "system", "content": SYSTEM_BASELINE}, {"role": "user", "content": "Hello"}], self.opts)

    def listen(self, source, tl: Timeline) -> np.ndarray:
        """Collect audio until the fixed hangover elapses after the last voiced frame."""
        ep = Endpointer(adaptive=False, base=HANGOVER_MS)
        self.vad.reset()
        frames: list[np.ndarray] = []
        for t, frame in source.frames():
            frames.append(frame)
            voiced = self.vad.prob(frame) > self.vad.threshold
            ev = ep.update(voiced, t)
            if ev and ev.kind == "start":
                tl.mark("speech_start", ev.t)
            if ev and ev.kind == "end":
                tl.mark("speech_end", ev.t)
                tl.hangover_ms = ev.hangover_ms
                break
        return np.concatenate(frames)

    def respond(self, audio: np.ndarray, tl: Timeline) -> None:
        self.asr.start_stream()
        self.asr.feed(audio)
        final = self.asr.finalize()
        tl.transcript = final.text
        tl.mark("asr_final")
        tl.mark("route_decision")
        tl.path = "baseline"
        messages = [{"role": "system", "content": SYSTEM_BASELINE}, {"role": "user", "content": final.text}]
        t0 = now()
        reply = "".join(self.llm.stream(messages, options=self.opts))
        tl.mark("llm_first_token", t0 + (self.llm.stats.first_token_s or 0))
        tl.reply, tl.tokens, tl.tokens_per_s = reply, self.llm.stats.tokens, self.llm.stats.tokens_per_s
        pcm = self.tts.synth(reply, do_normalize=False)
        tl.mark("tts_first_chunk_ready")
        self.player.reset_turn()
        self.player.play(pcm, self.tts.sample_rate)
        tl.mark("first_audio", self.player.first_audio_t)

    def run(self, source) -> Timeline:
        tl = Timeline(system=self.name)
        meter = self.rt.meter()
        audio = self.listen(source, tl)
        if "speech_end" not in tl.stamps:
            return tl
        self.respond(audio, tl)
        self.player.join()
        tl.mark("turn_end")
        m = meter.stop()
        tl.cpu_s, tl.energy_j, tl.rss_peak_mb = m["cpu_s"], m["energy_j"], m["rss_peak_mb"]
        tl.extra["energy_proxy_j"] = m["energy_proxy_j"]
        return tl

    def run_wav(self, wav: Path, realtime: bool = True) -> Timeline:
        tl = self.run(WavSource(load_wav16k(wav), realtime=realtime))
        tl.utterance = wav.name
        return tl


def main() -> None:
    ap = argparse.ArgumentParser(description="PocketVoice baseline loop")
    ap.add_argument("--wav", type=Path, help="input WAV instead of the microphone")
    ap.add_argument("--no-audio-out", action="store_true", help="write the reply to --out instead of speakers")
    ap.add_argument("--out", type=Path, default=path("data", "reply_baseline.wav"))
    ap.add_argument("--no-limit", action="store_true")
    args = ap.parse_args()
    rt = Runtime(limit=not args.no_limit, split=False)
    player = FilePlayer(args.out) if args.no_audio_out or args.wav else DevicePlayer()
    b = Baseline(rt, player)
    tl = b.run_wav(args.wav) if args.wav else b.run(MicSource())
    tl.log(path("data", "timelines", "baseline.jsonl"))
    print(f"transcript: {tl.transcript!r}\nreply: {tl.reply!r}")
    print(f"latency_ms={tl.latency_ms:.0f}  stages={ {k: round(v) for k, v in tl.stages_ms().items()} }")
    if args.no_audio_out or args.wav:
        print("reply wav:", args.out)
    rt.stop()


if __name__ == "__main__":
    main()
