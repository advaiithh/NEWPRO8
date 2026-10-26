"""The streaming voice pipeline: VAD/wake -> streaming ASR -> router -> LLM -> chunked TTS -> playback.

One frame loop (process_frame) drives a small state machine; each reply runs in its own worker thread so the
loop keeps listening (barge-in). Feature flags let the ablation ladder switch each idea on one at a time.
"""
from __future__ import annotations

import collections
import gc
import queue
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from pv.asr.moonshine_asr import MoonshineASR
from pv.audio.io import WavSource, load_wav16k
from pv.asr import langid
from pv.audio.io import normalize_gain
from pv.audio.playback import Player
from pv.config import config, model, path
from pv.governor.tiers import Tier, build as build_tiers
from pv.knowledge.kb import Knowledge
from pv.llm import prompts
from pv.llm.ollama_client import OllamaClient, OllamaOptions
from pv.metrics.timeline import Timeline, now
from pv.router import skills
from pv.router.cache import SemanticCache
from pv.router.router import Route, Router
from pv.runtime import Runtime
from pv.tts.chunker import Chunker
from pv.tts.piper_tts import PiperTTS
from pv.vad.endpoint import Endpointer
from pv.vad.silero_onnx import SileroVAD
from pv.wake.oww import WakeWord

IDLE, LISTEN, BUSY = "idle", "listen", "busy"
PREROLL_FRAMES = 14
BARGE_MS = 250


@dataclass(frozen=True)
class Features:
    hangover_ms: int = 500          # fixed endpointing hangover (A1)
    sentence_stream: bool = False   # D6 (A2)
    voice_prompt: bool = False      # short prompt, ctx/reply caps, tuned TTS + normalizer (A3)
    prefix_warm: bool = False       # D3 (A4)
    adaptive: bool = False          # D2 (A5)
    router: bool = False            # D1 (A6)
    governor: bool = False          # D4 (A7)
    barge_in: bool = False          # D5
    multilingual: bool = False      # D12: Hindi / Tamil ASR + voices, loaded on first use
    auto_language: bool = False     # detect the language of every utterance (needs ~2 s of Whisper-small per turn)
    knowledge: bool = False         # offline Wikipedia: definitions read from the article, facts as LLM context
    accurate_asr: str = ""          # e.g. "small.en": Whisper re-transcribes the finished utterance (slower, more accurate)

    @classmethod
    def ladder(cls, step: str) -> "Features":
        f = cls()
        for name, change in [("A2", dict(sentence_stream=True)), ("A3", dict(voice_prompt=True)),
                             ("A4", dict(prefix_warm=True)), ("A5", dict(adaptive=True)),
                             ("A6", dict(router=True)), ("A7", dict(governor=True))]:
            if step == "A1":
                break
            f = replace(f, **change)
            if step == name:
                break
        return f

    @classmethod
    def full(cls) -> "Features":
        return cls(hangover_ms=350, sentence_stream=True, voice_prompt=True, prefix_warm=True, adaptive=True, router=True,
                   governor=True, barge_in=True, multilingual=True, knowledge=True)


class Pipeline:
    def __init__(self, rt: Runtime, player: Player, features: Features, tier: int = 0, mode: str = "wake",
                 on_turn: Callable[[Timeline], None] | None = None, use_cache: bool = True) -> None:
        self.rt, self.player, self.f, self.mode, self.on_turn = rt, player, features, mode, on_turn
        self.tiers = build_tiers()
        self.tier = self.tiers[tier]
        self.vad = SileroVAD()
        self.wake = WakeWord() if mode == "wake" else None
        self.llm = OllamaClient()
        self.history = prompts.History()
        self.asrs: dict[str, MoonshineASR] = {}
        self.voices: dict[str, PiperTTS] = {}
        self.ctx = skills.SkillContext.create(path(config["paths"]["data"]), on_timer=self.say)
        cache = SemanticCache() if features.router and use_cache else None
        self.kb = Knowledge() if features.knowledge and features.router else None
        self.router = Router(self.ctx, cache, use_skills=features.router, use_cache=features.router and use_cache, kb=self.kb)
        self.ep = Endpointer(base=features.hangover_ms, adaptive=features.adaptive)
        self.state, self.armed_until = IDLE, 0.0
        self.preroll: collections.deque[np.ndarray] = collections.deque(maxlen=PREROLL_FRAMES)
        self.tl: Timeline | None = None
        self.meter = None
        self.abort = threading.Event()
        self.worker: threading.Thread | None = None
        self.turn_done = threading.Event()
        self.voiced_run_ms = 0.0
        self.partial = ""
        self.lang = "en"
        self.speed_scale = 0.0
        self.pending_tier: Tier | None = None
        self.governor = None
        self.mute_until = 0.0
        self.utt: list[np.ndarray] = []
        self.whisper = None
        self.lang_voice: dict[str, str] = {}      # language code -> Piper voice, filled when multilingual
        self.last_voice = ""
        self.save_dir: Path | None = None          # when set, every utterance is saved here as a WAV
        self.whisper_acc = None
        if features.accurate_asr:
            from pv.asr.whisper_asr import WhisperASR
            self.whisper_acc = WhisperASR(features.accurate_asr, threads=max(2, len(rt.limiter.audio_cpus) if rt.limiter else 2), fast=True)
            self.whisper_acc.warmup()
        self.language = "en"                       # sticky: "speak Tamil" / the L key / --language switch it
        if features.multilingual:
            self.lang_voice = {"hi": next(model("tts").glob("hi_IN-*.onnx")).stem, "ta": config["tts"]["voice_ta"]}
            if features.auto_language:
                self.multilingual_asr()
        self.warm_thread: threading.Thread | None = None
        self.warmup()

    # ---- models ---------------------------------------------------------
    def asr(self) -> MoonshineASR:
        size = self.tier.asr if self.f.voice_prompt else "small"
        if size not in self.asrs:
            self.asrs[size] = MoonshineASR(size)
            self.asrs[size].warmup()
        return self.asrs[size]

    def voice(self, name: str | None = None) -> PiperTTS:
        name = name or (self.tier.voice if self.f.voice_prompt else config["tts"]["voice_full"])
        if name not in self.voices:
            threads = max(1, len(self.rt.limiter.audio_cpus)) if self.rt.limiter else 1
            self.voices[name] = PiperTTS(name, threads, length_scale=None if self.f.voice_prompt else 1.0)
            self.voices[name].warmup()
        return self.voices[name]

    def llm_options(self) -> OllamaOptions:
        if not self.f.voice_prompt:
            return OllamaOptions(self.tier.llm or config["ollama"]["full_model"], default_options=True)
        threads = len(self.rt.limiter.llm_cpus) if self.rt.limiter else None
        think = False if (self.tier.llm or "").startswith("qwen3") else None       # no hidden reasoning tokens in a voice loop
        return OllamaOptions(self.tier.llm, self.tier.num_ctx, self.tier.num_predict, threads, think=think)

    def system_prompt(self) -> str:
        if not self.f.voice_prompt:
            return prompts.SYSTEM_BASELINE
        return f"{self.tier.system} Today is {time.strftime('%A, %B')} {time.localtime().tm_mday}, {time.localtime().tm_year}."

    def warmup(self) -> None:
        self.asr()
        self.voice()
        for name in self.lang_voice.values():
            self.voice(name)
        if self.tier.llm:
            self.llm.warm(self.history.messages(self.system_prompt(), 600) + [{"role": "user", "content": "Hello"}],
                          self.llm_options())

    # ---- control --------------------------------------------------------
    def trigger_wake(self, t: float | None = None, window_s: float = 30.0) -> None:
        """Wake word detected or push-to-talk pressed: arm, pre-activate ASR, warm the LLM prefix (D3)."""
        t = now() if t is None else t
        self.armed_until = t + window_s
        if self.state == IDLE:
            self._begin_turn(t)
            self.tl.mark("wake", t)
            self.asr().start_stream()
            if self.f.prefix_warm and self.tier.llm:
                self.warm_thread = threading.Thread(target=self._warm_prefix, daemon=True)
                self.warm_thread.start()

    def _warm_prefix(self) -> None:
        try:
            msgs = self.history.messages(self.system_prompt(), self.tier.num_ctx - self.tier.num_predict)
            self.llm.warm(msgs + [{"role": "user", "content": "Hello"}], self.llm_options())
        except Exception:                                      # warming is an optimisation; shutdown or a busy server is fine
            pass

    def _begin_turn(self, t: float) -> None:
        self.tl = Timeline(system="ours", tier=self.tier.id)
        self.meter = self.rt.meter()
        self.turn_done.clear()
        self.abort.clear()
        self.partial = ""

    def say(self, text: str) -> None:
        """Speak arbitrary text (timer alarms) when not busy."""
        if self.state == BUSY:
            return
        threading.Thread(target=lambda: self.player.play(self.voice().synth(text), self.voice().sample_rate),
                         daemon=True).start()

    # ---- frame loop -----------------------------------------------------
    strict = True          # tests and benchmarks want exceptions; the live app turns this off

    def process_frame(self, t: float, frame: np.ndarray) -> None:
        try:
            self._process_frame(t, frame)
        except Exception as e:                                  # one bad turn (e.g. out of memory) must not end the session
            if self.strict:
                raise
            print(f"[error] turn dropped: {type(e).__name__}: {str(e)[:120]}", flush=True)
            self.ep.reset()
            self.preroll.clear()
            self.state, self.tl = IDLE, None

    def _process_frame(self, t: float, frame: np.ndarray) -> None:
        prob = self.vad.prob(frame)
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        voiced = prob > self.vad.threshold and rms >= config["vad"].get("min_rms", 120)   # energy gate: fans/hum are not speech
        if self.state == BUSY:
            self._barge_check(t, frame, prob)             # half-duplex unless barge-in is enabled: frames are otherwise dropped
            return
        if t < self.mute_until:                           # the tail of our own voice still ringing in the room
            self.preroll.clear()
            return
        self.preroll.append(frame)
        if self.state == IDLE:
            if self.wake and self.wake.process(frame):
                self.trigger_wake(t)
                self.wake.reset()
            armed = self.mode == "always" or t < self.armed_until
            ev = self.ep.update(voiced, t) if armed else None
            if ev and ev.kind == "start":
                if self.tl is None or self.mode == "always":
                    self._begin_turn(t)
                    self.asr().start_stream()
                    if self.f.prefix_warm and self.tier.llm and self.mode == "always":
                        threading.Thread(target=self._warm_prefix, daemon=True).start()
                self._start_listening(ev.t)
        elif self.state == LISTEN:
            p = self.asr().feed(frame)
            self.utt.append(frame)
            self.partial = p.text or self.partial
            english = self.language == "en"            # the English recogniser's partials are noise in other languages
            known = self.f.adaptive and english and self.router.intent_known(self.partial)
            ev = self.ep.update(voiced, t, self.partial if english else "", known)
            if ev and ev.kind == "end":
                self._end_of_speech(ev)

    def _start_listening(self, t_start: float) -> None:
        self.state = LISTEN
        self.tl.mark("speech_start", t_start)
        self.utt = list(self.preroll)
        for fr in self.preroll:
            self.asr().feed(fr)
        self.preroll.clear()

    def multilingual_asr(self):
        """Whisper-small (multilingual), loaded the first time a non-English language is used."""
        if self.whisper is None:
            from pv.asr.whisper_asr import WhisperASR
            self.whisper = WhisperASR(config["asr"].get("multilingual", "small"), threads=max(2, len(self.rt.limiter.audio_cpus)), fast=True)
            self.whisper.warmup()
        return self.whisper

    def set_language(self, code: str) -> None:
        self.language = code
        if code != "en" and code in self.lang_voice:
            self.voice(self.lang_voice[code])
            self.multilingual_asr()

    @property
    def auto_language_on(self) -> bool:
        return self.f.auto_language and self.f.multilingual

    def _resolve_language(self) -> str:
        """Sticky language, or (auto mode) Whisper-small language-ID on the whole utterance (~2 s, accurate)."""
        if not self.auto_language_on:
            return self.language
        return langid.from_audio(self.multilingual_asr(), np.concatenate(self.utt))

    def _end_of_speech(self, ev) -> None:
        tl = self.tl
        tl.mark("speech_end", ev.t)
        tl.hangover_ms = ev.hangover_ms
        final = self.asr().finalize()
        lang = self._resolve_language()
        if lang != "en":
            final.text = self.multilingual_asr().transcribe(np.concatenate(self.utt), lang)[0]
        elif self.whisper_acc is not None:                  # slower but more accurate final transcript
            tl.extra["moonshine_text"] = final.text
            try:
                final.text = self.whisper_acc.transcribe(normalize_gain(np.concatenate(self.utt)))[0] or final.text
            except Exception as e:                          # e.g. out of memory: keep the fast transcript
                tl.extra["whisper_error"] = repr(e)
                print(f"[warn] accurate recogniser failed ({type(e).__name__}); using the fast transcript", flush=True)
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            sf.write(self.save_dir / f"{time.strftime('%H%M%S')}_{int(time.perf_counter() * 1000) % 100000}.wav",
                     np.concatenate(self.utt), 16000)
        tl.mark("asr_final")
        tl.transcript, tl.lang = final.text, lang
        self.state = BUSY
        self.voiced_run_ms = 0.0
        self.worker = threading.Thread(target=self._reply_worker, args=(tl, final.text), daemon=True)
        self.worker.start()

    def _barge_check(self, t: float, frame: np.ndarray, prob: float) -> None:
        if not self.f.barge_in or not (self.worker and self.worker.is_alive()):
            return
        loud = prob > config["vad"].get("barge_threshold", 0.85)
        self.voiced_run_ms = self.voiced_run_ms + 32 if loud else 0.0
        if self.voiced_run_ms >= BARGE_MS:
            self.interrupt()

    def interrupt(self) -> None:
        """D5 barge-in: stop playback, cancel the LLM stream, begin a new turn from the interrupting speech."""
        self.abort.set()
        self.llm.cancel()
        self.player.stop()
        if self.worker:
            self.worker.join(timeout=2)
        self._begin_turn(now())
        self.tl.extra["barge_in"] = True
        self.asr().start_stream()
        self.ep.reset()
        self._start_listening(now())
        self.state = LISTEN

    # ---- reply ------------------------------------------------------------
    def _reply_worker(self, tl: Timeline, text: str) -> None:
        try:
            if self._is_noise(text):
                tl.path = "ignored"
                return
            self.player.reset_turn()
            self.player.on_first_audio = lambda t: tl.mark("first_audio", t)
            if self.f.multilingual and (target := langid.switch_command(text)) and target != self.language:
                self.set_language(target)                       # "speak Tamil": confirm in the new language, no LLM
                tl.lang, tl.path = target, "command"
                tl.reply = self._speak_text(tl, langid.CONFIRM[target])
                self.player.join()
                return
            self._maybe_swap_tier()
            route = self.router.route(text, tl.lang) if self.f.router else Route("llm")
            tl.mark("route_decision")
            tl.path, tl.similarity = route.kind, route.sim
            tl.extra.update(intent=route.intent, skill=route.skill, margin=route.margin, kb=route.kb_title)
            reply = self._speak_route(tl, route, text)
            tl.reply = reply
            self.ctx.last_reply = reply
            for kind, delta in self.ctx.actions:
                self._apply_action(kind, delta)
            self.ctx.actions.clear()
            if not self.abort.is_set():
                self.history.add(text, reply)
            self.player.join()
        except Exception as e:                                   # keep the assistant alive; record the failure
            tl.extra["error"] = repr(e)
        finally:
            self._finish_turn(tl)

    JUNK = {"you", "the", "uh", "um", "hmm", "mm", "huh", "oh", "ah", "a", "i", "so", "and", "okay", "yeah"}

    @classmethod
    def _is_noise(cls, text: str) -> bool:
        """Empty or typical recogniser hallucinations on noise: never answered."""
        words = [w for w in text.lower().replace(".", " ").replace(",", " ").replace("?", " ").split() if w]
        return not words or (len(words) <= 2 and all(w in cls.JUNK for w in words))

    def _apply_action(self, kind: str, delta: float) -> None:
        if kind == "volume":
            self.player.volume = float(np.clip(self.player.volume + delta, 0.2, 2.0))
        elif kind == "speed":
            self.speed_scale = float(np.clip(self.speed_scale + delta, -0.3, 0.5))

    def _speak_route(self, tl: Timeline, route: Route, text: str) -> str:
        if route.kind == "cache" and route.audio is not None:
            tl.mark("tts_first_chunk_ready")
            self.player.play(*route.audio)
            return route.reply
        if route.kind in ("skill", "cache", "kb"):
            return self._speak_text(tl, route.reply)
        if self.tier.llm is None or (self.tier.llm_when_idle_only and self.rt.sampler.cpu_percent > 70 * self.rt.limiter.cores):
            tl.path = "survival"
            return self._speak_text(tl, prompts.SURVIVAL_REPLY)
        return self._llm_reply(tl, text, route.context)

    def _synth(self, text: str, lang: str = "en") -> tuple[np.ndarray, int]:
        v = self.voice(self.lang_voice.get(lang))
        self.last_voice = v.name
        scale = (config["tts"]["length_scale"] if self.f.voice_prompt else 1.0) + self.speed_scale
        return v.synth(text, scale, do_normalize=self.f.voice_prompt and lang == "en"), v.sample_rate

    def _speak_text(self, tl: Timeline, text: str) -> str:
        pcm, sr = self._synth(text, tl.lang)
        tl.mark("tts_first_chunk_ready")
        self.player.play(pcm, sr)
        return text

    def _llm_reply(self, tl: Timeline, text: str, context: str = "") -> str:
        budget = self.tier.num_ctx - self.tier.num_predict if self.f.voice_prompt else 3000
        system = prompts.SYSTEM_BY_LANG.get(tl.lang, "") if tl.lang != "en" and self.f.voice_prompt else self.system_prompt()
        system = system or self.system_prompt()
        asof = self.kb.asof if self.kb else "an older snapshot"
        ask = (f"Facts (Wikipedia, as of {asof}): {context}\nQuestion: {text}\nAnswer briefly. Trust the facts over your own memory when they are relevant."
               if context else text)
        msgs = self.history.messages(system, budget) + [{"role": "user", "content": ask}]
        t0 = now()
        chunker, reply = Chunker(), []
        q: queue.Queue[str | None] = queue.Queue(maxsize=2)
        worker = threading.Thread(target=self._tts_worker, args=(q, tl), daemon=True)
        stream = self.f.sentence_stream
        if stream:
            worker.start()
        for tok in self.llm.stream(msgs, options=self.llm_options()):
            if self.abort.is_set():
                break
            if "llm_first_token" not in tl.stamps:
                tl.mark("llm_first_token")
            reply.append(tok)
            if stream:
                for c in chunker.push(tok):
                    q.put(c)
        capped = self.f.voice_prompt and self.llm.stats.tokens >= self.tier.num_predict
        if stream:
            if not self.abort.is_set():
                spoke = chunker.emitted > 0
                for c in chunker.flush():
                    if capped and spoke and not c.rstrip().endswith((".", "!", "?")):
                        reply[:] = ["".join(reply).rsplit(c, 1)[0]]       # drop the cut-off fragment from the reply too
                        break
                    q.put(c)
            q.put(None)
            worker.join()
        elif not self.abort.is_set():
            pcm, sr = self._synth("".join(reply), tl.lang)
            tl.mark("tts_first_chunk_ready")
            self.player.play(pcm, sr)
        tl.tokens, tl.tokens_per_s = self.llm.stats.tokens, self.llm.stats.tokens_per_s
        tl.extra["llm_s"] = now() - t0
        return "".join(reply).strip()

    def _tts_worker(self, q: "queue.Queue[str | None]", tl: Timeline) -> None:
        """Two-slot pipeline: synthesise chunk k+1 while chunk k plays."""
        while (chunk := q.get()) is not None:
            if self.abort.is_set():
                continue
            pcm, sr = self._synth(chunk, tl.lang)
            tl.mark("tts_first_chunk_ready")
            self.player.play(pcm, sr)

    def _finish_turn(self, tl: Timeline) -> None:
        tl.mark("turn_end")
        if self.meter:
            m = self.meter.stop()
            tl.cpu_s, tl.energy_j, tl.rss_peak_mb = m["cpu_s"], m["energy_j"], m["rss_peak_mb"]
            tl.extra["energy_proxy_j"] = m["energy_proxy_j"]
        if not self.abort.is_set():
            self.ep.reset()
            self.mute_until = now() + (0.0 if tl.path in ("ignored", "empty") else 0.7)
            self.state = IDLE
            self.armed_until = now() + 8.0
            self.tl = None
            if self.governor:
                self.governor.observe_turn(tl.latency_ms, tl.tokens_per_s, tl.path == "llm")
            if self.on_turn:
                self.on_turn(tl)
            self.turn_done.set()

    def _maybe_swap_tier(self) -> None:
        if self.pending_tier is not None:
            new, self.pending_tier = self.pending_tier, None
            self._apply_tier(new)

    def set_tier(self, tier_id: int) -> None:
        """Immediate when idle, otherwise at the start of the next reply (never mid-sentence)."""
        new = self.tiers[tier_id]
        if self.state == IDLE:
            self._apply_tier(new)
        else:
            self.pending_tier = new

    def _apply_tier(self, new: Tier) -> None:
        """Swap models for the new tier and free what the old one needed (RAM is the scarce resource)."""
        old, self.tier = self.tier, new
        if old.llm and old.llm != new.llm:
            self.llm.unload(old.llm)
        if old.asr != new.asr:
            self.asrs.pop(old.asr, None)
        for name in [v for v in self.voices if v != new.voice and v not in self.lang_voice.values()]:
            del self.voices[name]
        gc.collect()
        self.asr()
        self.voice()
        if new.llm and not new.llm_when_idle_only:
            threading.Thread(target=self._preload_llm, daemon=True).start()

    def _preload_llm(self) -> None:
        self.llm.warm(self.history.messages(self.system_prompt(), 600) + [{"role": "user", "content": "Hello"}],
                      self.llm_options())

    # ---- driving ------------------------------------------------------------
    def run_wav(self, wav, realtime: bool = True) -> Timeline:
        """Loopback test: play a WAV into the pipeline as if spoken after a push-to-talk press."""
        done: list[Timeline] = []
        self.on_turn = done.append
        src = WavSource(load_wav16k(wav), realtime=realtime)
        self.state, self.tl = IDLE, None
        self.mute_until = 0.0
        self.turn_done.clear()
        self.ep.reset()
        self.vad.reset()
        woke = False
        for t, frame in src.frames():
            if not woke and t - src.t0 >= 0.15:
                woke = True
                self.trigger_wake(t)
            self.process_frame(t, frame)
            if done:
                break
        else:
            deadline = time.perf_counter() + 60
            while not done and time.perf_counter() < deadline:
                time.sleep(0.02)
        tl = done[-1] if done else Timeline(system="ours")
        tl.utterance = getattr(wav, "name", str(wav))
        return tl

    def run(self, source) -> None:
        for t, frame in source.frames():
            self.process_frame(t, frame)

    def close(self) -> None:
        self.ctx.timers.stop()
        self.llm.close()
        self.asrs.clear()
        self.voices.clear()
        self.whisper = None
        gc.collect()
