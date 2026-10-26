"""faster-whisper (CTranslate2, CPU, int8). Baseline ASR and non-English ASR; whole-utterance decoding."""
from __future__ import annotations

import time

import numpy as np
from faster_whisper import WhisperModel

from pv.asr.base import ASR, Final, Partial
from pv.config import model


# a few words in the target script keep Whisper writing that script (it confuses Hindi/Urdu and Tamil/Malayalam)
PROMPTS = {"hi": "यह हिंदी में एक वाक्य है।", "ta": "இது தமிழில் ஒரு வாக்கியம்."}


class WhisperASR(ASR):
    def __init__(self, name: str = "base.en", threads: int = 2, compute_type: str = "int8",
                 beam_size: int = 5, fast: bool = False) -> None:
        """fast=True is our tuned decode (beam 1, no previous-text conditioning, no timestamps)."""
        self.name, self.beam_size, self.fast = name, beam_size, fast
        self.model = WhisperModel(str(model("asr", f"whisper-{name}")), device="cpu", compute_type=compute_type,
                                  cpu_threads=threads, local_files_only=True)
        self._buf: list[np.ndarray] = []
        self._lang: str | None = None

    def start_stream(self, lang_hint: str | None = None) -> None:
        self._buf, self._lang = [], lang_hint

    def feed(self, pcm16: np.ndarray) -> Partial:
        self._buf.append(pcm16)
        return Partial("")

    def transcribe(self, pcm16: np.ndarray, lang: str | None = None) -> tuple[str, str, float]:
        audio = pcm16.astype(np.float32) / 32768.0
        kw = dict(beam_size=1, vad_filter=False, without_timestamps=True, condition_on_previous_text=False) \
            if self.fast else dict(beam_size=self.beam_size)
        prompt = PROMPTS.get(lang)   # keeps Devanagari output (base confuses Hindi/Urdu script)
        segments, info = self.model.transcribe(audio, language=lang if not self.name.endswith(".en") else None,
                                               initial_prompt=prompt, **kw)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, info.language, float(info.language_probability)

    def language_probs(self, pcm16: np.ndarray) -> dict[str, float]:
        """Whisper's probability for every language on the audio given (callers pass the first ~2 s)."""
        _, _, probs = self.model.detect_language(pcm16.astype(np.float32) / 32768.0)
        return dict(probs)

    def detect_language(self, pcm16: np.ndarray) -> tuple[str, float]:
        """Whisper language detection on the audio given (callers pass only the first ~2 s)."""
        lang, prob, _ = self.model.detect_language(pcm16.astype(np.float32) / 32768.0)
        return lang, float(prob)

    def finalize(self) -> Final:
        t0 = time.perf_counter()
        audio = np.concatenate(self._buf) if self._buf else np.zeros(1600, dtype=np.int16)
        text, lang, conf = self.transcribe(audio, self._lang)
        return Final(text, lang, conf, (time.perf_counter() - t0) * 1000)
