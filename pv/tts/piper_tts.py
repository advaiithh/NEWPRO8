"""Piper TTS on ONNX Runtime CPU. Voices stay loaded; sessions are created through the CPU-only gate."""
from __future__ import annotations

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig

from pv.config import config, model
from pv.guard.gpucheck import ort_session
from pv.tts.normalizer import normalize


class PiperTTS:
    def __init__(self, voice: str, threads: int = 1, length_scale: float | None = None) -> None:
        path = model("tts", f"{voice}.onnx")
        self.name = voice
        self.voice = PiperVoice.load(path, use_cuda=False)
        self.voice.session = ort_session(path, threads)
        self.sample_rate = self.voice.config.sample_rate
        self.length_scale = config["tts"]["length_scale"] if length_scale is None else length_scale

    def synth(self, text: str, speed_scale: float | None = None, do_normalize: bool = True) -> np.ndarray:
        """Return int16 mono PCM at self.sample_rate."""
        text = normalize(text) if do_normalize else text
        cfg = SynthesisConfig(length_scale=speed_scale if speed_scale is not None else self.length_scale)
        parts = [c.audio_int16_array for c in self.voice.synthesize(text, cfg)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)

    def warmup(self) -> None:
        self.synth("Ready.")
