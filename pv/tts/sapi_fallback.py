"""Starved-tier voice: Windows SAPI via pyttsx3 (offline, no model files, near-zero RAM)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pyttsx3
import soundfile as sf

from pv.tts.normalizer import normalize


class SapiTTS:
    name = "sapi"

    def __init__(self, rate: int = 190) -> None:
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.sample_rate = 22050

    def synth(self, text: str, speed_scale: float | None = None, do_normalize: bool = True) -> np.ndarray:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "s.wav"
            self.engine.save_to_file(normalize(text) if do_normalize else text, str(out))
            self.engine.runAndWait()
            data, sr = sf.read(out, dtype="int16")
        self.sample_rate = sr
        return data if data.ndim == 1 else data[:, 0]
