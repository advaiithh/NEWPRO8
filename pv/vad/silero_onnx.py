"""Silero VAD (v5) on ONNX Runtime CPU, no torch. 16 kHz, 512-sample (32 ms) frames, 64-sample context."""
from __future__ import annotations

import numpy as np

from pv.config import config, model
from pv.guard.gpucheck import ort_session

FRAME = 512
CONTEXT = 64


class SileroVAD:
    def __init__(self, threads: int = 1) -> None:
        self.session = ort_session(model("vad", "silero_vad.onnx"), threads)
        self.sr = np.array(16000, dtype=np.int64)
        self.threshold = config["vad"]["threshold"]
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, CONTEXT), dtype=np.float32)

    def prob(self, frame_i16: np.ndarray) -> float:
        """Speech probability for one 512-sample int16 frame."""
        x = frame_i16.astype(np.float32) / 32768.0
        inp = np.concatenate([self.context, x[None, :]], axis=1)
        out, self.state = self.session.run(None, {"input": inp, "state": self.state, "sr": self.sr})
        self.context = inp[:, -CONTEXT:]
        return float(out[0, 0])
