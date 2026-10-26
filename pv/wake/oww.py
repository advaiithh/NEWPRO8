"""openWakeWord (ONNX, CPU) on the shared 16 kHz frame stream. Push-to-talk and always-listening are handled in the pipeline."""
from __future__ import annotations

import numpy as np
from openwakeword.model import Model

from pv.config import config, model

CHUNK = 1280        # openWakeWord's native 80 ms step


class WakeWord:
    def __init__(self, phrase_model: str | None = None, threshold: float | None = None) -> None:
        name = phrase_model or config["wake"]["model"]
        self.threshold = threshold if threshold is not None else config["wake"]["threshold"]
        self.model = Model(
            wakeword_models=[str(model("wakeword", f"{name}.onnx"))],
            inference_framework="onnx",
            melspec_model_path=str(model("wakeword", "melspectrogram.onnx")),
            embedding_model_path=str(model("wakeword", "embedding_model.onnx")),
            device="cpu",
        )
        self.label = next(iter(self.model.models))
        self.buf = np.zeros(0, dtype=np.int16)
        self.last_score = 0.0

    def reset(self) -> None:
        self.model.reset()
        self.buf = np.zeros(0, dtype=np.int16)

    def process(self, frame: np.ndarray) -> bool:
        """Feed a frame of any size; True when the wake phrase fired in the completed 80 ms steps."""
        self.buf = np.concatenate([self.buf, frame])
        fired = False
        while len(self.buf) >= CHUNK:
            scores = self.model.predict(self.buf[:CHUNK])
            self.buf = self.buf[CHUNK:]
            self.last_score = float(scores[self.label])
            fired = fired or self.last_score >= self.threshold
        return fired
