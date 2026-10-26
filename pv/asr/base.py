"""ASR interface shared by Moonshine (streaming) and faster-whisper (whole utterance)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Partial:
    text: str = ""


@dataclass
class Final:
    text: str
    lang: str = "en"
    confidence: float = 1.0
    ms: float = 0.0           # time spent inside finalize()


class ASR(ABC):
    @abstractmethod
    def start_stream(self, lang_hint: str | None = None) -> None: ...

    @abstractmethod
    def feed(self, pcm16: np.ndarray) -> Partial: ...

    @abstractmethod
    def finalize(self) -> Final: ...

    def warmup(self) -> None:
        """Run one second of silence through the model so first-use costs are paid at startup."""
        self.start_stream()
        self.feed(np.zeros(16000, dtype=np.int16))
        self.finalize()
