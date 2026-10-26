"""Moonshine streaming ASR (moonshine-voice). Partials flow out while the user is still speaking."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from moonshine_voice.download import download_model_from_info, find_model_info
from moonshine_voice.moonshine_api import ModelArch
from moonshine_voice.transcriber import Transcriber, TranscriptEvent

from pv.asr.base import ASR, Final, Partial
from pv.config import model
from pv.guard import gpucheck  # noqa: F401  (sets CPU-only and Moonshine thread env vars on import)

ARCH = {"tiny": ModelArch.TINY_STREAMING, "small": ModelArch.SMALL_STREAMING}


def model_dir(size: str, lang: str = "en") -> tuple[Path, ModelArch]:
    """Resolve the model folder inside models/asr/moonshine (already downloaded at setup)."""
    info = find_model_info(lang, ARCH[size])
    path, arch = download_model_from_info(info, cache_root=model("asr", "moonshine"))
    return Path(path), arch


class MoonshineASR(ASR):
    def __init__(self, size: str = "small", update_interval: float = 0.3) -> None:
        self.size = size
        path, arch = model_dir(size)
        self.transcriber = Transcriber(path, arch, update_interval=update_interval)
        self.stream = None
        self._lines: dict[int, str] = {}

    def _on_event(self, ev: TranscriptEvent) -> None:
        self._lines[ev.line.line_id] = ev.line.text.strip()

    def _text(self) -> str:
        return " ".join(t for _, t in sorted(self._lines.items()) if t)

    def start_stream(self, lang_hint: str | None = None) -> None:
        if self.stream is not None:
            self.stream.close()
        self._lines = {}
        self.stream = self.transcriber.create_stream()
        self.stream.add_listener(self._on_event)
        self.stream.start()

    def feed(self, pcm16: np.ndarray) -> Partial:
        self.stream.add_audio((pcm16.astype(np.float32) / 32768.0).tolist(), 16000)
        return Partial(self._text())

    def finalize(self) -> Final:
        t0 = time.perf_counter()
        self.stream.stop()
        return Final(self._text(), "en", 1.0, (time.perf_counter() - t0) * 1000)
