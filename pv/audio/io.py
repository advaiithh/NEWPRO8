"""Audio sources: live microphone (sounddevice) or a WAV file paced in real time. Both yield 512-sample int16 frames."""
from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from pv.vad.silero_onnx import FRAME

SR = 16000


def load_wav16k(path: Path) -> np.ndarray:
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    x = data[:, 0]
    if sr != SR:
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x.astype(np.float32)).astype(np.int16)
    return x


def normalize_gain(pcm16: np.ndarray, target_peak: float = 0.7, max_gain: float = 12.0) -> np.ndarray:
    """Quiet microphones hurt recognition: bring the loud end of the utterance up to ~70% of full scale (max +21 dB)."""
    x = pcm16.astype(np.float32)
    peak = float(np.percentile(np.abs(x), 99.5)) if len(x) else 0.0
    if peak < 50:
        return pcm16
    gain = min(max_gain, target_peak * 32767 / peak)
    return np.clip(x * gain, -32768, 32767).astype(np.int16) if gain > 1.1 else pcm16


class WavSource:
    """Feeds a file as if it were a live microphone: each frame is released when its audio has 'elapsed'.

    realtime=False hands frames out as fast as they are consumed (used by unit tests, never by benchmarks).
    """

    def __init__(self, samples: np.ndarray, lead_s: float = 0.3, tail_s: float = 3.0, realtime: bool = True) -> None:
        pad = lambda s: np.zeros(int(s * SR), dtype=np.int16)
        self.samples = np.concatenate([pad(lead_s), samples, pad(tail_s)])
        self.realtime = realtime
        self.t0 = 0.0

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (audio_clock_time, frame): the instant the frame's last sample was spoken, on the perf_counter clock."""
        self.t0 = time.perf_counter()
        for i in range(0, len(self.samples) - FRAME + 1, FRAME):
            due = self.t0 + (i + FRAME) / SR
            if self.realtime:
                wait = due - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            yield due, self.samples[i:i + FRAME]


class MicSource:
    def __init__(self, device: int | None = None) -> None:
        import sounddevice as sd
        self.q: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(maxsize=400)
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=FRAME, device=device,
                                     callback=self._cb)

    def _cb(self, indata, frames, time_info, status) -> None:
        try:
            self.q.put_nowait((time.perf_counter(), indata[:, 0].copy()))      # stamped at capture, not at dequeue
        except queue.Full:
            pass

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        self.stream.start()
        try:
            while True:
                yield self.q.get()
        finally:
            self.stream.stop()
