"""Interruptible playback. DevicePlayer streams to the speakers; FilePlayer renders to a WAV (--no-audio-out).

T_first_audio is stamped when the first real sample is handed to the output (device callback / file writer).
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

OUT_RATE = 22050


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x
    n = int(len(x) * sr_out / sr_in)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x.astype(np.float32)).astype(np.int16)


class Player:
    """Common state: first-audio stamp, volume, playing flag."""

    def __init__(self) -> None:
        self.first_audio_t: float | None = None
        self.volume = 1.0
        self.on_first_audio = None

    def reset_turn(self) -> None:
        self.first_audio_t = None

    def _stamp(self) -> None:
        if self.first_audio_t is None:
            self.first_audio_t = time.perf_counter()
            if self.on_first_audio:
                self.on_first_audio(self.first_audio_t)


class DevicePlayer(Player):
    def __init__(self, device: int | None = None) -> None:
        super().__init__()
        import sounddevice as sd
        self.q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self.buf = np.zeros(0, dtype=np.int16)
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(samplerate=OUT_RATE, channels=1, dtype="int16", latency="low",
                                      device=device, callback=self._cb)
        self.stream.start()

    def _cb(self, outdata, frames, time_info, status) -> None:
        with self.lock:
            while len(self.buf) < frames:
                try:
                    self.buf = np.concatenate([self.buf, self.q.get_nowait()])
                except queue.Empty:
                    break
            n = min(frames, len(self.buf))
            outdata[:n, 0] = (self.buf[:n] * self.volume).astype(np.int16)
            outdata[n:, 0] = 0
            self.buf = self.buf[n:]
        if n:
            self._stamp()

    def play(self, pcm16: np.ndarray, sr: int) -> None:
        self.q.put(resample(pcm16, sr, OUT_RATE))

    @property
    def playing(self) -> bool:
        return not self.q.empty() or len(self.buf) > 0

    def stop(self) -> None:
        """Barge-in: drop everything queued."""
        with self.lock:
            self.buf = np.zeros(0, dtype=np.int16)
            while not self.q.empty():
                self.q.get_nowait()

    def join(self, timeout: float = 60) -> None:
        t0 = time.time()
        while self.playing and time.time() - t0 < timeout:
            time.sleep(0.02)
        time.sleep(0.1)

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


class FilePlayer(Player):
    """Silent renderer: collects chunks and writes <out>.wav; first_audio is stamped at the first chunk."""

    def __init__(self, out: Path | None = None) -> None:
        super().__init__()
        self.out = out
        self.chunks: list[np.ndarray] = []
        self.playing = False

    def play(self, pcm16: np.ndarray, sr: int) -> None:
        self._stamp()
        self.chunks.append(resample(pcm16, sr, OUT_RATE))

    def stop(self) -> None:
        self.chunks.clear()

    def join(self, timeout: float = 60) -> None:
        if self.out and self.chunks:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(self.out, np.concatenate(self.chunks), OUT_RATE)

    def reset_turn(self) -> None:
        super().reset_turn()
        self.chunks = []
