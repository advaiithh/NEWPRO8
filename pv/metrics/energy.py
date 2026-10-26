"""Energy: Windows Energy Meter (RAPL package power via PDH) when present, else a labeled CPU-seconds proxy."""
from __future__ import annotations

import threading
import time

import win32pdh

COUNTER = r"\Energy Meter(RAPL_Package0_PKG)\Power"


class EnergyMeter:
    """Integrates package power (mW -> W) over time. Whole-machine package power, not per-process."""

    def __init__(self, period_s: float = 0.5) -> None:
        self.period = period_s
        self.available = False
        self._lock = threading.Lock()
        self._joules = 0.0
        self._last_w = 0.0
        self._stop = threading.Event()
        try:
            self._q = win32pdh.OpenQuery()
            self._c = win32pdh.AddEnglishCounter(self._q, COUNTER)
            win32pdh.CollectQueryData(self._q)
            self.available = True
            threading.Thread(target=self._loop, name="energy", daemon=True).start()
        except win32pdh.error:
            self.available = False

    def _loop(self) -> None:
        last = time.perf_counter()
        while not self._stop.wait(self.period):
            win32pdh.CollectQueryData(self._q)
            _, mw = win32pdh.GetFormattedCounterValue(self._c, win32pdh.PDH_FMT_DOUBLE)
            t = time.perf_counter()
            with self._lock:
                self._last_w = mw / 1000
                self._joules += self._last_w * (t - last)
            last = t

    def joules(self) -> float:
        with self._lock:
            return self._joules

    def watts(self) -> float:
        with self._lock:
            return self._last_w

    def stop(self) -> None:
        self._stop.set()


def proxy_joules(cpu_seconds: float, w_per_core: float) -> float:
    """Labeled proxy: cpu_seconds x assumed W/core. Never present as a measurement."""
    return cpu_seconds * w_per_core
