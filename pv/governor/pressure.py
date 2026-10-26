"""Pressure signals sampled every 500 ms (SPEC 5.11)."""
from __future__ import annotations

import collections
from dataclasses import dataclass

import psutil


@dataclass
class Signals:
    mem_ratio: float          # total tree RSS / declared cap
    cpu_pct: float            # tree CPU % (100 = one core)
    cpu_budget_pct: float     # cores * 100
    tok_s: float              # rolling LLM tokens/s (0 if none yet)
    lat_p50_ms: float         # p50 of the last 5 turn latencies (0 if none yet)
    on_battery: bool


class Rolling:
    def __init__(self, n: int = 5) -> None:
        self.q: collections.deque[float] = collections.deque(maxlen=n)

    def add(self, x: float) -> None:
        self.q.append(x)

    def median(self) -> float:
        s = sorted(self.q)
        if not s:
            return 0.0
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2

    def mean(self) -> float:
        return sum(self.q) / len(self.q) if self.q else 0.0


def on_battery() -> bool:
    b = psutil.sensors_battery()
    return bool(b and b.power_plugged is False)
