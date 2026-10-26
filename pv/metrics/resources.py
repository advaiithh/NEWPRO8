"""Process-tree CPU/RSS sampler (psutil). Peak RSS, CPU-seconds and CPU % over arbitrary windows."""
from __future__ import annotations

import threading
import time
from typing import Callable

import psutil


class ResourceSampler:
    def __init__(self, tree: Callable[[], list[psutil.Process]], period_s: float = 0.5) -> None:
        self.tree, self.period = tree, period_s
        self.rss_mb = 0.0
        self.peak_mb = 0.0
        self.commit_mb = 0.0          # private (committed) bytes: what the Job Object cap counts
        self.peak_commit_mb = 0.0
        self.cpu_percent = 0.0
        self._cpu_by_pid: dict[int, float] = {}
        self._cpu_done = 0.0          # CPU seconds of processes that exited
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_cpu, self._last_t = 0.0, time.perf_counter()
        threading.Thread(target=self._loop, name="resources", daemon=True).start()

    def _loop(self) -> None:
        while not self._stop.wait(self.period):
            self.sample()

    def sample(self) -> None:
        rss, commit, seen = 0, 0, {}
        for p in self.tree():
            try:
                mi = p.memory_info()
                rss += mi.rss
                commit += mi.private
                t = p.cpu_times()
                seen[p.pid] = t.user + t.system
            except psutil.Error:
                continue
        with self._lock:
            for pid, c in seen.items():
                self._cpu_by_pid[pid] = max(c, self._cpu_by_pid.get(pid, 0.0))
            self.rss_mb = rss / 2**20
            self.peak_mb = max(self.peak_mb, self.rss_mb)
            self.commit_mb = commit / 2**20
            self.peak_commit_mb = max(self.peak_commit_mb, self.commit_mb)
            total, t = sum(self._cpu_by_pid.values()), time.perf_counter()
            self.cpu_percent = 100 * (total - self._last_cpu) / max(t - self._last_t, 1e-6)
            self._last_cpu, self._last_t = total, t

    def cpu_seconds(self) -> float:
        self.sample()
        with self._lock:
            return sum(self._cpu_by_pid.values())

    def reset_peak(self) -> None:
        with self._lock:
            self.peak_mb = self.rss_mb

    def stop(self) -> None:
        self._stop.set()


class TurnMeter:
    """Measures CPU-seconds, peak RSS and energy over one turn (whole process tree)."""

    def __init__(self, sampler: ResourceSampler, energy, w_per_core: float) -> None:
        self.sampler, self.energy, self.w_per_core = sampler, energy, w_per_core
        self.start()

    def start(self) -> None:
        self.t0, self.cpu0 = time.perf_counter(), self.sampler.cpu_seconds()
        self.j0 = self.energy.joules() if self.energy and self.energy.available else None
        self.sampler.reset_peak()

    def stop(self) -> dict[str, float | None]:
        cpu = self.sampler.cpu_seconds() - self.cpu0
        wall = time.perf_counter() - self.t0
        return {"cpu_s": cpu, "wall_s": wall, "rss_peak_mb": self.sampler.peak_mb,
                "energy_j": (self.energy.joules() - self.j0) if self.j0 is not None else None,
                "energy_proxy_j": cpu * self.w_per_core}
