"""D4 pressure-driven tier governor: downgrade fast, upgrade slowly (hysteresis), log every transition."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from pv.config import config, path
from pv.guard.limiter import commit_free_mb
from pv.governor.pressure import Rolling, Signals, on_battery
from pv.governor.tiers import Tier

FOOTPRINT_FILE = path("config", "tier_footprint.json")
# Conservative defaults until `bench.run_degrade` measures the real numbers (written to FOOTPRINT_FILE).
DEFAULT_FOOTPRINT_MB = {0: 3000, 1: 2700, 2: 1700, 3: 1400, 4: 700}
DEFAULT_MIN_CORES = {0: 3, 1: 2, 2: 2, 3: 1, 4: 1}


class Governor:
    def __init__(self, tiers: list[Tier], limiter, sampler, apply_tier: Callable[[int], None],
                 target_ms: float | None = None, upgrade_after_s: float = 20.0, period_s: float = 0.5,
                 start_tier: int | None = None, on_transition: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.tiers, self.limiter, self.sampler, self.apply_tier = tiers, limiter, sampler, apply_tier
        self.on_transition = on_transition
        self._cooldown_until = 0.0
        # A big model is slow by nature: the latency target belongs to the profile (1.5 s for the 1.5B edge config, 7 s for 4B).
        self.target_ms = target_ms or float(config.get("governor", {}).get("target_ms", 1500.0))
        self.upgrade_after_s, self.period = upgrade_after_s, period_s
        self.footprint = self._load_footprint()
        self.lat, self.tps = Rolling(5), Rolling(5)
        self.tier_id = self.plan() if start_tier is None else start_tier
        self.forced = False
        self.log: list[dict[str, Any]] = []
        self._headroom_since: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.signals: Signals | None = None

    # ---- capacity plan ---------------------------------------------------
    @staticmethod
    def _load_footprint() -> dict[int, int]:
        if FOOTPRINT_FILE.exists():
            return {int(k): int(v) for k, v in json.loads(FOOTPRINT_FILE.read_text()).items()}
        configured = config.get("governor", {}).get("footprint_mb")        # a profile may declare its own model sizes
        return {i: int(v) for i, v in enumerate(configured)} if configured else dict(DEFAULT_FOOTPRINT_MB)

    def plan(self) -> int:
        """Best tier whose measured footprint fits under 90% of the declared RAM and whose core need is met."""
        cap, cores = self.limiter.ram_mb, self.limiter.cores
        for t in range(len(self.tiers)):
            if self.footprint[t] <= 0.9 * cap and cores >= DEFAULT_MIN_CORES[t]:
                return t
        return len(self.tiers) - 1

    # ---- signals ---------------------------------------------------------
    def observe_turn(self, latency_ms: float | None, tok_s: float, used_llm: bool) -> None:
        with self._lock:
            if latency_ms is not None and used_llm:
                self.lat.add(latency_ms)
            if tok_s > 0:
                self.tps.add(tok_s)

    def read(self) -> Signals:
        s = Signals(self.limiter.rss_mb() / self.limiter.ram_mb, self.sampler.cpu_percent,
                    self.limiter.cores * 100.0, self.tps.mean(), self.lat.median(), on_battery())
        self.signals = s
        return s

    COOLDOWN_S = 10.0              # after a transition, speed/latency rules wait for fresh evidence
    MIN_LAT_SAMPLES = 3
    MIN_TPS_SAMPLES = 2
    LOW_FREE_MB = 500              # system free memory below which we shrink (was 1200: loading the model itself tripped it)
    MAX_SLOW_TIER = 2

    def _wanted(self, s: Signals, t: float) -> tuple[int, str]:
        """Tier the current pressure asks for, with the reason (never below the capacity plan)."""
        cur, floor = self.tier_id, self.plan()
        free = commit_free_mb()
        settled = t >= self._cooldown_until
        if free < 250 or (free < self.LOW_FREE_MB and settled):
            return max(cur + 1, floor), f"system memory low ({free:.0f} MB free): other apps need RAM"
        if s.mem_ratio > 0.92:
            return max(cur + 1, floor), f"memory {s.mem_ratio:.0%} of cap"
        # Slowness alone may only lower the model to T2 (still a real LLM). Falling to the no-LLM tiers is for memory.
        if settled and cur < self.MAX_SLOW_TIER and len(self.lat.q) >= self.MIN_LAT_SAMPLES and s.lat_p50_ms > 2 * self.target_ms:
            return max(cur + 1, floor), f"latency p50 {s.lat_p50_ms:.0f} ms > 2x target"
        if settled and cur < self.MAX_SLOW_TIER and len(self.tps.q) >= self.MIN_TPS_SAMPLES and 0 < s.tok_s < 5:
            return max(cur + 1, floor), f"LLM speed {s.tok_s:.1f} tok/s"
        if floor > cur:
            return floor, "declared limit tightened"
        return cur, ""

    # ---- control loop ----------------------------------------------------
    def tick(self, t: float | None = None) -> None:
        t = time.monotonic() if t is None else t
        s = self.read()
        if self.forced:
            return
        want, why = self._wanted(s, t)
        if want > self.tier_id:                                      # downgrade fast
            self._headroom_since = None
            self._transition(want, why, t)
            return
        healthy = (s.mem_ratio < 0.70 and (not self.lat.q or s.lat_p50_ms < self.target_ms)
                   and (s.tok_s == 0 or s.tok_s >= 8) and self.plan() < self.tier_id and commit_free_mb() > 2500)
        if healthy:                                                  # upgrade only after sustained headroom
            if self._headroom_since is None:
                self._headroom_since = t
            if t - self._headroom_since >= self.upgrade_after_s:
                self._headroom_since = None
                self._transition(self.tier_id - 1, f"{self.upgrade_after_s:.0f}s of headroom", t)
        else:
            self._headroom_since = None

    def _transition(self, new: int, reason: str, t: float | None = None) -> None:
        new = max(0, min(len(self.tiers) - 1, new))
        if new == self.tier_id:
            return
        self.log.append({"t": time.time(), "from": self.tier_id, "to": new, "name": self.tiers[new].name,
                         "reason": reason})
        self.tier_id = new
        self.lat.q.clear()                       # measurements from the old tier say nothing about the new one
        self.tps.q.clear()
        self._cooldown_until = (time.monotonic() if t is None else t) + self.COOLDOWN_S
        if self.on_transition:
            self.on_transition(self.log[-1])
        self.apply_tier(new)

    def force_tier(self, tier_id: int, reason: str = "forced") -> None:
        self.forced = True
        self._transition(tier_id, reason)

    def release_force(self) -> None:
        self.forced = False

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.wait(self.period):
                self.tick()
        threading.Thread(target=loop, name="governor", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
