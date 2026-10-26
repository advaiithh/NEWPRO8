"""Process-wide wiring: offline tripwire, device limit, resource and energy meters."""
from __future__ import annotations

from pv.config import config
from pv.guard import gpucheck, netguard
from pv.guard.limiter import Limiter
from pv.metrics.energy import EnergyMeter
from pv.metrics.resources import ResourceSampler, TurnMeter


class Runtime:
    def __init__(self, cores: int | None = None, ram_mb: int | None = None, limit: bool = True, split: bool = True) -> None:
        netguard.install()
        cores = cores or config["limit"]["cores"]
        ram_mb = ram_mb or config["limit"]["ram_mb"]
        gpucheck.set_threads(cores)
        self.limiter = Limiter(cores, ram_mb, split=split) if limit else None
        self.sampler = ResourceSampler(self.limiter.tree if self.limiter else (lambda: []))
        self.energy = EnergyMeter()
        if self.limiter:
            self.limiter.start_watchdog()
            print(self.limiter.banner())

    def meter(self) -> TurnMeter:
        return TurnMeter(self.sampler, self.energy, config["energy"]["assumed_w_per_core"])

    def stop(self) -> None:
        self.sampler.stop()
        self.energy.stop()
        if self.limiter:
            self.limiter.stop()
