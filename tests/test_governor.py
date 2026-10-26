"""D4: hysteresis and tier transitions with simulated pressure (no models needed)."""
from dataclasses import dataclass

from pv.governor.governor import Governor
from pv.governor.tiers import build


@dataclass
class FakeLimiter:
    ram_mb: int = 4096
    cores: int = 4
    rss: float = 1000.0

    def rss_mb(self) -> float:
        return self.rss


@dataclass
class FakeSampler:
    cpu_percent: float = 50.0


@__import__("pytest").fixture(autouse=True)
def _plenty_of_memory(monkeypatch):
    import pv.governor.governor as gmod
    monkeypatch.setattr(gmod, "commit_free_mb", lambda: 9000.0)


def make(limiter=None, start=None):
    applied = []
    g = Governor(build(), limiter or FakeLimiter(), FakeSampler(), applied.append, start_tier=start)
    return g, applied


def test_capacity_plan_follows_declared_limit():
    assert make(FakeLimiter(4096, 4))[0].tier_id == 0
    assert make(FakeLimiter(2048, 2))[0].tier_id >= 2
    assert make(FakeLimiter(1536, 2))[0].tier_id >= 3
    assert make(FakeLimiter(600, 1))[0].tier_id == 4


def test_memory_pressure_downgrades_immediately_one_step():
    lim = FakeLimiter()
    g, applied = make(lim)
    lim.rss = 0.95 * lim.ram_mb
    g.tick(t=0.0)
    assert g.tier_id == 1 and applied == [1]
    assert g.log[0]["reason"].startswith("memory")


def test_latency_triggers_downgrade_only_after_enough_samples():
    g, applied = make()
    g.observe_turn(4000, 20, True)
    g.observe_turn(4200, 20, True)
    g.tick(t=0.0)
    assert g.tier_id == 0                          # two slow turns are not enough
    g.observe_turn(4100, 20, True)
    g.tick(t=1.0)
    assert g.tier_id == 1 and "latency" in g.log[-1]["reason"]


def test_slow_llm_speed_downgrades():
    g, _ = make()
    g.observe_turn(900, 3.0, True)
    g.tick(t=0.0)
    assert g.tier_id == 0                          # one slow sample is not enough
    g.observe_turn(900, 3.2, True)
    g.tick(t=1.0)
    assert g.tier_id == 1 and "tok/s" in g.log[-1]["reason"]


def test_upgrade_needs_twenty_seconds_of_headroom():
    lim = FakeLimiter()
    g, applied = make(lim, start=2)
    for _ in range(3):
        g.observe_turn(600, 25, True)
    g.tick(t=0.0)
    g.tick(t=10.0)
    assert g.tier_id == 2                          # not yet
    g.tick(t=19.0)
    assert g.tier_id == 2
    g.tick(t=20.5)
    assert g.tier_id == 1 and applied == [1]       # one step at a time
    g.tick(t=21.0)
    assert g.tier_id == 1


def test_headroom_timer_resets_when_pressure_returns():
    lim = FakeLimiter()
    g, _ = make(lim, start=2)
    for _ in range(3):
        g.observe_turn(600, 25, True)
    g.tick(t=0.0)
    lim.rss = 0.80 * lim.ram_mb                     # not healthy (>70%) but not critical
    g.tick(t=15.0)
    lim.rss = 1000
    g.tick(t=25.0)                                  # timer restarted at 25
    assert g.tier_id == 2
    g.tick(t=46.0)
    assert g.tier_id == 1


def test_tightening_the_limit_forces_a_drop_without_waiting():
    lim = FakeLimiter()
    g, applied = make(lim)
    lim.ram_mb, lim.cores = 1536, 2                 # live "tighten limit"
    g.tick(t=0.0)
    assert g.tier_id >= 3 and applied
    assert g.log[-1]["reason"] == "declared limit tightened" or "memory" in g.log[-1]["reason"]


def test_force_tier_and_release():
    g, applied = make()
    g.force_tier(4)
    assert g.tier_id == 4
    g.tick(t=100.0)
    assert g.tier_id == 4                           # forced stays
    g.release_force()
    assert g.forced is False
    assert applied[-1] == 4


def test_every_transition_is_logged_with_reason():
    lim = FakeLimiter()
    g, _ = make(lim)
    lim.rss = 0.95 * lim.ram_mb
    for t in range(6):
        g.tick(t=float(t))
    assert [e["to"] for e in g.log] == [1, 2, 3, 4]
    assert all(e["reason"] for e in g.log)


def test_one_slow_turn_never_cascades_to_survival():
    """Live-test bug: one 5 s cold turn dropped T0->T4 in two seconds because stale latency kept re-triggering."""
    g, applied = make()
    for ms in (5054, 5054, 5054):
        g.observe_turn(ms, 25, True)
    for k in range(40):                           # 20 s of 0.5 s ticks with no new turns
        g.tick(t=k * 0.5)
    assert g.tier_id == 1 and applied == [1]      # exactly one step; the old samples were discarded
    assert len(g.log) == 1


def test_cooldown_blocks_a_second_latency_drop_until_fresh_evidence():
    g, applied = make()
    for ms in (5000, 5000, 5000):
        g.observe_turn(ms, 25, True)
    g.tick(t=0.0)
    for ms in (5000, 5000, 5000):
        g.observe_turn(ms, 25, True)
    g.tick(t=5.0)                                 # inside the 10 s cooldown
    assert g.tier_id == 1
    g.tick(t=11.0)                                # evidence is fresh and the cooldown is over
    assert g.tier_id == 2


def test_median_of_two_is_the_mean():
    from pv.governor.pressure import Rolling
    r = Rolling()
    r.add(5054)
    r.add(2831)
    assert r.median() == (5054 + 2831) / 2


def test_low_system_memory_downgrades_even_when_our_own_use_is_fine(monkeypatch):
    """Live crash: Chrome etc. ate the machine's memory; the assistant must shrink, not run out of RAM."""
    import pv.governor.governor as gmod
    g, applied = make()
    monkeypatch.setattr(gmod, "commit_free_mb", lambda: 400.0)
    g.tick(t=0.0)
    assert g.tier_id == 1 and "system memory low" in g.log[-1]["reason"]


def test_plenty_of_system_memory_does_not_trigger(monkeypatch):
    import pv.governor.governor as gmod
    g, _ = make()
    monkeypatch.setattr(gmod, "commit_free_mb", lambda: 9000.0)
    g.tick(t=0.0)
    assert g.tier_id == 0


def test_slowness_alone_never_drops_into_the_no_llm_tiers():
    """Live bug: the 4B model needs 4-7 s per answer, so a 1.5 s target pushed it down to the 'low on resources' tier."""
    g, applied = make(start=2)
    for _ in range(40):
        for _ in range(3):
            g.observe_turn(9000, 25, True)
        g.tick(t=g.COOLDOWN_S * (len(g.log) + 1) + 100)
    assert g.tier_id == 2 and applied == []                 # at T2 slowness cannot push further


def test_latency_target_comes_from_the_profile():
    from pv import config as cfgmod
    cfgmod.reload("quality")
    try:
        g, _ = make(FakeLimiter(8192, 6))
        assert g.target_ms == 7000
        for _ in range(3):
            g.observe_turn(6000, 12, True)                  # a normal 4B answer
        g.tick(t=0.0)
        assert g.tier_id == 0
    finally:
        cfgmod.reload(None)
    g, _ = make()
    assert g.target_ms == 1500


def test_memory_rule_needs_real_scarcity(monkeypatch):
    import pv.governor.governor as gmod
    g, _ = make()
    monkeypatch.setattr(gmod, "commit_free_mb", lambda: 900.0)       # the old rule (1200) would have fired here
    g.tick(t=0.0)
    assert g.tier_id == 0
