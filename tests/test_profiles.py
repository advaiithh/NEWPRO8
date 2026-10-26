"""Profiles pick the LLM per tier; the governor plans from the profile's measured footprints."""
import pytest

from pv import config as cfgmod
from pv.governor.governor import Governor
from pv.governor.tiers import build


@pytest.fixture(autouse=True)
def restore_config():
    yield
    cfgmod.reload(None)


class Lim:
    def __init__(self, ram, cores):
        self.ram_mb, self.cores = ram, cores

    def rss_mb(self):
        return 1000.0


class Samp:
    cpu_percent = 10.0


def plan(ram, cores):
    return Governor(build(), Lim(ram, cores), Samp(), lambda i: None).tier_id


def test_default_profile_uses_the_benchmarked_small_models():
    cfgmod.reload(None)
    t = build()
    assert t[0].llm == "qwen2.5:1.5b-instruct" and t[2].llm == "qwen2.5:0.5b-instruct" and t[4].llm is None


def test_quality_profile_puts_the_4b_model_on_top_and_falls_back_with_memory():
    cfgmod.reload("quality")
    t = build()
    assert [x.llm for x in t] == ["qwen3.5:4b", "qwen2.5:3b-instruct", "qwen2.5:1.5b-instruct", "qwen2.5:0.5b-instruct", None]
    assert cfgmod.config["limit"] == {"cores": 6, "ram_mb": 8192}
    assert plan(8192, 6) == 0                     # roomy device: best model
    assert plan(5120, 4) == 1                     # tighter: 3B
    assert plan(4096, 4) >= 2                     # the 4 GB edge limit cannot hold 4B or 3B plus the stack
    assert plan(1536, 2) == 4                     # survival: no LLM at all


def test_balanced_profile():
    cfgmod.reload("balanced")
    assert build()[0].llm == "qwen2.5:3b-instruct" and plan(5120, 4) == 0 and plan(4096, 4) >= 1
