"""Quality tiers (SPEC 5.11). Each tier fixes ASR size, LLM, context, reply cap and voice."""
from __future__ import annotations

from dataclasses import dataclass

from pv.config import config
from pv.llm import prompts


@dataclass(frozen=True)
class Tier:
    id: int
    name: str
    asr: str                  # moonshine size
    llm: str | None           # None = no LLM at all (survival)
    num_ctx: int
    num_predict: int
    voice: str
    system: str
    llm_when_idle_only: bool = False
    sapi: bool = False


def build() -> list[Tier]:
    full, low = config["ollama"]["full_model"], config["ollama"]["low_model"]
    m = config["ollama"].get("tier_models") or [full, full, low, low]       # profiles may give a model per tier
    vf, vl = config["tts"]["voice_full"], config["tts"]["voice_low"]
    return [
        Tier(0, "Full", "small", m[0], 1024, 80, vf, prompts.SYSTEM_FULL),
        Tier(1, "Medium", "tiny", m[1], 768, 56, vf, prompts.SYSTEM_FULL),
        Tier(2, "Low", "tiny", m[2], 768, 40, vl, prompts.SYSTEM_LOW),
        Tier(3, "Starved", "tiny", m[3], 512, 24, vl, prompts.SYSTEM_LOW, llm_when_idle_only=True),
        Tier(4, "Survival", "tiny", None, 512, 0, vl, prompts.SYSTEM_LOW),
    ]
