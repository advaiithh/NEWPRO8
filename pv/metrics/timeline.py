"""Per-turn timeline record. All stamps are time.perf_counter() seconds."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STAMPS = ["wake", "speech_start", "speech_end", "asr_final", "route_decision", "llm_first_token",
          "tts_first_chunk_ready", "first_audio", "turn_end"]


def now() -> float:
    return time.perf_counter()


@dataclass
class Timeline:
    system: str = ""
    utterance: str = ""
    stamps: dict[str, float] = field(default_factory=dict)
    tier: int = 0
    hangover_ms: float = 0.0
    path: str = ""              # skill | cache | llm | baseline
    similarity: float | None = None
    transcript: str = ""
    reply: str = ""
    lang: str = "en"
    tokens: int = 0
    tokens_per_s: float = 0.0
    cpu_s: float = 0.0
    energy_j: float | None = None
    rss_peak_mb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def mark(self, name: str, t: float | None = None) -> None:
        self.stamps.setdefault(name, now() if t is None else t)

    @property
    def latency_ms(self) -> float | None:
        s = self.stamps
        if "first_audio" in s and "speech_end" in s:
            return (s["first_audio"] - s["speech_end"]) * 1000
        return None

    def stages_ms(self) -> dict[str, float]:
        """Durations between consecutive recorded stamps after speech_end."""
        keys = [k for k in STAMPS[2:] if k in self.stamps]
        return {f"{a}->{b}": (self.stamps[b] - self.stamps[a]) * 1000 for a, b in zip(keys, keys[1:])}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["latency_ms"] = self.latency_ms
        d["stages_ms"] = self.stages_ms()
        return d

    def log(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict()) + "\n")
