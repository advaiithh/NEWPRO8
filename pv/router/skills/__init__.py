"""Deterministic skills, tried before any model is touched."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import calc, misc, notes, timers, units


@dataclass
class SkillContext:
    data_dir: Path
    last_reply: str = ""
    actions: list[tuple[str, float]] = field(default_factory=list)
    notes: notes.Notes | None = None
    timers: timers.Timers | None = None

    @classmethod
    def create(cls, data_dir: Path, on_timer: Any = None) -> "SkillContext":
        return cls(data_dir, notes=notes.Notes(data_dir / "notes.json"),
                   timers=timers.Timers(data_dir / "timers.json", on_timer))


ORDER = [("timer", timers), ("notes", notes), ("units", units), ("calc", calc), ("misc", misc)]


def run(text: str, ctx: SkillContext) -> tuple[str, str] | None:
    """Return (skill_name, reply) or None."""
    for name, mod in ORDER:
        if (reply := mod.match(text, ctx)) is not None:
            return name, reply
    return None


class _DryTimers:
    items: list = []

    def add(self, *a: Any) -> None: ...
    def clear(self) -> int:
        return 0

    def remaining(self) -> list:
        return []


class _DryNotes:
    items: list = []

    def add(self, *a: Any) -> None: ...
    def clear(self) -> None: ...


def would_match(text: str) -> bool:
    """True if some skill would answer this text. Uses throwaway state so nothing is stored or started."""
    dry = SkillContext(Path("."), notes=_DryNotes(), timers=_DryTimers())  # type: ignore[arg-type]
    return run(text, dry) is not None
