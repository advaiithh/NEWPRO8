"""Timers and alarms, persisted in data/timers.json and re-armed after restart."""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Callable

from .words import words_to_digits

UNIT_S = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}
DUR = re.compile(r"(\d+(?:\.\d+)?)\s*(second|sec|minute|min|hour|hr)s?\b")


class Timers:
    def __init__(self, path: Path, on_fire: Callable[[str], None] | None = None) -> None:
        self.path, self.on_fire = path, on_fire
        self.items: list[dict] = json.loads(path.read_text()) if path.exists() else []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, name="timers", daemon=True).start()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items))

    def add(self, seconds: float, label: str) -> None:
        with self._lock:
            self.items.append({"due": time.time() + seconds, "label": label})
            self._save()

    def clear(self) -> int:
        with self._lock:
            n, self.items = len(self.items), []
            self._save()
        return n

    def _loop(self) -> None:
        while not self._stop.wait(0.5):
            with self._lock:
                due = [i for i in self.items if i["due"] <= time.time()]
                if due:
                    self.items = [i for i in self.items if i not in due]
                    self._save()
            for i in due:
                if self.on_fire:
                    self.on_fire(f"Your {i['label']} is done.")

    def remaining(self) -> list[tuple[str, int]]:
        with self._lock:
            return [(i["label"], max(0, round(i["due"] - time.time()))) for i in self.items]

    def stop(self) -> None:
        self._stop.set()


def _spoken(seconds: int) -> str:
    parts = []
    for n, u in ((seconds // 3600, "hour"), (seconds % 3600 // 60, "minute"), (seconds % 60, "second")):
        if n:
            parts.append(f"{n} {u}{'s' if n != 1 else ''}")
    return " and ".join(parts) or "0 seconds"


def match(text: str, ctx) -> str | None:
    timers: Timers | None = getattr(ctx, "timers", None)
    if timers is None:
        return None
    t = words_to_digits(text.lower().replace("?", ""))
    if re.search(r"\b(timers?|alarms?)\b", t) and re.search(r"\b(set|start|create|add|begin)\b", t):
        total = sum(float(n) * UNIT_S[u] for n, u in DUR.findall(t))
        if total <= 0:
            return "For how long should I set the timer?"
        timers.add(total, f"{_spoken(int(total))} timer")
        return f"Okay, timer set for {_spoken(int(total))}."
    if re.search(r"\b(cancel|stop|clear|delete)\b.*\b(timers?|alarms?)\b", t):
        n = timers.clear()
        return f"Cleared {n} timer{'s' if n != 1 else ''}."
    if re.search(r"\b(how (?:much|long)|list|what|any|check)\b.*\b(timers?|alarms?)\b|\btime left\b", t):
        rem = timers.remaining()
        if not rem:
            return "You have no timers running."
        return " ".join(f"Your {label} has {_spoken(s)} left." for label, s in rem)
    return None
