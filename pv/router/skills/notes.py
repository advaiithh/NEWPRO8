"""Local notes in data/notes.json: 'remember that ...' / 'what did I ask you to remember'."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path


class Notes:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.items: list[dict] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items), encoding="utf-8")

    def add(self, text: str) -> None:
        self.items.append({"text": text, "ts": time.time()})
        self._save()

    def clear(self) -> None:
        self.items = []
        self._save()


ADD = re.compile(r"^(?:please )?(?:remember|note|make a note|take a note|write down)(?: that| down)?\s+(?P<x>.+)$")
READ = re.compile(r"\b(what did i (?:ask you to |tell you to )?(?:remember|note)|what (?:are|were) my notes|"
                  r"read (?:back )?my notes|what do you remember|my notes)\b")
CLEAR = re.compile(r"\b(clear|delete|erase|forget) (?:all )?(?:my )?notes\b")


def match(text: str, ctx) -> str | None:
    notes: Notes | None = getattr(ctx, "notes", None)
    if notes is None:
        return None
    t = text.lower().strip().rstrip("?.!")
    if CLEAR.search(t):
        notes.clear()
        return "Okay, I cleared your notes."
    if READ.search(t):
        if not notes.items:
            return "You have not asked me to remember anything yet."
        return "You asked me to remember: " + "; ".join(i["text"] for i in notes.items[-5:]) + "."
    if m := ADD.match(t):
        notes.add(m["x"])
        return f"Okay, I will remember that {m['x']}."
    return None
