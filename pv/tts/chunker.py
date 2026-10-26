"""D6: cut an LLM token stream into speakable chunks. Short first chunk, then sentence/clause cuts."""
from __future__ import annotations

import re

ABBREV = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "no"}
SENT = re.compile(r"[.?!]+[\"')]*\s")
CLAUSE = re.compile(r"[,;:]\s")


class Chunker:
    def __init__(self, first_min: int = 4, first_max: int = 10, clause_min: int = 6, max_words: int = 12) -> None:
        self.first_min, self.first_max = first_min, first_max
        self.clause_min, self.max_words = clause_min, max_words
        self.buf = ""
        self.emitted = 0

    def _words(self, s: str) -> int:
        return len(s.split())

    def _limits(self) -> tuple[int, int]:
        return (self.first_min, self.first_max) if self.emitted == 0 else (self.clause_min, self.max_words)

    def _cut(self) -> int | None:
        """Index just after the chunk to emit, or None if more text is needed."""
        lo, hi = self._limits()
        for m in SENT.finditer(self.buf):
            before = self.buf[:m.start()].split()
            if before and before[-1].lower().rstrip(".") in ABBREV:
                continue
            if self._words(self.buf[:m.end()]) >= (2 if self.emitted == 0 else 1):
                return m.end()
        for m in CLAUSE.finditer(self.buf):
            if self._words(self.buf[:m.end()]) >= lo:
                return m.end()
        if len(self.buf.split()) > hi and self.buf[-1:].isspace():
            return self._nth_word_end(hi)
        return None

    def _nth_word_end(self, n: int) -> int:
        count, i = 0, 0
        for m in re.finditer(r"\S+", self.buf):
            count += 1
            i = m.end()
            if count == n:
                break
        return i

    def push(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while (i := self._cut()) is not None:
            chunk, self.buf = self.buf[:i].strip(), self.buf[i:].lstrip()
            if chunk:
                out.append(chunk)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        if rest:
            self.emitted += 1
            return [rest]
        return []
