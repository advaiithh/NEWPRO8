"""D2: adaptive endpointing. The silence hangover depends on how complete the partial transcript looks."""
from __future__ import annotations

import re
from dataclasses import dataclass

INCOMPLETE_END = {"and", "but", "or", "the", "a", "an", "of", "to", "with", "for", "in", "on", "at", "if", "because",
                  "that", "so", "then", "is", "are", "my", "your", "by", "from", "about", "like", "what", "how", "than"}
QUESTION_START = {"what", "whats", "when", "where", "who", "why", "how", "which", "is", "are", "do", "does", "can",
                  "could", "will", "would", "should", "tell", "set", "convert", "remember", "roll", "flip"}


def hangover_ms(partial: str, intent_known: bool, base: float = 350, lo: float = 200, hi: float = 700) -> float:
    """Pure function: chosen hangover for the current partial transcript."""
    text = partial.strip().lower()
    words = re.findall(r"[a-z0-9']+", text)
    if not words:
        return base
    if text.endswith(",") or words[-1] in INCOMPLETE_END or len(words) < 2:
        return hi
    if intent_known:
        return lo
    if text.endswith("?") or (len(words) >= 3 and words[0] in QUESTION_START):
        return (base + lo) / 2
    return base


@dataclass
class EndpointEvent:
    kind: str                # "start" | "end"
    t: float                 # time of the last voiced frame (end) or first voiced frame (start)
    hangover_ms: float = 0.0


class Endpointer:
    """Consumes per-frame speech decisions; emits start/end events. Fixed mode ignores the partial."""

    def __init__(self, frame_ms: float = 32, min_speech_ms: float = 160, base: float = 350,
                 lo: float = 200, hi: float = 700, adaptive: bool = True) -> None:
        self.frame_ms, self.min_speech_ms = frame_ms, min_speech_ms
        self.base, self.lo, self.hi, self.adaptive = base, lo, hi, adaptive
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self.start_t = 0.0
        self.last_voice_t = 0.0
        self.hangover = self.base

    def update(self, voiced: bool, t: float, partial: str = "", intent_known: bool = False) -> EndpointEvent | None:
        if voiced:
            self.silence_ms = 0.0
            self.last_voice_t = t
            if not self.in_speech:
                self.speech_ms += self.frame_ms
                if self.speech_ms >= self.min_speech_ms:
                    self.in_speech = True
                    return EndpointEvent("start", t - self.speech_ms / 1000)
            return None
        if not self.in_speech:
            self.speech_ms = 0.0
            return None
        self.silence_ms += self.frame_ms
        self.hangover = (hangover_ms(partial, intent_known, self.base, self.lo, self.hi)
                         if self.adaptive else self.base)
        if self.silence_ms >= self.hangover:
            self.in_speech = False
            self.speech_ms = 0.0
            return EndpointEvent("end", self.last_voice_t, self.hangover)
        return None
