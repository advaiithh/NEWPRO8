"""Cheap language ID: script heuristics on text, Whisper detection on the first ~2 s of audio as fallback."""
from __future__ import annotations

import re

import numpy as np
from rapidfuzz import fuzz

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HINDI_WORDS = {"kya", "hai", "kaise", "aap", "mujhe", "batao", "samay", "kitna", "nahi", "haan", "mera", "tum",
               "kaun", "kahan", "aaj", "kal", "namaste", "dhanyavaad", "shukriya"}


def from_text(text: str) -> str | None:
    """'hi' when the text is Devanagari or clearly romanised Hindi, 'en' for plain English, None if unsure."""
    if not text.strip():
        return None
    letters = re.findall(r"\w", text)
    if letters and len(DEVANAGARI.findall(text)) / len(letters) > 0.3:
        return "hi"
    words = re.findall(r"[a-z']+", text.lower())
    if words and sum(w in HINDI_WORDS for w in words) / len(words) >= 0.3:
        return "hi"
    return "en" if words else None


def from_audio(whisper, pcm16: np.ndarray, allowed: tuple[str, ...] = ("en", "hi", "ta"), min_conf: float = 0.55) -> str:
    """Pick among the languages we ship (renormalising Whisper's probabilities), English unless another clearly wins.

    Short utterances are ambiguous (a lone Tamil word often scores as Malayalam or English), so a non-English
    language needs at least `min_conf` of the probability mass among the allowed languages.
    """
    probs = whisper.language_probs(pcm16[: 2 * 16000])
    mass = {k: probs.get(k, 0.0) for k in allowed}
    total = sum(mass.values()) or 1.0
    best = max(mass, key=mass.get)
    return best if best == "en" or mass[best] / total >= min_conf else "en"


CONFIRM = {"en": "Okay, I will speak English.", "ta": "\u0b9a\u0bb0\u0bbf, \u0b87\u0ba9\u0bbf \u0ba4\u0bae\u0bbf\u0bb4\u0bbf\u0bb2\u0bcd \u0baa\u0bc7\u0b9a\u0bc1\u0b95\u0bbf\u0bb1\u0bc7\u0ba9\u0bcd.",
           "hi": "\u0920\u0940\u0915 \u0939\u0948, \u0905\u092c \u092e\u0948\u0902 \u0939\u093f\u0902\u0926\u0940 \u092e\u0947\u0902 \u092c\u094b\u0932\u0942\u0902\u0917\u093e\u0964"}
NAMES = {"english": "en", "tamil": "ta", "hindi": "hi", "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd": "ta", "\u0b86\u0b99\u0bcd\u0b95\u0bbf\u0bb2\u0bae\u0bcd": "en",
         "\u0b86\u0b99\u0bcd\u0b95\u0bbf\u0bb2\u0ba4\u0bcd": "en", "\u0b87\u0ba8\u0bcd\u0ba4\u0bbf": "hi", "\u0939\u093f\u0902\u0926\u0940": "hi", "\u0905\u0902\u0917\u094d\u0930\u0947\u091c\u093c\u0940": "en",
         "\u0905\u0902\u0917\u094d\u0930\u0947\u091c\u0940": "en", "\u0924\u092e\u093f\u0932": "ta"}
VERBS = ("speak", "talk", "switch to", "reply in", "answer in", "respond in", "use", "change to", "go to",
         "\u0baa\u0bc7\u0b9a\u0bc1", "\u0baa\u0bc7\u0b9a\u0bc1\u0b99\u0bcd\u0b95\u0bb3\u0bcd", "\u092c\u094b\u0932\u094b", "\u092c\u094b\u0932\u093f\u090f", "\u092e\u0947\u0902", "\u0bbf\u0bb2\u0bcd")


def _has(text: str, word: str) -> bool:
    """Exact for Latin words; fuzzy for Tamil/Hindi, where recognisers slip on a letter or two."""
    return word in text if word.isascii() else fuzz.partial_ratio(word, text) >= 78


def switch_command(text: str) -> str | None:
    """'speak Tamil' / 'தமிழில் பேசு' / 'हिंदी में बोलो' / a bare 'Tamil' -> language code, else None. Short utterances only."""
    t = text.lower().strip().rstrip(".?!")
    words = t.split()
    if len(words) > 6:
        return None
    for name, code in NAMES.items():
        if _has(t, name) and (len(words) <= 2 or any(_has(t, v) for v in VERBS)):
            return code
    return None


NAMES.update({"தமிழ": "ta", "ஆங்கில": "en", "இந்தி": "hi"})          # word stems: Tamil inflects the ending (தமிழ், தமிழில்)
