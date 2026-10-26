"""Spoken numbers -> digits ("twenty three" -> 23, "two and a half" -> 2.5)."""
from __future__ import annotations

import re

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALE = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9, "lakh": 10**5, "crore": 10**7}
_NUMWORDS = set(_UNITS) | set(_TENS) | set(_SCALE) | {"and", "a", "half", "point"}


def _parse_run(words: list[str]) -> float | None:
    total, cur, i = 0.0, 0.0, 0
    seen = False
    while i < len(words):
        w = words[i]
        if w in _UNITS:
            cur += _UNITS[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w in _SCALE:
            cur = (cur or 1) * _SCALE[w]
            if _SCALE[w] >= 1000:
                total, cur = total + cur, 0
        elif w == "point":
            frac = ""
            for d in words[i + 1:]:
                if d not in _UNITS or _UNITS[d] > 9:
                    break
                frac += str(_UNITS[d])
            return total + cur + (float("0." + frac) if frac else 0)
        elif w == "and":
            if words[i + 1:i + 3] == ["a", "half"]:
                return total + cur + 0.5
            i += 1
            continue
        elif w == "half":
            return total + cur + 0.5
        else:
            return None
        seen = True
        i += 1
    return total + cur if seen else None


def words_to_digits(text: str) -> str:
    """Replace maximal runs of number words with digits; digits already in the text are untouched."""
    text = re.sub(r"\b(" + "|".join(_TENS) + r")-(" + "|".join(k for k in _UNITS if 0 < _UNITS[k] < 10) + r")\b", r"\1 \2", text.lower())
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-zA-Z']+|[^\sA-Za-z\d]", text.lower())
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in _NUMWORDS - {"and", "a", "half", "point"}:
            j = i
            while j < len(tokens) and tokens[j] in _NUMWORDS:
                j += 1
            while j > i and tokens[j - 1] in ("and", "a", "point"):
                j -= 1                                    # trailing connectors are not part of the number
            v = _parse_run(tokens[i:j])
            if v is not None:
                out.append(str(int(v)) if v == int(v) else str(v))
                i = j
                continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)
