"""Small deterministic skills: time, date, day, coin, dice, repeat, volume/speed, capabilities."""
from __future__ import annotations

import datetime as dt
import random
import re

from .words import words_to_digits


def _clock(n: dt.datetime) -> str:
    return n.strftime("%I:%M %p").lstrip("0")


def match(text: str, ctx) -> str | None:
    t = words_to_digits(text.lower().replace("?", "").replace("'", ""))
    now = dt.datetime.now()
    if re.search(r"\bwhat time is it\b|\bwhats the time\b|\bcurrent time\b|\btell me the time\b|\bwhat is the time\b|\btime now\b", t):
        return f"It is {_clock(now)}."
    if re.search(r"\b(todays date|what is the date|whats the date|what date is it|current date|date today)\b", t):
        return f"Today is {now.strftime('%A, %B')} {now.day}, {now.year}."
    if re.search(r"\bwhat day (?:is it|of the week)|\bwhat is today\b|\bwhats today\b|\bwhich day is it\b", t):
        return f"It is {now.strftime('%A')}."
    if re.search(r"\b(flip|toss) a coin\b|\bheads or tails\b", t):
        return f"It is {random.choice(['heads', 'tails'])}."
    if m := re.search(r"\broll (?:a |an |(\d+) )?(?:dice|die|dices)\b", t):
        rolls = [random.randint(1, 6) for _ in range(min(int(m.group(1) or 1), 6))]
        return "You rolled " + (str(rolls[0]) if len(rolls) == 1 else ", ".join(map(str, rolls)) + f", total {sum(rolls)}") + "."
    if re.search(r"\b(say that again|repeat that|repeat|come again|say it again)\b", t):
        return getattr(ctx, "last_reply", "") or "I have not said anything yet."
    if re.search(r"\b(louder|volume up|turn it up|speak up)\b", t):
        ctx.actions.append(("volume", +0.2))
        return "Okay, louder."
    if re.search(r"\b(quieter|softer|volume down|turn it down)\b", t):
        ctx.actions.append(("volume", -0.2))
        return "Okay, quieter."
    if re.search(r"\b(speak|talk) (?:more )?slow(?:er)?\b|\bslow down\b", t):
        ctx.actions.append(("speed", +0.1))
        return "Okay, slower."
    if re.search(r"\b(speak|talk) (?:more )?fast(?:er)?\b|\bspeed up\b", t):
        ctx.actions.append(("speed", -0.1))
        return "Okay, faster."
    if re.search(r"\bwhat can you do\b|\bwhat are your (?:skills|abilities|features)\b|\bwhat do you do\b", t):
        return ("I can tell the time and date, do math, convert units, set timers, remember notes, "
                "flip a coin, roll dice, and chat. All of it runs on this device, offline.")
    return None
