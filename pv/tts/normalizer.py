"""D7: make text speakable (digits, currency, units, dates, markdown) before TTS."""
from __future__ import annotations

import re

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = [(10**9, "billion"), (10**6, "million"), (10**3, "thousand")]
MONTHS = "January February March April May June July August September October November December".split()
ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
       "twelve": "twelfth"}
UNITS = {"km": "kilometers", "kg": "kilograms", "g": "grams", "mg": "milligrams", "cm": "centimeters",
         "mm": "millimeters", "m": "meters", "km/h": "kilometers per hour", "mph": "miles per hour",
         "ml": "milliliters", "l": "liters", "gb": "gigabytes", "mb": "megabytes", "kb": "kilobytes",
         "ghz": "gigahertz", "mhz": "megahertz", "hz": "hertz", "kw": "kilowatts", "w": "watts", "s": "seconds",
         "ms": "milliseconds", "min": "minutes", "h": "hours", "hr": "hours", "lb": "pounds", "lbs": "pounds",
         "ft": "feet", "in": "inches", "mi": "miles"}
CURRENCY = {"₹": ("rupee", "rupees"), "rs": ("rupee", "rupees"), "rs.": ("rupee", "rupees"),
            "$": ("dollar", "dollars"), "€": ("euro", "euros"), "£": ("pound", "pounds")}


def int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + (" " + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" " + int_to_words(n % 100) if n % 100 else "")
    for size, name in SCALES:
        if n >= size:
            rest = n % size
            return int_to_words(n // size) + " " + name + (" " + int_to_words(rest) if rest else "")
    return str(n)


def ordinal(n: int) -> str:
    w = int_to_words(n)
    head, _, last = w.rpartition(" ")
    last = ORD.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return (head + " " + last).strip()


def number_to_words(s: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        return f"{int_to_words(int(whole or 0))} point " + " ".join(ONES[int(c)] for c in frac)
    return int_to_words(int(s))


def _year(n: int) -> str:
    if 2000 <= n < 2010:
        return int_to_words(n)
    return int_to_words(n // 100) + " " + (int_to_words(n % 100) if n % 100 >= 10 else ("oh " + ONES[n % 100] if n % 100 else "hundred"))


def _money(m: re.Match) -> str:
    sym, num = m.group(1).lower(), m.group(2)
    one, many = CURRENCY[sym]
    whole, _, frac = num.replace(",", "").partition(".")
    out = f"{int_to_words(int(whole))} {one if whole == '1' else many}"
    if frac.strip("0"):
        out += f" and {int_to_words(int(frac.ljust(2, '0')[:2]))} cents"
    return out


def normalize(text: str) -> str:
    t = re.sub(r"```.*?```", " ", text, flags=re.S)
    t = re.sub(r"[*_`#>~]+", "", t)
    t = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b",
               lambda m: f"{MONTHS[int(m[2]) - 1]} {ordinal(int(m[3]))}, {_year(int(m[1]))}", t)
    t = re.sub(r"(₹|\$|€|£|\bRs\.?)\s?(\d[\d,]*(?:\.\d+)?)", _money, t)
    t = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?%", lambda m: number_to_words(m[1]) + " percent", t)
    t = re.sub(r"(-?)(\d+(?:\.\d+)?)\s?°\s?([CF])\b",
               lambda m: f"{'minus ' if m[1] else ''}{number_to_words(m[2])} degrees "
                         f"{'Celsius' if m[3] == 'C' else 'Fahrenheit'}", t)
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: ordinal(int(m[1])), t)
    t = re.sub(r"\b(\d[\d,]*(?:\.\d+)?)\s?(km/h|mph|kg|km|mg|cm|mm|ml|gb|mb|kb|ghz|mhz|hz|kw|lbs|lb|ft|ms|min|hr|g|m|l|w|s|h)\b",
               lambda m: f"{number_to_words(m[1])} {UNITS[m[2].lower()]}", t, flags=re.I)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b",
               lambda m: f"{int_to_words(int(m[1]))} " + (f"oh {ONES[int(m[2])]}" if m[2][0] == "0" and m[2] != "00" else
                                                          "o'clock" if m[2] == "00" else int_to_words(int(m[2]))), t)
    t = re.sub(r"\d[\d,]*(?:\.\d+)?", lambda m: number_to_words(m[0]), t)
    t = t.replace("&", " and ").replace("+", " plus ").replace("=", " equals ")
    return re.sub(r"\s+", " ", t).strip()
