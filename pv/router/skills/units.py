"""Unit conversion skill."""
from __future__ import annotations

import re

from .words import words_to_digits

# unit -> (dimension, factor to base)
U: dict[str, tuple[str, float]] = {}
for dim, table in {
    "len": {"meter": 1, "kilometer": 1000, "centimeter": 0.01, "millimeter": 0.001, "mile": 1609.344,
            "yard": 0.9144, "foot": 0.3048, "inch": 0.0254},
    "mass": {"kilogram": 1, "gram": 0.001, "pound": 0.45359237, "ounce": 0.028349523, "ton": 1000},
    "vol": {"liter": 1, "milliliter": 0.001, "gallon": 3.785411784, "cup": 0.2365882365, "pint": 0.473176473},
    "speed": {"kilometer per hour": 1, "mile per hour": 1.609344, "meter per second": 3.6},
    "data": {"megabyte": 1, "gigabyte": 1024, "kilobyte": 1 / 1024, "terabyte": 1024 * 1024},
}.items():
    for name, f in table.items():
        U[name] = (dim, f)
ALIASES = {"meters": "meter", "metres": "meter", "kilometers": "kilometer", "kilometres": "kilometer", "km": "kilometer",
           "kms": "kilometer", "kilos": "kilogram", "kilo": "kilogram", "kg": "kilogram", "kgs": "kilogram",
           "miles": "mile", "feet": "foot", "ft": "foot", "inches": "inch", "yards": "yard", "pounds": "pound",
           "lbs": "pound", "lb": "pound", "ounces": "ounce", "grams": "gram", "liters": "liter", "litres": "liter",
           "litre": "liter", "gallons": "gallon", "cups": "cup", "pints": "pint", "centimeters": "centimeter",
           "centimetres": "centimeter", "cm": "centimeter", "millimeters": "millimeter", "mm": "millimeter",
           "milliliters": "milliliter", "ml": "milliliter", "tons": "ton", "mph": "mile per hour",
           "kph": "kilometer per hour", "kmh": "kilometer per hour", "megabytes": "megabyte", "mb": "megabyte",
           "gigabytes": "gigabyte", "gb": "gigabyte", "kilobytes": "kilobyte", "terabytes": "terabyte",
           "miles per hour": "mile per hour", "kilometers per hour": "kilometer per hour"}
TEMP = {"celsius": "c", "centigrade": "c", "fahrenheit": "f", "kelvin": "k"}
NAMES = sorted({*U, *ALIASES, *TEMP}, key=len, reverse=True)
UNIT_RE = "|".join(re.escape(n) for n in NAMES)
PATTERNS = [
    re.compile(rf"convert\s+(?P<n>-?\d+(?:\.\d+)?)\s*(?:degrees?\s+)?(?P<a>{UNIT_RE})\s+(?:to|into|in)\s+(?:degrees?\s+)?(?P<b>{UNIT_RE})\b"),
    re.compile(rf"(?P<n>-?\d+(?:\.\d+)?)\s*(?:degrees?\s+)?(?P<a>{UNIT_RE})\s+(?:to|in|into)\s+(?:degrees?\s+)?(?P<b>{UNIT_RE})\b"),
    re.compile(rf"how many\s+(?P<b>{UNIT_RE})\s+(?:are |is )?(?:there )?in\s+(?:a |an |one )?(?P<n>\d+(?:\.\d+)?)?\s*(?P<a>{UNIT_RE})\b"),
]


def _canon(u: str) -> str:
    return ALIASES.get(u, u)


def _temp(v: float, a: str, b: str) -> float:
    c = {"c": v, "f": (v - 32) * 5 / 9, "k": v - 273.15}[a]
    return {"c": c, "f": c * 9 / 5 + 32, "k": c + 273.15}[b]


def _fmt(v: float) -> str:
    return str(round(v)) if abs(v - round(v)) < 1e-9 else f"{v:.2f}".rstrip("0").rstrip(".")


def match(text: str, ctx: object | None = None) -> str | None:
    t = words_to_digits(text.lower().replace("?", "").replace("°", " "))
    for pat in PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        n = float(m.group("n") or 1)
        a, b = m.group("a"), m.group("b")
        if a in TEMP and b in TEMP:
            return f"{_fmt(n)} degrees {a} is {_fmt(_temp(n, TEMP[a], TEMP[b]))} degrees {b}."
        (da, fa), (db, fb) = U.get(_canon(a), (None, 0)), U.get(_canon(b), (None, 0))
        if da and da == db:
            return f"{_fmt(n)} {a} is {_fmt(n * fa / fb)} {b}."
    return None
