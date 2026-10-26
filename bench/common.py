"""Shared bench helpers: request loading, WER, quality checks, percentiles."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jiwer
import numpy as np

from pv.router.skills.words import words_to_digits

HERE = Path(__file__).resolve().parent
TESTSET = HERE / "testset"
OUT = HERE / "out"


def load_manifest(subset: str = "main", lang: str = "en") -> list[dict[str, Any]]:
    """subset: main (60 en + 10 hi), heldout (20), all, q40 (skills+open). lang filters en/hi."""
    items = json.loads((TESTSET / "manifest.json").read_text(encoding="utf-8"))
    if subset == "q40":                       # the 40 fixed quality prompts: 20 skill + 20 open
        items = [m for m in items if m["kind"] in ("skill", "open") and not m.get("heldout") and m.get("lang", "en") == "en"]
        return items
    sel = [m for m in items if (m.get("heldout", False) == (subset == "heldout") or subset == "all")]
    return [m for m in sel if m.get("lang", "en") == lang]


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\sऀ-ॿ]", " ", words_to_digits(s.lower()))).strip()


def wer(ref: str, hyp: str) -> float:
    r, h = norm_text(ref), norm_text(hyp)
    return float(jiwer.wer(r, h)) if r else 0.0


def quality_ok(req: dict[str, Any], reply: str, intent: str = "") -> bool | None:
    """Keyword/regex rubric. skill: regex on reply; open: any keyword; cache-type: intent match (None if not routed)."""
    low = reply.lower()
    if req["kind"] == "skill":
        return bool(re.search(req["expect"], reply, re.I))
    if req["kind"] == "open":
        return any(k.lower() in low for k in req["any"])
    return None if not intent else intent == req["intent"]


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def stats(xs: list[float]) -> dict[str, float]:
    xs = [x for x in xs if x is not None]
    return {"p50": pct(xs, 50), "p95": pct(xs, 95), "mean": float(np.mean(xs)) if xs else float("nan"),
            "worst": float(max(xs)) if xs else float("nan"), "n": len(xs)}
