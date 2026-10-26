"""Semantic response cache: model2vec static embeddings over intent paraphrases -> pre-rendered audio."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from model2vec import StaticModel
from rapidfuzz import fuzz

from pv.config import config, model, path

INTENTS = Path(__file__).with_name("intents.yaml")
INTENTS_TA = Path(__file__).with_name("intents_ta.yaml")
EXACT = 0.97          # a near-verbatim paraphrase needs no margin over neighbouring intents


INDIC = "ऀ-ॿ஀-௿"          # Devanagari + Tamil: vowel signs are not \w, so keep the whole blocks


def clean(text: str) -> str:
    """Lowercase, drop punctuation (keeps Devanagari and Tamil letters and vowel signs), collapse spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s" + INDIC + "']", " ", text.lower())).strip()


@dataclass
class CacheHit:
    intent: str
    answer_idx: int
    text: str
    sim: float
    margin: float
    lang: str

    @property
    def audio_path(self) -> Path:
        return path(config["paths"]["cache"], "audio", self.lang, f"{self.intent}_{self.answer_idx}.wav")

    def audio(self) -> tuple[np.ndarray, int] | None:
        p = self.audio_path
        if not p.exists():
            return None
        data, sr = sf.read(p, dtype="int16")
        return data, sr


class SemanticCache:
    def __init__(self, tau: float | None = None, margin: float | None = None) -> None:
        self.tau = config["router"]["tau_hit"] if tau is None else tau
        self.margin = config["router"]["margin"] if margin is None else margin
        self.model = StaticModel.from_pretrained(str(model("embed", "potion-base-8M")))
        self.tau_indic = config["router"].get("tau_indic", 0.70)
        self.intents = [i for f in (INTENTS, INTENTS_TA) for i in yaml.safe_load(f.read_text(encoding="utf-8"))]
        texts, owner = [], []
        for i, it in enumerate(self.intents):
            for q in it["q"]:
                texts.append(clean(q))
                owner.append(i)
        self.texts = texts
        self.owner = np.array(owner)
        self.lang_of = np.array([self.intents[o]["lang"] for o in owner])
        self.matrix = self._embed(texts)

    def _embed(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def scores(self, text: str, lang: str = "en") -> list[tuple[float, int]]:
        """Best similarity per intent of this language, descending."""
        if lang == "en":
            sims = self.matrix @ self._embed([clean(text)])[0]
        else:        # the English static embeddings cannot read Indic scripts: use character similarity instead
            q = clean(text)
            sims = np.array([fuzz.ratio(q, p) / 100 for p in self.texts])
        best: dict[int, float] = {}
        for s, o, lg in zip(sims, self.owner, self.lang_of):
            if lg == lang and s > best.get(int(o), -1.0):
                best[int(o)] = float(s)
        return sorted(((s, o) for o, s in best.items()), reverse=True)

    def lookup(self, text: str, lang: str = "en", tau: float | None = None) -> tuple[CacheHit | None, float, float]:
        """Return (hit or None, top similarity, margin over the second-best intent)."""
        ranked = self.scores(text, lang)
        if not ranked or not clean(text):
            return None, 0.0, 0.0
        top, intent = ranked[0]
        margin = top - (ranked[1][0] if len(ranked) > 1 else 0.0)
        floor = tau if tau is not None else (self.tau if lang == "en" else self.tau_indic)
        if top >= floor and (margin >= self.margin or top >= EXACT):
            it = self.intents[intent]
            k = random.randrange(len(it["a"]))
            return CacheHit(it["id"], k, it["a"][k], top, margin, lang), top, margin
        return None, top, margin
