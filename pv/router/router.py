"""D1 compute-avoidance router: skills -> semantic cache -> LLM. Cheapest first."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pv.router import skills
from pv.router.cache import CacheHit, SemanticCache


@dataclass
class Route:
    kind: str                         # skill | cache | llm
    reply: str = ""
    skill: str = ""
    intent: str = ""
    sim: float | None = None
    margin: float | None = None
    audio: tuple[np.ndarray, int] | None = None
    context: str = ""                 # retrieved facts handed to the LLM
    kb_title: str = ""


class Router:
    def __init__(self, ctx: skills.SkillContext, cache: SemanticCache | None, use_skills: bool = True,
                 use_cache: bool = True, kb=None) -> None:
        self.ctx, self.cache, self.kb = ctx, cache, kb
        self.use_skills, self.use_cache = use_skills, use_cache and cache is not None

    def route(self, text: str, lang: str = "en") -> Route:
        if self.use_skills and lang == "en":
            if hit := skills.run(text, self.ctx):
                return Route("skill", reply=hit[1], skill=hit[0])
        sim = margin = None
        if self.use_cache:
            ch, sim, margin = self.cache.lookup(text, lang)
            if ch:
                return Route("cache", reply=ch.text, intent=ch.intent, sim=sim, margin=margin, audio=ch.audio())
        if self.kb is not None and lang == "en":
            kind, hit, snippet = self.kb.answer(text)
            if kind == "extract":                      # definition question: read the article lead, no model guessing
                return Route("kb", reply=snippet, kb_title=hit.title, sim=hit.score)
            if kind == "context":                      # other factual question: give the LLM the best passage
                return Route("llm", sim=sim, margin=margin, context=snippet, kb_title=hit.title)
        return Route("llm", sim=sim, margin=margin)

    def intent_known(self, partial: str, lang: str = "en") -> bool:
        """Side-effect-free probe used by adaptive endpointing on partial transcripts."""
        if self.use_skills and lang == "en" and skills.would_match(partial):
            return True
        if self.use_cache:
            return self.cache.lookup(partial, lang)[0] is not None
        return False
