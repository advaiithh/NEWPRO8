"""Cache honesty (SPEC 10.5): hit rate, wrong-answer rate and the effect of tau/margin.

dev  = cache-type requests in bench/requests.yaml; tau and margin are CHOSEN on dev only.
test = bench/heldout.yaml, evaluated once at the chosen setting (never used for tuning).
Negatives (must not be answered from the cache): all skill/open requests of the same file plus hand-written
near-domain questions. Text-level (no ASR noise); spoken-pipeline numbers come from bench/run_bench turns.
"""
from __future__ import annotations

import json

import numpy as np
import yaml

from bench.common import HERE, OUT
from pv.router.cache import SemanticCache

NEAR_DOMAIN_NEGATIVES = [
    "what is the name of the capital of France", "how old is the universe", "who made the first computer",
    "how are electric cars made", "what time does the sun set in london", "tell me about the history of jokes",
    "how does a gpu work", "what is a wake word", "how does the internet work", "who is the president of india",
    "what is the weather like on mars", "how do I hear better", "what is the meaning of the word hello",
    "play the song thank you", "is the cloud made of water", "how do robots learn", "why do we sleep",
    "what is a good name for a dog", "how fast is light", "can you recommend a movie about space",
]


def split(name: str) -> tuple[list[dict], list[str]]:
    rows = [r for r in yaml.safe_load((HERE / name).read_text(encoding="utf-8")) if r.get("lang", "en") == "en"]
    return [r for r in rows if r["kind"] == "cache"], [r["text"] for r in rows if r["kind"] != "cache"]


def evaluate(cache: SemanticCache, pos: list[dict], neg: list[str], tau: float, margin: float) -> dict:
    cache.margin = margin
    hits = wrong = 0
    for r in pos:
        h, _, _ = cache.lookup(r["text"], "en", tau)
        hits += h is not None
        wrong += h is not None and h.intent != r["intent"]
    fp = sum(cache.lookup(t, "en", tau)[0] is not None for t in neg)
    return {"tau": tau, "margin": margin, "hit_rate": hits / len(pos), "wrong_answer_rate_of_hits": wrong / hits if hits else 0.0,
            "false_hit_rate_on_negatives": fp / len(neg), "n_pos": len(pos), "n_neg": len(neg)}


def failures(cache: SemanticCache, pos: list[dict], neg: list[str], tau: float, margin: float) -> list[dict]:
    cache.margin = margin
    out = []
    for r in pos:
        h, sim, m = cache.lookup(r["text"], "en", tau)
        if h is None or h.intent != r["intent"]:
            out.append({"text": r["text"], "expected": r["intent"], "got": h.intent if h else None, "sim": round(sim, 2), "margin": round(m, 2)})
    for t in neg:
        h, sim, m = cache.lookup(t, "en", tau)
        if h:
            out.append({"text": t, "expected": None, "got": h.intent, "sim": round(sim, 2), "margin": round(m, 2)})
    return out


def main() -> None:
    cache = SemanticCache()
    dev_pos, dev_neg = split("requests.yaml")
    dev_neg += NEAR_DOMAIN_NEGATIVES
    test_pos, test_neg = split("heldout.yaml")
    taus = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]
    sweep = [evaluate(cache, dev_pos, dev_neg, t, m) for m in (0.03, 0.05) for t in taus]
    ok = [r for r in sweep if r["wrong_answer_rate_of_hits"] <= 0.10 and r["false_hit_rate_on_negatives"] <= 0.10]
    chosen = max(ok, key=lambda r: r["hit_rate"]) if ok else max(sweep, key=lambda r: -r["wrong_answer_rate_of_hits"])
    test = evaluate(cache, test_pos, test_neg, chosen["tau"], chosen["margin"])
    spec_default = evaluate(cache, test_pos, test_neg, 0.80, 0.05)
    res = {"sweep_dev": sweep, "chosen_on_dev": chosen, "heldout_at_chosen": test, "heldout_at_spec_default_0.80": spec_default,
           "selection_rule": "max dev hit rate s.t. wrong-answer rate <= 10% and false hits on negatives <= 10%",
           "failures_dev_at_chosen": failures(cache, dev_pos, dev_neg, chosen["tau"], chosen["margin"]),
           "failures_heldout_at_chosen": failures(cache, test_pos, test_neg, chosen["tau"], chosen["margin"])}
    OUT.mkdir(exist_ok=True)
    (OUT / "cache_eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for r in sweep:
        print(f"margin={r['margin']} tau={r['tau']:.2f} hit={r['hit_rate']:.2f} wrong/hit={r['wrong_answer_rate_of_hits']:.2f} false-hit={r['false_hit_rate_on_negatives']:.2f}")
    print("CHOSEN on dev:", chosen)
    print("HELD-OUT at chosen:", test)
    print("HELD-OUT at spec default 0.80/0.05:", spec_default)


if __name__ == "__main__":
    main()
