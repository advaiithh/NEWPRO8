"""Does the offline knowledge base help? 60 multi-area questions, text in -> spoken text out (no audio), same LLM settings.

  llm-only : voice prompt, 1.5B Q4, no retrieval
  with-kb  : definition questions read from the article lead; other facts get the best passage as LLM context
"""
from __future__ import annotations

import json
import subprocess
import time

import yaml

from bench.common import HERE, OUT
from pv import config as cfgmod
from pv.knowledge.kb import Knowledge
from pv.llm import prompts
from pv.llm.ollama_client import OllamaClient, OllamaOptions


def ask_llm(c: OllamaClient, opts: OllamaOptions, q: str, context: str = "") -> tuple[str, float]:
    user = (f"Facts: {context}\nQuestion: {q}\nAnswer briefly. Use the facts only if they are relevant to the question." if context else q)
    t0 = time.perf_counter()
    reply = "".join(c.stream([{"role": "system", "content": prompts.SYSTEM_FULL}, {"role": "user", "content": user}], options=opts)).strip()
    return reply, time.perf_counter() - t0


def main() -> None:
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/launch_ollama.ps1", "-IfDown"],
                   check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    qs = yaml.safe_load((HERE / "knowledge_qa.yaml").read_text(encoding="utf-8"))
    kb = Knowledge()
    c = OllamaClient()
    opts = OllamaOptions(cfgmod.config["ollama"]["full_model"], 1024, 80)
    ask_llm(c, opts, "hello")
    rows, by_area = [], {}
    for q in qs:
        base, t_base = ask_llm(c, opts, q["text"])
        kind, hit, snippet = kb.answer(q["text"])
        if kind == "extract":
            ans, t_kb = snippet, 0.005
        else:
            ans, t_kb = ask_llm(c, opts, q["text"], snippet if kind == "context" else "")
        ok = lambda r: any(str(k).lower() in r.lower() for k in q["any"])
        rows.append({"id": q["id"], "area": q["area"], "q": q["text"], "kind": kind, "title": hit.title if hit else None,
                     "llm_ok": ok(base), "kb_ok": ok(ans), "llm_s": t_base, "kb_s": t_kb, "llm": base, "kb": ans})
        a = by_area.setdefault(q["area"], [0, 0, 0])
        a[0] += ok(base)
        a[1] += ok(ans)
        a[2] += 1
        print(f"{q['id']} {kind:8} llm={'Y' if ok(base) else '-'} kb={'Y' if ok(ans) else '-'}  {q['text']}", flush=True)
    n = len(rows)
    summary = {"n": n, "llm_only_accuracy": sum(r["llm_ok"] for r in rows) / n, "with_kb_accuracy": sum(r["kb_ok"] for r in rows) / n,
               "kb_extract_share": sum(r["kind"] == "extract" for r in rows) / n, "kb_context_share": sum(r["kind"] == "context" for r in rows) / n,
               "llm_mean_s": sum(r["llm_s"] for r in rows) / n, "kb_mean_s": sum(r["kb_s"] for r in rows) / n,
               "by_area": {k: {"llm_only": v[0] / v[2], "with_kb": v[1] / v[2], "n": v[2]} for k, v in by_area.items()}}
    OUT.mkdir(exist_ok=True)
    (OUT / "knowledge_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
