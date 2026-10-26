"""Which LLM answers correctly? Same prompt/options for every model; CPU only; also speed, memory and VRAM.

  python -m bench.llm_compare [model ...]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import psutil
import yaml

from bench.common import HERE, OUT
from pv.llm import prompts
from pv.llm.ollama_client import OllamaClient, OllamaOptions

DEFAULT = ["qwen2.5:1.5b-instruct", "qwen2.5:3b-instruct"]


def runner_rss() -> float:
    runners = [p for p in psutil.process_iter(["name"]) if "llama" in (p.info["name"] or "").lower()]
    return max((p.memory_info().rss for p in runners), default=0) / 2**20


def main() -> None:
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/launch_ollama.ps1", "-IfDown"],
                   check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    qs = yaml.safe_load((HERE / "llm_hard_qa.yaml").read_text(encoding="utf-8"))
    c = OllamaClient()
    res = {}
    for model in (sys.argv[1:] or DEFAULT):
        opts = OllamaOptions(model, 1024, 80, think=False if "qwen3" in model else None)
        c.warm([{"role": "system", "content": prompts.SYSTEM_FULL}, {"role": "user", "content": "Hello"}], opts)
        rows = []
        for q in qs:
            t = time.perf_counter()
            reply = "".join(c.stream([{"role": "system", "content": prompts.SYSTEM_FULL}, {"role": "user", "content": q["text"]}], options=opts)).strip()
            rows.append({"id": q["id"], "q": q["text"], "reply": reply, "ok": any(str(k).lower() in reply.lower() for k in q["any"]),
                         "first_token_ms": c.stats.first_token_s * 1000, "tok_s": c.stats.tokens_per_s, "total_s": time.perf_counter() - t})
        ps = c.ps()
        res[model] = {"accuracy": sum(r["ok"] for r in rows) / len(rows), "n": len(rows),
                      "first_token_ms_p50": sorted(r["first_token_ms"] for r in rows)[len(rows) // 2],
                      "tok_s": sum(r["tok_s"] for r in rows) / len(rows), "runner_rss_mb": runner_rss(),
                      "size_vram": [m["size_vram"] for m in ps], "rows": rows}
        print(f"{model}: accuracy {res[model]['accuracy']:.2f}  first token p50 {res[model]['first_token_ms_p50']:.0f} ms  "
              f"{res[model]['tok_s']:.1f} tok/s  RSS {res[model]['runner_rss_mb']:.0f} MB  vram {res[model]['size_vram']}", flush=True)
        c.unload(model)
    OUT.mkdir(exist_ok=True)
    (OUT / "llm_compare.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    base = res[next(iter(res))]["rows"]
    for m, r in list(res.items())[1:]:
        for a, b in zip(base, r["rows"]):
            if a["ok"] != b["ok"]:
                print(("BETTER " if b["ok"] else "WORSE  "), m.split(":")[1], a["q"], "|", a["reply"][:50].replace("\n", " "), "=>", b["reply"][:50].replace("\n", " "))


if __name__ == "__main__":
    main()
