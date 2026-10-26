"""Generate reports/report.md, reports/summary.json, charts, and reports/WRITEUP.md from bench/out/*.json only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench.common import OUT  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, GRID = "#0b0b0b", "#e1e0d9"
LABELS = {"A0": "A0 baseline", "A1": "A1 +Moonshine stream", "A2": "A2 +sentence TTS", "A3": "A3 +voice prompt/caps",
          "A4": "A4 +prefix warm", "A5": "A5 +adaptive endpoint", "A6": "A6 +router", "A7": "A7 +governor"}


def load(name: str) -> Any:
    f = OUT / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def style(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)


def fig_ladder(abl: dict) -> None:
    keys = [k for k in LABELS if k in abl]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=130)
    x = range(len(keys))
    p50 = [abl[k]["latency_ms"]["p50"] / 1000 for k in keys]
    p95 = [abl[k]["latency_ms"]["p95"] / 1000 for k in keys]
    ax.bar([i - 0.2 for i in x], p50, 0.38, color=C[0], label="p50")
    ax.bar([i + 0.2 for i in x], p95, 0.38, color=C[1], label="p95")
    for i, (a, b) in enumerate(zip(p50, p95)):
        ax.text(i - 0.2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.text(i + 0.2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(list(x), [LABELS[k] for k in keys], rotation=20, ha="right")
    ax.set_ylabel("end of speech to first audio (s)", fontsize=9)
    ax.set_title("Ablation ladder: latency", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(REPORTS / "ablation_latency.png")
    plt.close(fig)


def stage_means(name: str) -> dict[str, float]:
    d = load(f"{name}.json")
    acc: dict[str, list[float]] = {}
    for t in d["turns"]:
        for k, v in (t.get("stages_ms") or {}).items():
            acc.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def fig_waterfall(abl: dict, systems: dict[str, str]) -> None:
    keys = [k for k in LABELS if k in abl]
    stages: dict[str, list[float]] = {}
    for k in keys:
        sm = stage_means(systems[k])
        hang = abl[k]["_hangover_ms"]
        parts = {"endpoint hangover": hang, "ASR finalise": sm.get("speech_end->asr_final", 0) - 0,
                 "route": sm.get("asr_final->route_decision", 0),
                 "LLM first token": sm.get("route_decision->llm_first_token", 0),
                 "LLM remainder + TTS": sm.get("llm_first_token->tts_first_chunk_ready", 0),
                 "skill / cache TTS": sm.get("route_decision->tts_first_chunk_ready", 0)}
        for n, v in parts.items():
            stages.setdefault(n, []).append(max(v, 0) / 1000)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=130)
    bottom = [0.0] * len(keys)
    for i, (n, vals) in enumerate(stages.items()):
        ax.bar(range(len(keys)), vals, 0.6, bottom=bottom, color=C[i % 8], label=n, edgecolor="#fcfcfb", linewidth=1.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(keys)), [LABELS[k] for k in keys], rotation=20, ha="right")
    ax.set_ylabel("mean seconds after speech ends", fontsize=9)
    ax.set_title("Where the time goes (mean per stage)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=7, ncol=3)
    style(ax)
    fig.tight_layout()
    fig.savefig(REPORTS / "ablation_waterfall.png")
    plt.close(fig)


def fig_degrade(deg: dict) -> None:
    steps = list(next(iter(deg.values())).keys())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=130)
    for ax, (key, title, scale) in zip(axes, [("p50", "latency p50 (s)", 1000), ("completion_rate", "completion rate", 0.01),
                                              ("quality_pass_rate", "quality (keyword pass rate)", 0.01)]):
        for i, (sys_, name) in enumerate([("FIXED", "fixed Full config"), ("FULL", "with governor")]):
            ys = []
            for s in steps:
                v = deg[sys_].get(s, {})
                y = v.get("latency_ms", {}).get(key) if key == "p50" else v.get(key)
                ys.append(float("nan") if y is None else (y / scale if key == "p50" else y))
            ax.plot(range(len(steps)), ys, marker="o", color=C[i], lw=2, label=name)
        ax.set_xticks(range(len(steps)), steps, rotation=25, fontsize=7)
        ax.set_title(title, loc="left", fontsize=10)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "degradation.png")
    plt.close(fig)


def fig_quant(q: dict) -> None:
    pts = [(k, v) for k, v in q["llm"].items() if "tok_s_mean" in v]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=130)
    for i, (k, v) in enumerate(pts):
        ax.scatter(v["first_token_ms_p50"], v["quality"], s=v["rss_mb"] / 3, color=C[i % 8], edgecolor="#fcfcfb", lw=1.5)
        ax.annotate(f"{k}\n{v['tok_s_mean']:.0f} tok/s, {v['rss_mb']:.0f} MB", (v["first_token_ms_p50"], v["quality"]),
                    fontsize=7, xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("first-token latency p50 (ms)", fontsize=9)
    ax.set_ylabel("answer quality (keyword pass rate)", fontsize=9)
    ax.set_title("LLM quantization Pareto (bubble = RSS)", loc="left", fontsize=11)
    style(ax)
    fig.tight_layout()
    fig.savefig(REPORTS / "quant_pareto.png")
    plt.close(fig)


def fig_cache(cv: dict) -> None:
    rows = [r for r in cv["sweep_dev"] if r["margin"] == cv["chosen_on_dev"]["margin"]]
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=130)
    for i, (k, n) in enumerate([("hit_rate", "hit rate (paraphrases)"), ("wrong_answer_rate_of_hits", "wrong answers among hits"),
                                ("false_hit_rate_on_negatives", "false hits on out-of-domain")]):
        ax.plot([r["tau"] for r in rows], [r[k] for r in rows], marker="o", color=C[i], lw=2, label=n)
    ax.axvline(cv["chosen_on_dev"]["tau"], color="#898781", ls="--", lw=1)
    ax.set_xlabel("similarity threshold tau", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Semantic cache: effect of tau", loc="left", fontsize=11)
    style(ax)
    fig.tight_layout()
    fig.savefig(REPORTS / "cache_tau.png")
    plt.close(fig)


def f(x: Any, d: int = 0) -> str:
    return "n/a" if x is None or x != x else f"{x:,.{d}f}"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    systems = {"A0": "baseline", **{k: k for k in ("A1", "A2", "A3", "A4", "A5", "A6", "A7")}}
    abl = {}
    for label, name in systems.items():
        d = load(f"{name}.json")
        if d:
            abl[label] = d["summary"]
            hs = [t["hangover_ms"] for t in d["turns"] if t.get("hangover_ms")]
            abl[label]["_hangover_ms"] = sum(hs) / len(hs) if hs else 0.0
    deg, quant, cache, qual = load("degrade.json"), load("quant_sweep.json"), load("cache_eval.json"), load("quality_by_tier.json")
    md: list[str] = ["# PocketVoice benchmark report", "",
                     "Generated by `bench/make_report.py` from `bench/out/*.json`. Every number below is read from those files.", ""]
    if abl:
        base = abl.get("A0")
        md += ["## 1. Baseline vs ladder (4 cores / 4096 MB, 60 spoken requests x 3 repeats, warm-up discarded)", "",
               "| Step | p50 ms | p95 ms | mean ms | worst ms | peak RSS MB | CPU-s/turn | J/turn (RAPL, net of idle) | J/turn (proxy) | WER | quality | LLM avoided |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in abl.items():
            n = v["n_turns"] or 1
            avoided = sum(v["paths"].get(p, 0) for p in ("skill", "cache")) / n
            md.append(f"| {LABELS[k]} | {f(v['latency_ms']['p50'])} | {f(v['latency_ms']['p95'])} | {f(v['latency_ms']['mean'])} | "
                      f"{f(v['latency_ms']['worst'])} | {f(v['peak_rss_mb'])} | {f(v['cpu_s_per_turn'], 1)} | "
                      f"{f(v['energy_j_per_turn_net_of_idle'], 1)} | {f(v['energy_proxy_j_per_turn'], 1)} | {f(v['wer_mean'], 3)} | "
                      f"{f(v['quality_pass_rate'], 2)} | {avoided:.0%} |")
        v0 = base
        md += ["", f"Energy source: {v0['energy_source']}. The proxy column is cpu-seconds x {v0['assumed_w_per_core_for_proxy']} W/core "
               "(an assumption, not a measurement). RAPL is whole-package power and includes background load, so it is reported "
               "net of the idle draw measured immediately before each run.", "",
               "Idle (always-on front end only): " + "; ".join(f"{LABELS[k]}: {f(v['idle_cpu_pct'], 1)}% CPU, {f(v['idle_rss_mb'])} MB RSS"
                                                                  for k, v in abl.items() if k in ("A0", "A7") or k == list(abl)[-1]), "",
               "![latency](ablation_latency.png)", "", "![waterfall](ablation_waterfall.png)", ""]
        fig_ladder(abl)
        fig_waterfall(abl, systems)
    if deg:
        md += ["## 2. Degradation: fixed Full config vs governor", "", "| Limit | config | completion | p50 ms | quality |", "|---|---|---|---|---|"]
        for sys_ in ("FIXED", "FULL"):
            for step, v in deg[sys_].items():
                md.append(f"| {step} | {'fixed' if sys_ == 'FIXED' else 'governor'} | {f(v.get('completion_rate', 0) * 100)}% | "
                          f"{f(v.get('latency_ms', {}).get('p50'))} | {f(v.get('quality_pass_rate'), 2)} "
                          f"{('(startup failed)' if v.get('startup_failed') else '')} |")
        md += ["", "![degradation](degradation.png)", ""]
        fig_degrade(deg)
    if quant:
        md += ["## 3. Quantization and offload", "", "| LLM | tok/s | first token ms (p50) | RSS MB | USS MB | quality (20 open prompts) |", "|---|---|---|---|---|---|"]
        for k, v in quant["llm"].items():
            md.append(f"| {k} | {f(v.get('tok_s_mean'), 1)} | {f(v.get('first_token_ms_p50'))} | {f(v.get('rss_mb'))} | {f(v.get('uss_mb'))} | {f(v.get('quality'), 2)} |"
                      if "tok_s_mean" in v else f"| {k} | not run ({v.get('missing')}) | | | | |")
        md += ["", "| KV cache / flash attention | tok/s | first token ms | RSS MB |", "|---|---|---|---|"]
        for k, v in quant["kv"].items():
            md.append(f"| {k} | {f(v['tok_s_mean'], 1)} | {f(v['first_token_ms_p50'])} | {f(v['rss_mb'])} |")
        md += ["", "| ASR | WER | RTF | process RSS MB |", "|---|---|---|---|"]
        for k, v in quant["asr"].items():
            md.append(f"| {k} | {f(v['wer'], 3)} | {f(v['rtf'], 2)} | {f(v['rss_mb'])} |")
        md += ["", "| TTS | RTF | Whisper-WER of output |", "|---|---|---|"]
        for k, v in quant["tts"].items():
            md.append(f"| {k} | {f(v['rtf'], 3)} | {f(v['whisper_wer_intelligibility'], 3)} |")
        o = quant["offload"]
        md += ["", f"Offload evidence: cold load {f(o['cold_load_s'], 2)} s, unload {f(o['unload_s'], 2)} s "
               f"(models resident afterwards: {o['models_loaded_after_unload']}); resident runner RSS {f(o['resident']['rss_mb'])} MB vs "
               f"private (USS) {f(o['resident']['uss_mb'])} MB (weights are loaded into private memory, not left memory-mapped); "
               f"pinned {f(o['pinning']['pinned_4_cores_tok_s'], 1)} tok/s vs unpinned {f(o['pinning']['unpinned_all_12_threads_tok_s'], 1)} tok/s.",
               "", "![pareto](quant_pareto.png)", ""]
        fig_quant(quant)
    if qual:
        md += ["## 4. Quality per tier (40 fixed prompts, 4c/4GB)", "", "| Tier | quality | p50 ms | peak RSS MB |", "|---|---|---|---|"]
        for k, v in qual.items():
            md.append(f"| {k} | {f(v.get('quality_pass_rate'), 2)} | {f(v.get('latency_ms', {}).get('p50'))} | {f(v.get('peak_rss_mb'))} |")
        md.append("")
    if cache:
        md += ["## 5. Cache honesty", "", "Dev set (tau and margin are chosen here only):", "",
               "| margin | tau | hit rate | wrong answers among hits | false hits on out-of-domain |", "|---|---|---|---|---|"]
        for r in cache["sweep_dev"]:
            md.append(f"| {r['margin']} | {r['tau']:.2f} | {r['hit_rate']:.2f} | {r['wrong_answer_rate_of_hits']:.2f} | {r['false_hit_rate_on_negatives']:.2f} |")
        h, d = cache["heldout_at_chosen"], cache["heldout_at_spec_default_0.80"]
        md += ["", f"Chosen on dev: tau={cache['chosen_on_dev']['tau']:.2f}, margin={cache['chosen_on_dev']['margin']} ({cache['selection_rule']}).",
               f"Held-out (n={h['n_pos']} positives, {h['n_neg']} negatives) at the chosen setting: hit {h['hit_rate']:.2f}, wrong {h['wrong_answer_rate_of_hits']:.2f}, "
               f"false hits {h['false_hit_rate_on_negatives']:.2f}; at the spec default 0.80/0.05: hit {d['hit_rate']:.2f}, wrong {d['wrong_answer_rate_of_hits']:.2f}.",
               "", "Failure cases at the chosen setting (held-out first):", ""]
        md += [f"- `{x['text']}` expected `{x['expected']}`, got `{x['got']}` (sim {x['sim']}, margin {x['margin']})"
               for x in (cache["failures_heldout_at_chosen"] + cache["failures_dev_at_chosen"])[:14]]
        md.append("\n![tau](cache_tau.png)\n")
        fig_cache(cache)
    (REPORTS / "report.md").write_text("\n".join(md), encoding="utf-8")
    if "A0" in abl and len(abl) > 1:
        from bench.writeup import build
        (REPORTS / "WRITEUP.md").write_text(build(abl, deg, quant, cache, qual, load("proofs.json")), encoding="utf-8")

    rows = [{"system": LABELS[k], "p50": v["latency_ms"]["p50"], "p95": v["latency_ms"]["p95"], "rss": v["peak_rss_mb"],
             "cpu_s": v["cpu_s_per_turn"]} for k, v in abl.items() if k in ("A0", "A7")]
    (REPORTS / "summary.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    print("wrote reports/report.md, summary.json and charts")


if __name__ == "__main__":
    main()
