"""Ablation ladder A0..A7: each step adds one change. One process per step (clean RSS/CPU numbers).

  python -m bench.run_ablation --reps 3            (A0 = baseline result is reused if bench/out/baseline.json exists)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from bench.common import OUT

STEPS = [("A0", "baseline"), ("A1", "A1"), ("A2", "A2"), ("A3", "A3"), ("A4", "A4"), ("A5", "A5"), ("A6", "A6"), ("A7", "A7")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="re-run steps that already have results")
    ap.add_argument("--steps", default=",".join(s for s, _ in STEPS))
    args = ap.parse_args()
    for label, system in STEPS:
        if label not in args.steps.split(","):
            continue
        if (OUT / f"{system}.json").exists() and not args.force:
            print(f"{label}: reusing bench/out/{system}.json")
            continue
        subprocess.run([sys.executable, "-m", "bench.run_bench", "--system", system, "--reps", str(args.reps)], check=True)
    rows = {}
    for label, system in STEPS:
        f = OUT / f"{system}.json"
        if f.exists():
            rows[label] = json.loads(f.read_text())["summary"]
    (OUT / "ablation.json").write_text(json.dumps(rows, indent=1))
    for k, v in rows.items():
        print(f"{k}: p50={v['latency_ms']['p50']:.0f} p95={v['latency_ms']['p95']:.0f} rss={v['peak_rss_mb']:.0f} "
              f"cpu_s={v['cpu_s_per_turn']:.1f} quality={v['quality_pass_rate']:.2f}")


if __name__ == "__main__":
    main()
