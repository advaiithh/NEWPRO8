"""Degradation test (SPEC 10.4): same 40 requests while the limit steps down; fixed Full config vs governor.

Each (config, step) is a fresh process with Ollama restarted under that limit, so the OS-enforced cap is real.
  python -m bench.run_degrade [--n 40]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from bench.common import OUT

STEPS = [(6, 6144), (4, 4096), (3, 3072), (2, 2048), (2, 1536)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--steps", default=None, help="e.g. 4:4096,2:2048,2:1536 (cores:MB)")
    args = ap.parse_args()
    steps = [tuple(map(int, x.split(":"))) for x in args.steps.split(",")] if args.steps else STEPS
    res = {}
    for system in ("FIXED", "FULL"):
        for cores, ram in steps:
            tag = f"deg_{cores}c{ram}"
            subprocess.run([sys.executable, "-m", "bench.run_bench", "--system", system, "--reps", "1",
                            "--subset", "q40", "--limit-n", str(args.n), "--cores", str(cores), "--ram", str(ram),
                            "--restart-ollama", "--warmup-turns", "0", "--idle-s", "1", "--tag", tag,
                            "--llm-timeout", str(args.timeout)], check=False)
            f = OUT / f"{system}_{tag}.json"
            res.setdefault(system, {})[f"{cores}c/{ram}MB"] = json.loads(f.read_text())["summary"] if f.exists() else {"missing": True}
    (OUT / "degrade.json").write_text(json.dumps(res, indent=1))
    for system, steps in res.items():
        for k, v in steps.items():
            lat = v.get("latency_ms", {})
            print(f"{system:5} {k:12} completion={v.get('completion_rate', 0):.2f} p50={lat.get('p50', float('nan')):.0f} "
                  f"quality={v.get('quality_pass_rate', float('nan')):.2f} {v.get('startup_failed', '')}")


if __name__ == "__main__":
    main()
