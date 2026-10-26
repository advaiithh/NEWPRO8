"""S4 quality per tier: the 40 fixed prompts (20 skill + 20 open) at each fixed tier T0..T4, 4 cores / 4 GB.

Machine-checked rubric (keywords / regex). Blind 1-5 human ratings are collected separately (docs/LIVE_TEST.md).
"""
from __future__ import annotations

import json
import subprocess
import sys

from bench.common import OUT


def main() -> None:
    res = {}
    for tier in range(5):
        tag = f"tier{tier}"
        subprocess.run([sys.executable, "-m", "bench.run_bench", "--system", "FIXED", "--tier", str(tier), "--reps", "1",
                        "--subset", "q40", "--warmup-turns", "0", "--idle-s", "1", "--tag", tag, "--restart-ollama"], check=False)
        f = OUT / f"FIXED_{tag}.json"
        res[f"T{tier}"] = json.loads(f.read_text())["summary"] if f.exists() else {"missing": True}
    (OUT / "quality_by_tier.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, "quality", v.get("quality_pass_rate"), "p50", v.get("latency_ms", {}).get("p50"), "rss", v.get("peak_rss_mb"))


if __name__ == "__main__":
    main()
