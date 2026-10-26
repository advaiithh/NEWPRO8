"""Re-apply the current scoring rubric (requests.yaml / heldout.yaml) to saved results and the manifest.

Needed when the rubric changes after audio was synthesised: replies are stored in every result file, so no run is repeated.
"""
from __future__ import annotations

import json

import numpy as np
import yaml

from bench.common import HERE, OUT, TESTSET, quality_ok


def main() -> None:
    reqs = {r["id"]: r for f in ("requests.yaml", "heldout.yaml") for r in yaml.safe_load((HERE / f).read_text(encoding="utf-8"))}
    mf = TESTSET / "manifest.json"
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    for m in manifest:                                    # keep voice/speed/seconds, refresh the rubric fields
        for k in ("expect", "any", "intent", "text"):
            if k in reqs[m["id"]]:
                m[k] = reqs[m["id"]][k]
    mf.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    for f in sorted(OUT.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(d, dict) or "turns" not in d or not d["turns"]:
            continue
        for t in d["turns"]:
            if t.get("req_id") in reqs and t.get("reply") is not None:
                t["ok"] = quality_ok(reqs[t["req_id"]], t["reply"], (t.get("extra") or {}).get("intent", ""))
        scored = [t["ok"] for t in d["turns"] if t.get("kind") in ("skill", "open") and t.get("ok") is not None]
        old = d["summary"].get("quality_pass_rate")
        d["summary"]["quality_pass_rate"], d["summary"]["quality_n"] = (float(np.mean(scored)) if scored else float("nan")), len(scored)
        f.write_text(json.dumps(d, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
        print(f"{f.stem:22} quality {old if old is None else round(old, 2)} -> {d['summary']['quality_pass_rate']:.2f} (n={len(scored)})")


if __name__ == "__main__":
    main()
