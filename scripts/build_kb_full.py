"""Setup (online, long): build the offline knowledge base from the CURRENT Wikipedia export (omarkamali/wikipedia-monthly).

Streams every train shard of one language, keeps only a short lead per article, and discards the shard. Keeps the N longest
(most developed) articles plus every office-holder / government / election page (those carry "who is the current ...").
Output (models/kb_<lang>/): emb.npy (float16), titles.json, leads.sqlite, meta.json (snapshot date).

  python scripts/build_kb_full.py --lang en --n 700000
"""
from __future__ import annotations

import argparse
import heapq
import json
import pickle
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPO = "omarkamali/wikipedia-monthly"
SKIP_TITLE = re.compile(r"^(list of|lists of|timeline of|index of|outline of|deaths|births|events|\d{3,4}\b)|[:(]|disambiguation|\bin \d{4}$|\b\d{4}s?$", re.I)
ALWAYS = re.compile(r"^(chief minister|prime minister|president|vice[- ]president|governor|speaker|chief justice|secretary[- ]general|mayor|"
                    r"minister|deputy chief minister|leader of the opposition|chief election commissioner|attorney general)\b.*\b(of|for)\b|"
                    r"^(government of|cabinet of|legislative assembly|parliament of)\b|^20(2[3-9]|3\d) .*\b(election|elections|referendum)\b|"
                    r"\b(legislative assembly|lok sabha|rajya sabha)\b", re.I)
# a biography whose own lead says the person currently holds an office ("X is ... serving as the Chief Minister of Y since 2026"):
# new office-holders have short articles, so they must be kept regardless of length
HOLDER_LEAD = re.compile(r"\b(is|are|has been|has served|serves|serving|currently)\b[^.]{0,160}?\b(prime minister|chief minister|deputy chief minister|"
                         r"minister|president|vice[- ]president|governor|lieutenant governor|speaker|mayor|chancellor|chief justice|"
                         r"secretary[- ]general|ambassador|king|queen|monarch|chief secretary|chief election commissioner)\b[^.]{0,8}?\bof\b[^.]{0,80}?\b(since|currently)\b|"
                         r"\b(currently|incumbent)\b[^.]{0,80}?\b(prime minister|chief minister|president|governor|speaker|mayor|chancellor|chief justice|secretary[- ]general)\b[^.]{0,8}?\bof\b", re.I)
DEFINING = re.compile(r"\b(is|are|was|were|refers to|means|are called|serves|served|held|won|elected|sworn)\b")


def lead_of(text: str) -> str:
    first = next((p for p in text.split("\n") if len(p.strip()) > 60), "")
    first = re.sub(r"\s*\([^)]*\)", "", first)
    first = re.sub(r"\s+", " ", first).strip()
    out = ""
    for s in re.split(r"(?<=[.!?])\s+", first)[:4]:
        if len(out) + len(s) > 560:
            break
        out = f"{out} {s}".strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--n", type=int, default=700000)
    ap.add_argument("--date", default="20261001")
    ap.add_argument("--max-shards", type=int, default=0, help="debug: only the first N shards")
    args = ap.parse_args()
    api = HfApi()
    shards = sorted(t.path for t in api.list_repo_tree(REPO, path_in_repo=f"{args.date}/{args.lang}/train", repo_type="dataset")
                    if t.path.endswith(".parquet"))
    if args.max_shards:
        shards = shards[: args.max_shards]
    print(f"{len(shards)} shards of {args.date}/{args.lang}", flush=True)
    heap: list[tuple[int, str, str]] = []                      # (length, title, lead): smallest on top
    always: dict[str, str] = {}
    seen = kept = 0
    t0 = time.time()

    def fetch(path: str) -> str:
        for attempt in range(8):                                # the CDN drops a request now and then: back off and retry
            try:
                return hf_hub_download(REPO, path, repo_type="dataset", local_dir=ROOT / "data" / "wm_tmp")
            except Exception as e:
                print(f"  download retry {attempt + 1} for {path.split('/')[-1]}: {type(e).__name__}", flush=True)
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"could not download {path}")

    ckpt = ROOT / "data" / f"kb_ckpt_{args.lang}.pkl"
    start = 0
    if ckpt.exists():                                           # resume after an interruption
        start, heap, always, seen = pickle.loads(ckpt.read_bytes())
        print(f"resuming at shard {start} ({len(heap):,} kept so far)", flush=True)

    with ThreadPoolExecutor(3) as pool:
        futures = {i: pool.submit(fetch, shards[i]) for i in range(start, min(start + 3, len(shards)))}
        for k in range(start, len(shards)):
            local = Path(futures.pop(k).result())
            if k + 3 < len(shards):
                futures[k + 3] = pool.submit(fetch, shards[k + 3])
            pf = pq.ParquetFile(local)
            for g in range(pf.num_row_groups):
                tbl = pf.read_row_group(g, columns=["title", "text"])
                for title, text in zip(tbl["title"].to_pylist(), tbl["text"].to_pylist()):
                    seen += 1
                    if not title or not text:
                        continue
                    prio = bool(ALWAYS.search(title)) or bool(HOLDER_LEAD.search(text[:900].replace("\n", " ")))
                    if SKIP_TITLE.search(title) and not prio:
                        continue
                    if not prio and len(heap) >= args.n and len(text) <= heap[0][0]:
                        continue                                # cheap early exit: shorter than the current cut-off
                    lead = lead_of(text)
                    if len(lead) < 80 or not DEFINING.search(re.split(r"(?<=[.!?])\s+", lead)[0]):
                        continue
                    if prio:
                        always[title] = lead
                    elif len(heap) < args.n:
                        heapq.heappush(heap, (len(text), title, lead))
                    else:
                        heapq.heapreplace(heap, (len(text), title, lead))
            pf.close()                                  # Windows cannot delete a file that is still open
            local.unlink(missing_ok=True)
            if (k + 1) % 50 == 0:
                ckpt.write_bytes(pickle.dumps((k + 1, heap, always, seen)))
            if (k + 1) % 25 == 0 or k + 1 == len(shards):
                print(f"shard {k + 1}/{len(shards)}  seen {seen:,}  kept {len(heap) + len(always):,}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)

    entries = {t: l for _, t, l in heap}
    entries.update(always)
    titles = sorted(entries)
    out = ROOT / "models" / f"kb_{args.lang}"
    out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(out / "leads.sqlite")
    db.execute("DROP TABLE IF EXISTS leads")
    db.execute("CREATE TABLE leads (i INTEGER PRIMARY KEY, lead TEXT)")
    db.executemany("INSERT INTO leads VALUES (?, ?)", ((i, entries[t]) for i, t in enumerate(titles)))
    db.commit()
    db.close()
    (out / "titles.json").write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")
    print(f"embedding {len(titles):,} articles ...", flush=True)
    from model2vec import StaticModel
    model = StaticModel.from_pretrained(str(ROOT / "models" / "embed" / "potion-base-8M"))
    vecs = []
    for s in range(0, len(titles), 8192):
        v = model.encode([f"{t}. {entries[t]}" for t in titles[s:s + 8192]])
        vecs.append((v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)).astype(np.float16))
    np.save(out / "emb.npy", np.concatenate(vecs))
    (out / "meta.json").write_text(json.dumps({"snapshot": args.date, "lang": args.lang, "articles": len(titles),
                                               "always_kept": len(always), "source": REPO}), encoding="utf-8")
    ckpt.unlink(missing_ok=True)
    print(f"done: {len(titles):,} articles ({len(always):,} office/government/election pages) in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
