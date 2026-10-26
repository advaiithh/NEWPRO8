"""Setup: build the offline knowledge base from the downloaded Simple English Wikipedia dump.

Keeps the N longest (most developed) articles, stores a short lead per article and a static embedding of
"title. lead" (float16 memmap). Runtime loads only these files.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model2vec import StaticModel  # noqa: E402

KB = ROOT / "models" / "kb"
SKIP_TITLE = re.compile(r"^(list of|lists of|timeline of|index of|deaths|births|events|\d{3,4}\b)|[:(]|disambiguation|\bin \d{4}$|\b\d{4}s?$", re.I)
DEFINING = re.compile(r"\b(is|are|was|were|refers to|means|are called)\b")      # first sentence reads like a definition


def lead_of(text: str) -> str:
    first = next((p for p in text.split("\n") if len(p.strip()) > 60), "")
    first = re.sub(r"\s*\([^)]*\)", "", first)                    # pronunciations, dates in brackets
    first = re.sub(r"\s+", " ", first).strip()
    sentences = re.split(r"(?<=[.!?])\s+", first)
    out = ""
    for s in sentences[:3]:
        if len(out) + len(s) > 420:
            break
        out = f"{out} {s}".strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160000)
    args = ap.parse_args()
    table = pq.read_table(KB / "raw" / "20231101.simple" / "train-00000-of-00001.parquet", columns=["title", "text"])
    titles, texts = table["title"].to_pylist(), table["text"].to_pylist()
    order = sorted(range(len(titles)), key=lambda i: -len(texts[i]))
    keep_t, keep_l = [], []
    for i in order:
        if SKIP_TITLE.search(titles[i]):
            continue
        lead = lead_of(texts[i])
        if len(lead) < 80 or not DEFINING.search(re.split(r"(?<=[.!?])\s+", lead)[0]):
            continue
        keep_t.append(titles[i])
        keep_l.append(lead)
        if len(keep_t) >= args.n:
            break
    model = StaticModel.from_pretrained(str(ROOT / "models" / "embed" / "potion-base-8M"))
    vecs = []
    for s in range(0, len(keep_t), 4096):
        v = model.encode([f"{t}. {l}" for t, l in zip(keep_t[s:s + 4096], keep_l[s:s + 4096])])
        vecs.append(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9))
    np.save(KB / "emb.npy", np.concatenate(vecs).astype(np.float16))      # half precision: 82 MB for 160k articles
    (KB / "titles.json").write_text(json.dumps(keep_t, ensure_ascii=False), encoding="utf-8")
    (KB / "leads.json").write_text(json.dumps(keep_l, ensure_ascii=False), encoding="utf-8")
    print(f"knowledge base: {len(keep_t)} articles from {len(titles)}; example: {keep_t[0]!r}: {keep_l[0][:100]!r}")


if __name__ == "__main__":
    main()
