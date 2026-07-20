"""Stage 01 — download BEIR + CodeSearchNet to data/ as corpus/queries/qrels.
Idempotent: skips a dataset whose three files already exist. Re-download with --force."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import datasets_io
import manifest


def main(force: bool = False):
    counts = {}
    for name in manifest.all_datasets():
        out = datasets_io.DATA / name
        done = all((out / f).exists() for f in ("corpus.jsonl", "queries.jsonl", "qrels.trec"))
        if done and not force:
            print(f"{name}: cached, skipping")
            corpus, queries, qrels = datasets_io.load(name)
            counts[name] = {"corpus": len(corpus), "queries": len(queries),
                            "judgments": sum(len(v) for v in qrels.values())}
            continue
        print(f"{name}: downloading...")
        c, q, j = datasets_io.download(name)
        counts[name] = {"corpus": c, "queries": q, "judgments": j}
        print(f"{name}: corpus={c} queries={q} judgments={j}")
    Path("results").mkdir(exist_ok=True)
    Path("results/data_counts.json").write_text(json.dumps(counts, indent=2))
    print("\n", json.dumps(counts, indent=2))


if __name__ == "__main__":
    main(force="--force" in sys.argv)
