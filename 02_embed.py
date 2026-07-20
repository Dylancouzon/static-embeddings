"""Stage 02 — embed corpora + queries (quality vectors, cached to disk) and measure
embed-only speed per system. Idempotent: skips vectors/speed files already written.

Vectors: results/vectors/<dataset>__<run_id>.{corpus,queries}.npy + .*_ids.json
Speed:   results/speed/<run_id>.json  (throughput, cold-start, latency, RSS, env)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

import datasets_io
import env
import manifest
import speed
import systems

VEC = Path("results/vectors")
SPD = Path("results/speed")


def embed_vectors():
    VEC.mkdir(parents=True, exist_ok=True)
    bs = manifest.qdrant()["upload_batch"]
    for e in manifest.dense_embedders():  # BM25 is sparse -> handled in 03/04
        embedder = None
        for ds in manifest.all_datasets():
            corpus, queries, _ = datasets_io.load(ds)
            for kind_name, items in (("corpus", corpus), ("queries", queries)):
                npy, idj = manifest.vec_paths(ds, e["run_id"], kind_name)
                if npy.exists():
                    continue
                if embedder is None:
                    print(f"building {e['run_id']} ...")
                    embedder = systems.build(e["kind"], e["model_id"])
                ids = list(items.keys())
                texts = [items[i] for i in ids]
                print(f"  embed {ds} {kind_name}: {len(texts)} docs")
                np.save(npy, embedder.embed(texts, batch_size=bs).astype(np.float32))
                idj.write_text(json.dumps(ids))


def measure_speed():
    # speed numbers are only comparable under pinned threads; refuse to run unpinned
    # (a stray standalone call would default ORT to 1 and pollute results/speed/).
    for var in ("OMP_NUM_THREADS", "ORT_INTRA_OP_THREADS"):
        if var not in os.environ:
            raise SystemExit(f"{var} unset — run via run_all.py so threads are pinned "
                             f"(speed results would be incomparable otherwise).")
    SPD.mkdir(parents=True, exist_ok=True)
    sp = manifest.speed()
    e_env = env.capture()
    # speed corpus: largest available (code track), fallback to first BEIR
    speed_ds = manifest.datasets()["code"][0]
    try:
        corpus, queries, _ = datasets_io.load(speed_ds)
    except FileNotFoundError:
        speed_ds = manifest.datasets()["beir"][0]
        corpus, queries, _ = datasets_io.load(speed_ds)
    corpus_texts = list(corpus.values())
    query_texts = list(queries.values())

    for e in manifest.embedders():
        out = SPD / f"{e['run_id']}.json"
        if out.exists():
            continue
        print(f"speed: {e['run_id']} (on {speed_ds})")
        embedder = systems.build(e["kind"], e["model_id"])
        emb = embedder.embed_sparse if e["kind"] == "bm25" else embedder.embed
        r = {
            "run_id": e["run_id"], "kind": e["kind"], "model_id": e["model_id"],
            "speed_dataset": speed_ds, "env": e_env,
            "throughput": speed.throughput(emb, corpus_texts, sp["throughput_batch_sizes"], sp["repeats"]),
            "query_latency": speed.query_latency(emb, query_texts, sp["query_latency_n"]),
            "peak_rss": speed.peak_rss_mb(emb, corpus_texts),
            "cold_start": speed.cold_start(e["kind"], e["model_id"], corpus_texts, sp["coldstart_processes"]),
        }
        out.write_text(json.dumps(r, indent=2))
        tp32 = r["throughput"].get("32", {}).get("median")
        print(f"  throughput@32={tp32:.0f} docs/s  load={r['cold_start']['model_load']['median']*1000:.0f}ms")

    # load/size-only models (multilingual): cold-start only, no quality
    for e in manifest.loadonly_specs():
        out = SPD / f"{e['id']}__{e['model']}.loadonly.json"
        if out.exists():
            continue
        print(f"speed(loadonly): {e['id']} {e['model']}")
        r = {"run_id": f"{e['id']}__{e['model']}", "kind": e["kind"], "model_id": e["model_id"],
             "env": e_env,
             "cold_start": speed.cold_start(e["kind"], e["model_id"], corpus_texts, sp["coldstart_processes"])}
        out.write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    if "--speed-only" not in sys.argv:
        embed_vectors()
    if "--vectors-only" not in sys.argv:
        measure_speed()
    print("stage 02 done")
