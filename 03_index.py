"""Stage 03 — build Qdrant collections from cached vectors and measure end-to-end
ingest. Dense collections from 02's npy; BM25 sparse computed here; hybrid = named
dense+sparse in one collection for server-side RRF (built in 04).

Collections: <dataset>__<run_id>. Point id = int index; payload.doc_id = original id.
Ingest timing: embed+upsert wall at batch 128 / 2 parallel streams, one config, on
the code dataset (results/ingest/<run_id>.json).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from qdrant_client import models

import datasets_io
import manifest
import qc
import systems

ING = Path("results/ingest")


def _load_vecs(dataset: str, run_id: str, kind: str):
    npy, idj = manifest.vec_paths(dataset, run_id, kind)
    return np.load(npy), json.loads(idj.read_text())


def _dense_points(vecs, ids):
    for i, (v, did) in enumerate(zip(vecs, ids)):
        yield models.PointStruct(id=i, vector=v.tolist(), payload={"doc_id": did})


def _bm25_corpus(dataset: str):
    """(sparse_embeddings, doc_ids) for the corpus via fastembed BM25."""
    corpus, _, _ = datasets_io.load(dataset)
    ids = list(corpus.keys())
    sparse = systems.build("bm25").embed_sparse([corpus[i] for i in ids])
    return sparse, ids


def build_collections():
    c = qc.client()
    up = manifest.qdrant()
    # dense (D1/D2/A/B/C × datasets)
    for e in manifest.dense_embedders():
        for ds in manifest.all_datasets():
            name = manifest.collection(ds, e["run_id"])
            vecs, ids = _load_vecs(ds, e["run_id"], "corpus")
            qc.recreate_dense(c, name, e["dim"])
            qc.bulk_upload(c, name, _dense_points(vecs, ids), up["upload_batch"], up["upload_parallel"])
            print(f"dense {name}: {len(ids)} pts")
    # BM25 sparse × datasets
    for ds in manifest.all_datasets():
        name = manifest.collection(ds, "BM25")
        sparse, ids = _bm25_corpus(ds)
        qc.recreate_sparse(c, name)
        pts = (models.PointStruct(id=i, payload={"doc_id": did},
               vector={qc.SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())})
               for i, (s, did) in enumerate(zip(sparse, ids)))
        qc.bulk_upload(c, name, pts, up["upload_batch"], up["upload_parallel"])
        print(f"sparse {name}: {len(ids)} pts")
    # hybrid: A dense (cached) + BM25 sparse in one collection
    for h in manifest.hybrid_specs():
        for ds in manifest.all_datasets():
            name = manifest.collection(ds, h["run_id"])
            vecs, ids = _load_vecs(ds, h["dense_run"], "corpus")
            sparse, sids = _bm25_corpus(ds)
            assert ids == sids, "dense/sparse doc id order mismatch"
            qc.recreate_hybrid(c, name, h["dim"])
            pts = (models.PointStruct(
                       id=i, payload={"doc_id": did},
                       vector={qc.DENSE: v.tolist(),
                               qc.SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())})
                   for i, (v, s, did) in enumerate(zip(vecs, sparse, ids)))
            qc.bulk_upload(c, name, pts, up["upload_batch"], up["upload_parallel"])
            print(f"hybrid {name}: {len(ids)} pts")


def measure_ingest():
    """End-to-end embed+upsert wall at fixed config, on the code dataset."""
    ING.mkdir(parents=True, exist_ok=True)
    c = qc.client()
    up = manifest.qdrant()
    ds = manifest.datasets()["code"][0]
    corpus, _, _ = datasets_io.load(ds)
    ids = list(corpus.keys())
    texts = [corpus[i] for i in ids]
    for e in manifest.dense_embedders():
        out = ING / f"{e['run_id']}.json"
        if out.exists():
            continue
        name = f"ingest_tmp__{e['run_id']}".replace("/", "_")
        embedder = systems.build(e["kind"], e["model_id"])
        t = time.perf_counter()
        vecs = embedder.embed(texts, batch_size=up["upload_batch"])
        t_embed = time.perf_counter() - t
        qc.recreate_dense(c, name, e["dim"])
        t = time.perf_counter()
        qc.bulk_upload(c, name, _dense_points(vecs, ids), up["upload_batch"], up["upload_parallel"])
        t_upsert = time.perf_counter() - t
        c.delete_collection(name)
        r = {"run_id": e["run_id"], "dataset": ds, "n_docs": len(ids),
             "batch": up["upload_batch"], "parallel": up["upload_parallel"],
             "embed_s": round(t_embed, 3), "upsert_s": round(t_upsert, 3),
             "total_s": round(t_embed + t_upsert, 3),
             "docs_per_s_total": round(len(ids) / (t_embed + t_upsert), 1)}
        out.write_text(json.dumps(r, indent=2))
        print(f"ingest {e['run_id']}: embed={t_embed:.1f}s upsert={t_upsert:.1f}s total={r['total_s']}s")


if __name__ == "__main__":
    import sys
    if "--ingest-only" not in sys.argv:
        build_collections()
    if "--collections-only" not in sys.argv:
        measure_ingest()
    print("stage 03 done")
