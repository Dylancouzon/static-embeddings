"""Follow-up measurements from the final review (outside the 01-05 pipeline):

1. Dense(MiniLM)+BM25 hybrid vs static+BM25 hybrid, and each hybrid's marginal add
   over BM25-alone — the honest "what does static actually contribute to a hybrid".
2. V1 paired bootstrap: is static's code NDCG@10 a statistical tie with BM25, or a
   real loss? (Cannot flip the FAIL — static still does not beat BM25 — but
   characterizes the gap.)

Writes results/extra.json. Reuses cached vectors + the running Qdrant.
"""
from __future__ import annotations

import json
import math

import numpy as np
from qdrant_client import models
from ranx import Qrels, Run, evaluate

import datasets_io
import manifest
import qc
import systems

BEIR = ["scifact", "nfcorpus", "fiqa"]
CODE = "codesearchnet-python"
ALL = BEIR + [CODE]
TOPK, PREFETCH, QBATCH = 10, 100, 64
RET = "potion-retrieval-32M"


def _qrels(ds):
    return Qrels.from_file(str(datasets_io.DATA / ds / "qrels.trec"), kind="trec")


def _ndcg(ds, run):
    return float(evaluate(_qrels(ds), Run(run), ["ndcg@10"]))


def _cached(ds, run_id, kind):
    npy, idj = manifest.vec_paths(ds, run_id, kind)
    return np.load(npy), json.loads(idj.read_text())


def _bm25_corpus(ds):
    corpus, _, _ = datasets_io.load(ds)
    ids = list(corpus.keys())
    sp = systems.build("bm25").embed_sparse([corpus[i] for i in ids])
    return sp, ids


def _bm25_queries(ds, qids):
    _, queries, _ = datasets_io.load(ds)
    return systems.build("bm25").embed_query([queries[q] for q in qids])


def dense_bm25_hybrid_ndcg(c, ds, dense_run, dim):
    """Build <dense_run>+BM25 hybrid, return RRF NDCG@10 (mirrors the pipeline's H)."""
    name = f"exhybrid__{ds}__{dense_run}".replace("/", "_")
    dv, ids = _cached(ds, dense_run, "corpus")
    sp, sids = _bm25_corpus(ds)
    assert ids == sids
    qc.recreate_hybrid(c, name, dim)
    pts = (models.PointStruct(
               id=i, payload={"doc_id": did},
               vector={qc.DENSE: v.tolist(),
                       qc.SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())})
           for i, (v, s, did) in enumerate(zip(dv, sp, ids)))
    qc.bulk_upload(c, name, pts, 128, 2)

    dqv, qids = _cached(ds, dense_run, "queries")
    qsp = _bm25_queries(ds, qids)
    run = {}
    for s in range(0, len(qids), QBATCH):
        reqs = []
        for v, sv in zip(dqv[s:s + QBATCH], qsp[s:s + QBATCH]):
            reqs.append(models.QueryRequest(
                prefetch=[
                    models.Prefetch(query=v.tolist(), using=qc.DENSE, limit=PREFETCH,
                                    params=models.SearchParams(exact=True)),
                    models.Prefetch(query=models.SparseVector(indices=sv.indices.tolist(),
                                    values=sv.values.tolist()), using=qc.SPARSE, limit=PREFETCH),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF), limit=TOPK, with_payload=True))
        for qid, resp in zip(qids[s:s + QBATCH], c.query_batch_points(name, requests=reqs)):
            run[qid] = {p.payload["doc_id"]: p.score for p in resp.points}
    c.delete_collection(name)
    return _ndcg(ds, run)


def _dense_run_on(c, ds, run_id):
    """Per-query top-10 doc_ids for a dense system (for the code bootstrap)."""
    dqv, qids = _cached(ds, run_id, "queries")
    run = {}
    for s in range(0, len(qids), QBATCH):
        reqs = [models.QueryRequest(query=v.tolist(), limit=TOPK, with_payload=True,
                                    params=models.SearchParams(exact=True)) for v in dqv[s:s + QBATCH]]
        for qid, resp in zip(qids[s:s + QBATCH], c.query_batch_points(manifest.collection(ds, run_id), requests=reqs)):
            run[qid] = [p.payload["doc_id"] for p in resp.points]
    return run, qids


def _bm25_run_on(c, ds):
    _, queries, _ = datasets_io.load(ds)
    qids = list(queries.keys())
    qsp = _bm25_queries(ds, qids)
    run = {}
    for s in range(0, len(qids), QBATCH):
        reqs = [models.QueryRequest(query=models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
                                    using=qc.SPARSE, limit=TOPK, with_payload=True,
                                    params=models.SearchParams(exact=True)) for sv in qsp[s:s + QBATCH]]
        for qid, resp in zip(qids[s:s + QBATCH], c.query_batch_points(manifest.collection(ds, "BM25"), requests=reqs)):
            run[qid] = [p.payload["doc_id"] for p in resp.points]
    return run, qids


def _per_query_ndcg_singlerel(run, qrels):
    """NDCG@10 per query for single-relevant-doc datasets (code): 1/log2(1+rank) or 0."""
    out = []
    for qid, docs in qrels.items():
        rel = next(iter(docs))
        ranked = run.get(qid, [])
        out.append(1 / math.log2(1 + (ranked.index(rel) + 1)) if rel in ranked else 0.0)
    return np.array(out)


def bootstrap_diff(a, b, iters=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = [(a[idx].mean() - b[idx].mean()) for idx in (rng.integers(0, n, n) for _ in range(iters))]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(np.mean(a) - np.mean(b)), "ci95": [float(lo), float(hi)],
            "p_static_worse": float(np.mean(np.array(diffs) < 0))}


def main():
    c = qc.client()
    out = {"dense_bm25_hybrid_ndcg": {}, "static_bm25_hybrid_ndcg": {}, "bm25_ndcg": {}}
    for ds in ALL:
        out["dense_bm25_hybrid_ndcg"][ds] = round(dense_bm25_hybrid_ndcg(c, ds, "D1", 384), 4)
        out["static_bm25_hybrid_ndcg"][ds] = round(
            json.loads((manifest.VECDIR.parent / "quality" / f"{ds}__H__{RET}.json").read_text())["metrics"]["ndcg@10"], 4)
        out["bm25_ndcg"][ds] = round(
            json.loads((manifest.VECDIR.parent / "quality" / f"{ds}__BM25.json").read_text())["metrics"]["ndcg@10"], 4)
        print(f"{ds}: dense+BM25={out['dense_bm25_hybrid_ndcg'][ds]}  "
              f"static+BM25={out['static_bm25_hybrid_ndcg'][ds]}  BM25={out['bm25_ndcg'][ds]}")

    beir_avg = lambda d: round(sum(out[d][x] for x in BEIR) / 3, 4)  # noqa: E731
    out["beir_avg"] = {k: beir_avg(k) for k in ("dense_bm25_hybrid_ndcg", "static_bm25_hybrid_ndcg", "bm25_ndcg")}
    out["static_marginal_add_over_bm25_beir"] = round(
        out["beir_avg"]["static_bm25_hybrid_ndcg"] - out["beir_avg"]["bm25_ndcg"], 4)

    # V1 bootstrap on code
    a_run, _ = _dense_run_on(c, CODE, f"A__{RET}")
    b_run, _ = _bm25_run_on(c, CODE)
    qrels = {qid: docs for qid, docs in ((line.split()[0], {line.split()[2]: 1})
             for line in open(datasets_io.DATA / CODE / "qrels.trec"))}
    a_nd = _per_query_ndcg_singlerel(a_run, qrels)
    b_nd = _per_query_ndcg_singlerel(b_run, qrels)
    out["v1_bootstrap_code"] = {"static_mean": float(a_nd.mean()), "bm25_mean": float(b_nd.mean()),
                                **bootstrap_diff(a_nd, b_nd)}
    print("\nBEIR avg:", out["beir_avg"])
    print("static marginal add over BM25 (BEIR):", out["static_marginal_add_over_bm25_beir"])
    print("V1 bootstrap (static-BM25 on code):", out["v1_bootstrap_code"])
    (manifest.VECDIR.parent / "extra.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
