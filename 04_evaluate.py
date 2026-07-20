"""Stage 04 — query every collection with exact=True, score NDCG@10 / Recall@10 via
ranx against TREC qrels. Covers dense (D1/D2/A/B/C), BM25 sparse, and hybrid RRF
(server-side prefetch + fusion). One local brute-force cosine run (scifact) confirms
Qdrant exact ranking reproduces a numpy top-k — separating model quality from client
effects. Writes results/quality/<dataset>__<run_id>.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qdrant_client import models
from ranx import Qrels, Run, evaluate

import datasets_io
import manifest
import qc
import systems

QUAL = Path("results/quality")
METRICS = ["ndcg@10", "recall@10"]
TOPK = 10
PREFETCH = 100  # hybrid per-branch candidate depth before RRF
QBATCH = 64


def _qrels(dataset: str) -> Qrels:
    return Qrels.from_file(str(datasets_io.DATA / dataset / "qrels.trec"), kind="trec")


def _score(qrels: Qrels, run_dict: dict) -> dict:
    res = evaluate(qrels, Run(run_dict), METRICS)
    return {m: float(res[m]) for m in METRICS}


def _run_from_batches(c, name, query_vecs, qids, make_request):
    run: dict[str, dict[str, float]] = {}
    for s in range(0, len(qids), QBATCH):
        reqs = [make_request(v) for v in query_vecs[s:s + QBATCH]]
        for qid, resp in zip(qids[s:s + QBATCH], c.query_batch_points(name, requests=reqs)):
            run[qid] = {p.payload["doc_id"]: p.score for p in resp.points}
    return run


def eval_dense():
    c = qc.client()
    for e in manifest.dense_embedders():
        for ds in manifest.all_datasets():
            _eval_one(c, ds, e["run_id"], _dense_query_vecs(ds, e["run_id"]),
                      lambda v: models.QueryRequest(query=v, limit=TOPK, with_payload=True,
                                                    params=models.SearchParams(exact=True)))


def _dense_query_vecs(ds, run_id):
    npy, idj = manifest.vec_paths(ds, run_id, "queries")
    return np.load(npy), json.loads(idj.read_text())


def _eval_one(c, ds, run_id, vecs_ids, make_request):
    vecs, qids = vecs_ids
    name = manifest.collection(ds, run_id)
    run = _run_from_batches(c, name, [v.tolist() for v in vecs], qids, make_request)
    _write(ds, run_id, _score(_qrels(ds), run))


def eval_bm25():
    c = qc.client()
    bm = systems.build("bm25")
    for ds in manifest.all_datasets():
        _, queries, _ = datasets_io.load(ds)
        qids = list(queries.keys())
        sparse = bm.embed_query([queries[q] for q in qids])
        svecs = [models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()) for s in sparse]
        name = manifest.collection(ds, "BM25")
        run = _run_from_batches(c, name, svecs, qids,
                                lambda v: models.QueryRequest(query=v, using=qc.SPARSE, limit=TOPK,
                                                              with_payload=True,
                                                              params=models.SearchParams(exact=True)))
        _write(ds, "BM25", _score(_qrels(ds), run))


def eval_hybrid():
    c = qc.client()
    bm = systems.build("bm25")
    for h in manifest.hybrid_specs():
        for ds in manifest.all_datasets():
            dvecs, qids = _dense_query_vecs(ds, h["dense_run"])
            _, queries, _ = datasets_io.load(ds)
            sparse = bm.embed_query([queries[q] for q in qids])
            name = manifest.collection(ds, h["run_id"])
            run: dict[str, dict[str, float]] = {}
            for s in range(0, len(qids), QBATCH):
                reqs = []
                for dv, sv in zip(dvecs[s:s + QBATCH], sparse[s:s + QBATCH]):
                    reqs.append(models.QueryRequest(
                        prefetch=[
                            models.Prefetch(query=dv.tolist(), using=qc.DENSE, limit=PREFETCH,
                                            params=models.SearchParams(exact=True)),
                            models.Prefetch(query=models.SparseVector(
                                indices=sv.indices.tolist(), values=sv.values.tolist()),
                                using=qc.SPARSE, limit=PREFETCH),
                        ],
                        query=models.FusionQuery(fusion=models.Fusion.RRF),
                        limit=TOPK, with_payload=True))
                for qid, resp in zip(qids[s:s + QBATCH], c.query_batch_points(name, requests=reqs)):
                    run[qid] = {p.payload["doc_id"]: p.score for p in resp.points}
            _write(ds, h["run_id"], _score(_qrels(ds), run))


def bruteforce_check(dataset: str = "scifact"):
    """Local numpy top-k must reproduce Qdrant exact ranking (quality vs client effects)."""
    out = {}
    for e in manifest.dense_embedders():
        cnpy, cidj = manifest.vec_paths(dataset, e["run_id"], "corpus")
        qnpy, qidj = manifest.vec_paths(dataset, e["run_id"], "queries")
        if not cnpy.exists():
            continue
        corpus = np.load(cnpy)
        cids = json.loads(cidj.read_text())
        qv = np.load(qnpy)
        qids = json.loads(qidj.read_text())
        cn = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-32)
        qn = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-32)
        sims = qn @ cn.T
        run = {}
        for i, qid in enumerate(qids):
            top = np.argsort(-sims[i])[:TOPK]
            run[qid] = {cids[j]: float(sims[i, j]) for j in top}
        bf = _score(_qrels(dataset), run)
        qdr = json.loads((QUAL / f"{dataset}__{e['run_id']}.json").read_text())["metrics"]
        out[e["run_id"]] = {"bruteforce": bf, "qdrant": qdr,
                            "ndcg_delta": round(abs(bf["ndcg@10"] - qdr["ndcg@10"]), 4)}
    (QUAL / "_bruteforce_check.json").write_text(json.dumps(out, indent=2))
    for rid, v in out.items():
        print(f"bruteforce {rid}: ndcg_delta={v['ndcg_delta']}")


def _write(ds, run_id, metrics):
    QUAL.mkdir(parents=True, exist_ok=True)
    (QUAL / f"{ds}__{run_id}.json").write_text(json.dumps(
        {"dataset": ds, "run_id": run_id, "metrics": metrics}, indent=2))
    print(f"{ds}__{run_id}: ndcg@10={metrics['ndcg@10']:.4f} recall@10={metrics['recall@10']:.4f}")


if __name__ == "__main__":
    eval_dense()
    eval_bm25()
    eval_hybrid()
    bruteforce_check()
    print("stage 04 done")
