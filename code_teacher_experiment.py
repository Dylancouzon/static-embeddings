"""Adversarial follow-up: does a CODE-distilled static model flip V1 / the hybrid case?

The pre-registration fixed potion-retrieval-32M (a general-text distillation) as THE
static model, then judged it on code. This distills a code-retrieval teacher into a
model2vec static model, loads it through candidate A's OWN path (StaticNumpy), and
measures on CSN-Python:

  standalone NDCG@10  -> V1 (static >= BM25 0.2955?)
  static+BM25 RRF     -> hybrid (>= BM25? by how much?)

Teacher: st-codesearch-distilroberta-base (ST fine-tuned on CodeSearchNet). It saw
CSN, so this is the OPTIMISTIC ceiling — if even this can't beat BM25 standalone,
no-go is airtight; if it does, the ceiling is real (confirm later on a non-CSN teacher).

Correctness discipline: A (StaticNumpy) must equal B (model2vec) on the distilled
model within atol=1e-5 before any number is trusted (CLAUDE.md gate, per-model).

Writes results/code_teacher.json. Reuses the running Qdrant. Run with distill deps:
  uv run --with "model2vec[distill]" python code_teacher_experiment.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from qdrant_client import models
from ranx import Qrels, Run, evaluate

import datasets_io
import qc
import systems
from static_numpy import StaticNumpy

TEACHER = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
OUTDIR = Path("models/potion-code-st")
DS = "codesearchnet-python"
PCA_DIMS = 256
TOPK, PREFETCH, QBATCH = 10, 100, 64
# frozen reference points from results/RESULTS.md + extra.json (general-text potion)
REF = {"bm25_code": 0.2955, "static_general_code": 0.2887, "hybrid_general_code": 0.3585}


def distill_teacher():
    if (OUTDIR / "model.safetensors").exists():
        print(f"distilled model present at {OUTDIR}")
        return
    from model2vec.distill import distill
    print(f"distilling {TEACHER} -> pca_dims={PCA_DIMS} (downloads torch/transformers/teacher)")
    m = distill(model_name=TEACHER, pca_dims=PCA_DIMS)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(str(OUTDIR))
    print(f"saved -> {OUTDIR}")


def correctness_check() -> dict:
    """Record A (numpy port) vs B (model2vec) parity on the distilled model.

    On potion this is 3e-8. On a RoBERTa byte-level-BPE tokenizer A drifts (~5e-4):
    A's encode_batch_fast on the saved tokenizer.json doesn't reproduce model2vec's
    tokenization for BPE. Quality below is measured with B (the oracle), so this is a
    SHIPPING caveat for candidate A on non-potion tokenizers, not a blocker here.
    """
    from model2vec import StaticModel
    paths = {f: str(OUTDIR / f) for f in StaticNumpy.FILES}
    A = StaticNumpy(str(OUTDIR), paths=paths)
    B = StaticModel.from_pretrained(str(OUTDIR))
    edges = ["", "   \n\t", "def f(x): return x", "\x00\x01 qwxzjkv",
             "🚀 日本語 café", "for i in range(10):\n    print(i*2)"]
    texts = datasets_io.corpus_sample([DS], 300) + edges
    va = A.embed(texts)
    vb = B.encode(texts, max_length=None, use_multiprocessing=False,
                  show_progress_bar=False).astype(np.float32)
    max_abs = float(np.abs(va - vb).max())
    ok = bool(np.allclose(va, vb, atol=1e-5))
    print(f"A-vs-B parity on distilled model: allclose(1e-5)={ok} max_abs={max_abs:.2e} dim={A.dim}")
    return {"a_vs_b_allclose_1e-5": ok, "a_vs_b_max_abs": max_abs, "dim": A.dim}


def _qrels() -> Qrels:
    return Qrels.from_file(str(datasets_io.DATA / DS / "qrels.trec"), kind="trec")


def _ndcg_recall(run: dict) -> dict:
    res = evaluate(_qrels(), Run(run), ["ndcg@10", "recall@10"])
    return {k: float(res[k]) for k in ("ndcg@10", "recall@10")}


def run_standalone_and_hybrid(dim: int) -> dict:
    corpus, queries, _ = datasets_io.load(DS)
    cids = list(corpus.keys())
    qids = list(queries.keys())

    # Use model2vec (B, the correctness oracle) for the code static vectors — A drifts
    # on this BPE tokenizer (see correctness_check); the quality question is the model's.
    from model2vec import StaticModel
    b = StaticModel.from_pretrained(str(OUTDIR))
    embed = lambda ts: b.encode(ts, max_length=None, use_multiprocessing=False,  # noqa: E731
                                show_progress_bar=False, batch_size=256).astype(np.float32)
    print(f"embedding {len(cids)} corpus + {len(qids)} queries (code-static, model2vec)...")
    cvec = embed([corpus[i] for i in cids])
    qvec = embed([queries[q] for q in qids])

    bm = systems.build("bm25")
    print("embedding BM25 sparse corpus + queries...")
    csp = bm.embed_sparse([corpus[i] for i in cids])
    qsp = bm.embed_query([queries[q] for q in qids])

    c = qc.client()

    # --- standalone dense (V1) ---
    name_d = f"codeexp__standalone__{DS}"
    qc.recreate_dense(c, name_d, dim)
    qc.bulk_upload(c, name_d, (
        models.PointStruct(id=i, payload={"doc_id": did}, vector=v.tolist())
        for i, (v, did) in enumerate(zip(cvec, cids))), 128, 2)
    run_d: dict = {}
    for s in range(0, len(qids), QBATCH):
        reqs = [models.QueryRequest(query=v.tolist(), limit=TOPK, with_payload=True,
                                    params=models.SearchParams(exact=True))
                for v in qvec[s:s + QBATCH]]
        for qid, resp in zip(qids[s:s + QBATCH], c.query_batch_points(name_d, requests=reqs)):
            run_d[qid] = {p.payload["doc_id"]: p.score for p in resp.points}
    standalone = _ndcg_recall(run_d)
    c.delete_collection(name_d)

    # --- static + BM25 hybrid, server-side RRF ---
    name_h = f"codeexp__hybrid__{DS}"
    qc.recreate_hybrid(c, name_h, dim)
    qc.bulk_upload(c, name_h, (
        models.PointStruct(id=i, payload={"doc_id": did},
            vector={qc.DENSE: v.tolist(),
                    qc.SPARSE: models.SparseVector(indices=s.indices.tolist(),
                                                   values=s.values.tolist())})
        for i, (v, s, did) in enumerate(zip(cvec, csp, cids))), 128, 2)
    run_h: dict = {}
    for s in range(0, len(qids), QBATCH):
        reqs = []
        for v, sv in zip(qvec[s:s + QBATCH], qsp[s:s + QBATCH]):
            reqs.append(models.QueryRequest(
                prefetch=[
                    models.Prefetch(query=v.tolist(), using=qc.DENSE, limit=PREFETCH,
                                    params=models.SearchParams(exact=True)),
                    models.Prefetch(query=models.SparseVector(indices=sv.indices.tolist(),
                                    values=sv.values.tolist()), using=qc.SPARSE, limit=PREFETCH),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF), limit=TOPK, with_payload=True))
        for qid, resp in zip(qids[s:s + QBATCH], c.query_batch_points(name_h, requests=reqs)):
            run_h[qid] = {p.payload["doc_id"]: p.score for p in resp.points}
    hybrid = _ndcg_recall(run_h)
    c.delete_collection(name_h)

    return {"standalone": standalone, "hybrid": hybrid}


def main():
    distill_teacher()
    parity = correctness_check()
    res = run_standalone_and_hybrid(parity["dim"])
    out = {"teacher": TEACHER, "pca_dims": PCA_DIMS, "dim": parity["dim"], "dataset": DS,
           "reference": REF, "a_vs_b_parity": parity, **res}
    sa = res["standalone"]["ndcg@10"]
    hy = res["hybrid"]["ndcg@10"]
    print("\n===== CODE-TEACHER STATIC on CSN-Python =====")
    print(f"standalone NDCG@10 = {sa:.4f}   (V1 needs >= BM25 {REF['bm25_code']};  "
          f"general-potion was {REF['static_general_code']})")
    print(f"  -> V1 {'PASS (beats BM25)' if sa >= REF['bm25_code'] else 'FAIL (loses to BM25)'}")
    print(f"hybrid NDCG@10     = {hy:.4f}   (BM25 alone {REF['bm25_code']};  "
          f"general-potion hybrid {REF['hybrid_general_code']})")
    print(f"  -> hybrid vs BM25: {hy - REF['bm25_code']:+.4f} "
          f"({100*(hy/REF['bm25_code']-1):+.1f}%)")
    Path("results").mkdir(exist_ok=True)
    Path("results/code_teacher.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/code_teacher.json")


if __name__ == "__main__":
    sys.exit(main())
