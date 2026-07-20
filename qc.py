"""Qdrant client + collection helpers. Query API only (query_points); one collection
per (dataset, run_id). Point ids are ints; original doc id lives in payload['doc_id']
so results map back to qrels."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import islice

from qdrant_client import QdrantClient, models

import manifest

DENSE = "dense"
SPARSE = "bm25"


def client() -> QdrantClient:
    return QdrantClient(url=manifest.qdrant()["url"], timeout=300)


def bulk_upload(c: QdrantClient, name: str, points, batch_size: int, parallel: int):
    """N concurrent upsert streams over HTTP. Replaces client.upload_points, whose
    fork-based worker pool segfaults on macOS + Python 3.14. Threads give the same
    'parallel streams' semantics (qdrant-client is thread-safe) without fork."""
    it = iter(points)
    batches = iter(lambda: list(islice(it, batch_size)), [])
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        list(ex.map(lambda b: c.upsert(name, points=b, wait=True), batches))


# fastembed Qdrant/bm25 emits TF-only values; IDF is applied server-side, so the
# sparse index MUST declare modifier=IDF or BM25 scores are term-frequency only.
_IDF = models.SparseVectorParams(modifier=models.Modifier.IDF)


def _recreate(c: QdrantClient, name: str, **kwargs):
    if c.collection_exists(name):
        c.delete_collection(name)
    c.create_collection(name, **kwargs)


def recreate_dense(c: QdrantClient, name: str, dim: int):
    _recreate(c, name, vectors_config=models.VectorParams(
        size=dim, distance=models.Distance.COSINE))


def recreate_sparse(c: QdrantClient, name: str):
    _recreate(c, name, vectors_config={}, sparse_vectors_config={SPARSE: _IDF})


def recreate_hybrid(c: QdrantClient, name: str, dim: int):
    _recreate(
        c, name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: _IDF})
