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


def recreate_dense(c: QdrantClient, name: str, dim: int):
    c.recreate_collection(name, vectors_config=models.VectorParams(
        size=dim, distance=models.Distance.COSINE))


def recreate_sparse(c: QdrantClient, name: str):
    c.recreate_collection(name, vectors_config={},
                          sparse_vectors_config={SPARSE: models.SparseVectorParams()})


def recreate_hybrid(c: QdrantClient, name: str, dim: int):
    c.recreate_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams()})
