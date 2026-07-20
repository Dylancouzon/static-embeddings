"""Expand bench.toml into concrete run specs. One source of truth for every stage.

An 'embedder' is one (system, model) that produces vectors. run_id uniquely names
it: baselines keep their id (BM25/D1/D2); static candidates are '{sys}__{model_key}'.
Qdrant collection per (dataset, run_id) — each system's vectors are distinct, so
they can't share a collection even at equal dims (CLAUDE.md's 'one collection per
model' assumes one system per model; we compare several).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

VECDIR = Path("results/vectors")


def vec_paths(dataset: str, run_id: str, kind: str) -> tuple[Path, Path]:
    """(vectors.npy, ids.json) for a (dataset, run, corpus|queries). Plain string
    join — Path.with_suffix would eat the .corpus/.queries segment."""
    base = VECDIR / f"{dataset}__{run_id}.{kind}"
    return Path(f"{base}.npy"), Path(f"{base}.ids.json")

_CFG = None


def load() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = tomllib.loads(Path("bench.toml").read_text())
    return _CFG


def qdrant() -> dict:
    return load()["qdrant"]


def threads() -> dict:
    return load()["threads"]


def speed() -> dict:
    return load()["speed"]


def model_id(model_key: str) -> str:
    return load()["models"][model_key]


def datasets() -> dict:
    return load()["datasets"]


def all_datasets() -> list[str]:
    # BENCH_DATASETS=scifact,fiqa restricts the run (smoke tests, resumes). Default: all.
    import os
    d = datasets()
    full = d["beir"] + d["code"]
    override = os.environ.get("BENCH_DATASETS")
    return [x for x in override.split(",") if x in full] if override else full


def embedders() -> list[dict]:
    """Every (system, model) that produces vectors for quality + throughput."""
    c = load()
    runs: list[dict] = []
    for r in c["runs"]:  # baselines (fixed model)
        runs.append({
            "run_id": r["id"], "system": r["id"], "kind": r["kind"],
            "model_key": None, "model_id": r.get("model"), "dim": r["dim"],
        })
    st = c["static"]  # static candidates × potion models
    for sys in st["systems"]:
        for mk in st["models"]:
            runs.append({
                "run_id": f"{sys}__{mk}", "system": sys, "kind": st["kinds"][sys],
                "model_key": mk, "model_id": model_id(mk), "dim": st["dims"][mk],
            })
    return runs


def dense_embedders() -> list[dict]:
    return [e for e in embedders() if e["kind"] != "bm25"]


def hybrid_specs() -> list[dict]:
    h = load()["hybrid"]
    out = []
    for mk in h["models"]:
        out.append({
            "run_id": f"{h['id']}__{mk}", "model_key": mk,
            "dense_run": f"{h['dense_system']}__{mk}", "sparse_run": h["sparse_system"],
            "dim": load()["static"]["dims"][mk],
        })
    return out


def loadonly_specs() -> list[dict]:
    return [{**r, "model_id": model_id(r["model"])} for r in load()["loadonly"]["runs"]]


def collection(dataset: str, run_id: str) -> str:
    return f"{dataset}__{run_id}".replace("/", "_").replace(".", "_")


if __name__ == "__main__":
    print("datasets:", all_datasets())
    for e in embedders():
        print(" embed", e["run_id"], e["kind"], e["model_id"], "dim", e["dim"])
    for h in hybrid_specs():
        print(" hybrid", h["run_id"], "dense", h["dense_run"], "sparse", h["sparse_run"])
