"""Dataset loaders + on-disk cache. Shared by 01_download, correctness_gate,
02_embed, 04_evaluate so every stage sees identical corpora/qrels.

On disk per dataset (data/<name>/):
  corpus.jsonl   {"id","text"}
  queries.jsonl  {"id","text"}
  qrels.trec     TREC 4-col: qid 0 docid rel   (loaded by ranx)

BEIR doc text = title + text (BEIR convention). CodeSearchNet: corpus = function
code, query = docstring first paragraph, qrels = query->its own function (one
relevant doc), deduped by identical code so qrels stay unambiguous.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path("data")
CODE_DATASET = "codesearchnet-python"
CSN_MAX_DOCS = 20000


def _write_jsonl(path: Path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> dict[str, str]:
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["id"]] = r["text"]
    return out


def _write_trec(path: Path, qrels: dict[str, dict[str, int]]):
    with open(path, "w") as f:
        for qid, docs in qrels.items():
            for docid, rel in docs.items():
                f.write(f"{qid} 0 {docid} {rel}\n")


def _first_paragraph(doc: str) -> str:
    return doc.strip().split("\n\n")[0].strip()


def _fetch_beir(name: str):
    from datasets import load_dataset
    corpus_ds = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries_ds = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{name}-qrels", split="test")
    corpus = [{"id": str(r["_id"]), "text": (r.get("title", "") + " " + r["text"]).strip()}
              for r in corpus_ds]
    queries_by_id = {str(r["_id"]): r["text"] for r in queries_ds}
    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_ds:
        if int(r["score"]) <= 0:
            continue
        qid = str(r["query-id"])
        qrels.setdefault(qid, {})[str(r["corpus-id"])] = int(r["score"])
    # keep only queries that have judgments
    queries = [{"id": qid, "text": queries_by_id[qid]} for qid in qrels if qid in queries_by_id]
    qrels = {q["id"]: qrels[q["id"]] for q in queries}
    return corpus, queries, qrels


def _fetch_codesearchnet():
    from datasets import load_dataset
    ds = load_dataset("code-search-net/code_search_net", name="python", split="test")
    corpus, queries, qrels = [], [], {}
    seen_code: set[str] = set()   # docstring-free code (dedup identical functions)
    seen_ids: set[str] = set()
    for r in ds:
        code = r["func_code_string"]
        doc = r["func_documentation_string"]
        if not code or not doc or not doc.strip():
            continue
        query = _first_paragraph(doc)
        if not query:
            continue
        # CSN docstrings live verbatim inside func_code_string. Leaving them in makes
        # the query a substring of its relevant doc -> substring retrieval, not code
        # search. Strip the docstring so the corpus is docstring-free code.
        code_text = code.replace(doc, " ").strip()
        if not code_text or code_text in seen_code:
            continue
        doc_id = r["func_code_url"]
        assert doc_id not in seen_ids, f"duplicate CSN doc_id: {doc_id}"
        seen_code.add(code_text)
        seen_ids.add(doc_id)
        corpus.append({"id": doc_id, "text": code_text})
        qid = f"q{len(queries)}"
        queries.append({"id": qid, "text": query})
        qrels[qid] = {doc_id: 1}
        if len(corpus) >= CSN_MAX_DOCS:
            break
    return corpus, queries, qrels


def download(name: str):
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    if name == CODE_DATASET:
        corpus, queries, qrels = _fetch_codesearchnet()
    else:
        corpus, queries, qrels = _fetch_beir(name)
    _write_jsonl(out / "corpus.jsonl", corpus)
    _write_jsonl(out / "queries.jsonl", queries)
    _write_trec(out / "qrels.trec", qrels)
    return len(corpus), len(queries), sum(len(v) for v in qrels.values())


def load(name: str):
    out = DATA / name
    corpus = _read_jsonl(out / "corpus.jsonl")
    queries = _read_jsonl(out / "queries.jsonl")
    qrels: dict[str, dict[str, int]] = {}
    for line in open(out / "qrels.trec"):
        qid, _, docid, rel = line.split()
        qrels.setdefault(qid, {})[docid] = int(rel)
    return corpus, queries, qrels


def corpus_sample(names: list[str], n: int, seed: int = 0) -> list[str]:
    """n documents sampled across datasets for the correctness gate."""
    import random
    rng = random.Random(seed)
    pool: list[str] = []
    for name in names:
        try:
            pool.extend((DATA / name / "corpus.jsonl").read_text().splitlines())
        except FileNotFoundError:
            continue
    rng.shuffle(pool)
    return [json.loads(line)["text"] for line in pool[:n]]
