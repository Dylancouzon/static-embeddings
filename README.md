# Static Embeddings Viability Benchmark

Should [fastembed](https://github.com/qdrant/fastembed) ship static embeddings (model2vec / potion)? This repo answers with data, measuring them against the baselines fastembed users already have.

**Decision: no.** The best static model does not beat BM25 — which fastembed already ships, free and with no model download — on any track we tested, including code search, the use case that motivated the request. Full reasoning, speed/footprint, and the when-to-use-what matrix: **[DECISION.md](DECISION.md)**.

## Best static vs BM25 — NDCG@10 (exact search)

| Dataset | BM25 (fastembed, no download) | potion-retrieval-32M (best static) |
|---|---|---|
| SciFact | **0.683** | 0.638 |
| NFCorpus | **0.324** | 0.307 |
| FiQA | **0.243** | 0.190 |
| CodeSearchNet (code) | **0.296** | 0.289 |

BM25 wins every track. On code, static's motivating use case, the gap is small but statistically real: a paired bootstrap over 20k queries puts the 95% interval below zero. Distilling a code-specific teacher into static to test the obvious fix scored *worse* (0.056), because averaging context-free token vectors discards the composition code search needs. For code-search quality neither lexical nor static wins anyway — bge-small (dense) scores 0.674. Use dense there.

One case we did not test could change this: cross-lingual retrieval, where BM25 can't match across languages and a multilingual static model can. It is outside the pre-registered English scope — see [DECISION.md](DECISION.md).

## Speed and cost

Speed is static's selling point, so this is where it has to win.

| System | Embed throughput (docs/s, batch 32) | Model download |
|---|---|---|
| BM25 | 15,944 | none |
| potion-retrieval-32M (static) | 9,121 | 129 MB |
| MiniLM (ONNX dense) | 301 | 90 MB |

Static embeds 30× faster than an ONNX dense model — a real, durable advantage. But BM25 is the baseline it has to beat, and BM25 embeds faster, downloads nothing, and scores higher on every track. Against the free option already in fastembed, static is slower, heavier, and less accurate.

## Methodology

- **Systems**, all behind one `embed()` signature: BM25, static (dependency-free numpy port + model2vec + potion-ONNX), and dense baselines (MiniLM, bge-small).
- **Retrieval** runs end-to-end through Qdrant (embed → upsert → query) using the Query API with `exact=True`, so approximate-search recall never confounds model quality. Scored with `ranx` against TREC qrels.
- **Datasets:** BEIR (SciFact, NFCorpus, FiQA) and CodeSearchNet Python (docstring → function).
- **Trust checks:** a correctness gate proves the numpy port matches model2vec to 3e-8; a local brute-force top-k reproduces Qdrant's exact ranking to 0.0 NDCG delta; all thresholds were fixed before any results existed.

Full generated numbers: [results/RESULTS.md](results/RESULTS.md).

## Repo layout

| File | Purpose |
|---|---|
| `run_all.py` | Runs the whole pipeline: gate → download → embed → index → evaluate → report |
| `01_download_data.py` … `05_report.py` | The pipeline stages |
| `correctness_gate.py` | Proves the numpy port matches model2vec before any benchmarking |
| `static_numpy.py` | Candidate implementation: the dependency-free numpy static embedder |
| `systems.py` | All embedders behind one `embed()` signature |
| `qc.py`, `speed.py`, `datasets_io.py`, `manifest.py`, `env.py` | Qdrant helpers, timing, data loading, config, environment capture |
| `bench.toml` | The systems × models × datasets manifest |
| `DECISION.md` | The verdict and the when-to-use-what matrix |
| `results/` | Generated numbers (`RESULTS.md`, per-run JSON) |

## Reproduce / run on your own data

```bash
docker compose up -d      # Qdrant (image pinned by digest)
uv sync
uv run python run_all.py  # correctness gate → download → embed → index → evaluate → report
```

`bench.toml` defines the systems × models × datasets. To benchmark your own corpus, drop `corpus.jsonl`, `queries.jsonl`, and `qrels.trec` into `data/<name>/` and add `<name>` to the manifest.
