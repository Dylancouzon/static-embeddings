# Static Embeddings Viability Benchmark

Decision harness for whether [fastembed](https://github.com/qdrant/fastembed)
should ship static embeddings such as model2vec / potion.

**Verdict: do not ship.** The best static model tested,
`minishlab/potion-retrieval-32M`, does not beat BM25 on English retrieval or on
CodeSearchNet, the motivating code-search workload. BM25 already ships in
fastembed, needs no model download, embeds faster, and scores higher.

The speed claim is real: static embeddings are about 30x faster than ONNX MiniLM
and load by mmap in 26 ms. The problem is product fit, not implementation. Static
is faster than dense, but BM25 is cheaper and better; dense is slower, but much
better where quality matters.

See [DECISION.md](DECISION.md) for the full recommendation and
[results/RESULTS.md](results/RESULTS.md) for generated numbers.

## Headline Results

NDCG@10, exact Qdrant search:

| Dataset | BM25 | Static, potion-retrieval-32M | MiniLM | bge-small |
|---|---:|---:|---:|---:|
| SciFact | 0.6830 | 0.6378 | 0.6239 | **0.7203** |
| NFCorpus | 0.3240 | 0.3073 | 0.3148 | **0.3387** |
| FiQA | 0.2433 | 0.1903 | 0.3663 | **0.3848** |
| CodeSearchNet Python | 0.2955 | 0.2887 | 0.5547 | **0.6742** |

Static loses to BM25 on every tested track. On code search, the gap is small but
statistically real: paired bootstrap over 20k queries gives a 95% interval of
`[-0.0128, -0.0009]` for static minus BM25.

Hybrid retrieval improves static, but not enough:

| System | BEIR avg | CodeSearchNet Python |
|---|---:|---:|
| BM25 | 0.4168 | 0.2955 |
| Static + BM25 | 0.4329 | 0.3585 |
| MiniLM + BM25 | 0.4829 | 0.5312 |
| bge-small + BM25 | **0.4926** | **0.6033** |

## Speed

Median embedding throughput at batch 32:

| System | Docs/s | Warm load | Query p50 |
|---|---:|---:|---:|
| BM25 | **15,944** | 44 ms | 0.01 ms |
| Static, potion-retrieval-32M | 9,121 | **26 ms** | 0.02 ms |
| MiniLM ONNX | 301 | 41 ms | 3.89 ms |
| bge-small ONNX | 31 | 52 ms | 2.07 ms |

Static is a credible implementation technique. It just does not create a better
fastembed option for the tested English and code-retrieval workloads.

## What This Repo Tests

- **Systems:** BM25, fastembed ONNX dense baselines, static via a dependency-light
  numpy port, model2vec reference implementation, potion ONNX, and hybrid
  BM25+static retrieval.
- **Datasets:** BEIR SciFact, NFCorpus, FiQA, and CodeSearchNet Python
  docstring-to-function retrieval.
- **Quality path:** embed, upsert to Qdrant, query with `exact=True`, then score
  with `ranx` against qrels.
- **Correctness checks:** the numpy static port matches model2vec to `3e-8`; a
  local brute-force top-k check reproduces Qdrant exact rankings with `0.0`
  NDCG delta.
- **Pre-registered thresholds:** correctness, BM25 parity on code, BEIR quality
  floor, throughput, startup time, hybrid usefulness, and implementation parity.

## Repo Layout

| Path | Purpose |
|---|---|
| [DECISION.md](DECISION.md) | Final verdict, recommendation matrix, caveats |
| [bench.toml](bench.toml) | Systems, models, datasets, thread settings |
| [run_all.py](run_all.py) | Full pipeline orchestrator |
| [01_download_data.py](01_download_data.py) - [05_report.py](05_report.py) | Pipeline stages |
| [correctness_gate.py](correctness_gate.py) | Hard-stop parity check for the numpy port |
| [static_numpy.py](static_numpy.py) | Candidate dependency-light static embedder |
| [systems.py](systems.py) | Embedders behind one `embed()` shape |
| [results/RESULTS.md](results/RESULTS.md) | Generated report |

## Reproduce

```bash
docker compose up -d
uv sync
uv run python run_all.py
```

`run_all.py` runs the correctness gate before benchmarking. If the gate fails,
the run stops before any quality or speed numbers are produced.

To add a dataset, create `data/<name>/corpus.jsonl`,
`data/<name>/queries.jsonl`, and `data/<name>/qrels.trec`, then add the dataset
to [bench.toml](bench.toml).
