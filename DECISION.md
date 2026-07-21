# Static embeddings in fastembed: viability decision

**Verdict: no-go.** The best static model loses to BM25 on code search, the workload that motivated the request. BM25 already ships in fastembed, downloads nothing, and adds no dependencies. Static gives up quality and returns nothing BM25 lacks.

This is not "the technology is broken." Static is fast (30× the embed throughput of ONNX MiniLM, 26 ms to load) and its general-retrieval quality clears the floor we set (87% of MiniLM on BEIR). The failure is specific: on the one job that justified the feature, a free baseline wins.

## Scorecard

Thresholds were fixed before any results existed. Viability required all of V1–V4.

| Check | Target | Result | Verdict |
|---|---|---|---|
| Correctness | numpy port matches model2vec | 3e-8 | pass |
| V1 | static ≥ BM25 on code | 0.289 vs 0.296 | **fail** |
| V2 | ≥ 75% of MiniLM on BEIR | 87% | pass |
| V3 | ≥ 20× MiniLM embed throughput | 30× | pass |
| V4 | mmap load ≤ 200 ms, cold-start under ONNX | 26 ms, yes | pass |
| Workflow | hybrid ≥ 90% of MiniLM on both tracks | BEIR 99%, code 65% | **fail** |
| Approach | numpy port within 10% of model2vec | 99.7% | pass |

V1 fails, so the verdict is no-go whatever the other rows say.

## Best static vs BM25, NDCG@10

| Dataset | BM25 | potion-retrieval-32M |
|---|---|---|
| SciFact | 0.683 | 0.638 |
| NFCorpus | 0.324 | 0.307 |
| FiQA | 0.243 | 0.190 |
| Code | 0.296 | 0.289 |

BM25 wins every track. The code gap is small but real: a paired bootstrap over 20k queries puts the 95% interval at [-0.013, -0.001], entirely below zero. Static also drops sharply on FiQA, where matching is semantic rather than lexical.

Neither method is good at code. Dense models are: bge-small scores 0.674, MiniLM 0.555. Code search needs learned semantics that averaging word vectors cannot produce.

## When to use what

| Need | Use | Why |
|---|---|---|
| Code search | BM25 for local, bge-small for quality | Static loses to BM25 and trails dense 2× |
| General retrieval, best quality | bge-small | Highest score on every general track |
| General retrieval, speed first | BM25 | Faster than static, higher BEIR, no download |
| Semantic-heavy data (FiQA-like) | Dense | Static collapses without context |

Static has no first-choice slot. Its only plausible role is the dense half of a hybrid, and a dense hybrid fills that better.

## Implementation, if it ships anyway

Ship the numpy port. It matches model2vec to 3e-8, adds no dependencies (numpy, tokenizers, huggingface-hub are already fastembed deps), loads by mmap in 26 ms, and reaches 99.7% of model2vec throughput once it uses `encode_batch_fast`. The ONNX path is slower and pulls onnxruntime. Precedents to follow: `fastembed/sparse/minicoil.py` for mmap, `fastembed/sparse/bm25.py` for the class shape.

## The hybrid path (not recommended)

The pre-registered hybrid test failed on code, so shipping static inside a hybrid is a separate decision. The numbers do not support it.

| System | BEIR avg | Code | FiQA |
|---|---|---|---|
| BM25 | 0.417 | 0.296 | 0.243 |
| static + BM25 | 0.433 | 0.359 | 0.254 |
| dense + BM25 | 0.483 | 0.531 | 0.380 |

Static adds 0.016 BEIR over BM25 alone. A dense hybrid adds four times that and wins every track. The one point in static's favor is speed: static + BM25 reaches MiniLM's average BEIR score at 30× the throughput with no added dependency. That fits a narrow case (general text, throughput-bound, tolerant of weak semantic matching, no room for onnxruntime). It is a judgment call, not a benchmark result.

## Notes

- Measured on Apple M5 Pro, macOS 26.4.1, Qdrant 1.15.1. Every quality number runs through Qdrant with `exact=True`; a local brute-force top-k reproduces the ranking, so scores reflect the model, not the index.
- Startup numbers use a warm OS cache. Cold-cache runs were skipped.
- Full tables: `results/RESULTS.md` and `results/extra.json`. Reproduce with `run_all.py`.
- bge-small throughput drops from 90 to 31 docs/s between batch 1 and 32, likely pad-to-longest. It touches no threshold; recheck before quoting it.
