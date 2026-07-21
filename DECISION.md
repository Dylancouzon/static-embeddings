# Static embeddings in fastembed: viability decision

**Verdict: no-go.** The best static model (potion-retrieval-32M) loses to BM25 on code search — the workload that motivated the request — and BM25 already ships in fastembed with no download and no dependencies. The technology itself works: static embeds 30× faster than ONNX MiniLM, loads in 26 ms, and clears the general-quality floor (87% of MiniLM on BEIR). It fails on the one job that justified building it.

## Scorecard

Thresholds were fixed before any results existed; viability required all of V1–V4. V1 fails, so the verdict is no-go regardless of the rest.

| Check | Target | Result | Verdict |
|---|---|---|---|
| Correctness | numpy port matches model2vec | 3e-8 | pass |
| V1 | static ≥ BM25 on code | 0.289 vs 0.296 | **fail** |
| V2 | ≥ 75% of MiniLM on BEIR | 87% | pass |
| V3 | ≥ 20× MiniLM embed throughput | 30× | pass |
| V4 | mmap load ≤ 200 ms, cold-start under ONNX | 26 ms, yes | pass |
| Workflow | hybrid ≥ 90% of MiniLM on both tracks | BEIR 99%, code 65% | **fail** |
| Approach | numpy port within 10% of model2vec throughput | 99.7% | pass |

## Why static loses

**Standalone, BM25 wins every track** (NDCG@10):

| Dataset | BM25 | static | MiniLM | bge |
|---|---|---|---|---|
| SciFact | 0.683 | 0.638 | 0.624 | 0.720 |
| NFCorpus | 0.324 | 0.307 | 0.315 | 0.339 |
| FiQA | 0.243 | 0.190 | 0.366 | 0.385 |
| Code | 0.296 | 0.289 | 0.555 | 0.674 |

Static never beats BM25, and trails the dense models by ~2× on code and FiQA. The code gap is small but real — a paired bootstrap over 20k queries puts the 95% interval at [-0.013, -0.001], entirely below zero. Static drops hardest on FiQA, where matching is semantic rather than lexical.

**A code-specific model makes it worse, not better.** The obvious objection is that potion-retrieval-32M is a general-text distillation. So we distilled a code-retrieval teacher — `st-codesearch-distilroberta-base`, fine-tuned on CodeSearchNet — into a static model and re-ran the code track.

| Model on code | Standalone | + BM25 hybrid |
|---|---|---|
| BM25 (incumbent) | 0.296 | — |
| general potion-retrieval-32M | 0.289 | 0.359 |
| code-teacher static | 0.056 | 0.247 |

The code teacher is 5× worse standalone, and its hybrid falls below BM25 alone — a weak dense signal drags RRF fusion down. This is a technology ceiling, not a model choice. Distillation keeps one context-free vector per token and averages them; code retrieval lives in composition, which averaging discards. The general model scores higher only because it soft-matches English subwords shared between docstrings and identifier names, and even that loses to BM25's exact matching.

**Hybrid does not rescue it** (NDCG@10):

| System | BEIR avg | Code | FiQA |
|---|---|---|---|
| BM25 | 0.417 | 0.296 | 0.243 |
| static + BM25 | 0.433 | 0.359 | 0.254 |
| MiniLM + BM25 | 0.483 | 0.531 | 0.380 |
| bge + BM25 | 0.493 | 0.603 | 0.376 |

static + BM25 beats BM25 on every track but loses to a dense hybrid everywhere; the stronger dense model (bge) widens the code gap to 0.603 vs 0.359. Static's one edge is cost: static + BM25 matches MiniLM's BEIR average at 30× the throughput with no added dependency. That fits a narrow niche — general text, throughput-bound, no room for onnxruntime — not a benchmark pass.

## When to use what

| Need | Use | Why |
|---|---|---|
| Code search, local | BM25 | Beats static, no download, no deps |
| Code search, quality | bge-small | Dense leads by ~2× |
| General retrieval, quality | bge-small | Highest on every general track |
| General retrieval, speed first | BM25 | Faster than static, higher BEIR, no download |
| Semantic-heavy data (FiQA-like) | Dense | Static collapses without context |

Static has no first-choice slot: BM25 sits above it on cost, dense above it on quality.

## Implementation, if it ships anyway

Ship the numpy port. It matches model2vec to 3e-8 on potion, adds no dependencies (numpy, tokenizers, huggingface-hub are already fastembed deps), loads by mmap in 26 ms, and reaches 99.7% of model2vec throughput with `encode_batch_fast`. The ONNX path is slower and pulls onnxruntime. Follow `fastembed/sparse/minicoil.py` for mmap and `fastembed/sparse/bm25.py` for the class shape.

One caveat for any non-potion model: the port matches model2vec to 3e-8 on potion's tokenizer but drifts 5.7e-4 on a byte-level BPE tokenizer (its `encode_batch_fast` path doesn't reproduce model2vec's BPE tokenization). Ship a non-potion static model only after a tokenizer-parity fix.

## What we didn't test

- **Multilingual / cross-lingual retrieval — the one untested case where static could beat BM25.** BM25 matches tokens, so it scores near zero across languages (a French query and English docs share no terms); a multilingual static model embeds both into one space and can match. `potion-multilingual-128M` was loaded for speed/size only. Recommended check before closing the file: potion-multilingual vs BM25 on a cross-lingual set. It is outside the pre-registered English scope, so it needs sign-off and a dataset wired.
- **Cold-cache startup.** Warm cache only; cold runs skipped.

## Notes

- Apple M5 Pro, macOS 26.4.1, Qdrant 1.15.1. Every quality number runs through Qdrant with `exact=True`; a local brute-force top-k reproduces the ranking, so scores reflect the model, not the index.
- Full tables: `results/RESULTS.md`, `results/extra.json`, `results/code_teacher.json`. Reproduce with `run_all.py`.
