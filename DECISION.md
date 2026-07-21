# Static Embeddings In Fastembed: Decision

**Verdict: no-go.** Do not ship static embeddings in fastembed for the tested
English and code-retrieval workloads.

`minishlab/potion-retrieval-32M`, the best static model in this benchmark, fails
the pre-registered viability check: it scores below BM25 on CodeSearchNet Python,
the workload that motivated the request. BM25 already exists in fastembed,
downloads nothing, embeds faster, and also beats static on the BEIR tracks.

Static embeddings are not broken. The numpy implementation matches model2vec,
loads quickly, and embeds about 30x faster than ONNX MiniLM. The issue is that
static has no first-choice product slot: BM25 wins the cheap path, and dense
models win the quality path.

## Scorecard

Viability required all V1-V4 checks. V1 fails, so the shipping decision is no-go.

| Check | Requirement | Result | Verdict |
|---|---|---:|---|
| Correctness | numpy port matches model2vec | max diff `3e-8` | pass |
| V1: code search | static >= BM25 | 0.2887 vs 0.2955 | **fail** |
| V2: BEIR floor | static >= 75% of MiniLM | 87.0% | pass |
| V3: throughput | static >= 20x MiniLM at batch 32 | 30.3x | pass |
| V4: startup | mmap load <= 200 ms, total below ONNX | 26 ms, yes | pass |
| Workflow | static+BM25 >= 90% of MiniLM on BEIR and code | 99.5%, 64.6% | **fail** |
| Approach | numpy port within 10% of model2vec throughput | 99.7% | pass |

## Main Evidence

NDCG@10, exact Qdrant search:

| Dataset | BM25 | Static | MiniLM | bge-small |
|---|---:|---:|---:|---:|
| SciFact | 0.6830 | 0.6378 | 0.6239 | **0.7203** |
| NFCorpus | 0.3240 | 0.3073 | 0.3148 | **0.3387** |
| FiQA | 0.2433 | 0.1903 | 0.3663 | **0.3848** |
| CodeSearchNet Python | 0.2955 | 0.2887 | 0.5547 | **0.6742** |

Static never beats BM25. On the code track, paired bootstrap over 20k queries
puts static minus BM25 at `[-0.0128, -0.0009]` for the 95% interval, so the small
gap is still real.

Dense retrieval is in a different tier for code and semantic-heavy data. bge-small
scores 0.6742 on CodeSearchNet, more than 2x static. MiniLM and bge-small also
dominate FiQA, where exact lexical overlap is weaker.

## Why The Code-Specific Escape Hatch Failed

The obvious objection is that `potion-retrieval-32M` is a general retrieval
model. We tested that objection by distilling a code-search teacher,
`flax-sentence-embeddings/st-codesearch-distilroberta-base`, into a static model
and re-running CodeSearchNet.

| Code model | Standalone | With BM25 |
|---|---:|---:|
| BM25 | 0.2955 | - |
| General static, potion-retrieval-32M | 0.2887 | 0.3585 |
| Code-teacher static | 0.0556 | 0.2471 |

The code-teacher static model is much worse, and its BM25 hybrid falls below BM25
alone. This points at a representation limit rather than a missing teacher:
static embeddings keep one context-free vector per token and average the result.
Code search depends heavily on composition, ordering, and structure; averaging
throws too much of that away.

## Hybrid Does Not Rescue The Product Case

NDCG@10:

| System | BEIR avg | CodeSearchNet Python | FiQA |
|---|---:|---:|---:|
| BM25 | 0.4168 | 0.2955 | 0.2433 |
| Static + BM25 | 0.4329 | 0.3585 | 0.2540 |
| MiniLM + BM25 | 0.4829 | 0.5312 | 0.3804 |
| bge-small + BM25 | **0.4926** | **0.6033** | 0.3757 |

Static+BM25 is better than BM25 alone, but it does not reach the pre-registered
workflow bar. It is close on BEIR and far behind on code. If a user can afford a
dense model, dense hybrid is the stronger default; if they cannot, BM25 is still
the simpler fast path.

## Recommendation Matrix

| Need | Recommend | Reason |
|---|---|---|
| Fast local English retrieval | BM25 | Better than static, faster than static, no download |
| Code search, lowest cost | BM25 | Beats static on the motivating workload |
| Code search, quality | bge-small or another dense code-capable model | Dense scores roughly 2x static |
| General retrieval, quality | bge-small | Best standalone and hybrid results here |
| Semantic-heavy data like FiQA | Dense | Static drops hardest when lexical overlap is weak |
| No ONNX runtime, no model server, BM25 inadequate | Static only as a niche option | Speed is real, but benchmark support is narrow |

Static has no recommended default slot for this scope.

## If Static Ships Anyway

Ship the numpy port, not the model2vec dependency and not the potion ONNX path.

Reasons:

- It matches model2vec on potion models to `3e-8`.
- It adds no new fastembed runtime dependency beyond numpy, tokenizers, and
  huggingface-hub.
- It uses mmap and loads `potion-retrieval-32M` in 26 ms warm-cache.
- It reaches 99.7% of model2vec throughput at batch 32.
- The ONNX path is slower and pulls onnxruntime into the static path.

Implementation shape should follow fastembed's sparse model classes, with mmap
precedent from MiniCOIL. Keep the correctness gate as a required test: static
tokenization details are easy to get subtly wrong.

Important caveat: the port matches model2vec on potion tokenizers, but the
code-teacher experiment showed `5.7e-4` max drift on a byte-level BPE tokenizer.
Do not ship a non-potion static model until tokenizer parity is fixed and gated.

## Untested Cases

**Cross-lingual retrieval remains open.** BM25 cannot match a query and document
that share no terms across languages, while a multilingual static model might.
`potion-multilingual-128M` was measured only for load and size, not quality. A
separate cross-lingual benchmark would be needed before making a decision for
that use case.

**Cold-cache startup was skipped.** V4 uses warm OS cache measurements. The mmap
load phase is isolated and comfortably below the threshold, but cold-cache claims
should not be made from this run.

**NIFE / pynife static models were not tested.** `pynife` (model2vec's author)
distills a teacher into a static model with a query-focused recipe. Pretrained
checkpoints exist on Hugging Face under `stephantulkens`, one in model2vec format
that loads here with no training. Inference is identical to candidate A, so V3/V4
are unaffected. Worth a cheap re-run on the semantic tracks where static drops
hardest (FiQA). It will not move the code verdict: the escape-hatch result above
shows the failure is static's averaging ceiling, which no training recipe removes.

## Reproduce And Audit

```bash
docker compose up -d
uv sync
uv run python run_all.py
```

Full generated results live in [results/RESULTS.md](results/RESULTS.md). Extra
checks are in [results/extra.json](results/extra.json),
[results/code_teacher.json](results/code_teacher.json), and
[results/correctness.json](results/correctness.json).

Environment for the recorded run: Apple M5 Pro, 24 GB RAM, Darwin 26.4.1,
Python 3.14.6, Qdrant 1.15.1 pinned by digest, OMP 4, ORT intra-op 4.
