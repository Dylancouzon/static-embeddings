# Static embeddings in fastembed — viability decision

**Verdict: no-go.** Static embeddings do not clear the pre-registered viability bar. They fail V1 — they are *worse* than BM25 on code search, the use case that motivated the request (0.289 vs 0.296 NDCG@10; paired bootstrap 95% CI [−0.013, −0.001], excludes zero). CLAUDE.md's V1 rule is explicit about the consequence: "ship nothing and update the docs to recommend BM25 for fast local indexing." That is the recommendation.

The speed claims are real (V3, V4 pass) and the general-quality floor is met (V2 passes), so this is not "the technology doesn't work" — it is "it does not win the job it was proposed for, and a free baseline we already ship beats it there." A general-retrieval hybrid path exists but is weak and would be a **new, unregistered decision** requiring your sign-off (see the last section) — it is not part of this verdict.

Environment: Apple M5 Pro, 24 GB, macOS 26.4.1, Python 3.14.6, OMP/ORT=4 threads (numpy BLAS = Accelerate — see caveats), Qdrant v1.15.1 (digest-pinned). Warm OS cache. All quality via Qdrant Query API with `exact=True`; local brute-force reproduces Qdrant ranking to 0.0 NDCG delta. Medians over ≥3 runs. Viability model: `potion-retrieval-32M`.

## Threshold scorecard (mirrors results/RESULTS.md)

| # | Threshold | Result | Verdict |
|---|-----------|--------|---------|
| Gate | A (numpy port) == model2vec within 1e-5 + edges | max abs diff 3e-8 | ✅ PASS |
| V1 | retrieval-32M NDCG@10 ≥ BM25 on code | 0.289 vs 0.296 (significant loss) | ❌ FAIL |
| V2 | retrieval-32M ≥ 75% MiniLM, BEIR avg | 0.379 vs 0.435 = 87% | ✅ PASS |
| V3 | ≥20× embed throughput vs ONNX MiniLM @32 | 9121 vs 301 = 30.3× | ✅ PASS |
| V4 | model-load ≤200ms warm + total cold-start under ONNX | load 26ms; total 0.204s < 0.253s | ✅ PASS |
| Workflow | hybrid ≥90% MiniLM on both tracks | BEIR 99.5%; code 64.6% | ❌ FAIL |
| Approach | ship A if gate + within 10% of model2vec throughput | 9121 vs 9145 = 99.7% | ✅ PASS |

Viability requires all of V1–V4. V1 fails, so the verdict is no-go regardless of the others.

## Quality — NDCG@10 (exact search)

| System | scifact | nfcorpus | fiqa | code | BEIR avg |
|--------|---------|----------|------|------|----------|
| BM25 (sparse, no download, 0 deps) | 0.683 | 0.324 | 0.243 | 0.296 | 0.417 |
| MiniLM-L6-v2 (ONNX dense) | 0.624 | 0.315 | 0.366 | 0.555 | 0.435 |
| bge-small-en-v1.5 (ONNX dense) | 0.720 | 0.339 | 0.385 | 0.674 | 0.481 |
| static retrieval-32M | 0.638 | 0.307 | 0.190 | 0.289 | 0.379 |
| static base-8M | 0.505 | 0.244 | 0.166 | 0.282 | 0.305 |

**BM25 beats static-alone on every dataset** — including throughput (15.9k vs 9.1k docs/s), with no 129 MB download and zero dependencies. Static's "87% of MiniLM" (V2) is true but not the decision-relevant comparison: against the free baseline we already ship, static-alone loses everywhere. Static collapses on fiqa (0.190 vs MiniLM 0.366) — averaging word-vectors has no context, so semantic-match datasets hurt it. On code, static and BM25 both trail dense by ~2× (0.29 vs 0.55–0.67): code retrieval rewards learned semantics neither token-overlap method has.

## Speed & footprint

| System | throughput @32 (docs/s) | model-load (ms) | total cold-start (s) | query p50 (ms) | on-disk | deps added |
|--------|------------------------|-----------------|----------------------|----------------|---------|------------|
| static retrieval-32M (A) | 9121 | 26 | 0.204 | 0.02 | 129 MB | 0 |
| static base-8M (A) | 9607 | 13 | 0.196 | 0.02 | 30 MB | 0 |
| model2vec (B) | 9145 | 146 | 0.288 | 0.03 | 129 MB | model2vec, safetensors |
| potion via ONNX (C) | 6786 | 54 | 0.253 | 0.03 | 129 MB | onnxruntime |
| MiniLM (ONNX) | 301 | 41 | 0.440 | 3.89 | 90 MB | onnxruntime |
| bge-small (ONNX) | 31 | 52 | 1.398 | 2.07 | 133 MB | onnxruntime |
| BM25 | 15944 | 44 | 0.300 | 0.01 | none | none |

The mmap claim holds: A loads the 32M model in 26 ms — fastest of all, including model2vec (146 ms) and ONNX (41–54 ms) — and its total cold-start (0.204 s) is under every ONNX system (V4 passes). Static's speed is genuine; it just does not buy enough quality to matter against BM25.

## Use case → recommendation

| Use case | Recommendation | Why |
|----------|----------------|-----|
| **Code search** (the motivating use case) | **BM25** for local/dependency-free; **bge-small** for quality | Static loses to BM25 (significant) and both trail dense ~2×; static adds a 129 MB download for a loss |
| **General retrieval, max quality** | **bge-small (ONNX dense)** | Best on every general track (BEIR 0.481) |
| **General retrieval, speed/cost critical** | **BM25 alone** | Faster than static (15.9k vs 9.1k docs/s), higher BEIR (0.417 vs 0.379), zero download |
| **Semantic-match data (fiqa-like)** | **Dense**, never static | Static collapses (0.190 vs 0.366) |
| Static embeddings | **No first-choice slot** | Only conceivable role is the dense half of a hybrid — which a dense hybrid does better (below) |

## Implementation choice (only if shipped despite the no-go)

**Candidate A — the native numpy port.** Passes the correctness gate (matches model2vec to 3e-8 incl. edge suite), adds **zero runtime dependencies** (numpy, tokenizers, huggingface-hub are already fastembed deps), loads via mmap in 26 ms, and after adopting `encode_batch_fast` reaches 99.7% of model2vec throughput. The ONNX path (C) is slower and pulls onnxruntime — no reason to prefer it. Follow `fastembed/sparse/minicoil.py` (mmap) and `fastembed/sparse/bm25.py` (class shape).

## Unregistered observation — the hybrid path (needs your sign-off before it becomes a decision)

The pre-registered "workflow inclusion" test (hybrid ≥90% MiniLM on *both* tracks) failed on code (64.6%). Shipping static as a hybrid component is therefore a new decision, not blessed by the thresholds. The data does not make a strong case for it:

| BEIR avg | code | fiqa | |
|---|---|---|---|
| BM25 alone | 0.417 | 0.296 | 0.243 |
| static + BM25 (RRF) | 0.433 | 0.359 | 0.254 |
| dense (MiniLM) + BM25 (RRF) | **0.483** | **0.531** | **0.380** |

- Static + BM25 adds only **+0.016 BEIR** over BM25 alone — the entire quality payoff for shipping a 129 MB model into the pipeline.
- A **dense + BM25 hybrid beats static + BM25 on every dataset** (BEIR 0.483 vs 0.433; fiqa 0.380 vs 0.254). If you will run a hybrid at all, the dense one is better.
- "Static + BM25 matches MiniLM" is only true as a BEIR *average* (0.433 vs 0.435); on fiqa it is 69% of MiniLM. The average is carried by the lexical-friendly sets where BM25 already matches MiniLM.

The one honest pitch for static + BM25: it reaches MiniLM-alone's *average* BEIR quality at ~30× the embed throughput and zero added dependencies. If a use case is (a) general text, (b) throughput-bound, (c) tolerant of weak semantic matching, and (d) unable to afford ONNX, static + BM25 is defensible. That is a narrow slot, and it is your call — not the benchmark's.

## Caveats

- **Cold-cache regime pending.** All startup numbers are warm OS cache (your call to defer). mmap defers page-in to first batch, so the cold-cache phase split should be measured before any final V4 claim.
- **V4 threshold history.** CLAUDE.md says only "meaningfully under" (no number). This report uses the registered strict-under-fastest-ONNX reading (passes). An earlier draft tightened this to ≤75% mid-project without sign-off; that was reverted. If you want a stricter margin, say so — `05_report.py` re-runs in seconds.
- **base-8M (the intended default) is materially weaker** than retrieval-32M (BEIR 0.305 vs 0.379). Any static discussion should lead with retrieval-32M.
- **bge-small throughput anomaly:** 90 docs/s @batch 1 → 31 @batch 32 (batching should not cost 3×; likely pad-to-longest). Immaterial to any threshold (V3 is vs MiniLM) but do not quote the D2 throughput row without re-checking.
