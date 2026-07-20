# Static Embeddings Viability Benchmark

This repo answers two questions, in order:

1. **Viability:** are static embeddings worth shipping in fastembed at all — do they beat the baselines users already have (BM25, small ONNX models) by enough, on the use cases that motivate them? And if yes, **when** should a user pick them over the alternatives?
2. **Approach:** if viable, which implementation should fastembed ship?

It is an internal decision benchmark, not a publication. The deliverable is `DECISION.md` containing: a go/no-go on shipping, a use-case → recommendation matrix ("when to use static vs BM25 vs ONNX dense vs hybrid"), and the implementation choice. The viability verdict may legitimately be "no" — a benchmark that can only say yes is worthless.

## Decision thresholds — pre-registered, read before running anything

These are set before any results exist so the numbers decide, not the narrative. Adjust them only with Dylan's sign-off, never after seeing results they would flip.

**Viability (ship it) requires ALL of:**
- **V1 — beats lexical where it claims to:** potion-retrieval-32M NDCG@10 ≥ BM25 NDCG@10 on the code-search track. If a static dense model can't beat BM25 on its motivating use case, ship nothing and update the docs to recommend BM25 for fast local indexing.
- **V2 — quality floor vs dense:** potion-retrieval-32M ≥ 75% of MiniLM-L6-v2's NDCG@10 averaged over the BEIR tracks.
- **V3 — the speed claim is real vs our own stack:** ≥ 20× embed throughput vs fastembed ONNX MiniLM at batch 32, same thread settings.
- **V4 — the startup claim is real:** model-load phase (mmap + tokenizer, excluding interpreter/import time) ≤ 200 ms warm-cache, and total cold-start meaningfully under the ONNX candidates'.

**Workflow inclusion (recommend in normal pipelines, not just niche):** hybrid static + BM25 (Qdrant server-side RRF fusion) reaches ≥ 90% of MiniLM's NDCG@10 on both tracks. If hybrid closes the gap, static graduates from "niche speed tool" to "default cheap pipeline"; if not, it stays scoped to speed-critical local indexing.

**Approach choice (only if viable):** ship candidate A if it passes the correctness gate and is within 10% of B's throughput; otherwise escalate — that result would contradict the research and needs investigation before any decision.

## Context

- Research report (July 2026): https://claude.ai/code/artifact/bf94232c-1b36-40cd-8818-5bf18eadf17f
- The feature request this feeds: https://github.com/qdrant/fastembed/issues/388
- fastembed source for reference: `../fastembed` (static class would follow `fastembed/sparse/bm25.py`; mmap precedent in `fastembed/sparse/minicoil.py`)
- fastembed team constraints: no new dependencies (numpy, tokenizers, huggingface-hub, onnxruntime available; pytorch, safetensors, sentence-transformers NOT), fast model load, ideally mmap.

## Systems under test

Baselines (the incumbents a user already has in fastembed today):

| ID | System | Why it's here |
|----|--------|---------------|
| BM25 | fastembed's own BM25 (sparse, no model download) | The kill-shot baseline: also fast, also dependency-free. Static must beat it somewhere to justify existing. |
| D1 | `all-MiniLM-L6-v2` via fastembed ONNX | Today's "fast dense" default |
| D2 | `BAAI/bge-small-en-v1.5` via fastembed ONNX | Today's quality-per-size default |

Static candidates:

| ID | System | Role |
|----|--------|------|
| A | Native numpy port: hand-parsed safetensors header + `np.memmap` matrix + tokenizers + mean-pool + L2 normalize | What fastembed would ship |
| B | `model2vec` library (`StaticModel.from_pretrained`) | Reference implementation: correctness oracle and speed baseline for A |
| C | Potion ONNX export (`onnx/model.onnx` in each potion repo) via onnxruntime | The "reuse fastembed's existing path" alternative |

Combination:

| ID | System | Role |
|----|--------|------|
| H | Hybrid: static (A) + BM25, fused server-side in Qdrant (Query API prefetch + RRF) | The workflow-inclusion question: does hybrid close the quality gap to dense? |

Candidate A rules:
- May import only numpy, tokenizers, huggingface_hub. Run with an import guard proving torch, safetensors, and model2vec never load in its process.
- safetensors parsing: 8-byte little-endian u64 header length, JSON header with per-tensor dtype/shape/data_offsets, then flat buffer. Serve the matrix with `np.memmap` at offset `8 + header_len + tensor_offset`.
- Replicate model2vec inference exactly: `encode_batch` with `add_special_tokens=False`, drop `unk_token_id` tokens, gather rows, mean over axis 0, zeros vector for empty token lists, L2 normalize with `+1e-32` guard, honor `normalize` from config.json. No tokenizer truncation, no padding (potion configs set `seq_length: 1000000`).

**Correctness gate before any benchmarking:** A's vectors must match B's within `np.allclose(atol=1e-5)` on 1k sampled documents per model PLUS a fixed edge-case suite: empty string, whitespace-only, single token, unknown-token-heavy text, a >10k-token document, code snippets, emoji/non-Latin text. Also record A-vs-C and B-vs-C vector parity (cosine similarity distribution) — C is a different tokenization/graph path and any drift must be known, not assumed away. Re-run the gate after every change to A.

## Models under test

- `minishlab/potion-base-8M` (256-d) — the intended fastembed default
- `minishlab/potion-retrieval-32M` (512-d) — best static retrieval quality; the model the viability thresholds are evaluated on
- `potion-multilingual-128M`: load-time/size check only; quality workloads are English.

## Workloads

1. **General retrieval:** SciFact, NFCorpus, FiQA (BEIR loaders or HF `BeIR/*`).
2. **Code search (the motivating use case):** CodeSearchNet Python split, ~20k functions. Specify exactly: corpus = function code strings; queries = docstring first paragraphs; qrels = the CSN-provided query→function pairs, one relevant doc per query, serialized to TREC-format qrels for `ranx`. Write this mapping down in the script before running — vague CSN handling produces unusable qrels.

Quality runs go end-to-end through Qdrant (embed → upsert → query → score). Additionally, run a local brute-force cosine top-k on one dataset per model and confirm it reproduces the Qdrant exact-search ranking — this separates embedding-quality effects from client/scoring effects. BM25 quality runs use Qdrant sparse vectors with fastembed's BM25 output.

## Metrics

Quality (per system × dataset): NDCG@10 and Recall@10 via `ranx` against qrels, with `exact=True` on every Qdrant query so ANN recall never confounds model quality. (Refs: https://qdrant.tech/documentation/tutorials-search-engineering/ann-recall/, https://qdrant.tech/documentation/improve-search/retrieval-relevance/)

Speed (per system × model, same corpus, same thread settings):
- **Cold start, phase-split** — report each phase separately, not one blended number: interpreter+imports / model download-cache check / model load (mmap or ORT session init + tokenizer) / first batch / second batch. Two regimes, both labeled: warm OS file cache (5 fresh processes, back to back) and cold cache (after `purge` or a machine reboot, 3 runs). The mmap claim lives or dies on the model-load phase, so it must be isolated.
- **Embed throughput:** docs/sec at batch sizes 1, 8, 32, 128 and fastembed's default (256). Embed-only, no Qdrant in the loop.
- **End-to-end ingest:** embed + upsert wall-clock at one fixed configuration (batch 128, 2 parallel streams) — fixed, not a range, so systems are comparable. Note that upsert time excludes deferred HNSW builds (`indexing_threshold_kb` default 20 MB); quality runs use exact search so this doesn't matter there.
- **Query latency:** single-query embed time, p50/p95 over 1k queries.
- **Peak RSS** during the embed run (`psutil`).
- Report median and IQR over ≥3 runs for every speed number, never a single run.

Environment controls, pinned and recorded in every results file: chip, RAM, macOS version, power source (plugged in, low-power mode off), `OMP_NUM_THREADS`, `TOKENIZERS_PARALLELISM`, ORT session thread settings (intra/inter op), package versions, Qdrant image digest. Numbers from different machines or different env settings never share a table.

Footprint (per system — this is a shipping decision, not just a speed contest): model artifact size on disk, files downloaded, transitive runtime deps added to fastembed (must be zero for A), and a one-line maintenance note (who owns the code path).

## Qdrant usage — do it the way we'd tell a customer to

- Qdrant in Docker, image pinned by digest in `docker-compose.yml` (not `latest`). Record the version in results.
- Python client `qdrant-client`, Query API (`query_points`) only — never the deprecated `search`.
- One collection per model (dims differ); cosine distance; default HNSW config (irrelevant for quality runs — those are exact).
- Hybrid (H): named vectors (dense + sparse) in one collection, server-side fusion via Query API `prefetch` + RRF. This is Qdrant's own fusion, not client-side merging.
- Bulk upload via `upload_points` at the fixed batch/parallelism above. (Ref: https://qdrant.tech/documentation/database-tutorials/bulk-upload/)
- Never call Qdrant a "vector database" in anything written here — it is a vector search engine.

## Repo conventions

- Python 3.11+, `uv`, all versions pinned in `pyproject.toml`, `ruff`.
- One `bench.toml` manifest defines systems × models × datasets × batch sizes; stage scripts read it rather than growing per-combination flags: `01_download_data.py`, `02_embed.py`, `03_index.py`, `04_evaluate.py`, `05_report.py`. Each writes JSON to `results/`, keyed by system/model/dataset.
- `05_report.py` renders `results/RESULTS.md` — generated tables plus a pass/fail line for each pre-registered threshold. Never hand-edited.
- Seed anything stochastic; embedding and evaluation must be deterministic re-runs.
- Torch may exist in the env (B/D tooling), but candidate A's measured process must prove it never loads torch/safetensors/model2vec.

## Working rules for Claude sessions here

- Keep it lean: a decision harness, not a framework. Five systems with the same `embed(texts) -> vectors` signature and a manifest is the whole architecture.
- Correctness gate first, always, after any change to A.
- The thresholds above are the verdict. If a number surprises you in either direction, check the harness (batching, cache regime, thread settings, exact search actually on) before believing it — then believe it, including when it says "don't ship."
- Findings go in `DECISION.md` as they firm up; raw numbers stay in `results/`.
