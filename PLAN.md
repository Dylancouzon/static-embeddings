# Build plan — Static Embeddings Viability Benchmark

Reference: `CLAUDE.md` (thresholds, systems, metrics, Qdrant rules are authoritative — this plan implements them, it does not restate them). This file exists for the pre-run Fable review and is deleted once the harness is stable.

## Environment (pinned, recorded in every results file)

Apple M5 Pro · 24 GB · macOS 26.4.1 (build 25E253) · AC power, low-power off. Python **3.12** via uv (system 3.14 lacks onnxruntime/tokenizers wheels). `OMP_NUM_THREADS` and ORT intra/inter-op pinned per run; `TOKENIZERS_PARALLELISM=false`. Qdrant in Docker, pinned by digest.

## Architecture (lean — one embed signature + a manifest)

Every system is a factory returning an object with `embed(texts: list[str]) -> np.ndarray` (dense, float32, L2-normalized where the model says so) or, for BM25, a sparse-vector producer. A registry in `systems.py` maps `system_id -> factory`.

```
pyproject.toml          # uv, pinned deps, ruff, py3.12
bench.toml              # systems × models × datasets × batch sizes (single manifest)
docker-compose.yml      # Qdrant pinned by digest
.gitignore              # data/, results/*.json artifacts, models cache
systems.py              # registry: BM25, D1, D2, B (model2vec), C (onnx) + hybrid helpers
static_numpy.py         # candidate A — standalone so its process imports only numpy/tokenizers/hf_hub
env.py                  # env capture (chip/ram/os/threads/versions/qdrant digest) -> dict
01_download_data.py     # BEIR (scifact/nfcorpus/fiqa) + CodeSearchNet python -> data/ + TREC qrels
02_embed.py             # embed corpus+queries per (system,model,dataset) -> results/vectors + speed JSON
03_index.py             # create collection per model, upload_points, hybrid named vectors
04_evaluate.py          # query_points exact=True -> ranx NDCG@10/Recall@10 -> results JSON
05_report.py            # results/RESULTS.md: generated tables + pass/fail per pre-registered threshold
correctness_gate.py     # A-vs-B allclose(1e-5) on 1k/model + edge suite; A/B/C cosine parity
speed.py                # cold-start phase-split, throughput, ingest, query latency, RSS (median+IQR)
run_all.py              # gate (hard stop) -> download -> embed -> index -> eval -> report
DECISION.md             # deliverable (hand-written from RESULTS.md)
```

Ponytail: no per-system files beyond A. A is separate only because its process must provably exclude torch/safetensors/model2vec.

## Candidate A (the thing we might ship) — exact inference

Per CLAUDE.md rules. Concretely:
1. `hf_hub_download` `model.safetensors`, `tokenizer.json`, `config.json`.
2. Parse safetensors: read 8-byte LE u64 header length → JSON header → find the embedding tensor (`embeddings`/`embedding`/`0.weight` — resolved empirically in probe step). `np.memmap(file, dtype, mode='r', offset=8+header_len+data_offsets[0], shape)`.
3. `Tokenizer.from_file`, `encode_batch(texts, add_special_tokens=False)`.
4. Per doc: token ids → drop `unk_token_id` → gather rows → mean axis 0 → zeros(dim) if empty → L2 normalize `v / (norm+1e-32)` iff `config.normalize`.
5. Import guard: at process end assert `torch`/`safetensors`/`model2vec` not in `sys.modules`.

Correctness gate (blocks all benchmarking, re-run after any A change): A vs B `np.allclose(atol=1e-5)` on 1k sampled docs/model + edge suite (empty, whitespace, single token, unk-heavy, >10k-token doc, code, emoji/non-Latin). Record A/B/C and B/C cosine-similarity distributions — C is a different graph, drift must be measured.

## Systems / models / datasets

Systems: BM25 (fastembed sparse), D1 all-MiniLM-L6-v2, D2 bge-small-en-v1.5 (fastembed ONNX), A, B (model2vec), C (potion ONNX via onnxruntime), H (A+BM25 server-side RRF). Models: potion-base-8M, potion-retrieval-32M (thresholds evaluated on this), potion-multilingual-128M (load/size only). Datasets: SciFact, NFCorpus, FiQA (BEIR via HF `BeIR/*`); CodeSearchNet Python test split (corpus=code, query=docstring 1st paragraph, qrels=CSN pairs → TREC).

## Qdrant

Docker digest-pinned. `qdrant-client` Query API `query_points` only, `exact=True` on every quality query. One collection/model, cosine. Hybrid: named dense+sparse vectors, server-side `prefetch`+RRF. Bulk via `upload_points` batch 128 / 2 streams. Sanity: local brute-force cosine top-k on one dataset/model must reproduce Qdrant exact ranking.

## Thresholds → RESULTS.md (auto pass/fail)

V1 retrieval-32M NDCG@10 ≥ BM25 on code · V2 retrieval-32M ≥ 75% MiniLM avg BEIR · V3 ≥20× embed throughput vs fastembed ONNX MiniLM @batch32 same threads · V4 model-load ≤200ms warm + total cold-start < ONNX (cold-cache regime pending per Dylan; warm reported now) · Workflow: hybrid ≥90% MiniLM on both tracks · Approach: ship A if gate passes and within 10% of B throughput.

## Run sequence (autonomous)

`run_all.py`: correctness gate → if fail, abort + write failure to RESULTS.md → else download → embed (speed JSON) → index → evaluate → report. All speed numbers median+IQR over ≥3 runs. Seeded, deterministic re-runs.

## Risk register (what Fable + the probes must scrutinize)

1. **A↔B parity** — tensor name, dtype (fp16 vs fp32 in memmap), normalize flag, unk handling. Probe the actual potion files before coding A.
2. **C ONNX I/O** — model2vec export is EmbeddingBag-style (input_ids + offsets), not (ids+mask). Probe the real signature; C is worthless if fed wrong inputs.
3. **CSN qrels** — one relevant doc/query, dedup identical functions, TREC format. Wrong qrels = unusable V1.
4. **V3 fairness** — numpy/BLAS threads (A) vs ORT threads (ONNX) must be pinned comparably or "20×" is an artifact.
5. **BM25 in Qdrant** — fastembed BM25 output → sparse vector; IDF is corpus-dependent, must fit on the actual corpus.
6. **model2vec/onnxruntime not pulling torch** — verify install footprint stays dependency-clean for A's claims.
