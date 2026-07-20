"""Cold-start phase timer — one fresh process per invocation. Prints JSON of phase
durations (seconds). Import-minimal at the top so the 'imports' phase reflects the
backend libs, not this harness. Backend is imported eagerly per kind, then the model
is loaded, so import vs load phases stay separate (CLAUDE.md V4 requirement).

Usage: python _coldstart_child.py <kind> <model_id|-> <texts.json>
"""
import time

t0 = time.perf_counter()
import json
import sys

kind, model_id, texts_path = sys.argv[1], sys.argv[2], sys.argv[3]
texts = json.loads(open(texts_path).read())
model = None if model_id == "-" else model_id

# eager backend import (the heavy part of "imports")
if kind == "static_numpy":
    import huggingface_hub  # noqa: F401
    import numpy  # noqa: F401
    import tokenizers  # noqa: F401
elif kind == "model2vec":
    import model2vec  # noqa: F401
elif kind == "static_onnx":
    import huggingface_hub  # noqa: F401
    import onnxruntime  # noqa: F401
    import tokenizers  # noqa: F401
elif kind == "bm25" or kind == "fastembed_dense":
    import fastembed  # noqa: F401
t_import = time.perf_counter()

import systems  # lazy internally; backend already imported above

# Split cache-check (file resolution/download) from model load (mmap/ORT+tokenizer)
# for A and C, per CLAUDE.md V4 phase list. B/D/BM25 resolve internally -> one phase.
if kind == "static_numpy":
    from static_numpy import StaticNumpy
    paths = StaticNumpy.resolve(model)
    t_cache = time.perf_counter()
    sysm = StaticNumpy(model, paths=paths)
elif kind == "static_onnx":
    paths = systems.StaticOnnx.resolve(model)
    t_cache = time.perf_counter()
    sysm = systems.StaticOnnx(model, paths=paths)
else:
    t_cache = time.perf_counter()
    sysm = systems.build(kind, model)
t_load = time.perf_counter()

batch = texts[:32] if len(texts) >= 32 else texts
emb = sysm.embed_sparse if kind == "bm25" else sysm.embed
list(emb(batch)) if kind == "bm25" else emb(batch)
t_b1 = time.perf_counter()
list(emb(batch)) if kind == "bm25" else emb(batch)
t_b2 = time.perf_counter()

if kind == "static_numpy":
    from static_numpy import assert_no_forbidden_imports
    assert_no_forbidden_imports()

print(json.dumps({
    "imports": t_import - t0,      # interpreter startup measured by parent (wall - child_total)
    "cache_check": t_cache - t_import,   # hf file resolution/download (0-ish for B/D internal)
    "model_load": t_load - t_cache,      # mmap+tokenizer (A) / ORT+tokenizer (C) / build (B/D)
    "first_batch": t_b1 - t_load,
    "second_batch": t_b2 - t_b1,
    "child_total": t_b2 - t0,
}))
