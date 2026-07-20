"""Correctness gate — blocks all benchmarking (CLAUDE.md). Re-run after any change to A.

Passes only if, for every static model:
  A vs B  np.allclose(atol=1e-5)  on 1k sampled docs + the fixed edge suite.
Also records A/B/C and B/C cosine-similarity distributions (C is a different graph;
drift must be measured, not assumed). A runs in a clean subprocess (embed_a.py),
which doubles as the import-guard proof.

Exit code 0 = pass, 1 = fail. Writes results/correctness.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import datasets_io
import manifest
import systems

EDGE_SUITE = {
    "empty": "",
    "whitespace": "   \t\n  ",
    "single_token": "hello",
    "unk_heavy": "\x00\x01\x02 ⿕⿖⿗ qwxzjkv",
    "long_doc": "the quick brown fox jumps over the lazy dog " * 1200,  # >10k tokens
    "code": "def f(x):\n    return [i*2 for i in range(x) if i % 3 == 0]",
    "emoji_nonlatin": "🚀 日本語 emoji test Ελληνικά café naïve 中文 🎉",
}
SAMPLE_N = 1000
STATIC_MODELS = ["potion-base-8M", "potion-retrieval-32M"]


def _embed_a(model_id: str, texts: list[str]) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        tin, tout = Path(d) / "in.json", Path(d) / "out.npy"
        tin.write_text(json.dumps(texts))
        subprocess.run([sys.executable, "embed_a.py", model_id, str(tin), str(tout)],
                       check=True)
        return np.load(tout)


def _cosine_stats(x: np.ndarray, y: np.ndarray) -> dict:
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-32)
    yn = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-32)
    cos = (xn * yn).sum(axis=1)
    return {"min": float(cos.min()), "mean": float(cos.mean()),
            "p01": float(np.percentile(cos, 1)), "p50": float(np.percentile(cos, 50))}


def run() -> bool:
    sample = datasets_io.corpus_sample(manifest.all_datasets(), SAMPLE_N)
    if len(sample) < SAMPLE_N:
        print(f"WARN: only {len(sample)} sampled docs (data not fully downloaded)")
    edge_texts = list(EDGE_SUITE.values())
    texts = sample + edge_texts

    report = {"sample_n": len(sample), "edge_cases": list(EDGE_SUITE), "models": {}}
    ok = True
    for mk in STATIC_MODELS:
        mid = manifest.model_id(mk)
        A = _embed_a(mid, texts)
        B = systems.build("model2vec", mid).embed(texts)
        C = systems.build("static_onnx", mid).embed(texts)

        ab = np.allclose(A, B, atol=1e-5)
        max_abs = float(np.abs(A - B).max())
        # per-edge-case pass detail
        edge = {name: bool(np.allclose(A[len(sample) + i], B[len(sample) + i], atol=1e-5))
                for i, name in enumerate(EDGE_SUITE)}
        report["models"][mk] = {
            "A_vs_B_allclose_1e-5": bool(ab),
            "A_vs_B_max_abs_diff": max_abs,
            "edge_pass": edge,
            "cosine_A_B": _cosine_stats(A, B),
            "cosine_A_C": _cosine_stats(A, C),
            "cosine_B_C": _cosine_stats(B, C),
        }
        status = "PASS" if ab and all(edge.values()) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{mk}: A vs B allclose={ab} max_abs={max_abs:.2e} edges={all(edge.values())} -> {status}")
        print(f"  cosine A/C mean={report['models'][mk]['cosine_A_C']['mean']:.4f} "
              f"min={report['models'][mk]['cosine_A_C']['min']:.4f}")

    report["passed"] = ok
    Path("results").mkdir(exist_ok=True)
    Path("results/correctness.json").write_text(json.dumps(report, indent=2))
    print("\nCORRECTNESS GATE:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
