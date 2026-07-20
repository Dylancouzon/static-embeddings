"""Speed measurements — throughput, cold-start phase-split, query latency, peak RSS.
Every number is median + IQR over >=3 runs (CLAUDE.md). Cold-start uses fresh
subprocesses (_coldstart_child.py); warm-cache regime only for now (cold-cache pending).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil


def _stats(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"median": float(np.median(a)),
            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
            "n": len(xs)}


def throughput(embed_fn, corpus: list[str], batch_sizes: list[int], repeats: int) -> dict:
    """docs/sec at each batch size. Embeds a fixed slice so systems are comparable."""
    out = {}
    for bs in batch_sizes:
        n = min(len(corpus), max(bs * 20, 512))  # enough docs to amortize per-call overhead
        slice_ = corpus[:n]
        runs = []
        for _ in range(repeats):
            t = time.perf_counter()
            embed_fn(slice_, batch_size=bs)
            dt = time.perf_counter() - t
            runs.append(n / dt)
        out[str(bs)] = _stats(runs)
    return out


def query_latency(embed_fn, queries: list[str], n: int, repeats: int = 1) -> dict:
    """Single-query embed time p50/p95 over n queries."""
    qs = (queries * (n // len(queries) + 1))[:n]
    lat = []
    for q in qs:
        t = time.perf_counter()
        embed_fn([q], batch_size=1)
        lat.append((time.perf_counter() - t) * 1000)  # ms
    a = np.asarray(lat)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "n": n}


def peak_rss_mb(embed_fn, corpus: list[str], batch_size: int = 256) -> dict:
    """RSS (MB) before and after a full-corpus embed run. 'peak' ~= after, since
    embed grows RSS monotonically (matrix pages in, vectors accumulate)."""
    proc = psutil.Process()
    before = proc.memory_info().rss / 1024**2
    embed_fn(corpus, batch_size=batch_size)
    after = proc.memory_info().rss / 1024**2
    return {"rss_before_mb": round(before, 1), "rss_after_mb": round(after, 1),
            "embed_delta_mb": round(after - before, 1)}


def cold_start(kind: str, model_id: str | None, sample_texts: list[str], processes: int) -> dict:
    """Phase-split cold-start over N fresh processes (warm OS cache). Reports median+IQR
    per phase. Interpreter startup = wall(parent) - child_total."""
    with tempfile.TemporaryDirectory() as d:
        tf = Path(d) / "texts.json"
        tf.write_text(json.dumps(sample_texts[:64]))
        phases: dict[str, list[float]] = {}
        interp: list[float] = []
        for _ in range(processes):
            launch = time.perf_counter()
            r = subprocess.run(
                [sys.executable, "_coldstart_child.py", kind, model_id or "-", str(tf)],
                capture_output=True, text=True)
            wall = time.perf_counter() - launch
            if r.returncode != 0:
                raise RuntimeError(f"coldstart child failed ({kind}/{model_id}):\n{r.stderr[-800:]}")
            child = json.loads(r.stdout.strip().splitlines()[-1])
            for k in ("imports", "cache_check", "model_load", "first_batch", "second_batch"):
                phases.setdefault(k, []).append(child[k])
            interp.append(wall - child["child_total"])
    result = {k: _stats(v) for k, v in phases.items()}
    result["interpreter_startup"] = _stats(interp)
    return result
