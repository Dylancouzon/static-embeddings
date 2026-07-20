"""Orchestrator: download -> correctness gate (HARD STOP) -> embed -> index ->
evaluate -> report. Thread settings pinned from bench.toml into the environment so
every subprocess (and the cold-start children) honor them identically.

A failed correctness gate aborts before any benchmarking, even in autonomous mode.
"""
from __future__ import annotations

import os
import subprocess
import sys

import manifest

STAGES = ["01_download_data.py", "02_embed.py", "03_index.py", "04_evaluate.py", "05_report.py"]


def _env() -> dict:
    t = manifest.threads()
    e = dict(os.environ)
    e.update({
        "OMP_NUM_THREADS": str(t["omp_num_threads"]),
        "TOKENIZERS_PARALLELISM": "false",
        "ORT_INTRA_OP_THREADS": str(t["ort_intra_op"]),
        "ORT_INTER_OP_THREADS": str(t["ort_inter_op"]),
        "OPENBLAS_NUM_THREADS": str(t["omp_num_threads"]),
        "MKL_NUM_THREADS": str(t["omp_num_threads"]),
        "VECLIB_MAXIMUM_THREADS": str(t["omp_num_threads"]),  # Accelerate (macOS numpy)
        "NUMEXPR_NUM_THREADS": str(t["omp_num_threads"]),
    })
    return e


def _run(script: str, env: dict):
    print(f"\n{'='*60}\n>>> {script}\n{'='*60}")
    r = subprocess.run([sys.executable, script], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{script} failed with code {r.returncode}")


def main():
    env = _env()
    print(f"threads: OMP={env['OMP_NUM_THREADS']} ORT_intra={env['ORT_INTRA_OP_THREADS']}")
    _run("01_download_data.py", env)

    print(f"\n{'='*60}\n>>> correctness_gate.py (HARD STOP)\n{'='*60}")
    if subprocess.run([sys.executable, "correctness_gate.py"], env=env).returncode != 0:
        raise SystemExit("CORRECTNESS GATE FAILED — aborting before benchmarking. "
                         "See results/correctness.json.")

    for script in STAGES[1:]:
        _run(script, env)
    print("\nrun_all complete — see results/RESULTS.md")


if __name__ == "__main__":
    main()
