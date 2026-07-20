"""Environment capture — pinned into every results file so numbers from different
machines/settings never silently share a table (CLAUDE.md metrics rules)."""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import subprocess


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "?"


def _ver(pkg: str) -> str:
    try:
        return md.version(pkg)
    except Exception:
        return "absent"


def qdrant_digest(compose_path: str = "docker-compose.yml") -> str:
    """Pinned image ref from docker-compose.yml (digest is the reproducibility anchor)."""
    try:
        for line in open(compose_path):
            line = line.strip()
            if line.startswith("image:"):
                return line.split("image:", 1)[1].strip()
    except Exception:
        pass
    return "?"


def capture() -> dict:
    return {
        "chip": _sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "ram_gb": round(int(_sh(["sysctl", "-n", "hw.memsize"]) or 0) / 1024**3, 1),
        "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
        "python": platform.python_version(),
        "power": _sh(["pmset", "-g", "batt"]).split("\n")[0][:60],
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", "unset"),
        "ort_intra_op": os.environ.get("ORT_INTRA_OP_THREADS", "unset"),
        "ort_inter_op": os.environ.get("ORT_INTER_OP_THREADS", "unset"),
        "qdrant_image": qdrant_digest(),
        "versions": {p: _ver(p) for p in
                     ["numpy", "tokenizers", "huggingface-hub", "model2vec",
                      "fastembed", "onnxruntime", "qdrant-client", "ranx", "datasets"]},
    }


if __name__ == "__main__":
    import json
    print(json.dumps(capture(), indent=2))
