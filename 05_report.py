"""Stage 05 — render results/RESULTS.md: generated tables + a pass/fail line per
pre-registered threshold (CLAUDE.md). Never hand-edited. Missing inputs render as
N/A rather than crashing, so partial runs still produce a readable report.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

R = Path("results")
RET = "potion-retrieval-32M"
BEIR = ["scifact", "nfcorpus", "fiqa"]
CODE = "codesearchnet-python"


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def quality(run_id: str, dataset: str):
    d = _load(R / "quality" / f"{dataset}__{run_id}.json")
    return d["metrics"] if d else None


def ndcg(run_id: str, dataset: str):
    m = quality(run_id, dataset)
    return m["ndcg@10"] if m else None


def beir_avg_ndcg(run_id: str):
    # require ALL BEIR tracks — a 2/3 average must not render as a confident verdict
    vals = [ndcg(run_id, d) for d in BEIR]
    return mean(vals) if all(v is not None for v in vals) else None


def speed(run_id: str):
    return _load(R / "speed" / f"{run_id}.json")


def tp32(run_id: str):
    s = speed(run_id)
    try:
        return s["throughput"]["32"]["median"]
    except (TypeError, KeyError):
        return None


def fmt(x, pct=False, suffix=""):
    if x is None:
        return "N/A"
    return f"{x*100:.1f}%" if pct else f"{x:.4f}{suffix}" if isinstance(x, float) else f"{x}{suffix}"


def _pf(ok):
    return "✅ PASS" if ok else ("❌ FAIL" if ok is not None else "⚠️ N/A")


def thresholds() -> list[str]:
    L = ["## Pre-registered thresholds (the verdict)\n"]
    gate = _load(R / "correctness.json")
    gate_ok = gate.get("passed") if gate else None
    L.append(f"- **Correctness gate** (A==B within 1e-5 + edges): {_pf(gate_ok)}")

    # V1: retrieval-32M NDCG@10 >= BM25 on code
    a_code = ndcg(f"A__{RET}", CODE)
    bm_code = ndcg("BM25", CODE)
    v1 = (a_code >= bm_code) if (a_code is not None and bm_code is not None) else None
    L.append(f"- **V1** (beats BM25 on code): A={fmt(a_code)} vs BM25={fmt(bm_code)} → {_pf(v1)}")

    # V2: retrieval-32M >= 75% MiniLM avg over BEIR
    a_beir = beir_avg_ndcg(f"A__{RET}")
    d1_beir = beir_avg_ndcg("D1")
    ratio = (a_beir / d1_beir) if (a_beir is not None and d1_beir) else None
    v2 = (ratio >= 0.75) if ratio is not None else None
    L.append(f"- **V2** (≥75% MiniLM on BEIR): A={fmt(a_beir)} MiniLM={fmt(d1_beir)} "
             f"ratio={fmt(ratio, pct=True)} → {_pf(v2)}")

    # V3: >=20x embed throughput vs fastembed MiniLM @batch32
    a_tp = tp32(f"A__{RET}")
    d1_tp = tp32("D1")
    speedup = (a_tp / d1_tp) if (a_tp is not None and d1_tp) else None
    v3 = (speedup >= 20) if speedup is not None else None
    L.append(f"- **V3** (≥20× throughput @32): A={fmt(a_tp)} MiniLM={fmt(d1_tp)} docs/s "
             f"→ {fmt(speedup, suffix='×') if speedup else 'N/A'} {_pf(v3)}")

    # V4: model-load <=200ms warm + total cold-start < ONNX candidates
    s = speed(f"A__{RET}")
    load_ms = s["cold_start"]["model_load"]["median"] * 1000 if s else None
    a_total = _total_coldstart(s) if s else None
    onnx_totals = [t for t in (_total_coldstart(speed(r)) for r in ("D1", "D2", f"C__{RET}")) if t]
    v4_load = (load_ms <= 200) if load_ms is not None else None
    # "meaningfully under": A total cold-start <= 75% of the fastest ONNX total.
    # Pre-registered here before results exist; warm-cache regime only (cold pending).
    v4_total = (a_total <= 0.75 * min(onnx_totals)) if (a_total is not None and onnx_totals) else None
    v4 = (v4_load and v4_total) if (v4_load is not None and v4_total is not None) else None
    L.append(f"- **V4** (load ≤200ms warm + total ≤75% ONNX, warm cache): "
             f"load={fmt(load_ms, suffix='ms') if load_ms else 'N/A'} "
             f"total={fmt(a_total, suffix='s') if a_total else 'N/A'} "
             f"min(ONNX total)={fmt(min(onnx_totals), suffix='s') if onnx_totals else 'N/A'} → {_pf(v4)}")

    # Workflow inclusion: hybrid >=90% MiniLM on both tracks
    h_code, h_beir = ndcg(f"H__{RET}", CODE), beir_avg_ndcg(f"H__{RET}")
    d1_code = ndcg("D1", CODE)
    wf_code = (h_code / d1_code) if (h_code is not None and d1_code) else None
    wf_beir = (h_beir / d1_beir) if (h_beir is not None and d1_beir) else None
    wf = (wf_code >= 0.90 and wf_beir >= 0.90) if (wf_code is not None and wf_beir is not None) else None
    L.append(f"- **Workflow** (hybrid ≥90% MiniLM both tracks): code={fmt(wf_code, pct=True)} "
             f"BEIR={fmt(wf_beir, pct=True)} → {_pf(wf)}")

    # Approach: A within 10% of B throughput + gate
    b_tp = tp32(f"B__{RET}")
    within = (a_tp >= 0.9 * b_tp) if (a_tp is not None and b_tp) else None
    appr = (within and gate_ok) if (within is not None and gate_ok is not None) else None
    L.append(f"- **Approach** (A within 10% of B tput + gate): A={fmt(a_tp)} B={fmt(b_tp)} docs/s "
             f"→ {_pf(appr)}")
    return L


def _total_coldstart(s) -> float | None:
    if not s or "cold_start" not in s:
        return None
    cs = s["cold_start"]
    return sum(cs[k]["median"] for k in
               ("interpreter_startup", "imports", "cache_check", "model_load", "first_batch"))


def quality_table() -> list[str]:
    import manifest
    runs = [e["run_id"] for e in manifest.embedders()] + [h["run_id"] for h in manifest.hybrid_specs()]
    cols = BEIR + [CODE]
    L = ["\n## Quality — NDCG@10 (exact search)\n", "| System | " + " | ".join(cols) + " |",
         "|" + "---|" * (len(cols) + 1)]
    for rid in runs:
        L.append("| " + rid + " | " + " | ".join(fmt(ndcg(rid, d)) for d in cols) + " |")
    return L


def speed_table() -> list[str]:
    import manifest
    L = ["\n## Speed — throughput (docs/s, median) & cold-start (warm OS cache)\n",
         "_load(ms) = mmap+tokenizer for A/C, resolution+init for B/D; cache-chk split out for A/C only._\n",
         "| System | tput@1 | tput@32 | tput@256 | load(ms) | cache-chk(ms) | total cold(s) | q p50(ms) |",
         "|---|---|---|---|---|---|---|---|"]
    for e in manifest.embedders():
        s = speed(e["run_id"])
        if not s:
            L.append(f"| {e['run_id']} | " + " N/A |" * 7)
            continue
        cs = s["cold_start"]
        L.append("| {} | {} | {} | {} | {:.0f} | {:.0f} | {:.3f} | {:.2f} |".format(
            e["run_id"],
            fmt(s["throughput"].get("1", {}).get("median")),
            fmt(s["throughput"].get("32", {}).get("median")),
            fmt(s["throughput"].get("256", {}).get("median")),
            cs["model_load"]["median"] * 1000, cs["cache_check"]["median"] * 1000,
            _total_coldstart(s) or 0, s["query_latency"]["p50_ms"]))
    return L


def bruteforce_note() -> list[str]:
    bf = _load(R / "quality" / "_bruteforce_check.json")
    if not bf:
        return []
    worst = max((v["ndcg_delta"] for v in bf.values()), default=0)
    return ["\n## Sanity: local brute-force vs Qdrant exact\n",
            f"Max NDCG@10 delta across systems on scifact: **{worst}** "
            f"(≈0 confirms Qdrant exact ranking reproduces numpy top-k)."]


def main():
    counts = _load(R / "data_counts.json") or {}
    lines = ["# Static Embeddings Viability — RESULTS", "",
             "_Generated by 05_report.py — do not hand-edit._", ""]
    env_ref = speed("D1") or speed(f"A__{RET}")
    if env_ref:
        e = env_ref["env"]
        lines += [f"Env: {e['chip']} · {e['ram_gb']}GB · {e['os']} · py{e['python']} · "
                  f"OMP={e['omp_num_threads']} ORT_intra={e['ort_intra_op']} · Qdrant `{e['qdrant_image'][:60]}`", ""]
    if counts:
        lines.append("Datasets: " + ", ".join(f"{k}({v['corpus']}d/{v['queries']}q)" for k, v in counts.items()))
    lines += [""] + thresholds() + quality_table() + speed_table() + bruteforce_note()
    (R / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(thresholds()))
    print(f"\nwrote {R / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
