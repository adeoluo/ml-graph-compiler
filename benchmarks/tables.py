"""Print the README and evaluation report tables from the committed results in benchmarks/results/.

uv run python benchmarks/tables.py [parity latency ablation memory mlir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent / "results"
MODELS = ("resnet50", "mobilenetv2", "bert-base", "gpt2")
ENGINES = ("mgc", "ort", "ort-noopt", "torch")
LABEL = {
    "resnet50": "ResNet-50",
    "mobilenetv2": "MobileNetV2",
    "bert-base": "BERT-base",
    "gpt2": "GPT-2",
}


def load(name: str) -> dict[str, Any] | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def parity() -> None:
    d = load("parity")
    if d is None:
        return
    print(
        "| Model | Inputs | Max abs error | Mean abs error | Top-1 agreement "
        "| Top-1 accuracy (mgc / ORT) |"
    )
    print("|---|---|---:|---:|---:|---:|")
    for m in MODELS:
        r = d["models"].get(m)
        if r is None:
            continue
        mx = max(o["max_abs_err"] for o in r["outputs"].values())
        mean = max(o["mean_abs_err"] for o in r["outputs"].values())
        acc = (
            f"{r['top1_accuracy_mgc']:.4f} / {r['top1_accuracy_ort']:.4f}"
            if "top1_accuracy_mgc" in r
            else "n/a"
        )
        agree = round(r["top1_agreement"] * r["top1_positions"])
        print(
            f"| {LABEL[m]} | {r['inputs']:,} {r['input_kind']} | {mx:.1e} | {mean:.1e} | "
            f"{agree:,} / {r['top1_positions']:,} | {acc} |"
        )


def latency() -> None:
    rows = []
    for name in ("latency_cnn", "latency_transformers"):
        d = load(name)
        if d is not None:
            rows += d["summary"]
            loads = [c["load_average_1m"] for c in d["cells"] if "load_average_1m" in c]
            print(
                f"{name}: {len(d['cells'])} cells, 1 minute load average during the cells "
                f"min {min(loads):.1f}, median {sorted(loads)[len(loads) // 2]:.1f}, "
                f"max {max(loads):.1f} (8 cores)"
            )
    if not rows:
        return
    by = {(r["model"], r["engine"], r["threads"]): r for r in rows}
    for threads in (1, 8):
        print(f"\nThreads = {threads}: median ms [95% CI] (peak RSS MB)\n")
        print(
            "| Model | mgc (this) | onnxruntime | onnxruntime, no graph opt "
            "| PyTorch eager | mgc vs ORT |"
        )
        print("|---|---:|---:|---:|---:|---|")
        for m in MODELS:
            cells = []
            for e in ENGINES:
                r = by.get((m, e, threads))
                cells.append(
                    "n/a"
                    if r is None
                    else f"{r['median_ms']:.1f} [{r['median_ci95_ms'][0]:.1f}, "
                    f"{r['median_ci95_ms'][1]:.1f}] ({r['peak_rss_mb']:.0f}"
                    + ("" if r["rounds"] == 3 else f"; {r['rounds']} rounds")
                    + ")"
                )
            a, b = by.get((m, "mgc", threads)), by.get((m, "ort", threads))
            verdict = "n/a"
            if a and b:
                if a["median_ci95_ms"][1] < b["median_ci95_ms"][0]:
                    verdict = f"win, {b['median_ms'] / a['median_ms']:.2f}x"
                elif b["median_ci95_ms"][1] < a["median_ci95_ms"][0]:
                    verdict = f"loss, {a['median_ms'] / b['median_ms']:.2f}x slower"
                else:
                    verdict = "tie (CIs overlap)"
            print(f"| {LABEL[m]} | " + " | ".join(cells) + f" | {verdict} |")


def load_ablation() -> dict[str, Any] | None:
    """Merge the per-model-group ablation files (ablation_cnn, ablation_transformers)."""
    merged: dict[str, Any] = {}
    for name in ("ablation", "ablation_cnn", "ablation_transformers"):
        d = load(name)
        if d is None:
            continue
        for key in ("cumulative", "leave_one_out"):
            merged.setdefault(key, {}).update(d.get(key, {}))
        if "memory" in d:
            merged.setdefault("memory", {}).update(d["memory"])
    return merged or None


def ablation() -> None:
    d = load_ablation()
    if d is None:
        return
    for study in ("cumulative", "leave_one_out"):
        if study not in d:
            continue
        print(f"\n{study}: nodes, arena MB, median ms at 1 / 8 threads, max abs err vs ORT\n")
        print("| Model | Config | Nodes | Arena MB | ms, 1 thread | ms, 8 threads | Max abs err |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for m in MODELS:
            for config, r in d[study].get(m, {}).items():
                if "error" in r:
                    print(f"| {LABEL[m]} | {config} | error: {r['error'][:60]} | | | | |")
                    continue
                print(
                    f"| {LABEL[m]} | {config} | {r['nodes']} | "
                    f"{r['arena_bytes'] / 2**20:.1f} | {r['median_ms_t1']:.1f} | "
                    f"{r['median_ms_t8']:.1f} | {r['max_abs_err_vs_ort']:.1e} |"
                )


def memory() -> None:
    d = load_ablation()
    if d is None or "memory" not in d:
        return
    print("\nArena MB by planner (scheduled graph, O3)\n")
    print(
        "| Model | Peak live, import order | Peak live, scheduled (lower bound) | naive (no reuse) "
        "| first_fit | greedy_by_size | greedy_by_size + in place |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        rows = [r for r in d["memory"].get(m, []) if r["scheduler"]]
        if not rows:
            continue

        def arena(strategy: str, inplace: bool, rows: list[dict[str, Any]] = rows) -> str:
            r = next(x for x in rows if x["strategy"] == strategy and x["inplace"] == inplace)
            return f"{r['arena_bytes'] / 2**20:.1f}"

        print(
            f"| {LABEL[m]} | {rows[0]['peak_live_bytes_import_order'] / 2**20:.1f} | "
            f"{rows[0]['lower_bound_bytes'] / 2**20:.1f} | "
            f"{arena('naive', False)} | "
            f"{arena('first_fit', False)} | {arena('greedy_by_size', False)} | "
            f"{arena('greedy_by_size', True)} |"
        )


def mlir() -> None:
    d = load("mlir")
    if d is None:
        return
    print(
        "\n| Model | Level | Nodes | MLIR lines | Fully lowered to linalg | Verified by mlir-opt |"
    )
    print("|---|---|---:|---:|---|---|")
    for m in MODELS:
        for level, r in d["verify"].get(m, {}).items():
            ok = "no: " + r["error"][:60] if "error" in r else "yes"
            print(
                f"| {LABEL[m]} | {level} | {r.get('nodes', '')} | {r.get('lines', '')} | "
                f"{r.get('fully_lowered', '')} | {ok} |"
            )
    for m, r in d.get("execute", {}).items():
        print(f"\nexecute {m}: {r}")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["parity", "latency", "ablation", "memory", "mlir"]
    for name in wanted:
        globals()[name]()
