"""Effect of each optimization pass, the scheduler and the memory planner on the real models.

Three studies, all on the four exported models with the C backend:

1. cumulative: pipelines O0 -> fold -> cse -> simplify -> fuse -> layout -> loopfuse (= O3), each
   adding one group of passes. Records nodes, C kernels, arena bytes, max abs error vs
   onnxruntime and median latency at 1 and 8 threads.
2. leave-one-out: O3 with one pass removed (fold, cse, simplify, fuse, layout, loopfuse).
3. memory: arena bytes for every planner strategy (naive, first_fit, greedy_by_size), with and
   without in-place reuse, with and without the memory-aware scheduler; no compilation needed.

Latencies are interleaved across configurations for ``--rounds`` rounds and pooled.

    uv run --group bench python benchmarks/ablation.py [--models ...] [--study cumulative ...]
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Any

import numpy as np
import onnxruntime as ort
from common import model_path, peak_rss_mb, write_result
from latency import bootstrap_median_ci, make_input

from ml_graph_compiler.codegen import compile_graph
from ml_graph_compiler.memory import STRATEGIES, plan_memory
from ml_graph_compiler.onnx_import import from_onnx
from ml_graph_compiler.passes import PIPELINES, optimize
from ml_graph_compiler.schedule import peak_live_bytes, schedule

CUMULATIVE = ["O0", "fold", "cse", "simplify", "fuse", "layout", "loopfuse"]
REMOVABLE = ["fold", "cse", "simplify", "fuse", "layout", "loopfuse"]


def pipeline(config: str) -> list[str]:
    if config.startswith("O3-no-"):
        dropped = config.removeprefix("O3-no-")
        return [p for p in PIPELINES["O3"] if p != dropped]
    return PIPELINES[config]


def reference(model: str, x: np.ndarray) -> dict[str, np.ndarray]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(model_path(model)), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, {sess.get_inputs()[0].name: x}), strict=True))


def time_runs(compiled: Any, feeds: dict[str, np.ndarray], threads: int, n: int) -> list[float]:
    compiled.set_threads(threads)
    return [1000 * t for t in compiled.benchmark(feeds, warmup=3, repeat=n)]


def compile_config(model: str, config: str) -> tuple[Any, dict[str, Any]]:
    graph = from_onnx(str(model_path(model)))
    nodes_before = len(graph.nodes)
    start = time.perf_counter()
    report = optimize(graph, pipeline(config))
    opt_seconds = time.perf_counter() - start
    schedule(graph)
    compiled = compile_graph(graph, name=f"{model}-{config}")
    r = compiled.report
    assert r is not None
    info = {
        "nodes_imported": nodes_before,
        "nodes": len(graph.nodes),
        "optimize_seconds": round(opt_seconds, 3),
        "pass_changes": [[p.name, p.changes] for p in report.records],
        "kernels": r.kernels,
        "arena_bytes": r.arena_bytes,
        "weights_bytes": r.weights_bytes,
        "c_lines": r.c_lines,
        "build_seconds": r.build_seconds,
    }
    return compiled, info


def latency_study(models: list[str], configs: list[str], rounds: int, repeat: int) -> dict:
    out: dict[str, dict[str, Any]] = {}
    for model in models:
        x = make_input(model)
        ref = reference(model, x)
        out[model] = {}
        # One compiled model alive at a time: the transformer weight blobs are ~0.5 GB each, so
        # every round recompiles (the clang build is cached) instead of keeping all configs.
        for r in range(rounds):
            for config in configs:
                if r > 0 and "error" in out[model][config]:
                    continue
                try:
                    compiled, info = compile_config(model, config)
                except Exception as exc:  # recorded, not hidden: some passes are prerequisites
                    out[model][config] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                    print(f"{model} {config}: {out[model][config]['error']}", flush=True)
                    continue
                feeds = {compiled.inputs[0][0]: x}
                if r == 0:
                    got = compiled.run(feeds)
                    info["max_abs_err_vs_ort"] = max(
                        float(np.abs(got[k] - ref[k]).max()) for k in ref if k in got
                    )
                    info["samples_ms"] = {"1": [], "8": []}
                    out[model][config] = info
                for threads in (1, 8):
                    out[model][config]["samples_ms"][str(threads)] += time_runs(
                        compiled, feeds, threads, repeat
                    )
                del compiled, feeds
                gc.collect()
        for config in configs:
            info = out[model][config]
            if "error" in info:
                continue
            for threads in ("1", "8"):
                s = np.array(info["samples_ms"][threads])
                lo, hi = bootstrap_median_ci(s)
                info[f"median_ms_t{threads}"] = float(np.median(s))
                info[f"median_ci95_ms_t{threads}"] = [lo, hi]
            print(
                f"{model} {config}: nodes {info['nodes']} "
                f"arena {info['arena_bytes'] / 2**20:.2f} MB "
                f"t1 {info['median_ms_t1']:.2f} ms t8 {info['median_ms_t8']:.2f} ms "
                f"err {info['max_abs_err_vs_ort']:.2e}",
                flush=True,
            )
        gc.collect()
    return out


def memory_study(models: list[str]) -> dict:
    out: dict[str, Any] = {}
    for model in models:
        rows = []
        for scheduled in (False, True):
            graph = from_onnx(str(model_path(model)))
            optimize(graph, "O3")
            live_before = peak_live_bytes(graph)
            if scheduled:
                schedule(graph)
            live = peak_live_bytes(graph)
            for strategy in STRATEGIES:
                for inplace in (False, True):
                    plan = plan_memory(graph, strategy=strategy, inplace=inplace)
                    plan.verify()
                    rows.append(
                        {
                            "scheduler": scheduled,
                            "strategy": strategy,
                            "inplace": inplace,
                            "arena_bytes": plan.arena_bytes,
                            "lower_bound_bytes": plan.lower_bound,
                            "peak_live_bytes": live,
                            "peak_live_bytes_import_order": live_before,
                        }
                    )
            del graph
            gc.collect()
        out[model] = rows
        best = min(r["arena_bytes"] for r in rows)
        worst = max(r["arena_bytes"] for r in rows)
        print(f"{model}: arena {worst / 2**20:.2f} MB (naive) -> {best / 2**20:.2f} MB", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet50", "mobilenetv2", "bert-base", "gpt2"])
    ap.add_argument(
        "--study",
        nargs="+",
        default=["memory", "cumulative", "leave-one-out"],
        choices=["memory", "cumulative", "leave-one-out"],
    )
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--name", default="ablation")
    args = ap.parse_args()
    result: dict[str, Any] = {"args": vars(args)}
    if "memory" in args.study:
        result["memory"] = memory_study(args.models)
    if "cumulative" in args.study:
        result["cumulative"] = latency_study(args.models, CUMULATIVE, args.rounds, args.repeat)
    if "leave-one-out" in args.study:
        configs = ["O3", *[f"O3-no-{p}" for p in REMOVABLE]]
        result["leave_one_out"] = latency_study(args.models, configs, args.rounds, args.repeat)
    result["peak_rss_mb"] = round(peak_rss_mb(), 1)
    print("wrote", write_result(args.name, result))


if __name__ == "__main__":
    main()
