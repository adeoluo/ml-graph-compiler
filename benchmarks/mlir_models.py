"""MLIR export of the real models: verification with mlir-opt and, optionally, execution.

For each model and level the graph is exported to linalg-on-tensors MLIR (weights elided for the
verification pass), parsed and verified by ``mlir-opt``. With ``--execute`` the full module
(weights inlined as dense resources) is lowered through one-shot bufferization, loops and the
LLVM dialect, compiled with clang, run on one seeded input and compared with onnxruntime.

    uv run --group bench python benchmarks/mlir_models.py [--execute resnet50 mobilenetv2]
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Any

import numpy as np
import onnxruntime as ort
from common import DATA, model_path, peak_rss_mb, write_result
from latency import make_input

from ml_graph_compiler.mlir import compile_mlir, find_mlir_tools, mlir_verify, to_mlir
from ml_graph_compiler.onnx_import import from_onnx
from ml_graph_compiler.passes import optimize


def export_and_verify(model: str, level: str) -> dict[str, Any]:
    graph = from_onnx(str(model_path(model)))
    optimize(graph, level)
    start = time.perf_counter()
    module = to_mlir(graph, "linalg", elide_constants=True)
    export_seconds = time.perf_counter() - start
    start = time.perf_counter()
    mlir_verify(module.text, allow_unregistered=not module.fully_lowered)
    return {
        "nodes": len(graph.nodes),
        "lines": len(module.text.splitlines()),
        "lowered_ops": module.lowered,
        "fallback_ops": module.fallback,
        "fully_lowered": module.fully_lowered,
        "export_seconds": round(export_seconds, 3),
        "verify_seconds": round(time.perf_counter() - start, 3),
    }


def execute(model: str, level: str, repeat: int) -> dict[str, Any]:
    graph = from_onnx(str(model_path(model)))
    optimize(graph, level)
    x = make_input(model)
    feeds = {graph.inputs[0].name: x}
    start = time.perf_counter()
    exe = compile_mlir(graph, cache_dir=DATA / "mlircache")
    build_seconds = time.perf_counter() - start
    out = exe.run(feeds)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        exe.run(feeds)
        times.append(1000 * (time.perf_counter() - t0))
    sess = ort.InferenceSession(str(model_path(model)), providers=["CPUExecutionProvider"])
    ref = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds), strict=True))
    return {
        "build_seconds": round(build_seconds, 1),
        "max_abs_err_vs_ort": max(float(np.abs(out[k] - ref[k]).max()) for k in ref),
        "top1_agree": bool(
            all(
                np.array_equal(
                    out[k].reshape(-1, out[k].shape[-1]).argmax(-1),
                    ref[k].reshape(-1, ref[k].shape[-1]).argmax(-1),
                )
                for k in ref
            )
        ),
        "median_ms_single_thread": float(np.median(times)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet50", "mobilenetv2", "bert-base", "gpt2"])
    ap.add_argument("--levels", nargs="+", default=["O0", "O3"])
    ap.add_argument("--execute", nargs="*", default=[], help="models to lower and run")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--name", default="mlir")
    args = ap.parse_args()
    tools = find_mlir_tools()
    result: dict[str, Any] = {"args": vars(args), "tools": str(tools), "verify": {}, "execute": {}}
    for model in args.models:
        result["verify"][model] = {}
        for level in args.levels:
            try:
                info = export_and_verify(model, level)
            except Exception as exc:
                info = {"error": f"{type(exc).__name__}: {exc}"[:800]}
            result["verify"][model][level] = info
            print(model, level, {k: v for k, v in info.items() if k != "lowered_ops"}, flush=True)
            gc.collect()
    for model in args.execute:
        try:
            info = execute(model, "O3", args.repeat)
        except Exception as exc:
            info = {"error": f"{type(exc).__name__}: {exc}"[:800]}
        result["execute"][model] = info
        print(model, "execute", info, flush=True)
        gc.collect()
    result["peak_rss_mb"] = round(peak_rss_mb(), 1)
    print("wrote", write_result(args.name, result))


if __name__ == "__main__":
    main()
