"""Latency and peak memory: compiled C model vs onnxruntime vs PyTorch eager, batch 1, CPU.

Every (engine, model, threads) cell runs in a fresh subprocess so that its peak resident set size
is its own. Within the worker the model is loaded, warmed up, then timed for ``--repeat`` calls;
the worker reports the median, p10/p90, a bootstrap 95% CI of the median, the post-import
baseline RSS and the peak RSS. The driver runs the cells round robin for ``--rounds`` rounds so
that background load on the shared machine hits every engine alike, and keeps every sample.

Engines:
  mgc       this compiler, -O3, C backend (clang -O3, NEON, OpenMP)
  ort       onnxruntime CPU provider, ORT_ENABLE_ALL graph optimizations, same ONNX file
  ort-noopt onnxruntime with ORT_DISABLE_ALL (its kernels without its graph optimizer)
  torch     PyTorch eager, torch.inference_mode (torchvision / transformers checkpoints)

    uv run --group bench python benchmarks/latency.py [--threads 1 8] [--rounds 3]
    uv run --group bench python benchmarks/latency.py --worker mgc resnet50 8   (one cell)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np
from common import DATA, exit_now, model_path, peak_rss_mb, write_result

ENGINES = ("mgc", "ort", "ort-noopt", "torch")
MODELS = ("resnet50", "mobilenetv2", "bert-base", "gpt2")
SEQ = 128


def current_rss_mb() -> float:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=False
    ).stdout.strip()
    return int(out or 0) / 1024


def make_input(model: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    if model in ("resnet50", "mobilenetv2"):
        return rng.standard_normal((1, 3, 224, 224), dtype=np.float32)
    vocab = 30522 if model == "bert-base" else 50257
    return rng.integers(0, vocab, (1, SEQ), dtype=np.int64)


def load_runner(engine: str, model: str, threads: int) -> Any:
    x = make_input(model)
    path = model_path(model)
    if engine == "mgc":
        from ml_graph_compiler.codegen import compile_model

        compiled = compile_model(str(path), opt="O3")
        compiled.set_threads(threads)
        feeds = {compiled.inputs[0][0]: x}
        return lambda: compiled.run(feeds)
    if engine.startswith("ort"):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            if engine == "ort-noopt"
            else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        feeds = {sess.get_inputs()[0].name: x}
        return lambda: sess.run(None, feeds)
    os.environ.setdefault("HF_HOME", str(DATA / "hf"))
    os.environ.setdefault("TORCH_HOME", str(DATA / "torch"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch

    torch.set_num_threads(threads)
    if model in ("resnet50", "mobilenetv2"):
        import torchvision

        ctor = (
            torchvision.models.resnet50 if model == "resnet50" else torchvision.models.mobilenet_v2
        )
        net = ctor(weights="IMAGENET1K_V1").eval()
        t = torch.from_numpy(x)
    else:
        from transformers import AutoModel, GPT2LMHeadModel

        if model == "gpt2":
            net = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        else:
            net = AutoModel.from_pretrained("bert-base-uncased").eval()
        t = torch.from_numpy(x)

    def run() -> Any:
        with torch.inference_mode():
            return net(t) if model in ("resnet50", "mobilenetv2") else net(input_ids=t)

    return run


def bootstrap_median_ci(samples: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(samples, (n, samples.size), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def worker(engine: str, model: str, threads: int, warmup: int, repeat: int) -> dict[str, Any]:
    base = current_rss_mb()
    start = time.perf_counter()
    run = load_runner(engine, model, threads)
    load_seconds = time.perf_counter() - start
    for _ in range(warmup):
        run()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    ms = np.array(times) * 1000
    lo, hi = bootstrap_median_ci(ms)
    return {
        "engine": engine,
        "model": model,
        "threads": threads,
        "load_seconds": round(load_seconds, 3),
        "median_ms": float(np.median(ms)),
        "p10_ms": float(np.percentile(ms, 10)),
        "p90_ms": float(np.percentile(ms, 90)),
        "median_ci95_ms": [lo, hi],
        "samples_ms": [round(v, 3) for v in ms.tolist()],
        "rss_before_load_mb": round(base, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "load_average_1m": round(os.getloadavg()[0], 2),
    }


def run_cell(engine: str, model: str, threads: int, warmup: int, repeat: int) -> dict[str, Any]:
    cmd = [sys.executable, __file__, "--worker", engine, model, str(threads)]
    cmd += ["--warmup", str(warmup), "--repeat", str(repeat)]
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if out.returncode != 0:
        return {"engine": engine, "model": model, "threads": threads, "error": out.stderr[-2000:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


def summarize(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = sorted({(c["model"], c["engine"], c["threads"]) for c in cells if "error" not in c})
    for model, engine, threads in keys:
        group = [
            c
            for c in cells
            if "error" not in c
            and (c["model"], c["engine"], c["threads"]) == (model, engine, threads)
        ]
        samples = np.concatenate([np.array(c["samples_ms"]) for c in group])
        lo, hi = bootstrap_median_ci(samples)
        rows.append(
            {
                "model": model,
                "engine": engine,
                "threads": threads,
                "rounds": len(group),
                "median_ms": float(np.median(samples)),
                "median_ci95_ms": [lo, hi],
                "p90_ms": float(np.percentile(samples, 90)),
                "best_round_median_ms": min(c["median_ms"] for c in group),
                "peak_rss_mb": float(np.median([c["peak_rss_mb"] for c in group])),
                "rss_delta_mb": float(
                    np.median([c["peak_rss_mb"] - c["rss_before_load_mb"] for c in group])
                ),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=3, metavar=("ENGINE", "MODEL", "THREADS"))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--engines", nargs="+", default=list(ENGINES))
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 8])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--name", default="latency")
    args = ap.parse_args()
    if args.worker:
        engine, model, threads = args.worker
        print(json.dumps(worker(engine, model, int(threads), args.warmup, args.repeat)))
        return
    cells = []
    for r in range(args.rounds):
        for threads in args.threads:
            for model in args.models:
                for engine in args.engines:
                    cell = run_cell(engine, model, threads, args.warmup, args.repeat)
                    cell["round"] = r
                    cells.append(cell)
                    msg = cell.get("error", "")[-300:] or (
                        f"{cell['median_ms']:.2f} ms, peak {cell['peak_rss_mb']:.0f} MB"
                    )
                    print(f"round {r} {model} {engine} t={threads}: {msg}", flush=True)
    rows = summarize(cells)
    path = write_result(args.name, {"args": vars(args), "summary": rows, "cells": cells})
    print("wrote", path)
    for row in rows:
        print(
            f"{row['model']:12s} {row['engine']:10s} t={row['threads']}: "
            f"{row['median_ms']:8.2f} ms  peak {row['peak_rss_mb']:7.1f} MB"
        )


if __name__ == "__main__":
    main()
    exit_now()
