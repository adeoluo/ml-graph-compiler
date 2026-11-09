"""Compare ONNX exports of the transformer models made with two transformers releases.

The BERT-base and GPT-2 graphs in every benchmark are exported from Hugging Face checkpoints, so a
transformers upgrade can change the graph the compiler sees. This script loads an older export
and the current one, compares the operator histograms, initializer counts and parameter bytes,
and runs both through onnxruntime (all graph optimizations, one thread) on the same 50 seeded
token sequences that ``parity.py`` uses. It also compiles both graphs at ``O3`` and reports the
node and kernel counts the compiler produces, which shows whether the downstream benchmarks see
the same program.

    uv run --group bench python benchmarks/compare_exports.py --old DIR [--models bert-base gpt2]
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from common import MODEL_FILES, exit_now, model_path, write_result

from ml_graph_compiler.codegen.c import generate_c
from ml_graph_compiler.onnx_import import from_onnx
from ml_graph_compiler.passes import optimize
from ml_graph_compiler.schedule import schedule

SEQ = 128
VOCAB = {"bert-base": 30522, "gpt2": 50257}


def graph_summary(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    ops = Counter(node.op_type for node in model.graph.node)
    params = sum(int(np.prod(t.dims)) for t in model.graph.initializer)
    return {
        "bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "parameters": params,
        "opset": next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
        "producer": f"{model.producer_name} {model.producer_version}",
        "ops": dict(sorted(ops.items())),
    }


ABLATION = ("O0", "fold", "cse", "simplify", "fuse", "layout", "loopfuse")


def compiled_summary(path: Path) -> dict[str, Any]:
    """Node counts after every ablation pipeline, and the O3 C source and weights digests."""
    out: dict[str, Any] = {}
    for config in ABLATION:
        graph = from_onnx(str(path))
        report = optimize(graph, config)
        out.setdefault("nodes_imported", report.nodes_before)
        out[f"nodes_{config}"] = report.nodes_after
    graph = from_onnx(str(path))
    optimize(graph, "O3")
    schedule(graph)
    generated = generate_c(graph)
    out["nodes_after_O3"] = len(graph.nodes)
    out["O3_kernels"] = generated.kernels
    out["O3_arena_bytes"] = generated.arena_bytes
    out["O3_c_lines"] = generated.lines
    out["O3_source_sha256"] = hashlib.sha256(generated.source.encode()).hexdigest()
    out["O3_weights_sha256"] = generated.weights.digest()
    return out


def session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def compare(name: str, old: Path, new: Path, n: int) -> dict[str, Any]:
    s_old, s_new = session(old), session(new)
    feed = s_old.get_inputs()[0].name
    max_abs = 0.0
    mean_abs = []
    agree = total = 0
    for seed in range(n):
        ids = np.random.default_rng(seed).integers(0, VOCAB[name], (1, SEQ), dtype=np.int64)
        a = s_old.run(None, {feed: ids})[0]
        b = s_new.run(None, {s_new.get_inputs()[0].name: ids})[0]
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        max_abs = max(max_abs, float(diff.max()))
        mean_abs.append(float(diff.mean()))
        agree += int((a.argmax(-1) == b.argmax(-1)).sum())
        total += int(a.argmax(-1).size)
    return {
        "inputs": n,
        "output": s_old.get_outputs()[0].name,
        "max_abs_diff": max_abs,
        "mean_abs_diff": float(np.mean(mean_abs)),
        "argmax_agreement": f"{agree} / {total}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True, help="directory with the older exports")
    ap.add_argument("--old-label", default="transformers 4.57.6")
    ap.add_argument("--new-label", default="transformers 5.22.2")
    ap.add_argument("--models", nargs="+", default=["bert-base", "gpt2"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--name", default="export_transformers5")
    args = ap.parse_args()
    out: dict[str, Any] = {"old": args.old_label, "new": args.new_label, "models": {}}
    for name in args.models:
        old, new = args.old / MODEL_FILES[name], model_path(name)
        s_old, s_new = graph_summary(old), graph_summary(new)
        s_old.update(compiled_summary(old))
        s_new.update(compiled_summary(new))
        changed = sorted(
            op
            for op in set(s_old["ops"]) | set(s_new["ops"])
            if s_old["ops"].get(op) != s_new["ops"].get(op)
        )
        out["models"][name] = {
            "old": s_old,
            "new": s_new,
            "ops_changed": {
                op: [s_old["ops"].get(op, 0), s_new["ops"].get(op, 0)] for op in changed
            },
            "onnxruntime_outputs": compare(name, old, new, args.n),
        }
        m = out["models"][name]
        m["O3_source_identical"] = s_old["O3_source_sha256"] == s_new["O3_source_sha256"]
        m["O3_weights_identical"] = s_old["O3_weights_sha256"] == s_new["O3_weights_sha256"]
        print(
            f"{name}: nodes {s_old['nodes']} -> {s_new['nodes']}, O3 nodes "
            f"{s_old['nodes_after_O3']} -> {s_new['nodes_after_O3']}, params "
            f"{s_old['parameters']} -> {s_new['parameters']}, ops changed {m['ops_changed']}, "
            f"max abs diff {m['onnxruntime_outputs']['max_abs_diff']:.2e}, O3 C source identical "
            f"{m['O3_source_identical']}, O3 weights identical {m['O3_weights_identical']}"
        )
    print(write_result(args.name, out))
    exit_now()


if __name__ == "__main__":
    main()
