"""Command line interface: ``python -m ml_graph_compiler`` or ``mgc``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ml_graph_compiler import __version__
from ml_graph_compiler.errors import CompilerError
from ml_graph_compiler.ir import Graph


def _parse_dims(items: Sequence[str]) -> dict[str, int]:
    dims: dict[str, int] = {}
    for item in items:
        name, _, value = item.partition("=")
        if not name or not value.isdigit():
            raise CompilerError(f"--dim expects NAME=INT, got {item!r}")
        dims[name] = int(value)
    return dims


def _parse_shapes(items: Sequence[str]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for item in items:
        name, _, value = item.partition("=")
        parts = value.lower().split("x") if value else []
        if not name or not parts or not all(p.isdigit() for p in parts):
            raise CompilerError(f"--shape expects NAME=D0xD1x..., got {item!r}")
        shapes[name] = tuple(int(p) for p in parts)
    return shapes


def load_model(args: argparse.Namespace) -> Graph:
    """Import the model named on the command line (an ONNX path or a bundled sample name)."""
    from ml_graph_compiler import samples
    from ml_graph_compiler.onnx_import import from_onnx

    source: Any = args.model
    if args.model in samples.SAMPLES:
        build: Any = samples.SAMPLES[args.model]
        source = build()
    elif not Path(args.model).exists():
        raise CompilerError(
            f"{args.model!r} is neither a file nor a bundled sample ({', '.join(samples.SAMPLES)})"
        )
    return from_onnx(source, _parse_shapes(args.shape), _parse_dims(args.dim))


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="ONNX file or bundled sample (mlp, tiny-cnn, tiny-gpt, tiny-bert)")
    p.add_argument(
        "--dim",
        action="append",
        default=[],
        metavar="NAME=N",
        help="value for a symbolic input dimension (repeatable)",
    )
    p.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="INPUT=D0xD1",
        help="full static shape for one input (repeatable)",
    )


def _print(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    for key, value in payload.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  {item}")
        else:
            print(f"{key}: {value}")


def cmd_inspect(args: argparse.Namespace) -> int:
    graph = load_model(args)
    if args.print_ir:
        print(graph)
        return 0
    _print(graph.summary(), args.json)
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    from ml_graph_compiler.passes import optimize

    graph = load_model(args)
    before = graph.op_counts()
    report = optimize(graph, args.opt)
    if args.print_ir:
        print(graph)
        return 0
    payload: dict[str, Any] = {
        "pipeline": args.opt,
        "nodes": f"{report.nodes_before} -> {report.nodes_after}",
        "passes": [
            f"{r.name:<9} changes={r.changes:<4} nodes={r.nodes_after:<5} {r.seconds * 1e3:7.1f} ms"
            for r in report.records
        ],
        "ops before": before,
        "ops after": graph.op_counts(),
    }
    if args.json:
        payload = {
            "pipeline": args.opt,
            **report.to_dict(),
            "ops_before": before,
            "ops_after": graph.op_counts(),
        }
    _print(payload, args.json)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from ml_graph_compiler.interpreter import Interpreter, Profile, random_feeds
    from ml_graph_compiler.passes import optimize

    graph = load_model(args)
    feeds = random_feeds(graph, args.seed, int_high=args.int_high)
    reference = None
    if args.check:
        reference = Interpreter(graph).run(feeds)
    optimize(graph, args.opt)
    profile = Profile()
    start = time.perf_counter()
    outputs = Interpreter(graph).run(feeds, profile)
    elapsed = time.perf_counter() - start
    payload: dict[str, Any] = {
        "backend": "interpreter",
        "pipeline": args.opt,
        "seconds": round(elapsed, 4),
        "peak_live_mb": round(profile.peak_live_bytes / 2**20, 2),
        "outputs": {
            k: f"{v.dtype}{list(v.shape)} mean={float(v.mean()):.6f}" for k, v in outputs.items()
        },
    }
    if reference is not None:
        payload["max_abs_diff_vs_O0"] = max(
            float(np.max(np.abs(outputs[k].astype(np.float64) - reference[k]))) for k in reference
        )
    _print(payload, args.json)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from ml_graph_compiler.memory import STRATEGIES, plan_memory
    from ml_graph_compiler.passes import optimize
    from ml_graph_compiler.schedule import schedule

    graph = load_model(args)
    optimize(graph, args.opt)
    report = schedule(graph)
    plans = {s: plan_memory(graph, s, inplace=not args.no_inplace).summary() for s in STRATEGIES}
    if args.json:
        _print({"schedule": report.__dict__, "plans": plans}, True)
        return 0
    mb = 2**20
    payload: dict[str, Any] = {
        "nodes": report.nodes,
        "peak live (source order)": f"{report.peak_before / mb:.2f} MB",
        "peak live (scheduled)": f"{report.peak_after / mb:.2f} MB",
        "wavefronts": f"{report.levels} levels, widest {report.max_width}",
        "critical path": f"{report.critical_flops / max(report.total_flops, 1):.1%} of FLOPs",
        "arena by strategy": [
            f"{name:<15} {p['arena_bytes'] / mb:8.2f} MB  (lower bound "  # type: ignore[operator]
            f"{p['lower_bound_bytes'] / mb:.2f} MB, no reuse {p['naive_bytes'] / mb:.2f} MB)"  # type: ignore[operator]
            for name, p in plans.items()
        ],
        "aliases": plans["greedy_by_size"]["aliases"],
        "in place": plans["greedy_by_size"]["inplace"],
    }
    _print(payload, False)
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    from ml_graph_compiler.codegen import BuildOptions, compile_graph
    from ml_graph_compiler.interpreter import Interpreter, random_feeds
    from ml_graph_compiler.passes import optimize
    from ml_graph_compiler.schedule import schedule

    graph = load_model(args)
    feeds = random_feeds(graph, args.seed, int_high=args.int_high)
    reference = Interpreter(graph).run(feeds) if args.check else None
    optimize(graph, args.opt)
    schedule(graph)
    options = BuildOptions(openmp=False if args.no_openmp else None, neon=not args.no_neon)
    compiled = compile_graph(graph, options, cache_dir=args.cache_dir)
    if args.threads:
        compiled.set_threads(args.threads)
    report = compiled.report
    assert report is not None
    payload: dict[str, Any] = {
        "backend": "c",
        "pipeline": args.opt,
        "library": str(compiled.library_path),
        "c_lines": report.c_lines,
        "kernels": report.kernels,
        "arena_mb": round(report.arena_bytes / 2**20, 2),
        "weights_mb": round(report.weights_bytes / 2**20, 2),
        "codegen_s": report.codegen_seconds,
        "build_s": report.build_seconds,
        "cached": report.cached,
        "openmp": compiled.openmp,
        "neon": compiled.neon,
        "threads": compiled.max_threads(),
    }
    outputs = compiled.run(feeds)
    if reference is not None:
        payload["max_abs_diff_vs_interpreter"] = max(
            float(np.max(np.abs(outputs[k].astype(np.float64) - reference[k]))) for k in reference
        )
    if args.bench:
        times = sorted(compiled.benchmark(feeds, warmup=2, repeat=args.bench))
        payload["latency_ms_median"] = round(1000 * times[len(times) // 2], 3)
        payload["latency_ms_min"] = round(1000 * times[0], 3)
    if args.output:
        payload["saved"] = str(compiled.save(args.output))
    if args.emit_c:
        Path(args.emit_c).write_text((compiled.library_path.parent / "model.c").read_text())
        payload["c_source"] = args.emit_c
    _print(payload, args.json)
    return 0


def cmd_mlir(args: argparse.Namespace) -> int:
    from ml_graph_compiler.mlir import to_mlir
    from ml_graph_compiler.passes import optimize

    graph = load_model(args)
    optimize(graph, args.opt)
    module = to_mlir(graph, args.dialect, elide_constants=args.elide_constants)
    if args.verify:
        from ml_graph_compiler.mlir import mlir_verify

        mlir_verify(
            module.text, allow_unregistered=not module.fully_lowered or args.dialect == "mgc"
        )
    if args.output:
        Path(args.output).write_text(module.text)
        lowered = sum(module.lowered.values())
        print(
            f"wrote {args.output}: {len(module.text.splitlines())} lines, {lowered} nodes lowered, "
            f"fallback {module.fallback or 'none'}, {module.resources} weight blobs"
            + (", verified by mlir-opt" if args.verify else "")
        )
    else:
        print(module.text, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mgc", description="Deep learning graph compiler: ONNX in, optimized code out."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="import a model and summarize the graph")
    _add_model_args(p)
    p.add_argument("--print-ir", action="store_true", help="print the IR text")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("optimize", help="run a pass pipeline and report the effect of each pass")
    _add_model_args(p)
    p.add_argument("-O", "--opt", default="O3", help="O0..O3, an ablation level, or pass,pass,...")
    p.add_argument("--print-ir", action="store_true", help="print the optimized IR")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_optimize)

    p = sub.add_parser("run", help="execute with the reference interpreter on random inputs")
    _add_model_args(p)
    p.add_argument("-O", "--opt", default="O3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--int-high", type=int, default=100, help="upper bound for integer inputs")
    p.add_argument("--check", action="store_true", help="compare against the unoptimized graph")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("plan", help="schedule the graph and plan its memory arena")
    _add_model_args(p)
    p.add_argument("-O", "--opt", default="O3")
    p.add_argument("--no-inplace", action="store_true", help="disable in-place reuse")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("compile", help="generate C, build it with the system compiler and run it")
    _add_model_args(p)
    p.add_argument("-O", "--opt", default="O3")
    p.add_argument("-o", "--output", help="directory to save the library, weights and metadata")
    p.add_argument("--emit-c", metavar="FILE", help="also write the generated model C source")
    p.add_argument("--cache-dir", help="build cache (default ~/.cache/ml-graph-compiler)")
    p.add_argument("--threads", type=int, default=0, help="OpenMP threads (0 keeps the default)")
    p.add_argument("--no-openmp", action="store_true", help="build without OpenMP")
    p.add_argument("--no-neon", action="store_true", help="use the portable C micro kernel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--int-high", type=int, default=100, help="upper bound for integer inputs")
    p.add_argument("--check", action="store_true", help="compare against the reference interpreter")
    p.add_argument("--bench", type=int, default=0, metavar="N", help="time N runs")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_compile)

    p = sub.add_parser("mlir", help="export the graph as MLIR text (linalg on tensors or mgc ops)")
    _add_model_args(p)
    p.add_argument("-O", "--opt", default="O3")
    p.add_argument("-o", "--output", help="write to a file instead of standard output")
    p.add_argument("--dialect", choices=["linalg", "mgc"], default="linalg")
    p.add_argument("--elide-constants", action="store_true", help="print large weights as elided")
    p.add_argument("--verify", action="store_true", help="parse and verify with mlir-opt")
    p.set_defaults(fn=cmd_mlir)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.fn(args))
    except CompilerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
