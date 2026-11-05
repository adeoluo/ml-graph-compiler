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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.fn(args))
    except CompilerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
