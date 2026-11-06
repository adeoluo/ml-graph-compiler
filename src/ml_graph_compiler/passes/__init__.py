"""Optimization passes and the pass manager."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ml_graph_compiler.errors import CompilerError
from ml_graph_compiler.ir import Graph
from ml_graph_compiler.passes.basic import (
    eliminate_common_subexpressions,
    eliminate_dead_code,
    fold_constants,
)
from ml_graph_compiler.passes.fusion import fuse
from ml_graph_compiler.passes.layout import to_nhwc
from ml_graph_compiler.passes.loopfuse import fuse_elementwise
from ml_graph_compiler.passes.simplify import simplify

PassFn = Callable[[Graph], int]

PASSES: dict[str, PassFn] = {
    "fold": fold_constants,
    "dce": eliminate_dead_code,
    "cse": eliminate_common_subexpressions,
    "simplify": simplify,
    "fuse": fuse,
    "layout": to_nhwc,
    "loopfuse": fuse_elementwise,
}

# Cumulative pipelines; each level adds one group of passes (used by the pass ablation).
PIPELINES: dict[str, list[str]] = {
    "O0": [],
    "fold": ["fold", "dce"],
    "cse": ["fold", "dce", "cse", "dce"],
    "simplify": ["fold", "dce", "cse", "simplify", "fold", "cse", "dce"],
    "fuse": ["fold", "dce", "cse", "simplify", "fold", "cse", "fuse", "simplify", "dce"],
    "layout": [
        "fold",
        "dce",
        "cse",
        "simplify",
        "fold",
        "cse",
        "fuse",
        "simplify",
        "layout",
        "cse",
        "dce",
    ],
}
PIPELINES["loopfuse"] = [*PIPELINES["layout"], "loopfuse", "dce"]
PIPELINES["O1"] = PIPELINES["simplify"]
PIPELINES["O2"] = PIPELINES["fuse"]
PIPELINES["O3"] = PIPELINES["loopfuse"]


@dataclass
class PassRecord:
    name: str
    changes: int
    nodes_after: int
    seconds: float


@dataclass
class PassReport:
    nodes_before: int
    records: list[PassRecord] = field(default_factory=list)

    @property
    def nodes_after(self) -> int:
        return self.records[-1].nodes_after if self.records else self.nodes_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes_before": self.nodes_before,
            "nodes_after": self.nodes_after,
            "passes": [r.__dict__ for r in self.records],
        }


def resolve(passes: str | Sequence[str]) -> list[str]:
    """A pipeline name (O0..O3, fold, cse, ...) or an explicit list of pass names."""
    if isinstance(passes, str):
        if passes in PIPELINES:
            return list(PIPELINES[passes])
        names = [p.strip() for p in passes.split(",") if p.strip()]
    else:
        names = list(passes)
    unknown = [n for n in names if n not in PASSES]
    if unknown:
        raise CompilerError(
            f"unknown passes {unknown}; known: {sorted(PASSES)} or {sorted(PIPELINES)}"
        )
    return names


def optimize(graph: Graph, passes: str | Sequence[str] = "O3", verify: bool = True) -> PassReport:
    """Run passes in order, in place. Verifies the graph after each pass by default."""
    report = PassReport(nodes_before=len(graph.nodes))
    for name in resolve(passes):
        start = time.perf_counter()
        changes = PASSES[name](graph)
        if verify:
            graph.verify()
        report.records.append(
            PassRecord(name, changes, len(graph.nodes), round(time.perf_counter() - start, 4))
        )
    return report


__all__ = [
    "PASSES",
    "PIPELINES",
    "PassReport",
    "eliminate_common_subexpressions",
    "eliminate_dead_code",
    "fold_constants",
    "fuse",
    "fuse_elementwise",
    "optimize",
    "simplify",
    "to_nhwc",
]
