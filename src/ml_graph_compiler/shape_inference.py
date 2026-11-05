"""Static shape and type inference with partial evaluation of small shape computations."""

from __future__ import annotations

import numpy as np

from ml_graph_compiler import ops
from ml_graph_compiler.errors import ShapeError
from ml_graph_compiler.ir import Graph, Node, TensorType, Value

# Shape arithmetic (Shape -> Gather -> Concat -> Reshape) produces tiny int64 tensors. Values up
# to this many elements are evaluated during inference so that operators whose output shape
# depends on a tensor value (Reshape, Expand, Slice, Range, ...) can be typed.
KNOWN_VALUE_LIMIT = 1 << 16


def _known(value: Value | None, known: dict[int, np.ndarray]) -> np.ndarray | None:
    if value is None:
        return None
    if value.data is not None:
        return value.data
    return known.get(id(value))


def infer_node(node: Node, known: dict[int, np.ndarray] | None = None) -> list[TensorType]:
    """Infer and set the output types of one node; record small known output values."""
    known = {} if known is None else known
    types = [v.type if v is not None else None for v in node.inputs]
    consts = [_known(v, known) for v in node.inputs]
    try:
        out_types = ops.infer(node, types, consts)
    except ShapeError as exc:
        raise ShapeError(f"{node.op} node {node.name!r}: {exc}") from exc
    for value, t in zip(node.outputs, out_types, strict=True):
        value.type = t
    small = all(t.size <= KNOWN_VALUE_LIMIT for t in out_types)
    if small and node.op in ops.SHAPE_ONLY_OPS:
        x = node.input(0).tensor_type
        known[id(node.output)] = (
            ops.shape_value(node, x.shape) if node.op == "Shape" else np.asarray(x.size, np.int64)
        )
    elif small and all(c is not None for c, v in zip(consts, node.inputs, strict=True) if v):
        results = ops.run(node, consts)
        for value, array in zip(node.outputs, results, strict=True):
            known[id(value)] = array
    return out_types


def infer_shapes(graph: Graph) -> None:
    """Assign a static type to every value in the graph (nodes must be in topological order)."""
    known: dict[int, np.ndarray] = {}
    for node in graph.nodes:
        infer_node(node, known)
    for value in graph.outputs:
        if value.type is None:
            raise ShapeError(f"graph output {value.name!r} has no type")
