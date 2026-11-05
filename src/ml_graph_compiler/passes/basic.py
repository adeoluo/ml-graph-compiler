"""Constant folding, dead code elimination and common subexpression elimination."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ml_graph_compiler import ops
from ml_graph_compiler.ir import Graph, Node, Value

# Folding never grows the constant data by more than this unless the result is small.
FOLD_GROWTH_LIMIT_BYTES = 16 << 20


def _keep_transpose_for_gemm(node: Node) -> bool:
    """A transposed 2-D weight that only feeds MatMul as the right operand stays a Transpose;
    fusion turns the pair into one FusedGemm with ``trans_b`` and no transposed copy."""
    if node.op != "Transpose" or node.output.type is None or node.output.type.rank != 2:
        return False
    users = node.output.users
    return bool(users) and all(u.op == "MatMul" and u.inputs[1] is node.output for u in users)


def fold_constants(graph: Graph) -> int:
    """Evaluate every node whose inputs are all constants (and Shape of a typed value).

    Returns the number of nodes removed.
    """
    folded: list[Node] = []
    for node in graph.nodes:
        present = [v for v in node.inputs if v is not None]
        if node.op in ops.SHAPE_ONLY_OPS:
            x = node.input(0).tensor_type
            data = ops.shape_value(node, x.shape) if node.op == "Shape" else np.asarray(x.size)
            results = [data.astype(np.int64)]
        elif present and all(v.is_constant for v in present):
            if _keep_transpose_for_gemm(node):
                continue
            in_bytes = sum(v.data.nbytes for v in present if v.data is not None)
            out_bytes = sum(v.tensor_type.nbytes for v in node.outputs if v.type is not None)
            if out_bytes > max(2 * in_bytes, FOLD_GROWTH_LIMIT_BYTES):
                continue
            results = ops.run(node, [v.data if v is not None else None for v in node.inputs])
        else:
            continue
        for value, data in zip(node.outputs, results, strict=True):
            new = graph.constant(data, value.name + "_folded")
            graph.replace_all_uses(value, new)
        folded.append(node)
    graph.remove_nodes(folded)
    return len(folded)


def eliminate_dead_code(graph: Graph) -> int:
    """Remove nodes that do not contribute to any graph output."""
    live: set[int] = {id(v) for v in graph.outputs}
    keep: list[Node] = []
    for node in reversed(graph.nodes):
        if any(id(o) in live for o in node.outputs):
            keep.append(node)
            live.update(id(v) for v in node.inputs if v is not None)
    keep_ids = {id(n) for n in keep}
    dead = [n for n in graph.nodes if id(n) not in keep_ids]
    return graph.remove_nodes(dead)


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return ("nd", value.dtype.str, value.shape, hashlib.sha1(value.tobytes()).hexdigest())
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    return value


_COMMUTATIVE = frozenset({"Add", "Mul", "Max", "Min", "Equal", "And", "Or", "Xor"})


def eliminate_common_subexpressions(graph: Graph) -> int:
    """Merge identical constants, then nodes with the same operator, attributes and inputs."""
    replaced = 0
    by_content: dict[Any, Value] = {}
    for value in list(graph.constants()):
        assert value.data is not None
        key = _freeze(value.data)
        first = by_content.setdefault(key, value)
        if first is not value:
            graph.replace_all_uses(value, first)
            replaced += 1
    seen: dict[Any, Node] = {}
    duplicates: list[Node] = []
    for node in graph.nodes:
        keys = [id(v) if v is not None else None for v in node.inputs]
        if node.op in _COMMUTATIVE and len(keys) == 2:
            keys = sorted(keys, key=lambda k: -1 if k is None else k)
        key = (node.op, tuple(keys), _freeze(node.attrs), len(node.outputs))
        original = seen.get(key)
        if original is None:
            seen[key] = node
            continue
        for old, new in zip(node.outputs, original.outputs, strict=True):
            graph.replace_all_uses(old, new)
        duplicates.append(node)
    graph.remove_nodes(duplicates)
    return replaced + len(duplicates)
