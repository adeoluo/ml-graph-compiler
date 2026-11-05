"""Helpers shared by the optimization passes."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml_graph_compiler.ir import Graph, Node, Value
from ml_graph_compiler.shape_inference import infer_node


def const(value: Value | None) -> np.ndarray | None:
    return None if value is None else value.data


def scalar(value: Value | None) -> float | None:
    """The value of a constant with exactly one element, else None."""
    data = const(value)
    if data is None or data.size != 1:
        return None
    return float(data.reshape(()))


def is_scalar(value: Value | None, target: float, rel: float = 1e-5) -> bool:
    s = scalar(value)
    return s is not None and abs(s - target) <= rel * max(1.0, abs(target))


def producer(value: Value | None, op: str | tuple[str, ...]) -> Node | None:
    """The producing node of ``value`` if its operator is ``op`` (or one of them)."""
    if value is None or value.producer is None:
        return None
    ops = (op,) if isinstance(op, str) else op
    return value.producer if value.producer.op in ops else None


def only_user(value: Value, graph: Graph) -> Node | None:
    """The single consuming node of ``value`` (and value is not a graph output)."""
    if len(value.users) != 1 or value in graph.outputs:
        return None
    return value.users[0]


def binary_with_const(node: Node | None, op: str, target: float) -> Value | None:
    """If ``node`` is ``op(x, c)`` (or ``op(c, x)`` for Add and Mul) with scalar c == target,
    return x."""
    if node is None or node.op != op or len(node.inputs) != 2:
        return None
    a, b = node.inputs[0], node.inputs[1]
    if is_scalar(b, target) and a is not None and not a.is_constant:
        return a
    if op in ("Add", "Mul") and is_scalar(a, target) and b is not None and not b.is_constant:
        return b
    return None


def insert_node(
    graph: Graph,
    op: str,
    inputs: list[Value | None],
    before: Node,
    attrs: dict[str, Any] | None = None,
    n_outputs: int = 1,
    name: str = "",
) -> Node:
    """Insert a node right before ``before`` and infer its output types."""
    node = graph.add_node(
        op, inputs, n_outputs=n_outputs, attrs=attrs, name=name, index=graph.nodes.index(before)
    )
    infer_node(node)
    return node


def remove_dead(graph: Graph, nodes: list[Node]) -> int:
    """Remove the given nodes (last first) when none of their outputs is used any more."""
    removed = 0
    for node in reversed(nodes):
        if node in graph.nodes and all(
            not o.users and o not in graph.outputs for o in node.outputs
        ):
            graph.remove_node(node)
            removed += 1
    return removed
