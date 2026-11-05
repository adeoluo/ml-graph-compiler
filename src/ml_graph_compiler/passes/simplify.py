"""Algebraic simplification and canonicalization.

Rewrites (each preserves the computed values up to float32 rounding):

* Identity, Dropout, Cast to the same type, Expand to the same shape, full Slice, single-input
  Concat: removed.
* ``x + 0``, ``x - 0``, ``x * 1``, ``x / 1``, ``x ^ 1``: removed when broadcasting does not change
  the shape of ``x``.
* ``x ^ 2`` to ``x * x``, ``x ^ 3`` to ``x * x * x``, ``x ^ 0.5`` to ``sqrt(x)``.
* ``x / c`` to ``x * (1 / c)`` for floating point constants.
* Flatten, Squeeze, Unsqueeze and Reshape become one ``Reshape`` with a static ``shape`` attribute;
  chains of reshapes collapse, and reshapes to the same shape disappear.
* ``Transpose(Transpose(x, p), q)`` composes to one Transpose (or to ``x``); a Transpose that only
  moves size-1 dimensions becomes a Reshape.
"""

from __future__ import annotations

import numpy as np

from ml_graph_compiler import ops
from ml_graph_compiler.ir import Graph, Node, Value
from ml_graph_compiler.passes.basic import eliminate_dead_code
from ml_graph_compiler.passes.util import insert_node, is_scalar

_RESHAPE_LIKE = frozenset({"Reshape", "Flatten", "Squeeze", "Unsqueeze"})


def _bypass(graph: Graph, node: Node, value: Value) -> bool:
    """Replace the node's single output by ``value`` when types agree."""
    out = node.outputs[0]
    if value.type != out.type:
        return False
    graph.replace_all_uses(out, value)
    return True


def _identity_rewrite(graph: Graph, node: Node) -> bool:
    op = node.op
    if op == "Identity" or (op == "Dropout" and not any(o.users for o in node.outputs[1:])):
        return _bypass(graph, node, node.input(0))
    x = node.inputs[0] if node.inputs else None
    if x is None:
        return False
    if op == "Cast" and x.type is not None and x.type == node.output.type:
        return _bypass(graph, node, x)
    if op in ("Expand", "Slice", "Tile", "Pad") and x.type == node.output.type:
        consts = [v.data if v is not None else None for v in node.inputs]
        rank = x.tensor_type.rank
        if op == "Pad" and ops.pad_amounts(node, rank, consts) != [0] * (2 * rank):
            return False
        return _bypass(graph, node, x)
    if op == "Concat" and len([v for v in node.inputs if v is not None]) == 1:
        return _bypass(graph, node, x)
    if op in ("Add", "Sub", "Mul", "Div", "Pow") and len(node.inputs) == 2:
        a, b = node.inputs[0], node.inputs[1]
        neutral = {"Add": 0.0, "Sub": 0.0, "Mul": 1.0, "Div": 1.0, "Pow": 1.0}[op]
        if (
            a is not None
            and b is not None
            and b.data is not None
            and np.all(b.data == neutral)
            and _bypass(graph, node, a)
        ):
            return True
        if (
            op in ("Add", "Mul")
            and a is not None
            and b is not None
            and a.data is not None
            and np.all(a.data == neutral)
            and _bypass(graph, node, b)
        ):
            return True
    return False


def _strength_reduce(graph: Graph, node: Node) -> bool:
    if len(node.inputs) != 2:
        return False
    x, c = node.inputs[0], node.inputs[1]
    if x is None or c is None or x.is_constant or x.type is None:
        return False
    if not np.issubdtype(np.dtype(x.type.dtype), np.floating):
        return False
    out = node.output
    if out.type != x.type:
        return False
    if node.op == "Pow":
        if is_scalar(c, 2.0):
            new = insert_node(graph, "Mul", [x, x], node).output
        elif is_scalar(c, 3.0):
            sq = insert_node(graph, "Mul", [x, x], node).output
            new = insert_node(graph, "Mul", [sq, x], node).output
        elif is_scalar(c, 0.5):
            new = insert_node(graph, "Sqrt", [x], node).output
        else:
            return False
        graph.replace_all_uses(out, new)
        return True
    if node.op == "Div" and c.data is not None and np.issubdtype(c.data.dtype, np.floating):
        if np.any(c.data == 0):
            return False
        recip = graph.constant((1.0 / c.data).astype(c.data.dtype), c.name + "_recip")
        new = insert_node(graph, "Mul", [x, recip], node).output
        graph.replace_all_uses(out, new)
        return True
    return False


def _reshape_rewrite(graph: Graph, node: Node) -> bool:
    if node.op not in _RESHAPE_LIKE:
        return False
    x = node.input(0)
    out = node.output
    if out.type is None or x.type is None:
        return False
    if x.type.shape == out.type.shape:
        return _bypass(graph, node, x)
    source = x
    while (
        source.producer is not None
        and source.producer.op == "Reshape"
        and "shape" in source.producer.attrs
    ):
        source = source.producer.input(0)
    canonical = (
        node.op == "Reshape"
        and "shape" in node.attrs
        and len([v for v in node.inputs if v is not None]) == 1
    )
    if canonical and source is x:
        return False
    if source.type is not None and source.type.shape == out.type.shape:
        return _bypass(graph, node, source)
    new = insert_node(graph, "Reshape", [source], node, attrs={"shape": list(out.type.shape)})
    graph.replace_all_uses(out, new.output)
    return True


def _transpose_rewrite(graph: Graph, node: Node) -> bool:
    if node.op != "Transpose":
        return False
    x = node.input(0)
    if x.type is None or node.output.type is None:
        return False
    perm = ops.transpose_perm(node, x.type.rank)
    if perm == list(range(len(perm))):
        return _bypass(graph, node, x)
    inner = x.producer
    if inner is not None and inner.op == "Transpose":
        p1 = ops.transpose_perm(inner, inner.input(0).tensor_type.rank)
        composed = [p1[p] for p in perm]
        src = inner.input(0)
        if composed == list(range(len(composed))):
            return _bypass(graph, node, src)
        new = insert_node(graph, "Transpose", [src], node, attrs={"perm": composed})
        graph.replace_all_uses(node.output, new.output)
        return True
    non_unit = [p for p in perm if x.type.shape[p] != 1]
    if non_unit == sorted(non_unit):
        new = insert_node(
            graph, "Reshape", [x], node, attrs={"shape": list(node.output.type.shape)}
        )
        graph.replace_all_uses(node.output, new.output)
        return True
    return False


_RULES = (_identity_rewrite, _strength_reduce, _reshape_rewrite, _transpose_rewrite)


def simplify(graph: Graph, max_rounds: int = 10) -> int:
    """Apply the rewrites until nothing changes. Returns the number of rewrites."""
    total = 0
    for _ in range(max_rounds):
        changed = 0
        for node in list(graph.nodes):
            if all(not o.users and o not in graph.outputs for o in node.outputs):
                continue
            for rule in _RULES:
                if rule(graph, node):
                    changed += 1
                    break
        if changed:
            eliminate_dead_code(graph)
        total += changed
        if not changed:
            break
    return total
