"""Loop fusion: chains of elementwise operators become one FusedElementwise node.

Each elementwise operator, executed alone, reads its inputs from memory and writes a full output
tensor. When an operator's only consumer is another elementwise operator with the same output
shape, the two can run in one loop that keeps the intermediate in a register:
``y = (x * s + m)`` costs one pass over memory instead of two, and the intermediate tensor needs
no buffer. The pass grows a group backwards from each elementwise node, absorbing producers
that are elementwise, have exactly one user inside the group, are not graph outputs and produce
the group's output shape (so no value is recomputed per element of a larger broadcast). The
group becomes a FusedElementwise node whose ``program`` lists the original operators; the
reference interpreter replays it and the C backend emits it as a single loop nest.
"""

from __future__ import annotations

from typing import Any

from ml_graph_compiler import ops
from ml_graph_compiler.ir import Graph, Node, Value
from ml_graph_compiler.passes.basic import eliminate_dead_code
from ml_graph_compiler.passes.util import insert_node

MAX_PROGRAM = 32


def _fusible(node: Node) -> bool:
    return len(node.outputs) == 1 and (ops.is_elementwise(node.op) or node.op == "FusedElementwise")


def fuse_elementwise(graph: Graph) -> int:
    """Group elementwise chains into FusedElementwise nodes; returns the number of groups."""
    position = {id(n): i for i, n in enumerate(graph.nodes)}
    assigned: set[int] = set()
    groups: list[list[Node]] = []
    for root in reversed(graph.nodes):
        if id(root) in assigned or not _fusible(root) or root.op == "FusedElementwise":
            continue
        shape = root.output.tensor_type.shape
        members = [root]
        frontier = [root]
        while frontier and len(members) < MAX_PROGRAM:
            node = frontier.pop()
            for v in node.inputs:
                p = v.producer if v is not None else None
                if (
                    p is None
                    or id(p) in assigned
                    or p in members
                    or not _fusible(p)
                    or p.op == "FusedElementwise"
                    or v in graph.outputs
                    or v is None
                    or v.tensor_type.shape != shape
                    or any(u not in members for u in v.users)
                ):
                    continue
                members.append(p)
                frontier.append(p)
        if len(members) < 2:
            continue
        assigned.update(id(m) for m in members)
        groups.append(sorted(members, key=lambda n: position[id(n)]))

    for members in groups:
        _build(graph, members)
    if groups:
        eliminate_dead_code(graph)
    return len(groups)


def _build(graph: Graph, members: list[Node]) -> None:
    root = members[-1]
    produced = {id(m.output): k for k, m in enumerate(members)}
    leaves: list[Value | None] = []
    leaf_index: dict[int, int] = {}
    for m in members:
        for v in m.inputs:
            if v is not None and id(v) not in produced and id(v) not in leaf_index:
                leaf_index[id(v)] = len(leaves)
                leaves.append(v)
    n_in = len(leaves)
    program: list[list[Any]] = []
    for m in members:
        args = []
        for v in m.inputs:
            if v is None:
                args.append(-1)
            elif id(v) in produced:
                args.append(n_in + produced[id(v)])
            else:
                args.append(leaf_index[id(v)])
        program.append([m.op, dict(m.attrs), args])
    fused = insert_node(graph, "FusedElementwise", leaves, root, attrs={"program": program})
    graph.replace_all_uses(root.output, fused.output)
