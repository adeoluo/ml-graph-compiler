"""Operator scheduling: memory-aware list scheduling and dataflow (wavefront) analysis.

Any topological order of the nodes computes the same values, but orders differ in how many
intermediate tensors are alive at once. Finding the order with the smallest peak is NP-hard in
general (it contains register sufficiency), so :func:`memory_aware_order` uses the standard
greedy list-scheduling heuristic: keep the set of *ready* nodes (all producers scheduled) and
repeatedly pick the one whose execution changes the live footprint the least,

    delta(n) = bytes(outputs of n) - bytes(inputs of n whose last remaining user is n),

breaking ties by the original program order so that the result is deterministic and close to
the source order. :func:`wavefronts` computes ASAP levels (the length of the longest path from
an input) which bounds the operator-level parallelism available to a multi-stream executor, and
:func:`critical_path_flops` the FLOP-weighted longest path.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ml_graph_compiler import ops
from ml_graph_compiler.errors import GraphError
from ml_graph_compiler.ir import Graph, Node, Value

# Operators that never allocate: their output is a view of their input (see memory.ALIAS_OPS).
_VIEW_OPS = frozenset({"Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity", "Dropout"})


def _bytes(value: Value) -> int:
    return 0 if value.is_constant or value.type is None else value.type.nbytes


def _producers(node: Node, members: set[int]) -> set[int]:
    return {
        id(v.producer)
        for v in node.inputs
        if v is not None and v.producer is not None and id(v.producer) in members
    }


def _root(value: Value) -> Value:
    """The tensor that owns the storage of ``value`` (following view operators)."""
    while value.producer is not None and value.producer.op in _VIEW_OPS:
        value = value.producer.input(0)
    return value


def peak_live_bytes(graph: Graph, order: Sequence[Node] | None = None) -> int:
    """Peak of the live set over a schedule, counting a node's inputs and outputs together.

    A tensor is alive from its definition until the last use of it or of any view of it; graph
    outputs stay alive to the end. Views (Reshape and friends) own no storage.
    """
    order = list(graph.nodes if order is None else order)
    last: dict[int, int] = {}
    for i, node in enumerate(order):
        for v in node.inputs:
            if v is not None and not v.is_constant:
                last[id(_root(v))] = i
    for v in graph.outputs:
        last[id(_root(v))] = len(order)
    live = sum(_bytes(v) for v in graph.inputs)
    peak = live
    dying: dict[int, list[Value]] = {}
    for v in graph.inputs:
        dying.setdefault(last.get(id(v), -1), []).append(v)
    live -= sum(_bytes(v) for v in dying.pop(-1, []))
    for i, node in enumerate(order):
        if node.op not in _VIEW_OPS:
            for out in node.outputs:
                live += _bytes(out)
                dying.setdefault(last.get(id(out), i), []).append(out)
        peak = max(peak, live)
        live -= sum(_bytes(v) for v in dying.pop(i, []))
    return peak


def memory_aware_order(graph: Graph) -> list[Node]:
    """A topological order chosen greedily to keep the live set small (see module docstring)."""
    nodes = list(graph.nodes)
    members = {id(n) for n in nodes}
    index = {id(n): i for i, n in enumerate(nodes)}
    pending = {id(n): len(_producers(n, members)) for n in nodes}
    remaining_users: dict[int, int] = {}
    for n in nodes:
        for v in {id(v): v for v in n.inputs if v is not None}.values():
            remaining_users[id(v)] = remaining_users.get(id(v), 0) + 1
    outputs = {id(v) for v in graph.outputs}

    def delta(node: Node) -> int:
        if node.op in _VIEW_OPS:
            return 0
        alloc = sum(_bytes(o) for o in node.outputs)
        freed = 0
        for v in {id(v): v for v in node.inputs if v is not None}.values():
            if remaining_users.get(id(v)) == 1 and id(v) not in outputs:
                freed += _bytes(v)
        return alloc - freed

    ready = [n for n in nodes if pending[id(n)] == 0]
    order: list[Node] = []
    while ready:
        best = min(ready, key=lambda n: (delta(n), index[id(n)]))
        ready.remove(best)
        order.append(best)
        for v in {id(v): v for v in best.inputs if v is not None}.values():
            remaining_users[id(v)] -= 1
        released: set[int] = set()
        for out in best.outputs:
            for user in out.users:
                if id(user) in members and id(user) not in released:
                    released.add(id(user))
                    pending[id(user)] -= 1
                    if pending[id(user)] == 0:
                        ready.append(user)
    if len(order) != len(nodes):
        raise GraphError("graph has a cycle")
    return order


def apply_order(graph: Graph, order: Iterable[Node]) -> None:
    """Replace the node order (must be a permutation of the nodes, in topological order)."""
    new = list(order)
    if len(new) != len(graph.nodes) or {id(n) for n in new} != {id(n) for n in graph.nodes}:
        raise GraphError("schedule is not a permutation of the graph's nodes")
    old = graph.nodes
    graph.nodes = new
    try:
        graph.verify()
    except GraphError:
        graph.nodes = old
        raise


def wavefronts(graph: Graph) -> list[list[Node]]:
    """ASAP levels: level(n) = 1 + max level of its producers; nodes in a level are independent."""
    level: dict[int, int] = {}
    levels: list[list[Node]] = []
    for node in graph.nodes:
        preds = [
            level[id(v.producer)] for v in node.inputs if v is not None and v.producer is not None
        ]
        lv = 1 + max(preds) if preds else 0
        level[id(node)] = lv
        while len(levels) <= lv:
            levels.append([])
        levels[lv].append(node)
    return levels


def critical_path_flops(graph: Graph) -> tuple[int, int]:
    """(FLOPs on the longest FLOP-weighted path, total FLOPs): their ratio bounds the speedup
    from running independent operators concurrently."""
    longest: dict[int, int] = {}
    total = 0
    for node in graph.nodes:
        f = ops.flops(node)
        total += f
        preds = [
            longest[id(v.producer)] for v in node.inputs if v is not None and v.producer is not None
        ]
        longest[id(node)] = f + (max(preds) if preds else 0)
    return (max(longest.values()) if longest else 0), total


@dataclass
class ScheduleReport:
    nodes: int
    peak_before: int
    peak_after: int
    levels: int
    max_width: int
    critical_flops: int
    total_flops: int


def schedule(graph: Graph) -> ScheduleReport:
    """Reorder the graph with :func:`memory_aware_order` when it lowers the peak live bytes."""
    before = peak_live_bytes(graph)
    order = memory_aware_order(graph)
    after = peak_live_bytes(graph, order)
    if after < before:
        apply_order(graph, order)
    else:
        after = before
    levels = wavefronts(graph)
    crit, total = critical_path_flops(graph)
    return ScheduleReport(
        nodes=len(graph.nodes),
        peak_before=before,
        peak_after=after,
        levels=len(levels),
        max_width=max((len(lv) for lv in levels), default=0),
        critical_flops=crit,
        total_flops=total,
    )
