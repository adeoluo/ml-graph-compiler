"""Static memory planning: liveness, aliasing, in-place reuse and offset assignment.

Every non-constant tensor gets a fixed byte offset in one arena that is allocated once, so the
generated code performs no allocation at run time. Planning has four steps.

1. **Storage groups.** View operators (Reshape, Flatten, Squeeze, Unsqueeze, Identity, Dropout)
   produce an alias of their input. An elementwise operator may also write its output over an
   input of the same shape and element size when that input dies at the operator (in place).
   Values that share storage are merged with union-find into one group.
2. **Liveness.** With nodes numbered in schedule order, a group is live on the closed interval
   ``[first definition, last use]``. Graph outputs live to the end; graph inputs and constants
   are external and need no arena space. A node's scratch space (for example the im2col buffer
   of a convolution) is a group live on ``[i, i]``.
3. **Offset assignment.** Two groups conflict when their intervals intersect; offsets must keep
   conflicting groups disjoint in memory. This is the dynamic storage allocation problem
   (NP-hard). The default strategy is *greedy by size* (Pisarchyk and Lee, "Efficient Memory
   Management for Deep Neural Net Inference", 2020): visit groups from largest to smallest and
   place each at the lowest aligned offset that does not overlap any already placed conflicting
   group. *First fit* visits groups in definition order instead, and *naive* gives every group
   its own space (no reuse).
4. **Bounds.** The maximum over time of the summed sizes of live groups is a lower bound on any
   valid arena; the report shows how close each strategy gets to it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from ml_graph_compiler import ops
from ml_graph_compiler.errors import GraphError
from ml_graph_compiler.ir import Graph, Node, Value

ALIAS_OPS = frozenset({"Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity", "Dropout"})
# Elementwise operators whose generated loop reads element i of an input before writing element i
# of the output, so the output may overwrite a same-shaped input.
INPLACE_OPS = frozenset(op for op in ops.REGISTRY if ops.is_elementwise(op)) | {"FusedElementwise"}

Strategy = Literal["greedy_by_size", "first_fit", "naive"]
STRATEGIES: tuple[Strategy, ...] = ("greedy_by_size", "first_fit", "naive")
WorkspaceFn = Callable[[Node], int]


def _align(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Location:
    """Where a value lives: ``arena`` (byte offset), ``input`` / ``output`` (index), ``const``."""

    kind: Literal["arena", "input", "output", "const"]
    index: int = 0


@dataclass
class Buffer:
    """One storage group: the values sharing it, its size and live interval, and its offset."""

    id: int
    size: int
    start: int
    end: int
    values: list[Value] = field(default_factory=list)
    offset: int = -1
    workspace_of: Node | None = None

    def overlaps_in_time(self, other: Buffer) -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass
class MemoryPlan:
    strategy: str
    alignment: int
    arena_bytes: int
    lower_bound: int
    naive_bytes: int
    buffers: list[Buffer]
    locations: dict[int, Location]
    workspaces: dict[int, int]
    copies: list[tuple[Value, int]]
    inplace: int
    aliases: int

    def location(self, value: Value) -> Location:
        if value.is_constant:
            return Location("const")
        try:
            return self.locations[id(value)]
        except KeyError as exc:
            raise GraphError(f"value {value.name!r} has no planned location") from exc

    def offset(self, value: Value) -> int:
        loc = self.location(value)
        if loc.kind != "arena":
            raise GraphError(f"value {value.name!r} is not in the arena ({loc.kind})")
        return loc.index

    def verify(self) -> None:
        """Check that no two buffers alive at the same time overlap in the arena."""
        placed = sorted((b for b in self.buffers if b.offset >= 0), key=lambda b: b.offset)
        for i, a in enumerate(placed):
            if a.offset % self.alignment:
                raise GraphError(f"buffer {a.id} is misaligned")
            if a.offset + a.size > self.arena_bytes:
                raise GraphError(f"buffer {a.id} exceeds the arena")
            for b in placed[i + 1 :]:
                if b.offset >= a.offset + a.size:
                    break
                if a.overlaps_in_time(b):
                    raise GraphError(f"buffers {a.id} and {b.id} overlap in time and space")

    def summary(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "arena_bytes": self.arena_bytes,
            "lower_bound_bytes": self.lower_bound,
            "naive_bytes": self.naive_bytes,
            "buffers": len(self.buffers),
            "aliases": self.aliases,
            "inplace": self.inplace,
            "reuse_factor": round(self.naive_bytes / self.arena_bytes, 2)
            if self.arena_bytes
            else 1.0,
        }


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _nbytes(value: Value) -> int:
    return value.tensor_type.nbytes


def _place(buffers: list[Buffer], order: list[Buffer], alignment: int) -> int:
    placed: list[Buffer] = []
    top = 0
    for buf in order:
        conflicts = sorted((p for p in placed if p.overlaps_in_time(buf)), key=lambda p: p.offset)
        offset = 0
        for p in conflicts:
            if offset + buf.size <= p.offset:
                break
            offset = max(offset, _align(p.offset + p.size, alignment))
        buf.offset = offset
        placed.append(buf)
        top = max(top, offset + buf.size)
    return _align(top, alignment)


def plan_memory(
    graph: Graph,
    strategy: Strategy = "greedy_by_size",
    alignment: int = 64,
    inplace: bool = True,
    workspace: WorkspaceFn | None = None,
) -> MemoryPlan:
    """Assign every non-constant value a location (see the module docstring)."""
    if strategy not in STRATEGIES:
        raise GraphError(f"unknown strategy {strategy!r}; known: {STRATEGIES}")
    n = len(graph.nodes)
    uf = _UnionFind()
    values: dict[int, Value] = {}
    external: dict[int, Location] = {}
    for i, v in enumerate(graph.inputs):
        values[id(v)] = v
        external[id(v)] = Location("input", i)

    last_use: dict[int, int] = {}
    for i, node in enumerate(graph.nodes):
        for used in node.inputs:
            if used is not None and not used.is_constant:
                last_use[id(used)] = i
    outputs = {id(v) for v in graph.outputs}
    for v in graph.outputs:
        last_use[id(v)] = n

    aliases = inplace_count = 0
    for i, node in enumerate(graph.nodes):
        for out in node.outputs:
            values[id(out)] = out
            uf.find(id(out))
        x = node.inputs[0] if node.inputs else None
        if node.op in ALIAS_OPS and x is not None and not x.is_constant:
            uf.union(id(x), id(node.outputs[0]))
            aliases += 1
            continue
        if node.op in ALIAS_OPS and x is not None and x.is_constant:
            external[id(node.outputs[0])] = Location("const")
            continue
        if not inplace or node.op not in INPLACE_OPS or len(node.outputs) != 1:
            continue
        out = node.output
        for cand in node.inputs:
            if cand is None or cand.is_constant or cand.type is None or out.type is None:
                continue
            root = uf.find(id(cand))
            if any(uf.find(e) == root for e in external):
                continue
            if cand.type.shape != out.type.shape or cand.type.itemsize != out.type.itemsize:
                continue
            member_ids = [k for k in values if uf.find(k) == root]
            if any(k in outputs for k in member_ids):
                continue
            if max(last_use.get(k, -1) for k in member_ids) != i:
                continue
            uf.union(root, id(out))
            inplace_count += 1
            break

    # Groups: root -> members; external roots (graph inputs / constants) take the external location.
    groups: dict[int, list[Value]] = {}
    for k, v in values.items():
        groups.setdefault(uf.find(k), []).append(v)
    locations: dict[int, Location] = {}
    copies: list[tuple[Value, int]] = []
    output_index: dict[int, int] = {}
    for j, v in enumerate(graph.outputs):
        if v.is_constant:
            copies.append((v, j))
            continue
        root = uf.find(id(v))
        if root in output_index or any(uf.find(e) == root for e in external):
            copies.append((v, j))
        else:
            output_index[root] = j

    defined_at: dict[int, int] = {}
    for i, node in enumerate(graph.nodes):
        for out in node.outputs:
            defined_at[id(out)] = i
    buffers: list[Buffer] = []
    for root, members in groups.items():
        ext = next((external[id(m)] for m in members if id(m) in external), None)
        if ext is not None:
            for m in members:
                locations[id(m)] = ext
            continue
        if root in output_index:
            for m in members:
                locations[id(m)] = Location("output", output_index[root])
            continue
        start = min(defined_at.get(id(m), 0) for m in members)
        end = max(max(last_use.get(id(m), -1) for m in members), start)
        size = _align(max(_nbytes(m) for m in members), alignment) if members else 0
        buffers.append(Buffer(len(buffers), size, start, end, members))

    workspaces: dict[int, int] = {}
    if workspace is not None:
        for i, node in enumerate(graph.nodes):
            size = workspace(node)
            if size > 0:
                buffers.append(
                    Buffer(len(buffers), _align(size, alignment), i, i, [], workspace_of=node)
                )

    lower = 0
    events: dict[int, int] = {}
    for b in buffers:
        events[b.start] = events.get(b.start, 0) + b.size
        events[b.end + 1] = events.get(b.end + 1, 0) - b.size
    live = 0
    for t in sorted(events):
        live += events[t]
        lower = max(lower, live)
    naive = sum(b.size for b in buffers)

    if strategy == "naive":
        offset = 0
        for b in buffers:
            b.offset = offset
            offset += b.size
        arena = offset
    elif strategy == "first_fit":
        arena = _place(buffers, sorted(buffers, key=lambda b: (b.start, -b.size)), alignment)
    else:
        arena = _place(buffers, sorted(buffers, key=lambda b: (-b.size, b.start)), alignment)

    for b in buffers:
        for m in b.values:
            locations[id(m)] = Location("arena", b.offset)
        if b.workspace_of is not None:
            workspaces[id(b.workspace_of)] = b.offset
    plan = MemoryPlan(
        strategy=strategy,
        alignment=alignment,
        arena_bytes=arena,
        lower_bound=lower,
        naive_bytes=naive,
        buffers=buffers,
        locations=locations,
        workspaces=workspaces,
        copies=copies,
        inplace=inplace_count,
        aliases=aliases,
    )
    plan.verify()
    return plan
