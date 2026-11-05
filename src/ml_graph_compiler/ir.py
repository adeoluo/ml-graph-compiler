"""Graph intermediate representation in static single assignment form.

A :class:`Graph` holds an ordered list of :class:`Node` objects. Every :class:`Value` is defined
exactly once: by a graph input, by a constant (an initializer with data), or by one node output.
Nodes keep references to their input values and each value keeps the list of nodes that use it,
so passes can rewrite the graph without rebuilding indexes.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from ml_graph_compiler.errors import GraphError

_DTYPES: dict[str, np.dtype[Any]] = {
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "float16": np.dtype(np.float16),
    "int64": np.dtype(np.int64),
    "int32": np.dtype(np.int32),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
    "bool": np.dtype(np.bool_),
}


def contiguous(array: Any, dtype: Any = None) -> np.ndarray:
    """A C-contiguous array that keeps zero-dimensional shapes (unlike ascontiguousarray)."""
    return np.asarray(array, dtype=dtype, order="C")


def numpy_dtype(name: str) -> np.dtype[Any]:
    """The numpy dtype for an IR dtype name."""
    try:
        return _DTYPES[name]
    except KeyError as exc:
        raise GraphError(f"unsupported dtype {name!r}") from exc


def dtype_name(dtype: np.dtype[Any] | type) -> str:
    """The IR dtype name for a numpy dtype."""
    dt = np.dtype(dtype)
    for name, candidate in _DTYPES.items():
        if candidate == dt:
            return name
    raise GraphError(f"unsupported numpy dtype {dt}")


@dataclass(frozen=True)
class TensorType:
    """Element type and static shape of a tensor."""

    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        numpy_dtype(self.dtype)
        if any((not isinstance(d, int)) or d < 0 for d in self.shape):
            raise GraphError(f"shape must be non-negative integers, got {self.shape}")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1

    @property
    def itemsize(self) -> int:
        return numpy_dtype(self.dtype).itemsize

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    def __str__(self) -> str:
        dims = "x".join(str(d) for d in self.shape)
        return f"{self.dtype}[{dims}]"

    @staticmethod
    def of(array: np.ndarray) -> TensorType:
        return TensorType(dtype_name(array.dtype), tuple(int(d) for d in array.shape))


class Value:
    """An SSA value: a graph input, a constant, or one output of one node."""

    __slots__ = ("data", "name", "producer", "type", "users")

    def __init__(
        self,
        name: str,
        type: TensorType | None = None,
        data: np.ndarray | None = None,
    ) -> None:
        self.name = name
        self.type = type
        self.data = data
        self.producer: Node | None = None
        self.users: list[Node] = []

    @property
    def is_constant(self) -> bool:
        return self.data is not None

    @property
    def tensor_type(self) -> TensorType:
        if self.type is None:
            raise GraphError(f"value {self.name!r} has no inferred type")
        return self.type

    def __repr__(self) -> str:
        return f"Value({self.name!r}, {self.type})"


class Node:
    """One operation. ``inputs`` may contain ``None`` for omitted optional ONNX inputs."""

    __slots__ = ("attrs", "inputs", "name", "op", "outputs")

    def __init__(
        self,
        op: str,
        inputs: list[Value | None],
        outputs: list[Value],
        attrs: dict[str, Any] | None = None,
        name: str = "",
    ) -> None:
        self.op = op
        self.inputs = inputs
        self.outputs = outputs
        self.attrs: dict[str, Any] = dict(attrs or {})
        self.name = name

    def input(self, i: int) -> Value:
        """The i-th input, which must be present."""
        if i >= len(self.inputs) or self.inputs[i] is None:
            raise GraphError(f"{self.op} node {self.name!r} is missing input {i}")
        value = self.inputs[i]
        assert value is not None
        return value

    def optional_input(self, i: int) -> Value | None:
        return self.inputs[i] if i < len(self.inputs) else None

    @property
    def output(self) -> Value:
        return self.outputs[0]

    def __repr__(self) -> str:
        ins = ", ".join(v.name if v is not None else "_" for v in self.inputs)
        outs = ", ".join(v.name for v in self.outputs)
        return f"{outs} = {self.op}({ins}) {self.attrs if self.attrs else ''}".rstrip()


class Graph:
    """A dataflow graph with ordered nodes, graph inputs, outputs and constants."""

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.opset = 17
        self.inputs: list[Value] = []
        self.outputs: list[Value] = []
        # Public output names stay fixed while passes replace the values behind them.
        self.output_names: list[str] = []
        self.nodes: list[Node] = []
        self._names: set[str] = set()
        self._counter = itertools.count()

    # -- naming -----------------------------------------------------------------------------
    def fresh_name(self, hint: str = "v") -> str:
        while True:
            name = f"{hint}_{next(self._counter)}"
            if name not in self._names:
                self._names.add(name)
                return name

    def _claim(self, name: str) -> str:
        if not name or name in self._names:
            return self.fresh_name(name or "v")
        self._names.add(name)
        return name

    # -- construction -----------------------------------------------------------------------
    def add_input(self, name: str, type: TensorType) -> Value:
        value = Value(self._claim(name), type)
        self.inputs.append(value)
        return value

    def constant(self, array: np.ndarray, name: str = "const") -> Value:
        """A new constant value (not attached to any node)."""
        data = contiguous(array)
        return Value(self._claim(name), TensorType.of(data), data)

    def add_node(
        self,
        op: str,
        inputs: list[Value | None],
        n_outputs: int = 1,
        attrs: dict[str, Any] | None = None,
        name: str = "",
        output_names: list[str] | None = None,
        index: int | None = None,
    ) -> Node:
        """Create a node (appended, or inserted at ``index``) with fresh output values."""
        names = output_names or [f"{name or op.lower()}_out" for _ in range(n_outputs)]
        outputs = [Value(self._claim(n)) for n in names]
        node = Node(op, list(inputs), outputs, attrs, name or self.fresh_name(op.lower()))
        for out in outputs:
            out.producer = node
        for value in inputs:
            if value is not None:
                value.users.append(node)
        if index is None:
            self.nodes.append(node)
        else:
            self.nodes.insert(index, node)
        return node

    # -- mutation ---------------------------------------------------------------------------
    def set_input(self, node: Node, i: int, value: Value | None) -> None:
        old = node.inputs[i]
        if old is not None:
            old.users.remove(node)
        node.inputs[i] = value
        if value is not None:
            value.users.append(node)

    def remove_node(self, node: Node) -> None:
        """Remove a node whose outputs are no longer used."""
        for out in node.outputs:
            if out.users or out in self.outputs:
                raise GraphError(f"cannot remove {node.op} {node.name!r}: output still used")
        for value in node.inputs:
            if value is not None:
                value.users.remove(node)
        self.nodes.remove(node)

    def remove_nodes(self, dead: Iterable[Node]) -> int:
        """Remove several nodes at once (their outputs must be unused outside the set)."""
        dead_set = set(dead)
        if not dead_set:
            return 0
        for node in dead_set:
            for value in node.inputs:
                if value is not None and node in value.users:
                    value.users.remove(node)
        self.nodes = [n for n in self.nodes if n not in dead_set]
        return len(dead_set)

    def replace_all_uses(self, old: Value, new: Value) -> None:
        """Point every user of ``old`` (and the graph outputs) at ``new``."""
        if old is new:
            return
        for user in list(old.users):
            for i, value in enumerate(user.inputs):
                if value is old:
                    user.inputs[i] = new
                    new.users.append(user)
        old.users = []
        self.outputs = [new if v is old else v for v in self.outputs]

    def add_output(self, value: Value, name: str | None = None) -> None:
        self.outputs.append(value)
        self.output_names.append(name or value.name)

    def named_outputs(self) -> list[tuple[str, Value]]:
        names = self.output_names or [v.name for v in self.outputs]
        return list(zip(names, self.outputs, strict=True))

    # -- queries ----------------------------------------------------------------------------
    def values(self) -> Iterator[Value]:
        """Every value reachable from the nodes, inputs and outputs, each once."""
        seen: set[int] = set()
        for value in itertools.chain(
            self.inputs,
            (v for n in self.nodes for v in n.inputs if v is not None),
            (v for n in self.nodes for v in n.outputs),
            self.outputs,
        ):
            if id(value) not in seen:
                seen.add(id(value))
                yield value

    def constants(self) -> list[Value]:
        return [v for v in self.values() if v.is_constant]

    def op_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.op] = counts.get(node.op, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def constant_bytes(self) -> int:
        return sum(v.data.nbytes for v in self.constants() if v.data is not None)

    # -- ordering and checks ----------------------------------------------------------------
    def topological_sort(self) -> None:
        """Reorder nodes topologically, keeping the current order where it is already valid."""
        position = {id(n): i for i, n in enumerate(self.nodes)}
        indegree: dict[int, int] = {}
        for node in self.nodes:
            preds = {
                id(v.producer)
                for v in node.inputs
                if v is not None and v.producer is not None and id(v.producer) in position
            }
            indegree[id(node)] = len(preds)
        import heapq

        ready = [(position[id(n)], n) for n in self.nodes if indegree[id(n)] == 0]
        heapq.heapify(ready)
        order: list[Node] = []
        while ready:
            _, node = heapq.heappop(ready)
            order.append(node)
            released: set[int] = set()
            for out in node.outputs:
                for user in out.users:
                    if id(user) in released or id(user) not in indegree:
                        continue
                    if any(v is not None and v.producer is node for v in user.inputs):
                        released.add(id(user))
                        indegree[id(user)] -= 1
                        if indegree[id(user)] == 0:
                            heapq.heappush(ready, (position[id(user)], user))
        if len(order) != len(self.nodes):
            raise GraphError("graph has a cycle")
        self.nodes = order

    def verify(self) -> None:
        """Check SSA form, topological order and use lists; raise GraphError on violation."""
        defined: set[int] = {id(v) for v in self.inputs}
        producers: set[int] = set()
        for node in self.nodes:
            for value in node.inputs:
                if value is None:
                    continue
                if not value.is_constant and id(value) not in defined:
                    raise GraphError(f"{node.op} {node.name!r} uses undefined {value.name!r}")
                if node not in value.users:
                    raise GraphError(f"use list of {value.name!r} misses {node.name!r}")
            for out in node.outputs:
                if id(out) in producers or id(out) in defined:
                    raise GraphError(f"value {out.name!r} is defined twice")
                if out.producer is not node:
                    raise GraphError(f"producer of {out.name!r} is wrong")
                producers.add(id(out))
                defined.add(id(out))
        live = {id(n) for n in self.nodes}
        for value in self.values():
            for user in value.users:
                if id(user) not in live:
                    raise GraphError(f"{value.name!r} lists removed user {user.name!r}")
        for value in self.outputs:
            if not value.is_constant and id(value) not in defined:
                raise GraphError(f"graph output {value.name!r} is undefined")

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nodes": len(self.nodes),
            "inputs": {v.name: str(v.type) for v in self.inputs},
            "outputs": {n: str(v.type) for n, v in self.named_outputs()},
            "constants": len(self.constants()),
            "constant_bytes": self.constant_bytes(),
            "ops": self.op_counts(),
        }

    def __str__(self) -> str:
        lines = [f"graph {self.name}("]
        lines += [f"  {v.name}: {v.type}," for v in self.inputs]
        lines.append(") {")
        for node in self.nodes:
            types = ", ".join(str(v.type) for v in node.outputs)
            lines.append(f"  {node!r}  -> {types}")
        returned = ", ".join(f"{n}={v.name}" for n, v in self.named_outputs())
        lines.append(f"  return {returned}")
        lines.append("}")
        return "\n".join(lines)
