"""Reference interpreter: executes the IR with the numpy semantics in :mod:`ops`."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ml_graph_compiler import ops
from ml_graph_compiler.errors import GraphError
from ml_graph_compiler.ir import Graph, Node, Value, contiguous, numpy_dtype


@dataclass
class Profile:
    """Wall-clock seconds per operator type, summed over one run."""

    seconds_by_op: dict[str, float] = field(default_factory=dict)
    peak_live_bytes: int = 0

    def add(self, op: str, seconds: float) -> None:
        self.seconds_by_op[op] = self.seconds_by_op.get(op, 0.0) + seconds


def check_feeds(graph: Graph, feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate input names, shapes and dtypes; return contiguous arrays keyed by name."""
    expected = {v.name for v in graph.inputs}
    missing = expected - set(feeds)
    extra = set(feeds) - expected
    if missing or extra:
        raise GraphError(f"inputs: missing {sorted(missing)}, unexpected {sorted(extra)}")
    out: dict[str, np.ndarray] = {}
    for value in graph.inputs:
        array = np.asarray(feeds[value.name])
        t = value.tensor_type
        if tuple(array.shape) != t.shape:
            raise GraphError(f"input {value.name!r}: expected shape {t.shape}, got {array.shape}")
        if array.dtype != numpy_dtype(t.dtype):
            raise GraphError(f"input {value.name!r}: expected {t.dtype}, got {array.dtype}")
        out[value.name] = contiguous(array)
    return out


class Interpreter:
    """Runs a graph node by node, freeing each intermediate after its last use."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        last_use: dict[int, int] = {}
        for i, node in enumerate(graph.nodes):
            for value in node.inputs:
                if value is not None:
                    last_use[id(value)] = i
        keep = {id(v) for v in graph.outputs}
        self._release: list[list[Value]] = [[] for _ in graph.nodes]
        seen: set[int] = set()
        for i, node in enumerate(graph.nodes):
            for value in node.inputs:
                if (
                    value is not None
                    and not value.is_constant
                    and id(value) not in keep
                    and id(value) not in seen
                    and last_use[id(value)] == i
                ):
                    self._release[i].append(value)
                    seen.add(id(value))

    def run(
        self, feeds: Mapping[str, np.ndarray], profile: Profile | None = None
    ) -> dict[str, np.ndarray]:
        env: dict[int, np.ndarray] = {}
        checked = check_feeds(self.graph, feeds)
        for value in self.graph.inputs:
            env[id(value)] = checked[value.name]
        live = sum(a.nbytes for a in env.values())
        for i, node in enumerate(self.graph.nodes):
            arrays = [self._get(env, v) for v in node.inputs]
            start = time.perf_counter()
            results = ops.run(node, arrays)
            if profile is not None:
                profile.add(node.op, time.perf_counter() - start)
            self._store(env, node, results)
            live += sum(r.nbytes for r in results)
            if profile is not None:
                profile.peak_live_bytes = max(profile.peak_live_bytes, live)
            for value in self._release[i]:
                released = env.pop(id(value), None)
                if released is not None:
                    live -= released.nbytes
        return {n: self._get_required(env, v) for n, v in self.graph.named_outputs()}

    @staticmethod
    def _get(env: dict[int, np.ndarray], value: Value | None) -> np.ndarray | None:
        if value is None:
            return None
        if value.data is not None:
            return value.data
        if id(value) not in env:
            raise GraphError(f"value {value.name!r} used before it is defined")
        return env[id(value)]

    def _get_required(self, env: dict[int, np.ndarray], value: Value) -> np.ndarray:
        array = self._get(env, value)
        assert array is not None
        return array

    @staticmethod
    def _store(env: dict[int, np.ndarray], node: Node, results: list[np.ndarray]) -> None:
        if len(results) != len(node.outputs):
            raise GraphError(f"{node.op} returned {len(results)} outputs")
        for value, array in zip(node.outputs, results, strict=True):
            t = value.type
            if t is not None and tuple(array.shape) != t.shape:
                raise GraphError(
                    f"{node.op} {node.name!r}: produced shape {array.shape}, inferred {t.shape}"
                )
            env[id(value)] = array


def run_graph(graph: Graph, feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convenience wrapper: interpret ``graph`` once."""
    return Interpreter(graph).run(feeds)


def random_feeds(graph: Graph, seed: int = 0, int_high: int = 100) -> dict[str, np.ndarray]:
    """Deterministic random inputs: standard normal floats, integers in [0, int_high)."""
    rng = np.random.default_rng(seed)
    feeds: dict[str, np.ndarray] = {}
    for value in graph.inputs:
        t = value.tensor_type
        dtype = numpy_dtype(t.dtype)
        if np.issubdtype(dtype, np.floating):
            feeds[value.name] = rng.standard_normal(t.shape).astype(dtype)
        elif dtype == np.bool_:
            feeds[value.name] = rng.integers(0, 2, t.shape).astype(np.bool_)
        else:
            feeds[value.name] = rng.integers(0, int_high, t.shape).astype(dtype)
    return feeds
