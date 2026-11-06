"""Load and run a compiled model through ctypes."""

from __future__ import annotations

import ctypes
import json
import shutil
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ml_graph_compiler.codegen.build import BuildOptions, BuildResult, build_library
from ml_graph_compiler.codegen.c import GeneratedModel, generate_c
from ml_graph_compiler.errors import CodegenError, GraphError
from ml_graph_compiler.ir import Graph, TensorType, numpy_dtype
from ml_graph_compiler.memory import Strategy


def _aligned_empty(nbytes: int, alignment: int = 64) -> np.ndarray:
    raw = np.empty(nbytes + alignment, np.uint8)
    start = (-raw.ctypes.data) % alignment
    return raw[start : start + nbytes]


@dataclass
class CompileReport:
    nodes: int
    c_lines: int
    arena_bytes: int
    weights_bytes: int
    codegen_seconds: float
    build_seconds: float
    cached: bool
    openmp: bool
    kernels: dict[str, int] = field(default_factory=dict)
    plan: dict[str, object] = field(default_factory=dict)


class CompiledModel:
    """A model compiled to native code: ``run(feeds)`` returns outputs as numpy arrays.

    The arena is allocated once; calls are serialized with a lock because they share it.
    """

    def __init__(
        self,
        library: Path,
        weights: np.ndarray,
        inputs: list[tuple[str, TensorType]],
        outputs: list[tuple[str, TensorType]],
        report: CompileReport | None = None,
    ) -> None:
        self.library_path = Path(library)
        self.lib = ctypes.CDLL(str(library))
        self.lib.mgc_run.restype = ctypes.c_int
        self.lib.mgc_run.argtypes = [ctypes.c_void_p] * 4
        self.lib.mgc_arena_bytes.restype = ctypes.c_size_t
        self.lib.mgc_weights_bytes.restype = ctypes.c_size_t
        self.lib.mgc_get_max_threads.restype = ctypes.c_int
        self.lib.mgc_has_openmp.restype = ctypes.c_int
        self.lib.mgc_has_neon.restype = ctypes.c_int
        self.lib.mgc_set_num_threads.argtypes = [ctypes.c_int]
        self.lib.mgc_set_blocking.argtypes = [ctypes.c_int] * 3
        expected = int(self.lib.mgc_weights_bytes())
        if weights.nbytes != expected:
            raise CodegenError(
                f"weights blob has {weights.nbytes} bytes, library expects {expected}"
            )
        self.weights = weights if weights.ctypes.data % 64 == 0 else self._realign(weights)
        self.node_names: list[tuple[str, str]] = []
        self.arena = _aligned_empty(int(self.lib.mgc_arena_bytes()))
        self.inputs = inputs
        self.outputs = outputs
        self.report = report
        self._lock = threading.Lock()
        self._in_ptrs = (ctypes.c_void_p * max(len(inputs), 1))()
        self._out_ptrs = (ctypes.c_void_p * max(len(outputs), 1))()

    @staticmethod
    def _realign(weights: np.ndarray) -> np.ndarray:
        out = _aligned_empty(weights.nbytes)
        out[:] = weights
        return out

    @property
    def openmp(self) -> bool:
        return bool(self.lib.mgc_has_openmp())

    @property
    def neon(self) -> bool:
        return bool(self.lib.mgc_has_neon())

    def set_threads(self, n: int) -> None:
        self.lib.mgc_set_num_threads(int(n))

    def max_threads(self) -> int:
        return int(self.lib.mgc_get_max_threads())

    def set_blocking(self, mc: int, kc: int, npb: int) -> None:
        self.lib.mgc_set_blocking(int(mc), int(kc), int(npb))

    def profile(self) -> list[tuple[str, str, float]]:
        """(op, node name, accumulated seconds) per node; needs a model built with profile=True."""
        if not hasattr(self.lib, "mgc_profile_data"):
            raise CodegenError("the model was not compiled with profile=True")
        self.lib.mgc_profile_data.restype = ctypes.POINTER(ctypes.c_double)
        self.lib.mgc_profile_size.restype = ctypes.c_int
        data = self.lib.mgc_profile_data()
        n = int(self.lib.mgc_profile_size())
        return [(op, name, float(data[i])) for i, (op, name) in enumerate(self.node_names[:n])]

    def run(self, feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        names = [n for n, _ in self.inputs]
        missing = set(names) - set(feeds)
        extra = set(feeds) - set(names)
        if missing or extra:
            raise GraphError(f"inputs: missing {sorted(missing)}, unexpected {sorted(extra)}")
        arrays = []
        for i, (name, t) in enumerate(self.inputs):
            a = np.asarray(feeds[name], order="C")
            if a.shape != t.shape or a.dtype != numpy_dtype(t.dtype):
                raise GraphError(f"input {name!r}: expected {t}, got {a.dtype}{list(a.shape)}")
            arrays.append(a)
            self._in_ptrs[i] = a.ctypes.data
        results = {}
        for j, (name, t) in enumerate(self.outputs):
            out = np.empty(t.shape, numpy_dtype(t.dtype))
            results[name] = out
            self._out_ptrs[j] = out.ctypes.data
        with self._lock:
            code = self.lib.mgc_run(
                ctypes.addressof(self._in_ptrs),
                ctypes.addressof(self._out_ptrs),
                self.arena.ctypes.data,
                self.weights.ctypes.data,
            )
        if code != 0:
            raise GraphError("compiled model reported an out-of-range Gather index")
        return results

    def benchmark(
        self, feeds: Mapping[str, np.ndarray], warmup: int = 3, repeat: int = 20
    ) -> list[float]:
        for _ in range(warmup):
            self.run(feeds)
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            self.run(feeds)
            times.append(time.perf_counter() - start)
        return times

    # -- persistence -----------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Write the library, weights and I/O metadata to ``directory``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        lib = d / self.library_path.name
        if lib.resolve() != self.library_path.resolve():
            lib.write_bytes(self.library_path.read_bytes())
        source = getattr(self.weights, "filename", None)
        target = d / "weights.bin"
        if source is not None and Path(source).exists():
            if Path(source).resolve() != target.resolve():
                shutil.copyfile(source, target)
        else:
            np.asarray(self.weights).tofile(target)
        meta = {
            "library": lib.name,
            "inputs": [[n, t.dtype, list(t.shape)] for n, t in self.inputs],
            "outputs": [[n, t.dtype, list(t.shape)] for n, t in self.outputs],
            "report": self.report.__dict__ if self.report else None,
            "nodes": [list(pair) for pair in self.node_names],
        }
        (d / "model.json").write_text(json.dumps(meta, indent=2) + "\n")
        return d

    @classmethod
    def load(cls, directory: str | Path) -> CompiledModel:
        d = Path(directory)
        meta = json.loads((d / "model.json").read_text())
        weights = np.memmap(d / "weights.bin", dtype=np.uint8, mode="r")
        io = {
            k: [(n, TensorType(dt, tuple(s))) for n, dt, s in meta[k]]
            for k in ("inputs", "outputs")
        }
        model = cls(d / meta["library"], weights, io["inputs"], io["outputs"])
        model.node_names = [(str(op), str(name)) for op, name in meta.get("nodes", [])]
        return model


def compile_graph(
    graph: Graph,
    options: BuildOptions | None = None,
    cache_dir: str | Path | None = None,
    strategy: Strategy = "greedy_by_size",
    inplace: bool = True,
    name: str = "model",
    profile: bool = False,
) -> CompiledModel:
    """Generate C for an (already optimized and scheduled) graph, build it and load it.

    The weights blob is written next to the library (named by its content hash) and memory
    mapped read-only, so the process never holds a private copy of the packed weights.
    """
    start = time.perf_counter()
    generated: GeneratedModel = generate_c(graph, strategy, inplace, profile)
    codegen_seconds = time.perf_counter() - start
    start = time.perf_counter()
    built: BuildResult = build_library(
        generated.source, options, Path(cache_dir) if cache_dir is not None else None, name
    )
    weights_path = built.directory / f"weights-{generated.weights.digest()}.bin"
    if not weights_path.exists():
        generated.weights.write(weights_path)
    weights = np.memmap(weights_path, dtype=np.uint8, mode="r")
    report = CompileReport(
        nodes=len(graph.nodes),
        c_lines=generated.lines,
        arena_bytes=generated.arena_bytes,
        weights_bytes=int(generated.weights.nbytes),
        codegen_seconds=round(codegen_seconds, 4),
        build_seconds=round(time.perf_counter() - start, 4),
        cached=built.cached,
        openmp=built.openmp,
        kernels=generated.kernels,
        plan=generated.plan.summary(),
    )
    model = CompiledModel(built.library, weights, generated.inputs, generated.outputs, report)
    model.node_names = [(n.op, n.name) for n in graph.nodes]
    return model


def compile_model(
    model: Any,
    opt: str = "O3",
    input_shapes: dict[str, tuple[int, ...]] | None = None,
    dims: dict[str, int] | None = None,
    reschedule: bool = True,
    **kwargs: Any,
) -> CompiledModel:
    """ONNX model (proto or path) to a loaded native model: import, optimize, schedule, compile."""
    from ml_graph_compiler.onnx_import import from_onnx
    from ml_graph_compiler.passes import optimize
    from ml_graph_compiler.schedule import schedule

    graph = from_onnx(model, input_shapes, dims)
    optimize(graph, opt)
    if reschedule:
        schedule(graph)
    return compile_graph(graph, **kwargs)
