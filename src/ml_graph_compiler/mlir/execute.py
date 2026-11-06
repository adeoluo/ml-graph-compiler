"""Verify, lower and run exported MLIR with the LLVM tools (mlir-opt, mlir-translate, clang).

The lowering pipeline is the standard tensor to LLVM path: one-shot bufferization with function
boundaries (identity layouts, so every argument and result is a contiguous memref), linalg to
loops, then the arith, math, memref, control flow and func conversions to the LLVM dialect.
``llvm.emit_c_interface`` on ``@main`` produces ``_mlir_ciface_main(results*, inputs*...)``,
which :class:`MlirExecutable` calls through ctypes with memref descriptors.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml_graph_compiler.codegen.build import LIB_SUFFIX, default_cache_dir
from ml_graph_compiler.errors import CodegenError, GraphError
from ml_graph_compiler.ir import Graph, TensorType, numpy_dtype
from ml_graph_compiler.mlir.emit import to_mlir

PIPELINE = (
    "builtin.module("
    "func.func(linalg-generalize-named-ops),"
    "one-shot-bufferize{bufferize-function-boundaries "
    "function-boundary-type-conversion=identity-layout-map},"
    "buffer-deallocation-pipeline,"
    "convert-linalg-to-loops,"
    "convert-bufferization-to-memref,"
    "expand-strided-metadata,"
    "lower-affine,"
    "convert-scf-to-cf,"
    "convert-math-to-llvm,"
    "convert-math-to-libm,"
    "finalize-memref-to-llvm,"
    "convert-func-to-llvm,"
    "convert-arith-to-llvm,"
    "convert-cf-to-llvm,"
    "reconcile-unrealized-casts)"
)

_SEARCH = (
    "/opt/homebrew/opt/llvm/bin",
    "/usr/local/opt/llvm/bin",
    *sorted(glob.glob("/usr/lib/llvm-*/bin"), reverse=True),
)


@dataclass(frozen=True)
class MlirTools:
    mlir_opt: str
    mlir_translate: str
    clang: str


def find_mlir_tools() -> MlirTools:
    """mlir-opt, mlir-translate and a clang from the same LLVM (``$MGC_MLIR_BIN`` first)."""
    dirs = [os.environ.get("MGC_MLIR_BIN", "")]
    found = shutil.which("mlir-opt")
    if found:
        dirs.append(str(Path(found).parent))
    dirs += list(_SEARCH)
    for d in dirs:
        if not d:
            continue
        tools = [Path(d) / name for name in ("mlir-opt", "mlir-translate", "clang")]
        if all(t.exists() for t in tools):
            return MlirTools(*(str(t) for t in tools))
    raise CodegenError("mlir-opt, mlir-translate and clang from one LLVM install were not found")


def _run(cmd: list[str], stdin: str | None = None) -> str:
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise CodegenError(f"{Path(cmd[0]).name} failed:\n{proc.stderr[-4000:]}")
    return proc.stdout


def mlir_verify(
    text: str, tools: MlirTools | None = None, allow_unregistered: bool = False
) -> None:
    """Parse and verify ``text`` with mlir-opt (raises CodegenError with the diagnostics)."""
    tools = tools or find_mlir_tools()
    cmd = [tools.mlir_opt, "--verify-diagnostics", "-o", os.devnull, "-"]
    if allow_unregistered:
        cmd.insert(1, "--allow-unregistered-dialect")
    _run(cmd, stdin=text)


def _runtime_link_flags(tools: MlirTools) -> list[str]:
    """Link the MLIR C runner utilities (``memrefCopy``) when the install ships them."""
    lib = Path(tools.mlir_opt).resolve().parent.parent / "lib"
    if any((lib / f"libmlir_c_runner_utils{suffix}").exists() for suffix in (".dylib", ".so")):
        return [f"-L{lib}", "-lmlir_c_runner_utils", f"-Wl,-rpath,{lib}"]
    return []


def _descriptor(rank: int) -> type[ctypes.Structure]:
    fields: list[tuple[str, Any]] = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
    ]
    if rank:
        fields += [("sizes", ctypes.c_int64 * rank), ("strides", ctypes.c_int64 * rank)]
    return type(f"MemRef{rank}D", (ctypes.Structure,), {"_fields_": fields})


def _strides(shape: tuple[int, ...]) -> list[int]:
    out = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        out[i] = out[i + 1] * shape[i + 1]
    return out


class MlirExecutable:
    """A module lowered to a shared library; ``run(feeds)`` returns numpy outputs."""

    _GLOBAL_SENTINEL = 0xDEADBEEF

    def __init__(
        self,
        library: Path,
        inputs: list[tuple[str, TensorType]],
        outputs: list[tuple[str, TensorType]],
    ) -> None:
        self.library = library
        self.inputs = inputs
        self.outputs = outputs
        self.lib = ctypes.CDLL(str(library))
        self.fn = self.lib._mlir_ciface_main
        self.fn.restype = None
        self._in_types = [_descriptor(len(t.shape)) for _, t in inputs]
        out_descs = [_descriptor(len(t.shape)) for _, t in outputs]
        if len(outputs) == 1:
            self._result_type: type[ctypes.Structure] = out_descs[0]
        else:
            fields = [(f"r{i}", d) for i, d in enumerate(out_descs)]
            self._result_type = type("Results", (ctypes.Structure,), {"_fields_": fields})
        self._libc = ctypes.CDLL(None)
        self._libc.free.argtypes = [ctypes.c_void_p]

    def run(self, feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        arrays, descs = [], []
        for (name, t), dtype in zip(self.inputs, self._in_types, strict=True):
            if name not in feeds:
                raise GraphError(f"missing input {name!r}")
            a = np.asarray(feeds[name], dtype=numpy_dtype(t.dtype), order="C")
            if a.shape != t.shape:
                raise GraphError(f"input {name!r}: expected {t}, got {list(a.shape)}")
            arrays.append(a)
            d = dtype()
            d.allocated = d.aligned = a.ctypes.data
            d.offset = 0
            if t.shape:
                d.sizes[:] = list(t.shape)
                d.strides[:] = _strides(t.shape)
            descs.append(d)
        result = self._result_type()
        self.fn(ctypes.byref(result), *(ctypes.byref(d) for d in descs))
        parts = (
            [result]
            if len(self.outputs) == 1
            else [getattr(result, f"r{i}") for i in range(len(self.outputs))]
        )
        inputs = {a.ctypes.data for a in arrays}
        freed: set[int] = set()
        out = {}
        for (name, t), d in zip(self.outputs, parts, strict=True):
            np_type = numpy_dtype(t.dtype)
            nbytes = t.size * np_type.itemsize
            base = int(d.aligned or 0) + int(d.offset) * np_type.itemsize
            raw = ctypes.string_at(base, nbytes) if nbytes else b""
            out[name] = np.frombuffer(raw, dtype=np_type).reshape(t.shape).copy()
            alloc = int(d.allocated or 0)
            if (
                alloc
                and alloc != self._GLOBAL_SENTINEL
                and alloc not in inputs
                and alloc not in freed
            ):
                freed.add(alloc)
        for alloc in freed:
            self._libc.free(alloc)
        return out


def compile_mlir(
    graph: Graph,
    cache_dir: str | Path | None = None,
    opt_level: str = "-O2",
    tools: MlirTools | None = None,
) -> MlirExecutable:
    """Export ``graph`` (fully lowered, with constants), lower it to LLVM and load it."""
    module = to_mlir(graph, "linalg")
    if not module.fully_lowered:
        raise CodegenError(f"operators without an MLIR lowering: {sorted(module.fallback)}")
    tools = tools or find_mlir_tools()
    key = hashlib.sha256((module.text + PIPELINE + opt_level + tools.clang).encode()).hexdigest()[
        :20
    ]
    base = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    directory = base / f"mlir-{key}"
    library = directory / f"libmlir_model{LIB_SUFFIX}"
    if not library.exists():
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.mlir").write_text(module.text)
        llvm_dialect = directory / "model.llvm.mlir"
        _run(
            [
                tools.mlir_opt,
                str(directory / "model.mlir"),
                f"--pass-pipeline={PIPELINE}",
                "-o",
                str(llvm_dialect),
            ]
        )
        _run(
            [
                tools.mlir_translate,
                "--mlir-to-llvmir",
                str(llvm_dialect),
                "-o",
                str(directory / "model.ll"),
            ]
        )
        tmp = library.with_suffix(f".tmp{os.getpid()}{LIB_SUFFIX}")
        _run(
            [
                tools.clang,
                opt_level,
                "-shared",
                "-fPIC",
                "-Wno-override-module",
                str(directory / "model.ll"),
                "-o",
                str(tmp),
                *_runtime_link_flags(tools),
                "-lm",
            ]
        )
        tmp.replace(library)
    inputs = [(v.name, v.tensor_type) for v in graph.inputs]
    outputs = [(n, v.tensor_type) for n, v in graph.named_outputs()]
    return MlirExecutable(library, inputs, outputs)
