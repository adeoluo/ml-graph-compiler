"""Print a graph as MLIR text (see the package docstring for the two forms)."""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ml_graph_compiler import ops
from ml_graph_compiler.errors import CodegenError
from ml_graph_compiler.ir import Graph, Node, TensorType, Value, numpy_dtype

ELEMENT_TYPES = {
    "float32": "f32",
    "float64": "f64",
    "float16": "f16",
    "int64": "i64",
    "int32": "i32",
    "int8": "i8",
    "uint8": "i8",
    "bool": "i1",
}
FLOATS = frozenset({"float32", "float64", "float16"})
INTS = frozenset({"int64", "int32", "int8", "uint8"})

# Constants with at most this many elements are printed inline; larger ones go to resources.
INLINE_LIMIT = 16


def element_type(dtype: str) -> str:
    try:
        return ELEMENT_TYPES[dtype]
    except KeyError as exc:
        raise CodegenError(f"no MLIR element type for {dtype}") from exc


def tensor_type(t: TensorType) -> str:
    dims = "".join(f"{d}x" for d in t.shape)
    return f"tensor<{dims}{element_type(t.dtype)}>"


def float_literal(value: float, dtype: str = "float32") -> str:
    """A float attribute literal; non-finite values use the hexadecimal bit pattern."""
    if math.isfinite(value):
        return f"{value:.9e}"
    if dtype == "float64":
        return "0x" + struct.pack(">d", value).hex().upper()
    return "0x" + struct.pack(">f", value).hex().upper()


def scalar_literal(value: Any, dtype: str) -> str:
    if dtype == "bool":
        return "true" if bool(value) else "false"
    if dtype in FLOATS:
        return float_literal(float(value), dtype)
    return str(int(value))


def _nested(array: np.ndarray, dtype: str) -> str:
    if array.ndim == 0:
        return scalar_literal(array.item(), dtype)
    return "[" + ", ".join(_nested(a, dtype) for a in array) + "]"


def dense_literal(array: np.ndarray, dtype: str) -> str:
    """``dense<...>``: a splat when every element is equal, nested lists otherwise."""
    flat = array.reshape(-1)
    if flat.size and np.all(flat == flat[0]):
        return f"dense<{scalar_literal(flat[0].item(), dtype)}>"
    return f"dense<{_nested(array, dtype)}>"


def _ident(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def attribute(value: Any) -> str:
    """An MLIR attribute for an IR attribute value."""
    if isinstance(value, bool | np.bool_):
        return "true" if value else "false"
    if isinstance(value, int | np.integer):
        return f"{int(value)} : i64"
    if isinstance(value, float | np.floating):
        return f"{float_literal(float(value))} : f32"
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, np.ndarray):
        dtype = value.dtype.name if value.dtype.name in ELEMENT_TYPES else "float32"
        t = TensorType(dtype, tuple(int(d) for d in value.shape))
        return f"{dense_literal(value.astype(dtype), dtype)} : {tensor_type(t)}"
    if isinstance(value, dict):
        items = ", ".join(f"{_ident(str(k))} = {attribute(v)}" for k, v in value.items())
        return "{" + items + "}"
    if isinstance(value, list | tuple):
        if all(isinstance(v, int | np.integer) and not isinstance(v, bool) for v in value):
            return (
                "array<i64" + (": " + ", ".join(str(int(v)) for v in value) if value else "") + ">"
            )
        if all(isinstance(v, float | np.floating) for v in value):
            return "array<f32: " + ", ".join(float_literal(float(v)) for v in value) + ">"
        return "[" + ", ".join(attribute(v) for v in value) + "]"
    return _string(str(value))


def snake(op: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", op).lower()


def broadcast_map(shape: Sequence[int], out_shape: Sequence[int]) -> list[str]:
    """Affine results that read a numpy-broadcast operand inside a loop nest over ``out_shape``."""
    rank = len(out_shape)
    results = []
    for j, d in enumerate(shape):
        i = j + rank - len(shape)
        results.append("0" if d == 1 and out_shape[i] != 1 else f"d{i}")
    return results


def term(dim: str, factor: int) -> str:
    return dim if factor == 1 else f"{dim} * {factor}"


def affine(rank: int, results: Sequence[str]) -> str:
    dims = ", ".join(f"d{i}" for i in range(rank))
    return f"affine_map<({dims}) -> ({', '.join(results)})>"


@dataclass
class Operand:
    """An SSA tensor and the affine results that index it inside a generic loop nest."""

    name: str
    type: TensorType
    results: list[str]


@dataclass
class MlirModule:
    """Printed module text plus statistics about the lowering."""

    text: str
    lowered: dict[str, int] = field(default_factory=dict)
    fallback: dict[str, int] = field(default_factory=dict)
    resources: int = 0
    resource_bytes: int = 0

    @property
    def fully_lowered(self) -> bool:
        return not self.fallback

    def __str__(self) -> str:
        return self.text


Body = Callable[["Emitter", list[str]], list[str]]


class Emitter:
    """Emits one ``func.func @main`` for a graph."""

    def __init__(self, graph: Graph, lower: bool, elide_constants: bool) -> None:
        self.graph = graph
        self.lower = lower
        self.elide = elide_constants
        self.lines: list[str] = []
        self.names: dict[int, str] = {}
        self.resources: list[tuple[str, str]] = []
        self.resource_bytes = 0
        self.counter = 0
        self.indent = 2
        self.lowered: dict[str, int] = {}
        self.fallback: dict[str, int] = {}

    # -- basics ---------------------------------------------------------------------------
    def fresh(self, hint: str = "v") -> str:
        self.counter += 1
        return f"%{hint}{self.counter}"

    def emit(self, line: str) -> None:
        self.lines.append(" " * self.indent + line)

    def value(self, v: Value) -> str:
        name = self.names.get(id(v))
        if name is None:
            if v.data is None:
                raise CodegenError(f"value {v.name!r} is used before it is defined")
            name = self.constant(v.data, v.tensor_type)
            self.names[id(v)] = name
        return name

    def constant(self, data: np.ndarray, t: TensorType) -> str:
        name = self.fresh("cst")
        data = np.asarray(data, dtype=numpy_dtype(t.dtype))
        if data.size <= INLINE_LIMIT or t.dtype == "bool" or np.all(data == data.reshape(-1)[0]):
            literal = dense_literal(data, t.dtype)
        elif self.elide:
            literal = "dense_resource<__elided__>"
        else:
            key = f"mgc_blob_{len(self.resources)}"
            align = max(data.dtype.itemsize, 4)
            blob = struct.pack("<I", align).hex() + np.ascontiguousarray(data).tobytes().hex()
            self.resources.append((key, "0x" + blob.upper()))
            self.resource_bytes += data.nbytes
            literal = f"dense_resource<{key}>"
        self.emit(f"{name} = arith.constant {literal} : {tensor_type(t)}")
        return name

    def scalar(self, value: float | int | bool, dtype: str) -> str:
        name = self.fresh("s")
        if dtype == "bool":
            self.emit(f"{name} = arith.constant {scalar_literal(value, dtype)}")
        else:
            self.emit(
                f"{name} = arith.constant {scalar_literal(value, dtype)} : {element_type(dtype)}"
            )
        return name

    def empty(self, t: TensorType) -> str:
        name = self.fresh("e")
        self.emit(f"{name} = tensor.empty() : {tensor_type(t)}")
        return name

    def filled(self, t: TensorType, value: float | int | bool) -> str:
        init = self.empty(t)
        s = self.scalar(value, t.dtype)
        name = self.fresh("f")
        self.emit(
            f"{name} = linalg.fill ins({s} : {element_type(t.dtype)}) "
            f"outs({init} : {tensor_type(t)}) -> {tensor_type(t)}"
        )
        return name

    def count(self, table: dict[str, int], op: str) -> None:
        table[op] = table.get(op, 0) + 1

    # -- linalg.generic -------------------------------------------------------------------
    def generic(
        self,
        rank: int,
        iterators: Sequence[str],
        inputs: Sequence[Operand],
        outputs: Sequence[tuple[TensorType, list[str], str | None]],
        body: Body,
        init: Sequence[float | int | bool | None] | None = None,
    ) -> list[str]:
        """Emit ``linalg.generic``; outputs are (type, affine results, init tensor or None).

        Outputs without an init tensor start from ``tensor.empty`` or, when ``init`` gives a
        value, from ``linalg.fill``. ``body`` receives the block arguments (inputs then outputs)
        and returns the yielded scalars.
        """
        outs = []
        for i, (t, _, given) in enumerate(outputs):
            if given is not None:
                outs.append(given)
            elif init is not None and init[i] is not None:
                fill = init[i]
                assert fill is not None
                outs.append(self.filled(t, fill))
            else:
                outs.append(self.empty(t))
        maps = [affine(rank, op.results) for op in inputs]
        maps += [affine(rank, results) for _, results, _ in outputs]
        its = ", ".join(f'"{it}"' for it in iterators)
        ins = ", ".join(op.name for op in inputs)
        in_types = ", ".join(tensor_type(op.type) for op in inputs)
        out_types = ", ".join(tensor_type(t) for t, _, _ in outputs)
        results = [self.fresh("g") for _ in outputs]
        head = (
            f"{', '.join(results) if len(results) == 1 else results[0] + ':' + str(len(results))}"
        )
        if len(results) > 1:
            base = results[0]
            results = [f"{base}#{i}" for i in range(len(outputs))]
        ins_part = f"ins({ins} : {in_types}) " if inputs else ""
        self.emit(
            f"{head} = linalg.generic {{indexing_maps = [{', '.join(maps)}], "
            f"iterator_types = [{its}]}} {ins_part}outs({', '.join(outs)} : {out_types}) {{"
        )
        args = [self.fresh("a") for _ in (*inputs, *outputs)]
        arg_types = [element_type(op.type.dtype) for op in inputs]
        arg_types += [element_type(t.dtype) for t, _, _ in outputs]
        sig = ", ".join(f"{a}: {t}" for a, t in zip(args, arg_types, strict=True))
        self.emit(f"^bb0({sig}):")
        self.indent += 2
        yields = body(self, args)
        ytypes = ", ".join(element_type(t.dtype) for t, _, _ in outputs)
        self.emit(f"linalg.yield {', '.join(yields)} : {ytypes}")
        self.indent -= 2
        self.emit(f"}} -> {out_types if len(outputs) == 1 else '(' + out_types + ')'}")
        return results

    def op(self, text: str, dtype: str) -> str:
        """``%r = <text> : <element type>`` inside a region."""
        name = self.fresh("x")
        self.emit(f"{name} = {text} : {element_type(dtype)}")
        return name

    # -- scalar expressions ---------------------------------------------------------------
    def fconst(self, value: float, dtype: str) -> str:
        return self.scalar(value, dtype)

    def activation(self, x: str, attrs: dict[str, Any], dtype: str) -> str:
        """The fused activation of FusedConv and FusedGemm on one scalar."""
        act = str(attrs.get("activation", "none"))
        if act == "none":
            return x
        if act == "relu":
            return self.op(f"arith.maximumf {x}, {self.fconst(0.0, dtype)}", dtype)
        if act == "clip":
            lo = self.fconst(float(attrs.get("clip_min", 0.0)), dtype)
            hi = self.fconst(float(attrs.get("clip_max", 6.0)), dtype)
            return self.op(
                f"arith.minimumf {self.op(f'arith.maximumf {x}, {lo}', dtype)}, {hi}", dtype
            )
        if act in ("gelu", "gelu_tanh"):
            return self.expr(
                "Gelu",
                {"approximate": "tanh" if act == "gelu_tanh" else "none"},
                [x],
                [dtype],
                dtype,
            )
        if act == "tanh":
            return self.op(f"math.tanh {x}", dtype)
        if act == "sigmoid":
            return self.expr("Sigmoid", {}, [x], [dtype], dtype)
        raise CodegenError(f"activation {act!r} has no MLIR lowering")

    def expr(
        self,
        op: str,
        attrs: dict[str, Any],
        args: list[str | None],
        dtypes: list[str],
        out_dtype: str,
    ) -> str:
        """Emit the scalar computation of one elementwise operator; returns the SSA result."""
        dt = dtypes[0] if dtypes else out_dtype
        a = args[0] if args else None
        if op == "Identity":
            assert a is not None
            return a
        if op == "Cast":
            assert a is not None
            return self.cast(a, dt, out_dtype)
        if op == "Where":
            c, x, y = args
            return self.op(f"arith.select {c}, {x}, {y}", out_dtype)
        if op in ("And", "Or", "Xor"):
            return self.op(f"arith.{op.lower()}i {a}, {args[1]}", "bool")
        if op == "Not":
            return self.op(f"arith.xori {a}, {self.scalar(True, 'bool')}", "bool")
        if op in _CMP:
            fpred, ipred = _CMP[op]
            if dt in FLOATS:
                return self._cmp("cmpf", fpred, a, args[1], dt)
            return self._cmp("cmpi", ipred, a, args[1], dt)
        if dt in FLOATS:
            return self.float_expr(op, attrs, args, dt)
        if dt in INTS:
            return self.int_expr(op, attrs, args, dt)
        raise CodegenError(f"{op} on {dt} has no MLIR lowering")

    def _cmp(self, kind: str, pred: str, a: str | None, b: str | None, dtype: str) -> str:
        name = self.fresh("x")
        self.emit(f"{name} = arith.{kind} {pred}, {a}, {b} : {element_type(dtype)}")
        return name

    def cast(self, a: str, src: str, dst: str) -> str:
        if src == dst or (src in INTS and dst in INTS and element_type(src) == element_type(dst)):
            return a
        s, d = element_type(src), element_type(dst)
        name = self.fresh("x")
        if dst == "bool":
            zero = self.scalar(0, src)
            kind = "cmpf une" if src in FLOATS else "cmpi ne"
            self.emit(f"{name} = arith.{kind.split()[0]} {kind.split()[1]}, {a}, {zero} : {s}")
            return name
        if src == "bool":
            text = f"arith.uitofp {a}" if dst in FLOATS else f"arith.extui {a}"
        elif src in FLOATS and dst in FLOATS:
            wider = numpy_dtype(dst).itemsize > numpy_dtype(src).itemsize
            text = f"arith.{'extf' if wider else 'truncf'} {a}"
        elif src in FLOATS:
            text = f"arith.fptosi {a}"
        elif dst in FLOATS:
            text = f"arith.sitofp {a}"
        else:
            wider = numpy_dtype(dst).itemsize > numpy_dtype(src).itemsize
            text = f"arith.{'extsi' if wider else 'trunci'} {a}"
        self.emit(f"{name} = {text} : {s} to {d}")
        return name

    def float_expr(self, op: str, attrs: dict[str, Any], args: list[str | None], dt: str) -> str:
        a, b = args[0], args[1] if len(args) > 1 else None
        if op in _FBIN:
            return self.op(f"arith.{_FBIN[op]} {a}, {b}", dt)
        if op == "Pow":
            return self.op(f"math.powf {a}, {b}", dt)
        if op in _FUN:
            return self.op(f"{_FUN[op]} {a}", dt)
        if op == "Relu":
            return self.op(f"arith.maximumf {a}, {self.fconst(0.0, dt)}", dt)
        if op == "Reciprocal":
            return self.op(f"arith.divf {self.fconst(1.0, dt)}, {a}", dt)
        if op == "Sigmoid":
            neg = self.op(f"arith.negf {a}", dt)
            den = self.op(
                f"arith.addf {self.fconst(1.0, dt)}, {self.op(f'math.exp {neg}', dt)}", dt
            )
            return self.op(f"arith.divf {self.fconst(1.0, dt)}, {den}", dt)
        if op == "Softplus":
            # max(x, 0) + log1p(exp(-|x|)) avoids overflow for large x
            pos = self.op(f"arith.maximumf {a}, {self.fconst(0.0, dt)}", dt)
            nabs = self.op(f"arith.negf {self.op(f'math.absf {a}', dt)}", dt)
            tail = self.op(f"math.log1p {self.op(f'math.exp {nabs}', dt)}", dt)
            return self.op(f"arith.addf {pos}, {tail}", dt)
        if op == "LeakyRelu":
            alpha = self.fconst(float(attrs.get("alpha", 0.01)), dt)
            scaled = self.op(f"arith.mulf {a}, {alpha}", dt)
            ge = self._cmp("cmpf", "oge", a, self.fconst(0.0, dt), dt)
            return self.op(f"arith.select {ge}, {a}, {scaled}", dt)
        if op == "HardSigmoid":
            alpha = self.fconst(float(attrs.get("alpha", 0.2)), dt)
            beta = self.fconst(float(attrs.get("beta", 0.5)), dt)
            y = self.op(f"arith.addf {self.op(f'arith.mulf {a}, {alpha}', dt)}, {beta}", dt)
            y = self.op(f"arith.maximumf {y}, {self.fconst(0.0, dt)}", dt)
            return self.op(f"arith.minimumf {y}, {self.fconst(1.0, dt)}", dt)
        if op == "Gelu":
            half_x = self.op(f"arith.mulf {a}, {self.fconst(0.5, dt)}", dt)
            if attrs.get("approximate", "none") == "tanh":
                x3 = self.op(f"arith.mulf {self.op(f'arith.mulf {a}, {a}', dt)}, {a}", dt)
                cubic = self.op(f"arith.mulf {x3}, {self.fconst(0.044715, dt)}", dt)
                inner = self.op(f"arith.addf {a}, {cubic}", dt)
                k = self.fconst(math.sqrt(2 / math.pi), dt)
                t = self.op(f"math.tanh {self.op(f'arith.mulf {inner}, {k}', dt)}", dt)
            else:
                k = self.fconst(1 / math.sqrt(2), dt)
                t = self.op(f"math.erf {self.op(f'arith.mulf {a}, {k}', dt)}", dt)
            return self.op(
                f"arith.mulf {half_x}, {self.op(f'arith.addf {t}, {self.fconst(1.0, dt)}', dt)}", dt
            )
        if op == "Clip":
            assert a is not None
            y = a
            lo: str | None = args[1] if len(args) > 1 and args[1] else None
            hi: str | None = args[2] if len(args) > 2 and args[2] else None
            if lo is None and "min" in attrs:
                lo = self.fconst(float(attrs["min"]), dt)
            if hi is None and "max" in attrs:
                hi = self.fconst(float(attrs["max"]), dt)
            if lo is not None:
                y = self.op(f"arith.maximumf {y}, {lo}", dt)
            if hi is not None:
                y = self.op(f"arith.minimumf {y}, {hi}", dt)
            assert y is not None
            return y
        raise CodegenError(f"{op} on {dt} has no MLIR lowering")

    def int_expr(self, op: str, attrs: dict[str, Any], args: list[str | None], dt: str) -> str:
        a, b = args[0], args[1] if len(args) > 1 else None
        if op in _IBIN:
            return self.op(f"arith.{_IBIN[op]} {a}, {b}", dt)
        if op == "Neg":
            return self.op(f"arith.subi {self.scalar(0, dt)}, {a}", dt)
        if op == "Abs":
            return self.op(f"math.absi {a}", dt)
        if op == "Clip":
            y = a
            if len(args) > 1 and args[1]:
                y = self.op(f"arith.maxsi {y}, {args[1]}", dt)
            if len(args) > 2 and args[2]:
                y = self.op(f"arith.minsi {y}, {args[2]}", dt)
            assert y is not None
            return y
        raise CodegenError(f"{op} on {dt} has no MLIR lowering")

    # -- nodes ----------------------------------------------------------------------------
    def define(self, value: Value, name: str) -> None:
        self.names[id(value)] = name

    def node(self, node: Node) -> None:
        self.emit(f"// {node.op} {node.name}")
        if not self.lower:
            self.opaque(node)
            return
        present = [v for v in node.inputs if v is not None]
        if node.op in ("Shape", "Size") or (present and all(v.is_constant for v in present)):
            self.fold(node)
            return
        handler = getattr(self, f"lower_{node.op}", None)
        saved = (len(self.lines), dict(self.names), len(self.resources), self.resource_bytes)
        try:
            if handler is not None:
                handler(node)
            elif ops.is_elementwise(node.op):
                self.lower_elementwise(node)
            else:
                raise CodegenError(f"{node.op} has no MLIR lowering")
            self.count(self.lowered, node.op)
        except CodegenError:
            del self.lines[saved[0] :]
            self.names = saved[1]
            del self.resources[saved[2] :]
            self.resource_bytes = saved[3]
            self.opaque(node)

    def fold(self, node: Node) -> None:
        """Nodes whose inputs are all constants (or that only read shapes) become constants."""
        arrays: list[np.ndarray | None] = []
        for v in node.inputs:
            if v is None:
                arrays.append(None)
            elif v.data is not None:
                arrays.append(v.data)
            else:
                arrays.append(np.zeros(v.tensor_type.shape, numpy_dtype(v.tensor_type.dtype)))
        for out, data in zip(node.outputs, ops.run(node, arrays), strict=False):
            self.define(out, self.constant(np.asarray(data), out.tensor_type))
        self.count(self.lowered, node.op)

    def opaque(self, node: Node) -> None:
        """A generic ``"mgc.<op>"`` operation carrying the node's attributes."""
        operands = [v for v in node.inputs if v is not None]
        names = [self.value(v) for v in operands]
        attrs = dict(node.attrs)
        if any(v is None for v in node.inputs):
            attrs["operand_present"] = [int(v is not None) for v in node.inputs]
        attr_text = ""
        if attrs:
            attr_text = (
                " {" + ", ".join(f"{_ident(k)} = {attribute(v)}" for k, v in attrs.items()) + "}"
            )
        in_types = ", ".join(tensor_type(v.tensor_type) for v in operands)
        out_types = [tensor_type(o.tensor_type) for o in node.outputs]
        result = self.fresh("op")
        head = result if len(node.outputs) == 1 else f"{result}:{len(node.outputs)}"
        outs = out_types[0] if len(out_types) == 1 else "(" + ", ".join(out_types) + ")"
        operands_text = ", ".join(names)
        self.emit(
            f'{head} = "mgc.{snake(node.op)}"({operands_text}){attr_text} : ({in_types}) -> {outs}'
        )
        for i, out in enumerate(node.outputs):
            self.define(out, result if len(node.outputs) == 1 else f"{result}#{i}")
        if self.lower:
            self.count(self.fallback, node.op)

    def operand(self, v: Value, out_shape: Sequence[int]) -> Operand:
        t = v.tensor_type
        return Operand(self.value(v), t, broadcast_map(t.shape, out_shape))

    # -- elementwise ----------------------------------------------------------------------
    def lower_elementwise(self, node: Node) -> None:
        program = [(node.op, dict(node.attrs), list(range(len(node.inputs))))]
        self.elementwise_program(node, program)

    def lower_FusedElementwise(self, node: Node) -> None:
        program = [
            (str(op), dict(a), [int(x) for x in args]) for op, a, args in node.attrs["program"]
        ]
        self.elementwise_program(node, program)

    def elementwise_program(
        self, node: Node, program: list[tuple[str, dict[str, Any], list[int]]]
    ) -> None:
        out_t = node.output.tensor_type
        leaves = list(node.inputs)
        inputs = [self.operand(v, out_t.shape) for v in leaves if v is not None]
        slot = {}
        k = 0
        for i, v in enumerate(leaves):
            if v is not None:
                slot[i] = k
                k += 1
        leaf_types = [v.tensor_type.dtype if v is not None else "" for v in leaves]

        def body(em: Emitter, args: list[str]) -> list[str]:
            values: list[str | None] = [
                args[slot[i]] if i in slot else None for i in range(len(leaves))
            ]
            dtypes = list(leaf_types)
            for op, attrs, refs in program:
                sub = Node(op, [], [Value("_")], dict(attrs))
                ts = [TensorType(dtypes[r], ()) if r >= 0 and dtypes[r] else None for r in refs]
                out_dtype = ops.spec(op).infer(sub, ts, [None] * len(refs))[0].dtype
                result = em.expr(
                    op,
                    attrs,
                    [values[r] if r >= 0 else None for r in refs],
                    [dtypes[r] for r in refs if r >= 0],
                    out_dtype,
                )
                values.append(result)
                dtypes.append(out_dtype)
            last = values[-1]
            assert last is not None
            return [last]

        rank = len(out_t.shape)
        identity = [f"d{i}" for i in range(rank)]
        (result,) = self.generic(rank, ["parallel"] * rank, inputs, [(out_t, identity, None)], body)
        self.define(node.output, result)

    def lower_Dropout(self, node: Node) -> None:
        self.define(node.output, self.value(node.input(0)))
        if len(node.outputs) > 1:
            self.define(node.outputs[1], self.filled(node.outputs[1].tensor_type, True))

    def lower_BatchNormalization(self, node: Node) -> None:
        x = node.input(0)
        t = x.tensor_type
        rank = t.rank
        ch = rank - 1 if node.attrs.get("layout") == "NHWC" else 1
        eps = float(node.attrs.get("epsilon", 1e-5))
        inputs = [self.operand(x, t.shape)]
        inputs += [
            Operand(self.value(node.input(i)), node.input(i).tensor_type, [f"d{ch}"])
            for i in range(1, 5)
        ]
        dt = t.dtype

        def body(em: Emitter, a: list[str]) -> list[str]:
            xv, scale, bias, mean, var = a[:5]
            inv = em.op(f"math.rsqrt {em.op(f'arith.addf {var}, {em.fconst(eps, dt)}', dt)}", dt)
            centered = em.op(f"arith.subf {xv}, {mean}", dt)
            y = em.op(f"arith.mulf {centered}, {em.op(f'arith.mulf {inv}, {scale}', dt)}", dt)
            return [em.op(f"arith.addf {y}, {bias}", dt)]

        identity = [f"d{i}" for i in range(rank)]
        (result,) = self.generic(rank, ["parallel"] * rank, inputs, [(t, identity, None)], body)
        self.define(node.output, result)

    # -- matrix products ------------------------------------------------------------------
    def contraction(
        self, a: Value, b: Value, out_t: TensorType, trans_a: bool = False, trans_b: bool = False
    ) -> str:
        """``A @ B`` with numpy batch broadcasting as one ``linalg.generic`` reduction over K."""
        at, bt = a.tensor_type, b.tensor_type
        has_m, has_n = at.rank >= 2, bt.rank >= 2
        batch = out_t.rank - int(has_m) - int(has_n)
        dm, dn, dk = batch, batch + int(has_m), batch + int(has_m) + int(has_n)
        rank = dk + 1
        out_batch = out_t.shape[:batch]

        def batch_results(t: TensorType) -> list[str]:
            return broadcast_map(t.shape[:-2], out_batch) if t.rank > 2 else []

        if has_m:
            a_res = batch_results(at) + ([f"d{dk}", f"d{dm}"] if trans_a else [f"d{dm}", f"d{dk}"])
        else:
            a_res = [f"d{dk}"]
        if has_n:
            b_res = batch_results(bt) + ([f"d{dn}", f"d{dk}"] if trans_b else [f"d{dk}", f"d{dn}"])
        else:
            b_res = [f"d{dk}"]
        dt = out_t.dtype
        floating = dt in FLOATS

        def body(em: Emitter, args: list[str]) -> list[str]:
            x, y, acc = args
            prod = em.op(f"arith.{'mulf' if floating else 'muli'} {x}, {y}", dt)
            return [em.op(f"arith.{'addf' if floating else 'addi'} {acc}, {prod}", dt)]

        out_res = [f"d{i}" for i in range(dk)]
        iterators = ["parallel"] * dk + ["reduction"]
        (result,) = self.generic(
            rank,
            iterators,
            [Operand(self.value(a), at, a_res), Operand(self.value(b), bt, b_res)],
            [(out_t, out_res, None)],
            body,
            init=[0],
        )
        return result

    def epilogue(
        self,
        acc: str,
        out_t: TensorType,
        extras: list[Operand],
        combine: Callable[[Emitter, str, list[str]], str],
        attrs: dict[str, Any],
    ) -> str:
        """One elementwise pass after a reduction: bias, residual, scaling and activation."""
        rank = out_t.rank
        identity = [f"d{i}" for i in range(rank)]

        def body(em: Emitter, args: list[str]) -> list[str]:
            y = combine(em, args[0], args[1 : len(extras) + 1])
            return [em.activation(y, attrs, out_t.dtype)]

        (result,) = self.generic(
            rank,
            ["parallel"] * rank,
            [Operand(acc, out_t, identity), *extras],
            [(out_t, identity, None)],
            body,
        )
        return result

    def lower_MatMul(self, node: Node) -> None:
        self.define(
            node.output, self.contraction(node.input(0), node.input(1), node.output.tensor_type)
        )

    def lower_Gemm(self, node: Node) -> None:
        out_t = node.output.tensor_type
        acc = self.contraction(
            node.input(0),
            node.input(1),
            out_t,
            bool(int(node.attrs.get("transA", 0))),
            bool(int(node.attrs.get("transB", 0))),
        )
        alpha, beta = float(node.attrs.get("alpha", 1.0)), float(node.attrs.get("beta", 1.0))
        c = node.optional_input(2)
        extras = [self.operand(c, out_t.shape)] if c is not None else []
        dt = out_t.dtype

        def combine(em: Emitter, y: str, rest: list[str]) -> str:
            if alpha != 1.0:
                y = em.op(f"arith.mulf {y}, {em.fconst(alpha, dt)}", dt)
            if rest:
                cv = rest[0]
                if beta != 1.0:
                    cv = em.op(f"arith.mulf {cv}, {em.fconst(beta, dt)}", dt)
                y = em.op(f"arith.addf {y}, {cv}", dt)
            return y

        if alpha == 1.0 and not extras:
            self.define(node.output, acc)
            return
        self.define(node.output, self.epilogue(acc, out_t, extras, combine, {}))

    def _add_all(self, em: Emitter, y: str, rest: list[str], dt: str) -> str:
        for r in rest:
            y = em.op(f"arith.addf {y}, {r}", dt)
        return y

    def lower_FusedGemm(self, node: Node) -> None:
        out_t = node.output.tensor_type
        acc = self.contraction(
            node.input(0), node.input(1), out_t, False, bool(int(node.attrs.get("trans_b", 0)))
        )
        extras = [self.operand(v, out_t.shape) for v in node.inputs[2:4] if v is not None]
        self.finish_fused(node, acc, out_t, extras)

    def finish_fused(self, node: Node, acc: str, out_t: TensorType, extras: list[Operand]) -> None:
        if not extras and node.attrs.get("activation", "none") == "none":
            self.define(node.output, acc)
            return
        dt = out_t.dtype
        result = self.epilogue(
            acc, out_t, extras, lambda em, y, rest: self._add_all(em, y, rest, dt), dict(node.attrs)
        )
        self.define(node.output, result)

    # -- convolution and pooling ----------------------------------------------------------
    def pad(
        self,
        name: str,
        t: TensorType,
        low: Sequence[int],
        high: Sequence[int],
        value: float | int | bool,
    ) -> tuple[str, TensorType]:
        if not any(low) and not any(high):
            return name, t
        if min(*low, *high) < 0:
            raise CodegenError("negative padding has no MLIR lowering")
        shape = tuple(d + lo + hi for d, lo, hi in zip(t.shape, low, high, strict=True))
        out_t = TensorType(t.dtype, shape)
        result = self.fresh("pad")
        idx = ", ".join(f"{self.fresh('i')}: index" for _ in shape)
        self.emit(
            f"{result} = tensor.pad {name} low[{', '.join(map(str, low))}] "
            f"high[{', '.join(map(str, high))}] {{"
        )
        self.emit(f"^bb0({idx}):")
        self.indent += 2
        s = self.scalar(value, t.dtype)
        self.emit(f"tensor.yield {s} : {element_type(t.dtype)}")
        self.indent -= 2
        self.emit(f"}} : {tensor_type(t)} to {tensor_type(out_t)}")
        return result, out_t

    def lower_Conv(self, node: Node) -> None:
        self.lower_FusedConv(node)

    def lower_FusedConv(self, node: Node) -> None:
        x, w = node.input(0), node.input(1)
        xt, wt, out_t = x.tensor_type, w.tensor_type, node.output.tensor_type
        nhwc = ops.conv_layout(node) == "NHWC"
        group = int(node.attrs.get("group", 1))
        if nhwc:
            kh, kw, cg, k = wt.shape
            hw = (xt.shape[1], xt.shape[2])
        else:
            k, cg, kh, kw = wt.shape
            hw = (xt.shape[2], xt.shape[3])
        g = ops.window_geometry(node, hw, (kh, kw))
        pt, pl, pb, pr = g.pads
        low = (0, pt, pl, 0) if nhwc else (0, 0, pt, pl)
        high = (0, pb, pr, 0) if nhwc else (0, 0, pb, pr)
        xp, xpt = self.pad(self.value(x), xt, low, high, 0.0)
        sh, sw = g.strides
        dh, dw = g.dilations
        kg = k // group
        if nhwc:  # loops (n, oh, ow, oc, kh, kw, ic)
            chan = "d6" if group == 1 else f"(d3 floordiv {kg}) * {cg} + d6"
            x_res = [
                "d0",
                f"{term('d1', sh)} + {term('d4', dh)}",
                f"{term('d2', sw)} + {term('d5', dw)}",
                chan,
            ]
            w_res = ["d4", "d5", "d6", "d3"]
        else:  # loops (n, oc, oh, ow, ic, kh, kw)
            chan = "d4" if group == 1 else f"(d1 floordiv {kg}) * {cg} + d4"
            x_res = [
                "d0",
                chan,
                f"{term('d2', sh)} + {term('d5', dh)}",
                f"{term('d3', sw)} + {term('d6', dw)}",
            ]
            w_res = ["d1", "d4", "d5", "d6"]
        dt = out_t.dtype

        def body(em: Emitter, args: list[str]) -> list[str]:
            xv, wv, acc = args
            return [em.op(f"arith.addf {acc}, {em.op(f'arith.mulf {xv}, {wv}', dt)}", dt)]

        iterators = ["parallel"] * 4 + ["reduction"] * 3
        (acc,) = self.generic(
            7,
            iterators,
            [Operand(xp, xpt, x_res), Operand(self.value(w), wt, w_res)],
            [(out_t, ["d0", "d1", "d2", "d3"], None)],
            body,
            init=[0.0],
        )
        extras = []
        bias = node.optional_input(2)
        if bias is not None:
            extras.append(Operand(self.value(bias), bias.tensor_type, ["d3" if nhwc else "d1"]))
        residual = node.optional_input(3)
        if residual is not None:
            extras.append(self.operand(residual, out_t.shape))
        self.finish_fused(node, acc, out_t, extras)

    def pool_divisor(self, node: Node, hw: tuple[int, int], g: ops.Conv2dGeometry) -> np.ndarray:
        """Per output position element counts for AveragePool (ONNX count_include_pad rules)."""
        pt, pl, pb, pr = g.pads
        kh, kw = g.kernel
        sh, sw = g.strides
        dh, dw = g.dilations
        oh, ow = g.out_hw
        h, w = hw
        include = bool(int(node.attrs.get("count_include_pad", 0)))
        lo_h, hi_h = (-pt, h + pb) if include else (0, h)
        lo_w, hi_w = (-pl, w + pr) if include else (0, w)
        rows = np.arange(oh)[:, None] * sh - pt + np.arange(kh)[None, :] * dh
        cols = np.arange(ow)[:, None] * sw - pl + np.arange(kw)[None, :] * dw
        rc = ((rows >= lo_h) & (rows < hi_h)).sum(axis=1)
        cc = ((cols >= lo_w) & (cols < hi_w)).sum(axis=1)
        return np.asarray(rc[:, None] * cc[None, :], dtype=np.float32)

    def _pool(self, node: Node, kind: str) -> None:
        x = node.input(0)
        xt, out_t = x.tensor_type, node.output.tensor_type
        nhwc = ops.conv_layout(node) == "NHWC"
        hw = (xt.shape[1], xt.shape[2]) if nhwc else (xt.shape[2], xt.shape[3])
        kernel = tuple(int(k) for k in node.attrs["kernel_shape"])
        g = ops.window_geometry(
            node, hw, (kernel[0], kernel[1]), bool(node.attrs.get("ceil_mode", 0))
        )
        pt, pl, pb, pr = g.pads
        kh, kw = g.kernel
        sh, sw = g.strides
        dh, dw = g.dilations
        oh, ow = g.out_hw
        eb = max((oh - 1) * sh + (kh - 1) * dh + 1 - (hw[0] + pt + pb), 0)
        er = max((ow - 1) * sw + (kw - 1) * dw + 1 - (hw[1] + pl + pr), 0)
        low = (0, pt, pl, 0) if nhwc else (0, 0, pt, pl)
        high = (0, pb + eb, pr + er, 0) if nhwc else (0, 0, pb + eb, pr + er)
        fill = -math.inf if kind == "max" else 0.0
        xp, xpt = self.pad(self.value(x), xt, low, high, fill)
        window = self.empty(TensorType(xt.dtype, (kh, kw)))
        if nhwc:  # loops (n, oh, ow, c, kh, kw)
            x_res = [
                "d0",
                f"{term('d1', sh)} + {term('d4', dh)}",
                f"{term('d2', sw)} + {term('d5', dw)}",
                "d3",
            ]
        else:  # loops (n, c, oh, ow, kh, kw)
            x_res = [
                "d0",
                "d1",
                f"{term('d2', sh)} + {term('d4', dh)}",
                f"{term('d3', sw)} + {term('d5', dw)}",
            ]
        dt = out_t.dtype

        def body(em: Emitter, args: list[str]) -> list[str]:
            xv, _, acc = args
            return [em.op(f"arith.{'maximumf' if kind == 'max' else 'addf'} {acc}, {xv}", dt)]

        iterators = ["parallel"] * 4 + ["reduction"] * 2
        (acc,) = self.generic(
            6,
            iterators,
            [
                Operand(xp, xpt, x_res),
                Operand(window, TensorType(xt.dtype, (kh, kw)), ["d4", "d5"]),
            ],
            [(out_t, ["d0", "d1", "d2", "d3"], None)],
            body,
            init=[fill],
        )
        if kind == "max":
            self.define(node.output, acc)
            return
        div = self.pool_divisor(node, hw, g)
        div_t = TensorType("float32", div.shape)
        div_op = Operand(self.constant(div, div_t), div_t, ["d1", "d2"] if nhwc else ["d2", "d3"])
        result = self.epilogue(
            acc, out_t, [div_op], lambda em, y, rest: em.op(f"arith.divf {y}, {rest[0]}", dt), {}
        )
        self.define(node.output, result)

    def lower_MaxPool(self, node: Node) -> None:
        self._pool(node, "max")

    def lower_AveragePool(self, node: Node) -> None:
        self._pool(node, "avg")

    # -- reductions and normalization -----------------------------------------------------
    def reduce(
        self,
        name: str,
        t: TensorType,
        axes: Sequence[int],
        kind: str,
        prologue: Callable[[Emitter, str, list[str]], str] | None = None,
        extras: Sequence[Operand] = (),
    ) -> tuple[str, TensorType]:
        """Reduce ``axes`` away (no kept dims). ``prologue`` maps each element before combining."""
        rank = t.rank
        keep = [i for i in range(rank) if i not in axes]
        out_t = TensorType(t.dtype, tuple(t.shape[i] for i in keep))
        dt = t.dtype
        floating = dt in FLOATS
        init: float | int = {"sum": 0, "max": -math.inf, "min": math.inf, "prod": 1}[kind]
        if not floating and kind in ("max", "min"):
            info = np.iinfo(numpy_dtype(dt))
            init = int(info.min) if kind == "max" else int(info.max)
        combine = {
            "sum": "addf" if floating else "addi",
            "max": "maximumf" if floating else "maxsi",
            "min": "minimumf" if floating else "minsi",
            "prod": "mulf" if floating else "muli",
        }[kind]

        def body(em: Emitter, args: list[str]) -> list[str]:
            v = args[0]
            if prologue is not None:
                v = prologue(em, v, args[1:-1])
            return [em.op(f"arith.{combine} {args[-1]}, {v}", dt)]

        iterators = ["reduction" if i in axes else "parallel" for i in range(rank)]
        identity = [f"d{i}" for i in range(rank)]
        (result,) = self.generic(
            rank,
            iterators,
            [Operand(name, t, identity), *extras],
            [(out_t, [f"d{i}" for i in keep], None)],
            body,
            init=[init],
        )
        return result, out_t

    def reshape(self, name: str, src: TensorType, dst: TensorType) -> str:
        """Any static reshape as collapse to 1-D followed by expand (both are views in MLIR)."""
        if src.shape == dst.shape:
            return name
        n = src.size
        flat_t = TensorType(src.dtype, (n,))
        if src.rank == 0:
            flat = self.fresh("r")
            self.emit(
                f"{flat} = tensor.expand_shape {name} [] output_shape [1] : "
                f"{tensor_type(src)} into {tensor_type(flat_t)}"
            )
        elif src.rank == 1:
            flat = name
        else:
            flat = self.fresh("r")
            group = ", ".join(str(i) for i in range(src.rank))
            self.emit(
                f"{flat} = tensor.collapse_shape {name} [[{group}]] : "
                f"{tensor_type(src)} into {tensor_type(flat_t)}"
            )
        if dst.rank == 1:
            return flat
        result = self.fresh("r")
        if dst.rank == 0:
            self.emit(
                f"{result} = tensor.collapse_shape {flat} [] : "
                f"{tensor_type(flat_t)} into {tensor_type(dst)}"
            )
        else:
            group = ", ".join(str(i) for i in range(dst.rank))
            self.emit(
                f"{result} = tensor.expand_shape {flat} [[{group}]] output_shape "
                f"[{', '.join(map(str, dst.shape))}] : "
                f"{tensor_type(flat_t)} into {tensor_type(dst)}"
            )
        return result

    def keep_reduced(self, node: Node, name: str, t: TensorType) -> None:
        self.define(node.output, self.reshape(name, t, node.output.tensor_type))

    def _reduce_node(self, node: Node, kind: str, mean: bool = False) -> None:
        x = node.input(0)
        xt = x.tensor_type
        consts = [v.data if v is not None else None for v in node.inputs]
        axes = ops.reduce_axes(node, xt.rank, consts)
        if not axes:
            self.define(node.output, self.value(x))
            return
        name, t = self.reduce(self.value(x), xt, axes, kind)
        if mean:
            count = int(np.prod([xt.shape[a] for a in axes]))
            name = self.scale(name, t, 1.0 / count)
        self.keep_reduced(node, name, t)

    def scale(self, name: str, t: TensorType, factor: float) -> str:
        rank = t.rank
        identity = [f"d{i}" for i in range(rank)]

        def body(em: Emitter, args: list[str]) -> list[str]:
            return [em.op(f"arith.mulf {args[0]}, {em.fconst(factor, t.dtype)}", t.dtype)]

        (result,) = self.generic(
            rank, ["parallel"] * rank, [Operand(name, t, identity)], [(t, identity, None)], body
        )
        return result

    def lower_ReduceSum(self, node: Node) -> None:
        self._reduce_node(node, "sum")

    def lower_ReduceMean(self, node: Node) -> None:
        if node.input(0).tensor_type.dtype not in FLOATS:
            raise CodegenError("integer ReduceMean")
        self._reduce_node(node, "sum", mean=True)

    def lower_ReduceMax(self, node: Node) -> None:
        self._reduce_node(node, "max")

    def lower_ReduceMin(self, node: Node) -> None:
        self._reduce_node(node, "min")

    def lower_ReduceProd(self, node: Node) -> None:
        self._reduce_node(node, "prod")

    def _global(self, node: Node, kind: str) -> None:
        x = node.input(0)
        xt = x.tensor_type
        axes = (
            tuple(range(1, xt.rank - 1))
            if ops.conv_layout(node) == "NHWC"
            else tuple(range(2, xt.rank))
        )
        name, t = self.reduce(self.value(x), xt, axes, kind)
        if kind == "sum":
            name = self.scale(name, t, 1.0 / int(np.prod([xt.shape[a] for a in axes])))
        self.keep_reduced(node, name, t)

    def lower_GlobalAveragePool(self, node: Node) -> None:
        self._global(node, "sum")

    def lower_GlobalMaxPool(self, node: Node) -> None:
        self._global(node, "max")

    def lower_ArgMax(self, node: Node) -> None:
        x = node.input(0)
        xt = x.tensor_type
        if xt.dtype not in FLOATS:
            raise CodegenError("integer ArgMax")
        axis = int(node.attrs.get("axis", 0)) % xt.rank
        keep = [i for i in range(xt.rank) if i != axis]
        val_t = TensorType(xt.dtype, tuple(xt.shape[i] for i in keep))
        idx_t = TensorType("int64", val_t.shape)
        dt = xt.dtype

        def body(em: Emitter, args: list[str]) -> list[str]:
            v, best, best_i = args
            i = em.fresh("x")
            em.emit(f"{i} = linalg.index {axis} : index")
            i64 = em.fresh("x")
            em.emit(f"{i64} = arith.index_cast {i} : index to i64")
            gt = em._cmp("cmpf", "ogt", v, best, dt)
            return [
                em.op(f"arith.select {gt}, {v}, {best}", dt),
                em.op(f"arith.select {gt}, {i64}, {best_i}", "int64"),
            ]

        iterators = ["reduction" if i == axis else "parallel" for i in range(xt.rank)]
        out_res = [f"d{i}" for i in keep]
        results = self.generic(
            xt.rank,
            iterators,
            [Operand(self.value(x), xt, [f"d{i}" for i in range(xt.rank)])],
            [(val_t, out_res, None), (idx_t, out_res, None)],
            body,
            init=[-math.inf, 0],
        )
        self.keep_reduced(node, results[1], idx_t)

    def _softmax(self, node: Node, log: bool) -> None:
        x = node.input(0)
        xt = x.tensor_type
        axis = int(node.attrs.get("axis", -1)) % xt.rank
        name = self.value(x)
        rank = xt.rank
        identity = [f"d{i}" for i in range(rank)]
        red = [f"d{i}" for i in range(rank) if i != axis]
        dt = xt.dtype
        mx, mt = self.reduce(name, xt, (axis,), "max")

        def shifted_exp(em: Emitter, v: str, rest: list[str]) -> str:
            return em.op(f"math.exp {em.op(f'arith.subf {v}, {rest[0]}', dt)}", dt)

        total, tt = self.reduce(name, xt, (axis,), "sum", shifted_exp, [Operand(mx, mt, red)])

        def body(em: Emitter, args: list[str]) -> list[str]:
            v, m, s = args[:3]
            d = em.op(f"arith.subf {v}, {m}", dt)
            if log:
                return [em.op(f"arith.subf {d}, {em.op(f'math.log {s}', dt)}", dt)]
            return [em.op(f"arith.divf {em.op(f'math.exp {d}', dt)}, {s}", dt)]

        (result,) = self.generic(
            rank,
            ["parallel"] * rank,
            [Operand(name, xt, identity), Operand(mx, mt, red), Operand(total, tt, red)],
            [(xt, identity, None)],
            body,
        )
        self.define(node.output, result)

    def lower_Softmax(self, node: Node) -> None:
        self._softmax(node, False)

    def lower_LogSoftmax(self, node: Node) -> None:
        self._softmax(node, True)

    def lower_LayerNormalization(self, node: Node) -> None:
        x = node.input(0)
        xt = x.tensor_type
        if xt.dtype not in FLOATS:
            raise CodegenError("integer LayerNormalization")
        axis = int(node.attrs.get("axis", -1)) % xt.rank
        eps = float(node.attrs.get("epsilon", 1e-5))
        axes = tuple(range(axis, xt.rank))
        count = int(np.prod([xt.shape[a] for a in axes]))
        rank = xt.rank
        dt = xt.dtype
        name = self.value(x)
        red = [f"d{i}" for i in range(axis)]
        total, st = self.reduce(name, xt, axes, "sum")
        mean = self.scale(total, st, 1.0 / count)

        def centered_square(em: Emitter, v: str, rest: list[str]) -> str:
            c = em.op(f"arith.subf {v}, {rest[0]}", dt)
            return em.op(f"arith.mulf {c}, {c}", dt)

        sq, _ = self.reduce(name, xt, axes, "sum", centered_square, [Operand(mean, st, red)])
        identity = [f"d{i}" for i in range(rank)]
        inputs = [Operand(name, xt, identity), Operand(mean, st, red), Operand(sq, st, red)]
        params = [node.optional_input(1), node.optional_input(2)]
        inputs += [self.operand(v, xt.shape) for v in params if v is not None]

        def body(em: Emitter, args: list[str]) -> list[str]:
            v, m, s = args[:3]
            var = em.op(f"arith.mulf {s}, {em.fconst(1.0 / count, dt)}", dt)
            inv = em.op(f"math.rsqrt {em.op(f'arith.addf {var}, {em.fconst(eps, dt)}', dt)}", dt)
            y = em.op(f"arith.mulf {em.op(f'arith.subf {v}, {m}', dt)}, {inv}", dt)
            k = 3
            if params[0] is not None:
                y = em.op(f"arith.mulf {y}, {args[k]}", dt)
                k += 1
            if params[1] is not None:
                y = em.op(f"arith.addf {y}, {args[k]}", dt)
            return [y]

        (result,) = self.generic(rank, ["parallel"] * rank, inputs, [(xt, identity, None)], body)
        self.define(node.output, result)

    # -- data movement --------------------------------------------------------------------
    def copy_with_map(self, node: Node, src: Value, results: list[str], out_t: TensorType) -> None:
        rank = out_t.rank
        (result,) = self.generic(
            rank,
            ["parallel"] * rank,
            [Operand(self.value(src), src.tensor_type, results)],
            [(out_t, [f"d{i}" for i in range(rank)], None)],
            lambda em, a: [a[0]],
        )
        self.define(node.output, result)

    def lower_Transpose(self, node: Node) -> None:
        x = node.input(0)
        xt, out_t = x.tensor_type, node.output.tensor_type
        perm = ops.transpose_perm(node, xt.rank)
        init = self.empty(out_t)
        result = self.fresh("t")
        self.emit(
            f"{result} = linalg.transpose ins({self.value(x)} : {tensor_type(xt)}) "
            f"outs({init} : {tensor_type(out_t)}) permutation = [{', '.join(map(str, perm))}]"
        )
        self.define(node.output, result)

    def _view(self, node: Node) -> None:
        x = node.input(0)
        self.define(
            node.output, self.reshape(self.value(x), x.tensor_type, node.output.tensor_type)
        )

    lower_Reshape = _view
    lower_Flatten = _view
    lower_Squeeze = _view
    lower_Unsqueeze = _view
    lower_Identity = _view

    def lower_Expand(self, node: Node) -> None:
        x, out_t = node.input(0), node.output.tensor_type
        self.copy_with_map(node, x, broadcast_map(x.tensor_type.shape, out_t.shape), out_t)

    def lower_Tile(self, node: Node) -> None:
        x = node.input(0)
        shape = x.tensor_type.shape
        results = [
            f"d{i} mod {d}" if d != o else f"d{i}"
            for i, (d, o) in enumerate(zip(shape, node.output.tensor_type.shape, strict=True))
        ]
        self.copy_with_map(node, x, results, node.output.tensor_type)

    def lower_Slice(self, node: Node) -> None:
        x = node.input(0)
        consts = [v.data if v is not None else None for v in node.inputs]
        slices = ops.slice_params(node, x.tensor_type.shape, consts)
        results = []
        for i, (sl, d) in enumerate(zip(slices, x.tensor_type.shape, strict=True)):
            start, _, step = sl.indices(d)
            results.append(f"d{i} * {step} + {start}" if (start, step) != (0, 1) else f"d{i}")
        self.copy_with_map(node, x, results, node.output.tensor_type)

    def lower_Concat(self, node: Node) -> None:
        parts = [v for v in node.inputs if v is not None]
        out_t = node.output.tensor_type
        axis = int(node.attrs.get("axis", 0)) % out_t.rank
        if len(parts) == 1:
            self.define(node.output, self.value(parts[0]))
            return
        names = ", ".join(self.value(v) for v in parts)
        types = ", ".join(tensor_type(v.tensor_type) for v in parts)
        result = self.fresh("cat")
        self.emit(
            f"{result} = tensor.concat dim({axis}) {names} : ({types}) -> {tensor_type(out_t)}"
        )
        self.define(node.output, result)

    def lower_Split(self, node: Node) -> None:
        x = node.input(0)
        xt = x.tensor_type
        axis = int(node.attrs.get("axis", 0)) % xt.rank
        offset = 0
        for out in node.outputs:
            ot = out.tensor_type
            offsets = [offset if i == axis else 0 for i in range(xt.rank)]
            result = self.fresh("sl")
            self.emit(
                f"{result} = tensor.extract_slice {self.value(x)}[{', '.join(map(str, offsets))}] "
                f"[{', '.join(map(str, ot.shape))}] [{', '.join('1' for _ in ot.shape)}] : "
                f"{tensor_type(xt)} to {tensor_type(ot)}"
            )
            self.define(out, result)
            offset += ot.shape[axis]

    def lower_Pad(self, node: Node) -> None:
        x = node.input(0)
        xt = x.tensor_type
        if node.attrs.get("mode", "constant") != "constant":
            raise CodegenError("only constant Pad")
        consts = [v.data if v is not None else None for v in node.inputs]
        pads = ops.pad_amounts(node, xt.rank, consts)
        value_v = node.optional_input(2)
        value = 0.0
        if value_v is not None:
            if value_v.data is None:
                raise CodegenError("Pad with a dynamic value")
            value = float(np.asarray(value_v.data).reshape(()))
        fill: float | int = value if xt.dtype in FLOATS else int(value)
        name, _ = self.pad(self.value(x), xt, pads[: xt.rank], pads[xt.rank :], fill)
        self.define(node.output, name)

    def lower_Gather(self, node: Node) -> None:
        data, idx = node.input(0), node.input(1)
        dt_t, it = data.tensor_type, idx.tensor_type
        out_t = node.output.tensor_type
        axis = int(node.attrs.get("axis", 0)) % dt_t.rank
        rank = out_t.rank
        idx_res = [f"d{axis + j}" for j in range(it.rank)]
        size = dt_t.shape[axis]
        table = self.value(data)

        def body(em: Emitter, args: list[str]) -> list[str]:
            i = args[0]
            if it.dtype != "int64":
                i = em.cast(i, it.dtype, "int64")
            neg = em._cmp("cmpi", "slt", i, em.scalar(0, "int64"), "int64")
            wrapped = em.op(f"arith.addi {i}, {em.scalar(size, 'int64')}", "int64")
            i = em.op(f"arith.select {neg}, {wrapped}, {i}", "int64")
            ii = em.fresh("x")
            em.emit(f"{ii} = arith.index_cast {i} : i64 to index")
            coords = []
            for d in range(dt_t.rank):
                if d == axis:
                    coords.append(ii)
                    continue
                loop = d if d < axis else d + it.rank - 1
                c = em.fresh("x")
                em.emit(f"{c} = linalg.index {loop} : index")
                coords.append(c)
            v = em.fresh("x")
            em.emit(f"{v} = tensor.extract {table}[{', '.join(coords)}] : {tensor_type(dt_t)}")
            return [v]

        (result,) = self.generic(
            rank,
            ["parallel"] * rank,
            [Operand(self.value(idx), it, idx_res)],
            [(out_t, [f"d{i}" for i in range(rank)], None)],
            body,
        )
        self.define(node.output, result)

    # -- module ---------------------------------------------------------------------------
    def function(self) -> list[str]:
        g = self.graph
        args = []
        for i, v in enumerate(g.inputs):
            name = f"%arg{i}"
            self.names[id(v)] = name
            args.append(f"{name}: {tensor_type(v.tensor_type)}")
        for node in g.nodes:
            self.node(node)
        outs = [(n, v) for n, v in g.named_outputs()]
        names = [self.value(v) for _, v in outs]
        types = [tensor_type(v.tensor_type) for _, v in outs]
        self.emit(f"return {', '.join(names)} : {', '.join(types)}")
        body = self.lines
        result_types = ", ".join(types)
        inputs_doc = ", ".join(f"{_ident(v.name)}" for v in g.inputs)
        outputs_doc = ", ".join(_ident(n) for n, _ in outs)
        head = [
            f"// graph {g.name}: inputs ({inputs_doc}) outputs ({outputs_doc})",
            f"func.func @main({', '.join(args)}) -> ({result_types}) "
            "attributes {llvm.emit_c_interface} {",
        ]
        return head + body + ["}"]


def to_mlir(graph: Graph, dialect: str = "linalg", elide_constants: bool = False) -> MlirModule:
    """Print ``graph`` as an MLIR module (``dialect`` is "linalg" or "mgc")."""
    if dialect not in ("linalg", "mgc"):
        raise CodegenError(f"unknown MLIR form {dialect!r} (use 'linalg' or 'mgc')")
    em = Emitter(graph, lower=dialect == "linalg", elide_constants=elide_constants)
    func = em.function()
    lines = [f"module attributes {{mgc.graph = {_string(graph.name)}}} {{"]
    lines += ["  " + line if line else line for line in func]
    lines.append("}")
    if em.resources:
        lines += ["", "{-#", "  dialect_resources: {", "    builtin: {"]
        lines += [f'      {k}: "{v}",' for k, v in em.resources[:-1]]
        k, v = em.resources[-1]
        lines.append(f'      {k}: "{v}"')
        lines += ["    }", "  }", "#-}"]
    return MlirModule(
        "\n".join(lines) + "\n",
        lowered=dict(sorted(em.lowered.items())),
        fallback=dict(sorted(em.fallback.items())),
        resources=len(em.resources),
        resource_bytes=em.resource_bytes,
    )


_CMP = {
    "Equal": ("oeq", "eq"),
    "Less": ("olt", "slt"),
    "Greater": ("ogt", "sgt"),
    "LessOrEqual": ("ole", "sle"),
    "GreaterOrEqual": ("oge", "sge"),
}
_FBIN = {
    "Add": "addf",
    "Sub": "subf",
    "Mul": "mulf",
    "Div": "divf",
    "Max": "maximumf",
    "Min": "minimumf",
}
_IBIN = {
    "Add": "addi",
    "Sub": "subi",
    "Mul": "muli",
    "Div": "divsi",
    "Max": "maxsi",
    "Min": "minsi",
}
_FUN = {
    "Tanh": "math.tanh",
    "Erf": "math.erf",
    "Exp": "math.exp",
    "Log": "math.log",
    "Sqrt": "math.sqrt",
    "Sin": "math.sin",
    "Cos": "math.cos",
    "Floor": "math.floor",
    "Ceil": "math.ceil",
    "Abs": "math.absf",
    "Neg": "arith.negf",
}
