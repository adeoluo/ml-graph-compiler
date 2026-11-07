"""C code generation.

The emitter walks the scheduled graph once and writes one C function, ``mgc_run``, that executes
the model with no allocation: every tensor has the byte offset chosen by the memory planner in
one arena, constants live in a separate weights blob, and graph inputs and outputs are
caller-provided pointers. Shapes are static, so every loop bound, stride and offset is a
compile-time constant in the generated code.

Lowering strategy per operator family:

* Elementwise operators (and FusedElementwise programs from loop fusion) become one loop nest
  over the output with per-operand broadcast strides; adjacent dimensions that are contiguous
  for every operand are collapsed, scalar constants become literals, and the body is a sequence
  of typed temporaries, so clang can vectorize it.
* Transpose, Expand, Slice, Concat, Split and Pad are the same loop nest with an identity body
  and permuted, broadcast or offset strides.
* MatMul, Gemm, FusedGemm, Conv and FusedConv call the packed GEMM driver of the runtime.
  Constant right-hand sides are packed into micro-kernel panels at compile time and stored in
  the weights blob; convolutions read the image through implicit im2col packing. Depthwise
  convolutions use a direct kernel vectorized over channels.
* Views (Reshape, Flatten, Squeeze, Unsqueeze, Identity, Dropout) emit nothing: the planner made
  their output an alias of their input.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ml_graph_compiler import ops
from ml_graph_compiler.errors import CodegenError
from ml_graph_compiler.ir import Graph, Node, TensorType, Value, comment_text, numpy_dtype
from ml_graph_compiler.memory import ALIAS_OPS, MemoryPlan, Strategy, plan_memory

MR, NR = 8, 12
ALIGN = 64

CTYPES = {
    "float32": "float",
    "float64": "double",
    "int64": "int64_t",
    "int32": "int32_t",
    "int8": "int8_t",
    "uint8": "uint8_t",
    "bool": "uint8_t",
}

ACT_CODES = {
    "none": "MGC_ACT_NONE",
    "relu": "MGC_ACT_RELU",
    "clip": "MGC_ACT_CLIP",
    "gelu": "MGC_ACT_GELU",
    "gelu_tanh": "MGC_ACT_GELU_TANH",
    "tanh": "MGC_ACT_TANH",
    "sigmoid": "MGC_ACT_SIGMOID",
}

# Elementwise expression templates for float32; {0}, {1}, {2} are operand expressions.
_FLOAT_EXPR = {
    "Add": "({0} + {1})",
    "Sub": "({0} - {1})",
    "Mul": "({0} * {1})",
    "Div": "({0} / {1})",
    "Pow": "powf({0}, {1})",
    "Max": "fmaxf({0}, {1})",
    "Min": "fminf({0}, {1})",
    "Relu": "({0} > 0.0f ? {0} : 0.0f)",
    "Sigmoid": "mgc_sigmoidf({0})",
    "Tanh": "mgc_tanhf({0})",
    "Erf": "mgc_erff({0})",
    "Sqrt": "sqrtf({0})",
    "Exp": "mgc_expf({0})",
    "Log": "logf({0})",
    "Neg": "(-{0})",
    "Abs": "fabsf({0})",
    "Reciprocal": "(1.0f / {0})",
    "Floor": "floorf({0})",
    "Ceil": "ceilf({0})",
    "Sin": "sinf({0})",
    "Cos": "cosf({0})",
    "Softplus": "mgc_softplusf({0})",
    "Identity": "{0}",
}
_INT_EXPR = {
    "Add": "({0} + {1})",
    "Sub": "({0} - {1})",
    "Mul": "({0} * {1})",
    "Div": "({0} / {1})",
    "Max": "({0} > {1} ? {0} : {1})",
    "Min": "({0} < {1} ? {0} : {1})",
    "Neg": "(-{0})",
    "Abs": "({0} < 0 ? -{0} : {0})",
    "Relu": "({0} > 0 ? {0} : 0)",
    "Identity": "{0}",
}
_COMPARE_EXPR = {
    "Equal": "({0} == {1})",
    "Less": "({0} < {1})",
    "Greater": "({0} > {1})",
    "LessOrEqual": "({0} <= {1})",
    "GreaterOrEqual": "({0} >= {1})",
    "And": "({0} && {1})",
    "Or": "({0} || {1})",
    "Xor": "(!{0} != !{1})",
}


def _literal(value: float | int | bool, dtype: str) -> str:
    if dtype == "bool":
        return "1" if value else "0"
    if dtype in ("float32", "float64"):
        v = float(value)
        if math.isnan(v):
            return "NAN"
        if math.isinf(v):
            return "INFINITY" if v > 0 else "(-INFINITY)"
        text = repr(v)
        return f"{text}f" if dtype == "float32" else text
    return f"((int64_t){int(value)})" if dtype == "int64" else str(int(value))


def _strides(shape: Sequence[int]) -> list[int]:
    out = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        out[i] = out[i + 1] * int(shape[i + 1])
    return out


def _broadcast_strides(shape: Sequence[int], out_shape: Sequence[int]) -> list[int]:
    """Element strides of a tensor of ``shape`` broadcast (right-aligned) to ``out_shape``."""
    src = _strides(shape)
    pad = len(out_shape) - len(shape)
    result = []
    for d in range(len(out_shape)):
        k = d - pad
        if k < 0 or shape[k] == 1:
            result.append(0)
        else:
            result.append(src[k])
    return result


@dataclass
class Operand:
    """One array in a loop nest: a C pointer (or a literal) plus element strides over the loops."""

    ptr: str | None
    ctype: str
    strides: list[int]
    base: int = 0
    literal: str | None = None

    def at(self, index: str) -> str:
        if self.literal is not None:
            return self.literal
        return f"{self.ptr}[{index}]"


@dataclass
class WeightsBuilder:
    """Constants laid out in one blob with 64-byte aligned offsets.

    Chunks are kept as references to the original arrays and only written out by
    :meth:`write`, so building the blob never holds a second copy of the weights in memory.
    """

    chunks: list[tuple[int, np.ndarray]] = field(default_factory=list)
    size: int = 0
    by_key: dict[Any, int] = field(default_factory=dict)

    def add(self, array: np.ndarray, key: Any = None) -> int:
        if key is not None and key in self.by_key:
            return self.by_key[key]
        data = np.ascontiguousarray(array)
        offset = self.size
        self.chunks.append((offset, data))
        self.size += data.nbytes
        self.size += (-self.size) % ALIGN
        if key is not None:
            self.by_key[key] = offset
        return offset

    @property
    def nbytes(self) -> int:
        return max(self.size, ALIGN)

    def digest(self) -> str:
        h = hashlib.blake2b(digest_size=12)
        for offset, data in self.chunks:
            h.update(offset.to_bytes(8, "little"))
            h.update(memoryview(data.reshape(-1).view(np.uint8)))
        return h.hexdigest()

    def write(self, path: Path) -> None:
        """Stream the blob to ``path`` (zero padding between aligned chunks)."""
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            pos = 0
            for offset, data in self.chunks:
                if offset > pos:
                    f.write(b"\0" * (offset - pos))
                f.write(memoryview(data.reshape(-1).view(np.uint8)))
                pos = offset + data.nbytes
            if self.nbytes > pos:
                f.write(b"\0" * (self.nbytes - pos))
        tmp.replace(path)

    def blob(self) -> np.ndarray:
        out = np.zeros(self.nbytes, np.uint8)
        for offset, data in self.chunks:
            out[offset : offset + data.nbytes] = data.reshape(-1).view(np.uint8)
        return out


def pack_b_panels(b: np.ndarray) -> np.ndarray:
    """Pack a K x N float32 matrix into the micro-kernel panel layout (see mgc_pack_b)."""
    b = np.asarray(b, np.float32)
    k, n = b.shape
    full, rest = divmod(n, NR)
    out = np.empty((full + (1 if rest else 0), k, NR), np.float32)
    if full:
        out[:full] = b[:, : full * NR].reshape(k, full, NR).transpose(1, 0, 2)
    if rest:
        out[full, :, :rest] = b[:, full * NR :]
        out[full, :, rest:] = 0.0
    return out


@dataclass
class GeneratedModel:
    source: str
    weights: WeightsBuilder
    arena_bytes: int
    inputs: list[tuple[str, TensorType]]
    outputs: list[tuple[str, TensorType]]
    plan: MemoryPlan
    lines: int
    kernels: dict[str, int]


class CEmitter:
    """Generates the C source and weights blob for one graph (see the module docstring)."""

    SUPPORTED = frozenset(
        {
            *(op for op in ops.REGISTRY if ops.is_elementwise(op)),
            "FusedElementwise",
            "Transpose",
            "Expand",
            "Slice",
            "Concat",
            "Split",
            "Pad",
            "Tile",
            "Gather",
            "Shape",
            "Size",
            "ConstantOfShape",
            "Range",
            "ReduceMean",
            "ReduceSum",
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
            "ArgMax",
            "Softmax",
            "LogSoftmax",
            "LayerNormalization",
            "BatchNormalization",
            "MatMul",
            "Gemm",
            "FusedGemm",
            "Conv",
            "FusedConv",
            "MaxPool",
            "AveragePool",
            "GlobalAveragePool",
            "GlobalMaxPool",
            *ALIAS_OPS,
        }
    )

    def __init__(
        self,
        graph: Graph,
        strategy: Strategy = "greedy_by_size",
        inplace: bool = True,
        profile: bool = False,
    ) -> None:
        self.profile = profile
        unsupported = sorted({n.op for n in graph.nodes} - self.SUPPORTED)
        if unsupported:
            raise CodegenError(f"the C backend does not support: {', '.join(unsupported)}")
        for v in graph.values():
            if v.type is not None and v.type.dtype == "float16":
                raise CodegenError(f"float16 tensors are not supported ({v.name})")
        self.graph = graph
        self.plan = plan_memory(graph, strategy=strategy, inplace=inplace)
        self.weights = WeightsBuilder()
        self.body: list[str] = []
        self.statics: list[str] = []
        self.kernels: dict[str, int] = {}
        self._tmp = 0

    # -- pointers ---------------------------------------------------------------------------
    def _root(self, value: Value) -> Value:
        while value.data is None and value.producer is not None and value.producer.op in ALIAS_OPS:
            value = value.producer.input(0)
        return value

    def const_offset(self, value: Value) -> int:
        data = value.data
        assert data is not None
        return self.weights.add(data, key=("raw", id(value)))

    def ptr(self, value: Value, ctype: str | None = None) -> str:
        ct = ctype or CTYPES[value.tensor_type.dtype]
        loc = self.plan.location(value)
        if loc.kind == "const":
            root = self._root(value)
            if root.data is None:
                raise CodegenError(f"constant {value.name!r} has no data")
            return f"((const {ct} *)(W + {self.const_offset(root)}))"
        if loc.kind == "input":
            return f"(({ct} *)inputs[{loc.index}])"
        if loc.kind == "output":
            return f"(({ct} *)outputs[{loc.index}])"
        return f"(({ct} *)(A + {loc.index}))"

    def operand(self, value: Value, out_shape: Sequence[int]) -> Operand:
        t = value.tensor_type
        ct = CTYPES[t.dtype]
        if value.is_constant and value.data is not None and value.data.size == 1:
            lit = _literal(value.data.reshape(()).item(), t.dtype)
            return Operand(None, ct, [0] * len(out_shape), literal=lit)
        return Operand(self.ptr(value), ct, _broadcast_strides(t.shape, out_shape))

    def out_operand(self, value: Value) -> Operand:
        t = value.tensor_type
        return Operand(self.ptr(value), CTYPES[t.dtype], _strides(t.shape))

    def fresh(self, hint: str = "t") -> str:
        self._tmp += 1
        return f"{hint}{self._tmp}"

    def emit(self, line: str) -> None:
        self.body.append(line)

    def count(self, kernel: str) -> None:
        self.kernels[kernel] = self.kernels.get(kernel, 0) + 1

    # -- loop nests -------------------------------------------------------------------------
    def loops(
        self, shape: Sequence[int], out: Operand, leaves: list[Operand], body: list[str]
    ) -> None:
        """Emit ``for`` loops over ``shape``. ``body`` lines may use ``{o}`` (output index) and
        ``{i0}``, ``{i1}``, ... (leaf indices); collapsible dimensions are merged first."""
        dims = [int(d) for d in shape]
        operands = [out, *leaves]
        keep = [i for i, d in enumerate(dims) if d != 1]
        dims = [dims[i] for i in keep]
        strides = [[op.strides[i] for i in keep] for op in operands]
        d = len(dims) - 1
        while d > 0:
            if all(s[d - 1] == s[d] * dims[d] for s in strides):
                dims[d - 1] *= dims[d]
                del dims[d]
                for s in strides:
                    s[d - 1] = s[d]
                    del s[d]
            d -= 1
        total = int(np.prod(dims)) if dims else 1
        if total == 0:
            return
        names = [f"j{k}" for k in range(len(dims))]

        def index(op_i: int) -> str:
            terms = [str(operands[op_i].base)] if operands[op_i].base else []
            for name, s in zip(names, strides[op_i], strict=True):
                if s == 1:
                    terms.append(name)
                elif s:
                    terms.append(f"{name} * {s}L")
            return " + ".join(terms) if terms else "0"

        self.emit("{")
        for k, (name, extent) in enumerate(zip(names, dims, strict=True)):
            if k == 0 and total >= 65536 and (len(dims) > 1 or extent >= 65536):
                self.emit("#pragma omp parallel for schedule(static)")
            self.emit(f"for (long {name} = 0; {name} < {extent}L; {name}++) {{")
        subs = {"o": index(0)}
        for k in range(len(leaves)):
            subs[f"i{k}"] = index(k + 1)
        for line in body:
            self.emit("  " + line.format(**subs))
        for _ in dims:
            self.emit("}")
        self.emit("}")

    def copy(self, shape: Sequence[int], out: Operand, src: Operand) -> None:
        self.loops(shape, out, [src], [f"{out.ptr}[{{o}}] = {src.at('{i0}')};"])

    # -- elementwise --------------------------------------------------------------------------
    def expression(
        self, op: str, attrs: dict[str, Any], args: list[str], types: list[str], out_dtype: str
    ) -> str:
        in_dtype = types[0] if types else out_dtype
        if op in _COMPARE_EXPR:
            return _COMPARE_EXPR[op].format(*args)
        if op == "Not":
            return f"(!{args[0]})"
        if op == "Where":
            return f"({args[0]} ? {args[1]} : {args[2]})"
        if op == "Cast":
            target = CTYPES[out_dtype]
            if out_dtype == "bool":
                return f"(({args[0]}) != 0)"
            return f"(({target})({args[0]}))"
        if in_dtype == "float32":
            if op == "Clip":
                lo = (
                    args[1]
                    if len(args) > 1 and args[1]
                    else _literal(attrs.get("min", -np.inf), "float32")
                )
                hi = (
                    args[2]
                    if len(args) > 2 and args[2]
                    else _literal(attrs.get("max", np.inf), "float32")
                )
                return f"fminf(fmaxf({args[0]}, {lo}), {hi})"
            if op == "Gelu":
                fn = (
                    "mgc_gelu_tanhf"
                    if attrs.get("approximate", "none") == "tanh"
                    else "mgc_gelu_erff"
                )
                return f"{fn}({args[0]})"
            if op == "LeakyRelu":
                alpha = _literal(attrs.get("alpha", 0.01), "float32")
                return f"({args[0]} >= 0.0f ? {args[0]} : {alpha} * {args[0]})"
            if op == "HardSigmoid":
                a = _literal(attrs.get("alpha", 0.2), "float32")
                b = _literal(attrs.get("beta", 0.5), "float32")
                return f"fminf(fmaxf({a} * {args[0]} + {b}, 0.0f), 1.0f)"
            if op in _FLOAT_EXPR:
                return _FLOAT_EXPR[op].format(*args)
        elif in_dtype in ("int64", "int32", "int8", "uint8", "bool"):
            if op == "Clip":
                ilo = args[1] if len(args) > 1 and args[1] else None
                ihi = args[2] if len(args) > 2 and args[2] else None
                e = args[0]
                if ilo:
                    e = f"({e} < {ilo} ? {ilo} : {e})"
                if ihi:
                    e = f"({e} > {ihi} ? {ihi} : {e})"
                return e
            if op in _INT_EXPR:
                return _INT_EXPR[op].format(*args)
        elif in_dtype == "float64" and op in ("Add", "Sub", "Mul", "Div", "Neg", "Identity"):
            return _FLOAT_EXPR[op].format(*args)
        raise CodegenError(f"elementwise {op} on {in_dtype} is not supported by the C backend")

    def elementwise(
        self,
        out_value: Value,
        program: list[tuple[str, dict[str, Any], list[int]]],
        leaves: list[Value | None],
    ) -> None:
        shape = out_value.tensor_type.shape
        out = self.out_operand(out_value)
        operands: list[Operand] = []
        leaf_expr: list[str | None] = []
        leaf_types: list[str | None] = []
        for v in leaves:
            if v is None:
                leaf_expr.append(None)
                leaf_types.append(None)
                continue
            operand = self.operand(v, shape)
            leaf_types.append(v.tensor_type.dtype)
            if operand.literal is not None:
                leaf_expr.append(operand.literal)
            else:
                leaf_expr.append(operand.at(f"{{i{len(operands)}}}"))
                operands.append(operand)
        temps: list[str] = []
        types: list[str] = []
        lines: list[str] = []
        types_all: list[TensorType | None] = [v.type if v is not None else None for v in leaves]
        consts: list[np.ndarray | None] = [v.data if v is not None else None for v in leaves]
        for k, (op, attrs, args) in enumerate(program):
            node = Node(op, [], [Value("_")], dict(attrs))
            arg_types = [types_all[a] if a >= 0 else None for a in args]
            out_t = ops.spec(op).infer(
                node, arg_types, [consts[a] if a >= 0 else None for a in args]
            )[0]
            types_all.append(out_t)
            consts.append(None)
            arg_exprs = []
            arg_dtypes = []
            for a in args:
                if a < 0:
                    arg_exprs.append("")
                    continue
                if a < len(leaves):
                    expr = leaf_expr[a]
                    assert expr is not None
                    arg_exprs.append(expr)
                    arg_dtypes.append(str(leaf_types[a]))
                else:
                    arg_exprs.append(temps[a - len(leaves)])
                    arg_dtypes.append(types[a - len(leaves)])
            name = f"v{k}"
            expr = self.expression(op, attrs, arg_exprs, arg_dtypes, out_t.dtype)
            lines.append(f"{CTYPES[out_t.dtype]} {name} = {expr};")
            temps.append(name)
            types.append(out_t.dtype)
        lines.append(f"{out.ptr}[{{o}}] = {temps[-1]};")
        # braces in literals are not expected, but escape any to keep str.format safe
        self.loops(shape, out, operands, lines)
        self.count("elementwise")

    # -- per-operator emitters ----------------------------------------------------------------
    def node(self, node: Node) -> None:
        types = ", ".join(str(o.type) for o in node.outputs)
        self.emit(f"/* {node.op} {comment_text(node.name)} -> {types} */")
        if node.op in ALIAS_OPS:
            if node.op == "Dropout" and len(node.outputs) > 1:
                mask = node.outputs[1]
                self.emit(f"memset({self.ptr(mask)}, 1, {mask.tensor_type.nbytes});")
            self.count("view")
            return
        if node.op == "FusedElementwise":
            program = [
                (str(op), dict(a), [int(x) for x in args]) for op, a, args in node.attrs["program"]
            ]
            self.elementwise(node.output, program, list(node.inputs))
            return
        if ops.is_elementwise(node.op):
            args = list(range(len(node.inputs)))
            args = [i if node.inputs[i] is not None else -1 for i in args]
            self.elementwise(node.output, [(node.op, dict(node.attrs), args)], list(node.inputs))
            return
        handler = getattr(self, f"op_{node.op}")
        handler(node)

    def _consts(self, node: Node) -> list[np.ndarray | None]:
        return [v.data if v is not None else None for v in node.inputs]

    def op_Transpose(self, node: Node) -> None:
        x = node.input(0)
        perm = ops.transpose_perm(node, x.tensor_type.rank)
        src_strides = _strides(x.tensor_type.shape)
        src = Operand(self.ptr(x), CTYPES[x.tensor_type.dtype], [src_strides[p] for p in perm])
        self.copy(node.output.tensor_type.shape, self.out_operand(node.output), src)
        self.count("transpose")

    def op_Expand(self, node: Node) -> None:
        shape = node.output.tensor_type.shape
        self.copy(shape, self.out_operand(node.output), self.operand(node.input(0), shape))
        self.count("copy")

    def op_Tile(self, node: Node) -> None:
        x = node.input(0)
        xs, os_ = x.tensor_type.shape, node.output.tensor_type.shape
        src_strides = _strides(xs)
        out = self.out_operand(node.output)
        total = int(np.prod(os_)) if os_ else 1
        ct = CTYPES[x.tensor_type.dtype]
        self.emit("{")
        self.emit(f"for (long o = 0; o < {total}L; o++) {{")
        self.emit("  long rem = o, src = 0;")
        for d in range(len(os_) - 1, -1, -1):
            self.emit(
                f"  src += (rem % {os_[d]}L % {xs[d]}L) * {src_strides[d]}L; rem /= {os_[d]}L;"
            )
        self.emit(f"  {out.ptr}[o] = (({ct} *){self.ptr(x)})[src];")
        self.emit("}")
        self.emit("}")
        self.count("copy")

    def op_Slice(self, node: Node) -> None:
        x = node.input(0)
        xs = x.tensor_type.shape
        slices = ops.slice_params(node, xs, self._consts(node))
        src_strides = _strides(xs)
        base = 0
        strides = []
        for sl, st in zip(slices, src_strides, strict=True):
            base += int(sl.start or 0) * st
            strides.append(int(sl.step or 1) * st)
        src = Operand(self.ptr(x), CTYPES[x.tensor_type.dtype], strides, base)
        self.copy(node.output.tensor_type.shape, self.out_operand(node.output), src)
        self.count("copy")

    def op_Concat(self, node: Node) -> None:
        out_t = node.output.tensor_type
        axis = int(node.attrs.get("axis", 0))
        axis = axis + out_t.rank if axis < 0 else axis
        out_strides = _strides(out_t.shape)
        offset = 0
        for v in node.inputs:
            if v is None:
                continue
            t = v.tensor_type
            if t.size:
                out = Operand(
                    self.ptr(node.output), CTYPES[t.dtype], out_strides, offset * out_strides[axis]
                )
                self.copy(t.shape, out, self.operand(v, t.shape))
            offset += t.shape[axis]
        self.count("copy")

    def op_Split(self, node: Node) -> None:
        x = node.input(0)
        xt = x.tensor_type
        axis = int(node.attrs.get("axis", 0))
        axis = axis + xt.rank if axis < 0 else axis
        src_strides = _strides(xt.shape)
        offset = 0
        for out_v in node.outputs:
            t = out_v.tensor_type
            if out_v.users or out_v in self.graph.outputs:
                src = Operand(
                    self.ptr(x), CTYPES[xt.dtype], src_strides, offset * src_strides[axis]
                )
                self.copy(t.shape, self.out_operand(out_v), src)
            offset += t.shape[axis]
        self.count("copy")

    def op_Pad(self, node: Node) -> None:
        mode = node.attrs.get("mode", "constant")
        if mode != "constant":
            raise CodegenError(f"Pad mode {mode!r} is not supported by the C backend")
        x = node.input(0)
        xt, ot = x.tensor_type, node.output.tensor_type
        pads = ops.pad_amounts(node, xt.rank, self._consts(node))
        if any(p < 0 for p in pads):
            raise CodegenError("negative Pad amounts are not supported by the C backend")
        value_v = node.optional_input(2)
        fill = 0.0
        if value_v is not None:
            if value_v.data is None:
                raise CodegenError("Pad value must be a constant")
            fill = value_v.data.reshape(()).item()
        out = self.out_operand(node.output)
        self.loops(ot.shape, out, [], [f"{out.ptr}[{{o}}] = {_literal(fill, ot.dtype)};"])
        out_strides = _strides(ot.shape)
        base = sum(p * s for p, s in zip(pads[: xt.rank], out_strides, strict=True))
        inner = Operand(out.ptr, out.ctype, out_strides, base)
        self.copy(xt.shape, inner, self.operand(x, xt.shape))
        self.count("copy")

    def op_Gather(self, node: Node) -> None:
        x, idx = node.input(0), node.input(1)
        xt, it = x.tensor_type, idx.tensor_type
        axis = int(node.attrs.get("axis", 0))
        axis = axis + xt.rank if axis < 0 else axis
        outer = int(np.prod(xt.shape[:axis])) if axis else 1
        dim = xt.shape[axis]
        inner = int(np.prod(xt.shape[axis + 1 :])) if axis + 1 < xt.rank else 1
        n = it.size
        ct = CTYPES[xt.dtype]
        item = xt.itemsize
        ict = CTYPES[it.dtype]
        self.emit("{")
        self.emit(f"const {ict} *ix = {self.ptr(idx)};")
        self.emit(f"const {ct} *src = {self.ptr(x)};")
        self.emit(f"{ct} *dst = {self.ptr(node.output)};")
        self.emit(f"for (long t = 0; t < {n}L; t++) {{")
        self.emit(f"  long k = (long)ix[t]; if (k < 0) k += {dim}L;")
        self.emit(f"  if (k < 0 || k >= {dim}L) return 1;")
        self.emit(f"  for (long o = 0; o < {outer}L; o++)")
        self.emit(
            f"    memcpy(dst + (o * {n}L + t) * {inner}L, src + (o * {dim}L + k) * {inner}L, "
            f"{inner * item});"
        )
        self.emit("}")
        self.emit("}")
        self.count("gather")

    def _write_constant(self, value: Value, array: np.ndarray) -> None:
        out = self.out_operand(value)
        flat = array.reshape(-1)
        if flat.size and np.all(flat == flat[0]):
            self.loops(
                value.tensor_type.shape,
                out,
                [],
                [f"{out.ptr}[{{o}}] = {_literal(flat[0].item(), value.tensor_type.dtype)};"],
            )
            return
        off = self.weights.add(array.astype(numpy_dtype(value.tensor_type.dtype)))
        self.emit(f"memcpy({out.ptr}, W + {off}, {value.tensor_type.nbytes});")

    def op_Shape(self, node: Node) -> None:
        self._write_constant(node.output, ops.shape_value(node, node.input(0).tensor_type.shape))
        self.count("constant")

    def op_Size(self, node: Node) -> None:
        self._write_constant(node.output, np.asarray([node.input(0).tensor_type.size], np.int64))
        self.count("constant")

    def op_ConstantOfShape(self, node: Node) -> None:
        value = node.attrs.get("value")
        fill = np.asarray(value).reshape(-1)[0].item() if value is not None else 0.0
        out = self.out_operand(node.output)
        self.loops(
            node.output.tensor_type.shape,
            out,
            [],
            [f"{out.ptr}[{{o}}] = {_literal(fill, node.output.tensor_type.dtype)};"],
        )
        self.count("constant")

    def op_Range(self, node: Node) -> None:
        t = node.output.tensor_type
        ct = CTYPES[t.dtype]
        start = self.operand(node.input(0), ())
        delta = self.operand(node.input(2), ())
        out = self.out_operand(node.output)
        self.emit(
            f"for (long o = 0; o < {t.size}L; o++) "
            f"{out.ptr}[o] = ({ct})({start.at('0')} + ({ct})o * {delta.at('0')});"
        )
        self.count("constant")

    def _reduce(self, node: Node, kind: str) -> None:
        x = node.input(0)
        xt = x.tensor_type
        axes = ops.reduce_axes(node, xt.rank, self._consts(node))
        out = node.output
        ct = CTYPES[xt.dtype]
        if not axes:
            self.copy(xt.shape, self.out_operand(out), self.operand(x, xt.shape))
            return
        if list(axes) != list(range(axes[0], axes[-1] + 1)):
            raise CodegenError(f"{node.op} over non-adjacent axes {axes} is not supported")
        outer = int(np.prod(xt.shape[: axes[0]])) if axes[0] else 1
        n = int(np.prod([xt.shape[a] for a in axes]))
        inner = int(np.prod(xt.shape[axes[-1] + 1 :])) if axes[-1] + 1 < xt.rank else 1
        if kind in ("mean", "max") and ct == "float":
            self.emit(
                f"mgc_reduce_mid({int(kind == 'max')}, {outer}L, {n}L, {inner}L, "
                f"{self.ptr(x)}, {self.ptr(out)});"
            )
            self.count("reduce")
            return
        init = {
            "sum": "0",
            "mean": "0",
            "prod": "1",
            "max": "-INFINITY" if ct == "float" else "INT64_MIN",
            "min": "INFINITY" if ct == "float" else "INT64_MAX",
        }[kind]
        update = {
            "sum": "acc += v;",
            "mean": "acc += v;",
            "prod": "acc *= v;",
            "max": "acc = v > acc ? v : acc;",
            "min": "acc = v < acc ? v : acc;",
        }[kind]
        acc_t = "double" if ct == "float" and kind in ("sum", "mean", "prod") else ct
        self.emit("{")
        self.emit(f"const {ct} *src = {self.ptr(x)}; {ct} *dst = {self.ptr(out)};")
        self.emit(f"for (long o = 0; o < {outer}L; o++) for (long i = 0; i < {inner}L; i++) {{")
        self.emit(f"  {acc_t} acc = {init};")
        self.emit(
            f"  for (long k = 0; k < {n}L; k++) {{ "
            f"{ct} v = src[(o * {n}L + k) * {inner}L + i]; {update} }}"
        )
        div = f" / {n}.0" if kind == "mean" else ""
        self.emit(f"  dst[o * {inner}L + i] = ({ct})(acc{div});")
        self.emit("}")
        self.emit("}")
        self.count("reduce")

    def op_ReduceMean(self, node: Node) -> None:
        self._reduce(node, "mean")

    def op_ReduceSum(self, node: Node) -> None:
        self._reduce(node, "sum")

    def op_ReduceMax(self, node: Node) -> None:
        self._reduce(node, "max")

    def op_ReduceMin(self, node: Node) -> None:
        self._reduce(node, "min")

    def op_ReduceProd(self, node: Node) -> None:
        self._reduce(node, "prod")

    def op_ArgMax(self, node: Node) -> None:
        x = node.input(0)
        xt = x.tensor_type
        axis = int(node.attrs.get("axis", 0))
        axis = axis + xt.rank if axis < 0 else axis
        outer = int(np.prod(xt.shape[:axis])) if axis else 1
        n = xt.shape[axis]
        inner = int(np.prod(xt.shape[axis + 1 :])) if axis + 1 < xt.rank else 1
        ct = CTYPES[xt.dtype]
        self.emit("{")
        self.emit(f"const {ct} *src = {self.ptr(x)}; int64_t *dst = {self.ptr(node.output)};")
        self.emit(f"for (long o = 0; o < {outer}L; o++) for (long i = 0; i < {inner}L; i++) {{")
        self.emit(f"  long best = 0; {ct} bv = src[o * {n}L * {inner}L + i];")
        self.emit(
            f"  for (long k = 1; k < {n}L; k++) {{ {ct} v = src[(o * {n}L + k) * {inner}L + i]; "
            "if (v > bv) { bv = v; best = k; } }"
        )
        self.emit(f"  dst[o * {inner}L + i] = best;")
        self.emit("}")
        self.emit("}")
        self.count("reduce")

    def _softmax(self, node: Node, log: int) -> None:
        x = node.input(0)
        xt = x.tensor_type
        self._require_f32(node, xt)
        axis = int(node.attrs.get("axis", -1))
        axis = axis + xt.rank if axis < 0 else axis
        outer = int(np.prod(xt.shape[:axis])) if axis else 1
        inner = int(np.prod(xt.shape[axis + 1 :])) if axis + 1 < xt.rank else 1
        self.emit(
            f"mgc_softmax({outer}L, {xt.shape[axis]}L, {inner}L, {self.ptr(x)}, "
            f"{self.ptr(node.output)}, {log});"
        )
        self.count("softmax")

    def op_Softmax(self, node: Node) -> None:
        self._softmax(node, 0)

    def op_LogSoftmax(self, node: Node) -> None:
        self._softmax(node, 1)

    def _require_f32(self, node: Node, t: TensorType) -> None:
        if t.dtype != "float32":
            raise CodegenError(f"{node.op} on {t.dtype} is not supported by the C backend")

    def _vector_ptr(self, value: Value | None, n: int, what: str) -> str:
        if value is None:
            return "NULL"
        if value.tensor_type.size != n:
            raise CodegenError(f"{what} must have {n} elements, got {value.tensor_type.shape}")
        return self.ptr(value, "float")

    def op_LayerNormalization(self, node: Node) -> None:
        x = node.input(0)
        xt = x.tensor_type
        self._require_f32(node, xt)
        axis = int(node.attrs.get("axis", -1))
        axis = axis + xt.rank if axis < 0 else axis
        cols = int(np.prod(xt.shape[axis:]))
        rows = xt.size // max(cols, 1)
        scale = self._vector_ptr(node.optional_input(1), cols, "LayerNormalization scale")
        bias = self._vector_ptr(node.optional_input(2), cols, "LayerNormalization bias")
        eps = _literal(float(node.attrs.get("epsilon", 1e-5)), "float32")
        self.emit(
            f"mgc_layer_norm({rows}L, {cols}L, {self.ptr(x)}, {scale}, {bias}, {eps}, "
            f"{self.ptr(node.output)});"
        )
        self.count("layer_norm")

    def op_BatchNormalization(self, node: Node) -> None:
        x = node.input(0)
        xt = x.tensor_type
        self._require_f32(node, xt)
        params = [v.data if v is not None else None for v in node.inputs[1:5]]
        if any(p is None for p in params):
            raise CodegenError("BatchNormalization parameters must be constants")
        scale, bias, mean, var = (np.asarray(p, np.float64) for p in params)
        inv = scale / np.sqrt(var + float(node.attrs.get("epsilon", 1e-5)))
        shift = bias - mean * inv
        c = inv.size
        nhwc = node.attrs.get("layout") == "NHWC"
        bshape = (c,) if nhwc else (c,) + (1,) * (xt.rank - 2)
        shape = xt.shape
        out = self.out_operand(node.output)
        mul = Operand(
            f"((const float *)(W + {self.weights.add(inv.astype(np.float32))}))",
            "float",
            _broadcast_strides(bshape, shape),
        )
        add = Operand(
            f"((const float *)(W + {self.weights.add(shift.astype(np.float32))}))",
            "float",
            _broadcast_strides(bshape, shape),
        )
        src = self.operand(x, shape)
        self.loops(
            shape,
            out,
            [src, mul, add],
            [f"{out.ptr}[{{o}}] = {src.at('{i0}')} * {mul.at('{i1}')} + {add.at('{i2}')};"],
        )
        self.count("elementwise")

    # -- GEMM family ---------------------------------------------------------------------------
    def _offsets(self, values: list[int]) -> str:
        if not any(values):
            return "NULL"
        name = self.fresh("off")
        self.statics.append(
            f"static const long {name}[{len(values)}] = {{{', '.join(map(str, values))}}};"
        )
        return name

    def _gemm_call(
        self,
        m: int,
        n: int,
        k: int,
        a: str,
        b: str,
        c: str,
        ldc: int,
        batch: int = 1,
        offs: tuple[str, str, str, str] = ("NULL", "NULL", "NULL", "NULL"),
        alpha: float = 1.0,
        bias: str = "NULL",
        bias_mode: int = 0,
        res: str = "NULL",
        ldr: int = 0,
        beta: float = 1.0,
        act: str = "none",
        lo: float = 0.0,
        hi: float = 6.0,
    ) -> None:
        self.emit("{")
        self.emit("mgc_gemm_args g;")
        self.emit("memset(&g, 0, sizeof g);")
        self.emit(f"g.M = {m}; g.N = {n}; g.K = {k};")
        self.emit(a)
        self.emit(b)
        self.emit(f"g.c = {c}; g.ldc = {ldc}L;")
        self.emit(
            f"g.ep.alpha = {_literal(alpha, 'float32')}; g.ep.beta = {_literal(beta, 'float32')};"
        )
        self.emit(f"g.ep.bias = {bias}; g.ep.bias_mode = {bias_mode};")
        self.emit(f"g.ep.res = {res}; g.ep.ldr = {ldr}L;")
        self.emit(
            f"g.ep.act = {ACT_CODES[act]}; g.ep.lo = {_literal(lo, 'float32')}; "
            f"g.ep.hi = {_literal(hi, 'float32')};"
        )
        self.emit(f"mgc_gemm(&g, {batch}, {offs[0]}, {offs[1]}, {offs[2]}, {offs[3]});")
        self.emit("}")
        self.count("gemm")

    @staticmethod
    def _strided(field_: str, ptr: str, rs: int, cs: int) -> str:
        return (
            f"g.{field_}.kind = MGC_STRIDED; g.{field_}.p = {ptr}; "
            f"g.{field_}.rs = {rs}L; g.{field_}.cs = {cs}L;"
        )

    def _packed(self, matrix: np.ndarray, key: Any) -> str:
        off = self.weights.add(pack_b_panels(np.asarray(matrix, np.float32)), key=key)
        return f"g.b.kind = MGC_PACKED; g.b.p = (const float *)(W + {off});"

    def _act(self, node: Node) -> tuple[str, float, float]:
        return (
            str(node.attrs.get("activation", "none")),
            float(node.attrs.get("clip_min", 0.0)),
            float(node.attrs.get("clip_max", 6.0)),
        )

    def op_FusedGemm(self, node: Node) -> None:
        a, b = node.input(0), node.input(1)
        at, bt = a.tensor_type, b.tensor_type
        self._require_f32(node, at)
        trans = int(node.attrs.get("trans_b", 0))
        k, n = (bt.shape[1], bt.shape[0]) if trans else bt.shape
        m = at.size // max(k, 1)
        if b.data is not None:
            bmat = b.data.T if trans else b.data
            bspec = self._packed(bmat, ("packed", id(b), trans))
        else:
            bspec = self._strided("b", self.ptr(b), 1 if trans else n, k if trans else 1)
        bias_v = node.optional_input(2)
        res_v = node.optional_input(3)
        act, lo, hi = self._act(node)
        self._gemm_call(
            m,
            n,
            k,
            self._strided("a", self.ptr(a), k, 1),
            bspec,
            self.ptr(node.output),
            n,
            bias=self._vector_ptr(bias_v, n, "FusedGemm bias") if bias_v is not None else "NULL",
            bias_mode=1 if bias_v is not None else 0,
            res=self.ptr(res_v) if res_v is not None else "NULL",
            ldr=n,
            act=act,
            lo=lo,
            hi=hi,
        )

    def op_Gemm(self, node: Node) -> None:
        a, b, c = node.input(0), node.input(1), node.optional_input(2)
        at, bt = a.tensor_type, b.tensor_type
        self._require_f32(node, at)
        ta, tb = int(node.attrs.get("transA", 0)), int(node.attrs.get("transB", 0))
        m, k = (at.shape[1], at.shape[0]) if ta else at.shape
        n = bt.shape[0] if tb else bt.shape[1]
        alpha = float(node.attrs.get("alpha", 1.0))
        beta = float(node.attrs.get("beta", 1.0))
        aspec = self._strided("a", self.ptr(a), 1 if ta else k, m if ta else 1)
        if b.data is not None:
            bspec = self._packed(b.data.T if tb else b.data, ("packed", id(b), tb))
        else:
            bspec = self._strided("b", self.ptr(b), 1 if tb else n, k if tb else 1)
        bias, mode, res, ldr = "NULL", 0, "NULL", 0
        if c is not None:
            ct = c.tensor_type
            full = np.broadcast_shapes(ct.shape, (m, n))
            if c.data is not None and ct.size in (1, n) and (ct.rank < 2 or ct.shape[0] == 1):
                vec = np.broadcast_to(c.data.reshape(-1) * beta, (n,)).astype(np.float32)
                bias, mode = f"((const float *)(W + {self.weights.add(vec)}))", 1
            elif c.data is not None and ct.rank == 2 and ct.shape == (m, 1):
                vec = (c.data.reshape(-1) * beta).astype(np.float32)
                bias, mode = f"((const float *)(W + {self.weights.add(vec)}))", 2
            elif tuple(ct.shape) == (m, n) and full == (m, n):
                res, ldr = self.ptr(c), n
            else:
                raise CodegenError(f"Gemm bias of shape {ct.shape} is not supported")
        self._gemm_call(
            m,
            n,
            k,
            aspec,
            bspec,
            self.ptr(node.output),
            n,
            alpha=alpha,
            bias=bias,
            bias_mode=mode,
            res=res,
            ldr=ldr,
            beta=beta,
        )

    def op_MatMul(self, node: Node) -> None:
        a, b = node.input(0), node.input(1)
        at, bt = a.tensor_type, b.tensor_type
        self._require_f32(node, at)
        ashape = (1, *at.shape) if at.rank == 1 else at.shape
        bshape = (*bt.shape, 1) if bt.rank == 1 else bt.shape
        m, k = ashape[-2], ashape[-1]
        n = bshape[-1]
        batch_shape = tuple(int(d) for d in np.broadcast_shapes(ashape[:-2], bshape[:-2]))
        batch = int(np.prod(batch_shape)) if batch_shape else 1
        a_bs = _broadcast_strides(ashape[:-2], batch_shape)
        b_bs = _broadcast_strides(bshape[:-2], batch_shape)
        a_off: list[int] = []
        b_off: list[int] = []
        c_off: list[int] = []
        for idx in np.ndindex(*batch_shape) if batch_shape else [()]:
            a_off.append(sum(i * s for i, s in zip(idx, a_bs, strict=True)) * m * k)
            b_off.append(sum(i * s for i, s in zip(idx, b_bs, strict=True)) * k * n)
            c_off.append(len(c_off) * m * n)
        if b.data is not None and not any(b_off):
            bspec = self._packed(
                b.data.reshape(k, n) if bt.rank > 1 else b.data.reshape(k, 1), ("packed", id(b), 0)
            )
        else:
            bspec = self._strided("b", self.ptr(b), n, 1)
        offs = (self._offsets(a_off), self._offsets(b_off), self._offsets(c_off), "NULL")
        self._gemm_call(
            m,
            n,
            k,
            self._strided("a", self.ptr(a), k, 1),
            bspec,
            self.ptr(node.output),
            n,
            batch=batch,
            offs=offs,
        )

    # -- convolution and pooling ----------------------------------------------------------------
    @staticmethod
    def _depthwise(node: Node) -> bool:
        if node.op not in ("Conv", "FusedConv"):
            return False
        group = int(node.attrs.get("group", 1))
        w = node.input(1).tensor_type
        nhwc = node.attrs.get("layout") == "NHWC"
        cg, k = (w.shape[2], w.shape[3]) if nhwc else (w.shape[1], w.shape[0])
        return group > 1 and cg == 1 and k == group

    def op_Conv(self, node: Node) -> None:
        self.op_FusedConv(node)

    def op_FusedConv(self, node: Node) -> None:
        x, w = node.input(0), node.input(1)
        xt, wt = x.tensor_type, w.tensor_type
        self._require_f32(node, xt)
        nhwc = ops.conv_layout(node) == "NHWC"
        group = int(node.attrs.get("group", 1))
        if nhwc:
            nb, h, wd, c = xt.shape
            kh, kw, cg, kout = wt.shape
        else:
            nb, c, h, wd = xt.shape
            kout, cg, kh, kw = wt.shape
        g = ops.window_geometry(node, (h, wd), (kh, kw))
        oh, ow = g.out_hw
        pt, pl, _, _ = g.pads
        bias_v = node.optional_input(2)
        res_v = node.optional_input(3)
        act, lo, hi = self._act(node) if node.op == "FusedConv" else ("none", 0.0, 6.0)
        bias = self._vector_ptr(bias_v, kout, "Conv bias") if bias_v is not None else "NULL"
        res = self.ptr(res_v) if res_v is not None else "NULL"
        geo = (
            f"{g.strides[0]}, {g.strides[1]}, {g.dilations[0]}, {g.dilations[1]}, {pt}, {pl}, "
            f"{oh}, {ow}"
        )
        if self._depthwise(node):
            if w.data is None:
                raise CodegenError("depthwise convolution weights must be constant")
            fn = "mgc_depthwise_nhwc" if nhwc else "mgc_depthwise_nchw"
            self.emit(
                f"{fn}({nb}, {h}, {wd}, {c}, {kh}, {kw}, {geo}, {self.ptr(x)}, "
                f"{self.ptr(w)}, {bias}, {res}, {ACT_CODES[act]}, {_literal(lo, 'float32')}, "
                f"{_literal(hi, 'float32')}, {self.ptr(node.output)});"
            )
            self.count("depthwise")
            return
        if w.data is None:
            raise CodegenError("convolution weights must be constant for the C backend")
        kg = kout // group
        img_in = h * wd * c
        img_out = oh * ow * kout
        out_ptr = self.ptr(node.output)
        for gi in range(group):
            in_offs = [i * img_in for i in range(nb)]
            out_offs = [i * img_out for i in range(nb)]
            if nhwc:
                wmat = w.data[:, :, :, gi * kg : (gi + 1) * kg].reshape(kh * kw * cg, kg)
                bspec = self._packed(wmat, ("conv", id(w), gi))
                pointwise = kh == kw == 1 and g.strides == (1, 1) and g.pads == (0, 0, 0, 0)
                if pointwise:
                    aspec = self._strided("a", f"({self.ptr(x)} + {gi * cg})", c, 1)
                else:
                    aspec = (
                        f"g.a.kind = MGC_CONV_NHWC; g.a.p = {self.ptr(x)}; "
                        f"g.a.H = {h}; g.a.W = {wd}; "
                        f"g.a.C = {c}; g.a.KH = {kh}; g.a.KW = {kw}; g.a.SH = {g.strides[0]}; "
                        f"g.a.SW = {g.strides[1]}; "
                        f"g.a.DH = {g.dilations[0]}; g.a.DW = {g.dilations[1]}; "
                        f"g.a.PT = {pt}; g.a.PL = {pl}; g.a.OW = {ow}; g.a.c_off = {gi * cg}; "
                        f"g.a.cg = {cg};"
                    )
                offs = (
                    self._offsets(in_offs),
                    "NULL",
                    self._offsets(out_offs),
                    self._offsets(out_offs) if res_v is not None else "NULL",
                )
                self._gemm_call(
                    oh * ow,
                    kg,
                    kh * kw * cg,
                    aspec,
                    bspec,
                    f"({out_ptr} + {gi * kg})",
                    kout,
                    batch=nb,
                    offs=offs,
                    bias=f"({bias} + {gi * kg})" if bias != "NULL" else "NULL",
                    bias_mode=1 if bias != "NULL" else 0,
                    res=f"({res} + {gi * kg})" if res != "NULL" else "NULL",
                    ldr=kout,
                    act=act,
                    lo=lo,
                    hi=hi,
                )
            else:
                woff = self.weights.add(
                    np.ascontiguousarray(
                        w.data[gi * kg : (gi + 1) * kg].reshape(kg, cg * kh * kw), np.float32
                    ),
                    key=("convA", id(w), gi),
                )
                aspec = self._strided("a", f"((const float *)(W + {woff}))", cg * kh * kw, 1)
                bspec = (
                    f"g.b.kind = MGC_CONV_NCHW; g.b.p = {self.ptr(x)}; g.b.H = {h}; g.b.W = {wd}; "
                    f"g.b.C = {c}; g.b.KH = {kh}; g.b.KW = {kw}; g.b.SH = {g.strides[0]}; "
                    f"g.b.SW = {g.strides[1]}; "
                    f"g.b.DH = {g.dilations[0]}; g.b.DW = {g.dilations[1]}; "
                    f"g.b.PT = {pt}; g.b.PL = {pl}; g.b.OW = {ow}; "
                    f"g.b.c_off = {gi * cg}; g.b.cg = {cg};"
                )
                plane = oh * ow
                offs = (
                    "NULL",
                    self._offsets(in_offs),
                    self._offsets(out_offs),
                    self._offsets(out_offs) if res_v is not None else "NULL",
                )
                self._gemm_call(
                    kg,
                    plane,
                    cg * kh * kw,
                    aspec,
                    bspec,
                    f"({out_ptr} + {gi * kg * plane})",
                    plane,
                    batch=nb,
                    offs=offs,
                    bias=f"({bias} + {gi * kg})" if bias != "NULL" else "NULL",
                    bias_mode=2 if bias != "NULL" else 0,
                    res=f"({res} + {gi * kg * plane})" if res != "NULL" else "NULL",
                    ldr=plane,
                    act=act,
                    lo=lo,
                    hi=hi,
                )

    def _pool(self, node: Node, is_max: int) -> None:
        x = node.input(0)
        xt = x.tensor_type
        self._require_f32(node, xt)
        nhwc = ops.conv_layout(node) == "NHWC"
        nb, h, wd, c = xt.shape if nhwc else (xt.shape[0], xt.shape[2], xt.shape[3], xt.shape[1])
        kernel = tuple(int(k) for k in node.attrs["kernel_shape"])
        g = ops.window_geometry(
            node, (h, wd), (kernel[0], kernel[1]), bool(node.attrs.get("ceil_mode", 0))
        )
        pt, pl, pb, pr = g.pads
        oh, ow = g.out_hw
        include = int(node.attrs.get("count_include_pad", 0))
        fn = "mgc_pool_nhwc" if nhwc else "mgc_pool_nchw"
        self.emit(
            f"{fn}({is_max}, {nb}, {h}, {wd}, {c}, {kernel[0]}, {kernel[1]}, {g.strides[0]}, "
            f"{g.strides[1]}, {g.dilations[0]}, {g.dilations[1]}, {pt}, {pl}, {pb}, {pr}, "
            f"{oh}, {ow}, {include}, {self.ptr(x)}, {self.ptr(node.output)});"
        )
        self.count("pool")

    def op_MaxPool(self, node: Node) -> None:
        self._pool(node, 1)

    def op_AveragePool(self, node: Node) -> None:
        self._pool(node, 0)

    def _global_pool(self, node: Node, is_max: int) -> None:
        x = node.input(0)
        xt = x.tensor_type
        self._require_f32(node, xt)
        if ops.conv_layout(node) == "NHWC":
            outer, n, inner = xt.shape[0], int(np.prod(xt.shape[1:-1])), xt.shape[-1]
        else:
            outer, n, inner = xt.shape[0] * xt.shape[1], int(np.prod(xt.shape[2:])), 1
        self.emit(
            f"mgc_reduce_mid({is_max}, {outer}L, {n}L, {inner}L, {self.ptr(x)}, "
            f"{self.ptr(node.output)});"
        )
        self.count("reduce")

    def op_GlobalAveragePool(self, node: Node) -> None:
        self._global_pool(node, 0)

    def op_GlobalMaxPool(self, node: Node) -> None:
        self._global_pool(node, 1)

    # -- whole program --------------------------------------------------------------------------
    def generate(self) -> GeneratedModel:
        for k, node in enumerate(self.graph.nodes):
            if self.profile:
                self.emit("{ double mgc_t = mgc_now();")
            self.node(node)
            if self.profile:
                self.emit(f"mgc_prof[{k}] += mgc_now() - mgc_t; }}")
        for value, j in self.plan.copies:
            t = value.tensor_type
            self.emit(f"memcpy(outputs[{j}], {self.ptr(value)}, {t.nbytes});")
        weights = self.weights
        arena = max(self.plan.arena_bytes, ALIGN)
        head = [
            "/* Generated by ml-graph-compiler. */",
            '#include "mgc_runtime.h"',
            "",
            f"/* graph {comment_text(self.graph.name)}: {len(self.graph.nodes)} nodes, "
            f"arena {arena} bytes, "
            f"weights {weights.nbytes} bytes */",
            "",
            *self.statics,
            "",
            *(
                [
                    f"static double mgc_prof[{max(len(self.graph.nodes), 1)}];",
                    f"int mgc_profile_size(void) {{ return {len(self.graph.nodes)}; }}",
                    "const double *mgc_profile_data(void) { return mgc_prof; }",
                    "",
                ]
                if self.profile
                else []
            ),
            f"size_t mgc_arena_bytes(void) {{ return {arena}UL; }}",
            f"size_t mgc_weights_bytes(void) {{ return {weights.nbytes}UL; }}",
            "",
            "int mgc_run(const void *const *inputs, void *const *outputs, unsigned char *A,",
            "            const unsigned char *W) {",
            "(void)inputs; (void)outputs; (void)A; (void)W;",
        ]
        tail = ["return 0;", "}", ""]
        source = "\n".join(head + self.body + tail)
        return GeneratedModel(
            source=source,
            weights=weights,
            arena_bytes=arena,
            inputs=[(v.name, v.tensor_type) for v in self.graph.inputs],
            outputs=[(name, v.tensor_type) for name, v in self.graph.named_outputs()],
            plan=self.plan,
            lines=source.count("\n"),
            kernels=dict(sorted(self.kernels.items())),
        )


def generate_c(
    graph: Graph,
    strategy: Strategy = "greedy_by_size",
    inplace: bool = True,
    profile: bool = False,
) -> GeneratedModel:
    """Generate C source and the weights blob for a graph (nodes in execution order).

    With ``profile=True`` the code accumulates the wall time of every node, readable through
    ``mgc_profile_data``.
    """
    return CEmitter(graph, strategy, inplace, profile).generate()
