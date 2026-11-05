"""Operator registry: shape and type inference plus numpy reference semantics.

Each operator has an ``infer`` function (input types and any known constant inputs to output
types) and a ``run`` function (numpy arrays in, numpy arrays out). The reference interpreter,
constant folding and shape inference all use this one table, so an operator behaves the same in
every stage. Semantics follow the ONNX operator specification for the opsets that the importer
accepts (11 to 21). Operators in the ``mgc`` namespace (FusedConv, FusedGemm, Gelu) are produced
by the fusion passes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ml_graph_compiler.errors import ShapeError, UnsupportedOpError
from ml_graph_compiler.ir import Node, TensorType, Value, contiguous, dtype_name

Arrays = list[np.ndarray | None]
Consts = list[np.ndarray | None]
InferFn = Callable[[Node, list[TensorType | None], Consts], list[TensorType]]
RunFn = Callable[[Node, Arrays], list[np.ndarray]]


@dataclass(frozen=True)
class OpSpec:
    infer: InferFn
    run: RunFn
    elementwise: bool = False


REGISTRY: dict[str, OpSpec] = {}

# Operators whose output values are fully determined by the input *types* (no data needed).
SHAPE_ONLY_OPS = frozenset({"Shape", "Size"})

_NO_VALUE = Value("_")

ACTIVATIONS = ("none", "relu", "clip", "gelu", "gelu_tanh", "tanh", "sigmoid")


def register(name: str, infer: InferFn, elementwise: bool = False) -> Callable[[RunFn], RunFn]:
    def wrap(run: RunFn) -> RunFn:
        REGISTRY[name] = OpSpec(infer, run, elementwise)
        return run

    return wrap


def spec(op: str) -> OpSpec:
    try:
        return REGISTRY[op]
    except KeyError as exc:
        raise UnsupportedOpError(f"operator {op!r} is not supported") from exc


def supported_ops() -> list[str]:
    return sorted(REGISTRY)


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def _t(types: Sequence[TensorType | None], i: int) -> TensorType:
    if i >= len(types) or types[i] is None:
        raise ShapeError(f"missing type for input {i}")
    t = types[i]
    assert t is not None
    return t


def _const(consts: Consts, i: int, what: str) -> np.ndarray:
    if i >= len(consts) or consts[i] is None:
        raise ShapeError(f"{what} must be a constant (static shapes are required)")
    c = consts[i]
    assert c is not None
    return c


def _arr(arrays: Arrays, i: int) -> np.ndarray:
    a = arrays[i] if i < len(arrays) else None
    if a is None:
        raise UnsupportedOpError(f"missing input {i}")
    return a


def _opt(arrays: Arrays, i: int) -> np.ndarray | None:
    return arrays[i] if i < len(arrays) else None


def _axis(axis: int, rank: int) -> int:
    a = axis + rank if axis < 0 else axis
    if not 0 <= a < max(rank, 1):
        raise ShapeError(f"axis {axis} out of range for rank {rank}")
    return a


def _ints(values: Any) -> list[int]:
    return [int(v) for v in np.asarray(values).reshape(-1)]


def _same(node: Node, types: list[TensorType | None], consts: Consts) -> list[TensorType]:
    return [_t(types, 0)]


def _broadcast(node: Node, types: list[TensorType | None], consts: Consts) -> list[TensorType]:
    present = [t for t in types if t is not None]
    try:
        shape = np.broadcast_shapes(*(t.shape for t in present))
    except ValueError as exc:
        raise ShapeError(
            f"{node.op}: shapes do not broadcast: {[t.shape for t in present]}"
        ) from exc
    return [TensorType(present[0].dtype, tuple(int(d) for d in shape))]


def _compare(node: Node, types: list[TensorType | None], consts: Consts) -> list[TensorType]:
    return [TensorType("bool", _broadcast(node, types, consts)[0].shape)]


def _keep(x: np.ndarray, like: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=like.dtype)


# ------------------------------------------------------------------------------------------
# elementwise
# ------------------------------------------------------------------------------------------
def _int_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(np.trunc(a / b), dtype=a.dtype)


_BINARY: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "Add": np.add,
    "Sub": np.subtract,
    "Mul": np.multiply,
    "Max": np.maximum,
    "Min": np.minimum,
}

for _name, _fn in _BINARY.items():

    def _make(fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> RunFn:
        def run(node: Node, xs: Arrays) -> list[np.ndarray]:
            a, b = _arr(xs, 0), _arr(xs, 1)
            return [_keep(fn(a, b), a)]

        return run

    register(_name, _broadcast, elementwise=True)(_make(_fn))


@register("Div", _broadcast, elementwise=True)
def _div(node: Node, xs: Arrays) -> list[np.ndarray]:
    a, b = _arr(xs, 0), _arr(xs, 1)
    if np.issubdtype(a.dtype, np.integer):
        return [_int_div(a, b)]
    return [_keep(a / b, a)]


@register("Pow", _broadcast, elementwise=True)
def _pow(node: Node, xs: Arrays) -> list[np.ndarray]:
    a, b = _arr(xs, 0), _arr(xs, 1)
    return [_keep(np.power(a, b.astype(a.dtype)), a)]


_COMPARE: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "Equal": np.equal,
    "Less": np.less,
    "Greater": np.greater,
    "LessOrEqual": np.less_equal,
    "GreaterOrEqual": np.greater_equal,
    "And": np.logical_and,
    "Or": np.logical_or,
    "Xor": np.logical_xor,
}
for _name, _fn in _COMPARE.items():

    def _make_cmp(fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> RunFn:
        def run(node: Node, xs: Arrays) -> list[np.ndarray]:
            return [np.asarray(fn(_arr(xs, 0), _arr(xs, 1)), dtype=np.bool_)]

        return run

    register(_name, _compare, elementwise=True)(_make_cmp(_fn))


def _where_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    return [TensorType(_t(t, 1).dtype, _broadcast(n, t, c)[0].shape)]


@register("Where", _where_infer, elementwise=True)
def _where(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 1)
    return [_keep(np.where(_arr(xs, 0), x, _arr(xs, 2)), x)]


_ERF_ALPHA = (
    -2.72614225801306e-10,
    2.77068142495902e-08,
    -2.10102402082508e-06,
    -5.69250639462346e-05,
    -7.34990630326855e-04,
    -2.95459980854025e-03,
    -1.60960333262415e-02,
)
_ERF_BETA = (
    -1.45660718464996e-05,
    -2.13374055278905e-04,
    -1.68282697438203e-03,
    -7.37332916720468e-03,
    -1.42647390514189e-02,
)


def erf(x: np.ndarray) -> np.ndarray:
    """Vectorized error function (rational approximation, about 1e-7 absolute error)."""
    x64 = np.clip(np.asarray(x, dtype=np.float64), -4.0, 4.0)
    x2 = x64 * x64
    p = np.full_like(x64, _ERF_ALPHA[0])
    for c in _ERF_ALPHA[1:]:
        p = p * x2 + c
    p = p * x64
    q = np.full_like(x64, _ERF_BETA[0])
    for c in _ERF_BETA[1:]:
        q = q * x2 + c
    return np.asarray(p / q, dtype=np.asarray(x).dtype)


def gelu(x: np.ndarray, approximate: str = "none") -> np.ndarray:
    if approximate == "tanh":
        c = math.sqrt(2.0 / math.pi)
        return _keep(0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x))), x)
    return _keep(0.5 * x * (1.0 + erf(x / math.sqrt(2.0))), x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return _keep(0.5 * (1.0 + np.tanh(0.5 * x)), x)


_UNARY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Relu": lambda x: np.maximum(x, 0),
    "Sigmoid": sigmoid,
    "Tanh": np.tanh,
    "Erf": erf,
    "Sqrt": np.sqrt,
    "Exp": np.exp,
    "Log": np.log,
    "Neg": np.negative,
    "Abs": np.abs,
    "Reciprocal": np.reciprocal,
    "Floor": np.floor,
    "Ceil": np.ceil,
    "Sin": np.sin,
    "Cos": np.cos,
    "Softplus": lambda x: np.logaddexp(0, x),
    "Identity": lambda x: x.copy(),
}
for _name, _ufn in _UNARY.items():

    def _make_unary(fn: Callable[[np.ndarray], np.ndarray]) -> RunFn:
        def run(node: Node, xs: Arrays) -> list[np.ndarray]:
            x = _arr(xs, 0)
            return [_keep(fn(x), x)]

        return run

    register(_name, _same, elementwise=True)(_make_unary(_ufn))


@register("Not", _same, elementwise=True)
def _not(node: Node, xs: Arrays) -> list[np.ndarray]:
    return [np.logical_not(_arr(xs, 0))]


@register("Gelu", _same, elementwise=True)
def _gelu(node: Node, xs: Arrays) -> list[np.ndarray]:
    return [gelu(_arr(xs, 0), str(node.attrs.get("approximate", "none")))]


@register("LeakyRelu", _same, elementwise=True)
def _leaky(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    alpha = float(node.attrs.get("alpha", 0.01))
    return [_keep(np.where(x >= 0, x, alpha * x), x)]


@register("HardSigmoid", _same, elementwise=True)
def _hard_sigmoid(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    alpha, beta = float(node.attrs.get("alpha", 0.2)), float(node.attrs.get("beta", 0.5))
    return [_keep(np.clip(alpha * x + beta, 0, 1), x)]


def clip_bounds(node: Node, xs: Sequence[np.ndarray | None]) -> tuple[float, float]:
    """Clip bounds from attributes (opset < 11) or optional inputs (opset >= 11)."""
    lo = node.attrs.get("min", -np.inf)
    hi = node.attrs.get("max", np.inf)
    if len(xs) > 1 and xs[1] is not None:
        lo = float(np.asarray(xs[1]).reshape(()))
    if len(xs) > 2 and xs[2] is not None:
        hi = float(np.asarray(xs[2]).reshape(()))
    return float(lo), float(hi)


@register("Clip", _same, elementwise=True)
def _clip(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    lo, hi = clip_bounds(node, xs)
    return [_keep(np.clip(x, lo, hi), x)]


_CAST_TO = {
    1: "float32",
    6: "int32",
    7: "int64",
    9: "bool",
    11: "float64",
    10: "float16",
    2: "uint8",
    3: "int8",
}


def cast_dtype(node: Node) -> str:
    to = node.attrs.get("to", 1)
    if isinstance(to, str):
        return to
    if int(to) not in _CAST_TO:
        raise UnsupportedOpError(f"Cast to ONNX type {to} is not supported")
    return _CAST_TO[int(to)]


@register("Cast", lambda n, t, c: [TensorType(cast_dtype(n), _t(t, 0).shape)], elementwise=True)
def _cast(node: Node, xs: Arrays) -> list[np.ndarray]:
    return [_arr(xs, 0).astype(cast_dtype(node))]


def _dropout_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    x = _t(t, 0)
    outs = [x]
    if len(n.outputs) > 1:
        outs.append(TensorType("bool", x.shape))
    return outs


@register("Dropout", _dropout_infer)
def _dropout(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    outs = [x.copy()]
    if len(node.outputs) > 1:
        outs.append(np.ones(x.shape, dtype=np.bool_))
    return outs


# ------------------------------------------------------------------------------------------
# shape manipulation
# ------------------------------------------------------------------------------------------
def reshape_target(
    shape_in: tuple[int, ...], spec_: Sequence[int], allowzero: int = 0
) -> tuple[int, ...]:
    out = list(spec_)
    for i, d in enumerate(out):
        if d == 0 and not allowzero:
            if i >= len(shape_in):
                raise ShapeError("Reshape copies a dimension that does not exist")
            out[i] = shape_in[i]
    total = int(np.prod(shape_in, dtype=np.int64))
    if out.count(-1) > 1:
        raise ShapeError("Reshape has more than one -1")
    if -1 in out:
        known = int(np.prod([d for d in out if d != -1], dtype=np.int64))
        if known == 0 or total % known:
            raise ShapeError(f"cannot reshape {shape_in} to {list(spec_)}")
        out[out.index(-1)] = total // known
    if int(np.prod(out, dtype=np.int64)) != total:
        raise ShapeError(f"cannot reshape {shape_in} to {list(spec_)}")
    return tuple(int(d) for d in out)


def _reshape_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    x = _t(t, 0)
    target = list(n.attrs["shape"]) if "shape" in n.attrs else _ints(_const(c, 1, "Reshape shape"))
    return [TensorType(x.dtype, reshape_target(x.shape, target, int(n.attrs.get("allowzero", 0))))]


@register("Reshape", _reshape_infer)
def _reshape(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    target = list(node.attrs["shape"]) if "shape" in node.attrs else _ints(_arr(xs, 1))
    return [x.reshape(reshape_target(x.shape, target, int(node.attrs.get("allowzero", 0)))).copy()]


def _flatten_shape(shape: tuple[int, ...], axis: int) -> tuple[int, int]:
    a = axis + len(shape) if axis < 0 else axis
    return int(np.prod(shape[:a], dtype=np.int64)), int(np.prod(shape[a:], dtype=np.int64))


@register(
    "Flatten",
    lambda n, t, c: [
        TensorType(_t(t, 0).dtype, _flatten_shape(_t(t, 0).shape, int(n.attrs.get("axis", 1))))
    ],
)
def _flatten(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    return [x.reshape(_flatten_shape(x.shape, int(node.attrs.get("axis", 1)))).copy()]


def transpose_perm(node: Node, rank: int) -> list[int]:
    perm = node.attrs.get("perm")
    return list(range(rank))[::-1] if perm is None else [int(p) for p in perm]


@register(
    "Transpose",
    lambda n, t, c: [
        TensorType(
            _t(t, 0).dtype,
            tuple(_t(t, 0).shape[p] for p in transpose_perm(n, _t(t, 0).rank)),
        )
    ],
)
def _transpose(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    return [contiguous(x.transpose(transpose_perm(node, x.ndim)))]


def _axes_arg(node: Node, consts: Sequence[np.ndarray | None], index: int) -> list[int] | None:
    if "axes" in node.attrs:
        return [int(a) for a in node.attrs["axes"]]
    if len(consts) > index and consts[index] is not None:
        return _ints(consts[index])
    if len(node.inputs) > index and node.inputs[index] is not None:
        raise ShapeError(f"{node.op} axes must be a constant")
    return None


def squeeze_shape(shape: tuple[int, ...], axes: list[int] | None) -> tuple[int, ...]:
    if axes is None:
        return tuple(d for d in shape if d != 1)
    norm = {_axis(a, len(shape)) for a in axes}
    for a in norm:
        if shape[a] != 1:
            raise ShapeError(f"cannot squeeze axis {a} of size {shape[a]}")
    return tuple(d for i, d in enumerate(shape) if i not in norm)


def unsqueeze_shape(shape: tuple[int, ...], axes: list[int]) -> tuple[int, ...]:
    rank = len(shape) + len(axes)
    norm = sorted(_axis(a, rank) for a in axes)
    out = list(shape)
    for a in norm:
        out.insert(a, 1)
    return tuple(out)


@register(
    "Squeeze",
    lambda n, t, c: [TensorType(_t(t, 0).dtype, squeeze_shape(_t(t, 0).shape, _axes_arg(n, c, 1)))],
)
def _squeeze(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    return [x.reshape(squeeze_shape(x.shape, _axes_arg(node, xs, 1))).copy()]


def _unsqueeze_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    axes = _axes_arg(n, c, 1)
    if axes is None:
        raise ShapeError("Unsqueeze needs axes")
    return [TensorType(_t(t, 0).dtype, unsqueeze_shape(_t(t, 0).shape, axes))]


@register("Unsqueeze", _unsqueeze_infer)
def _unsqueeze(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    axes = _axes_arg(node, xs, 1)
    assert axes is not None
    return [x.reshape(unsqueeze_shape(x.shape, axes)).copy()]


def _concat_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    present = [x for x in t if x is not None]
    first = present[0]
    axis = _axis(int(n.attrs.get("axis", 0)), first.rank)
    shape = list(first.shape)
    shape[axis] = sum(x.shape[axis] for x in present)
    for x in present:
        if x.rank != first.rank or any(
            x.shape[i] != first.shape[i] for i in range(first.rank) if i != axis
        ):
            raise ShapeError(f"Concat shapes differ: {[p.shape for p in present]}")
    return [TensorType(first.dtype, tuple(shape))]


@register("Concat", _concat_infer)
def _concat(node: Node, xs: Arrays) -> list[np.ndarray]:
    arrays = [x for x in xs if x is not None]
    axis = int(node.attrs.get("axis", 0))
    return [np.concatenate(arrays, axis=axis)]


def split_sizes(
    node: Node, shape: tuple[int, ...], consts: Sequence[np.ndarray | None]
) -> list[int]:
    axis = _axis(int(node.attrs.get("axis", 0)), len(shape))
    if "split" in node.attrs:
        return [int(s) for s in node.attrs["split"]]
    if len(consts) > 1 and consts[1] is not None:
        return _ints(consts[1])
    n = len(node.outputs)
    if "num_outputs" in node.attrs:
        n = int(node.attrs["num_outputs"])
        size = -(-shape[axis] // n)
        sizes = [size] * (n - 1)
        return [*sizes, shape[axis] - size * (n - 1)]
    if shape[axis] % n:
        raise ShapeError("Split cannot divide the axis evenly")
    return [shape[axis] // n] * n


def _split_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    x = _t(t, 0)
    axis = _axis(int(n.attrs.get("axis", 0)), x.rank)
    out = []
    for s in split_sizes(n, x.shape, c):
        shape = list(x.shape)
        shape[axis] = s
        out.append(TensorType(x.dtype, tuple(shape)))
    return out


@register("Split", _split_infer)
def _split(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    axis = _axis(int(node.attrs.get("axis", 0)), x.ndim)
    sizes = split_sizes(node, x.shape, xs)
    bounds = np.cumsum(sizes)[:-1]
    return [p.copy() for p in np.split(x, bounds, axis=axis)]


def slice_params(
    node: Node, shape: tuple[int, ...], consts: Sequence[np.ndarray | None]
) -> list[slice]:
    """Normalized python slices for every axis (ONNX Slice semantics, opset 1 and 10+)."""
    if "starts" in node.attrs:
        starts = [int(v) for v in node.attrs["starts"]]
        ends = [int(v) for v in node.attrs["ends"]]
        axes = [int(v) for v in node.attrs.get("axes", range(len(starts)))]
        steps = [1] * len(starts)
    else:
        starts = _ints(_const(list(consts), 1, "Slice starts"))
        ends = _ints(_const(list(consts), 2, "Slice ends"))
        axes = (
            _ints(consts[3])
            if len(consts) > 3 and consts[3] is not None
            else list(range(len(starts)))
        )
        steps = _ints(consts[4]) if len(consts) > 4 and consts[4] is not None else [1] * len(starts)
    slices = [slice(None)] * len(shape)
    for s, e, a, st in zip(starts, ends, axes, steps, strict=True):
        ax = _axis(a, len(shape))
        if st == 0:
            raise ShapeError("Slice step cannot be 0")
        dim = shape[ax]
        # Clamp like ONNX: huge sentinel values mean "to the end".
        s = max(-dim - 1, min(dim, s))
        e = max(-dim - 1, min(dim, e))
        start, stop, step = slice(s, e, st).indices(dim)
        slices[ax] = slice(start, stop if stop >= 0 else None, step)
    return slices


def _slice_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    x = _t(t, 0)
    slices = slice_params(n, x.shape, c)
    shape = tuple(len(range(*s.indices(d))) for s, d in zip(slices, x.shape, strict=True))
    return [TensorType(x.dtype, shape)]


@register("Slice", _slice_infer)
def _slice(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    return [contiguous(x[tuple(slice_params(node, x.shape, xs))])]


def _gather_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    x, idx = _t(t, 0), _t(t, 1)
    axis = _axis(int(n.attrs.get("axis", 0)), x.rank)
    return [TensorType(x.dtype, x.shape[:axis] + idx.shape + x.shape[axis + 1 :])]


@register("Gather", _gather_infer)
def _gather(node: Node, xs: Arrays) -> list[np.ndarray]:
    x, idx = _arr(xs, 0), _arr(xs, 1)
    axis = _axis(int(node.attrs.get("axis", 0)), x.ndim)
    idx = np.where(idx < 0, idx + x.shape[axis], idx)
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= x.shape[axis]):
        raise ShapeError(f"Gather index out of range for axis {axis} of size {x.shape[axis]}")
    return [np.take(x, idx, axis=axis)]


@register("Shape", lambda n, t, c: [TensorType("int64", (_shape_len(n, _t(t, 0)),))])
def _shape(node: Node, xs: Arrays) -> list[np.ndarray]:
    return [shape_value(node, _arr(xs, 0).shape)]


def _shape_len(node: Node, x: TensorType) -> int:
    return len(shape_value(node, x.shape))


def shape_value(node: Node, shape: tuple[int, ...]) -> np.ndarray:
    start = int(node.attrs.get("start", 0))
    end = node.attrs.get("end")
    dims = shape[start:] if end is None else shape[start : int(end)]
    return np.asarray(dims, dtype=np.int64)


@register("Size", lambda n, t, c: [TensorType("int64", ())])
def _size(node: Node, xs: Arrays) -> list[np.ndarray]:
    return [np.asarray(_arr(xs, 0).size, dtype=np.int64)]


def _cos_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    shape = tuple(_ints(_const(c, 0, "ConstantOfShape shape")))
    value = n.attrs.get("value")
    dtype = dtype_name(value.dtype) if isinstance(value, np.ndarray) else "float32"
    return [TensorType(dtype, shape)]


@register("ConstantOfShape", _cos_infer)
def _constant_of_shape(node: Node, xs: Arrays) -> list[np.ndarray]:
    value = node.attrs.get("value")
    fill = np.asarray(value).reshape(-1)[0] if value is not None else np.float32(0)
    dtype = value.dtype if isinstance(value, np.ndarray) else np.float32
    return [np.full(tuple(_ints(_arr(xs, 0))), fill, dtype=dtype)]


def _expand_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    x = _t(t, 0)
    target = tuple(_ints(_const(c, 1, "Expand shape")))
    return [TensorType(x.dtype, tuple(int(d) for d in np.broadcast_shapes(x.shape, target)))]


@register("Expand", _expand_infer)
def _expand(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    shape = np.broadcast_shapes(x.shape, tuple(_ints(_arr(xs, 1))))
    return [contiguous(np.broadcast_to(x, shape))]


def _tile_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    x = _t(t, 0)
    reps = _ints(_const(c, 1, "Tile repeats"))
    return [TensorType(x.dtype, tuple(d * r for d, r in zip(x.shape, reps, strict=True)))]


@register("Tile", _tile_infer)
def _tile(node: Node, xs: Arrays) -> list[np.ndarray]:
    return [np.tile(_arr(xs, 0), _ints(_arr(xs, 1)))]


def _range_values(start: np.ndarray, limit: np.ndarray, delta: np.ndarray) -> np.ndarray:
    s, lim, d = (np.asarray(v).reshape(()) for v in (start, limit, delta))
    n = max(math.ceil((float(lim) - float(s)) / float(d)), 0)
    return np.asarray(s + np.arange(n, dtype=s.dtype) * d, dtype=s.dtype)


def _range_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    values = _range_values(_const(c, 0, "Range"), _const(c, 1, "Range"), _const(c, 2, "Range"))
    return [TensorType(_t(t, 0).dtype, values.shape)]


@register("Range", _range_infer)
def _range(node: Node, xs: Arrays) -> list[np.ndarray]:
    return [_range_values(_arr(xs, 0), _arr(xs, 1), _arr(xs, 2))]


@register("Trilu", _same)
def _trilu(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    k = int(np.asarray(xs[1]).reshape(())) if len(xs) > 1 and xs[1] is not None else 0
    return [np.triu(x, k) if int(node.attrs.get("upper", 1)) else np.tril(x, k)]


def pad_amounts(node: Node, rank: int, consts: Sequence[np.ndarray | None]) -> list[int]:
    if "pads" in node.attrs:
        pads = [int(p) for p in node.attrs["pads"]]
    else:
        pads = _ints(_const(list(consts), 1, "Pad pads"))
    if len(consts) > 3 and consts[3] is not None:
        axes = [_axis(a, rank) for a in _ints(consts[3])]
        full = [0] * (2 * rank)
        for i, a in enumerate(axes):
            full[a], full[a + rank] = pads[i], pads[i + len(axes)]
        pads = full
    return pads


def _pad_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    x = _t(t, 0)
    pads = pad_amounts(n, x.rank, c)
    return [
        TensorType(x.dtype, tuple(d + pads[i] + pads[i + x.rank] for i, d in enumerate(x.shape)))
    ]


@register("Pad", _pad_infer)
def _pad(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    mode = node.attrs.get("mode", "constant")
    if mode != "constant":
        raise UnsupportedOpError(f"Pad mode {mode!r} is not supported")
    pads = pad_amounts(node, x.ndim, xs)
    value = float(np.asarray(xs[2]).reshape(())) if len(xs) > 2 and xs[2] is not None else 0.0
    width = [(pads[i], pads[i + x.ndim]) for i in range(x.ndim)]
    return [np.pad(x, width, constant_values=value)]


# ------------------------------------------------------------------------------------------
# reductions and normalization
# ------------------------------------------------------------------------------------------
def reduce_axes(node: Node, rank: int, consts: Sequence[np.ndarray | None]) -> tuple[int, ...]:
    axes = _axes_arg(node, consts, 1)
    if not axes:
        if int(node.attrs.get("noop_with_empty_axes", 0)):
            return ()
        return tuple(range(rank))
    return tuple(sorted(_axis(a, rank) for a in axes))


def _reduce_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    x = _t(t, 0)
    axes = reduce_axes(n, x.rank, c)
    keep = int(n.attrs.get("keepdims", 1))
    shape = tuple(1 if i in axes else d for i, d in enumerate(x.shape) if keep or i not in axes)
    return [TensorType(x.dtype, shape)]


_REDUCE: dict[str, Callable[..., np.ndarray]] = {
    "ReduceMean": np.mean,
    "ReduceSum": np.sum,
    "ReduceMax": np.max,
    "ReduceMin": np.min,
    "ReduceProd": np.prod,
}
for _name, _rfn in _REDUCE.items():

    def _make_reduce(fn: Callable[..., np.ndarray]) -> RunFn:
        def run(node: Node, xs: Arrays) -> list[np.ndarray]:
            x = _arr(xs, 0)
            axes = reduce_axes(node, x.ndim, xs)
            if not axes:
                return [x.copy()]
            keep = bool(int(node.attrs.get("keepdims", 1)))
            return [_keep(fn(x, axis=axes, keepdims=keep), x)]

        return run

    register(_name, _reduce_infer)(_make_reduce(_rfn))


def _argmax_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    x = _t(t, 0)
    axis = _axis(int(n.attrs.get("axis", 0)), x.rank)
    keep = int(n.attrs.get("keepdims", 1))
    shape = tuple(1 if i == axis else d for i, d in enumerate(x.shape) if keep or i != axis)
    return [TensorType("int64", shape)]


@register("ArgMax", _argmax_infer)
def _argmax(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    axis = _axis(int(node.attrs.get("axis", 0)), x.ndim)
    out = np.argmax(x, axis=axis)
    if int(node.attrs.get("keepdims", 1)):
        out = np.expand_dims(out, axis)
    return [out.astype(np.int64)]


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return _keep(e / np.sum(e, axis=axis, keepdims=True), x)


@register("Softmax", _same)
def _softmax(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    return [softmax(x, _axis(int(node.attrs.get("axis", -1)), x.ndim))]


@register("LogSoftmax", _same)
def _log_softmax(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    axis = _axis(int(node.attrs.get("axis", -1)), x.ndim)
    m = np.max(x, axis=axis, keepdims=True)
    return [_keep(x - m - np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)), x)]


def layer_norm(
    x: np.ndarray, scale: np.ndarray | None, bias: np.ndarray | None, axis: int, eps: float
) -> np.ndarray:
    axes = tuple(range(axis, x.ndim))
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=axes, keepdims=True)
    var = ((x64 - mean) ** 2).mean(axis=axes, keepdims=True)
    y = (x64 - mean) / np.sqrt(var + eps)
    if scale is not None:
        y = y * scale
    if bias is not None:
        y = y + bias
    return _keep(y, x)


def _ln_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    if len(n.outputs) > 1:
        raise UnsupportedOpError("LayerNormalization mean and inverse std outputs")
    return [_t(t, 0)]


@register("LayerNormalization", _ln_infer)
def _layer_norm(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    axis = _axis(int(node.attrs.get("axis", -1)), x.ndim)
    return [layer_norm(x, _opt(xs, 1), _opt(xs, 2), axis, float(node.attrs.get("epsilon", 1e-5)))]


@register("BatchNormalization", lambda n, t, c: [_t(t, 0)])
def _batch_norm(node: Node, xs: Arrays) -> list[np.ndarray]:
    x, scale, bias, mean, var = (_arr(xs, i) for i in range(5))
    eps = float(node.attrs.get("epsilon", 1e-5))
    channel_axis = x.ndim - 1 if node.attrs.get("layout") == "NHWC" else 1
    shape = [1] * x.ndim
    shape[channel_axis] = -1
    inv = (scale / np.sqrt(var + eps)).reshape(shape)
    return [_keep((x - mean.reshape(shape)) * inv + bias.reshape(shape), x)]


# ------------------------------------------------------------------------------------------
# matrix products
# ------------------------------------------------------------------------------------------
def matmul_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    if not a or not b:
        raise ShapeError("MatMul operands must have rank >= 1")
    a2 = (1, *a) if len(a) == 1 else a
    b2 = (*b, 1) if len(b) == 1 else b
    if a2[-1] != b2[-2]:
        raise ShapeError(f"MatMul inner dimensions differ: {a} x {b}")
    batch = np.broadcast_shapes(a2[:-2], b2[:-2])
    out = [*batch, a2[-2], b2[-1]]
    if len(a) == 1:
        out.pop(-2)
    if len(b) == 1:
        out.pop(-1)
    return tuple(int(d) for d in out)


@register(
    "MatMul",
    lambda n, t, c: [TensorType(_t(t, 0).dtype, matmul_shape(_t(t, 0).shape, _t(t, 1).shape))],
)
def _matmul(node: Node, xs: Arrays) -> list[np.ndarray]:
    a, b = _arr(xs, 0), _arr(xs, 1)
    return [_keep(np.matmul(a, b), a)]


def _gemm_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    a, b = _t(t, 0), _t(t, 1)
    m = a.shape[1] if int(n.attrs.get("transA", 0)) else a.shape[0]
    k = a.shape[0] if int(n.attrs.get("transA", 0)) else a.shape[1]
    kb, nn = (b.shape[1], b.shape[0]) if int(n.attrs.get("transB", 0)) else b.shape
    if k != kb:
        raise ShapeError(f"Gemm inner dimensions differ: {a.shape} x {b.shape}")
    return [TensorType(a.dtype, (m, nn))]


@register("Gemm", _gemm_infer)
def _gemm(node: Node, xs: Arrays) -> list[np.ndarray]:
    a, b, c = _arr(xs, 0), _arr(xs, 1), _opt(xs, 2)
    if int(node.attrs.get("transA", 0)):
        a = a.T
    if int(node.attrs.get("transB", 0)):
        b = b.T
    y = float(node.attrs.get("alpha", 1.0)) * (a @ b)
    if c is not None:
        y = y + float(node.attrs.get("beta", 1.0)) * c
    return [_keep(y, a)]


def apply_activation(y: np.ndarray, node: Node) -> np.ndarray:
    act = str(node.attrs.get("activation", "none"))
    if act == "none":
        return y
    if act == "relu":
        return np.asarray(np.maximum(y, 0), dtype=y.dtype)
    if act == "clip":
        lo, hi = float(node.attrs.get("clip_min", 0.0)), float(node.attrs.get("clip_max", 6.0))
        return np.asarray(np.clip(y, lo, hi), dtype=y.dtype)
    if act == "gelu":
        return gelu(y, "none")
    if act == "gelu_tanh":
        return gelu(y, "tanh")
    if act == "tanh":
        return np.tanh(y)
    if act == "sigmoid":
        return sigmoid(y)
    raise UnsupportedOpError(f"activation {act!r}")


def _fused_gemm_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    a, b = _t(t, 0), _t(t, 1)
    k, nn = (b.shape[1], b.shape[0]) if int(n.attrs.get("trans_b", 0)) else b.shape
    if b.rank != 2 or a.shape[-1] != k:
        raise ShapeError(f"FusedGemm shapes: {a.shape} x {b.shape}")
    return [TensorType(a.dtype, (*a.shape[:-1], nn))]


@register("FusedGemm", _fused_gemm_infer)
def _fused_gemm(node: Node, xs: Arrays) -> list[np.ndarray]:
    """``act(A @ B + bias + residual)`` with A of any rank >= 1 and a 2-D B."""
    a, b, bias = _arr(xs, 0), _arr(xs, 1), _opt(xs, 2)
    residual = _opt(xs, 3)
    if int(node.attrs.get("trans_b", 0)):
        b = b.T
    y = a @ b
    if bias is not None:
        y = y + bias
    if residual is not None:
        y = y + residual
    return [_keep(apply_activation(y, node), a)]


# ------------------------------------------------------------------------------------------
# convolution and pooling
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Conv2dGeometry:
    """Normalized 2-D window geometry; pads are (top, left, bottom, right)."""

    kernel: tuple[int, int]
    strides: tuple[int, int]
    dilations: tuple[int, int]
    pads: tuple[int, int, int, int]
    out_hw: tuple[int, int]


def window_geometry(
    node: Node, in_hw: tuple[int, int], kernel: tuple[int, int], ceil_mode: bool = False
) -> Conv2dGeometry:
    strides = tuple(int(s) for s in node.attrs.get("strides", (1, 1)))
    dilations = tuple(int(d) for d in node.attrs.get("dilations", (1, 1)))
    pads = [int(p) for p in node.attrs.get("pads", (0, 0, 0, 0))]
    if len(strides) != 2 or len(dilations) != 2 or len(pads) != 4:
        raise UnsupportedOpError(f"{node.op}: only 2-D windows are supported")
    auto_pad = node.attrs.get("auto_pad", "NOTSET")
    eff = [(kernel[i] - 1) * dilations[i] + 1 for i in range(2)]
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        out = [-(-in_hw[i] // strides[i]) for i in range(2)]
        total = [max((out[i] - 1) * strides[i] + eff[i] - in_hw[i], 0) for i in range(2)]
        small = [t // 2 for t in total]
        big = [t - t // 2 for t in total]
        if auto_pad == "SAME_UPPER":
            pads = [small[0], small[1], big[0], big[1]]
        else:
            pads = [big[0], big[1], small[0], small[1]]
    elif auto_pad == "VALID":
        pads = [0, 0, 0, 0]
    out_hw = []
    for i in range(2):
        span = in_hw[i] + pads[i] + pads[i + 2] - eff[i]
        if span < 0:
            raise ShapeError(f"{node.op}: window larger than the padded input")
        o = (-(-span // strides[i]) if ceil_mode else span // strides[i]) + 1
        if ceil_mode and (o - 1) * strides[i] >= in_hw[i] + pads[i]:
            o -= 1
        out_hw.append(o)
    return Conv2dGeometry(
        (kernel[0], kernel[1]),
        (strides[0], strides[1]),
        (dilations[0], dilations[1]),
        (pads[0], pads[1], pads[2], pads[3]),
        (out_hw[0], out_hw[1]),
    )


def conv_layout(node: Node) -> str:
    layout = str(node.attrs.get("layout", "NCHW"))
    if layout not in ("NCHW", "NHWC"):
        raise UnsupportedOpError(f"layout {layout!r}")
    return layout


def _conv_dims(node: Node, x: TensorType, w: TensorType) -> tuple[int, int, int, Conv2dGeometry]:
    """(batch, out_channels, in_channels, geometry) for Conv and FusedConv."""
    if x.rank != 4 or w.rank != 4:
        raise UnsupportedOpError(f"{node.op}: only 2-D convolution (rank 4) is supported")
    group = int(node.attrs.get("group", 1))
    if conv_layout(node) == "NHWC":
        n, h, wd, c = x.shape
        kh, kw, cg, k = w.shape
    else:
        n, c, h, wd = x.shape
        k, cg, kh, kw = w.shape
    if cg * group != c or k % group:
        raise ShapeError(
            f"{node.op}: channels {c} do not match weights {w.shape} with group {group}"
        )
    return n, k, c, window_geometry(node, (h, wd), (kh, kw))


def _conv_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    x, w = _t(t, 0), _t(t, 1)
    batch, k, _, g = _conv_dims(n, x, w)
    oh, ow = g.out_hw
    shape = (batch, oh, ow, k) if conv_layout(n) == "NHWC" else (batch, k, oh, ow)
    return [TensorType(x.dtype, shape)]


def conv2d_nchw(x: np.ndarray, w: np.ndarray, geometry: Conv2dGeometry, group: int) -> np.ndarray:
    """Grouped 2-D convolution, NCHW input and OIHW weights, via strided window views."""
    n, _c, _, _ = x.shape
    k, cg, kh, kw = w.shape
    pt, pl, pb, pr = geometry.pads
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    dh, dw = geometry.dilations
    sh, sw = geometry.strides
    oh, ow = geometry.out_hw
    win = np.lib.stride_tricks.sliding_window_view(
        xp, ((kh - 1) * dh + 1, (kw - 1) * dw + 1), axis=(2, 3)
    )[:, :, : (oh - 1) * sh + 1 : sh, : (ow - 1) * sw + 1 : sw, ::dh, ::dw]
    # win: [n, c, oh, ow, kh, kw]
    kg = k // group
    win = win.reshape(n, group, cg, oh, ow, kh, kw)
    wg = w.reshape(group, kg, cg, kh, kw)
    out = np.einsum("ngchwij,gkcij->ngkhw", win, wg, optimize=True)
    return contiguous(out.reshape(n, k, oh, ow), dtype=x.dtype)


def _conv_core(node: Node, x: np.ndarray, w: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    group = int(node.attrs.get("group", 1))
    nhwc = conv_layout(node) == "NHWC"
    xn = x.transpose(0, 3, 1, 2) if nhwc else x
    wn = w.transpose(3, 2, 0, 1) if nhwc else w
    geometry = window_geometry(node, (xn.shape[2], xn.shape[3]), (wn.shape[2], wn.shape[3]))
    y = conv2d_nchw(xn, wn, geometry, group)
    if b is not None:
        y = y + b.reshape(1, -1, 1, 1)
    return y.transpose(0, 2, 3, 1) if nhwc else y


@register("Conv", _conv_infer)
def _conv(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    return [_keep(contiguous(_conv_core(node, x, _arr(xs, 1), _opt(xs, 2))), x)]


@register("FusedConv", _conv_infer)
def _fused_conv(node: Node, xs: Arrays) -> list[np.ndarray]:
    """``act(conv(X, W) + B + residual)``; layout NCHW (OIHW weights) or NHWC (HWIO weights)."""
    x = _arr(xs, 0)
    y = _conv_core(node, x, _arr(xs, 1), _opt(xs, 2))
    residual = _opt(xs, 3)
    if residual is not None:
        y = y + residual
    return [_keep(contiguous(apply_activation(y, node)), x)]


def _pool_dims(node: Node, x: TensorType) -> tuple[Conv2dGeometry, tuple[int, ...]]:
    if x.rank != 4:
        raise UnsupportedOpError(f"{node.op}: only 2-D pooling is supported")
    kernel = tuple(int(k) for k in node.attrs["kernel_shape"])
    nhwc = conv_layout(node) == "NHWC"
    hw = (x.shape[1], x.shape[2]) if nhwc else (x.shape[2], x.shape[3])
    g = window_geometry(node, hw, (kernel[0], kernel[1]), bool(node.attrs.get("ceil_mode", 0)))
    oh, ow = g.out_hw
    shape = (x.shape[0], oh, ow, x.shape[3]) if nhwc else (x.shape[0], x.shape[1], oh, ow)
    return g, shape


def _pool_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    if len(n.outputs) > 1:
        raise UnsupportedOpError("MaxPool indices output is not supported")
    x = _t(t, 0)
    return [TensorType(x.dtype, _pool_dims(n, x)[1])]


def _pool(node: Node, x: np.ndarray, kind: str) -> np.ndarray:
    nhwc = conv_layout(node) == "NHWC"
    xn = x.transpose(0, 3, 1, 2) if nhwc else x
    g, _ = _pool_dims(node, TensorType.of(xn) if not nhwc else TensorType.of(x))
    pt, pl, pb, pr = g.pads
    kh, kw = g.kernel
    sh, sw = g.strides
    dh, dw = g.dilations
    oh, ow = g.out_hw
    # extra bottom/right padding so ceil_mode windows exist; they are excluded below
    eb = max((oh - 1) * sh + (kh - 1) * dh + 1 - (xn.shape[2] + pt + pb), 0)
    er = max((ow - 1) * sw + (kw - 1) * dw + 1 - (xn.shape[3] + pl + pr), 0)
    fill = -np.inf if kind == "max" else 0.0
    xp = np.pad(xn, ((0, 0), (0, 0), (pt, pb + eb), (pl, pr + er)), constant_values=fill)
    win = np.lib.stride_tricks.sliding_window_view(
        xp, ((kh - 1) * dh + 1, (kw - 1) * dw + 1), axis=(2, 3)
    )[:, :, : (oh - 1) * sh + 1 : sh, : (ow - 1) * sw + 1 : sw, ::dh, ::dw]
    if kind == "max":
        y = win.max(axis=(4, 5))
    else:
        total = win.sum(axis=(4, 5))
        if int(node.attrs.get("count_include_pad", 0)):
            ones = np.pad(
                np.ones(xn.shape[2:], dtype=np.float64), ((pt, pb), (pl, pr)), constant_values=1
            )
            ones = np.pad(ones, ((0, eb), (0, er)))
        else:
            ones = np.pad(np.ones(xn.shape[2:], dtype=np.float64), ((pt, pb + eb), (pl, pr + er)))
        cnt = np.lib.stride_tricks.sliding_window_view(
            ones, ((kh - 1) * dh + 1, (kw - 1) * dw + 1)
        )[: (oh - 1) * sh + 1 : sh, : (ow - 1) * sw + 1 : sw, ::dh, ::dw].sum(axis=(2, 3))
        y = total / cnt
    y = contiguous(y.astype(x.dtype))
    return contiguous(y.transpose(0, 2, 3, 1)) if nhwc else y


@register("MaxPool", _pool_infer)
def _max_pool(node: Node, xs: Arrays) -> list[np.ndarray]:
    return [_pool(node, _arr(xs, 0), "max")]


@register("AveragePool", _pool_infer)
def _avg_pool(node: Node, xs: Arrays) -> list[np.ndarray]:
    return [_pool(node, _arr(xs, 0), "avg")]


def _global_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    x = _t(t, 0)
    if x.rank < 3:
        raise ShapeError(f"{n.op} needs rank >= 3")
    if conv_layout(n) == "NHWC":
        return [TensorType(x.dtype, (x.shape[0], *([1] * (x.rank - 2)), x.shape[-1]))]
    return [TensorType(x.dtype, (*x.shape[:2], *([1] * (x.rank - 2))))]


def _spatial_axes(node: Node, rank: int) -> tuple[int, ...]:
    return tuple(range(1, rank - 1)) if conv_layout(node) == "NHWC" else tuple(range(2, rank))


@register("GlobalAveragePool", _global_infer)
def _gap(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    return [_keep(x.mean(axis=_spatial_axes(node, x.ndim), keepdims=True), x)]


@register("GlobalMaxPool", _global_infer)
def _gmp(node: Node, xs: Arrays) -> list[np.ndarray]:
    x = _arr(xs, 0)
    return [x.max(axis=_spatial_axes(node, x.ndim), keepdims=True)]


# ------------------------------------------------------------------------------------------
# fused elementwise programs (produced by the loop fusion pass)
# ------------------------------------------------------------------------------------------
# A FusedElementwise node carries ``program``: a list of instructions ``[op, attrs, args]``.
# An argument ``a`` refers to node input ``a`` when ``a < len(inputs)``, to the result of
# instruction ``a - len(inputs)`` otherwise, and ``-1`` is an omitted optional input. The last
# instruction's result is the node's output. Every instruction is an elementwise operator.
def _program_nodes(node: Node) -> list[tuple[Node, list[int]]]:
    program = node.attrs.get("program")
    if not program:
        raise UnsupportedOpError("FusedElementwise without a program")
    out = []
    for op, attrs, args in program:
        if not is_elementwise(str(op)):
            raise UnsupportedOpError(f"FusedElementwise: {op} is not elementwise")
        out.append((Node(str(op), [], [_NO_VALUE], dict(attrs)), [int(a) for a in args]))
    return out


def _fused_elementwise_infer(n: Node, t: list[TensorType | None], c: Consts) -> list[TensorType]:
    types: list[TensorType | None] = list(t)
    consts: Consts = list(c)
    for sub, args in _program_nodes(n):
        sub_types = [types[a] if a >= 0 else None for a in args]
        sub_consts = [consts[a] if a >= 0 else None for a in args]
        types.append(spec(sub.op).infer(sub, sub_types, sub_consts)[0])
        consts.append(None)
    last = types[-1]
    assert last is not None
    return [last]


@register("FusedElementwise", _fused_elementwise_infer)
def _fused_elementwise(node: Node, xs: Arrays) -> list[np.ndarray]:
    """Evaluate the program instruction by instruction with the reference semantics."""
    values: Arrays = list(xs)
    for sub, args in _program_nodes(node):
        values.append(spec(sub.op).run(sub, [values[a] if a >= 0 else None for a in args])[0])
    last = values[-1]
    assert last is not None
    return [last]


# ------------------------------------------------------------------------------------------
# public helpers
# ------------------------------------------------------------------------------------------
def infer(node: Node, types: list[TensorType | None], consts: Consts) -> list[TensorType]:
    out = spec(node.op).infer(node, types, consts)
    if len(out) < len(node.outputs):
        raise ShapeError(f"{node.op}: inferred {len(out)} outputs, node has {len(node.outputs)}")
    return out[: len(node.outputs)]


def run(node: Node, arrays: Arrays) -> list[np.ndarray]:
    return spec(node.op).run(node, arrays)


def is_elementwise(op: str) -> bool:
    s = REGISTRY.get(op)
    return bool(s and s.elementwise)


def flops(node: Node) -> int:
    """Approximate floating point operations (a multiply-add counts as 2)."""
    out = node.outputs[0].type
    if out is None:
        return 0
    if node.op in ("Conv", "FusedConv"):
        w = node.input(1).tensor_type
        k_elems = w.size // (w.shape[3] if conv_layout(node) == "NHWC" else w.shape[0])
        return 2 * out.size * k_elems
    if node.op == "FusedElementwise":
        return out.size * len(node.attrs.get("program", ()))
    if node.op in ("MatMul", "FusedGemm", "Gemm"):
        a = node.input(0).tensor_type
        k = a.shape[0] if node.op == "Gemm" and int(node.attrs.get("transA", 0)) else a.shape[-1]
        return 2 * out.size * k
    return out.size
