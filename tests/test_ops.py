"""Every operator's numpy semantics and inferred types against onnxruntime."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from helpers import assert_matches_ort, op_model
from ml_graph_compiler import ops
from ml_graph_compiler.errors import ShapeError, UnsupportedOpError
from ml_graph_compiler.ir import Node, TensorType

rng = np.random.default_rng(0)


def f32(*shape: int) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32)


def pos(*shape: int) -> np.ndarray:
    return (rng.random(shape) + 0.5).astype(np.float32)


def i64(values: list[int] | int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


UNARY = [
    "Relu",
    "Sigmoid",
    "Tanh",
    "Erf",
    "Exp",
    "Neg",
    "Abs",
    "Floor",
    "Ceil",
    "Sin",
    "Cos",
    "Softplus",
    "Identity",
]


@pytest.mark.parametrize("op", UNARY)
def test_unary(op: str) -> None:
    model, feeds = op_model(op, [f32(3, 5)])
    assert_matches_ort(model, feeds, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("op", ["Sqrt", "Log", "Reciprocal"])
def test_unary_positive(op: str) -> None:
    model, feeds = op_model(op, [pos(4, 6)])
    assert_matches_ort(model, feeds)


@pytest.mark.parametrize("op", ["Add", "Sub", "Mul", "Div", "Max", "Min"])
@pytest.mark.parametrize(
    "shapes", [((2, 3, 4), (4,)), ((2, 1, 4), (3, 1)), ((5,), (5,)), ((), (2, 2))]
)
def test_binary_broadcast(op: str, shapes: tuple[tuple[int, ...], tuple[int, ...]]) -> None:
    model, feeds = op_model(op, [f32(*shapes[0]), pos(*shapes[1])])
    assert_matches_ort(model, feeds)


@pytest.mark.parametrize("op", ["Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual"])
def test_compare(op: str) -> None:
    a = rng.integers(0, 3, (3, 4)).astype(np.float32)
    model, feeds = op_model(op, [a, rng.integers(0, 3, (4,)).astype(np.float32)])
    assert_matches_ort(model, feeds)


@pytest.mark.parametrize("op", ["And", "Or", "Xor"])
def test_logical(op: str) -> None:
    model, feeds = op_model(op, [rng.random((3, 4)) > 0.5, rng.random((4,)) > 0.5])
    assert_matches_ort(model, feeds)


def test_not_where_and_pow() -> None:
    model, feeds = op_model("Not", [rng.random((5,)) > 0.5])
    assert_matches_ort(model, feeds)
    model, feeds = op_model("Where", [rng.random((3, 1)) > 0.5, f32(3, 4), f32(4)])
    assert_matches_ort(model, feeds)
    model, feeds = op_model("Pow", [pos(3, 4), np.asarray(2.5, np.float32)], const={1})
    assert_matches_ort(model, feeds)


def test_integer_division_truncates() -> None:
    model, feeds = op_model("Div", [i64([7, -7, 9]), i64([2, 2, -4])])
    out = assert_matches_ort(model, feeds)
    assert out["out0"].tolist() == [3, -3, -2]


def test_activations_with_attributes() -> None:
    model, feeds = op_model("LeakyRelu", [f32(4, 4)], attrs={"alpha": 0.2})
    assert_matches_ort(model, feeds)
    model, feeds = op_model("HardSigmoid", [f32(4, 4)], attrs={"alpha": 0.3, "beta": 0.4})
    assert_matches_ort(model, feeds)
    lo, hi = np.asarray(-0.5, np.float32), np.asarray(0.7, np.float32)
    model, feeds = op_model("Clip", [f32(4, 4), lo, hi], const={1, 2})
    assert_matches_ort(model, feeds)
    model, feeds = op_model("Clip", [f32(4, 4), None, hi], const={2})
    assert_matches_ort(model, feeds)


def test_gelu_opset20_and_erf_accuracy() -> None:
    x = np.linspace(-6, 6, 2001).astype(np.float32)
    import math

    expected = np.asarray([math.erf(v) for v in x.astype(np.float64)])
    assert np.max(np.abs(ops.erf(x).astype(np.float64) - expected)) < 2e-7
    for approx in ("none", "tanh"):
        model, feeds = op_model("Gelu", [f32(3, 7)], attrs={"approximate": approx}, opset=20)
        assert_matches_ort(model, feeds, atol=2e-6)


@pytest.mark.parametrize("to", [1, 6, 7, 9])
def test_cast(to: int) -> None:
    model, feeds = op_model("Cast", [f32(3, 3) * 4], attrs={"to": to})
    assert_matches_ort(model, feeds)


def test_cast_rejects_unknown_type() -> None:
    node = Node("Cast", [], [], {"to": 16})
    with pytest.raises(UnsupportedOpError):
        ops.cast_dtype(node)


def test_dropout_inference() -> None:
    model, feeds = op_model("Dropout", [f32(2, 3)], n_outputs=2)
    assert_matches_ort(model, feeds)


@pytest.mark.parametrize(
    "shape,target", [((2, 3, 4), [4, -1]), ((2, 3, 4), [0, 12]), ((6,), [2, 3]), ((2, 3), [-1])]
)
def test_reshape(shape: tuple[int, ...], target: list[int]) -> None:
    model, feeds = op_model("Reshape", [f32(*shape), i64(target)], const={1})
    assert_matches_ort(model, feeds)


@pytest.mark.parametrize(
    "shape,target",
    [((2, 3), [4, -1]), ((6,), [-1, -1]), ((5,), [0, 5]), ((2, 3), [7])],
)
def test_reshape_target_errors(shape: tuple[int, ...], target: list[int]) -> None:
    with pytest.raises(ShapeError):
        ops.reshape_target(shape, target)


@given(st.lists(st.integers(1, 4), min_size=1, max_size=4), st.data())
@settings(max_examples=60, deadline=None)
def test_reshape_target_property(shape: list[int], data: st.DataObject) -> None:
    total = int(np.prod(shape))
    divisors = [d for d in range(1, total + 1) if total % d == 0]
    first = data.draw(st.sampled_from(divisors))
    out = ops.reshape_target(tuple(shape), [first, -1])
    assert out == (first, total // first)
    assert int(np.prod(out)) == total


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_flatten(axis: int) -> None:
    model, feeds = op_model("Flatten", [f32(2, 3, 4)], attrs={"axis": axis})
    assert_matches_ort(model, feeds)


@pytest.mark.parametrize("perm", [None, [0, 2, 1], [2, 0, 1]])
def test_transpose(perm: list[int] | None) -> None:
    attrs = {"perm": perm} if perm else {}
    model, feeds = op_model("Transpose", [f32(2, 3, 4)], attrs=attrs)
    assert_matches_ort(model, feeds)


def test_squeeze_unsqueeze() -> None:
    model, feeds = op_model("Squeeze", [f32(1, 3, 1, 2), i64([0, -2])], const={1})
    assert_matches_ort(model, feeds)
    model, feeds = op_model("Squeeze", [f32(1, 3, 1, 2)])
    assert_matches_ort(model, feeds)
    model, feeds = op_model("Unsqueeze", [f32(3, 2), i64([0, 3])], const={1})
    assert_matches_ort(model, feeds)
    model, feeds = op_model("Unsqueeze", [f32(3, 2)], attrs={"axes": [1]}, opset=11)
    assert_matches_ort(model, feeds)


def test_squeeze_non_unit_axis_fails() -> None:
    with pytest.raises(ShapeError):
        ops.squeeze_shape((2, 3), [0])


def test_concat_split_slice() -> None:
    model, feeds = op_model("Concat", [f32(2, 3), f32(2, 1), f32(2, 2)], attrs={"axis": -1})
    assert_matches_ort(model, feeds)
    model, feeds = op_model("Split", [f32(6, 4), i64([1, 2, 3])], const={1}, n_outputs=3)
    assert_matches_ort(model, feeds)
    model, feeds = op_model("Split", [f32(2, 6)], attrs={"axis": 1}, n_outputs=3)
    assert_matches_ort(model, feeds)
    model, feeds = op_model(
        "Split", [f32(2, 7)], attrs={"axis": 1, "num_outputs": 3}, n_outputs=3, opset=18
    )
    assert_matches_ort(model, feeds)


@pytest.mark.parametrize(
    "starts,ends,axes,steps",
    [
        ([1], [3], [0], [1]),
        ([0, 1], [-1, 100], [0, 1], [1, 2]),
        ([-1], [-100], [1], [-1]),
        ([2], [9223372036854775807], [1], [1]),
    ],
)
def test_slice(starts: list[int], ends: list[int], axes: list[int], steps: list[int]) -> None:
    model, feeds = op_model(
        "Slice", [f32(4, 6), i64(starts), i64(ends), i64(axes), i64(steps)], const={1, 2, 3, 4}
    )
    assert_matches_ort(model, feeds)


def test_slice_opset1_attributes() -> None:
    node = Node("Slice", [], [], {"starts": [1], "ends": [3], "axes": [1]})
    slices = ops.slice_params(node, (2, 5), [])
    assert slices[1] == slice(1, 3, 1)


def test_gather_variants() -> None:
    model, feeds = op_model("Gather", [f32(5, 3), i64([[0, 4], [-1, 2]])], attrs={"axis": 0})
    assert_matches_ort(model, feeds)
    model, feeds = op_model("Gather", [f32(2, 5, 3), i64(1)], attrs={"axis": 1}, const={1})
    assert_matches_ort(model, feeds)


def test_shape_size_constant_of_shape_expand_tile_range() -> None:
    model, feeds = op_model("Shape", [f32(2, 3, 4)], attrs={"start": 1}, opset=15)
    assert_matches_ort(model, feeds)
    model, feeds = op_model("Size", [f32(2, 3)])
    assert_matches_ort(model, feeds)
    from onnx import numpy_helper

    value = numpy_helper.from_array(np.asarray([1.5], np.float32))
    model, feeds = op_model("ConstantOfShape", [i64([2, 3])], const={0}, attrs={"value": value})
    assert_matches_ort(model, feeds)
    model, feeds = op_model("Expand", [f32(3, 1), i64([2, 3, 4])], const={1})
    assert_matches_ort(model, feeds)
    model, feeds = op_model("Tile", [f32(2, 3), i64([2, 1])], const={1})
    assert_matches_ort(model, feeds)
    model, feeds = op_model("Range", [i64(2), i64(11), i64(3)], const={0, 1, 2})
    assert_matches_ort(model, feeds)


@pytest.mark.parametrize("upper", [0, 1])
def test_trilu(upper: int) -> None:
    model, feeds = op_model("Trilu", [f32(2, 4, 4), i64(1)], const={1}, attrs={"upper": upper})
    assert_matches_ort(model, feeds)


def test_pad_constant_and_unsupported_mode() -> None:
    model, feeds = op_model(
        "Pad", [f32(2, 3), i64([1, 0, 0, 2]), np.asarray(0.5, np.float32)], const={1, 2}
    )
    assert_matches_ort(model, feeds)
    node = Node("Pad", [], [], {"mode": "reflect", "pads": [0, 0]})
    with pytest.raises(UnsupportedOpError):
        ops.REGISTRY["Pad"].run(node, [f32(2)])


@pytest.mark.parametrize("op", ["ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd"])
@pytest.mark.parametrize("keepdims", [0, 1])
def test_reductions(op: str, keepdims: int) -> None:
    if op == "ReduceSum":
        model, feeds = op_model(
            op, [f32(2, 3, 4), i64([1, -1])], const={1}, attrs={"keepdims": keepdims}, opset=13
        )
    else:
        model, feeds = op_model(op, [f32(2, 3, 4)], attrs={"axes": [1, -1], "keepdims": keepdims})
    assert_matches_ort(model, feeds, rtol=1e-5)


def test_reduce_all_axes_and_argmax() -> None:
    model, feeds = op_model("ReduceMean", [f32(2, 3)])
    assert_matches_ort(model, feeds)
    model, feeds = op_model("ArgMax", [f32(3, 5)], attrs={"axis": 1, "keepdims": 0})
    assert_matches_ort(model, feeds)


@pytest.mark.parametrize("op", ["Softmax", "LogSoftmax"])
@pytest.mark.parametrize("axis", [-1, 1])
def test_softmax(op: str, axis: int) -> None:
    model, feeds = op_model(op, [f32(2, 3, 5) * 5], attrs={"axis": axis}, opset=13)
    assert_matches_ort(model, feeds)


def test_layer_norm_and_batch_norm() -> None:
    model, feeds = op_model(
        "LayerNormalization",
        [f32(2, 3, 8), f32(8), f32(8)],
        const={1, 2},
        attrs={"epsilon": 1e-5},
        opset=17,
    )
    assert_matches_ort(model, feeds)
    model, feeds = op_model(
        "BatchNormalization", [f32(2, 3, 4, 4), f32(3), f32(3), f32(3), pos(3)], const={1, 2, 3, 4}
    )
    assert_matches_ort(model, feeds)


@pytest.mark.parametrize(
    "a,b",
    [
        ((3, 4), (4, 5)),
        ((2, 3, 4), (4, 6)),
        ((2, 1, 3, 4), (5, 4, 2)),
        ((4,), (4, 3)),
        ((3, 4), (4,)),
    ],
)
def test_matmul(a: tuple[int, ...], b: tuple[int, ...]) -> None:
    model, feeds = op_model("MatMul", [f32(*a), f32(*b)])
    assert_matches_ort(model, feeds, rtol=1e-4)


@pytest.mark.parametrize("trans_a,trans_b", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_gemm(trans_a: int, trans_b: int) -> None:
    a = f32(3, 4) if not trans_a else f32(4, 3)
    b = f32(4, 5) if not trans_b else f32(5, 4)
    model, feeds = op_model(
        "Gemm",
        [a, b, f32(5)],
        const={2},
        attrs={"transA": trans_a, "transB": trans_b, "alpha": 0.5, "beta": 2.0},
    )
    assert_matches_ort(model, feeds, rtol=1e-4)


def test_matmul_shape_errors() -> None:
    with pytest.raises(ShapeError):
        ops.matmul_shape((2, 3), (4, 5))
    with pytest.raises(ShapeError):
        ops.matmul_shape((), (4, 5))


@pytest.mark.parametrize(
    "attrs,channels,group",
    [
        ({"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}, 4, 1),
        ({"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 0, 1, 0]}, 4, 1),
        ({"kernel_shape": [3, 3], "dilations": [2, 2], "pads": [2, 2, 2, 2]}, 4, 2),
        ({"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}, 6, 6),
        ({"kernel_shape": [1, 1]}, 4, 1),
        ({"kernel_shape": [3, 3], "auto_pad": "SAME_UPPER", "strides": [2, 2]}, 4, 1),
        ({"kernel_shape": [2, 2], "auto_pad": "SAME_LOWER"}, 4, 1),
        ({"kernel_shape": [3, 3], "auto_pad": "VALID"}, 4, 1),
    ],
)
def test_conv(attrs: dict[str, object], channels: int, group: int) -> None:
    k = 6
    w = f32(k, channels // group, *attrs["kernel_shape"])  # type: ignore[misc]
    model, feeds = op_model(
        "Conv", [f32(2, channels, 9, 8), w, f32(k)], const={1, 2}, attrs={**attrs, "group": group}
    )
    assert_matches_ort(model, feeds, rtol=1e-4, atol=1e-5)


@given(
    k=st.integers(1, 4),
    s=st.integers(1, 3),
    d=st.integers(1, 2),
    p=st.integers(0, 2),
    hw=st.integers(7, 10),
    group=st.sampled_from([1, 2]),
)
@settings(max_examples=25, deadline=None)
def test_conv_property(k: int, s: int, d: int, p: int, hw: int, group: int) -> None:
    w = f32(4, 4 // group, k, k)
    model, feeds = op_model(
        "Conv",
        [f32(1, 4, hw, hw + 1), w],
        const={1},
        attrs={
            "kernel_shape": [k, k],
            "strides": [s, s],
            "dilations": [d, d],
            "pads": [p] * 4,
            "group": group,
        },
    )
    assert_matches_ort(model, feeds, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "attrs",
    [
        {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]},
        {"kernel_shape": [2, 2], "strides": [2, 2]},
        {"kernel_shape": [3, 3], "strides": [2, 2], "ceil_mode": 1},
        {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "count_include_pad": 1},
        {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1], "ceil_mode": 1},
    ],
)
@pytest.mark.parametrize("op", ["MaxPool", "AveragePool"])
def test_pooling(op: str, attrs: dict[str, object]) -> None:
    if op == "MaxPool" and "count_include_pad" in attrs:
        attrs = {k: v for k, v in attrs.items() if k != "count_include_pad"}
    model, feeds = op_model(op, [f32(2, 3, 9, 8)], attrs=attrs)
    assert_matches_ort(model, feeds)


@pytest.mark.parametrize("op", ["GlobalAveragePool", "GlobalMaxPool"])
def test_global_pooling(op: str) -> None:
    model, feeds = op_model(op, [f32(2, 3, 5, 4)])
    assert_matches_ort(model, feeds)


def test_nhwc_and_fused_operators_match_nchw() -> None:
    x, w, b, r = f32(1, 4, 7, 7), f32(6, 4, 3, 3), f32(6), f32(1, 6, 7, 7)
    attrs = {"pads": [1, 1, 1, 1], "group": 1, "activation": "relu"}
    nchw = Node("FusedConv", [], [], attrs)
    nhwc = Node("FusedConv", [], [], {**attrs, "layout": "NHWC"})
    y1 = ops.run(nchw, [x, w, b, r])[0]
    y2 = ops.run(
        nhwc, [x.transpose(0, 2, 3, 1), w.transpose(2, 3, 1, 0), b, r.transpose(0, 2, 3, 1)]
    )[0]
    np.testing.assert_allclose(y1, y2.transpose(0, 3, 1, 2), rtol=1e-5, atol=1e-5)
    reference = np.maximum(
        ops.run(Node("Conv", [], [], {"pads": [1, 1, 1, 1]}), [x, w, b])[0] + r, 0
    )
    np.testing.assert_allclose(y1, reference, rtol=1e-5, atol=1e-5)
    for op in ("MaxPool", "AveragePool"):
        a = {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]}
        p1 = ops.run(Node(op, [], [], a), [x])[0]
        p2 = ops.run(Node(op, [], [], {**a, "layout": "NHWC"}), [x.transpose(0, 2, 3, 1)])[0]
        np.testing.assert_allclose(p1, p2.transpose(0, 3, 1, 2), rtol=1e-6)
    g = ops.run(Node("GlobalAveragePool", [], [], {"layout": "NHWC"}), [x.transpose(0, 2, 3, 1)])[0]
    assert g.shape == (1, 1, 1, 4)


@pytest.mark.parametrize("act", ops.ACTIVATIONS)
def test_fused_gemm_activations(act: str) -> None:
    a, b, bias, res = f32(2, 3, 4), f32(4, 5), f32(5), f32(2, 3, 5)
    node = Node("FusedGemm", [], [], {"activation": act, "clip_min": -0.5, "clip_max": 0.5})
    y = ops.run(node, [a, b, bias, res])[0]
    pre = a @ b + bias + res
    expected = {
        "none": pre,
        "relu": np.maximum(pre, 0),
        "clip": np.clip(pre, -0.5, 0.5),
        "gelu": ops.gelu(pre),
        "gelu_tanh": ops.gelu(pre, "tanh"),
        "tanh": np.tanh(pre),
        "sigmoid": 1 / (1 + np.exp(-pre)),
    }[act]
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-6)
    yt = ops.run(
        Node(
            "FusedGemm",
            [],
            [],
            {"activation": act, "trans_b": 1, "clip_min": -0.5, "clip_max": 0.5},
        ),
        [a, b.T.copy(), bias, res],
    )[0]
    np.testing.assert_allclose(yt, y, rtol=1e-5, atol=1e-6)


def test_unknown_activation_and_operator() -> None:
    with pytest.raises(UnsupportedOpError):
        ops.apply_activation(
            np.zeros(2, np.float32), Node("FusedGemm", [], [], {"activation": "x"})
        )
    with pytest.raises(UnsupportedOpError):
        ops.spec("NoSuchOp")
    assert "Conv" in ops.supported_ops()


def test_infer_reports_rank_and_layout_errors() -> None:
    conv = Node("Conv", [], [], {})
    with pytest.raises(UnsupportedOpError):
        ops.infer(conv, [TensorType("float32", (1, 2, 3)), TensorType("float32", (1, 2, 3))], [])
    with pytest.raises(ShapeError):
        ops.infer(
            conv, [TensorType("float32", (1, 3, 5, 5)), TensorType("float32", (2, 2, 3, 3))], []
        )
    with pytest.raises(UnsupportedOpError):
        ops.conv_layout(Node("Conv", [], [], {"layout": "NCDHW"}))


def test_flops() -> None:
    from ml_graph_compiler.ir import Value

    x, w = (
        Value("x", TensorType("float32", (1, 3, 8, 8))),
        Value("w", TensorType("float32", (4, 3, 3, 3))),
    )
    y = Value("y", TensorType("float32", (1, 4, 8, 8)))
    node = Node("Conv", [x, w], [y], {"pads": [1, 1, 1, 1]})
    assert ops.flops(node) == 2 * 4 * 64 * 27
    a, b = Value("a", TensorType("float32", (2, 3))), Value("b", TensorType("float32", (3, 5)))
    out = Value("o", TensorType("float32", (2, 5)))
    assert ops.flops(Node("MatMul", [a, b], [out])) == 2 * 10 * 3
    assert ops.flops(Node("Relu", [a], [Value("r", TensorType("float32", (2, 3)))])) == 6
