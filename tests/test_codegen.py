"""C backend: generated code checked against onnxruntime and the reference interpreter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from onnx import TensorProto, helper, numpy_helper

from helpers import op_model, ort_run
from ml_graph_compiler import samples
from ml_graph_compiler.codegen import (
    BuildOptions,
    CompiledModel,
    compile_graph,
    compile_model,
    generate_c,
)
from ml_graph_compiler.codegen.build import find_compiler
from ml_graph_compiler.codegen.c import pack_b_panels
from ml_graph_compiler.errors import CodegenError, GraphError
from ml_graph_compiler.interpreter import random_feeds, run_graph
from ml_graph_compiler.ir import Graph, TensorType
from ml_graph_compiler.memory import STRATEGIES
from ml_graph_compiler.onnx_import import from_onnx
from ml_graph_compiler.passes import PIPELINES, optimize

try:
    find_compiler()
except CodegenError:  # pragma: no cover
    pytest.skip("no C compiler", allow_module_level=True)

rng = np.random.default_rng(1)


@pytest.fixture(scope="session")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("mgc-cache")


def f32(*shape: int) -> np.ndarray:
    return np.asarray(rng.standard_normal(shape), dtype=np.float32)


def pos(*shape: int) -> np.ndarray:
    return np.asarray(rng.random(shape) + 0.5, dtype=np.float32)


def check(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray],
    cache: Path,
    opt: str = "O0",
    rtol: float = 1e-4,
    atol: float = 1e-5,
    **kwargs: Any,
) -> CompiledModel:
    compiled = compile_model(model, opt=opt, cache_dir=cache, **kwargs)
    got = compiled.run(feeds)
    expected = ort_run(model, feeds)
    assert set(got) == set(expected)
    for name, value in expected.items():
        assert got[name].shape == value.shape, name
        assert got[name].dtype == value.dtype, name
        np.testing.assert_allclose(got[name], value, rtol=rtol, atol=atol, err_msg=name)
    return compiled


# ------------------------------------------------------------------------------------------
# single operators (compiled without optimization, so each lowering is exercised directly)
# ------------------------------------------------------------------------------------------
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
def test_unary(op: str, cache: Path) -> None:
    x = f32(3, 70) * 3
    model, feeds = op_model(op, [x])
    check(model, feeds, cache, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("op", ["Sqrt", "Log", "Reciprocal"])
def test_unary_positive(op: str, cache: Path) -> None:
    model, feeds = op_model(op, [pos(4, 33)])
    check(model, feeds, cache)


@pytest.mark.parametrize("op", ["Add", "Sub", "Mul", "Div", "Max", "Min", "Pow"])
@pytest.mark.parametrize(
    "shapes", [((2, 3, 4), (4,)), ((2, 1, 4), (3, 1)), ((5,), (5,)), ((), (2, 2))]
)
def test_binary_broadcast(
    op: str, shapes: tuple[tuple[int, ...], tuple[int, ...]], cache: Path
) -> None:
    a = pos(*shapes[0]) if op == "Pow" else f32(*shapes[0])
    model, feeds = op_model(op, [a, pos(*shapes[1])])
    check(model, feeds, cache)


@pytest.mark.parametrize("op", ["Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual"])
def test_compare(op: str, cache: Path) -> None:
    a = rng.integers(0, 3, (3, 4)).astype(np.float32)
    model, feeds = op_model(op, [a, rng.integers(0, 3, (4,)).astype(np.float32)])
    check(model, feeds, cache)


@pytest.mark.parametrize("op", ["And", "Or", "Xor"])
def test_logical(op: str, cache: Path) -> None:
    model, feeds = op_model(op, [rng.random((3, 4)) > 0.5, rng.random((4,)) > 0.5])
    check(model, feeds, cache)


def test_not_where_cast_and_integer_math(cache: Path) -> None:
    model, feeds = op_model("Not", [rng.random((5,)) > 0.5])
    check(model, feeds, cache)
    model, feeds = op_model("Where", [rng.random((3, 1)) > 0.5, f32(3, 4), f32(4)])
    check(model, feeds, cache)
    for to, src in [
        (TensorProto.INT64, f32(6) * 5),
        (TensorProto.FLOAT, np.arange(6, dtype=np.int64)),
        (TensorProto.BOOL, np.asarray([0, 1, 2, 0], np.float32)),
    ]:
        model, feeds = op_model("Cast", [src], attrs={"to": to})
        check(model, feeds, cache)
    ints = np.asarray([7, -7, 9, 4], np.int64)
    for op in ("Add", "Sub", "Mul", "Div", "Max", "Min"):
        model, feeds = op_model(op, [ints, np.asarray([2, 2, -4, 3], np.int64)])
        check(model, feeds, cache)


@pytest.mark.parametrize(
    "op, attrs, inputs, const",
    [
        (
            "Clip",
            {},
            [f32(3, 5), np.asarray(-0.5, np.float32), np.asarray(0.5, np.float32)],
            {1, 2},
        ),
        ("Clip", {}, [f32(3, 5), None, np.asarray(0.2, np.float32)], {2}),
        ("LeakyRelu", {"alpha": 0.2}, [f32(3, 5)], set()),
        ("HardSigmoid", {}, [f32(3, 5)], set()),
        ("Gelu", {}, [f32(3, 5)], set()),
        ("Gelu", {"approximate": "tanh"}, [f32(3, 5)], set()),
    ],
)
def test_parameterized_elementwise(
    op: str, attrs: dict[str, Any], inputs: list[Any], const: set[int], cache: Path
) -> None:
    opset = 20 if op == "Gelu" else 14
    model, feeds = op_model(op, inputs, const=const, attrs=attrs, opset=opset)
    check(model, feeds, cache, rtol=1e-5, atol=1e-6)


def test_integer_clip(cache: Path) -> None:
    model, feeds = op_model(
        "Clip",
        [np.arange(-5, 5, dtype=np.int64), np.asarray(-2, np.int64), np.asarray(3, np.int64)],
        const={1, 2},
    )
    check(model, feeds, cache)


@pytest.mark.parametrize(
    "op, inputs, const, attrs",
    [
        ("Transpose", [f32(2, 3, 4, 5)], set(), {"perm": [0, 2, 3, 1]}),
        ("Transpose", [f32(3, 4)], set(), {}),
        ("Expand", [f32(3, 1), np.asarray([2, 3, 4], np.int64)], {1}, {}),
        ("Tile", [f32(2, 3), np.asarray([2, 2], np.int64)], {1}, {}),
        (
            "Slice",
            [
                f32(6, 8),
                np.asarray([1, -6], np.int64),
                np.asarray([5, 100], np.int64),
                np.asarray([0, 1], np.int64),
                np.asarray([2, 3], np.int64),
            ],
            {1, 2, 3, 4},
            {},
        ),
        (
            "Slice",
            [
                f32(6, 8),
                np.asarray([4], np.int64),
                np.asarray([0], np.int64),
                np.asarray([1], np.int64),
                np.asarray([-2], np.int64),
            ],
            {1, 2, 3, 4},
            {},
        ),
        ("Concat", [f32(2, 3), f32(2, 5), f32(2, 1)], set(), {"axis": 1}),
        ("Concat", [f32(1, 3), f32(2, 3)], {0}, {"axis": 0}),
        (
            "Pad",
            [f32(2, 3), np.asarray([1, 0, 2, 3], np.int64), np.asarray(1.5, np.float32)],
            {1, 2},
            {},
        ),
        ("Gather", [f32(10, 4), np.asarray([[1, -1], [3, 0]], np.int64)], set(), {"axis": 0}),
        ("Gather", [f32(3, 10, 2), np.asarray([4, 2], np.int64)], {1}, {"axis": 1}),
        ("Shape", [f32(2, 3, 4)], set(), {}),
        ("Size", [f32(2, 3, 4)], set(), {}),
        (
            "ConstantOfShape",
            [np.asarray([2, 3], np.int64)],
            {0},
            {"value": numpy_helper.from_array(np.asarray([7], np.int64))},
        ),
        (
            "Range",
            [np.asarray(1.0, np.float32), np.asarray(9.0, np.float32), np.asarray(1.5, np.float32)],
            {0, 1, 2},
            {},
        ),
        ("ReduceMean", [f32(2, 3, 4)], set(), {"axes": [1, 2], "keepdims": 1}),
        ("ReduceMean", [f32(2, 3, 4)], set(), {"axes": [-1], "keepdims": 0}),
        ("ReduceSum", [f32(2, 3, 4), np.asarray([0, 1], np.int64)], {1}, {"keepdims": 0}),
        ("ReduceMax", [f32(2, 3, 4)], set(), {"axes": [1]}),
        ("ReduceMin", [f32(2, 3, 4)], set(), {"axes": [2]}),
        ("ReduceProd", [pos(2, 3)], set(), {"axes": [1]}),
        ("ArgMax", [f32(3, 7)], set(), {"axis": 1, "keepdims": 0}),
        ("Softmax", [f32(2, 3, 9)], set(), {"axis": -1}),
        ("Softmax", [f32(2, 7, 3)], set(), {"axis": 1}),
        ("LogSoftmax", [f32(4, 9)], set(), {"axis": -1}),
        ("Flatten", [f32(2, 3, 4)], set(), {"axis": 2}),
        ("Unsqueeze", [f32(2, 3), np.asarray([0, 3], np.int64)], {1}, {}),
        ("Squeeze", [f32(2, 1, 3), np.asarray([1], np.int64)], {1}, {}),
        ("Identity", [f32(4)], set(), {}),
    ],
)
def test_data_movement_and_reductions(
    op: str, inputs: list[np.ndarray], const: set[int], attrs: dict[str, Any], cache: Path
) -> None:
    model, feeds = op_model(op, inputs, const=const, attrs=attrs)
    check(model, feeds, cache)


def test_split_and_dropout(cache: Path) -> None:
    model, feeds = op_model(
        "Split",
        [f32(4, 9), np.asarray([2, 3, 4], np.int64)],
        const={1},
        attrs={"axis": 1},
        n_outputs=3,
    )
    check(model, feeds, cache)
    model, feeds = op_model("Dropout", [f32(3, 4)], n_outputs=2)
    check(model, feeds, cache)


def test_layer_and_batch_norm(cache: Path) -> None:
    model, feeds = op_model(
        "LayerNormalization",
        [f32(2, 5, 16), pos(16), f32(16)],
        const={1, 2},
        attrs={"axis": -1, "epsilon": 1e-5},
        opset=17,
    )
    check(model, feeds, cache)
    model, feeds = op_model(
        "LayerNormalization", [f32(2, 3, 4), pos(3, 4)], attrs={"axis": 1}, opset=17
    )
    check(model, feeds, cache)
    model, feeds = op_model(
        "BatchNormalization", [f32(2, 3, 4, 5), pos(3), f32(3), f32(3), pos(3)], const={1, 2, 3, 4}
    )
    check(model, feeds, cache)


@pytest.mark.parametrize(
    "a_shape, b_shape, const_b",
    [
        ((5, 7), (7, 3), False),
        ((5, 7), (7, 30), True),
        ((2, 3, 5, 7), (7, 9), True),
        ((2, 3, 5, 7), (2, 3, 7, 4), False),
        ((2, 1, 5, 7), (3, 7, 4), False),
        ((7,), (7, 5), False),
        ((4, 7), (7,), False),
        ((1, 1), (1, 1), False),
        ((33, 65), (65, 29), True),
    ],
)
def test_matmul(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...], const_b: bool, cache: Path
) -> None:
    model, feeds = op_model(
        "MatMul", [f32(*a_shape), f32(*b_shape)], const={1} if const_b else set()
    )
    check(model, feeds, cache)


@pytest.mark.parametrize(
    "attrs, c_shape",
    [
        ({}, None),
        ({"transA": 1}, (3,)),
        ({"transB": 1, "alpha": 0.5, "beta": 2.0}, (1, 3)),
        ({"beta": 0.5}, (4, 3)),
        ({}, (4, 1)),
        ({}, ()),
    ],
)
def test_gemm(attrs: dict[str, Any], c_shape: tuple[int, ...] | None, cache: Path) -> None:
    a = f32(5, 4) if attrs.get("transA") else f32(4, 5)
    b = f32(3, 5) if attrs.get("transB") else f32(5, 3)
    inputs = [a, b] + ([f32(*c_shape)] if c_shape is not None else [])
    model, feeds = op_model("Gemm", inputs, const={1, 2}, attrs=attrs)
    check(model, feeds, cache)
    model, feeds = op_model("Gemm", inputs, attrs=attrs)  # dynamic weights and bias
    if c_shape in ((4, 3), None):
        check(model, feeds, cache)


@pytest.mark.parametrize(
    "attrs, x_shape, w_shape",
    [
        ({"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}, (1, 4, 9, 8), (6, 4, 3, 3)),
        (
            {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 0, 1, 2]},
            (2, 3, 11, 9),
            (5, 3, 3, 3),
        ),
        ({"kernel_shape": [2, 3], "dilations": [2, 1]}, (1, 2, 9, 9), (4, 2, 2, 3)),
        ({"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "group": 2}, (1, 4, 6, 6), (6, 2, 3, 3)),
        ({"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "group": 4}, (1, 4, 7, 7), (4, 1, 3, 3)),
        ({"kernel_shape": [3, 3], "group": 4, "strides": [2, 2]}, (2, 4, 9, 9), (8, 1, 3, 3)),
        ({"kernel_shape": [1, 1]}, (1, 16, 5, 5), (24, 16, 1, 1)),
        (
            {"kernel_shape": [3, 3], "auto_pad": "SAME_UPPER", "strides": [2, 2]},
            (1, 3, 8, 8),
            (4, 3, 3, 3),
        ),
    ],
)
@pytest.mark.parametrize("opt", ["O0", "O3"])
def test_conv(
    attrs: dict[str, Any], x_shape: tuple[int, ...], w_shape: tuple[int, ...], opt: str, cache: Path
) -> None:
    model, feeds = op_model(
        "Conv", [f32(*x_shape), f32(*w_shape) * 0.3, f32(w_shape[0])], const={1, 2}, attrs=attrs
    )
    check(model, feeds, cache, opt=opt)


@pytest.mark.parametrize(
    "attrs",
    [
        {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]},
        {"kernel_shape": [2, 2], "strides": [2, 2]},
        {"kernel_shape": [3, 3], "strides": [2, 2], "ceil_mode": 1},
        {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "count_include_pad": 1},
        {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1], "ceil_mode": 1},
        {"kernel_shape": [2, 2], "dilations": [2, 2]},
    ],
)
@pytest.mark.parametrize("op", ["MaxPool", "AveragePool"])
@pytest.mark.parametrize("opt", ["O0", "O3"])
def test_pooling(op: str, attrs: dict[str, Any], opt: str, cache: Path) -> None:
    if op == "AveragePool" and "dilations" in attrs:
        pytest.skip("onnxruntime rejects dilated AveragePool at opset 14")
    if op == "MaxPool":
        attrs = {k: v for k, v in attrs.items() if k != "count_include_pad"}
    model, feeds = op_model(op, [f32(2, 3, 9, 10)], attrs=attrs)
    check(model, feeds, cache, opt=opt)


@pytest.mark.parametrize("op", ["GlobalAveragePool", "GlobalMaxPool"])
@pytest.mark.parametrize("opt", ["O0", "O3"])
def test_global_pooling(op: str, opt: str, cache: Path) -> None:
    model, feeds = op_model(op, [f32(2, 5, 4, 3)])
    check(model, feeds, cache, opt=opt)


# ------------------------------------------------------------------------------------------
# whole models
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("level", list(PIPELINES))
@pytest.mark.parametrize("name", sorted(samples.SAMPLES))
def test_samples_at_every_level(name: str, level: str, cache: Path) -> None:
    model = samples.SAMPLES[name]()
    graph = from_onnx(model)
    feeds = random_feeds(graph, 4, int_high=100)
    expected = run_graph(graph, feeds)
    compiled = compile_model(model, opt=level, cache_dir=cache)
    got = compiled.run(feeds)
    for k, v in expected.items():
        np.testing.assert_allclose(got[k], v, rtol=1e-4, atol=2e-5, err_msg=k)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("inplace", [True, False])
def test_memory_plans_give_identical_results(strategy: str, inplace: bool, cache: Path) -> None:
    model = samples.tiny_transformer(gelu="tanh", causal=True)
    graph = from_onnx(model)
    feeds = random_feeds(graph, 2)
    reference = compile_model(model, cache_dir=cache).run(feeds)
    got = compile_model(model, cache_dir=cache, strategy=strategy, inplace=inplace).run(feeds)
    for k in reference:
        np.testing.assert_array_equal(got[k], reference[k])


@pytest.mark.parametrize(
    "options",
    [
        BuildOptions(neon=False),
        BuildOptions(openmp=False),
        BuildOptions(opt_level="-O1", native=False),
    ],
    ids=["no-neon", "no-openmp", "O1-portable"],
)
def test_build_variants_agree(options: BuildOptions, cache: Path) -> None:
    model = samples.tiny_cnn()
    graph = from_onnx(model)
    feeds = random_feeds(graph, 5)
    expected = run_graph(graph, feeds)
    got = compile_model(model, cache_dir=cache, options=options).run(feeds)
    np.testing.assert_allclose(got["logits"], expected["logits"], rtol=1e-4, atol=1e-5)


def test_threads_and_blocking_do_not_change_results(cache: Path) -> None:
    model = samples.tiny_transformer(gelu="erf", causal=False, d_model=48, seq=20)
    graph = from_onnx(model)
    feeds = random_feeds(graph, 6)
    compiled = compile_model(model, cache_dir=cache)
    reference = compiled.run(feeds)["logits"]
    for threads, blocking in [(1, (8, 16, 1)), (3, (16, 32, 2)), (2, (64, 256, 8))]:
        compiled.set_threads(threads)
        compiled.set_blocking(*blocking)
        np.testing.assert_allclose(compiled.run(feeds)["logits"], reference, rtol=1e-5, atol=1e-6)
    assert compiled.max_threads() >= 1
    compiled.set_blocking(64, 256, 8)


# ------------------------------------------------------------------------------------------
# GEMM driver properties
# ------------------------------------------------------------------------------------------
def gemm_model(
    m: int, k: int, n: int, act: str | None, bias: bool, residual: bool
) -> onnx.ModelProto:
    nodes = [helper.make_node("MatMul", ["a", "w"], ["y0"])]
    inits = [numpy_helper.from_array(f32(k, n), "w")]
    cur = "y0"
    if bias:
        nodes.append(helper.make_node("Add", [cur, "b"], ["y1"]))
        inits.append(numpy_helper.from_array(f32(n), "b"))
        cur = "y1"
    inputs = [helper.make_tensor_value_info("a", TensorProto.FLOAT, [m, k])]
    if residual:
        nodes.append(helper.make_node("Add", [cur, "r"], ["y2"]))
        inputs.append(helper.make_tensor_value_info("r", TensorProto.FLOAT, [m, n]))
        cur = "y2"
    if act:
        nodes.append(helper.make_node(act, [cur], ["y3"]))
        cur = "y3"
    nodes.append(helper.make_node("Identity", [cur], ["y"]))
    graph = helper.make_graph(
        nodes, "g", inputs, [helper.make_tensor_value_info("y", TensorProto.FLOAT, [m, n])], inits
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    return model


@given(
    m=st.integers(1, 40),
    k=st.integers(1, 70),
    n=st.integers(1, 40),
    act=st.sampled_from([None, "Relu", "Tanh", "Sigmoid"]),
    bias=st.booleans(),
    residual=st.booleans(),
)
@settings(max_examples=25, deadline=None)
def test_fused_gemm_property(
    m: int,
    k: int,
    n: int,
    act: str | None,
    bias: bool,
    residual: bool,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    cache = tmp_path_factory.getbasetemp() / "mgc-gemm-cache"
    model = gemm_model(m, k, n, act, bias, residual)
    feeds = {"a": f32(m, k)}
    if residual:
        feeds["r"] = f32(m, n)
    compiled = compile_model(model, opt="O3", cache_dir=cache)
    compiled.set_blocking(8, 16, 1)  # small blocks exercise every edge path
    got = compiled.run(feeds)["y"]
    compiled.set_blocking(64, 256, 8)
    expected = ort_run(model, feeds)["y"]
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


@given(
    k=st.integers(1, 3),
    s=st.integers(1, 2),
    d=st.integers(1, 2),
    p=st.integers(0, 2),
    hw=st.integers(6, 11),
    group=st.sampled_from([1, 2, 4]),
    opt=st.sampled_from(["O0", "O3"]),
)
@settings(max_examples=25, deadline=None)
def test_conv_property(
    k: int,
    s: int,
    d: int,
    p: int,
    hw: int,
    group: int,
    opt: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    cache = tmp_path_factory.getbasetemp() / "mgc-conv-cache"
    model, feeds = op_model(
        "Conv",
        [f32(1, 4, hw, hw + 1), f32(4, 4 // group, k, k)],
        const={1},
        attrs={
            "kernel_shape": [k, k],
            "strides": [s, s],
            "dilations": [d, d],
            "pads": [p] * 4,
            "group": group,
        },
    )
    check(model, feeds, cache, opt=opt)


def test_pack_b_panels_layout() -> None:
    b = np.arange(5 * 14, dtype=np.float32).reshape(5, 14)
    packed = pack_b_panels(b)
    assert packed.shape == (2, 5, 12)
    np.testing.assert_array_equal(packed[0], b[:, :12])
    np.testing.assert_array_equal(packed[1, :, :2], b[:, 12:])
    assert not packed[1, :, 2:].any()


# ------------------------------------------------------------------------------------------
# runtime behaviour and errors
# ------------------------------------------------------------------------------------------
def test_gather_out_of_range_raises(cache: Path) -> None:
    model = samples.tiny_transformer(vocab=16)
    compiled = compile_model(model, cache_dir=cache)
    with pytest.raises(GraphError, match="Gather"):
        compiled.run({"input_ids": np.full((1, 8), 99, np.int64)})
    with pytest.raises(GraphError, match="missing"):
        compiled.run({})
    with pytest.raises(GraphError, match="expected"):
        compiled.run({"input_ids": np.zeros((1, 8), np.int32)})


def test_save_load_and_profile(cache: Path, tmp_path: Path) -> None:
    model = samples.tiny_cnn()
    graph = from_onnx(model)
    feeds = random_feeds(graph, 7)
    optimize(graph, "O3")
    compiled = compile_graph(graph, cache_dir=cache, profile=True)
    first = compiled.run(feeds)
    prof = compiled.profile()
    assert len(prof) == len(graph.nodes) and all(t >= 0 for _, _, t in prof)
    assert compiled.report is not None and compiled.report.kernels["gemm"] == 9
    compiled.save(tmp_path / "saved")
    loaded = CompiledModel.load(tmp_path / "saved")
    np.testing.assert_array_equal(loaded.run(feeds)["logits"], first["logits"])
    assert [(op, name) for op, name, _ in loaded.profile()] == [(op, name) for op, name, _ in prof]
    plain = compile_graph(graph, cache_dir=cache)
    with pytest.raises(CodegenError, match="profile"):
        plain.profile()
    times = loaded.benchmark(feeds, warmup=1, repeat=3)
    assert len(times) == 3
    # a cached rebuild of the same graph
    again = compile_graph(graph, cache_dir=cache, profile=True)
    assert again.report is not None and again.report.cached


def test_output_that_is_an_input_or_constant_is_copied(cache: Path) -> None:
    g = Graph("copies")
    x = g.add_input("x", TensorType("float32", (2, 3)))
    view = g.add_node("Reshape", [x], attrs={"shape": [3, 2]})
    view.output.type = TensorType("float32", (3, 2))
    g.add_output(view.output, "reshaped")
    g.add_output(g.constant(np.arange(4, dtype=np.float32), "c"), "const")
    compiled = compile_graph(g, cache_dir=cache)
    xv = f32(2, 3)
    out = compiled.run({"x": xv})
    np.testing.assert_array_equal(out["reshaped"], xv.reshape(3, 2))
    np.testing.assert_array_equal(out["const"], np.arange(4, dtype=np.float32))


def test_unsupported_programs_raise() -> None:
    model, _ = op_model("Trilu", [f32(3, 3)])
    with pytest.raises(CodegenError, match="Trilu"):
        generate_c(from_onnx(model))
    model, _ = op_model(
        "Pad", [f32(2, 3), np.asarray([1, 1, 1, 1], np.int64)], const={1}, attrs={"mode": "reflect"}
    )
    with pytest.raises(CodegenError, match="reflect"):
        generate_c(from_onnx(model))
    model, _ = op_model("Sqrt", [np.ones((2,), np.int64)])
    graph = from_onnx(model)
    with pytest.raises(CodegenError, match="Sqrt on int64"):
        generate_c(graph)
    g = Graph()
    h = g.add_input("h", TensorType("float16", (2,)))
    node = g.add_node("Relu", [h])
    node.output.type = TensorType("float16", (2,))
    g.add_output(node.output)
    with pytest.raises(CodegenError, match="float16"):
        generate_c(g)


def test_generated_source_shape() -> None:
    graph = from_onnx(samples.tiny_cnn())
    optimize(graph, "O3")
    generated = generate_c(graph)
    assert "int mgc_run(" in generated.source
    assert generated.kernels["gemm"] == 9 and generated.kernels["depthwise"] == 1
    assert generated.arena_bytes == max(generated.plan.arena_bytes, 64)
    blob = generated.weights.blob()
    assert blob.nbytes == generated.weights.nbytes and blob.nbytes % 64 == 0
    assert len(generated.weights.digest()) == 24


def test_compile_errors_are_reported(cache: Path) -> None:
    from ml_graph_compiler.codegen.build import build_library

    with pytest.raises(CodegenError, match="compilation failed"):
        build_library("this is not C", cache_dir=cache)
