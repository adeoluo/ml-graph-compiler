"""MLIR export: printed text, mlir-opt verification and execution against onnxruntime.

Verification and execution need mlir-opt, mlir-translate and clang from one LLVM install; those
tests are skipped when the tools are missing (the text tests always run).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, numpy_helper

from helpers import op_model, ort_run
from ml_graph_compiler import samples
from ml_graph_compiler.cli import main
from ml_graph_compiler.errors import CodegenError, GraphError
from ml_graph_compiler.interpreter import random_feeds, run_graph
from ml_graph_compiler.ir import Graph, TensorType
from ml_graph_compiler.mlir import compile_mlir, find_mlir_tools, mlir_verify, to_mlir
from ml_graph_compiler.mlir.emit import (
    attribute,
    broadcast_map,
    dense_literal,
    float_literal,
    snake,
    tensor_type,
)
from ml_graph_compiler.onnx_import import from_onnx
from ml_graph_compiler.passes import optimize

rng = np.random.default_rng(3)

try:
    TOOLS = find_mlir_tools()
except CodegenError:
    TOOLS = None
needs_mlir = pytest.mark.skipif(TOOLS is None, reason="LLVM MLIR tools are not installed")


@pytest.fixture(scope="session")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("mlir-cache")


def f32(*shape: int) -> np.ndarray:
    return np.asarray(rng.standard_normal(shape), dtype=np.float32)


def pos(*shape: int) -> np.ndarray:
    return np.asarray(rng.random(shape) + 0.5, dtype=np.float32)


# ------------------------------------------------------------------------------------------
# printing helpers
# ------------------------------------------------------------------------------------------
def test_types_and_literals() -> None:
    assert tensor_type(TensorType("float32", (2, 3))) == "tensor<2x3xf32>"
    assert tensor_type(TensorType("bool", ())) == "tensor<i1>"
    assert tensor_type(TensorType("int64", (5,))) == "tensor<5xi64>"
    assert float_literal(1.5) == "1.500000000e+00"
    assert float_literal(-math.inf) == "0xFF800000"
    assert float_literal(math.inf, "float64") == "0x7FF0000000000000"
    assert dense_literal(np.zeros((2, 2), np.float32), "float32") == "dense<0.000000000e+00>"
    assert dense_literal(np.asarray([[1, 2]], np.int64), "int64") == "dense<[[1, 2]]>"
    assert dense_literal(np.asarray([True, False]), "bool") == "dense<[true, false]>"
    assert snake("FusedConv") == "fused_conv" and snake("Relu") == "relu"


def test_attributes() -> None:
    assert attribute(3) == "3 : i64"
    assert attribute(True) == "true"
    assert attribute(0.5) == "5.000000000e-01 : f32"
    assert attribute("NHWC") == '"NHWC"'
    assert attribute('a"b') == '"a\\"b"'
    assert attribute([1, 2]) == "array<i64: 1, 2>"
    assert attribute([]) == "array<i64>"
    assert attribute([0.5]) == "array<f32: 5.000000000e-01>"
    assert attribute(["Add", {}, [0, 1]]) == '["Add", {}, array<i64: 0, 1>]'
    assert attribute({"k-1": 2}) == "{k_1 = 2 : i64}"
    assert attribute(np.asarray([1.0], np.float32)) == "dense<1.000000000e+00> : tensor<1xf32>"
    assert attribute(object()).startswith('"<object')


def test_broadcast_map() -> None:
    assert broadcast_map((4,), (2, 3, 4)) == ["d2"]
    assert broadcast_map((2, 1, 4), (2, 3, 4)) == ["d0", "0", "d2"]
    assert broadcast_map((), (2, 2)) == []
    assert broadcast_map((1, 1), (1, 1)) == ["d0", "d1"]


def test_mgc_form_is_one_op_per_node() -> None:
    graph = from_onnx(samples.SAMPLES["tiny-gpt"]())
    optimize(graph, "O3")
    module = to_mlir(graph, "mgc")
    text = module.text
    assert text.count('"mgc.') == len(graph.nodes)
    assert '"mgc.fused_gemm"' in text and '"mgc.layer_normalization"' in text
    assert "func.func @main(" in text and "return " in text
    assert module.fallback == {} and module.lowered == {}


def test_linalg_form_lowers_every_sample_operator() -> None:
    for name, build in samples.SAMPLES.items():
        for level in ("O0", "O3"):
            graph = from_onnx(build())
            optimize(graph, level)
            module = to_mlir(graph)
            assert module.fully_lowered, (name, level, module.fallback)
            assert '"mgc.' not in module.text
            assert "linalg.generic" in module.text


def test_resources_and_elision() -> None:
    graph = from_onnx(samples.tiny_cnn())
    full = to_mlir(graph)
    assert full.resources > 0 and full.resource_bytes > 0
    assert "dialect_resources" in full.text and "mgc_blob_0:" in full.text
    elided = to_mlir(graph, elide_constants=True)
    assert elided.resources == 0 and "dense_resource<__elided__>" in elided.text
    assert len(elided.text) < len(full.text)


def test_unknown_form_and_fallback() -> None:
    with pytest.raises(CodegenError, match="unknown MLIR form"):
        to_mlir(Graph(), "tosa")
    model, _ = op_model("Trilu", [f32(3, 3)])
    module = to_mlir(from_onnx(model))
    assert module.fallback == {"Trilu": 1} and '"mgc.trilu"' in module.text
    with pytest.raises(CodegenError, match="Trilu"):
        compile_mlir(from_onnx(model), tools=TOOLS)


def test_failed_lowering_rolls_back_partial_output() -> None:
    model, _ = op_model(
        "Pad", [f32(2, 3), np.asarray([1, 1, 1, 1], np.int64)], const={1}, attrs={"mode": "reflect"}
    )
    module = to_mlir(from_onnx(model))
    assert module.fallback == {"Pad": 1}
    assert "tensor.pad" not in module.text


def test_integer_reduce_mean_and_argmax_fall_back() -> None:
    model, _ = op_model(
        "ReduceMean", [np.arange(6, dtype=np.int64).reshape(2, 3)], attrs={"axes": [1]}
    )
    assert to_mlir(from_onnx(model)).fallback == {"ReduceMean": 1}
    model, _ = op_model("ArgMax", [np.arange(6, dtype=np.int64).reshape(2, 3)], attrs={"axis": 1})
    assert to_mlir(from_onnx(model)).fallback == {"ArgMax": 1}


# ------------------------------------------------------------------------------------------
# mlir-opt verification and execution
# ------------------------------------------------------------------------------------------
def run_both(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray],
    cache: Path,
    opt: str = "O0",
    rtol: float = 1e-4,
    atol: float = 2e-5,
) -> None:
    graph = from_onnx(model)
    optimize(graph, opt)
    module = to_mlir(graph)
    assert module.fully_lowered, module.fallback
    mlir_verify(module.text, TOOLS)
    mlir_verify(to_mlir(graph, "mgc").text, TOOLS, allow_unregistered=True)
    got = compile_mlir(graph, cache_dir=cache, tools=TOOLS).run(feeds)
    expected = ort_run(model, feeds)
    for name, value in expected.items():
        assert got[name].shape == value.shape and got[name].dtype == value.dtype, name
        np.testing.assert_allclose(got[name], value, rtol=rtol, atol=atol, err_msg=name)


CASES: list[tuple[str, list[Any], set[int], dict[str, Any], int]] = [
    ("Relu", [f32(3, 7)], set(), {}, 14),
    ("Sigmoid", [f32(3, 7)], set(), {}, 14),
    ("Tanh", [f32(3, 7)], set(), {}, 14),
    ("Erf", [f32(3, 7)], set(), {}, 14),
    ("Exp", [f32(3, 7)], set(), {}, 14),
    ("Log", [pos(3, 7)], set(), {}, 14),
    ("Sqrt", [pos(3, 7)], set(), {}, 14),
    ("Reciprocal", [pos(3, 7)], set(), {}, 14),
    ("Neg", [f32(3, 7)], set(), {}, 14),
    ("Abs", [f32(3, 7)], set(), {}, 14),
    ("Floor", [f32(3, 7) * 3], set(), {}, 14),
    ("Ceil", [f32(3, 7) * 3], set(), {}, 14),
    ("Sin", [f32(3, 7)], set(), {}, 14),
    ("Cos", [f32(3, 7)], set(), {}, 14),
    ("Softplus", [f32(3, 7) * 30], set(), {}, 14),
    ("LeakyRelu", [f32(3, 7)], set(), {"alpha": 0.2}, 14),
    ("HardSigmoid", [f32(3, 7)], set(), {}, 14),
    ("Gelu", [f32(3, 7)], set(), {}, 20),
    ("Gelu", [f32(3, 7)], set(), {"approximate": "tanh"}, 20),
    ("Clip", [f32(3, 5), np.asarray(-0.5, np.float32), np.asarray(0.5, np.float32)], set(), {}, 14),
    ("Add", [f32(2, 3, 4), f32(4)], set(), {}, 14),
    ("Sub", [f32(2, 1, 4), f32(3, 1)], set(), {}, 14),
    ("Mul", [f32(5), f32(5)], set(), {}, 14),
    ("Div", [f32(2, 2), pos()], set(), {}, 14),
    ("Max", [f32(2, 3), f32(3)], set(), {}, 14),
    ("Min", [f32(2, 3), f32(3)], set(), {}, 14),
    ("Pow", [pos(2, 3), pos(3)], set(), {}, 14),
    (
        "Equal",
        [
            rng.integers(0, 3, (3, 4)).astype(np.float32),
            rng.integers(0, 3, (4,)).astype(np.float32),
        ],
        set(),
        {},
        14,
    ),
    ("Less", [f32(3, 4), f32(4)], set(), {}, 14),
    ("GreaterOrEqual", [f32(3, 4), f32(4)], set(), {}, 14),
    ("And", [rng.random((3, 4)) > 0.5, rng.random((4,)) > 0.5], set(), {}, 14),
    ("Xor", [rng.random((3, 4)) > 0.5, rng.random((4,)) > 0.5], set(), {}, 14),
    ("Not", [rng.random((5,)) > 0.5], set(), {}, 14),
    ("Where", [rng.random((3, 1)) > 0.5, f32(3, 4), f32(4)], set(), {}, 14),
    ("Cast", [f32(6) * 5], set(), {"to": TensorProto.INT64}, 14),
    ("Cast", [np.arange(6, dtype=np.int64)], set(), {"to": TensorProto.FLOAT}, 14),
    ("Cast", [np.asarray([0, 1, 2, 0], np.float32)], set(), {"to": TensorProto.BOOL}, 14),
    ("Add", [np.asarray([7, -7, 9], np.int64), np.asarray([2, 2, -4], np.int64)], set(), {}, 14),
    ("Div", [np.asarray([7, -7, 9], np.int64), np.asarray([2, 2, -4], np.int64)], set(), {}, 14),
    ("MatMul", [f32(5, 7), f32(7, 3)], set(), {}, 14),
    ("MatMul", [f32(2, 3, 5, 7), f32(7, 4)], {1}, {}, 14),
    ("MatMul", [f32(2, 1, 5, 7), f32(3, 7, 4)], set(), {}, 14),
    ("MatMul", [f32(7), f32(7, 5)], set(), {}, 14),
    ("MatMul", [f32(4, 7), f32(7)], set(), {}, 14),
    (
        "Gemm",
        [f32(5, 4), f32(3, 5), f32(1, 3)],
        {1, 2},
        {"transA": 1, "transB": 1, "alpha": 0.5, "beta": 2.0},
        14,
    ),
    ("Gemm", [f32(4, 5), f32(5, 3)], set(), {}, 14),
    (
        "Conv",
        [f32(1, 4, 9, 8), f32(6, 4, 3, 3), f32(6)],
        {1, 2},
        {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]},
        14,
    ),
    (
        "Conv",
        [f32(2, 3, 11, 9), f32(5, 3, 3, 3)],
        {1},
        {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 0, 1, 2]},
        14,
    ),
    (
        "Conv",
        [f32(1, 2, 9, 9), f32(4, 2, 2, 3)],
        {1},
        {"kernel_shape": [2, 3], "dilations": [2, 1]},
        14,
    ),
    (
        "Conv",
        [f32(1, 4, 6, 6), f32(6, 2, 3, 3)],
        {1},
        {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "group": 2},
        14,
    ),
    (
        "Conv",
        [f32(2, 4, 9, 9), f32(8, 1, 3, 3)],
        {1},
        {"kernel_shape": [3, 3], "group": 4, "strides": [2, 2]},
        14,
    ),
    (
        "MaxPool",
        [f32(2, 3, 9, 10)],
        set(),
        {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]},
        14,
    ),
    (
        "MaxPool",
        [f32(2, 3, 9, 10)],
        set(),
        {"kernel_shape": [3, 3], "strides": [2, 2], "ceil_mode": 1},
        14,
    ),
    ("AveragePool", [f32(2, 3, 9, 10)], set(), {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}, 14),
    (
        "AveragePool",
        [f32(2, 3, 9, 10)],
        set(),
        {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "count_include_pad": 1},
        14,
    ),
    (
        "AveragePool",
        [f32(2, 3, 9, 10)],
        set(),
        {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1], "ceil_mode": 1},
        14,
    ),
    ("GlobalAveragePool", [f32(2, 5, 4, 3)], set(), {}, 14),
    ("GlobalMaxPool", [f32(2, 5, 4, 3)], set(), {}, 14),
    ("ReduceMean", [f32(2, 3, 4)], set(), {"axes": [1, 2], "keepdims": 1}, 14),
    ("ReduceSum", [f32(2, 3, 4), np.asarray([0, 1], np.int64)], {1}, {"keepdims": 0}, 14),
    ("ReduceMax", [f32(2, 3, 4)], set(), {"axes": [1]}, 14),
    ("ReduceMin", [f32(2, 3, 4)], set(), {"axes": [2]}, 14),
    ("ReduceProd", [pos(2, 3)], set(), {"axes": [1]}, 14),
    (
        "ReduceSum",
        [np.arange(12, dtype=np.int64).reshape(3, 4), np.asarray([1], np.int64)],
        {1},
        {},
        14,
    ),
    ("ArgMax", [f32(3, 7)], set(), {"axis": 1, "keepdims": 0}, 14),
    ("Softmax", [f32(2, 3, 9)], set(), {"axis": -1}, 14),
    ("Softmax", [f32(2, 7, 3)], set(), {"axis": 1}, 14),
    ("LogSoftmax", [f32(4, 9)], set(), {"axis": -1}, 14),
    (
        "LayerNormalization",
        [f32(2, 5, 16), pos(16), f32(16)],
        {1, 2},
        {"axis": -1, "epsilon": 1e-5},
        17,
    ),
    ("LayerNormalization", [f32(2, 3, 4), pos(3, 4)], set(), {"axis": 1}, 17),
    ("BatchNormalization", [f32(2, 3, 4, 5), pos(3), f32(3), f32(3), pos(3)], {1, 2, 3, 4}, {}, 14),
    ("Transpose", [f32(2, 3, 4, 5)], set(), {"perm": [0, 2, 3, 1]}, 14),
    ("Reshape", [f32(2, 3, 4), np.asarray([4, -1], np.int64)], {1}, {}, 14),
    ("Flatten", [f32(2, 3, 4)], set(), {"axis": 2}, 14),
    ("Unsqueeze", [f32(2, 3), np.asarray([0, 3], np.int64)], {1}, {}, 14),
    ("Squeeze", [f32(2, 1, 3), np.asarray([1], np.int64)], {1}, {}, 14),
    ("Squeeze", [f32(1, 1)], set(), {}, 14),
    ("Expand", [f32(3, 1), np.asarray([2, 3, 4], np.int64)], {1}, {}, 14),
    ("Tile", [f32(2, 3), np.asarray([2, 2], np.int64)], {1}, {}, 14),
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
        14,
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
        14,
    ),
    ("Concat", [f32(2, 3), f32(2, 5), f32(2, 1)], set(), {"axis": 1}, 14),
    (
        "Pad",
        [f32(2, 3), np.asarray([1, 0, 2, 3], np.int64), np.asarray(1.5, np.float32)],
        {1, 2},
        {},
        14,
    ),
    ("Gather", [f32(10, 4), np.asarray([[1, -1], [3, 0]], np.int64)], set(), {"axis": 0}, 14),
    ("Gather", [f32(3, 10, 2), np.asarray([4, 2], np.int64)], {1}, {"axis": 1}, 14),
    ("Identity", [f32(4)], set(), {}, 14),
]


@needs_mlir
@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_operator_executes_like_onnxruntime(
    case: tuple[str, list[Any], set[int], dict[str, Any], int], cache: Path
) -> None:
    op, inputs, const, attrs, opset = case
    model, feeds = op_model(op, inputs, const=const, attrs=attrs, opset=opset)
    run_both(model, feeds, cache)


@needs_mlir
def test_split_dropout_and_constant_folding(cache: Path) -> None:
    model, feeds = op_model(
        "Split",
        [f32(4, 9), np.asarray([2, 3, 4], np.int64)],
        const={1},
        attrs={"axis": 1},
        n_outputs=3,
    )
    run_both(model, feeds, cache)
    model, feeds = op_model("Dropout", [f32(3, 4)], n_outputs=2)
    run_both(model, feeds, cache)
    model, feeds = op_model("Shape", [f32(2, 3, 4)])
    run_both(model, feeds, cache)
    model, feeds = op_model(
        "ConstantOfShape",
        [np.asarray([2, 3], np.int64)],
        const={0},
        attrs={"value": numpy_helper.from_array(np.asarray([7], np.int64))},
    )
    run_both(model, feeds, cache)


@needs_mlir
@pytest.mark.parametrize("name", list(samples.SAMPLES))
@pytest.mark.parametrize("opt", ["O0", "O2", "O3"])
def test_samples_execute_like_the_interpreter(name: str, opt: str, cache: Path) -> None:
    graph = from_onnx(samples.SAMPLES[name]())
    feeds = random_feeds(graph, 5, int_high=50)
    expected = run_graph(graph, feeds)
    optimize(graph, opt)
    got = compile_mlir(graph, cache_dir=cache, tools=TOOLS).run(feeds)
    for key, value in expected.items():
        np.testing.assert_allclose(got[key], value, rtol=1e-4, atol=2e-5, err_msg=key)


@needs_mlir
def test_executable_checks_inputs_and_frees_outputs(cache: Path) -> None:
    graph = from_onnx(samples.mlp())
    exe = compile_mlir(graph, cache_dir=cache, tools=TOOLS)
    with pytest.raises(GraphError, match="missing"):
        exe.run({})
    with pytest.raises(GraphError, match="expected"):
        exe.run({"x": np.zeros((2, 2), np.float32)})
    feeds = random_feeds(graph, 1)
    first = exe.run(feeds)
    for _ in range(20):  # repeated calls must not leak or corrupt outputs
        again = exe.run(feeds)
    np.testing.assert_array_equal(first["y"], again["y"])
    cached = compile_mlir(graph, cache_dir=cache, tools=TOOLS)
    assert cached.library == exe.library


@needs_mlir
def test_output_that_is_an_input_or_constant(cache: Path) -> None:
    g = Graph("copies")
    x = g.add_input("x", TensorType("float32", (2, 3)))
    g.add_output(x, "same")
    g.add_output(g.constant(np.arange(4, dtype=np.float32), "c"), "const")
    exe = compile_mlir(g, cache_dir=cache, tools=TOOLS)
    xv = f32(2, 3)
    out = exe.run({"x": xv})
    np.testing.assert_array_equal(out["same"], xv)
    np.testing.assert_array_equal(out["const"], np.arange(4, dtype=np.float32))


@needs_mlir
def test_verify_reports_diagnostics() -> None:
    with pytest.raises(CodegenError, match="mlir-opt failed"):
        mlir_verify("func.func @f() { return %x : f32 }", TOOLS)


@needs_mlir
def test_cli_writes_and_verifies(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "m.mlir"
    assert main(["mlir", "tiny-gpt", "-o", str(out), "--verify", "--elide-constants"]) == 0
    assert "verified by mlir-opt" in capsys.readouterr().out
    assert "linalg.generic" in out.read_text()
    assert main(["mlir", "mlp", "--dialect", "mgc", "-O", "O0"]) == 0
    assert '"mgc.mat_mul"' in capsys.readouterr().out


def test_cli_prints_without_tools(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["mlir", "mlp", "--dialect", "mgc"]) == 0
    assert "func.func @main" in capsys.readouterr().out


def test_hostile_names_cannot_break_mlir_line_comments() -> None:
    from onnx import helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Relu", ["x"], ["y"], name="r\nfunc.func @evil() {")
    model = helper.make_model(
        helper.make_graph([node], "g\nmodule {", [x], [y]),
        opset_imports=[helper.make_opsetid("", 14)],
    )
    text = to_mlir(from_onnx(model)).text
    assert "\nfunc.func @evil" not in text
    assert "\nmodule {" not in text
