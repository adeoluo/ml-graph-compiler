"""ONNX import, shape inference and the reference interpreter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from helpers import assert_matches_ort, ort_run
from ml_graph_compiler import samples
from ml_graph_compiler.errors import GraphError, ModelImportError, ShapeError, UnsupportedOpError
from ml_graph_compiler.interpreter import Interpreter, Profile, check_feeds, random_feeds, run_graph
from ml_graph_compiler.ir import TensorType
from ml_graph_compiler.onnx_import import from_onnx, opset_version
from ml_graph_compiler.shape_inference import infer_shapes


def model_of(nodes, inputs, outputs, inits=(), opset=14, domain=""):  # type: ignore[no-untyped-def]
    graph = helper.make_graph(nodes, "m", inputs, outputs, list(inits))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid(domain, opset)])
    model.ir_version = 8
    return model


def vi(name: str, shape: list[object], elem: int = TensorProto.FLOAT) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem, shape)


@pytest.mark.parametrize("name", sorted(samples.SAMPLES))
def test_samples_match_onnxruntime(name: str) -> None:
    model = samples.SAMPLES[name]()
    graph = from_onnx(model)
    feeds = random_feeds(graph, 1, int_high=50)
    assert_matches_ort(model, feeds, rtol=1e-4, atol=1e-5)


def test_import_from_path(tmp_path: Path) -> None:
    path = tmp_path / "mlp.onnx"
    onnx.save(samples.mlp(), str(path))
    graph = from_onnx(path)
    assert graph.inputs[0].type == TensorType("float32", (4, 16))
    assert graph.opset == 14


def test_symbolic_dims_and_input_shapes() -> None:
    model = model_of(
        [helper.make_node("Relu", ["x"], ["y"])], [vi("x", ["batch", 3])], [vi("y", ["batch", 3])]
    )
    with pytest.raises(ModelImportError, match="dynamic dimension 'batch'"):
        from_onnx(model)
    assert from_onnx(model, dims={"batch": 5}).outputs[0].type == TensorType("float32", (5, 3))
    assert from_onnx(model, input_shapes={"x": (7, 3)}).outputs[0].tensor_type.shape == (7, 3)
    with pytest.raises(ModelImportError, match="unknown inputs"):
        from_onnx(model, input_shapes={"nope": (1,)}, dims={"batch": 1})


def test_unknown_dimension_without_name() -> None:
    model = model_of(
        [helper.make_node("Relu", ["x"], ["y"])], [vi("x", [None, 3])], [vi("y", None)]
    )
    with pytest.raises(ModelImportError, match="'\\?'"):
        from_onnx(model)


def test_opset_bounds_and_domains() -> None:
    nodes = [helper.make_node("Relu", ["x"], ["y"])]
    with pytest.raises(ModelImportError, match="opset"):
        from_onnx(model_of(nodes, [vi("x", [2])], [vi("y", [2])], opset=6))
    custom = model_of(
        [helper.make_node("Foo", ["x"], ["y"], domain="com.example")],
        [vi("x", [2])],
        [vi("y", [2])],
    )
    custom.opset_import.append(helper.make_opsetid("com.example", 1))
    with pytest.raises(ModelImportError, match="domain"):
        from_onnx(custom)
    other = model_of(nodes, [vi("x", [2])], [vi("y", [2])], domain="com.example")
    assert opset_version(other) == 21


def test_unsupported_input_type_and_undefined_values() -> None:
    nodes = [helper.make_node("Relu", ["x"], ["y"])]
    with pytest.raises(ModelImportError, match="element type"):
        from_onnx(model_of(nodes, [vi("x", [2], TensorProto.STRING)], [vi("y", [2])]))
    bad = [helper.make_node("Relu", ["missing"], ["y"])]
    with pytest.raises(ModelImportError, match="undefined value"):
        from_onnx(model_of(bad, [vi("x", [2])], [vi("y", [2])]))
    with pytest.raises(ModelImportError, match="never defined"):
        from_onnx(model_of(nodes, [vi("x", [2])], [vi("z", [2])]))


def test_unsupported_operator() -> None:
    nodes = [helper.make_node("NonMaxSuppression", ["x"], ["y"])]
    with pytest.raises(UnsupportedOpError):
        from_onnx(model_of(nodes, [vi("x", [2])], [vi("y", [2])]))


def test_control_flow_is_rejected() -> None:
    body = helper.make_graph([], "b", [], [])
    node = helper.make_node("If", ["c"], ["y"], then_branch=body, else_branch=body)
    with pytest.raises(ModelImportError, match="control flow"):
        from_onnx(model_of([node], [vi("c", [], TensorProto.BOOL)], [vi("y", [2])]))


@pytest.mark.parametrize(
    "attr, expected",
    [
        ({"value": numpy_helper.from_array(np.arange(3, dtype=np.float32))}, [0.0, 1.0, 2.0]),
        ({"value_float": 2.5}, 2.5),
        ({"value_floats": [1.0, 2.0]}, [1.0, 2.0]),
        ({"value_int": 4}, 4),
        ({"value_ints": [1, 2]}, [1, 2]),
    ],
)
def test_constant_node_forms(attr: dict[str, object], expected: object) -> None:
    nodes = [
        helper.make_node("Constant", [], ["c"], **attr),
        helper.make_node("Identity", ["c"], ["y"]),
    ]
    graph = from_onnx(model_of(nodes, [], [vi("y", None)]))
    out = run_graph(graph, {})["y"]
    assert out.tolist() == expected


def test_constant_node_unsupported_form() -> None:
    node = helper.make_node("Constant", [], ["c"], value_string="hi")
    with pytest.raises(ModelImportError, match="Constant"):
        from_onnx(model_of([node], [], [vi("c", None)]))


def test_string_and_tensor_list_attributes_import() -> None:
    nodes = [helper.make_node("Gelu", ["x"], ["y"], approximate="tanh")]
    graph = from_onnx(model_of(nodes, [vi("x", [3])], [vi("y", [3])], opset=20))
    assert graph.nodes[0].attrs["approximate"] == "tanh"


def test_legacy_softmax_last_axis_and_rejection() -> None:
    nodes = [helper.make_node("Softmax", ["x"], ["y"], axis=1)]
    model = model_of(nodes, [vi("x", [2, 5])], [vi("y", [2, 5])], opset=11)
    graph = from_onnx(model)
    assert graph.nodes[0].attrs["axis"] == -1
    feeds = {"x": np.random.default_rng(0).standard_normal((2, 5)).astype(np.float32)}
    np.testing.assert_allclose(run_graph(graph, feeds)["y"], ort_run(model, feeds)["y"], rtol=1e-6)
    model3 = model_of(nodes, [vi("x", [2, 3, 4])], [vi("y", [2, 3, 4])], opset=11)
    with pytest.raises(UnsupportedOpError, match="2-D coercion"):
        from_onnx(model3)


def test_optional_inputs_and_unused_outputs() -> None:
    nodes = [
        helper.make_node("Clip", ["x", "", "hi"], ["c"]),
        helper.make_node("Dropout", ["c"], ["y", ""]),
    ]
    hi = numpy_helper.from_array(np.asarray(0.5, np.float32), "hi")
    model = model_of(nodes, [vi("x", [4])], [vi("y", [4])], inits=[hi])
    feeds = {"x": np.asarray([-1, 0, 1, 2], np.float32)}
    out = assert_matches_ort(model, feeds)
    assert out["y"].tolist() == [-1, 0, 0.5, 0.5]


def test_shape_arithmetic_is_partially_evaluated() -> None:
    graph = from_onnx(samples.tiny_transformer())
    reshapes = [n for n in graph.nodes if n.op == "Reshape"]
    assert reshapes and all(n.output.type is not None for n in reshapes)


def test_infer_shapes_reports_node_and_missing_output_type() -> None:
    graph = from_onnx(samples.mlp())
    graph.nodes[0].input(1).data = np.zeros((3, 3), np.float32)
    graph.nodes[0].input(1).type = TensorType("float32", (3, 3))
    with pytest.raises(ShapeError, match="MatMul node"):
        infer_shapes(graph)


def test_check_feeds_validation() -> None:
    graph = from_onnx(samples.mlp())
    good = random_feeds(graph)
    assert check_feeds(graph, good)["x"].flags.c_contiguous
    with pytest.raises(GraphError, match="missing"):
        check_feeds(graph, {})
    with pytest.raises(GraphError, match="shape"):
        check_feeds(graph, {"x": np.zeros((1, 16), np.float32)})
    with pytest.raises(GraphError, match="expected float32"):
        check_feeds(graph, {"x": np.zeros((4, 16), np.float64)})


def test_random_feeds_types() -> None:
    model = model_of(
        [
            helper.make_node("Where", ["c", "x", "x"], ["y"]),
            helper.make_node("Cast", ["i"], ["j"], to=TensorProto.FLOAT),
        ],
        [vi("c", [3], TensorProto.BOOL), vi("x", [3]), vi("i", [3], TensorProto.INT64)],
        [vi("y", [3]), vi("j", [3])],
    )
    graph = from_onnx(model)
    feeds = random_feeds(graph, 0, int_high=7)
    assert feeds["c"].dtype == np.bool_ and feeds["i"].max() < 7
    assert_matches_ort(model, feeds)


def test_interpreter_profile_and_release() -> None:
    graph = from_onnx(samples.tiny_cnn())
    profile = Profile()
    Interpreter(graph).run(random_feeds(graph), profile)
    assert profile.peak_live_bytes > 0
    assert set(profile.seconds_by_op) >= {"Conv", "BatchNormalization"}


def test_interpreter_detects_shape_mismatch_and_bad_order() -> None:
    graph = from_onnx(samples.mlp())
    graph.nodes[0].output.type = TensorType("float32", (1, 1))
    with pytest.raises(GraphError, match="produced shape"):
        run_graph(graph, random_feeds(graph))
    graph = from_onnx(samples.mlp())
    graph.nodes.reverse()
    with pytest.raises(GraphError, match="before it is defined"):
        run_graph(graph, random_feeds(graph))


def test_gather_out_of_range_is_an_error() -> None:
    graph = from_onnx(samples.tiny_transformer(vocab=16))
    with pytest.raises(ShapeError, match="out of range"):
        run_graph(graph, {"input_ids": np.full((1, 8), 99, np.int64)})
