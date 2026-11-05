"""Test helpers: one-operator ONNX models and comparison with onnxruntime."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from ml_graph_compiler.interpreter import run_graph
from ml_graph_compiler.onnx_import import from_onnx

_ONNX_TYPES = {
    np.dtype(np.float32): onnx.TensorProto.FLOAT,
    np.dtype(np.int64): onnx.TensorProto.INT64,
    np.dtype(np.bool_): onnx.TensorProto.BOOL,
    np.dtype(np.int32): onnx.TensorProto.INT32,
}


def op_model(
    op: str,
    inputs: list[np.ndarray | None],
    const: set[int] | None = None,
    attrs: dict[str, Any] | None = None,
    n_outputs: int = 1,
    opset: int = 14,
) -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    """A model with a single node; inputs listed in ``const`` become initializers."""
    const = const or set()
    names, graph_inputs, inits, feeds = [], [], [], {}
    for i, array in enumerate(inputs):
        if array is None:
            names.append("")
            continue
        name = f"in{i}"
        names.append(name)
        if i in const:
            inits.append(numpy_helper.from_array(array, name))
        else:
            graph_inputs.append(
                helper.make_tensor_value_info(name, _ONNX_TYPES[array.dtype], list(array.shape))
            )
            feeds[name] = array
    outs = [f"out{i}" for i in range(n_outputs)]
    node = helper.make_node(op, names, outs, **(attrs or {}))
    graph = helper.make_graph(
        [node],
        "single",
        graph_inputs,
        [helper.make_tensor_value_info(o, onnx.TensorProto.UNDEFINED, None) for o in outs],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    return model, feeds


def ort_run(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, feeds), strict=True))


def assert_matches_ort(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-5,
    passes: str | None = None,
) -> dict[str, np.ndarray]:
    graph = from_onnx(model)
    if passes:
        from ml_graph_compiler.passes import optimize

        optimize(graph, passes)
    ours = run_graph(graph, feeds)
    ref = ort_run(model, feeds)
    assert set(ours) == set(ref)
    for name, expected in ref.items():
        got = ours[name]
        assert got.shape == expected.shape, (name, got.shape, expected.shape)
        assert got.dtype == expected.dtype, (name, got.dtype, expected.dtype)
        np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol, err_msg=name)
    return ours
