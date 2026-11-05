"""ONNX model importer: protobuf graph to the SSA IR, with static shapes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from ml_graph_compiler.errors import ModelImportError, UnsupportedOpError
from ml_graph_compiler.ir import Graph, TensorType, Value
from ml_graph_compiler.shape_inference import infer_shapes

_ELEM_TYPES: dict[int, str] = {
    onnx.TensorProto.FLOAT: "float32",
    onnx.TensorProto.DOUBLE: "float64",
    onnx.TensorProto.FLOAT16: "float16",
    onnx.TensorProto.INT64: "int64",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.INT8: "int8",
    onnx.TensorProto.UINT8: "uint8",
    onnx.TensorProto.BOOL: "bool",
}

MIN_OPSET = 7
MAX_OPSET = 21


def _attribute(attr: onnx.AttributeProto) -> Any:
    value = onnx.helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, onnx.TensorProto):
        return numpy_helper.to_array(value).copy()
    if isinstance(value, list):
        if value and isinstance(value[0], bytes):
            return [v.decode("utf-8") for v in value]
        if value and isinstance(value[0], onnx.TensorProto):
            return [numpy_helper.to_array(v).copy() for v in value]
        if value and isinstance(value[0], onnx.GraphProto):
            raise ModelImportError("graph-valued attributes (control flow) are not supported")
        return list(value)
    if isinstance(value, onnx.GraphProto):
        raise ModelImportError("graph-valued attributes (control flow) are not supported")
    return value


def _constant_node_value(attrs: dict[str, Any]) -> np.ndarray:
    if "value" in attrs:
        return np.asarray(attrs["value"])
    if "value_float" in attrs:
        return np.asarray(attrs["value_float"], dtype=np.float32)
    if "value_floats" in attrs:
        return np.asarray(attrs["value_floats"], dtype=np.float32)
    if "value_int" in attrs:
        return np.asarray(attrs["value_int"], dtype=np.int64)
    if "value_ints" in attrs:
        return np.asarray(attrs["value_ints"], dtype=np.int64)
    raise ModelImportError(f"unsupported Constant attributes {sorted(attrs)}")


def _value_type(
    info: onnx.ValueInfoProto,
    input_shapes: dict[str, tuple[int, ...]],
    dims: dict[str, int],
) -> TensorType:
    tensor = info.type.tensor_type
    if tensor.elem_type not in _ELEM_TYPES:
        raise ModelImportError(
            f"input {info.name!r} has unsupported element type {tensor.elem_type}"
        )
    dtype = _ELEM_TYPES[tensor.elem_type]
    if info.name in input_shapes:
        return TensorType(dtype, tuple(int(d) for d in input_shapes[info.name]))
    shape: list[int] = []
    for d in tensor.shape.dim:
        if d.HasField("dim_value"):
            shape.append(int(d.dim_value))
        elif d.dim_param and d.dim_param in dims:
            shape.append(int(dims[d.dim_param]))
        else:
            name = d.dim_param or "?"
            raise ModelImportError(
                f"input {info.name!r} has a dynamic dimension {name!r}; pass a value for it "
                f"(dims={{'{name}': N}} or --dim {name}=N) or a full input shape"
            )
    return TensorType(dtype, tuple(shape))


def opset_version(model: onnx.ModelProto) -> int:
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return int(entry.version)
    return MAX_OPSET


def from_onnx(
    model: onnx.ModelProto | str | Path,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
    dims: dict[str, int] | None = None,
) -> Graph:
    """Import an ONNX model into the IR and infer every static shape.

    ``input_shapes`` fixes whole input shapes by name; ``dims`` gives values to symbolic
    dimensions (for example ``{"batch": 1, "sequence": 128}``).
    """
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(str(model))
    input_shapes = dict(input_shapes or {})
    dims = dict(dims or {})
    opset = opset_version(model)
    if not MIN_OPSET <= opset <= MAX_OPSET:
        raise ModelImportError(
            f"opset {opset} is outside the supported range {MIN_OPSET}..{MAX_OPSET}"
        )
    g = model.graph
    graph = Graph(g.name or "model")
    env: dict[str, Value] = {}

    for init in g.initializer:
        env[init.name] = graph.constant(numpy_helper.to_array(init), init.name)
    for info in g.input:
        if info.name in env:
            continue
        env[info.name] = graph.add_input(info.name, _value_type(info, input_shapes, dims))
    unknown = set(input_shapes) - {v.name for v in graph.inputs}
    if unknown:
        raise ModelImportError(f"input_shapes names unknown inputs: {sorted(unknown)}")

    for proto in g.node:
        if proto.domain not in ("", "ai.onnx"):
            raise ModelImportError(
                f"operator domain {proto.domain!r} ({proto.op_type}) is not supported"
            )
        attrs = {a.name: _attribute(a) for a in proto.attribute}
        if proto.op_type == "Constant":
            array = _constant_node_value(attrs)
            env[proto.output[0]] = graph.constant(array, proto.output[0])
            continue
        if proto.op_type in ("Softmax", "LogSoftmax", "Hardmax") and opset < 13:
            attrs.setdefault("axis", 1)
            attrs["_legacy_coerce_2d"] = 1
        inputs: list[Value | None] = []
        for name in proto.input:
            if not name:
                inputs.append(None)
            elif name in env:
                inputs.append(env[name])
            else:
                raise ModelImportError(f"node {proto.name!r} uses undefined value {name!r}")
        node = graph.add_node(
            proto.op_type,
            inputs,
            n_outputs=len(proto.output),
            attrs=attrs,
            name=proto.name,
            output_names=[o or f"{proto.op_type.lower()}_unused" for o in proto.output],
        )
        for name, value in zip(proto.output, node.outputs, strict=True):
            if name:
                env[name] = value

    for info in g.output:
        if info.name not in env:
            raise ModelImportError(f"graph output {info.name!r} is never defined")
        graph.add_output(env[info.name], info.name)
    graph.opset = opset
    infer_shapes(graph)
    _check_legacy_softmax(graph)
    return graph


def _check_legacy_softmax(graph: Graph) -> None:
    for node in graph.nodes:
        if node.attrs.pop("_legacy_coerce_2d", None) is None:
            continue
        rank = node.input(0).tensor_type.rank
        axis = int(node.attrs["axis"])
        if (axis + rank if axis < 0 else axis) != rank - 1:
            raise UnsupportedOpError(
                f"{node.op} with opset < 13 over axis {axis} of a rank {rank} tensor "
                "(2-D coercion) is not supported"
            )
        node.attrs["axis"] = -1
