"""Small ONNX models built with ``onnx.helper`` for tests, demos and smoke checks.

They use the same operator patterns as the real exports (torch.onnx at opset 14): decomposed
LayerNorm, erf and tanh GELU, Shape-driven Reshape targets, Conv + BatchNormalization + ReLU
blocks with residual adds, depthwise convolutions with Clip(0, 6), and tied embedding heads.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 14


class _Builder:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.count = 0

    def name(self, hint: str) -> str:
        self.count += 1
        return f"{hint}_{self.count}"

    def const(self, array: np.ndarray, hint: str = "w") -> str:
        name = self.name(hint)
        self.inits.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def weight(self, shape: tuple[int, ...], scale: float | None = None, hint: str = "w") -> str:
        fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
        s = scale if scale is not None else 1.0 / math.sqrt(max(fan_in, 1))
        return self.const((self.rng.standard_normal(shape) * s).astype(np.float32), hint)

    def op(self, op_type: str, inputs: list[str], n_out: int = 1, **attrs: Any) -> list[str]:
        outs = [self.name(op_type.lower()) for _ in range(n_out)]
        self.nodes.append(helper.make_node(op_type, inputs, outs, name=self.name(op_type), **attrs))
        return outs

    def op1(self, op_type: str, inputs: list[str], **attrs: Any) -> str:
        return self.op(op_type, inputs, 1, **attrs)[0]

    def scalar(self, value: float) -> str:
        return self.const(np.asarray(value, dtype=np.float32), "s")

    def model(
        self, inputs: list[onnx.ValueInfoProto], outputs: list[onnx.ValueInfoProto], name: str
    ) -> onnx.ModelProto:
        graph = helper.make_graph(self.nodes, name, inputs, outputs, self.inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        return model


def mlp(
    batch: int = 4, sizes: tuple[int, ...] = (16, 32, 32, 10), seed: int = 0
) -> onnx.ModelProto:
    """MatMul + Add + Relu layers, then Softmax."""
    b = _Builder(seed)
    x = "x"
    for i in range(len(sizes) - 1):
        x = b.op1("MatMul", [x, b.weight((sizes[i], sizes[i + 1]))])
        x = b.op1("Add", [x, b.weight((sizes[i + 1],), 0.1)])
        if i < len(sizes) - 2:
            x = b.op1("Relu", [x])
    b.op("Softmax", [x], axis=-1)
    b.nodes[-1].output[0] = "y"
    return b.model(
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, sizes[0]])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, sizes[-1]])],
        "mlp",
    )


def _conv_bn(
    b: _Builder, x: str, cin: int, cout: int, k: int, stride: int = 1, group: int = 1
) -> str:
    w = b.weight((cout, cin // group, k, k))
    y = b.op1(
        "Conv",
        [x, w],
        kernel_shape=[k, k],
        strides=[stride, stride],
        pads=[k // 2] * 4,
        group=group,
    )
    scale = b.const((1.0 + 0.1 * b.rng.standard_normal(cout)).astype(np.float32), "bn_s")
    bias = b.const((0.1 * b.rng.standard_normal(cout)).astype(np.float32), "bn_b")
    mean = b.const((0.1 * b.rng.standard_normal(cout)).astype(np.float32), "bn_m")
    var = b.const((0.5 + b.rng.random(cout)).astype(np.float32), "bn_v")
    return b.op1("BatchNormalization", [y, scale, bias, mean, var], epsilon=1e-5)


def tiny_cnn(
    batch: int = 1, size: int = 32, channels: int = 16, classes: int = 10, seed: int = 0
) -> onnx.ModelProto:
    """ResNet and MobileNetV2 style blocks: stem, residual block, inverted residual, head."""
    b = _Builder(seed)
    x = b.op1("Relu", [_conv_bn(b, "image", 3, channels, 3, stride=2)])
    x = b.op1("MaxPool", [x], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    # basic residual block
    y = b.op1("Relu", [_conv_bn(b, x, channels, channels, 3)])
    y = _conv_bn(b, y, channels, channels, 3)
    x = b.op1("Relu", [b.op1("Add", [y, x])])
    # inverted residual with a depthwise convolution and ReLU6
    lo, hi = b.scalar(0.0), b.scalar(6.0)
    e = b.op1("Clip", [_conv_bn(b, x, channels, channels * 2, 1), lo, hi])
    e = b.op1("Clip", [_conv_bn(b, e, channels * 2, channels * 2, 3, group=channels * 2), lo, hi])
    e = _conv_bn(b, e, channels * 2, channels, 1)
    x = b.op1("Add", [x, e])
    # downsampling block with a strided 1x1 projection shortcut
    y = b.op1("Relu", [_conv_bn(b, x, channels, channels * 2, 3, stride=2)])
    y = _conv_bn(b, y, channels * 2, channels * 2, 3)
    shortcut = _conv_bn(b, x, channels, channels * 2, 1, stride=2)
    x = b.op1("Relu", [b.op1("Add", [y, shortcut])])
    x = b.op1("GlobalAveragePool", [x])
    x = b.op1("Flatten", [x], axis=1)
    b.op("Gemm", [x, b.weight((classes, channels * 2)), b.weight((classes,), 0.1)], transB=1)
    b.nodes[-1].output[0] = "logits"
    return b.model(
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [batch, 3, size, size])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [batch, classes])],
        "tiny_cnn",
    )


def _layer_norm(b: _Builder, x: str, d: int) -> str:
    """LayerNorm as torch.onnx exports it below opset 17."""
    mean = b.op1("ReduceMean", [x], axes=[-1], keepdims=1)
    diff = b.op1("Sub", [x, mean])
    var = b.op1("ReduceMean", [b.op1("Pow", [diff, b.scalar(2.0)])], axes=[-1], keepdims=1)
    std = b.op1("Sqrt", [b.op1("Add", [var, b.scalar(1e-5)])])
    y = b.op1("Div", [diff, std])
    gamma = b.const((1.0 + 0.1 * b.rng.standard_normal(d)).astype(np.float32), "ln_g")
    beta = b.const((0.1 * b.rng.standard_normal(d)).astype(np.float32), "ln_b")
    return b.op1("Add", [b.op1("Mul", [y, gamma]), beta])


def _gelu(b: _Builder, x: str, kind: str) -> str:
    if kind == "erf":  # BERT: x * 0.5 * (1 + erf(x / sqrt(2)))
        e = b.op1("Erf", [b.op1("Div", [x, b.scalar(math.sqrt(2.0))])])
        return b.op1("Mul", [b.op1("Mul", [x, b.op1("Add", [e, b.scalar(1.0)])]), b.scalar(0.5)])
    # GPT-2: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cube = b.op1("Pow", [x, b.scalar(3.0)])
    inner = b.op1("Add", [x, b.op1("Mul", [b.scalar(0.044715), cube])])
    t = b.op1("Tanh", [b.op1("Mul", [b.scalar(math.sqrt(2.0 / math.pi)), inner])])
    return b.op1("Mul", [b.op1("Mul", [b.scalar(0.5), x]), b.op1("Add", [b.scalar(1.0), t])])


def _dense(b: _Builder, x: str, din: int, dout: int) -> str:
    return b.op1("Add", [b.op1("MatMul", [x, b.weight((din, dout))]), b.weight((dout,), 0.02)])


def _dynamic_reshape(b: _Builder, x: str, tail: list[int]) -> str:
    """Reshape whose target is computed from Shape at run time, as torch.onnx exports it."""
    shape = b.op1("Shape", [x])
    dims = [
        b.op1(
            "Unsqueeze",
            [
                b.op1("Gather", [shape, b.const(np.asarray(i, np.int64), "i")], axis=0),
                b.const(np.asarray([0], np.int64), "ax"),
            ],
        )
        for i in (0, 1)
    ]
    target = b.op1("Concat", [*dims, b.const(np.asarray(tail, np.int64), "tail")], axis=0)
    return b.op1("Reshape", [x, target])


def tiny_transformer(
    batch: int = 1,
    seq: int = 8,
    d_model: int = 32,
    heads: int = 4,
    layers: int = 2,
    vocab: int = 128,
    gelu: str = "tanh",
    causal: bool = True,
    seed: int = 0,
) -> onnx.ModelProto:
    """A pre-LN (causal, tanh GELU: GPT-2 style) or post-LN (erf GELU: BERT style) encoder."""
    b = _Builder(seed)
    dh = d_model // heads
    wte = b.weight((vocab, d_model), 0.1, "wte")
    wpe = b.weight((seq, d_model), 0.1, "wpe")
    x = b.op1("Gather", [wte, "input_ids"], axis=0)
    positions = b.const(np.arange(seq, dtype=np.int64)[None, :], "pos")
    x = b.op1("Add", [x, b.op1("Gather", [wpe, positions], axis=0)])
    mask = np.triu(np.ones((seq, seq), dtype=bool), 1)[None, None]
    pre_ln = gelu == "tanh"
    for _ in range(layers):
        h = _layer_norm(b, x, d_model) if pre_ln else x
        qkv = _dense(b, h, d_model, 3 * d_model)
        q, k, v = b.op(
            "Split", [qkv, b.const(np.asarray([d_model] * 3, np.int64), "split")], 3, axis=-1
        )

        def heads_first(t: str) -> str:
            return b.op1("Transpose", [_dynamic_reshape(b, t, [heads, dh])], perm=[0, 2, 1, 3])

        q, k, v = heads_first(q), heads_first(k), heads_first(v)
        kt = b.op1("Transpose", [k], perm=[0, 1, 3, 2])
        scores = b.op1("Div", [b.op1("MatMul", [q, kt]), b.scalar(math.sqrt(dh))])
        if causal:
            neg = b.const(np.asarray(-1e4, dtype=np.float32), "neg")
            scores = b.op1("Where", [b.const(mask, "mask"), neg, scores])
        else:
            zero_mask = b.op1(
                "Mul",
                [
                    b.op1("Sub", [b.scalar(1.0), b.const(np.ones((batch, 1, 1, seq), np.float32))]),
                    b.scalar(-1e4),
                ],
            )
            scores = b.op1("Add", [scores, zero_mask])
        attn = b.op1("MatMul", [b.op1("Softmax", [scores], axis=-1), v])
        attn = b.op1("Transpose", [attn], perm=[0, 2, 1, 3])
        attn = _dynamic_reshape(b, attn, [d_model])
        attn = _dense(b, attn, d_model, d_model)
        x = b.op1("Add", [x, attn])
        if not pre_ln:
            x = _layer_norm(b, x, d_model)
        h = _layer_norm(b, x, d_model) if pre_ln else x
        f = _gelu(b, _dense(b, h, d_model, 4 * d_model), gelu)
        f = _dense(b, f, 4 * d_model, d_model)
        x = b.op1("Add", [x, f])
        if not pre_ln:
            x = _layer_norm(b, x, d_model)
    if pre_ln:
        x = _layer_norm(b, x, d_model)
    b.op("MatMul", [x, b.op1("Transpose", [wte], perm=[1, 0])])
    b.nodes[-1].output[0] = "logits"
    return b.model(
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [batch, seq])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [batch, seq, vocab])],
        "tiny_transformer",
    )


SAMPLES = {
    "mlp": mlp,
    "tiny-cnn": tiny_cnn,
    "tiny-gpt": lambda: tiny_transformer(gelu="tanh", causal=True),
    "tiny-bert": lambda: tiny_transformer(gelu="erf", causal=False),
}
