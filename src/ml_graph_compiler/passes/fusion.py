"""Operator fusion.

* Conv + BatchNormalization: the normalization is folded into the convolution weights and bias.
* Conv and Gemm/MatMul with a constant 2-D weight become FusedConv and FusedGemm, which then
  absorb a constant bias add, a residual add of a same-shaped tensor, and one activation
  (ReLU, Clip such as ReLU6, GELU erf or tanh, Tanh, Sigmoid): ``act(op(x) + bias + residual)``.
* ``MatMul(x, Transpose(W))`` with a constant W becomes FusedGemm with ``trans_b`` (no transposed
  copy of W is stored).
* The decomposed LayerNorm that torch.onnx emits below opset 17 (ReduceMean, Sub, Pow,
  ReduceMean, Add, Sqrt, Div, Mul, Add) becomes LayerNormalization.
* The erf GELU (BERT) and tanh GELU (GPT-2) subgraphs become one Gelu node.
"""

from __future__ import annotations

import math

import numpy as np

from ml_graph_compiler import ops
from ml_graph_compiler.ir import Graph, Node, Value
from ml_graph_compiler.passes.basic import eliminate_dead_code
from ml_graph_compiler.passes.util import (
    binary_with_const,
    insert_node,
    is_scalar,
    only_user,
    producer,
    scalar,
)


def _last_axis_mean(node: Node | None, x: Value) -> bool:
    if node is None or node.op != "ReduceMean" or node.inputs[0] is not x or x.type is None:
        return False
    consts = [v.data if v is not None else None for v in node.inputs]
    try:
        axes = ops.reduce_axes(node, x.type.rank, consts)
    except Exception:
        return False
    return axes == (x.type.rank - 1,) and int(node.attrs.get("keepdims", 1)) == 1


def _single(value: Value, graph: Graph) -> bool:
    return only_user(value, graph) is not None


def _channel_vector(value: Value | None, n: int, axis_from_end: int) -> np.ndarray | None:
    """A constant that broadcasts like a per-channel vector of length n, flattened."""
    if value is None or value.data is None:
        return None
    data = value.data
    shape = data.shape
    if data.size == 1:
        return np.full(n, float(data.reshape(())), dtype=np.float32)
    if data.size != n or len(shape) < axis_from_end:
        return None
    if shape[len(shape) - axis_from_end] != n:
        return None
    return data.reshape(n).astype(np.float32)


# ------------------------------------------------------------------------------------------
# LayerNorm and GELU patterns
# ------------------------------------------------------------------------------------------
def fuse_layer_norm(graph: Graph) -> int:
    count = 0
    for div in list(graph.nodes):
        if div.op != "Div" or div not in graph.nodes:
            continue
        d, s = div.inputs[0], div.inputs[1]
        sub = producer(d, "Sub")
        sqrt = producer(s, "Sqrt")
        if sub is None or sqrt is None or d is None:
            continue
        x, mean = sub.inputs[0], sub.inputs[1]
        if x is None or mean is None or not _last_axis_mean(mean.producer, x):
            continue
        add = producer(sqrt.input(0), "Add")
        if add is None:
            continue
        a0, a1 = add.inputs[0], add.inputs[1]
        if scalar(a1) is not None and a0 is not None and not a0.is_constant:
            var, eps = a0, scalar(a1)
        elif scalar(a0) is not None and a1 is not None and not a1.is_constant:
            var, eps = a1, scalar(a0)
        else:
            continue
        if eps is None:
            continue
        rm2 = var.producer
        if rm2 is None or rm2.op != "ReduceMean":
            continue
        sq = rm2.input(0)
        sq_node = sq.producer
        if not _last_axis_mean(rm2, sq) or sq_node is None:
            continue
        if not (
            (sq_node.op == "Mul" and sq_node.inputs[0] is d and sq_node.inputs[1] is d)
            or (
                sq_node.op == "Pow" and sq_node.inputs[0] is d and is_scalar(sq_node.inputs[1], 2.0)
            )
        ):
            continue
        internal = [mean, sq, var, add.output, sqrt.output]
        if not all(_single(v, graph) for v in internal):
            continue
        if any(u not in (sq_node, div) for u in d.users) or d in graph.outputs:
            continue
        dim = x.tensor_type.shape[-1]
        tail, gamma, beta = div, None, None
        user = only_user(div.output, graph)
        if user is not None and user.op == "Mul":
            other = user.inputs[1] if user.inputs[0] is div.output else user.inputs[0]
            gamma = _channel_vector(other, dim, 1)
            if gamma is not None:
                tail = user
        user = only_user(tail.output, graph)
        if user is not None and user.op == "Add":
            other = user.inputs[1] if user.inputs[0] is tail.output else user.inputs[0]
            beta = _channel_vector(other, dim, 1)
            if beta is not None:
                tail = user
        if tail.output.type != x.type:
            continue
        g = graph.constant(gamma if gamma is not None else np.ones(dim, np.float32), "ln_gamma")
        b = graph.constant(beta if beta is not None else np.zeros(dim, np.float32), "ln_beta")
        ln = insert_node(
            graph, "LayerNormalization", [x, g, b], tail, attrs={"axis": -1, "epsilon": float(eps)}
        )
        graph.replace_all_uses(tail.output, ln.output)
        count += 1
    if count:
        eliminate_dead_code(graph)
    return count


def _gelu_tail(graph: Graph, s: Node, x: Value) -> Node | None:
    """Given ``s = 1 + f(x)``, find the node computing ``0.5 * x * s`` in any association."""
    user = only_user(s.output, graph)
    if user is None or user.op != "Mul":
        return None
    other = user.inputs[1] if user.inputs[0] is s.output else user.inputs[0]
    if other is x:  # (x * s) * 0.5
        last = only_user(user.output, graph)
        if last is not None and binary_with_const(last, "Mul", 0.5) is user.output:
            return last
        return None
    half = producer(other, "Mul")
    if (
        half is not None
        and binary_with_const(half, "Mul", 0.5) is x
        and _single(half.output, graph)
    ):
        return user  # (0.5 * x) * s
    if is_scalar(other, 0.5):  # (s * 0.5) * x
        last = only_user(user.output, graph)
        if last is not None and last.op == "Mul" and x in last.inputs:
            return last
    return None


def _cube_of(value: Value | None) -> Value | None:
    node = producer(value, ("Pow", "Mul"))
    if node is None:
        return None
    if node.op == "Pow" and is_scalar(node.inputs[1], 3.0):
        return node.inputs[0]
    a, b = node.inputs[0], node.inputs[1]
    for sq, x in ((a, b), (b, a)):
        sq_node = producer(sq, "Mul")
        if sq_node is not None and sq_node.inputs[0] is x and sq_node.inputs[1] is x:
            return x
    return None


def fuse_gelu(graph: Graph) -> int:
    count = 0
    for node in list(graph.nodes):
        if node.op not in ("Erf", "Tanh") or not node.output.users:
            continue
        x: Value | None = None
        approximate = "none"
        arg = node.input(0)
        if node.op == "Erf":
            scaled = producer(arg, ("Div", "Mul"))
            if scaled is not None and scaled.op == "Div":
                x = binary_with_const(scaled, "Div", math.sqrt(2.0))
            elif scaled is not None:
                x = binary_with_const(scaled, "Mul", 1.0 / math.sqrt(2.0))
        else:
            approximate = "tanh"
            inner = binary_with_const(producer(arg, "Mul"), "Mul", math.sqrt(2.0 / math.pi))
            add = producer(inner, "Add")
            if add is not None:
                for xa, rest in ((add.inputs[0], add.inputs[1]), (add.inputs[1], add.inputs[0])):
                    cube = binary_with_const(producer(rest, "Mul"), "Mul", 0.044715)
                    if xa is not None and _cube_of(cube) is xa:
                        x = xa
                        break
        if x is None:
            continue
        s = only_user(node.output, graph)
        if s is None or binary_with_const(s, "Add", 1.0) is not node.output:
            continue
        tail = _gelu_tail(graph, s, x)
        if tail is None or tail.output.type != x.type:
            continue
        gelu = insert_node(graph, "Gelu", [x], tail, attrs={"approximate": approximate})
        graph.replace_all_uses(tail.output, gelu.output)
        count += 1
    if count:
        eliminate_dead_code(graph)
    return count


# ------------------------------------------------------------------------------------------
# Conv and Gemm with epilogues
# ------------------------------------------------------------------------------------------
def _fold_batch_norm(graph: Graph, conv: Node) -> bool:
    bn = only_user(conv.output, graph)
    if bn is None or bn.op != "BatchNormalization" or bn.inputs[0] is not conv.output:
        return False
    if len(bn.outputs) > 1 and any(o.users for o in bn.outputs[1:]):
        return False
    params = [v.data if v is not None else None for v in bn.inputs[1:5]]
    w, b = conv.input(1).data, conv.optional_input(2)
    if w is None or any(p is None for p in params) or (b is not None and b.data is None):
        return False
    scale, beta, mean, var = (np.asarray(p, dtype=np.float64) for p in params)
    factor = scale / np.sqrt(var + float(bn.attrs.get("epsilon", 1e-5)))
    k = w.shape[0]
    new_w = (w.astype(np.float64) * factor.reshape(k, 1, 1, 1)).astype(np.float32)
    old_b = b.data.astype(np.float64) if b is not None and b.data is not None else np.zeros(k)
    new_b = ((old_b - mean) * factor + beta).astype(np.float32)
    inputs: list[Value | None] = [
        conv.input(0),
        graph.constant(new_w, "conv_w_bn"),
        graph.constant(new_b, "conv_b_bn"),
    ]
    fused = insert_node(graph, "Conv", inputs, bn, attrs=dict(conv.attrs))
    graph.replace_all_uses(bn.outputs[0], fused.output)
    return True


def _activation(node: Node) -> dict[str, object] | None:
    if node.op == "Relu":
        return {"activation": "relu"}
    if node.op == "Tanh":
        return {"activation": "tanh"}
    if node.op == "Sigmoid":
        return {"activation": "sigmoid"}
    if node.op == "Gelu":
        approx = node.attrs.get("approximate", "none")
        return {"activation": "gelu_tanh" if approx == "tanh" else "gelu"}
    if node.op == "Clip":
        if any(v is not None and not v.is_constant for v in node.inputs[1:]):
            return None
        lo, hi = ops.clip_bounds(node, [v.data if v is not None else None for v in node.inputs])
        if lo == 0.0 and math.isinf(hi) and hi > 0:
            return {"activation": "relu"}
        return {"activation": "clip", "clip_min": lo, "clip_max": hi}
    return None


def _absorb_epilogue(graph: Graph, node: Node, channel_from_end: int) -> bool:
    """Absorb one bias add, residual add or activation that consumes ``node``'s output."""
    user = only_user(node.output, graph)
    if user is None:
        return False
    act = str(node.attrs.get("activation", "none"))
    inputs = list(node.inputs) + [None] * (4 - len(node.inputs))
    out_type = node.output.tensor_type
    if user.op == "Add" and act == "none":
        other = user.inputs[1] if user.inputs[0] is node.output else user.inputs[0]
        if other is None or user.output.type != out_type:
            return False
        n = out_type.shape[-channel_from_end]
        vector = _channel_vector(other, n, channel_from_end) if other.is_constant else None
        if vector is not None:
            old = inputs[2].data if inputs[2] is not None else None
            inputs[2] = graph.constant(
                vector if old is None else (old + vector).astype(np.float32), "bias"
            )
        elif not other.is_constant and other.type == out_type and inputs[3] is None:
            inputs[3] = other
        else:
            return False
        attrs = dict(node.attrs)
    elif act == "none" and (extra := _activation(user)) is not None:
        if user.output.type != out_type:
            return False
        attrs = {**node.attrs, **extra}
    else:
        return False
    while inputs and inputs[-1] is None:
        inputs.pop()
    fused = insert_node(graph, node.op, inputs, user, attrs=attrs)
    graph.replace_all_uses(user.output, fused.output)
    return True


def _to_fused_gemm(graph: Graph, node: Node) -> bool:
    if node.op == "Gemm":
        a, b, c = node.input(0), node.input(1), node.optional_input(2)
        if (
            int(node.attrs.get("transA", 0))
            or b.data is None
            or float(node.attrs.get("alpha", 1.0)) != 1.0
            or (c is not None and (c.data is None or float(node.attrs.get("beta", 1.0)) != 1.0))
        ):
            return False
        n = node.output.tensor_type.shape[1]
        bias = _channel_vector(c, n, 1) if c is not None else None
        if c is not None and bias is None:
            return False
        inputs: list[Value | None] = [a, b]
        if bias is not None:
            inputs.append(graph.constant(bias, "gemm_bias"))
        attrs = {"trans_b": int(node.attrs.get("transB", 0))}
    elif node.op == "MatMul":
        a, b = node.input(0), node.input(1)
        if a.tensor_type.rank < 1:
            return False
        transposed = producer(b, "Transpose")
        if b.data is not None and b.tensor_type.rank == 2:
            inputs, attrs = [a, b], {"trans_b": 0}
        elif (
            transposed is not None
            and transposed.input(0).data is not None
            and transposed.input(0).tensor_type.rank == 2
        ):
            inputs, attrs = [a, transposed.input(0)], {"trans_b": 1}
        else:
            return False
    else:
        return False
    fused = insert_node(graph, "FusedGemm", inputs, node, attrs={**attrs, "activation": "none"})
    graph.replace_all_uses(node.output, fused.output)
    return True


def fuse_conv(graph: Graph) -> int:
    count = 0
    for node in list(graph.nodes):
        if node.op == "Conv" and node.output.users and _fold_batch_norm(graph, node):
            count += 1
    eliminate_dead_code(graph)
    for node in list(graph.nodes):
        if node.op == "Conv" and node.input(1).is_constant and _used(graph, node):
            fused = insert_node(
                graph,
                "FusedConv",
                list(node.inputs),
                node,
                attrs={**node.attrs, "activation": "none"},
            )
            graph.replace_all_uses(node.output, fused.output)
            count += 1
    eliminate_dead_code(graph)
    return count + _absorb_all(graph, "FusedConv", 3)


def _lift_reshapes(graph: Graph, node: Node) -> bool:
    """``Reshape(FusedGemm(Reshape(x, [M, K])), lead + [N])`` -> ``FusedGemm(x')`` with x' of
    shape ``lead + [K]``.

    torch exports GPT-2's Conv1D as a flatten to 2-D, addmm and a reshape back; FusedGemm takes
    an operand of any rank, so both reshapes go away and the epilogue sees the real consumers.
    """
    reshape = producer(node.input(0), "Reshape")
    if reshape is None or "shape" not in reshape.attrs or len(node.inputs) > 3:
        return False
    out = node.output.tensor_type
    after = only_user(node.output, graph)
    if out.rank != 2 or after is None or after.op != "Reshape" or "shape" not in after.attrs:
        return False
    target = after.output.tensor_type.shape
    if not target or target[-1] != out.shape[1] or int(np.prod(target[:-1])) != out.shape[0]:
        return False
    x = reshape.input(0)
    a_shape = (*target[:-1], node.input(0).tensor_type.shape[1])
    if x.tensor_type.shape != a_shape:
        x = insert_node(graph, "Reshape", [x], after, attrs={"shape": list(a_shape)}).output
    lifted = insert_node(graph, "FusedGemm", [x, *node.inputs[1:]], after, attrs=dict(node.attrs))
    graph.replace_all_uses(after.output, lifted.output)
    return True


def fuse_gemm(graph: Graph) -> int:
    count = 0
    for node in list(graph.nodes):
        if node.op in ("Gemm", "MatMul") and _used(graph, node) and _to_fused_gemm(graph, node):
            count += 1
    eliminate_dead_code(graph)
    for node in list(graph.nodes):
        if node.op == "FusedGemm" and node.output.users and _lift_reshapes(graph, node):
            count += 1
    eliminate_dead_code(graph)
    return count + _absorb_all(graph, "FusedGemm", 1)


def _absorb_all(graph: Graph, op: str, channel_from_end: int) -> int:
    count = 0
    changed = True
    while changed:
        changed = False
        for node in list(graph.nodes):
            if (
                node.op == op
                and node.output.users
                and _absorb_epilogue(graph, node, channel_from_end)
            ):
                count += 1
                changed = True
        eliminate_dead_code(graph)
    return count


def fuse(graph: Graph) -> int:
    """Run every fusion; returns the number of rewrites."""
    return fuse_layer_norm(graph) + fuse_gelu(graph) + fuse_conv(graph) + fuse_gemm(graph)


def _used(graph: Graph, node: Node) -> bool:
    return any(o.users or o in graph.outputs for o in node.outputs)
