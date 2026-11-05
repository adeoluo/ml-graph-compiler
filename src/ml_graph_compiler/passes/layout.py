"""Layout transformation: NCHW convolution and pooling regions to NHWC.

Each NCHW FusedConv, MaxPool, AveragePool and GlobalAveragePool is rewritten as its NHWC form
between two Transposes (NCHW to NHWC before, NHWC to NCHW after), and convolution weights get a
Transpose from OIHW to HWIO (folded into the constant). Unary elementwise operators and binary
ones whose operands are transposed the same way are then moved after the Transpose, and
simplification cancels every back-to-back pair, so a chain of convolutions runs in NHWC with
one transpose at the graph input. NHWC keeps the channels of a pixel contiguous, which turns a
1x1 convolution into a plain GEMM and lets depthwise kernels vectorize over channels.
"""

from __future__ import annotations

from ml_graph_compiler import ops
from ml_graph_compiler.ir import Graph, Node, Value
from ml_graph_compiler.passes.basic import eliminate_dead_code, fold_constants
from ml_graph_compiler.passes.simplify import simplify
from ml_graph_compiler.passes.util import insert_node

TO_NHWC = [0, 2, 3, 1]
TO_NCHW = [0, 3, 1, 2]
OIHW_TO_HWIO = [2, 3, 1, 0]
_LAYOUT_OPS = frozenset(
    {"FusedConv", "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool"}
)


def _transpose(graph: Graph, value: Value, perm: list[int], before: Node) -> Value:
    return insert_node(graph, "Transpose", [value], before, attrs={"perm": perm}).output


def _convert(graph: Graph, node: Node) -> None:
    inputs: list[Value | None] = list(node.inputs)
    inputs[0] = _transpose(graph, node.input(0), TO_NHWC, node)
    if node.op == "FusedConv":
        inputs[1] = _transpose(graph, node.input(1), OIHW_TO_HWIO, node)
        if len(inputs) > 3 and inputs[3] is not None:
            inputs[3] = _transpose(graph, inputs[3], TO_NHWC, node)
    new = insert_node(graph, node.op, inputs, node, attrs={**node.attrs, "layout": "NHWC"})
    back = insert_node(graph, "Transpose", [new.output], node, attrs={"perm": TO_NCHW})
    graph.replace_all_uses(node.output, back.output)


def _sink_transposes(graph: Graph) -> int:
    """Move a Transpose below elementwise users: f(T(a)) -> T(f(a)), g(T(a), T(b)) -> T(g(a, b))."""
    moved = 0
    for node in list(graph.nodes):
        if not ops.is_elementwise(node.op) or not node.output.users or node.op == "Where":
            continue
        present = [v for v in node.inputs if v is not None]
        transposes = [
            v.producer for v in present if v.producer is not None and v.producer.op == "Transpose"
        ]
        tensors = [v for v in present if not _is_scalar_const(v)]
        if not transposes or len(transposes) != len(tensors):
            continue
        perms = {tuple(ops.transpose_perm(t, t.input(0).tensor_type.rank)) for t in transposes}
        if len(perms) != 1:
            continue
        perm = list(perms.pop())
        new_inputs: list[Value | None] = []
        for v in node.inputs:
            if v is None or _is_scalar_const(v):
                new_inputs.append(v)
            else:
                assert v.producer is not None
                new_inputs.append(v.producer.input(0))
        inner = insert_node(graph, node.op, new_inputs, node, attrs=dict(node.attrs))
        out = _transpose(graph, inner.output, perm, node)
        graph.replace_all_uses(node.output, out)
        moved += 1
    eliminate_dead_code(graph)
    return moved


def _is_scalar_const(value: Value) -> bool:
    return value.is_constant and value.data is not None and value.data.size == 1


def to_nhwc(graph: Graph) -> int:
    """Rewrite NCHW convolution and pooling to NHWC; returns the number of converted nodes."""
    converted = 0
    for node in list(graph.nodes):
        if node.op not in _LAYOUT_OPS or node.attrs.get("layout", "NCHW") != "NCHW":
            continue
        if node.input(0).tensor_type.rank != 4 or not (
            node.output.users or node.output in graph.outputs
        ):
            continue
        if node.op == "FusedConv" and not node.input(1).is_constant:
            continue
        _convert(graph, node)
        converted += 1
    if not converted:
        return 0
    eliminate_dead_code(graph)
    for _ in range(8):
        simplify(graph)
        if not _sink_transposes(graph):
            break
    fold_constants(graph)
    simplify(graph)
    eliminate_dead_code(graph)
    return converted
