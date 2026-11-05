"""The SSA graph IR: construction, mutation, ordering and verification."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml_graph_compiler.errors import GraphError
from ml_graph_compiler.ir import Graph, Node, TensorType, Value, dtype_name, numpy_dtype

F = TensorType("float32", (2, 3))


def chain(n: int = 3) -> tuple[Graph, list[Node]]:
    g = Graph("chain")
    x = g.add_input("x", F)
    nodes = []
    v = x
    for _ in range(n):
        node = g.add_node("Relu", [v])
        node.output.type = F
        nodes.append(node)
        v = node.output
    g.add_output(v, "y")
    return g, nodes


def test_tensor_type_properties() -> None:
    t = TensorType("float32", (2, 3, 4))
    assert (t.rank, t.size, t.itemsize, t.nbytes) == (3, 24, 4, 96)
    assert str(t) == "float32[2x3x4]"
    assert TensorType("int64", ()).size == 1
    assert TensorType.of(np.zeros((5, 1), np.int64)) == TensorType("int64", (5, 1))


def test_tensor_type_rejects_bad_shapes_and_dtypes() -> None:
    with pytest.raises(GraphError):
        TensorType("float32", (-1, 2))
    with pytest.raises(GraphError):
        TensorType("complex64", (1,))
    with pytest.raises(GraphError):
        dtype_name(np.complex64)
    with pytest.raises(GraphError):
        numpy_dtype("string")


@pytest.mark.parametrize(
    "name", ["float32", "float64", "float16", "int64", "int32", "int8", "uint8", "bool"]
)
def test_dtype_roundtrip(name: str) -> None:
    assert dtype_name(numpy_dtype(name)) == name


def test_value_and_node_accessors() -> None:
    v = Value("a")
    with pytest.raises(GraphError):
        _ = v.tensor_type
    assert "Value('a'" in repr(v)
    node = Node("Add", [v, None], [Value("o")], {"k": 1}, "add0")
    assert node.input(0) is v
    with pytest.raises(GraphError):
        node.input(1)
    with pytest.raises(GraphError):
        node.input(5)
    assert node.optional_input(1) is None and node.optional_input(9) is None
    assert "Add(a, _)" in repr(node)


def test_names_are_unique() -> None:
    g = Graph()
    a = g.add_input("x", F)
    b = g.constant(np.zeros(2, np.float32), "x")
    assert a.name != b.name
    names = {g.fresh_name("t") for _ in range(50)}
    assert len(names) == 50


def test_use_lists_track_mutation() -> None:
    g, (n0, n1, _n2) = chain()
    assert n0.output.users == [n1]
    g.set_input(n1, 0, g.inputs[0])
    assert n1 in g.inputs[0].users and not n0.output.users
    g.remove_node(n0)
    g.verify()
    assert n0 not in g.nodes


def test_remove_node_with_users_fails() -> None:
    g, (n0, _, n2) = chain()
    with pytest.raises(GraphError):
        g.remove_node(n0)
    with pytest.raises(GraphError):
        g.remove_node(n2)


def test_replace_all_uses_updates_outputs() -> None:
    g, (n0, n1, n2) = chain()
    g.replace_all_uses(n2.output, n0.output)
    assert g.outputs[0] is n0.output
    assert g.named_outputs()[0][0] == "y"
    g.replace_all_uses(n0.output, n0.output)
    assert g.remove_nodes([n1, n2]) == 2
    assert g.remove_nodes([]) == 0
    g.verify()


def test_verify_detects_violations() -> None:
    g, (_n0, n1, _) = chain()
    g.nodes[0], g.nodes[1] = g.nodes[1], g.nodes[0]
    with pytest.raises(GraphError, match="undefined"):
        g.verify()
    g.topological_sort()
    g.verify()
    n1.output.users.append(Node("Relu", [], [], name="ghost"))
    with pytest.raises(GraphError, match="removed user"):
        g.verify()


def test_verify_detects_missing_use_and_double_definition() -> None:
    g, (n0, n1, _) = chain()
    n0.output.users.remove(n1)
    with pytest.raises(GraphError, match="use list"):
        g.verify()
    g2, (m0, m1, _) = chain()
    m1.outputs = [m0.output]
    with pytest.raises(GraphError):
        g2.verify()
    g3, (k0, _, _) = chain()
    k0.output.producer = None
    with pytest.raises(GraphError, match="producer"):
        g3.verify()


def test_cycle_is_detected() -> None:
    g, (n0, n1, _) = chain()
    g.set_input(n0, 0, n1.output)
    with pytest.raises(GraphError, match="cycle"):
        g.topological_sort()


def test_undefined_output() -> None:
    g = Graph()
    g.add_output(Value("dangling", F))
    with pytest.raises(GraphError, match="output"):
        g.verify()


@given(st.permutations(list(range(6))))
@settings(max_examples=40, deadline=None)
def test_topological_sort_restores_any_permutation(order: list[int]) -> None:
    g, nodes = chain(6)
    g.nodes = [nodes[i] for i in order]
    g.topological_sort()
    assert g.nodes == nodes
    g.verify()


def test_topological_sort_keeps_valid_order_stable() -> None:
    g = Graph()
    x = g.add_input("x", F)
    a = g.add_node("Relu", [x])
    b = g.add_node("Neg", [x])
    c = g.add_node("Add", [b.output, a.output])
    g.add_output(c.output)
    before = list(g.nodes)
    g.topological_sort()
    assert g.nodes == before


def test_summary_text_and_queries() -> None:
    g, _ = chain()
    k = g.constant(np.ones((4,), np.float32), "k")
    node = g.add_node("Add", [g.outputs[0], k])
    node.output.type = F
    assert g.constant_bytes() == 16
    assert g.constants() == [k]
    assert g.op_counts() == {"Relu": 3, "Add": 1}
    s = g.summary()
    assert s["nodes"] == 4 and s["inputs"] == {"x": "float32[2x3]"}
    text = str(g)
    assert text.startswith("graph chain(") and "return y=" in text
    assert len(list(g.values())) == len({id(v) for v in g.values()})


def test_add_node_at_index() -> None:
    g, _nodes = chain(2)
    first = g.add_node("Neg", [g.inputs[0]], index=0)
    assert g.nodes[0] is first and first.output.producer is first
