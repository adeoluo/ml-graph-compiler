"""Static memory planner, scheduler and loop fusion."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from onnx import TensorProto, helper, numpy_helper

from ml_graph_compiler import samples
from ml_graph_compiler.errors import GraphError
from ml_graph_compiler.interpreter import random_feeds, run_graph
from ml_graph_compiler.ir import Graph, TensorType
from ml_graph_compiler.memory import STRATEGIES, Buffer, Location, plan_memory
from ml_graph_compiler.onnx_import import from_onnx
from ml_graph_compiler.passes import fuse_elementwise, optimize
from ml_graph_compiler.schedule import (
    apply_order,
    critical_path_flops,
    memory_aware_order,
    peak_live_bytes,
    schedule,
    wavefronts,
)

F = TensorType("float32", (256,))  # 1 KiB


def build(nodes, inputs, outputs, inits=()):  # type: ignore[no-untyped-def]
    graph = helper.make_graph(
        nodes, "t", inputs, outputs, [numpy_helper.from_array(a, n) for n, a in inits]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    return model


def vi(name: str, shape: list[int] | None) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def chain_graph(n: int, inplace_ops: str = "Relu") -> Graph:
    g = Graph()
    v = g.add_input("x", F)
    for _ in range(n):
        node = g.add_node(inplace_ops, [v])
        node.output.type = F
        v = node.output
    g.add_output(v, "y")
    return g


# ------------------------------------------------------------------------------------------
# memory planning
# ------------------------------------------------------------------------------------------
def test_chain_without_inplace_needs_two_buffers() -> None:
    g = chain_graph(6)
    plan = plan_memory(g, inplace=False)
    # the last node writes the output directly; five intermediates ping-pong between 2 slots
    assert plan.arena_bytes == 2 * 1024
    assert plan.naive_bytes == 5 * 1024
    assert plan.lower_bound == 2 * 1024
    assert plan.location(g.outputs[0]) == Location("output", 0)
    assert plan.location(g.inputs[0]) == Location("input", 0)


def test_chain_inplace_writes_everything_into_the_output() -> None:
    g = chain_graph(6)
    plan = plan_memory(g)
    # the first node cannot overwrite the graph input; the rest run in place in the output
    assert plan.inplace == 5
    assert plan.arena_bytes == 0
    assert all(plan.location(n.output).kind == "output" for n in g.nodes)


def test_views_alias_their_input() -> None:
    model = build(
        [
            helper.make_node("Relu", ["x"], ["a"]),
            helper.make_node("Reshape", ["a", "s"], ["b"]),
            helper.make_node("Flatten", ["b"], ["c"], axis=0),
            helper.make_node("Neg", ["c"], ["y"]),
        ],
        [vi("x", [4, 8])],
        [vi("y", [1, 32])],
        [("s", np.asarray([8, 4], np.int64))],
    )
    g = from_onnx(model)
    plan = plan_memory(g, inplace=False)
    a, b, c = (n.output for n in g.nodes[:3])
    assert plan.offset(a) == plan.offset(b) == plan.offset(c)
    assert plan.aliases == 2
    view_of_input = from_onnx(
        build(
            [helper.make_node("Reshape", ["x", "s"], ["y"])],
            [vi("x", [4, 8])],
            [vi("y", [8, 4])],
            [("s", np.asarray([8, 4], np.int64))],
        )
    )
    plan = plan_memory(view_of_input)
    assert plan.location(view_of_input.outputs[0]) == Location("input", 0)
    assert len(plan.copies) == 1 and plan.copies[0][1] == 0


def test_view_of_constant_and_constant_output() -> None:
    g = Graph()
    c = g.constant(np.ones((2, 2), np.float32), "c")
    node = g.add_node("Reshape", [c], attrs={"shape": [4]})
    node.output.type = TensorType("float32", (4,))
    g.add_output(node.output, "y")
    g.add_output(c, "c_out")
    plan = plan_memory(g)
    assert plan.location(node.output) == Location("const")
    assert plan.location(c) == Location("const")
    assert [j for _, j in plan.copies] == [0, 1]


def test_same_value_twice_as_output_is_copied() -> None:
    g = chain_graph(2)
    g.add_output(g.outputs[0], "again")
    plan = plan_memory(g)
    assert [j for _, j in plan.copies] == [1]


def test_inplace_requires_same_shape_and_death() -> None:
    model = build(
        [
            helper.make_node("Relu", ["x"], ["a"]),
            helper.make_node("Add", ["a", "b"], ["c"]),  # a is [4,1] broadcast to [4,8]: no reuse
            helper.make_node("Neg", ["c"], ["d"]),  # c is used again below: no reuse
            helper.make_node("Relu", ["c"], ["c2"]),  # c dies here: reuse
            helper.make_node("Add", ["d", "c2"], ["y"]),
        ],
        [vi("x", [4, 1]), vi("b", [4, 8])],
        [vi("y", [4, 8])],
    )
    g = from_onnx(model)
    plan = plan_memory(g)
    plan.verify()
    a, c, d, c2, _ = (n.output for n in g.nodes)
    assert plan.location(c) != plan.location(a)
    assert plan.location(d) != plan.location(c)
    assert plan.location(c2) == plan.location(c)
    assert plan.inplace == 2  # Relu over c, and the final Add over d into the output


def test_workspace_buffers() -> None:
    g = chain_graph(3)
    plan = plan_memory(g, inplace=False, workspace=lambda n: 100 if n is g.nodes[1] else 0)
    assert id(g.nodes[1]) in plan.workspaces
    assert plan.workspaces[id(g.nodes[1])] % 64 == 0
    assert plan.arena_bytes >= 1024 + 128


def test_unknown_strategy_and_missing_location() -> None:
    g = chain_graph(2)
    with pytest.raises(GraphError):
        plan_memory(g, strategy="best")  # type: ignore[arg-type]
    plan = plan_memory(g)
    from ml_graph_compiler.ir import Value

    with pytest.raises(GraphError):
        plan.location(Value("ghost", F))
    with pytest.raises(GraphError):
        plan.offset(g.inputs[0])


def test_verify_catches_overlap() -> None:
    g = chain_graph(4)
    plan = plan_memory(g, inplace=False)
    plan.buffers.append(Buffer(99, 1024, 0, 10, offset=plan.buffers[0].offset))
    with pytest.raises(GraphError, match="overlap"):
        plan.verify()
    plan.buffers[-1] = Buffer(99, 1024, 50, 60, offset=8)
    with pytest.raises(GraphError, match="misaligned"):
        plan.verify()
    plan.buffers[-1] = Buffer(99, 1024, 50, 60, offset=10**6)
    with pytest.raises(GraphError, match="exceeds"):
        plan.verify()


@pytest.mark.parametrize("name", sorted(samples.SAMPLES))
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_plans_are_valid_on_samples(name: str, strategy: str) -> None:
    g = from_onnx(samples.SAMPLES[name]())
    optimize(g, "O3")
    plan = plan_memory(g, strategy=strategy)  # type: ignore[arg-type]
    plan.verify()
    assert plan.lower_bound <= plan.arena_bytes <= plan.naive_bytes
    summary = plan.summary()
    assert summary["buffers"] == len(plan.buffers)
    for node in g.nodes:
        for out in node.outputs:
            assert plan.location(out).kind in ("arena", "output", "input", "const")


@st.composite
def random_dags(draw: st.DrawFn) -> Graph:
    g = Graph()
    values = [g.add_input("x", TensorType("float32", (draw(st.integers(1, 64)),)))]
    for _ in range(draw(st.integers(1, 25))):
        size = draw(st.integers(1, 512))
        k = draw(st.integers(1, min(3, len(values))))
        ins = draw(st.lists(st.sampled_from(values), min_size=k, max_size=k))
        node = g.add_node("Concat", list(ins), attrs={"axis": 0})
        node.output.type = TensorType("float32", (size,))
        values.append(node.output)
    for v in draw(st.lists(st.sampled_from(values[1:]), min_size=1, max_size=3, unique_by=id)):
        g.add_output(v)
    return g


@given(random_dags(), st.sampled_from(STRATEGIES), st.sampled_from([16, 64]))
@settings(max_examples=120, deadline=None)
def test_random_plans_never_overlap(g: Graph, strategy: str, alignment: int) -> None:
    plan = plan_memory(g, strategy=strategy, alignment=alignment)  # type: ignore[arg-type]
    plan.verify()
    assert plan.lower_bound <= plan.arena_bytes <= plan.naive_bytes
    # brute force: every pair alive at the same step is disjoint
    for i, a in enumerate(plan.buffers):
        for b in plan.buffers[i + 1 :]:
            if a.overlaps_in_time(b):
                assert a.offset + a.size <= b.offset or b.offset + b.size <= a.offset


@given(random_dags())
@settings(max_examples=60, deadline=None)
def test_greedy_by_size_is_never_worse_than_naive(g: Graph) -> None:
    greedy = plan_memory(g, "greedy_by_size").arena_bytes
    assert greedy <= plan_memory(g, "naive").arena_bytes


# ------------------------------------------------------------------------------------------
# scheduling
# ------------------------------------------------------------------------------------------
def fan_graph() -> Graph:
    """Two independent branches; the source order keeps both big tensors alive at once."""
    g = Graph()
    x = g.add_input("x", TensorType("float32", (8,)))
    big = TensorType("float32", (4096,))
    small = TensorType("float32", (8,))
    a1 = g.add_node("Expand", [x])
    a1.output.type = big
    b1 = g.add_node("Expand", [x])
    b1.output.type = big
    a2 = g.add_node("ReduceSum", [a1.output])
    a2.output.type = small
    b2 = g.add_node("ReduceSum", [b1.output])
    b2.output.type = small
    out = g.add_node("Add", [a2.output, b2.output])
    out.output.type = small
    g.add_output(out.output)
    return g


def test_memory_aware_order_reduces_peak() -> None:
    g = fan_graph()
    before = peak_live_bytes(g)
    order = memory_aware_order(g)
    after = peak_live_bytes(g, order)
    assert after < before
    assert before == 2 * 16384 + 32
    report = schedule(g)
    assert report.peak_after == after and report.peak_before == before
    g.verify()
    # running the scheduler again changes nothing
    assert schedule(g).peak_after == after


def test_schedule_keeps_order_when_no_gain() -> None:
    g = chain_graph(4)
    before = list(g.nodes)
    report = schedule(g)
    assert g.nodes == before and report.peak_after == report.peak_before


def test_apply_order_rejects_bad_orders() -> None:
    g = chain_graph(3)
    with pytest.raises(GraphError, match="permutation"):
        apply_order(g, g.nodes[:2])
    with pytest.raises(GraphError):
        apply_order(g, list(reversed(g.nodes)))
    assert [n.op for n in g.nodes] == ["Relu"] * 3
    g.verify()


def test_cycle_detection() -> None:
    g = chain_graph(2)
    g.set_input(g.nodes[0], 0, g.nodes[1].output)
    with pytest.raises(GraphError, match="cycle"):
        memory_aware_order(g)


def test_wavefronts_and_critical_path() -> None:
    g = fan_graph()
    levels = wavefronts(g)
    assert [len(lv) for lv in levels] == [2, 2, 1]
    crit, total = critical_path_flops(g)
    assert total == 4096 * 2 + 8 * 2 + 8 and crit == 4096 + 8 + 8
    assert critical_path_flops(Graph()) == (0, 0)


@pytest.mark.parametrize("name", sorted(samples.SAMPLES))
def test_schedule_preserves_numerics(name: str) -> None:
    g = from_onnx(samples.SAMPLES[name]())
    optimize(g, "O3")
    feeds = random_feeds(g, 0, int_high=50)
    expected = run_graph(g, feeds)
    report = schedule(g)
    assert report.peak_after <= report.peak_before
    assert report.levels >= 1 and report.max_width >= 1
    got = run_graph(g, feeds)
    for k in expected:
        np.testing.assert_array_equal(got[k], expected[k])


@given(random_dags())
@settings(max_examples=60, deadline=None)
def test_memory_aware_order_is_topological(g: Graph) -> None:
    order = memory_aware_order(g)
    seen: set[int] = set()
    for node in order:
        for v in node.inputs:
            if v is not None and v.producer is not None:
                assert id(v.producer) in seen
        seen.add(id(node))
    assert len(order) == len(g.nodes)


# ------------------------------------------------------------------------------------------
# loop fusion
# ------------------------------------------------------------------------------------------
def test_loop_fusion_builds_one_program() -> None:
    model = build(
        [
            helper.make_node("Mul", ["x", "s"], ["a"]),
            helper.make_node("Add", ["a", "m"], ["b"]),
            helper.make_node("Clip", ["b", "", "hi"], ["c"]),
            helper.make_node("Softmax", ["c"], ["y"], axis=-1),
        ],
        [vi("x", [2, 4, 4]), vi("m", [1, 4, 4])],
        [vi("y", [2, 4, 4])],
        [("s", np.asarray(0.5, np.float32)), ("hi", np.asarray(1.0, np.float32))],
    )
    g = from_onnx(model)
    feeds = random_feeds(g)
    expected = run_graph(g, feeds)["y"]
    assert fuse_elementwise(g) == 1
    assert g.op_counts() == {"FusedElementwise": 1, "Softmax": 1}
    fused = g.nodes[0]
    assert [ins[0] for ins in fused.attrs["program"]] == ["Mul", "Add", "Clip"]
    assert -1 in fused.attrs["program"][2][2]
    np.testing.assert_allclose(run_graph(g, feeds)["y"], expected, rtol=1e-6)
    assert fuse_elementwise(g) == 0


def test_loop_fusion_respects_multiple_users_outputs_and_shapes() -> None:
    model = build(
        [
            helper.make_node("Relu", ["x"], ["a"]),
            helper.make_node("Neg", ["a"], ["b"]),
            helper.make_node("Add", ["a", "b"], ["c"]),  # a has two users inside the group: ok
            helper.make_node("Relu", ["v"], ["small"]),
            helper.make_node("Add", ["c", "small"], ["y"]),  # small is [4]: different shape
            helper.make_node("Tanh", ["y"], ["z"]),
        ],
        [vi("x", [3, 4]), vi("v", [4])],
        [vi("y", [3, 4]), vi("z", [3, 4])],
    )
    g = from_onnx(model)
    feeds = random_feeds(g)
    expected = run_graph(g, feeds)
    fuse_elementwise(g)
    ops = g.op_counts()
    assert ops["FusedElementwise"] == 1 and ops["Relu"] == 1 and ops["Tanh"] == 1
    got = run_graph(g, feeds)
    for k in expected:
        np.testing.assert_allclose(got[k], expected[k], rtol=1e-6)


def test_loop_fusion_rejects_bad_programs() -> None:
    from ml_graph_compiler import ops as registry
    from ml_graph_compiler.errors import UnsupportedOpError
    from ml_graph_compiler.ir import Node

    with pytest.raises(UnsupportedOpError):
        registry.run(Node("FusedElementwise", [], [], {}), [])
    with pytest.raises(UnsupportedOpError):
        registry.run(Node("FusedElementwise", [], [], {"program": [["MatMul", {}, [0, 0]]]}), [])
