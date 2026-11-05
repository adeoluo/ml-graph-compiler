"""Optimization passes: each rewrite is checked for effect and for preserved numerics."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from onnx import TensorProto, helper, numpy_helper

from ml_graph_compiler import samples
from ml_graph_compiler.errors import CompilerError
from ml_graph_compiler.interpreter import random_feeds, run_graph
from ml_graph_compiler.ir import Graph
from ml_graph_compiler.onnx_import import from_onnx
from ml_graph_compiler.passes import (
    PIPELINES,
    eliminate_common_subexpressions,
    eliminate_dead_code,
    fold_constants,
    fuse,
    optimize,
    simplify,
    to_nhwc,
)
from ml_graph_compiler.passes import resolve as resolve_pipeline

rng = np.random.default_rng(0)


def build(nodes, inputs, outputs, inits=()):  # type: ignore[no-untyped-def]
    graph = helper.make_graph(
        nodes, "t", inputs, outputs, [numpy_helper.from_array(a, n) for n, a in inits]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    return model


def vi(name: str, shape: list[int] | None, elem: int = TensorProto.FLOAT) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem, shape)


def same_numerics(
    model: onnx.ModelProto, passes: str, rtol: float = 1e-5, atol: float = 1e-5
) -> Graph:
    reference_graph = from_onnx(model)
    feeds = random_feeds(reference_graph, 3, int_high=20)
    expected = run_graph(reference_graph, feeds)
    graph = from_onnx(model)
    optimize(graph, passes)
    got = run_graph(graph, feeds)
    for name, value in expected.items():
        assert got[name].shape == value.shape
        np.testing.assert_allclose(got[name], value, rtol=rtol, atol=atol, err_msg=name)
    return graph


# ------------------------------------------------------------------------------------------
# pipelines on the bundled models
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("level", list(PIPELINES))
@pytest.mark.parametrize("name", sorted(samples.SAMPLES))
def test_every_pipeline_preserves_numerics(name: str, level: str) -> None:
    graph = same_numerics(samples.SAMPLES[name](), level, rtol=1e-4, atol=1e-5)
    graph.verify()


@pytest.mark.parametrize(
    "name, expected",
    [
        ("mlp", {"FusedGemm": 3, "Softmax": 1}),
        (
            "tiny-cnn",
            {
                "FusedConv": 9,
                "FusedGemm": 1,
                "GlobalAveragePool": 1,
                "MaxPool": 1,
                "Reshape": 1,
                "Transpose": 1,
            },
        ),
    ],
)
def test_o3_operator_mix(name: str, expected: dict[str, int]) -> None:
    graph = from_onnx(samples.SAMPLES[name]())
    optimize(graph, "O3")
    assert graph.op_counts() == expected


@pytest.mark.parametrize("name", ["tiny-gpt", "tiny-bert"])
def test_transformer_patterns_are_fused(name: str) -> None:
    graph = from_onnx(samples.SAMPLES[name]())
    report = optimize(graph, "O2")
    ops = graph.op_counts()
    assert ops["LayerNormalization"] in (4, 5)
    assert "Erf" not in ops and "Tanh" not in ops and "Pow" not in ops and "ReduceMean" not in ops
    assert ops["FusedGemm"] == 9
    acts = {n.attrs["activation"] for n in graph.nodes if n.op == "FusedGemm"}
    assert acts == {"none", "gelu_tanh" if name == "tiny-gpt" else "gelu"}
    assert report.nodes_after < report.nodes_before / 3


def test_pass_report_and_resolve() -> None:
    graph = from_onnx(samples.mlp())
    report = optimize(graph, ["fold", "dce", "fuse"], verify=False)
    d = report.to_dict()
    assert d["nodes_before"] == 9 and d["nodes_after"] == 4
    assert [p["name"] for p in d["passes"]] == ["fold", "dce", "fuse"]
    assert resolve_pipeline("fold, cse") == ["fold", "cse"]
    assert resolve_pipeline("O0") == []
    with pytest.raises(CompilerError, match="unknown passes"):
        resolve_pipeline("fold,warp")
    assert optimize(from_onnx(samples.mlp()), "O0").nodes_after == 9


# ------------------------------------------------------------------------------------------
# constant folding, DCE, CSE
# ------------------------------------------------------------------------------------------
def test_constant_folding_and_shape_folding() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    model = build(
        [
            helper.make_node("Mul", ["a", "a"], ["sq"]),
            helper.make_node("Shape", ["x"], ["s"]),
            helper.make_node("Size", ["x"], ["n"]),
            helper.make_node("Cast", ["n"], ["nf"], to=TensorProto.FLOAT),
            helper.make_node("Reshape", ["x", "s"], ["xr"]),
            helper.make_node("Add", ["xr", "sq"], ["y0"]),
            helper.make_node("Add", ["y0", "nf"], ["y"]),
        ],
        [vi("x", [2, 3])],
        [vi("y", [2, 3])],
        [("a", a)],
    )
    graph = from_onnx(model)
    removed = fold_constants(graph)
    assert removed == 4
    assert graph.op_counts() == {"Add": 2, "Reshape": 1}
    same_numerics(model, "fold")


def test_folding_skips_large_growth() -> None:
    model = build(
        [
            helper.make_node("Expand", ["c", "shape"], ["big"]),
            helper.make_node("Add", ["x", "big"], ["y"]),
        ],
        [vi("x", [4096, 2048])],
        [vi("y", [4096, 2048])],
        [("c", np.ones((1,), np.float32)), ("shape", np.asarray([4096, 2048], np.int64))],
    )
    graph = from_onnx(model)
    fold_constants(graph)
    assert "Expand" in graph.op_counts()


def test_dead_code_elimination() -> None:
    model = build(
        [helper.make_node("Relu", ["x"], ["unused"]), helper.make_node("Neg", ["x"], ["y"])],
        [vi("x", [3])],
        [vi("y", [3])],
    )
    graph = from_onnx(model)
    assert eliminate_dead_code(graph) == 1
    assert graph.op_counts() == {"Neg": 1}
    assert eliminate_dead_code(graph) == 0


def test_cse_merges_nodes_constants_and_commutative_operands() -> None:
    c = np.ones(3, np.float32)
    model = build(
        [
            helper.make_node("Add", ["x", "c1"], ["a"]),
            helper.make_node("Add", ["c2", "x"], ["b"]),
            helper.make_node("Sub", ["x", "c1"], ["s1"]),
            helper.make_node("Sub", ["c1", "x"], ["s2"]),
            helper.make_node("Mul", ["a", "b"], ["m"]),
            helper.make_node("Mul", ["s1", "s2"], ["n"]),
            helper.make_node("Add", ["m", "n"], ["y"]),
        ],
        [vi("x", [3])],
        [vi("y", [3])],
        [("c1", c), ("c2", c.copy())],
    )
    graph = from_onnx(model)
    changes = eliminate_common_subexpressions(graph)
    assert changes == 2  # one duplicate constant, one duplicate Add; Sub is not commutative
    assert graph.op_counts() == {"Add": 2, "Sub": 2, "Mul": 2}
    same_numerics(model, "cse")


def test_cse_distinguishes_attributes() -> None:
    model = build(
        [
            helper.make_node("Softmax", ["x"], ["a"], axis=0),
            helper.make_node("Softmax", ["x"], ["b"], axis=1),
            helper.make_node("Add", ["a", "b"], ["y"]),
        ],
        [vi("x", [2, 3])],
        [vi("y", [2, 3])],
    )
    graph = from_onnx(model)
    assert eliminate_common_subexpressions(graph) == 0


# ------------------------------------------------------------------------------------------
# algebraic simplification
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "op, const, first",
    [
        ("Add", 0.0, False),
        ("Sub", 0.0, False),
        ("Mul", 1.0, False),
        ("Div", 1.0, False),
        ("Pow", 1.0, False),
        ("Add", 0.0, True),
        ("Mul", 1.0, True),
    ],
)
def test_neutral_elements_are_removed(op: str, const: float, first: bool) -> None:
    inputs = ["k", "x"] if first else ["x", "k"]
    model = build(
        [helper.make_node(op, inputs, ["t"]), helper.make_node("Neg", ["t"], ["y"])],
        [vi("x", [2, 3])],
        [vi("y", [2, 3])],
        [("k", np.asarray(const, np.float32))],
    )
    graph = from_onnx(model)
    assert simplify(graph) >= 1
    assert graph.op_counts() == {"Neg": 1}


def test_neutral_add_that_broadcasts_is_kept() -> None:
    model = build(
        [helper.make_node("Add", ["x", "k"], ["y"])],
        [vi("x", [3])],
        [vi("y", [2, 3])],
        [("k", np.zeros((2, 3), np.float32))],
    )
    graph = from_onnx(model)
    simplify(graph)
    assert graph.op_counts() == {"Add": 1}


@pytest.mark.parametrize(
    "exponent, ops", [(2.0, {"Mul": 1}), (3.0, {"Mul": 2}), (0.5, {"Sqrt": 1})]
)
def test_pow_strength_reduction(exponent: float, ops: dict[str, int]) -> None:
    model = build(
        [helper.make_node("Pow", ["x", "e"], ["y"])],
        [vi("x", [4])],
        [vi("y", [4])],
        [("e", np.asarray(exponent, np.float32))],
    )
    graph = same_numerics(model, "simplify")
    assert graph.op_counts() == ops


def test_div_by_constant_becomes_mul_but_not_by_zero_or_integers() -> None:
    model = build(
        [helper.make_node("Div", ["x", "c"], ["y"])],
        [vi("x", [4])],
        [vi("y", [4])],
        [("c", np.asarray([2, 4, 8, 16], np.float32))],
    )
    assert same_numerics(model, "simplify").op_counts() == {"Mul": 1}
    zero = build(
        [helper.make_node("Div", ["x", "c"], ["y"])],
        [vi("x", [2])],
        [vi("y", [2])],
        [("c", np.asarray([0, 1], np.float32))],
    )
    g = from_onnx(zero)
    simplify(g)
    assert g.op_counts() == {"Div": 1}
    ints = build(
        [helper.make_node("Div", ["x", "c"], ["y"])],
        [vi("x", [2], TensorProto.INT64)],
        [vi("y", [2], TensorProto.INT64)],
        [("c", np.asarray([2, 3], np.int64))],
    )
    g = from_onnx(ints)
    simplify(g)
    assert g.op_counts() == {"Div": 1}


def test_identity_like_ops_are_removed() -> None:
    model = build(
        [
            helper.make_node("Identity", ["x"], ["a"]),
            helper.make_node("Dropout", ["a"], ["b"]),
            helper.make_node("Cast", ["b"], ["c"], to=TensorProto.FLOAT),
            helper.make_node("Expand", ["c", "shape"], ["d"]),
            helper.make_node("Concat", ["d"], ["e"], axis=0),
            helper.make_node("Pad", ["e", "pads"], ["f"]),
            helper.make_node("Relu", ["f"], ["y"]),
        ],
        [vi("x", [2, 3])],
        [vi("y", [2, 3])],
        [("shape", np.asarray([2, 3], np.int64)), ("pads", np.zeros(4, np.int64))],
    )
    graph = same_numerics(model, "simplify")
    assert graph.op_counts() == {"Relu": 1}


def test_nonzero_pad_is_kept() -> None:
    model = build(
        [helper.make_node("Pad", ["x", "pads"], ["y"])],
        [vi("x", [2, 3])],
        [vi("y", [4, 3])],
        [("pads", np.asarray([1, 0, 1, 0], np.int64))],
    )
    assert same_numerics(model, "simplify").op_counts() == {"Pad": 1}


def test_reshape_chains_collapse() -> None:
    model = build(
        [
            helper.make_node("Unsqueeze", ["x", "ax"], ["a"]),
            helper.make_node("Flatten", ["a"], ["b"], axis=1),
            helper.make_node("Reshape", ["b", "s"], ["c"]),
            helper.make_node("Squeeze", ["c", "ax"], ["d"]),
            helper.make_node("Relu", ["d"], ["y"]),
        ],
        [vi("x", [2, 6])],
        [vi("y", [3, 4])],
        [("ax", np.asarray([0], np.int64)), ("s", np.asarray([1, 3, 4], np.int64))],
    )
    graph = same_numerics(model, "simplify")
    assert graph.op_counts() == {"Reshape": 1, "Relu": 1}
    roundtrip = build(
        [
            helper.make_node("Reshape", ["x", "s"], ["a"]),
            helper.make_node("Reshape", ["a", "back"], ["y0"]),
            helper.make_node("Neg", ["y0"], ["y"]),
        ],
        [vi("x", [2, 6])],
        [vi("y", [2, 6])],
        [("s", np.asarray([3, 4], np.int64)), ("back", np.asarray([2, 6], np.int64))],
    )
    assert same_numerics(roundtrip, "simplify").op_counts() == {"Neg": 1}


def test_transpose_composition_and_unit_dims() -> None:
    model = build(
        [
            helper.make_node("Transpose", ["x"], ["a"], perm=[1, 2, 0]),
            helper.make_node("Transpose", ["a"], ["b"], perm=[2, 0, 1]),
            helper.make_node("Transpose", ["b"], ["c"], perm=[0, 2, 1]),
            helper.make_node("Transpose", ["c"], ["d"], perm=[1, 0, 2]),
            helper.make_node("Relu", ["d"], ["y"]),
        ],
        [vi("x", [2, 3, 4])],
        [vi("y", [4, 2, 3])],
    )
    graph = same_numerics(model, "simplify")
    assert graph.op_counts() == {"Transpose": 1, "Relu": 1}
    unit = build(
        [
            helper.make_node("Transpose", ["x"], ["t"], perm=[1, 0, 2]),
            helper.make_node("Neg", ["t"], ["y"]),
        ],
        [vi("x", [1, 5, 4])],
        [vi("y", [5, 1, 4])],
    )
    assert same_numerics(unit, "simplify").op_counts() == {"Reshape": 1, "Neg": 1}
    ident = build(
        [
            helper.make_node("Transpose", ["x"], ["t"], perm=[0, 1]),
            helper.make_node("Neg", ["t"], ["y"]),
        ],
        [vi("x", [2, 3])],
        [vi("y", [2, 3])],
    )
    assert same_numerics(ident, "simplify").op_counts() == {"Neg": 1}


_UNARY = ["Relu", "Neg", "Tanh", "Sigmoid", "Abs", "Identity"]


@st.composite
def elementwise_models(draw: st.DrawFn) -> onnx.ModelProto:
    """Random DAGs of unary/binary elementwise ops with neutral and duplicate constants."""
    nodes, inits = [], []
    values = ["x"]
    for i in range(draw(st.integers(1, 12))):
        kind = draw(st.sampled_from(["unary", "binary", "neutral", "pow"]))
        a = draw(st.sampled_from(values))
        out = f"v{i}"
        if kind == "unary":
            nodes.append(helper.make_node(draw(st.sampled_from(_UNARY)), [a], [out]))
        elif kind == "binary":
            b = draw(st.sampled_from(values))
            nodes.append(
                helper.make_node(draw(st.sampled_from(["Add", "Sub", "Mul", "Max"])), [a, b], [out])
            )
        elif kind == "neutral":
            op, c = draw(st.sampled_from([("Add", 0.0), ("Mul", 1.0), ("Sub", 0.0), ("Div", 1.0)]))
            name = f"c{i}"
            inits.append((name, np.asarray(c, np.float32)))
            nodes.append(helper.make_node(op, [a, name], [out]))
        else:
            name = f"p{i}"
            inits.append((name, np.asarray(draw(st.sampled_from([2.0, 3.0, 1.0])), np.float32)))
            nodes.append(helper.make_node("Pow", [a, name], [out]))
        values.append(out)
    last = values[-1]
    nodes.append(helper.make_node("Identity", [last], ["y"]))
    return build(nodes, [vi("x", [3, 4])], [vi("y", [3, 4])], inits)


@given(elementwise_models(), st.sampled_from(["O1", "O2", "O3", "cse"]))
@settings(max_examples=60, deadline=None)
def test_random_elementwise_graphs_keep_their_values(model: onnx.ModelProto, level: str) -> None:
    reference_graph = from_onnx(model)
    feeds = {"x": np.random.default_rng(0).uniform(-2, 2, (3, 4)).astype(np.float32)}
    expected = run_graph(reference_graph, feeds)["y"]
    graph = from_onnx(model)
    report = optimize(graph, level)
    assert report.nodes_after <= report.nodes_before
    np.testing.assert_allclose(run_graph(graph, feeds)["y"], expected, rtol=2e-5, atol=1e-5)


# ------------------------------------------------------------------------------------------
# fusion
# ------------------------------------------------------------------------------------------
def conv_model(
    act: str | None, residual: bool, bn: bool, bias_add: bool, group: int = 1
) -> onnx.ModelProto:
    c = 4
    nodes = [
        helper.make_node(
            "Conv", ["x", "w"], ["c0"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=group
        )
    ]
    inits = [("w", rng.standard_normal((c, c // group, 3, 3)).astype(np.float32))]
    cur = "c0"
    if bn:
        nodes.append(helper.make_node("BatchNormalization", [cur, "s", "b", "m", "v"], ["bn"]))
        inits += [
            ("s", rng.random(c).astype(np.float32) + 0.5),
            ("b", rng.standard_normal(c).astype(np.float32)),
            ("m", rng.standard_normal(c).astype(np.float32)),
            ("v", rng.random(c).astype(np.float32) + 0.5),
        ]
        cur = "bn"
    if bias_add:
        nodes.append(helper.make_node("Add", [cur, "cb"], ["ba"]))
        inits.append(("cb", rng.standard_normal((1, c, 1, 1)).astype(np.float32)))
        cur = "ba"
    if residual:
        nodes.append(helper.make_node("Add", ["x", cur], ["res"]))
        cur = "res"
    if act == "relu6":
        nodes.append(helper.make_node("Clip", [cur, "lo", "hi"], ["act"]))
        inits += [("lo", np.asarray(0.0, np.float32)), ("hi", np.asarray(6.0, np.float32))]
        cur = "act"
    elif act:
        nodes.append(helper.make_node(act, [cur], ["act"]))
        cur = "act"
    nodes.append(helper.make_node("Identity", [cur], ["y"]))
    return build(nodes, [vi("x", [1, c, 6, 6])], [vi("y", [1, c, 6, 6])], inits)


@pytest.mark.parametrize("act", [None, "Relu", "relu6", "Sigmoid", "Tanh"])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("bn", [False, True])
def test_conv_epilogue_fusion(act: str | None, residual: bool, bn: bool) -> None:
    model = conv_model(act, residual, bn, bias_add=True)
    graph = same_numerics(model, "O2", rtol=1e-4, atol=1e-4)
    assert graph.op_counts() == {"FusedConv": 1}
    node = graph.nodes[0]
    expected_act = {
        None: "none",
        "Relu": "relu",
        "relu6": "clip",
        "Sigmoid": "sigmoid",
        "Tanh": "tanh",
    }[act]
    assert node.attrs["activation"] == expected_act
    assert (len(node.inputs) == 4) == residual


def test_depthwise_conv_fusion_and_nhwc() -> None:
    model = conv_model("Relu", residual=True, bn=True, bias_add=False, group=4)
    graph = same_numerics(model, "O3", rtol=1e-4, atol=1e-4)
    assert graph.op_counts()["FusedConv"] == 1
    conv = next(n for n in graph.nodes if n.op == "FusedConv")
    assert conv.attrs["layout"] == "NHWC" and conv.attrs["group"] == 4


def test_batchnorm_with_two_users_is_not_folded() -> None:
    model = build(
        [
            helper.make_node("Conv", ["x", "w"], ["c"], kernel_shape=[1, 1]),
            helper.make_node("BatchNormalization", ["c", "s", "b", "m", "v"], ["bn"]),
            helper.make_node("Add", ["c", "bn"], ["y"]),
        ],
        [vi("x", [1, 2, 3, 3])],
        [vi("y", [1, 2, 3, 3])],
        [("w", rng.standard_normal((2, 2, 1, 1)).astype(np.float32))]
        + [(n, np.ones(2, np.float32)) for n in "sbmv"],
    )
    graph = same_numerics(model, "O2", rtol=1e-4)
    assert "BatchNormalization" in graph.op_counts()


@pytest.mark.parametrize("act", ["Relu", "Tanh", "Sigmoid"])
def test_gemm_and_matmul_fusion(act: str) -> None:
    model = build(
        [
            helper.make_node("Gemm", ["x", "w", "b"], ["g"], transB=1),
            helper.make_node(act, ["g"], ["h"]),
            helper.make_node("Transpose", ["w2"], ["w2t"], perm=[1, 0]),
            helper.make_node("MatMul", ["h", "w2t"], ["m"]),
            helper.make_node("Add", ["m", "h"], ["y"]),
        ],
        [vi("x", [3, 5])],
        [vi("y", [3, 6])],
        [
            ("w", rng.standard_normal((6, 5)).astype(np.float32)),
            ("b", rng.standard_normal(6).astype(np.float32)),
            ("w2", rng.standard_normal((6, 6)).astype(np.float32)),
        ],
    )
    graph = same_numerics(model, "O2", rtol=1e-4, atol=1e-5)
    assert graph.op_counts() == {"FusedGemm": 2}
    second = graph.nodes[1]
    assert second.attrs["trans_b"] == 1 and len(second.inputs) == 4


def test_gemm_with_alpha_or_transa_is_not_fused() -> None:
    for attrs in ({"alpha": 2.0}, {"transA": 1}):
        model = build(
            [helper.make_node("Gemm", ["x", "w"], ["y"], **attrs)],
            [vi("x", [4, 4])],
            [vi("y", [4, 4])],
            [("w", rng.standard_normal((4, 4)).astype(np.float32))],
        )
        graph = same_numerics(model, "O2")
        assert graph.op_counts() == {"Gemm": 1}


def test_matmul_with_dynamic_weight_is_not_fused() -> None:
    model = build(
        [helper.make_node("MatMul", ["a", "b"], ["y"])],
        [vi("a", [2, 3]), vi("b", [3, 4])],
        [vi("y", [2, 4])],
    )
    graph = same_numerics(model, "O3")
    assert graph.op_counts() == {"MatMul": 1}


def test_fuse_returns_zero_on_plain_graph() -> None:
    model = build([helper.make_node("Neg", ["x"], ["y"])], [vi("x", [3])], [vi("y", [3])])
    graph = from_onnx(model)
    assert fuse(graph) == 0


# ------------------------------------------------------------------------------------------
# layout
# ------------------------------------------------------------------------------------------
def test_nhwc_conversion_leaves_one_input_transpose() -> None:
    graph = from_onnx(samples.tiny_cnn())
    optimize(graph, "O2")
    converted = to_nhwc(graph)
    assert converted == 11
    transposes = [n for n in graph.nodes if n.op == "Transpose"]
    assert len(transposes) == 1 and transposes[0].inputs[0] is graph.inputs[0]
    assert all(
        n.attrs.get("layout") == "NHWC" for n in graph.nodes if n.op in ("FusedConv", "MaxPool")
    )


def test_layout_skips_non_constant_weights_and_rank3() -> None:
    model = build(
        [helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[1, 1])],
        [vi("x", [1, 2, 3, 3]), vi("w", [2, 2, 1, 1])],
        [vi("y", [1, 2, 3, 3])],
    )
    graph = from_onnx(model)
    assert to_nhwc(graph) == 0
    pool = build(
        [helper.make_node("GlobalAveragePool", ["x"], ["y"])],
        [vi("x", [1, 2, 3])],
        [vi("y", [1, 2, 1])],
    )
    assert to_nhwc(from_onnx(pool)) == 0


def test_layout_on_pooling_only_graph_preserves_numerics() -> None:
    model = build(
        [
            helper.make_node("AveragePool", ["x"], ["p"], kernel_shape=[2, 2], strides=[2, 2]),
            helper.make_node("Relu", ["p"], ["r"]),
            helper.make_node("GlobalMaxPool", ["r"], ["y"]),
        ],
        [vi("x", [1, 3, 8, 8])],
        [vi("y", [1, 3, 1, 1])],
    )
    graph = same_numerics(model, "O3")
    assert graph.op_counts()["Transpose"] <= 2
