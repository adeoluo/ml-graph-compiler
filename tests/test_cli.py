"""Command line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import onnx
import pytest

from ml_graph_compiler import samples
from ml_graph_compiler.cli import main


def test_inspect_sample(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", "tiny-cnn"]) == 0
    out = capsys.readouterr().out
    assert "nodes: 32" in out and "Conv: 9" in out


def test_inspect_json_and_ir(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = tmp_path / "m.onnx"
    onnx.save(samples.mlp(), str(path))
    assert main(["inspect", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["inputs"] == {"x": "float32[4x16]"}
    assert main(["inspect", str(path), "--print-ir"]) == 0
    assert capsys.readouterr().out.startswith("graph mlp(")


def test_optimize_reports_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["optimize", "tiny-gpt", "-O", "O3"]) == 0
    out = capsys.readouterr().out
    assert "fuse" in out and "LayerNormalization" in out
    assert main(["optimize", "mlp", "-O", "fold,fuse", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["nodes_after"] == 4
    assert main(["optimize", "mlp", "--print-ir"]) == 0
    assert "FusedGemm" in capsys.readouterr().out


def test_run_with_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "tiny-bert", "--check", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["max_abs_diff_vs_O0"] < 1e-4
    assert main(["run", "mlp"]) == 0
    assert "peak_live_mb" in capsys.readouterr().out


def test_dims_and_shapes(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    model = samples.mlp()
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    path = tmp_path / "dyn.onnx"
    onnx.save(model, str(path))
    assert main(["inspect", str(path)]) == 2
    assert "dynamic dimension" in capsys.readouterr().err
    assert main(["inspect", str(path), "--dim", "batch=2", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["inputs"]["x"] == "float32[2x16]"
    assert main(["inspect", str(path), "--shape", "x=3x16", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["inputs"]["x"] == "float32[3x16]"


@pytest.mark.parametrize(
    "argv",
    [
        ["inspect", "nope.onnx"],
        ["inspect", "mlp", "--dim", "batch"],
        ["inspect", "mlp", "--shape", "x=ax2"],
        ["optimize", "mlp", "-O", "bogus"],
    ],
)
def test_errors_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_module_entrypoint() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "ml_graph_compiler", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().startswith("mgc ")


def test_plan(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "tiny-cnn"]) == 0
    out = capsys.readouterr().out
    assert "greedy_by_size" in out and "lower bound" in out
    assert main(["plan", "tiny-gpt", "--json", "--no-inplace"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plans"]["naive"]["inplace"] == 0
    assert payload["schedule"]["peak_after"] <= payload["schedule"]["peak_before"]


def test_compile_checks_benchmarks_and_saves(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cache = str(tmp_path / "cache")
    args = [
        "compile",
        "tiny-cnn",
        "--cache-dir",
        cache,
        "--check",
        "--bench",
        "3",
        "--threads",
        "2",
        "-o",
        str(tmp_path / "out"),
        "--emit-c",
        str(tmp_path / "model.c"),
        "--json",
    ]
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "c" and payload["max_abs_diff_vs_interpreter"] < 1e-3
    assert payload["latency_ms_median"] > 0 and payload["kernels"]["gemm"] == 9
    assert (tmp_path / "out" / "model.json").exists()
    assert "int mgc_run(" in (tmp_path / "model.c").read_text()


def test_compile_portable_build(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    args = [
        "compile",
        "tiny-gpt",
        "--cache-dir",
        str(tmp_path),
        "--no-openmp",
        "--no-neon",
        "-O",
        "O1",
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "openmp: False" in out and "neon: False" in out
