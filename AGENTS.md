# AGENTS.md

Working notes for coding agents and contributors in **ML Graph Compiler**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

A deep learning graph compiler: ONNX models in, an SSA graph IR through optimization passes and a static memory planner, native code out through generated C and clang, with an MLIR export.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docs/`: documentation, architecture diagram source and render, design records
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `src/ml_graph_compiler/__init__.py`: ml-graph-compiler: ONNX import, SSA graph IR, optimization passes and code generation.
- `src/ml_graph_compiler/cli.py`: Command line interface: ``python -m ml_graph_compiler`` or ``mgc``.
- `src/ml_graph_compiler/codegen/__init__.py`: C code generation backend: emitter, build driver and ctypes runtime.
- `src/ml_graph_compiler/codegen/build.py`: Compile generated C into a shared library with the system C compiler.
- `src/ml_graph_compiler/codegen/c.py`: C code generation.
- `src/ml_graph_compiler/codegen/runtime_lib.py`: Load and run a compiled model through ctypes.
- `src/ml_graph_compiler/errors.py`: Exception types raised by the compiler.
- `src/ml_graph_compiler/interpreter.py`: Reference interpreter: executes the IR with the numpy semantics in :mod:`ops`.
- `src/ml_graph_compiler/ir.py`: Graph intermediate representation in static single assignment form.
- `src/ml_graph_compiler/memory.py`: Static memory planning: liveness, aliasing, in-place reuse and offset assignment.
- `src/ml_graph_compiler/mlir/__init__.py`: MLIR textual export and an optional MLIR execution path.
- `src/ml_graph_compiler/mlir/emit.py`: Print a graph as MLIR text (see the package docstring for the two forms).
- `src/ml_graph_compiler/mlir/execute.py`: Verify, lower and run exported MLIR with the LLVM tools (mlir-opt, mlir-translate, clang).
- `src/ml_graph_compiler/onnx_import.py`: ONNX model importer: protobuf graph to the SSA IR, with static shapes.
- `src/ml_graph_compiler/ops.py`: Operator registry: shape and type inference plus numpy reference semantics.
- `src/ml_graph_compiler/passes/__init__.py`: Optimization passes and the pass manager.
- `src/ml_graph_compiler/passes/basic.py`: Constant folding, dead code elimination and common subexpression elimination.
- `src/ml_graph_compiler/passes/fusion.py`: Operator fusion.
- `src/ml_graph_compiler/passes/layout.py`: Layout transformation: NCHW convolution and pooling regions to NHWC.
- `src/ml_graph_compiler/passes/loopfuse.py`: Loop fusion: chains of elementwise operators become one FusedElementwise node.
- `src/ml_graph_compiler/passes/simplify.py`: Algebraic simplification and canonicalization.
- `src/ml_graph_compiler/passes/util.py`: Helpers shared by the optimization passes.
- `src/ml_graph_compiler/samples.py`: Small ONNX models built with ``onnx.helper`` for tests, demos and smoke checks.
- `src/ml_graph_compiler/schedule.py`: Operator scheduling: memory-aware list scheduling and dataflow (wavefront) analysis.
- `src/ml_graph_compiler/shape_inference.py`: Static shape and type inference with partial evaluation of small shape computations.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make run` | Optimize and run a bundled sample model |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.
