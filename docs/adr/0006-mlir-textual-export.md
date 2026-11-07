# ADR-0006: MLIR as a textual export, verified and optionally executed with LLVM tools

- **Status:** Accepted
- **Date:** 2026-11-06

## Context
MLIR is the common IR for production ML compilers. Showing that the graph IR lowers cleanly to
upstream MLIR dialects is useful, but the MLIR Python bindings are not on PyPI for every
platform and would make the package hard to install.

## Decision
Print MLIR text directly (`mlir/emit.py`): operators lower to `linalg.generic` with affine maps,
`linalg.fill`, `linalg.transpose`, `tensor` and `arith`/`math` ops on tensors; anything without
a lowering is printed as a generic `"mgc.<op>"` operation. A second form prints every node as an
`mgc` operation. When `mlir-opt` is installed, the export is verified, and fully lowered modules
can be bufferized, lowered to LLVM, compiled and run through ctypes to compare with the C backend
and onnxruntime.

## Alternatives considered
- **MLIR Python bindings** - typed construction and in-process verification, but a large native
  dependency that is not installable everywhere the package must run.
- **A custom dialect defined in C++ (ODS)** - the proper way to register `mgc` ops, but it needs
  an out-of-tree LLVM build; generic operation syntax with `--allow-unregistered-dialect` is
  enough for a textual export.

## Consequences
- The package has no MLIR dependency; MLIR tests skip when `mlir-opt` is absent (as on CI).
- Text emission must follow MLIR syntax exactly; tests verify the output with `mlir-opt` locally.
