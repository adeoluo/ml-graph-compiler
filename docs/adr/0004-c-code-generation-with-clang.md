# ADR-0004: Generate C and build it with clang, NEON and OpenMP

- **Status:** Accepted
- **Date:** 2026-11-06

## Context
The compiler needs a native backend that runs the four real models fast enough to compare with
onnxruntime and PyTorch on an Apple M1, and that also builds on the Linux CI runners.

## Decision
Emit one C translation unit per model (`codegen/c.py`) with every loop bound, stride and arena
offset as a compile-time constant, and build it into a shared library with the system C compiler
(`codegen/build.py`, clang `-O3 -mcpu=native`, OpenMP through Homebrew libomp on macOS).
Matrix multiplications and convolutions call a packed GEMM driver in a small runtime with an
8x12 NEON micro-kernel (a portable fallback elsewhere); constant weights are packed into panels
at compile time, and convolutions read the image through implicit im2col packing. Elementwise
chains become single loop nests that clang vectorizes. The library is cached under a hash of the
source and flags, and the packed weights are memory mapped read-only.

## Alternatives considered
- **LLVM IR through llvmlite, or MLIR as the main backend** - more control over code generation,
  but a heavy dependency for the main path; MLIR is kept as an optional export (ADR-0006).
- **Calling Accelerate or OpenBLAS for GEMM** - likely faster for large matrices, but then the
  project would measure a vendor library rather than its own code generation, and fusion of
  bias, residual and activation into the GEMM epilogue would be lost.
- **A Rust or C++ runtime with an interpreter loop over kernels** - avoids a C compiler at
  compile time but gives up the constant-folded loop bounds and offsets.

## Consequences
- A C compiler is required to compile a model; the numpy interpreter remains available without
  one.
- Compilation of a large model takes seconds (cached afterwards).
- Kernel quality is the project's own; where onnxruntime's MLAS kernels are faster, the
  benchmarks show it.
