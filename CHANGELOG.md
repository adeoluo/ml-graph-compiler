# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.1.1] 2026-11-09

### Documentation

- Complete citation metadata and bump it on release (#10)
- Render every architecture diagram in the house style (#13)
- Diagrams on white in the shared theme, make diagram for every diagram, and a self-contained Definition of Done (#14)

### Build and CI

- Build and attach the sdist and wheel to every release (#15)
- Lock transformers 5.22.2 (was 4.57.6) and rerun the transformer benchmarks (#12)

### Maintenance

- Latest toolchain and CI versions (#9)
- Release script keeps hard-coded version strings in step (#11)
- Release script keeps nested crates and packages on the release version (#16)

## [0.1.0] 2026-11-07

### Features

- SSA IR, ONNX importer, interpreter and optimization passes (#1)
- Memory-aware scheduler, static memory planner and loop fusion (#2)
- C code generation backend compiled with clang, NEON and OpenMP (#3)
- MLIR textual export to linalg on tensors and an mgc dialect (#4)
- Real-model parity, latency, memory and pass ablation benchmarks (#5)

### Bug fixes

- Keep model-supplied names from escaping comments in generated C and MLIR (#6)

### Documentation

- Design records, threat model and layered architecture diagram (#7)

### Maintenance

- Scaffold project
- Release tooling, changelog and contributor notes (#8)
