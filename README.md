<h1 align="center">ML Graph Compiler</h1>

<p align="center">
  <em>A deep learning graph compiler: ONNX models in, an SSA graph IR through optimization passes and a static memory planner, native code out through generated C and clang, with an MLIR export.</em>
</p>

<p align="center">
  <a href="https://github.com/adeoluo/ml-graph-compiler/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adeoluo/ml-graph-compiler/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/adeoluo/ml-graph-compiler/releases"><img alt="Release" src="https://img.shields.io/github/v/release/adeoluo/ml-graph-compiler"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
  <img alt="C" src="https://img.shields.io/badge/C-C11%20%7C%20clang-blue">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

---

## Motivation

Inference runtimes such as onnxruntime, TVM and XLA all solve the same problem: take a dataflow
graph exported from a training framework and turn it into fast code for one machine. The hard
parts are not the arithmetic but the transformations in between: deciding which operators to
fuse, which layout each tensor should have, in what order to run the nodes, and how to fit every
intermediate tensor into as little memory as possible, all while keeping the outputs equal to
the original model within float rounding.

This project implements that pipeline end to end and measures it on four real networks
(ResNet-50, MobileNetV2, BERT-base and GPT-2). An ONNX importer builds a typed SSA graph with
static shapes; constant folding, dead code elimination, CSE, algebraic simplification, operator
fusion, an NCHW to NHWC layout pass and loop fusion rewrite it; a memory-aware scheduler orders
it and a static planner packs every tensor into one arena; the backend emits C with a NEON GEMM
micro-kernel and OpenMP, builds it with clang and loads it through ctypes. A numpy interpreter
runs any graph as the reference, and an MLIR export (linalg on tensors) is verified with
`mlir-opt` and can be lowered to LLVM and run.

## Architecture

<p align="center">
  <img src="docs/architecture/architecture.svg" alt="Architecture diagram" width="860">
</p>

<sub>Legend: amber = entry points and external systems, indigo = core logic, green = data and state, blue = validation and control, grey = observability and outputs.</sub>

A model enters through the `mgc` CLI or the library API. The **frontend** parses the ONNX
protobuf into SSA values and nodes, and shape inference gives every value a static shape and
dtype using one operator table that also holds each operator's numpy semantics. The
**optimizer** runs a named pipeline (`O0` to `O3`) of passes, each verified after it runs; the
**scheduler** picks an order that lowers peak live memory, and the **memory planner** computes
liveness and assigns arena offsets with in-place reuse. The **backends** either interpret the
graph with numpy, print it as MLIR, or generate one C function with constant offsets into the
arena and a packed weights blob, which clang compiles into a cached shared library. Stage by
stage detail is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); design choices are in the
[ADRs](docs/adr/).

## Results

All numbers below were produced on one Apple M1 laptop (8 cores, 8 GB, macOS) with float32,
batch 1, images 224x224 and 128 tokens, by the scripts in [`benchmarks/`](benchmarks/). Raw
results are in [`benchmarks/results/`](benchmarks/results/), and
`uv run python benchmarks/tables.py` prints every table here from them. The full write-up,
including threats to validity, is [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md).

### Numerical parity with onnxruntime

Compiled at `-O3` with the C backend, each model is compared with onnxruntime on the same ONNX
file. Images are the full Imagenette validation split (3,925 ImageNet photos); the text models
get 50 seeded random sequences of 128 tokens (6,400 token positions each).
Reproduce: `uv run --group bench python benchmarks/parity.py`.

@@PARITY@@

Every compiled model picks the same top-1 class as onnxruntime at every image and token
position, and the image models reach the same Imagenette top-1 accuracy. The unoptimized graph
run by the numpy interpreter (`O0`) also matches onnxruntime (max abs error 1.5e-05 to 2.6e-04
on a subset), so each pass is checked against both references.

### Latency and peak memory against onnxruntime and PyTorch

Median latency of one inference in milliseconds with a bootstrap 95% confidence interval, and
the median peak resident set size of the process in MB. Every cell runs in a fresh process;
cells are interleaved round robin for 3 rounds (30 timed runs per round for the CNNs, 10 for
the transformers) and the samples are pooled. "onnxruntime" uses all its graph optimizations;
"no graph opt" disables them, which isolates its kernels. A win or loss is called only when the
two confidence intervals do not overlap.
Reproduce: `uv run --group bench python benchmarks/latency.py --models resnet50 mobilenetv2 --name latency_cnn`
and `... --models bert-base gpt2 --warmup 3 --repeat 10 --name latency_transformers`.

@@LATENCY@@

The machine was shared with other heavy jobs during these runs; the load averages above are
far above its 8 cores, so absolute times are inflated and the 8 thread numbers in particular
measure behaviour under contention. Read the table as a comparison under the same conditions
(engines were interleaved), not as best case latency.

@@LATENCY_READING@@

### Effect of each pass

Cumulative pipelines, each adding one group of passes to the previous one, compiled with the C
backend; nodes, generated kernels, arena size, median latency and the max abs error against
onnxruntime. Reproduce: `uv run --group bench python benchmarks/ablation.py`.

@@ABLATION@@

### Memory planning

Arena size for each placement strategy on the scheduled `O3` graph. The lower bound is the
largest total size of tensors alive at the same step, which no placement can beat.

@@MEMORY@@

### MLIR export

@@MLIR@@

## Quickstart

```bash
git clone https://github.com/adeoluo/ml-graph-compiler && cd ml-graph-compiler
uv sync --dev                                   # Python 3.11+, clang on the PATH
uv run mgc compile tiny-cnn --check --bench 50  # build a bundled model, compare, time it
```

On macOS, `brew install libomp` enables OpenMP (the build falls back to one thread without it);
`brew install llvm` provides `mlir-opt` for the MLIR checks. Linux needs `clang` or `gcc`.

## Usage

### Command line

```bash
uv run mgc inspect model.onnx                       # graph summary, operator counts, shapes
uv run mgc optimize model.onnx -O O3                # run a pipeline, report every pass
uv run mgc optimize model.onnx -O fold,cse,fuse     # or any list of passes
uv run mgc plan model.onnx                          # schedule and arena size per strategy
uv run mgc run model.onnx -O O2                     # numpy interpreter, compare with O0
uv run mgc compile model.onnx --check --bench 100 --threads 8 -o build/
uv run mgc compile model.onnx --emit-c model.c      # keep the generated C
uv run mgc mlir model.onnx --verify -o model.mlir   # linalg on tensors, checked by mlir-opt
```

`model` is an ONNX file or a bundled sample (`mlp`, `tiny-cnn`, `tiny-gpt`, `tiny-bert`).
Symbolic dimensions are bound with `--dim batch=1` or `--shape input_ids=1x128`. Every command
accepts `--json`.

### Library

```python
import numpy as np
from ml_graph_compiler.codegen import compile_model
from ml_graph_compiler.interpreter import run_graph
from ml_graph_compiler.onnx_import import from_onnx
from ml_graph_compiler.samples import tiny_cnn

model = tiny_cnn()
compiled = compile_model(model, opt="O3")  # import, optimize, schedule, generate C, clang
name, ttype = compiled.inputs[0]
x = np.random.default_rng(0).standard_normal(ttype.shape, dtype=np.float32)
out = compiled.run({name: x})

reference = run_graph(from_onnx(model), {name: x})  # numpy interpreter, unoptimized graph
print({k: float(np.abs(out[k] - reference[k]).max()) for k in out})  # {'logits': 9.5e-07}
```

The stages are also usable one at a time: `from_onnx`, `passes.optimize(graph, "O3")`,
`schedule.schedule(graph)`, `memory.plan_memory(graph)`, `codegen.compile_graph(graph)` and
`mlir.to_mlir(graph)`.

### Benchmarks and data

`uv sync --group bench` adds PyTorch, torchvision and transformers.
`benchmarks/export_models.py` downloads the four checkpoints and exports them to ONNX at
opset 14 with static shapes; `benchmarks/parity.py` downloads Imagenette (about 100 MB). Data
and weights go to `../.bench-data/ml-graph-compiler/` (override with `MGC_DATA`). See
[docs/DATA_CARD.md](docs/DATA_CARD.md).

### Demos

Recorded from real runs with [vhs](https://github.com/charmbracelet/vhs); each tape is in `docs/demo/`.

The `O3` pipeline on MobileNetV2: what each pass changed, and the operators before and after.

<p align="center">
  <img src="docs/demo/optimize.gif" alt="The O3 pipeline on MobileNetV2: what each pass changed, and the operators before and after" width="860">
</p>

Scheduling and memory planning for ResNet-50: arena size per placement strategy.

<p align="center">
  <img src="docs/demo/plan.gif" alt="Scheduling and memory planning for ResNet-50: arena size per placement strategy" width="860">
</p>

Compiling MobileNetV2 to C with clang, checking it against the interpreter and timing it.

<p align="center">
  <img src="docs/demo/compile.gif" alt="Compiling MobileNetV2 to C with clang, checking it against the interpreter and timing it" width="860">
</p>

Exporting a small transformer to MLIR and verifying it with mlir-opt.

<p align="center">
  <img src="docs/demo/mlir.gif" alt="Exporting a small transformer to MLIR and verifying it with mlir-opt" width="860">
</p>

## Design decisions

- **An SSA graph IR with static shapes and one operator table**
  ([ADR-0002](docs/adr/0002-ssa-ir-with-static-shapes.md)). Every value has one producer and a
  known shape, so passes can rewrite locally and the planner can size every buffer ahead of
  time. Shape inference and the interpreter share the operator table, so a pass is checked
  against the same semantics it was written for. The cost: one compilation per input shape.
- **One static arena, greedy by size** ([ADR-0003](docs/adr/0003-static-arena-memory-planning.md)).
  Tensors are placed largest first into the lowest offset that does not overlap a tensor alive
  at the same time, with in-place reuse for elementwise ops; the plan is verified pairwise.
  No allocation happens at run time.
- **Generate C and build it with clang** ([ADR-0004](docs/adr/0004-c-code-generation-with-clang.md)).
  C is portable, debuggable and lets clang do register allocation and vectorization; the hot
  GEMM is a hand written NEON micro-kernel over packed weight panels, with a portable fallback.
- **NHWC inside convolution regions** ([ADR-0005](docs/adr/0005-nhwc-layout-transformation.md)).
  Channels last makes the implicit im2col GEMM read contiguous memory; transposes are sunk and
  cancelled so only region boundaries pay for them.
- **MLIR as a textual export** ([ADR-0006](docs/adr/0006-mlir-textual-export.md)). The graph is
  printed as linalg on tensors (with an `mgc` dialect fallback), verified by `mlir-opt` and
  optionally lowered through bufferization to LLVM, without linking MLIR's C++ API.

## Testing

```bash
uv run pytest                                                  # full suite
uv run pytest --cov=src --cov-report=term-missing              # with coverage (CI gate 80%)
make all                                                       # lint, types, coverage
```

@@TESTS@@ tests cover:

- **Operators**: shape and type inference and numpy semantics for every supported operator,
  compared with onnxruntime on generated single node models.
- **Passes**: each pass on targeted graphs and on the bundled models, with outputs compared
  against the interpreter and onnxruntime before and after; property based tests (hypothesis)
  run random elementwise graphs through every pipeline and check the values are unchanged.
- **Memory and scheduling**: plans on random DAGs (hypothesis) never overlap and greedy by size
  is never worse than naive; schedules stay topological; aliasing and in-place rules.
- **C backend**: every kernel against the interpreter with and without OpenMP and NEON,
  input validation, bounds checked Gather, and hostile node names that try to escape comments.
- **MLIR**: text for every operator, verification by `mlir-opt` and lowered execution when the
  LLVM tools are installed (skipped otherwise).
- **CLI**: every subcommand on the bundled samples.

The real model parity, latency and ablation runs are scripts rather than tests because they need
the downloaded weights and minutes to hours of compute.

## Responsible AI

- **Model card:** [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
- **Data card:** [`docs/DATA_CARD.md`](docs/DATA_CARD.md)
- **Threat model:** [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md). A compiled model is native
  code loaded into the calling process; model supplied names never reach generated code
  unescaped, indices are bounds checked, and untrusted models should be compiled in a sandbox.

## Limitations

- Static shapes only: a model with a symbolic batch or sequence length is compiled for one
  binding at a time. No dynamic shapes, control flow (`If`, `Loop`) or quantized operators.
- CPU only (NEON on Apple silicon and ARM Linux, portable C elsewhere); no GPU backend.
- Single thread ResNet-50 is slower than PyTorch eager, whose convolutions use a tuned vendor
  library; the generated GEMM is one micro-kernel shape without autotuning.
- Attention is lowered as separate MatMul, Softmax and MatMul kernels, not a fused
  attention kernel.
- The latency numbers come from a heavily loaded shared laptop (see the load averages above).
- Imagenette is a 10 class subset; text models are fed random token ids, not natural text.

## Roadmap

- [ ] Autotune GEMM tile sizes per shape and add an int8 kernel path.
- [ ] Fused attention (tiled softmax) for the transformer models.
- [ ] Dynamic batch dimension with shape specialised kernels chosen at run time.
- [ ] Rerun the latency tables on an idle machine.

## License

Released under the [MIT License](LICENSE).
