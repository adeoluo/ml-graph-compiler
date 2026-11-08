<h1 align="center">ML Graph Compiler</h1>

<p align="center">
  <em>A deep learning graph compiler: ONNX models in, an SSA graph IR through optimization passes and a static memory planner, native code out through generated C and clang, with an MLIR export.</em>
</p>

<p align="center">
  <a href="https://github.com/adeoluo/ml-graph-compiler/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adeoluo/ml-graph-compiler/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/adeoluo/ml-graph-compiler/releases"><img alt="Release" src="https://img.shields.io/github/v/release/adeoluo/ml-graph-compiler"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue">
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

| Model | Inputs | Max abs error | Mean abs error | Top-1 agreement | Top-1 accuracy (mgc / ORT) |
|---|---|---:|---:|---:|---:|
| ResNet-50 | 3,925 imagenette2-320 val images | 4.0e-05 | 1.1e-06 | 3,925 / 3,925 | 0.8471 / 0.8471 |
| MobileNetV2 | 3,925 imagenette2-320 val images | 5.0e-05 | 2.5e-06 | 3,925 / 3,925 | 0.7903 / 0.7903 |
| BERT-base | 50 random token ids, seq 128 | 2.9e-05 | 4.6e-07 | 6,400 / 6,400 | n/a |
| GPT-2 | 50 random token ids, seq 128 | 2.2e-04 | 2.2e-05 | 6,400 / 6,400 | n/a |

Every compiled model picks the same top-1 class as onnxruntime at every image and token
position, and the image models reach the same Imagenette top-1 accuracy. The unoptimized graph
run by the numpy interpreter (`O0`) also matches onnxruntime (max abs error 1.5e-05 to 2.6e-04
on a subset), so each pass is checked against both references. The text models are exported with
transformers 5.22.2; its graphs differ from the 4.57.6 exports (BERT-base 1,212 instead of 1,266
ONNX nodes, GPT-2 2,043 instead of 2,127) but give bit-identical onnxruntime outputs on the 50
sequences and compile at `-O3` to the same node and kernel counts
([`export_transformers5.json`](benchmarks/results/export_transformers5.json),
`benchmarks/compare_exports.py`).

### Latency and peak memory against onnxruntime and PyTorch

Median latency of one inference in milliseconds with a bootstrap 95% confidence interval, and
the median peak resident set size of the process in MB. Every cell runs in a fresh process;
cells are interleaved round robin for 3 rounds (30 timed runs per round for the CNNs, 10 for
the transformers) and the samples are pooled. "onnxruntime" uses all its graph optimizations;
"no graph opt" disables them, which isolates its kernels. A win or loss is called only when the
two confidence intervals do not overlap.
Reproduce: `uv run --group bench python benchmarks/latency.py --models resnet50 mobilenetv2 --name latency_cnn`
and `... --models bert-base gpt2 --warmup 3 --repeat 10 --name latency_transformers` (run twice, the second
with `--name latency_transformers_run2`).

latency_cnn (latency_cnn): 48 cells, 1 minute load average during the cells min 11.2, median 50.0, max 89.3 (8 cores)
latency_transformers (latency_transformers, latency_transformers_run2): 96 cells, 1 minute load average during the cells min 7.2, median 17.3, max 118.2 (8 cores)

Threads = 1: median ms [95% CI] (peak RSS MB)

| Model | mgc (this) | onnxruntime | onnxruntime, no graph opt | PyTorch eager | mgc vs ORT | mgc vs PyTorch |
|---|---:|---:|---:|---:|---|---|
| ResNet-50 | 224.3 [207.0, 285.4] (431) | 212.9 [180.5, 267.4] (264) | 220.1 [196.1, 256.5] (185) | 61.3 [57.2, 68.7] (362) | tie | loss, 3.66x slower |
| MobileNetV2 | 16.0 [15.7, 17.0] (120) | 21.8 [18.8, 23.2] (113) | 18.0 [17.2, 20.1] (108) | 25.7 [23.9, 27.6] (237) | win, 1.37x | win, 1.61x |
| BERT-base | 1985.7 [1297.5, 2804.0] (572) | 1000.1 [795.7, 1233.0] (428) | 1314.1 [1131.3, 1546.4] (627) | 195.4 [172.8, 237.9] (399; 5 rounds) | loss, 1.99x slower | loss, 10.16x slower |
| GPT-2 | 5021.5 [3823.8, 6863.8] (832) | 1960.4 [1604.7, 2513.0] (640) | 1685.3 [1310.5, 3483.0] (623) | 1316.4 [389.2, 2376.0] (577; 5 rounds) | loss, 2.56x slower | loss, 3.81x slower |

Threads = 8: median ms [95% CI] (peak RSS MB)

| Model | mgc (this) | onnxruntime | onnxruntime, no graph opt | PyTorch eager | mgc vs ORT | mgc vs PyTorch |
|---|---:|---:|---:|---:|---|---|
| ResNet-50 | 58.6 [54.4, 61.6] (470) | 196.9 [149.1, 235.5] (281) | 194.3 [169.3, 232.2] (202) | 101.8 [88.2, 116.0] (462) | win, 3.36x | win, 1.74x |
| MobileNetV2 | 17.5 [15.9, 22.5] (116) | 30.9 [26.1, 46.0] (112) | 39.1 [27.7, 47.6] (111) | 78.9 [75.7, 88.1] (226) | win, 1.77x | win, 4.50x |
| BERT-base | 398.9 [312.2, 483.7] (577) | 1257.3 [1085.0, 1391.2] (459) | 2520.3 [1725.2, 2998.7] (580) | 199.3 [176.6, 251.1] (384; 5 rounds) | win, 3.15x | loss, 2.00x slower |
| GPT-2 | 429.8 [273.2, 499.8] (559) | 892.2 [513.1, 1081.6] (633) | 1074.8 [952.4, 1246.4] (695) | 2182.5 [1596.3, 2729.0] (470; 5 rounds) | win, 2.08x | win, 5.08x |

The machine was shared with other heavy jobs during these runs; the load averages above are
far above its 8 cores, so absolute times are inflated and the 8 thread numbers in particular
measure behaviour under contention. Read the tables as a comparison under the same conditions
(engines were interleaved), not as best case latency.

The transformer cells pool two complete runs (`latency_transformers.json` and
`latency_transformers_run2.json`, 6 rounds per cell; PyTorch has 5 because its first round in the
first run failed to start, before PyTorch was installed in the environment; the failure is kept
in the raw file). On ResNet-50 and MobileNetV2 the compiled code
beats onnxruntime at 8 threads (3.36x and 1.77x) and PyTorch eager (1.74x and 4.50x); single
threaded it ties onnxruntime on ResNet-50, wins on MobileNetV2 and loses to PyTorch's tuned
convolutions on ResNet-50 by 3.66x. On the transformers it wins against onnxruntime at 8 threads
(BERT-base 3.15x, GPT-2 2.08x) and loses single threaded (1.99x and 2.56x slower); PyTorch eager
is the fastest engine on BERT-base at both thread counts. The single thread transformer numbers
are the least reliable: the compiled model memory maps 0.4 to 0.5 GB of weights, and pages
evicted under the machine's memory pressure turn into multi-second outliers (see the threats to
validity).

### Effect of each pass

Cumulative pipelines, each adding one group of passes to the previous one, compiled with the C
backend; nodes, generated kernels, arena size, median latency and the max abs error against
onnxruntime. Leave-one-out runs `O3` with one pass removed. CNN runs use 2 rounds of 10 timed
runs per thread count, transformer runs 2 rounds of 5.
Reproduce: `uv run --group bench python benchmarks/ablation.py --models resnet50 mobilenetv2 --name ablation_cnn`
and `... --models bert-base gpt2 --repeat 5 --name ablation_transformers`.

cumulative: nodes, arena MB, median ms at 1 / 8 threads, max abs err vs ORT

| Model | Config | Nodes | Arena MB | ms, 1 thread | ms, 8 threads | Max abs err |
|---|---|---:|---:|---:|---:|---:|
| ResNet-50 | O0 | error: CodegenError: BatchNormalization parameters must be constant | | | | |
| ResNet-50 | fold | 175 | 9.2 | 2020.6 | 250.2 | 2.9e-06 |
| ResNet-50 | cse | 175 | 9.2 | 986.1 | 250.4 | 2.9e-06 |
| ResNet-50 | simplify | 175 | 9.2 | 694.3 | 285.3 | 2.9e-06 |
| ResNet-50 | fuse | 57 | 6.9 | 1710.5 | 373.7 | 1.9e-06 |
| ResNet-50 | layout | 58 | 6.9 | 468.9 | 119.9 | 2.9e-06 |
| ResNet-50 | loopfuse | 58 | 6.9 | 453.5 | 138.0 | 2.9e-06 |
| MobileNetV2 | O0 | 152 | 9.2 | 161.6 | 73.4 | 9.5e-06 |
| MobileNetV2 | fold | 152 | 9.2 | 151.2 | 63.4 | 9.5e-06 |
| MobileNetV2 | cse | 152 | 9.2 | 138.0 | 53.2 | 9.5e-06 |
| MobileNetV2 | simplify | 152 | 9.2 | 158.8 | 63.0 | 9.5e-06 |
| MobileNetV2 | fuse | 55 | 5.7 | 137.9 | 50.6 | 6.0e-06 |
| MobileNetV2 | layout | 56 | 5.7 | 47.0 | 28.3 | 6.6e-06 |
| MobileNetV2 | loopfuse | 56 | 5.7 | 48.7 | 32.1 | 6.6e-06 |
| BERT-base | O0 | 927 | 11.2 | 4515.6 | 2145.4 | 1.6e-05 |
| BERT-base | fold | 687 | 11.2 | 2719.0 | 878.8 | 1.6e-05 |
| BERT-base | cse | 687 | 11.2 | 2225.9 | 1685.3 | 1.6e-05 |
| BERT-base | simplify | 675 | 11.2 | 2640.0 | 499.5 | 2.1e-05 |
| BERT-base | fuse | 246 | 2.2 | 1127.7 | 303.3 | 1.7e-05 |
| BERT-base | layout | 246 | 2.2 | 2401.5 | 440.9 | 1.7e-05 |
| BERT-base | loopfuse | 245 | 2.2 | 3353.1 | 552.8 | 1.7e-05 |
| GPT-2 | O0 | 1296 | 147.6 | 3937.8 | 1711.6 | 9.9e-05 |
| GPT-2 | fold | 687 | 147.6 | 2881.9 | 827.9 | 9.9e-05 |
| GPT-2 | cse | 687 | 147.6 | 3144.6 | 1365.4 | 9.9e-05 |
| GPT-2 | simplify | 661 | 147.6 | 5436.5 | 3400.8 | 1.1e-04 |
| GPT-2 | fuse | 244 | 3.0 | 2229.2 | 639.9 | 1.1e-04 |
| GPT-2 | layout | 244 | 3.0 | 3151.8 | 503.1 | 1.1e-04 |
| GPT-2 | loopfuse | 232 | 3.0 | 1549.8 | 484.0 | 1.1e-04 |

leave_one_out: nodes, arena MB, median ms at 1 / 8 threads, max abs err vs ORT

| Model | Config | Nodes | Arena MB | ms, 1 thread | ms, 8 threads | Max abs err |
|---|---|---:|---:|---:|---:|---:|
| ResNet-50 | O3 | 58 | 6.9 | 588.0 | 167.1 | 2.9e-06 |
| ResNet-50 | O3-no-fold | 58 | 6.9 | 464.9 | 152.9 | 2.9e-06 |
| ResNet-50 | O3-no-cse | 58 | 6.9 | 471.2 | 177.5 | 2.9e-06 |
| ResNet-50 | O3-no-simplify | 58 | 6.9 | 536.9 | 189.9 | 2.9e-06 |
| ResNet-50 | O3-no-fuse | 162 | 9.2 | 1073.2 | 298.4 | 2.9e-06 |
| ResNet-50 | O3-no-layout | 57 | 6.9 | 1344.6 | 477.1 | 1.9e-06 |
| ResNet-50 | O3-no-loopfuse | 58 | 6.9 | 594.5 | 244.6 | 2.9e-06 |
| MobileNetV2 | O3 | 56 | 5.7 | 51.6 | 28.0 | 6.6e-06 |
| MobileNetV2 | O3-no-fold | 56 | 5.7 | 52.3 | 25.4 | 6.6e-06 |
| MobileNetV2 | O3-no-cse | 56 | 5.7 | 55.8 | 28.9 | 6.6e-06 |
| MobileNetV2 | O3-no-simplify | 56 | 5.7 | 57.7 | 38.5 | 6.6e-06 |
| MobileNetV2 | O3-no-fuse | 153 | 9.2 | 187.6 | 90.8 | 9.5e-06 |
| MobileNetV2 | O3-no-layout | 55 | 5.7 | 167.8 | 83.5 | 6.0e-06 |
| MobileNetV2 | O3-no-loopfuse | 56 | 5.7 | 56.8 | 35.8 | 6.6e-06 |
| BERT-base | O3 | 245 | 2.2 | 932.1 | 273.6 | 1.7e-05 |
| BERT-base | O3-no-fold | 247 | 2.2 | 979.9 | 323.4 | 1.7e-05 |
| BERT-base | O3-no-cse | 245 | 2.2 | 1581.2 | 409.8 | 1.7e-05 |
| BERT-base | O3-no-simplify | 245 | 2.2 | 2950.4 | 535.6 | 1.7e-05 |
| BERT-base | O3-no-fuse | 515 | 11.2 | 960.3 | 576.7 | 1.9e-05 |
| BERT-base | O3-no-layout | 245 | 2.2 | 2149.8 | 336.5 | 1.7e-05 |
| BERT-base | O3-no-loopfuse | 246 | 2.2 | 786.1 | 276.3 | 1.7e-05 |
| GPT-2 | O3 | 232 | 3.0 | 1051.5 | 372.5 | 1.1e-04 |
| GPT-2 | O3-no-fold | 235 | 3.0 | 846.7 | 332.3 | 1.1e-04 |
| GPT-2 | O3-no-cse | 232 | 3.0 | 821.5 | 323.9 | 1.1e-04 |
| GPT-2 | O3-no-simplify | 378 | 3.0 | 849.7 | 345.9 | 1.1e-04 |
| GPT-2 | O3-no-fuse | 478 | 147.6 | 1371.3 | 602.9 | 9.9e-05 |
| GPT-2 | O3-no-layout | 232 | 3.0 | 1122.5 | 339.5 | 1.1e-04 |
| GPT-2 | O3-no-loopfuse | 244 | 3.0 | 807.9 | 325.8 | 1.1e-04 |

Two passes carry most of the speed. Operator fusion folds BatchNorm into convolutions and merges
bias, residual and activation epilogues into the GEMM: it removes 63% to 67% of the nodes
(ResNet-50 175 to 57, MobileNetV2 152 to 55, BERT-base 675 to 246, GPT-2 661 to 244) and
shrinks the arena (GPT-2 from 147.6 MB to 3.0 MB: the output projection reads the 50257 x 768
embedding matrix through a transpose, which FusedGemm absorbs instead of materializing). The NHWC layout pass is
what makes the convolution GEMMs fast: removing it from `O3` makes MobileNetV2 3.3 times and
ResNet-50 2.3 times slower at 1 thread, and removing fusion costs 3.6 and 1.8 times. Constant
folding is required for the C backend on ResNet-50 (BatchNorm parameters must be constants) and
removes 26% of the BERT-base and 47% of the GPT-2 nodes, mostly shape arithmetic. CSE, algebraic
simplification and loop fusion change few nodes on these exported graphs, and their latency
effect is within the run to run noise of this loaded machine. The transformer latencies in this
study varied by 2 to 3 times between configurations that compile to nearly the same code, which
is paging noise (see the threats to validity), so only the node counts, arena sizes and errors
are reliable there. Every configuration stays within 1.2e-04 of onnxruntime.

### Memory planning

Arena size in MB for each placement strategy on the `O3` graph. Peak live is the largest total
size of tensors alive at the same step: in the importer's order, and after the memory-aware
scheduler (the lower bound for any placement in that order). Naive gives every tensor its own
buffer. Weights are not in the arena.

Arena MB by planner (scheduled graph, O3)

| Model | Peak live, import order | Peak live, scheduled (lower bound) | naive (no reuse) | first_fit | greedy_by_size | greedy_by_size + in place |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 6.9 | 6.9 | 43.7 | 10.5 | 6.9 | 6.9 |
| MobileNetV2 | 5.7 | 5.7 | 26.1 | 6.5 | 5.7 | 5.7 |
| BERT-base | 2.2 | 2.2 | 99.8 | 2.6 | 2.2 | 2.2 |
| GPT-2 | 24.9 | 2.6 | 113.6 | 3.0 | 3.0 | 3.0 |

The greedy by size planner reaches the lower bound on three of the four models and is within
0.4 MB on GPT-2. The scheduler matters most on GPT-2: the memory-aware order
cuts peak live memory from 24.9 MB to 2.6 MB, and the arena from 113.6 MB with no reuse
to 3.0 MB. In-place reuse does not change the arena at `O3` on these models.

### MLIR export

Each model is exported as linalg on tensors with weights elided and checked by `mlir-opt`
(LLVM 23 from Homebrew); operators without a linalg lowering would stay as generic `mgc`
operations. MobileNetV2 was also lowered all the way (one-shot bufferization, loops, LLVM
dialect), built with clang and run. Reproduce: `uv run --group bench python benchmarks/mlir_models.py --execute mobilenetv2`.

| Model | Level | Nodes | MLIR lines | Ops left in mgc dialect | Verified by mlir-opt |
|---|---|---:|---:|---|---|
| ResNet-50 | O0 | 179 | 2,137 | none | yes |
| ResNet-50 | O3 | 58 | 1,216 | none | yes |
| MobileNetV2 | O0 | 152 | 2,011 | none | yes |
| MobileNetV2 | O3 | 56 | 1,224 | none | yes |
| BERT-base | O0 | 927 | 5,922 | ConstantOfShape x1 | yes |
| BERT-base | O3 | 245 | 3,469 | none | yes |
| GPT-2 | O0 | 1,296 | 6,837 | none | yes |
| GPT-2 | O3 | 232 | 3,132 | none | yes |

| Model | Lowered and built (s) | Max abs err vs ORT | Top-1 agrees | ms, 1 thread |
|---|---:|---:|---|---:|
| MobileNetV2 | 32 | 1.0e-05 | yes | 585 |

Every model at `O3` lowers completely to standard dialects. The lowered MobileNetV2 matches
onnxruntime, but at 585 ms single threaded it is more than 30 times slower than the C backend: the default MLIR
lowering emits naive loops with no tiling or vectorization, which is the work the C backend's
packed GEMM does by hand.

## Quickstart

```bash
git clone https://github.com/adeoluo/ml-graph-compiler && cd ml-graph-compiler
uv sync --dev                                   # Python 3.12+, clang on the PATH
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

666 tests (94% line and branch coverage) cover:

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

CI runs the suite on Python 3.12, 3.13 and 3.14. Python 3.15 is not in the matrix yet:
onnxruntime, the reference most operator and backend tests compare against, publishes no 3.15
wheels (1.31.1 is the latest release), so the suite cannot run there.

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
- Single threaded, ResNet-50 is 3.66x slower than PyTorch eager, whose convolutions use a tuned
  vendor library, and BERT-base and GPT-2 are 2.0x to 2.6x slower than onnxruntime; the
  generated GEMM is one micro-kernel shape without autotuning.
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
