# Evaluation Report: ML Graph Compiler

Every number here comes from a script in [`benchmarks/`](../benchmarks/), with the raw output in
[`benchmarks/results/`](../benchmarks/results/). `uv run python benchmarks/tables.py` regenerates
each table from those files.

## Setup

- **Hardware:** one Apple M1 laptop, 8 cores (4 performance, 4 efficiency), 8 GB RAM, macOS 27
  (1-minute load averages 3.0 to 8.2 across the benchmark cells, recorded per cell in the raw JSON).
  No GPU is used.
- **Software:** Python 3.12, numpy 2.5.4, Apple clang 21 with Homebrew libomp, onnx 1.23.2,
  onnxruntime 1.31.1 (CPU execution provider), PyTorch 2.15.0 with torchvision 0.29.2 and
  transformers 5.22.2 for the exports and eager baselines, LLVM/MLIR from Homebrew for the MLIR
  checks.
- **Models:** ResNet-50 and MobileNetV2 (torchvision ImageNet weights), BERT-base uncased and
  GPT-2 124M (Hugging Face), exported by `benchmarks/export_models.py` to ONNX opset 14 with
  static shapes (batch 1, 3x224x224 images, 128 tokens) and with export time constant folding off,
  so this compiler and onnxruntime start from the same graph.
- **Data:** Imagenette 320 px validation split (3,925 images, 10 ImageNet classes) for the image
  models; 50 seeded random token sequences (seeds 0 to 49) for the text models. See
  [DATA_CARD.md](DATA_CARD.md).
- **Compiler configuration:** `-O3` pipeline (fold, DCE, CSE, simplify, fuse, layout, loop fusion),
  memory-aware schedule, greedy-by-size arena with in-place reuse, C backend built with
  `clang -O3 -mcpu=native` with NEON and OpenMP.

## 1. Numerical parity

`uv run --group bench python benchmarks/parity.py` (raw: `results/parity.json`). Each input runs
through the compiled model and onnxruntime (all graph optimizations on); errors are the max and
mean absolute difference over all outputs, and top-1 agreement compares the argmax over classes
(images) or over the vocabulary at each token position (text). The unoptimized graph is also run
by the numpy interpreter on a subset (40 images or 1 sequence) as an independent reference.

| Model | Inputs | Max abs error | Mean abs error | Top-1 agreement | Top-1 accuracy (mgc / ORT) |
|---|---|---:|---:|---:|---:|
| ResNet-50 | 3,925 imagenette2-320 val images | 4.0e-05 | 1.1e-06 | 3,925 / 3,925 | 0.8471 / 0.8471 |
| MobileNetV2 | 3,925 imagenette2-320 val images | 5.0e-05 | 2.5e-06 | 3,925 / 3,925 | 0.7903 / 0.7903 |
| BERT-base | 50 random token ids, seq 128 | 2.9e-05 | 4.6e-07 | 6,400 / 6,400 | n/a |
| GPT-2 | 50 random token ids, seq 128 | 2.2e-04 | 2.2e-05 | 6,400 / 6,400 | n/a |

Interpreter at `O0` against onnxruntime: max abs error 1.5e-05 (ResNet-50), 2.6e-05
(MobileNetV2), 1.9e-05 (BERT-base) and 2.6e-04 (GPT-2), with full top-1 agreement. GPT-2 has the
largest absolute error because its logits are large (order 100); its relative error is 1.9e-06.

**transformers 4.57.6 to 5.22.2.** `benchmarks/compare_exports.py` (raw:
`results/export_transformers5.json`) compares the exports from both releases. transformers 5
builds BERT's token type ids with a gather instead of a slice unless they are passed explicitly,
and GPT-2's eager attention no longer applies its own causal mask, so `export_models.py` now
passes zero token type ids and a fixed additive causal mask. The resulting graphs are smaller
(BERT-base 1,212 instead of 1,266 ONNX nodes, GPT-2 2,043 instead of 2,127; the `Where` and
`ConstantOfShape` nodes that built attention masks at run time are gone), produce bit-identical onnxruntime outputs on the 50 sequences (max abs
difference 0.0, 6,400 of 6,400 argmax positions equal), and compile at `-O3` to the same node,
kernel and arena counts (245 nodes and 2.25 MiB for BERT-base, 232 nodes and 3 MiB for GPT-2).
The parity table above was rerun on the transformers 5 exports and is unchanged to two
significant figures.

## 2. Latency and peak memory

`benchmarks/latency.py` (raw: `results/latency_cnn.json`, `results/latency_transformers.json`).
Each (engine, model, threads) cell is a fresh process: load, warm up (5 runs for CNNs, 3 for
transformers), time 30 (CNNs) or 10 (transformers) runs. Cells are interleaved round robin for 3
rounds and samples pooled; the confidence interval is a 2,000 sample bootstrap of the median.
Peak memory is the process peak RSS (`ru_maxrss`), which includes the Python interpreter and the
framework, and for this compiler the touched pages of the memory mapped weights file.

latency_cnn (latency_cnn): 48 cells, 1 minute load average during the cells min 3.2, median 3.4, max 4.0 (8 cores)
latency_transformers (latency_transformers): 48 cells, 1 minute load average during the cells min 3.0, median 5.0, max 8.2 (8 cores)

Threads = 1: median ms [95% CI] (peak RSS MB)

| Model | mgc (this) | onnxruntime | onnxruntime, no graph opt | PyTorch eager | mgc vs ORT | mgc vs PyTorch |
|---|---:|---:|---:|---:|---|---|
| ResNet-50 | 113.9 [113.8, 114.1] (596) | 101.4 [101.2, 101.6] (285) | 103.3 [103.3, 103.5] (204) | 31.3 [31.2, 31.5] (486) | loss, 1.12x slower | loss, 3.64x slower |
| MobileNetV2 | 13.3 [13.3, 13.3] (138) | 11.6 [11.6, 11.7] (114) | 12.3 [12.3, 12.4] (112) | 14.6 [14.6, 14.7] (323) | loss, 1.15x slower | win, 1.10x |
| BERT-base | 290.9 [290.7, 291.3] (1074) | 256.9 [256.4, 257.9] (957) | 304.3 [304.0, 304.7] (1111) | 52.4 [52.1, 52.5] (668) | loss, 1.13x slower | loss, 5.56x slower |
| GPT-2 | 415.4 [414.6, 415.9] (1592) | 357.2 [356.9, 357.7] (1032) | 399.4 [398.9, 399.6] (1040) | 97.3 [96.4, 100.2] (846) | loss, 1.16x slower | loss, 4.27x slower |

Threads = 8: median ms [95% CI] (peak RSS MB)

| Model | mgc (this) | onnxruntime | onnxruntime, no graph opt | PyTorch eager | mgc vs ORT | mgc vs PyTorch |
|---|---:|---:|---:|---:|---|---|
| ResNet-50 | 29.8 [29.7, 30.0] (597) | 40.9 [40.1, 41.9] (287) | 41.6 [41.0, 42.5] (204) | 28.1 [28.0, 28.2] (487) | win, 1.37x | loss, 1.06x slower |
| MobileNetV2 | 6.0 [5.9, 6.0] (138) | 6.0 [5.7, 6.3] (116) | 7.3 [6.8, 7.6] (113) | 43.6 [43.5, 43.9] (324) | tie | win, 7.33x |
| BERT-base | 72.0 [71.6, 73.2] (1257) | 88.8 [85.9, 103.1] (956) | 127.1 [122.6, 130.5] (1086) | 54.5 [53.3, 55.8] (670) | win, 1.23x | loss, 1.32x slower |
| GPT-2 | 109.7 [106.0, 120.4] (1833) | 149.4 [138.9, 199.9] (1031) | 165.3 [145.8, 188.5] (1041) | 91.5 [90.6, 103.8] (847) | win, 1.36x | loss, 1.20x slower |

All cells ran on the 8-core Apple M1 (4 performance + 4 efficiency cores; 1-minute load averages
3.2 to 4.0 for CNNs, 3.0 to 8.2 with median 5.0 for transformers). At **4 threads** (matching the
4 performance cores, `results/latency_t4.json`), `onnxruntime`'s MLAS kernels lead on all four
models by **1.05x to 1.38x** (`33.3 vs 35.7 ms` ResNet-50, `4.1 vs 5.6 ms` MobileNetV2,
`75.0 vs 84.2 ms` BERT-base, `111.2 vs 117.2 ms` GPT-2). At **8 threads** (where heterogeneous
P+E core speed differences penalize `onnxruntime`'s static thread-pool partitioning relative to
OpenMP dynamic tile scheduling), `mgc` beats `onnxruntime` CPU on three of
four models (**1.37x** on ResNet-50, **1.23x** on BERT-base, **1.36x** on GPT-2) and **ties** on
MobileNetV2 (`6.0 ms` vs `6.0 ms`). At **1 thread**, `onnxruntime`'s hand-tuned MLAS assembly
kernels beat `mgc`'s portable NEON micro-kernel by **1.12x to 1.16x** across all four models.
PyTorch eager dispatches matrix multiplies on macOS to Apple Accelerate (`BLAS_INFO=accelerate`,
using Apple AMX hardware blocks) and leads on ResNet-50 (`28.1 ms` vs `29.8 ms` at 8 threads,
`31.3 ms` vs `113.9 ms` at 1 thread), BERT-base (`1.32x` faster at 8 threads), and GPT-2 (`1.20x`
faster at 8 threads), while `mgc` leads PyTorch eager on MobileNetV2 (`1.10x` at 1 thread, `7.33x`
at 8 threads where batch-1 depthwise convolutions suffer thread-pool dispatch overhead).

## 3. Pass ablation

`benchmarks/ablation.py` (raw: `results/ablation_cnn.json`, `results/ablation_transformers.json`).
Cumulative pipelines add one group of passes at a time; leave-one-out removes one pass from
`O3`. Latency is the median over 2 rounds of interleaved runs at 1 and 8 threads.

cumulative: nodes, arena MB, median ms at 1 / 8 threads, max abs err vs ORT

| Model | Config | Nodes | Arena MB | ms, 1 thread | ms, 8 threads | Max abs err |
|---|---|---:|---:|---:|---:|---:|
| ResNet-50 | O0 | error: CodegenError: BatchNormalization parameters must be constant | | | | |
| ResNet-50 | fold | 175 | 9.2 | 200.2 | 54.0 | 2.9e-06 |
| ResNet-50 | cse | 175 | 9.2 | 200.8 | 53.3 | 2.9e-06 |
| ResNet-50 | simplify | 175 | 9.2 | 199.3 | 53.1 | 2.9e-06 |
| ResNet-50 | fuse | 57 | 6.9 | 205.9 | 52.5 | 1.9e-06 |
| ResNet-50 | layout | 58 | 6.9 | 114.1 | 30.8 | 2.9e-06 |
| ResNet-50 | loopfuse | 58 | 6.9 | 113.8 | 30.3 | 2.9e-06 |
| MobileNetV2 | O0 | 152 | 9.2 | 54.0 | 17.7 | 9.5e-06 |
| MobileNetV2 | fold | 152 | 9.2 | 54.0 | 17.6 | 9.5e-06 |
| MobileNetV2 | cse | 152 | 9.2 | 54.0 | 17.7 | 9.5e-06 |
| MobileNetV2 | simplify | 152 | 9.2 | 51.8 | 17.1 | 9.5e-06 |
| MobileNetV2 | fuse | 55 | 5.7 | 42.4 | 13.8 | 6.0e-06 |
| MobileNetV2 | layout | 56 | 5.7 | 13.3 | 6.1 | 6.6e-06 |
| MobileNetV2 | loopfuse | 56 | 5.7 | 13.3 | 6.0 | 6.6e-06 |
| BERT-base | O0 | 905 | 11.2 | 410.0 | 127.0 | 1.6e-05 |
| BERT-base | fold | 675 | 11.2 | 408.5 | 126.9 | 1.6e-05 |
| BERT-base | cse | 675 | 11.2 | 417.1 | 140.5 | 1.6e-05 |
| BERT-base | simplify | 675 | 11.2 | 414.4 | 132.8 | 2.1e-05 |
| BERT-base | fuse | 246 | 2.2 | 292.8 | 109.3 | 1.7e-05 |
| BERT-base | layout | 246 | 2.2 | 291.6 | 75.6 | 1.7e-05 |
| BERT-base | loopfuse | 245 | 2.2 | 291.2 | 74.4 | 1.7e-05 |
| GPT-2 | O0 | 1248 | 147.6 | 593.5 | 252.5 | 9.9e-05 |
| GPT-2 | fold | 675 | 147.6 | 588.8 | 254.1 | 9.9e-05 |
| GPT-2 | cse | 675 | 147.6 | 589.3 | 261.8 | 9.9e-05 |
| GPT-2 | simplify | 661 | 147.6 | 563.2 | 246.4 | 1.1e-04 |
| GPT-2 | fuse | 244 | 3.0 | 443.3 | 111.6 | 1.1e-04 |
| GPT-2 | layout | 244 | 3.0 | 428.7 | 111.2 | 1.1e-04 |
| GPT-2 | loopfuse | 232 | 3.0 | 415.4 | 110.8 | 1.1e-04 |

leave_one_out: nodes, arena MB, median ms at 1 / 8 threads, max abs err vs ORT

| Model | Config | Nodes | Arena MB | ms, 1 thread | ms, 8 threads | Max abs err |
|---|---|---:|---:|---:|---:|---:|
| ResNet-50 | O3 | 58 | 6.9 | 114.0 | 29.9 | 2.9e-06 |
| ResNet-50 | O3-no-fold | 58 | 6.9 | 113.9 | 30.0 | 2.9e-06 |
| ResNet-50 | O3-no-cse | 58 | 6.9 | 114.0 | 30.0 | 2.9e-06 |
| ResNet-50 | O3-no-simplify | 58 | 6.9 | 114.0 | 30.2 | 2.9e-06 |
| ResNet-50 | O3-no-fuse | 162 | 9.2 | 197.8 | 52.4 | 2.9e-06 |
| ResNet-50 | O3-no-layout | 57 | 6.9 | 205.6 | 50.6 | 1.9e-06 |
| ResNet-50 | O3-no-loopfuse | 58 | 6.9 | 114.1 | 30.2 | 2.9e-06 |
| MobileNetV2 | O3 | 56 | 5.7 | 13.2 | 6.0 | 6.6e-06 |
| MobileNetV2 | O3-no-fold | 56 | 5.7 | 13.2 | 6.0 | 6.6e-06 |
| MobileNetV2 | O3-no-cse | 56 | 5.7 | 13.3 | 6.0 | 6.6e-06 |
| MobileNetV2 | O3-no-simplify | 56 | 5.7 | 13.3 | 6.1 | 6.6e-06 |
| MobileNetV2 | O3-no-fuse | 153 | 9.2 | 38.8 | 14.6 | 9.5e-06 |
| MobileNetV2 | O3-no-layout | 55 | 5.7 | 42.4 | 13.9 | 6.0e-06 |
| MobileNetV2 | O3-no-loopfuse | 56 | 5.7 | 13.3 | 6.0 | 6.6e-06 |
| BERT-base | O3 | 245 | 2.2 | 291.4 | 77.1 | 1.7e-05 |
| BERT-base | O3-no-fold | 247 | 2.2 | 291.0 | 76.6 | 1.7e-05 |
| BERT-base | O3-no-cse | 245 | 2.2 | 292.4 | 76.4 | 1.7e-05 |
| BERT-base | O3-no-simplify | 245 | 2.2 | 291.3 | 76.3 | 1.7e-05 |
| BERT-base | O3-no-fuse | 515 | 11.2 | 404.9 | 122.4 | 1.9e-05 |
| BERT-base | O3-no-layout | 245 | 2.2 | 293.1 | 76.0 | 1.7e-05 |
| BERT-base | O3-no-loopfuse | 246 | 2.2 | 291.4 | 76.7 | 1.7e-05 |
| GPT-2 | O3 | 232 | 3.0 | 418.8 | 103.7 | 1.1e-04 |
| GPT-2 | O3-no-fold | 233 | 3.0 | 416.0 | 105.3 | 1.1e-04 |
| GPT-2 | O3-no-cse | 232 | 3.0 | 415.3 | 106.3 | 1.1e-04 |
| GPT-2 | O3-no-simplify | 378 | 3.0 | 418.6 | 105.2 | 1.1e-04 |
| GPT-2 | O3-no-fuse | 478 | 147.6 | 560.1 | 237.5 | 9.9e-05 |
| GPT-2 | O3-no-layout | 232 | 3.0 | 415.5 | 106.2 | 1.1e-04 |
| GPT-2 | O3-no-loopfuse | 244 | 3.0 | 415.0 | 103.7 | 1.1e-04 |

The `O3` / `loopfuse` rows in both ablation studies match the independent subprocess latency
benchmark (`ResNet-50` `114.0 ms` vs `113.9 ms` at 1 thread, `29.9 ms` vs `29.8 ms` at 8 threads;
`MobileNetV2` `13.2 ms` vs `13.3 ms` at 1 thread, `6.0 ms` vs `6.0 ms` at 8 threads; `BERT-base`
`291.4 ms` vs `290.9 ms` at 1 thread, `74.4-77.1 ms` vs `72.0 ms` at 8 threads; `GPT-2`
`415.4-418.8 ms` vs `415.4 ms` at 1 thread, `103.7-110.8 ms` vs `109.7 ms` at 8 threads).

Two passes carry most of the speed. Operator fusion folds BatchNorm into convolutions and merges
bias, residual and activation epilogues into the GEMM: it removes 63% to 67% of the nodes
(ResNet-50 175 to 57, MobileNetV2 152 to 55, BERT-base 675 to 246, GPT-2 661 to 244) and
shrinks the arena (GPT-2 from 147.6 MB to 3.0 MB: the output projection reads the 50257 x 768
embedding matrix through a transpose, which FusedGemm absorbs instead of materializing). Removing
fusion from `O3` slows 8-thread inference by **1.75x** on ResNet-50, **2.43x** on MobileNetV2,
**1.59x** on BERT-base, and **2.29x** on GPT-2. The NHWC layout pass is what makes the convolution
GEMMs fast: removing it from `O3` slows MobileNetV2 **3.21x** (t=1) / **2.32x** (t=8) and ResNet-50
**1.80x** (t=1) / **1.69x** (t=8). Constant folding is required for the C backend on ResNet-50
(BatchNorm parameters must be constants) and removes 25% of the BERT-base and 46% of the GPT-2
nodes, mostly shape arithmetic. Every configuration stays within 1.2e-04 of onnxruntime.

## 4. Memory planning

Arena MB by planner (scheduled graph, O3)

| Model | Peak live, import order | Peak live, scheduled (lower bound) | naive (no reuse) | first_fit | greedy_by_size | greedy_by_size + in place |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 6.9 | 6.9 | 43.7 | 10.5 | 6.9 | 6.9 |
| MobileNetV2 | 5.7 | 5.7 | 26.1 | 6.5 | 5.7 | 5.7 |
| BERT-base | 2.2 | 2.2 | 99.8 | 2.6 | 2.2 | 2.2 |
| GPT-2 | 24.9 | 2.6 | 113.6 | 3.0 | 3.0 | 3.0 |

## 5. MLIR export

`uv run --group bench python benchmarks/mlir_models.py` (raw: `results/mlir.json`).

| Model | Level | Nodes | MLIR lines | Ops left in mgc dialect | Verified by mlir-opt |
|---|---|---:|---:|---|---|
| ResNet-50 | O0 | 179 | 2,137 | none | yes |
| ResNet-50 | O3 | 58 | 1,216 | none | yes |
| MobileNetV2 | O0 | 152 | 2,011 | none | yes |
| MobileNetV2 | O3 | 56 | 1,224 | none | yes |
| BERT-base | O0 | 905 | 5,773 | none | yes |
| BERT-base | O3 | 245 | 3,469 | none | yes |
| GPT-2 | O0 | 1,248 | 6,633 | none | yes |
| GPT-2 | O3 | 232 | 3,131 | none | yes |

| Model | Lowered and built (s) | Max abs err vs ORT | Top-1 agrees | ms, 1 thread |
|---|---:|---:|---|---:|
| MobileNetV2 | 13 | 1.0e-05 | yes | 291 |

## Threats to validity

- **Single consumer ARM64 machine.** All runs execute on one 8-core Apple M1 (4 performance + 4
  efficiency cores, 8 GB unified memory; 1-minute load average 3.2 to 4.0 for CNNs, 3.0 to 8.2
  with median 5.0 for transformers). Engines are interleaved round robin across three rounds;
  where 95% bootstrap confidence intervals overlap, the result is called a tie.
- **Memory mapped weights.** The compiled model maps its weights file; on an 8 GB machine the OS
  can still reclaim clean pages of the 0.4 to 0.5 GB transformer weight blobs between subprocess
  rounds, while onnxruntime and PyTorch hold weights in anonymous memory.
- **Peak RSS** counts the Python process, not only the model; engines that import larger
  frameworks (PyTorch) carry a higher baseline.
- **Inputs.** Imagenette is easier than ImageNet; the text models see random token ids, not
  natural text. Parity is about identical outputs, not accuracy.
- **One shape.** Everything is batch 1 at one input size; other shapes are not measured.
- **Single machine.** Results on x86 or other ARM chips will differ, especially for the hand
  written NEON GEMM.
