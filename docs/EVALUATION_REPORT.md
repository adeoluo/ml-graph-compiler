# Evaluation Report: ML Graph Compiler

Every number here comes from a script in [`benchmarks/`](../benchmarks/), with the raw output in
[`benchmarks/results/`](../benchmarks/results/). `uv run python benchmarks/tables.py` regenerates
each table from those files.

## Setup

- **Hardware:** one Apple M1 laptop, 8 cores (4 performance, 4 efficiency), 8 GB RAM, macOS 27,
  shared with other jobs during the runs (load averages are recorded per cell). No GPU is used.
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

`benchmarks/latency.py` (raw: `results/latency_cnn.json`, `results/latency_transformers.json`,
`results/latency_transformers_run2.json`).
Each (engine, model, threads) cell is a fresh process: load, warm up (5 runs for CNNs, 3 for
transformers), time 30 (CNNs) or 10 (transformers) runs. Cells are interleaved round robin for 3
rounds and samples pooled; the confidence interval is a 2,000 sample bootstrap of the median.
Peak memory is the process peak RSS (`ru_maxrss`), which includes the Python interpreter and the
framework, and for this compiler the touched pages of the memory mapped weights file.

latency_cnn (latency_cnn): 48 cells, 1 minute load average during the cells min 11.2, median 50.0, max 89.3 (8 cores)
latency_transformers (latency_transformers, latency_transformers_run2): 96 cells, 1 minute load average during the cells min 3.0, median 5.5, max 12.2 (8 cores)

Threads = 1: median ms [95% CI] (peak RSS MB)

| Model | mgc (this) | onnxruntime | onnxruntime, no graph opt | PyTorch eager | mgc vs ORT | mgc vs PyTorch |
|---|---:|---:|---:|---:|---|---|
| ResNet-50 | 224.3 [207.0, 285.4] (431) | 212.9 [180.5, 267.4] (264) | 220.1 [196.1, 256.5] (185) | 61.3 [57.2, 68.7] (362) | tie | loss, 3.66x slower |
| MobileNetV2 | 16.0 [15.7, 17.0] (120) | 21.8 [18.8, 23.2] (113) | 18.0 [17.2, 20.1] (108) | 25.7 [23.9, 27.6] (237) | win, 1.37x | win, 1.61x |
| BERT-base | 483.1 [435.6, 545.3] (932) | 407.2 [399.7, 423.9] (647) | 535.7 [500.2, 557.0] (778) | 75.4 [73.8, 77.9] (668) | loss, 1.19x slower | loss, 6.41x slower |
| GPT-2 | 603.3 [586.4, 615.1] (1184) | 553.0 [544.3, 565.6] (836) | 615.9 [602.0, 639.5] (906) | 136.0 [133.8, 138.9] (847) | loss, 1.09x slower | loss, 4.44x slower |

Threads = 8: median ms [95% CI] (peak RSS MB)

| Model | mgc (this) | onnxruntime | onnxruntime, no graph opt | PyTorch eager | mgc vs ORT | mgc vs PyTorch |
|---|---:|---:|---:|---:|---|---|
| ResNet-50 | 58.6 [54.4, 61.6] (470) | 196.9 [149.1, 235.5] (281) | 194.3 [169.3, 232.2] (202) | 101.8 [88.2, 116.0] (462) | win, 3.36x | win, 1.74x |
| MobileNetV2 | 17.5 [15.9, 22.5] (116) | 30.9 [26.1, 46.0] (112) | 39.1 [27.7, 47.6] (111) | 78.9 [75.7, 88.1] (226) | win, 1.77x | win, 4.50x |
| BERT-base | 176.4 [164.1, 191.0] (895) | 307.6 [294.5, 336.8] (736) | 397.9 [279.4, 635.5] (836) | 76.5 [72.5, 81.5] (660) | win, 1.74x | loss, 2.31x slower |
| GPT-2 | 247.6 [232.7, 254.0] (1298) | 391.8 [367.6, 426.3] (865) | 404.1 [297.2, 450.7] (944) | 135.4 [133.0, 136.5] (840) | win, 1.58x | loss, 1.83x slower |

The CNN cells ran while the machine was shared with other heavy jobs (load averages far above
its 8 cores). The transformer cells were rerun on the transformers 5.22.2 exports with less load
(median 5.5, max 12.2) while another job was still running, and pool two complete runs
(`latency_transformers.json` and `latency_transformers_run2.json`, 6 rounds per cell). On
ResNet-50 and MobileNetV2 the compiled code beats onnxruntime at 8 threads (3.36x and 1.77x) and
PyTorch eager (1.74x and 4.50x); single threaded it ties onnxruntime on ResNet-50, wins on
MobileNetV2 and loses to PyTorch's tuned convolutions on ResNet-50 by 3.66x. On the transformers
it beats onnxruntime at 8 threads (BERT-base 1.74x, GPT-2 1.58x) and loses to it single threaded
(1.19x and 1.09x slower). PyTorch eager is the fastest engine on both transformers at both thread
counts (2.31x and 1.83x faster than mgc at 8 threads, 6.41x and 4.44x single threaded). Its macOS
build sends matrix multiplies to Apple Accelerate (`BLAS_INFO=accelerate` in
`torch.__config__.show()`) and runs as fast on 1 thread as on 8, while mgc's GEMM is portable
NEON code.

The previous transformer cells (transformers 4.57.6 exports, load averages from 7.2 to 118.2,
median 17.3) gave, at 8 threads, mgc 398.9 ms against onnxruntime 1257.3 ms and PyTorch 199.3 ms
on BERT-base, and 429.8 ms against 892.2 ms and 2182.5 ms on GPT-2; single threaded mgc lost to
onnxruntime by 1.99x and 2.56x with multi-second outliers from evicted weight pages. The rerun
replaces them: under less contention onnxruntime's and PyTorch's thread pools recover more than
mgc does, and the outliers are gone (the widest mgc interval is now 435.6 to 545.3 ms).

## 3. Pass ablation

`benchmarks/ablation.py` (raw: `results/ablation_cnn.json`, `results/ablation_transformers.json`).
Cumulative pipelines add one group of passes at a time; leave-one-out removes one pass from
`O3`. Latency is the median over 2 rounds of interleaved runs at 1 and 8 threads.

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
| BERT-base | O0 | 905 | 11.2 | 410.1 | 123.8 | 1.6e-05 |
| BERT-base | fold | 675 | 11.2 | 405.7 | 125.1 | 1.6e-05 |
| BERT-base | cse | 675 | 11.2 | 406.4 | 122.4 | 1.6e-05 |
| BERT-base | simplify | 675 | 11.2 | 404.5 | 121.6 | 2.1e-05 |
| BERT-base | fuse | 246 | 2.2 | 293.8 | 70.7 | 1.7e-05 |
| BERT-base | layout | 246 | 2.2 | 290.4 | 70.2 | 1.7e-05 |
| BERT-base | loopfuse | 245 | 2.2 | 290.8 | 69.8 | 1.7e-05 |
| GPT-2 | O0 | 1248 | 147.6 | 587.4 | 248.3 | 9.9e-05 |
| GPT-2 | fold | 675 | 147.6 | 607.9 | 247.8 | 9.9e-05 |
| GPT-2 | cse | 675 | 147.6 | 600.4 | 251.0 | 9.9e-05 |
| GPT-2 | simplify | 661 | 147.6 | 563.3 | 238.7 | 1.1e-04 |
| GPT-2 | fuse | 244 | 3.0 | 435.4 | 110.3 | 1.1e-04 |
| GPT-2 | layout | 244 | 3.0 | 415.2 | 102.9 | 1.1e-04 |
| GPT-2 | loopfuse | 232 | 3.0 | 439.5 | 104.7 | 1.1e-04 |

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
| BERT-base | O3 | 245 | 2.2 | 362.8 | 117.0 | 1.7e-05 |
| BERT-base | O3-no-fold | 247 | 2.2 | 370.2 | 118.4 | 1.7e-05 |
| BERT-base | O3-no-cse | 245 | 2.2 | 390.3 | 128.3 | 1.7e-05 |
| BERT-base | O3-no-simplify | 245 | 2.2 | 464.4 | 175.2 | 1.7e-05 |
| BERT-base | O3-no-fuse | 515 | 11.2 | 893.4 | 295.3 | 1.9e-05 |
| BERT-base | O3-no-layout | 245 | 2.2 | 737.4 | 194.8 | 1.7e-05 |
| BERT-base | O3-no-loopfuse | 246 | 2.2 | 714.6 | 195.4 | 1.7e-05 |
| GPT-2 | O3 | 232 | 3.0 | 1276.4 | 359.0 | 1.1e-04 |
| GPT-2 | O3-no-fold | 233 | 3.0 | 1105.5 | 324.5 | 1.1e-04 |
| GPT-2 | O3-no-cse | 232 | 3.0 | 1103.4 | 330.2 | 1.1e-04 |
| GPT-2 | O3-no-simplify | 378 | 3.0 | 774.0 | 279.9 | 1.1e-04 |
| GPT-2 | O3-no-fuse | 478 | 147.6 | 1111.4 | 440.6 | 9.9e-05 |
| GPT-2 | O3-no-layout | 232 | 3.0 | 680.1 | 246.0 | 1.1e-04 |
| GPT-2 | O3-no-loopfuse | 244 | 3.0 | 681.2 | 240.9 | 1.1e-04 |

Two passes carry most of the speed. Operator fusion folds BatchNorm into convolutions and merges
bias, residual and activation epilogues into the GEMM: it removes 63% to 67% of the nodes
(ResNet-50 175 to 57, MobileNetV2 152 to 55, BERT-base 675 to 246, GPT-2 661 to 244) and
shrinks the arena (GPT-2 from 147.6 MB to 3.0 MB: the output projection reads the 50257 x 768
embedding matrix through a transpose, which FusedGemm absorbs instead of materializing). The NHWC layout pass is
what makes the convolution GEMMs fast: removing it from `O3` makes MobileNetV2 3.3 times and
ResNet-50 2.3 times slower at 1 thread, and removing fusion costs 3.6 and 1.8 times. Constant
folding is required for the C backend on ResNet-50 (BatchNorm parameters must be constants) and
removes 25% of the BERT-base and 46% of the GPT-2 nodes, mostly shape arithmetic. CSE, algebraic
simplification and loop fusion change few nodes on these exported graphs, and their latency
effect is within the run to run noise of this loaded machine. The transformer ablation was rerun
on the transformers 5.22.2 exports. The cumulative transformer runs had the machine mostly to
themselves: fusion is the one pass with a clear effect there (BERT-base 404 to 294 ms at 1 thread
and 122 to 71 ms at 8 threads, GPT-2 563 to 435 ms and 239 to 110 ms), and layout and loop
fusion change nothing measurable on graphs without convolutions. Another job started during the
leave-one-out transformer runs, and their latencies drift upward in run order (`O3` itself
measured 363 ms for BERT-base and 1,276 ms for GPT-2 single threaded, against 291 ms and 440 ms
for the identical `loopfuse` pipeline in the cumulative study), so only their node counts, arena
sizes and errors are reliable. Every configuration stays within 1.2e-04 of onnxruntime.

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

- **Shared, overloaded machine.** The laptop ran other benchmark jobs at the same time; the 1
  minute load average during the latency cells is reported with each table. For the CNN cells
  it was far above the 8 cores and the machine was swapping; for the transformer cells (rerun
  on the transformers 5.22.2 exports) it was lower (median 5.5, max 12.2) but not idle. Interleaving the engines round robin spreads this
  noise across engines, but absolute times are inflated and the 8 thread results reflect
  behaviour under contention (spin-waiting thread pools suffer most). Where confidence
  intervals overlap the result is called a tie.
- **Memory mapped weights.** The compiled model maps its weights file; under memory pressure the
  OS can evict those pages between runs, which shows up as multi-second outliers for the 0.4 to
  0.5 GB transformer weights. onnxruntime and PyTorch hold weights in anonymous memory, which is
  swapped instead. These outliers dominated the earlier, heavily loaded transformer run and are
  absent from the rerun.
- **Peak RSS** counts the Python process, not only the model; engines that import larger
  frameworks (PyTorch) carry a higher baseline.
- **Inputs.** Imagenette is easier than ImageNet; the text models see random token ids, not
  natural text. Parity is about identical outputs, not accuracy.
- **One shape.** Everything is batch 1 at one input size; other shapes are not measured.
- **Single machine.** Results on x86 or other ARM chips will differ, especially for the hand
  written NEON GEMM.
