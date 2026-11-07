# ADR-0005: Convert convolution regions to NHWC with a transpose-sinking pass

- **Status:** Accepted
- **Date:** 2026-11-05

## Context
ONNX convolution networks are NCHW. For the C backend, NCHW makes a 1x1 convolution a strided
GEMM and forces depthwise kernels to vectorize along the spatial width, which is short in late
MobileNetV2 stages.

## Decision
`passes/layout.py` rewrites each NCHW convolution and pooling node as its NHWC form between two
Transposes, folds the OIHW to HWIO weight transpose into the constant, sinks Transposes through
elementwise operators, and lets simplification cancel adjacent inverse pairs. A chain of
convolutions then runs in NHWC with one transpose at the input and one before the classifier.

## Alternatives considered
- **Stay in NCHW and specialise kernels** - no graph rewrite, but every kernel needs an NCHW
  variant and 1x1 convolutions cannot be plain GEMMs.
- **Choose a layout per node with a cost model** - more general, but the two-transpose rewrite
  plus cancellation already removes every interior transpose on ResNet-50 and MobileNetV2.

## Consequences
- 1x1 convolutions are plain GEMMs and depthwise kernels vectorize over channels.
- The pass is measured separately in the pass ablation, including on BERT and GPT-2 where it
  should change nothing.
