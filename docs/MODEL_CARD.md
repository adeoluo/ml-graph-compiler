# System Card: ML Graph Compiler

## Overview
- **What it is:** a compiler from ONNX graphs to native code (C built with clang) with a numpy
  reference interpreter and an MLIR text export. It does not train or change model weights.
- **Models exercised:** ResNet-50, MobileNetV2 (torchvision ImageNet weights), BERT-base uncased
  and GPT-2 124M (Hugging Face), exported to ONNX at opset 14.
- **Version:** 0.1.0.

## Intended use
- Inference of float32 ONNX models with static shapes on CPU (Apple silicon with NEON, x86 and
  ARM Linux with the portable kernels), and study of graph optimizations, scheduling and memory
  planning.

## Out of scope
- Training, dynamic shapes, quantized (int8) graphs, GPUs, and operators outside the supported
  table (custom domains are rejected).
- Compiling untrusted models outside a sandbox (see [THREAT_MODEL.md](THREAT_MODEL.md)).

## Performance
Numerical parity, latency and memory against onnxruntime and PyTorch are in
[EVALUATION_REPORT.md](EVALUATION_REPORT.md). Outputs differ from onnxruntime only by float32
summation order; the measured maximum errors and top-1 agreement are reported per model.

## Risks and limitations
- A compiled model is specialised to one input shape; another shape needs another compilation.
- Numerical differences grow with depth and with large logits (GPT-2), within float32 rounding.
- The generated code is loaded into the calling process; a compiler bug in bounds or memory
  planning would show as wrong outputs or a crash, which the plan verifier and parity tests
  are designed to catch.

## Footprint
All runs use one Apple M1 laptop (8 cores, 8 GB); no GPU.
