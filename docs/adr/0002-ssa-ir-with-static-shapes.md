# ADR-0002: An SSA graph IR with static shapes and one operator table

- **Status:** Accepted
- **Date:** 2026-11-04

## Context
The compiler imports ONNX graphs exported by torch.onnx (ResNet-50, MobileNetV2, BERT-base,
GPT-2). Those graphs carry Shape, Gather and Concat arithmetic that computes Reshape targets at
run time, decomposed LayerNorm and GELU, and initializers stored as raw protobuf. Every later
stage (folding, fusion, memory planning, C and MLIR emission) needs to know each tensor's dtype
and full shape, and needs to rewrite the graph cheaply.

## Decision
Import into a static single assignment graph (`ir.Graph`): every `Value` is defined once (graph
input, constant or node output) and keeps its list of users, so a rewrite is a local edit with
no index rebuild. The importer requires static shapes: symbolic dimensions are bound with
`--dim` or `--shape`, and shape inference partially evaluates small integer computations so that
Shape-driven Reshape targets become constants. One registry (`ops.REGISTRY`) gives each operator
its type inference and its numpy semantics; the reference interpreter, constant folding and shape
inference all call it, so an operator cannot behave differently in two stages.

## Alternatives considered
- **Keep the ONNX protobuf as the IR** - no translation step, but protobuf edits are slow and
  awkward, and def-use chains must be rebuilt after every rewrite.
- **Dynamic shapes with symbolic dimensions throughout** - covers more models, but memory
  planning and C emission would need run-time sizes, which removes the compile-time constant
  loop bounds and offsets that make the generated code simple and fast.

## Consequences
- Every tensor has a byte size at compile time, which the static memory planner relies on.
- A model compiled for batch 1 and sequence 128 runs only at that shape; another shape is
  another compilation (the build cache makes repeats free).
- Parity tests compare the interpreter against onnxruntime operator by operator, so the
  semantics in the table are themselves validated.
