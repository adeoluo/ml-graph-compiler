# Threat Model: ML Graph Compiler

The compiler reads an ONNX file, generates C source, builds it with the system C compiler and
loads the resulting shared library into the calling Python process. The ONNX file is the
untrusted input; everything downstream of it executes as native code with the caller's rights.

## Assets
- The integrity of the host process and machine that compiles and runs a model.
- The correctness of the compiled model's outputs (a silent numerical error is a failure).
- The build cache (`~/.cache/ml-graph-compiler` or `$MGC_CACHE_DIR`), whose libraries are loaded
  without recompilation.

## Trust boundaries
1. ONNX file to importer: protobuf parsing, initializers, node and graph names, attributes.
2. IR to generated C or MLIR text: anything model-controlled that is printed into source.
3. Generated source to clang and the loaded shared library (`ctypes.CDLL`).
4. Caller-supplied input tensors to the compiled `mgc_run` function.

## Threats

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Code injection into generated C | A node or graph name containing `*/` or a line break printed in a comment | Arbitrary native code compiled and run | `ir.comment_text` makes names printable and breaks comment terminators before printing (C and MLIR); identifiers in code are generated, never taken from the model; regression tests with hostile names |
| Out-of-bounds read through Gather | Token or index input outside the table | Memory disclosure or crash | The generated Gather checks every index and `mgc_run` returns an error, raised as `GraphError` |
| Out-of-bounds write from shape mismatch | Caller passes an array of the wrong shape or dtype | Heap corruption | `CompiledModel.run` checks every input's shape and dtype against the compiled signature before the call |
| Arena overlap bug | Planner places two live tensors at overlapping offsets | Silent wrong outputs | `MemoryPlan.verify` checks every pair of buffers alive at the same time; it runs in tests and benchmarks; outputs are checked against onnxruntime |
| Unsupported or custom operators | Domain other than `ai.onnx`, unknown op type, out-of-range opset | Undefined behavior | The importer rejects foreign domains and opsets outside 11 to 21; unknown ops fail in type inference before code generation |
| Resource exhaustion | Very large initializers or shapes | Memory or disk exhaustion | Shapes are static and sizes known before allocation; the CLI reports arena and weight sizes; run untrusted models in a container with memory limits |
| External data path traversal | ONNX external-data locations pointing outside the model directory | Reading arbitrary files into weights | Loading is delegated to `onnx.load`, which validates external-data paths in current releases; keep `onnx` up to date |
| Cache poisoning | Another local user writes a library into the cache directory | Code execution when the cache hit loads it | Cache keys hash the source and flags; the cache lives under the user's own home directory by default; set `MGC_CACHE_DIR` to a private directory on shared machines |
| Supply chain | Malicious dependency | Code execution | Two runtime dependencies (numpy, onnx), pinned in `uv.lock`; CodeQL runs on every push to main |

## Residual risk
- The compiler trusts the system C compiler and the loaded library completely; a library in a
  writable cache directory is as trusted as the code that wrote it.
- Numerical behavior follows float32 rounding of a different summation order than onnxruntime;
  differences are measured and reported, not eliminated.
- Compiling a model from an unknown source should happen in a sandbox or container, as with any
  tool that turns data into native code.
