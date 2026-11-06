"""MLIR textual export and an optional MLIR execution path.

``to_mlir(graph)`` prints the graph as an MLIR module. Two forms are available:

* ``dialect="linalg"`` (default) lowers operators to upstream dialects on tensors: ``linalg``
  (``linalg.generic`` with affine indexing maps, ``linalg.fill``, ``linalg.transpose``),
  ``tensor`` (``empty``, ``pad``, ``concat``, ``extract_slice``, ``collapse_shape``,
  ``expand_shape``, ``extract``), ``arith`` and ``math``. Operators without a lowering are kept as
  generic ``"mgc.<op>"`` operations, which ``mlir-opt --allow-unregistered-dialect`` accepts.
* ``dialect="mgc"`` prints every node as a generic ``"mgc.<op>"`` operation with its attributes,
  a one to one image of the IR.

When the LLVM tools are installed, :func:`compile_mlir` lowers a fully lowered module through
``mlir-opt`` (one-shot bufferization, loops, LLVM dialect), ``mlir-translate`` and clang into a
shared library, and :class:`MlirExecutable` runs it through the C interface, which validates the
export numerically.
"""

from ml_graph_compiler.mlir.emit import MlirModule, to_mlir
from ml_graph_compiler.mlir.execute import (
    MlirExecutable,
    MlirTools,
    compile_mlir,
    find_mlir_tools,
    mlir_verify,
)

__all__ = [
    "MlirExecutable",
    "MlirModule",
    "MlirTools",
    "compile_mlir",
    "find_mlir_tools",
    "mlir_verify",
    "to_mlir",
]
