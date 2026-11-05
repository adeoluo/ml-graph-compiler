"""Exception types raised by the compiler."""

from __future__ import annotations


class CompilerError(Exception):
    """Base class for every error raised by this package."""


class GraphError(CompilerError):
    """The graph is malformed (not SSA, cyclic, wrong types)."""


class ModelImportError(CompilerError):
    """The ONNX model cannot be imported."""


class UnsupportedOpError(CompilerError):
    """An operator (or an attribute value) is not supported by a stage of the compiler."""


class ShapeError(CompilerError):
    """Shape or type inference failed."""


class CodegenError(CompilerError):
    """C code generation or compilation failed."""
