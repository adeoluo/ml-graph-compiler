"""Entrypoint: `python -m ml_graph_compiler`."""

from __future__ import annotations

import sys

from ml_graph_compiler.cli import main

if __name__ == "__main__":
    sys.exit(main())
