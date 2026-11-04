"""Entrypoint: `python -m ml_graph_compiler`."""

from __future__ import annotations

from ml_graph_compiler import __version__
from ml_graph_compiler.configuration import settings


def main() -> None:
    header = f"Ml Graph Compiler v{__version__}"
    status = f"environment={settings.environment}, seed={settings.seed}"
    print(f"{header} - {status}")


if __name__ == "__main__":
    main()
