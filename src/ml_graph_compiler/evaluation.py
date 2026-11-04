"""Reproduce the README results table: `python -m ml_graph_compiler.evaluation`.

Replace the stub with the real benchmark. Always report mean plus or minus a
confidence interval over multiple seeds.
"""

from __future__ import annotations


def run_evaluation() -> dict[str, float]:
    # TODO: implement the real evaluation; return the headline metrics.
    return {"primary_metric": 0.0}


def main() -> None:
    results = run_evaluation()
    for name, value in results.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
