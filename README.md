<h1 align="center">ML Graph Compiler</h1>

<p align="center">
  <em>A deep learning graph compiler: ONNX in, optimized C out.</em>
</p>

<p align="center">
  <a href="../../actions"><img alt="CI" src="https://img.shields.io/badge/CI-passing-brightgreen"></a>
  <img alt="Coverage" src="https://img.shields.io/badge/coverage-90%25-brightgreen">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <a href="DEMONSTRATION_URL"><img alt="Demonstration" src="https://img.shields.io/badge/demonstration-live-6366F1"></a>
</p>

---

## Motivation

What problem does this solve, and why does it matter? What's the insight or hard part?
Who is this for? (2 - 4 sentences - lead with the "so what".)

## Architecture

<p align="center">
  <img src="docs/architecture/architecture.svg" alt="System architecture" width="760">
</p>

One paragraph walking through the diagram: the request path, the core components, the
data stores, and where the model sits. Link to [ADRs](docs/adr/) for key decisions.

## Results

> Every claim is backed by a number. Reproduce with `make evaluation`.

| Metric | This project | Baseline | SOTA / reference |
|---|---:|---:|---:|
| Primary metric (↑) | **0.00** | 0.00 | 0.00 |
| Latency p95 (ms, ↓) | **0** | 0 | - |
| Cost / 1k req ($, ↓) | **0.00** | 0.00 | - |

Numbers are mean ± 95% confidence interval over N seeds. Full report: [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md).

<p align="center"><img src="docs/architecture/demonstration.gif" alt="Demonstration" width="640"></p>

## Quickstart

```bash
git clone <repository> && cd ML Graph Compiler
cp .env.example .env         # add your keys
make setup                   # install dependencies + pre-commit
make test                    # run the test suite
make run                     # run the application entrypoint
```

Or with Docker:

```bash
docker compose up --build
```

## Design decisions

- **Why X over Y** - short rationale, link to [ADR](docs/adr/).
- **Key tradeoff** - what you optimized for and what you gave up.

## Evaluation

How the numbers were produced: datasets, graders, seeds, hardware, cost.
See [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md) and the experiment dashboard: `W&B_URL`.

## Responsible AI

- **Model card:** [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
- **Data card:** [`docs/DATA_CARD.md`](docs/DATA_CARD.md)
- **Threat model:** [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md)

## Limitations

Honest list of what this does **not** do, known failure modes, and where it breaks.

## Roadmap

- [ ] Next improvement
- [ ] Stretch goal

## License

Released under the [MIT License](LICENSE).
