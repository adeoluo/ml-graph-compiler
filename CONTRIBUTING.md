# Contributing

## Development setup

```bash
make setup     # dependencies + pre-commit hooks
make all       # local gate: lint + type + test (mirrors continuous integration)
```

## Workflow

1. Branch from `main` (`feat/...`, `fix/...`, `docs/...`).
2. Use **Conventional Commits** (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
3. Keep pull requests focused; fill in the pull request template's Definition-of-Done checklist.
4. Continuous integration must be green before merge.

## Standards

This repository follows the portfolio-wide [Standard / Definition of Done](docs/DEFINITION_OF_DONE.md).
Non-obvious design choices get an [ADR](docs/adr/). Every claim in the README is backed
by a reproducible number (`make evaluation`).
