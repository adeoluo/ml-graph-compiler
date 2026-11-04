# Definition of Done

> Per-project checklist derived from the portfolio-wide [Standard](../../STANDARD.md).
> A PR is mergeable only when every applicable box is checked.

## Engineering
- [ ] Core logic built from scratch; frameworks justified in README.
- [ ] Typed, modular, no god-files. ADR added for non-obvious decisions.

## CI/CD
- [ ] `make all` passes locally (format · lint · type · test).
- [ ] CI green on PR; coverage ≥ 80%.
- [ ] One-command run (`make run`) + Docker build works.

## Diagram
- [ ] Architecture diagram present, **centered title**, shared palette, legend.
- [ ] `.mmd` source + rendered `.svg` committed; embedded in README.

## Docs
- [ ] README: motivation · diagram · results table · design · limitations · roadmap.
- [ ] Demonstration GIF / video / live link.

## Evidence
- [ ] Results table with numbers vs. baseline.
- [ ] Multiple seeds + error bars / CI.
- [ ] Reproducible via `make evaluation`; dashboard linked.

## Responsible AI
- [ ] Model card, data card, threat model current.

## Security
- [ ] No secrets in history; CodeQL + gitleaks pass.

## Hygiene
- [ ] Conventional commits; LICENSE, CITATION, CONTRIBUTING present.
- [ ] Links to portfolio site; listed in profile README index.
