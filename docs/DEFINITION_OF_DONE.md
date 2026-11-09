# Definition of Done

A pull request is ready to merge when every item that applies is true. The pull request
template repeats this list as checkboxes.

## Code
- Typed, modular code with tests for every change in behavior; non-obvious design choices
  are recorded as an ADR in [docs/adr/](adr/).
- `make all` passes locally and continuous integration is green: format, lint, type check,
  and tests with coverage at or above the gate set in the project configuration.

## Results
- Every number in the README and the docs comes from a committed result file and says how
  to reproduce it (`make evaluation` or the benchmark script named next to the table), on
  what hardware, at what data size, and with which seeds or confidence intervals.

## Documentation
- The README keeps the standard section order, and the architecture diagram matches the code.
- Diagrams are `.mmd` sources in `docs/architecture/`, rendered with `make diagram` (shared
  theme, white background); Markdown files embed the rendered SVG, never a raw mermaid block.
- The model card, data card and threat model are current where the project has them.
- Demo GIFs show the current output.

## Security and hygiene
- No secrets in the history; CodeQL passes.
- The pull request title is a Conventional Commit; it becomes the changelog entry.
