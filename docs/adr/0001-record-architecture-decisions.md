# ADR-0001: Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
We need a lightweight, durable record of non-obvious engineering decisions so reviewers
understand the *why*, not just the *what*.

## Decision
Use Architecture Decision Records (ADRs). One Markdown file per significant decision,
numbered sequentially, using [`template.md`](template.md).

## Consequences
- Decisions are discoverable and reviewable in version control.
- Small ongoing cost to author an ADR per significant choice.
