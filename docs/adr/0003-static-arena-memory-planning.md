# ADR-0003: Static arena memory planning with greedy-by-size placement

- **Status:** Accepted
- **Date:** 2026-11-05

## Context
An inference runtime that allocates each intermediate tensor separately pays for allocation on
every call and holds memory for tensors that are already dead. With static shapes (ADR-0002) and
a fixed schedule, every tensor's size and live interval are known before the model runs.

## Decision
Plan one arena at compile time (`memory.plan_memory`). Views (Reshape, Flatten, Squeeze,
Unsqueeze, Identity, Dropout) alias their input; an elementwise node may write over a same-shaped
input that dies at that node (in place). Alias groups are merged with union-find, live intervals
come from the schedule, and offsets are assigned with the greedy-by-size heuristic of Pisarchyk
and Lee (2020): largest group first, each at the lowest aligned offset that does not collide in
time and space with a placed group. The plan is verified (no two buffers alive together overlap)
and reported against the lower bound (the peak sum of live sizes). Before planning, a
memory-aware list scheduler reorders ready nodes to reduce that peak when it can.

## Alternatives considered
- **Per-tensor allocation at run time (malloc or a caching allocator)** - simple and supports
  dynamic shapes, but adds work per call and keeps fragmentation decisions out of reach of the
  compiler.
- **First fit in definition order** - implemented as `first_fit` for comparison; it is usually a
  little worse than greedy by size on these models (see the memory ablation).
- **Optimal placement with an ILP solver** - offset assignment is NP-hard; greedy by size is
  already at or near the lower bound on the four models, so a solver would add a dependency for
  little gain.

## Consequences
- The generated code performs no allocation; the caller allocates the arena once.
- The arena is a measurable, reported number per model and per pass pipeline.
- The planner must be told about aliasing precisely; a missed alias would corrupt data, so the
  plan verifier runs in tests and the benchmarks.
