# constellaration-methods

Code and evidence from an independent attempt at the three [ConStellaration](https://github.com/proximafusion/constellaration)
stellarator design problems, written up so that someone else can rerun the methods, not just read the scores.

## What is in here, in three lines

- A pipeline that takes a starting shape to a design that passes the benchmark's own checks: an optimiser
  with checkpointing, a feasibility restoration step, a population method that grows a Pareto front, and a
  confirmation step that re-evaluates every claimed design in fresh processes before it is called feasible.
- The results of running it: own-lineage designs on problems 1 and 2 that are on the public board, and on
  problem 3 the first design produced from a public seed database with our own pipeline, plus measurements
  of how fast the pipeline recovers when the problem definition changes and how different equally good
  designs can be.
- The discipline: every experiment was registered and frozen before it ran, and the frozen copies are in
  [docs/preregistration](docs/preregistration/), so the bars can be checked against the outcomes.

## Headline numbers

| Problem | Result | Whose lineage |
|---|---|---|
| P1 geometrical | 0.97522, on the public board | ours |
| P2 simple-to-build QI | 0.5818, on the public board | ours |
| P3 MHD-stable QI | three feasible designs from three database seeds through the identical chain, 4 to 6.5 hours each after a shared 2.8-hour generator step, and a confirmed 65-member front grown from the first of them (hypervolume 94.3) | ours, database CC BY 4.0 |
| P3, from a public leaderboard design | fronts up to hypervolume 124.2; a definition change recovered in 165 evaluations | derived, measurement only |

The full table with every experiment, its grade and its limits is in [docs/overview.md](docs/overview.md).
What we can and cannot claim is stated there before the numbers.

## Try it

```bash
podman build -t localhost/constellaration-methods:test .
podman run --rm -v "$(pwd)":/workspace:Z --entrypoint pytest localhost/constellaration-methods:test tests/ -q
```

That builds the pinned environment and runs 645 tests inside it. The runners are in `scripts/`; each has a
docstring that says what it does and what it reads. Start with `score_mhd_stable.py` to evaluate a boundary
the way the benchmark does, then `p3_final_restoration.py` to see the restoration step.

## Where to read next

- [docs/methods.md](docs/methods.md): the methods and what the evidence supports, refutes, or leaves open.
- [docs/overview.md](docs/overview.md): the complete results table, honesty guards, layout, and the scale-out option.
- [QUESTIONS.md](QUESTIONS.md): what we would like to ask the benchmark's authors.
- [PROVENANCE.md](PROVENANCE.md): where every design came from and what may be reused.

Apache-2.0. The P3 designs derived from another entrant's public boundary are shared as metrics only, not as
coordinates, until reuse terms exist.
