# Pre-registration snapshots

Each file is a dated copy of the campaign's live pre-registration note, taken at the moment an
experiment was locked and committed before its launch, so that the bars, labels and stop rules can
be checked against the results afterwards. The originals live in the private campaign repository,
where each snapshot's commit message records its sha256. The copies here are verbatim except for two
redactions, both the names of private container images, each replaced by a placeholder wherever it
appeared: `<leaderboard-fidelity image>` (the image used for the leaderboard-pinned fidelity
re-check, present in every snapshot) and, from the E21 snapshot onward, `<phase1-generator image>`
(a second, phase-specific image built for the generator work). Nothing else, including the numbers,
has been changed.

19 snapshots, spanning E12 through E23 (the own-lineage front baseline from the first clean-provenance
P3-feasible design), including the E12b launch note, the atlas-probe and D1 diversity locks, the
E20/E20b construction locks, the E21 near-axis database seed screen, the E22 restoration lock (the
own-lineage feasibility link) and its E22bc repeatability-link lock, and the E23 front-builder lock.
