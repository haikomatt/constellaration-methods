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

35 snapshots, spanning E12 through the E33 lock, including the E12b launch
note, the atlas-probe and D1 diversity locks, the E20/E20b construction locks, the E21 near-axis
database seed screen, the E22 restoration lock (the own-lineage feasibility link) and its E22bc
repeatability-link lock, the E23 front-builder lock, the E23b lock plus its pop-size amendment (the
first container exited at argument parsing before any evaluation ran), the E23c lock (third front
link) and the E23 host-runner auto-chain lock that carried the chain rule through to its 2% stop,
the E24-E26 breadth-queue lock (banks, conditional restoration, union front, mirror-023
definition change) plus its E24r-link-2/E25-re-launch amendment lock, and the refresh set of
2026-10-22 to 2026-11-01: the E24r link 3 and E25b lock and the E25 re-launch amendment, the E27
strict-nominal restoration lock, the E28 low-aspect front lock and its result-and-E28b amendment,
and the E29, E30, E31, E32 and E33 locks. E34 was a measurement with no bars and has no lock
snapshot. Later snapshots are copies of the whole live note as it stood at the lock, so each repeats
the earlier sections of that note.
