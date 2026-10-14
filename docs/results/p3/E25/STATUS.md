# E25 status

**Launch FAILED twice (no evaluations run). Both causes fixed. Re-launch queued as a new lock.**

E25 was registered as a union front: every verified own-lineage feasible point available when
E24r finished (E22 `R0:L5:t0.5`, E22b `R0:L15:t0.5`, E22c `R0:L16:t0.5`, plus any E24r point whose
runner label is `verified_p3_feasibility`), merged into one E15-format `REPORT.json` by
`scripts/p3_union_anchors.py` and fed to `b3_nsga2_mhd.py --anchors` (the same anchors mode as
E23b/E23c), container `p3-e25`.

**First failure (merger bug).** The union anchors report produced by `p3_union_anchors.py` lacked
the `solo_hypervolume` field, which the builder reads before it will start; the container exited at
startup with no canonical evaluation run (`results/p3-e25-anchors/`,
`results/p3-e25-anchors-dryrun/` -- no `REPORT.json`, nothing to strip or ship). `p3_union_anchors.py`
was corrected to populate `solo_hypervolume` on every merged anchor, with a test added, 2026-10-04.

**Second failure (container-name collision).** The re-launch under
`scripts/p3_breadth_chain_runner2.sh` collided on a container name on 2026-10-04 and again ran no
evaluation. The chain runner was fixed on a branch and merged 2026-10-14 (`fix/chain-runner-name-collision`).

**Where it stands (2026-10-14).** Both launch failures are fixed and merged. The union now has four
verified own-lineage feasible anchors to draw on: E22, E22b, E22c and E24r B link 2 (aspect 12,
`docs/results/p3/E24r-b14-2/`). The re-launch is a new pre-registered lock and is still queued; no
E25 evaluation has run, so there is no result to report. This file will be updated once a
confirmation exists.
