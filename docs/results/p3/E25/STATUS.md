# E25 status

**Launch FAILED (bug, no evaluations run). Fixed. Re-launch queued.**

E25 was registered as a union front: every verified own-lineage feasible point available when
E24r finished (E22 `R0:L5:t0.5`, E22b `R0:L15:t0.5`, E22c `R0:L16:t0.5`, plus any E24r point whose
runner label is `verified_p3_feasibility`), merged into one E15-format `REPORT.json` by
`scripts/p3_union_anchors.py` and fed to `b3_nsga2_mhd.py --anchors` (the same anchors mode as
E23b/E23c), container `p3-e25`.

**What happened.** The union anchors report produced by `p3_union_anchors.py` lacked the
`solo_hypervolume` field, which the builder reads before it will start; the container exited at
startup with no canonical evaluation run (`results/p3-e25-anchors/`,
`results/p3-e25-anchors-dryrun/` -- no `REPORT.json`, nothing to strip or ship).

**Fix.** `p3_union_anchors.py` was corrected to populate `solo_hypervolume` on every merged anchor,
with a test added, 2026-10-04.

**Re-launch (queued).** A re-launch with the fixed merger is registered under
`scripts/p3_breadth_chain_runner2.sh`: anchors = E22, E22b, E22c plus any E24r link-2 point whose
fresh re-check is feasible, chained under the same 2% rule as E23b/E23c, hard stop for new launches
2026-10-06 06:00. As of this writing it has not run -- queued. No result to report; this file will
be updated once a confirmation exists.
