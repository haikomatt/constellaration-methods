# E25 status

**Label: Material (E25 and E25b), with a correction: the union was not a union of the verified designs. See the correction below.**

The sections up to "Where it stands (2026-10-14)" record the two launch failures as they stood on
2026-10-14. What happened next follows them.

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

## Third launch failure and the re-launch (2026-10-22)

The front builder rebuilds each anchor from `<safe tag>.boundary.json` beside the anchors report, and
the union script never wrote those files. The re-launch died at once, with no evaluation. Fixed under
the regression ratchet (failing test first, then the fix): `write_anchor_boundaries` now writes them
and refuses to overwrite a file with a different boundary. The re-launch used the same lock, with five
anchors (E22, E22b, E22c, E24r B link 2 and the E24r A link-3 design, which had been verified that day;
see `docs/results/p3/E24r-a85-3/STATUS.md`).

## Result (adjudicated 2026-10-23 by the locked bars)

E25 (`docs/results/p3/E25-front/`): 2,968 canonical evaluations, 53 generations, stop `max_evals`;
confirmed front **45 members, hypervolume 95.81**, aspect 10.06 to 12.03, min_len 7.61 to 8.79.
Bars: material (>= 5 members) met, substantial (hypervolume >= 111.30) not met. **Material.** E25b
(`docs/results/p3/E25b-front/`), from E25's confirmed front: 72 members (55 strictly nominal),
hypervolume **97.34**, aspect 10.01 to 11.96, +1.6% on E25, under the 2% rule, so the chain stopped.
E25b at 97.34 is the best single own-lineage front we hold.

## Correction (2026-10-26): the two E24r anchors were drawn as different shapes

The engine never moves `r_cos(0,0)`, the major radius, so the run's template supplies it for every
coordinate vector. E22's template has 0.8143, the link-3 lineage 0.8016 and the aspect-12 lineage
0.8153. E25 seeded the link-3 design's exact coordinates (difference 1e-16) and evaluated them at
aspect 7.49 with worst residual 0.026, not 7.37 and 0.0100. The aspect-12 design came out at 11.959
instead of 11.973 but stayed feasible. The union script had checked mode counts, field periods and
scaling only.

What stands: every E25 and E25b front member is a real boundary, evaluated and confirmed canonically in
E22's representation, so the fronts, their hypervolumes (95.81, 97.34) and their labels stand. What is
withdrawn: E25 was not a union of the five designs, and the earlier reading that the low-aspect design
"dropped out because its perturbations were infeasible" was wrong, because the design was never
evaluated as itself.

Fix: the union script and the front builder both refuse anchors whose major radius differs from the
run's, and a new check (`p3_anchor_render_check.py`) compares every anchor's stored residuals with the
verbatim generation-0 evaluation in the archive. Run on E25's archive afterwards, it showed that only
E22's own anchor reproduced exactly: E22b and E22c (different database seeds) were re-rendered by 6e-4
and 1e-3 in residual, E24r B by 7e-4 and the link-3 design by 0.016. So E25's seeding was E22 plus four
nearby re-rendered shapes. A low-aspect front has to be built in its own representation, which is what
E28 onward do.
