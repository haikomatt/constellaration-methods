# E24 bank A (aspect 8.5) status

**Label: Material.**

Rebuild of the E21 primary selection (`results/p3-e21-nae-db/selection_primary.json`, 960 seeds,
same provenance) with the authors' MHD completion at well target W = +0.02, geometry gate
rmin/r00 >= 0.5, exported at aspect 8.5 by the boundary's own estimate
(`nae_db_p3_generator.py build --aspect 8.5`), replacements from the spares as in E21. 901 seeds
built (59 replaced); canonical aspect achieved 7.2-8.4, median 7.74 (requested 8.5). Host build and
once-only canonical screen on cores 2-13.

**Screen result.** 867/901 (96.2%) convergent. V median 0.542. **7 seeds at confirmed V <= 0.30**
with iota, mirror and vacuum well satisfied (the E21 material bar, aspect condition canonical
<= 10.5). Best confirmed seed `db00398_n2_s364`, worst-of-three **V = 0.2761**. QI tail empty at
this aspect (0% at log10 qi <= -2.5); flux tail 16% at flux compression <= 0.9.

**Reading.** Low aspect buys flux-compression headroom and costs QI headroom, as E20b's
construction-survey trend predicted at aspect ~8-9 for the first-order generator; here it holds for
the second-order database generator too.

**Label: MATERIAL** (a seed exists with confirmed V <= 0.30 and iota/mirror/well satisfied; null
not met). `REPORT.json` is the screen's full record (`results/p3-e24-a85/eval/REPORT.json` in the
source campaign, boundary/candidate coefficient arrays stripped by `tools/strip_boundaries.py`;
per-row distributions, convergence fractions, stratum/N summaries and the confirmed-seed table are
all present, no boundary coordinates).

**Not shipped.** The 901 individual seed boundaries (`results/p3-e24-a85/seeds/`,
`results/p3-e24-a85/eval/boundaries/`) carry Fourier boundary coefficients and are bank seed files,
not a submittable design -- they are reported only as counts and distributions above, never
redistributed.

**Follow-on.** Material -> conditional restoration from the best confirmed seed; see
`docs/results/p3/E24r-a85/STATUS.md`.
