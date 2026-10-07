# E24 bank B (aspect 14) status

**Label: Material.**

Rebuild of the E21 primary selection (960 seeds, same provenance) with the authors' MHD completion
at well target W = +0.02, geometry gate rmin/r00 >= 0.5, exported at aspect 14.0 by the boundary's
own estimate (`nae_db_p3_generator.py build --aspect 14.0`), replacements from the spares as in
E21. 960/960 seeds built; canonical aspect achieved 11.9-13.7, median 12.18 (requested 14.0). Host
build and once-only canonical screen on cores 2-13.

**Screen result.** 960/960 (100%) convergent. V median 0.399. **214 seeds at confirmed V <= 0.30**
with iota, mirror and vacuum well satisfied (the E21 material bar, aspect condition canonical
<= 16) -- compare E21 at aspect 10.4, which had 151. Best confirmed seed `db00537_n2_s231`,
worst-of-three **V = 0.1136**. 51% at log10 qi <= -2.5; 6% at flux compression <= 0.9.

**Reading.** The generator's best operating aspect for P3 is closer to ~12 than to E21/E22's ~10:
more seeds clear the material bar at this aspect (214 of 960, versus 151 of 960 at aspect 10.4),
and the best confirmed seed (0.1136) is materially closer to feasibility than either bank A's
0.2761 or E21's 0.1443.

**Label: MATERIAL** (a seed exists with confirmed V <= 0.30 and iota/mirror/well satisfied; null
not met). `REPORT.json` is the screen's full record (`results/p3-e24-b14/eval/REPORT.json` in the
source campaign, boundary/candidate coefficient arrays stripped by `tools/strip_boundaries.py`;
per-row distributions, convergence fractions, stratum/N summaries and the confirmed-seed table are
all present, no boundary coordinates).

**Not shipped.** The 960 individual seed boundaries (`results/p3-e24-b14/seeds/`,
`results/p3-e24-b14/eval/boundaries/`) carry Fourier boundary coefficients and are bank seed files,
not a submittable design -- they are reported only as counts and distributions above, never
redistributed.

**Follow-on.** Material -> conditional restoration from the best confirmed seed; see
`docs/results/p3/E24r-b14/STATUS.md`.
