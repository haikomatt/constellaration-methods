# E24r bank B (aspect ~12.0) status

**Label: Substantial (<= 0.030). Not feasible.**

Runner `p3_final_restoration.py` (the E22 command, unchanged), started from bank B's best confirmed
seed `db00537_n2_s231` (worst-of-three V 0.1136, via `nae_db_p3_generator.py candidate`, native
10x10, source budget 8x8). Container `p3-e24r-b14`, cores 2-13. Chain rule: material bar = start V
- 0.010 = 0.1036; substantial <= 0.030; goal feasible; null gain < 0.0025.

**Adjudication.** Best point `R6:L13:t0.5`, worst-of-three **V = 0.0143**, infeasible (QI residual
-3.45 as log10, mirror 0.208, flux 0.913, well +0.007; raw **aspect 12.01, min_len 8.80**). 1,831
optimisation evaluations, 3.4 h on 12 cores. The two rows still outside tolerance (QI 0.014, flux
0.014) are inside the recovery runner's usual reach.

**Objective-quality note.** The objective value reached here (min_len 8.80 at aspect 12.01) is
above every member of the own-lineage front (E23c's maximum 8.24) and sits in the range of the
davidkh-lineage E15 front at the same aspect -- the best indication yet that the aspect-14 bank's
operating point, not just its feasibility margin, is more competitive than the aspect-10 lineage.

**Label: SUBSTANTIAL** (V <= 0.030 cleared; goal, benchmark feasibility, not reached in this
link). `REPORT.json` is the runner's full adjudication record
(`results/p3-e24r-b14/REPORT.json` in the source campaign, boundary/candidate coefficient arrays
stripped by `tools/strip_boundaries.py`); `best_recheck.json` is the best point's residuals, raw
metrics and feasibility flag (no boundary coordinates).

**Not shipped.** `best_boundary.json` and `best_candidate.json` carry the design's Fourier boundary
coefficients; since the point is not feasible there is no own-lineage design to ship, and these
files are not copied here.

**Link 2 (queued).** A second restoration link from this link's adjudicated best point was
launched first, 2026-10-04 (`p3-e24r-b14-2`, identical command, material bar = this V - 0.010,
goal feasible). As of this writing it has not been adjudicated -- queued.
