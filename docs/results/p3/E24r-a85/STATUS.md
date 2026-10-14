# E24r bank A (aspect ~7.4) status

**Label: Material by the chain rule (not substantial). Not feasible.**

Runner `p3_final_restoration.py` (the E22 command, unchanged), started from bank A's best confirmed
seed `db00398_n2_s364` (worst-of-three V 0.2761, via `nae_db_p3_generator.py candidate`, native
10x10, source budget 8x8). Container `p3-e24r-a85`, cores 2-13. Chain rule: material bar = start V
- 0.010 = 0.2661; substantial <= 0.030; goal feasible; null gain < 0.0025. Runner label
`substantial_but_infeasible_progress`.

**Adjudication.** Best point `R0:L7:t1.0`, worst-of-three **V = 0.0751**, infeasible (QI residual
-3.32 as log10, mirror 0.264, flux 0.968, well +0.016; raw aspect 7.38, min_len 5.09). 1,853
optimisation evaluations, 2.9 h on 12 cores. Gain against the material bar (0.2661 -> 0.0751) is
well clear; the restoration did not reach feasibility in this link.

**Label: MATERIAL by the chain rule** (progress clears the material and substantial thresholds but
the point is not feasible; goal unmet). `REPORT.json` is the runner's full adjudication record
(`results/p3-e24r-a85/REPORT.json` in the source campaign, boundary/candidate coefficient arrays
stripped by `tools/strip_boundaries.py`); `best_recheck.json` is the best point's residuals, raw
metrics and feasibility flag (no boundary coordinates -- carries no key from
`tools/strip_boundaries.py`'s banned set).

**Not shipped.** `best_boundary.json` and `best_candidate.json` carry the design's Fourier boundary
coefficients; since the point is not feasible there is no own-lineage design to ship, and these
files are not copied here.

**Link 2 (adjudicated 2026-10-14).** A second restoration link from this link's adjudicated best
point (`p3-e24r-a85-2`, identical command, material bar = this V - 0.010, goal feasible) reached
worst-of-three V 0.0212, substantial but not feasible; see `docs/results/p3/E24r-a85-2/STATUS.md`.
