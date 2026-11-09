# E27 status: strict-nominal restoration of the link-3 design

**Label: partial (against bars set before the run).**

The E24r link-3 design (aspect 7.37) was tolerance-feasible only (QI 0.009972, mirror 0.009995, flux
0.007284 against 0.01). E27 restores it toward strict nominal with the link command unchanged plus the
new `--success-tol 0.0` option, so the run stops only on a confirmed strictly nominal point.
Registered and locked 2026-10-23 (`docs/preregistration/20261023T105026Z-E27-nominal-restoration-lock.md`).
Bars on the adjudicated point: goal strictly nominal on all three evaluations; material worst V <=
0.0050; null worst V > 0.0075.

**Result.** 1,760 optimisation evaluations, 3.3 h, stop `recovery_failed`, **0 accepted steps**. Best
adjudicated point `JR:277:-` (a Jacobian probe), worst-of-three **V 0.0060** (from 0.0100), not
strictly nominal: QI 0.00527, mirror 0.00593, flux 0.00600, a three-way tie. Material (<= 0.0050)
not met, null (> 0.0075) not met, so **partial**. Aspect 7.372, min_len 5.046, solo hypervolume
76.36: the objective barely moved.

**Reading.** The margin on the tightest row grew from 5e-6 to 0.004, but the method did not reach
strict nominal: no trust-region step was accepted and the gain came from probe points. Strict
nominal at aspect 7.4 is not shown reachable by this method from here.

**Verified in both images.** `phase1`: both exports feasible. `leaderboard-fidelity` (0.2.1): the
reflected export is feasible with identical residuals; the un-reflected export fails the iota sign
only. Exports in `own-lineage/`, un-stripped under the own-lineage allowlist.
