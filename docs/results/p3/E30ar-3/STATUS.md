# E30a bank (aspect 9.0 request), restoration link 3 status

**Label: Supported (goal). The seventh own-lineage P3-feasible design.**

Bank request 9.0: 951 built, best confirmed seed `db00398_n2_s364` 0.2566 (material). Link 1 took it
to 0.0408 and link 2 to 0.0154 (both material). **Link 3** (1,793 evaluations): point `JR:261:+`,
residuals [-0.254, -0.00159, 0.00509, 0.00358, -0.156], worst V 0.0051, tolerance-feasible.
**Aspect 7.637, min_len 5.308, solo hypervolume 77.98.** It came out at 7.64, not the 8.0 expected, so
the 7.4 to 8.6 gap is only partly filled. Registered and locked 2026-10-27
(`docs/preregistration/20261027T211746Z-E30-lock.md`).

Verified in both images through the reflected export: `phase1` both exports feasible;
`leaderboard-fidelity` (0.2.1) reflected feasible with identical residuals, un-reflected fails the
iota sign only. Feasible under 0.2.6 for both exports (E34).
