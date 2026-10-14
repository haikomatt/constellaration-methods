# E24r bank A, link 2 (aspect ~7.4) status

**Label: Substantial, not feasible. Link 3 not launched.**

Runner `p3_final_restoration.py` (the E22 command, unchanged), started from link 1's adjudicated
best point (worst-of-three V 0.0751 at aspect 7.4; see `docs/results/p3/E24r-a85/STATUS.md`), which
itself came from bank A's best confirmed database seed `db00398_n2_s364`. Container
`p3-e24r-a85-2`, cores 2-13. Chain rule: material bar = start V - 0.010 = 0.0651; substantial
<= 0.030; goal feasible. Runner label `substantial_but_infeasible_progress`, stop reason
`refresh_requested_after_recovery`.

**Adjudication.** Best confirmed point `D28:0:-`, worst-of-three **V = 0.0212**, infeasible: the
QI (0.0212), mirror (0.0212) and flux (0.0206) residuals are tied just outside tolerance, iota and
well strict. Aspect about 7.4. 1,843 optimisation evaluations, 9 verification evaluations, 4.08 h
on 12 cores, 6 accepted steps. Material and substantial are both met.

**What this shows, and what it does not.** The chain rule would fund a third link (material bar
0.0112); it was not launched. Against bank B, which reached feasibility in its second link at
aspect 12, bank A has not reached feasibility at aspect 7.4 in two links, with three rows tied at
0.021. That is one seed and two links, not a refutation of low-aspect designs; read with E24 it
says the generator's best operating aspect for P3 is about 12 to 14, and feasibility at 7.4 has not
been reached.

**Not shipped.** The run's boundary and candidate exports carry Fourier coefficients of a design
that is not feasible, so there is no own-lineage design to ship and they are not copied here.
`REPORT.json` is the runner's adjudication record with coefficient arrays stripped by
`tools/strip_boundaries.py`.
