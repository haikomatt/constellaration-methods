# Questions for the Proxima team

Nine questions this repository's results raise, each with one line on why we're asking.

1. **Which definition changes do you actually make, how often, and which new metrics are
   coming?** Our one measured case (E18b: tightening the edge mirror bound) is a plausible guess,
   not something you've told us -- knowing the real pattern (a new constraint vs. promoting a
   metric to an objective) tells us which kind of change to measure re-optimisation against next.

2. **What compute and wall-clock budget does a re-optimisation get internally?** Our runs cost
   ~3,000 canonical evaluations per 9 hours on 12 cores; if you need an answer in an hour on a
   cluster, cheap surrogate-guided search matters more than our canonical-in-loop rigour, and if
   you can wait days, our approach is already in range.

3. **Which unmodelled costs worry you most?** Our diversity probe (D1) uses surface curvature and
   high-mode spectral energy as cost proxies with no coil evaluator; if you can name a real one
   (e.g. a coil-complexity solver), we can test whether the two geometrically distinct families we
   found actually separate on it.

4. **Is building fronts inside the 0.01 tolerance intended?** The public top fronts sit at
   residuals 0.006-0.010, i.e. they use the tolerance as free margin, while our E13 point is
   strictly feasible (V=0.0); knowing whether tolerance-riding is intended changes how we'd report
   any future front and whether to target V=0 or V=0.01.

5. **What are the reuse terms for derivatives of public leaderboard boundaries?** Our P3 work
   derived from a public P2 entry's boundary is measurement only without agreed terms (our eleven
   own-lineage designs do not depend on the answer), and warm-starting from a public front (our E18b case) may or may not be a
   legitimate method for a re-optimisation service depending on the answer.

6. **How would you use design-space freedom if it exists?** Our D1 measurement can report that two
   design families reach the same metrics, but only you know which real-world cost (manufacturing,
   maintenance, magnet stress) should decide between them.

7. **Would a "definition in, confirmed front out" service be useful, and in what form?** Code you
   run yourselves, a container, or a hosted job -- the answer decides what we'd build a
   demonstration of next.

8. **What produced your internal ~178 reference, and what beat it?** Knowing whether the leading
   methods are population search, gradient-based, or hand-seeded tells us whether our
   population-plus-canonical-confirmation route is in the same family as what already works for
   you, or a genuinely different one.

9. **Do fronts in one aspect region grow from a single design while fronts in another barely do?**
   In our results, fronts grown from a single design at aspect 9 to 11 reach 10 to 19% above that
   design's own hypervolume, and at aspect 7.3 to 8.6 they reach 1 to 5% (E30b, E32, E33 against
   E28, E29, E30a, E31; two pre-registered seeds, so only suggested). Is that something you see
   internally, and is aspect ratio a sensible axis for the diversity question you raised?
