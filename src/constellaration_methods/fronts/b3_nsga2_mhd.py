#!/usr/bin/env python3
"""B3 -- NSGA-II population on MHDStableQIStellarator (pre-registration:
`an-nsga-ii-population-reproduces-a-competitive-mhd-stable-pareto-front.md`).
Doubles as M1+M2 of `mhd-stable-entry-plan.md`.

Hypothesis: davidkh's P3 entry is a 200-design population with
tolerance-riding margins -- the NSGA-II fingerprint. ONE NSGA-II run on
`constellaration.problems.MHDStableQIStellarator` (objectives: maximise
minimum normalized magnetic gradient scale length, minimise aspect ratio;
feasibility: its own 5-constraint set) should reproduce a competitive
Pareto front from our own lineage.

Algorithm source: `pymoo.algorithms.moo.nsga2.NSGA2` (confirmed importable
in the pinned image, v0.6.1.5) -- used via its own default operators
(SBX crossover, polynomial mutation for continuous variables), driven with
a manual ask/tell loop (not `pymoo.optimize.minimize`) so every generation
can be banked incrementally and the two/three real `evaluate()` calls
placed exactly where the pre-registration puts them. A from-scratch
minimal NSGA-II (non-dominated sort + crowding distance + tournament +
SBX/gaussian mutation, ~150 lines, e.g. Deb et al. 2002 "A Fast and Elitist
Multiobjective Genetic Algorithm: NSGA-II") was NOT needed -- documented
here in case the image changes and this script needs to fall back to it.

Representation: the SAME masked/scaled 8x8-mode-budget coordinate space
B1 uses (`alm_ngopt_optimizer.masked_coefficients_and_scale`) -- every
seed boundary (whatever its native mode budget) is re-expressed in this
one 288-dim scaled vector space so pymoo's continuous operators (SBX,
polynomial mutation) act in a space where per-coefficient scale is
meaningful, exactly the reason B1 uses it for CMA.

In-loop fitness: CHEAP fidelity (`alm_ngopt_optimizer._search_settings_for`,
reused verbatim). F = [-min_grad_scale_length, aspect_ratio] (both
minimised, matching `problem.get_objectives`' maximise/minimise flags);
G = `problem._normalized_constraint_violations(metrics)` (the problem's
own 5-line constraint vector, <=0 feasible -- pymoo's own convention).
Non-convergent candidates get a large finite objective/constraint penalty
(never nan/inf).

Final/real scoring: `problem.evaluate(<list of SurfaceRZFourier>)` -- the
EXACT leaderboard hypervolume call -- capped at <=3 total calls per the
pre-registration (used here: one mid-run, one final; the third is held in
reserve, never fired without being asked for). This is expensive (full
`default_high_fidelity` VMEC++ + Boozer per boundary in the list), so it
is never called per-generation.

Seeding (own-lineage only -- NO leader boundaries, per the pre-registration
and this campaign's standing rule): our feasible P2 bank --
`results/family-study/climb/c8_climb_bank.jsonl` (13 records, includes
call_index 30548) + `results/family-study/diagnostics/b1_probe_bank.jsonl`
(8 records, native 4x4 budget, zero-padded to 8x8 by
`masked_coefficients_and_scale`'s own convention) -- plus Gaussian
perturbations of those 21 seeds and fresh own-lineage samples built the
same way B1's `scratch` arm builds its generic seed
(`constellaration.initial_guess.generate_nae`, randomised own-recipe
parameters, own RNG -- never the leaderboard). Exact mix is documented in
`build_seed_population`.

Caps (HARD, pre-registration): <=24h wall, <=3 real `evaluate()` calls.
Every generation's population (X, cheap F/G) is banked to
`results/b3-nsga2/generations.jsonl`; every real `evaluate()` call's result
to `results/b3-nsga2/real_evaluate_log.jsonl`.

`--smoke`: population 12, 3 generations, cores 14-15 ONLY, NO real
`evaluate()` call -- must run end-to-end in the container. Never touches
cores 2-13 (the live `p1-overnight` run).

Usage (inside the pinned image, smoke, cores 14-15 ONLY):
    podman run --rm -v "$(pwd)":/workspace:Z -w /workspace \\
        localhost/private-image:phase1 \\
        nice taskset -c 14,15 python scripts/b3_nsga2_mhd.py --smoke

Real run (cores 2-13, NOT to be fired until that range is free):
    podman run --rm -v "$(pwd)":/workspace:Z -w /workspace \\
        --name b3-nsga2-mhd \\
        localhost/private-image:phase1 \\
        nice taskset -c 2-13 python scripts/b3_nsga2_mhd.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent import futures
from pathlib import Path
from typing import Any

import numpy as np

from constellaration_methods.engines import thread_limits

# `alm_ngopt_optimizer` pulls in jax/nevergrad/constellaration at ITS OWN module
# level, so every name from it is imported lazily, inside the functions that
# need it, rather than here -- the same convention `p3_promote_candidate.py`
# and `p3_feasibility_restoration.py` already use for the identical helpers,
# and the one this file needs so its pure/E17 logic stays host-importable
# (and unit-testable without the pinned image) exactly like those sibling
# scripts, none of which import anything heavier than numpy at module level.
from constellaration_methods.fronts.p3_front_builder import _is_fake, load_anchors
from constellaration_methods.fronts.p3_front_from_archive import (
    best_v_of_rows,
    hypervolume,
    nondominated,
    tolerance_feasible_rows,
    v_curve_from_rows,
)
from constellaration_methods.problems.overrides import (
    parse_constraint_overrides,
    problem_with_overrides,
)
from constellaration_methods.restoration.p3_feasibility_restoration import (
    TOL,
    Budget,
    CanonicalEvaluator,
)
from constellaration_methods.restoration.p3_final_restoration import representation_of
from constellaration_methods.restoration.p3_promote_candidate import (
    coefficient_space_multiplier,
    old_mode_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

C8_CLIMB_BANK = REPO_ROOT / "results" / "family-study" / "climb" / "c8_climb_bank.jsonl"
B1_PROBE_BANK = REPO_ROOT / "results" / "family-study" / "diagnostics" / "b1_probe_bank.jsonl"
OUT_DIR = REPO_ROOT / "results" / "b3-nsga2"

MAX_POLOIDAL_MODE = 8
MAX_TOROIDAL_MODE = 8
N_FIELD_PERIODS = 3

# ---------------------------------------------------------------------------
# HARD CAPS -- pre-registration, enforced below, not conventions.
# ---------------------------------------------------------------------------
MAX_WALL_SECONDS = 24 * 3600.0
MAX_REAL_EVALUATE_CALLS = 3
POP_SIZE_REAL = 200

NON_CONVERGENT_OBJECTIVE_PENALTY = 1.0e2  # large but finite; never inf/nan
NON_CONVERGENT_CONSTRAINT_PENALTY = 1.0e2

# Thresholds from the pre-registration (reported, not enforced in-driver --
# scoring happens after the run, against the real evaluate() score).
SUPPORTED_SCORE = 129.8  # published ALM-NGOpt reference
STRONG_SCORE = 133.6  # ExLLM

# Documented choice: the pre-registration fixes population (200) and caps
# (wall, <=3 real evaluate() calls) but not a generation COUNT -- the run
# is wall-clock-bounded, not generation-bounded. This default only sets
# where the "mid-run" real evaluate() call fires (roughly the halfway
# point of a run expected to fit the wall cap); it is a scheduling
# convenience, not a substantive parameter of the method.
DEFAULT_MID_RUN_GENERATION = 50

# Perturbation / fresh-sample mix for the initial population (own-lineage
# only; documented since the pre-registration doesn't pin exact counts).
GAUSSIAN_PERTURBATION_SIGMA = 0.05  # scaled-space sigma, small local jitter
BOUND_FACTOR = 3.0  # per-dim box = BOUND_FACTOR * observed seed spread
BOUND_MIN_RADIUS = 0.05


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_state(path: Path) -> dict[str, Any] | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# search-space plumbing (shared shape with b1_cma_long.py)
# ---------------------------------------------------------------------------


def boundary_dict_to_scaled_x(
    boundary_dict: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Any]:
    from constellaration_methods.engines.alm_ngopt_optimizer import (
        masked_coefficients_and_scale,
    )

    boundary, mask, masked_coeffs, unravel_and_unmask_fn, scale = masked_coefficients_and_scale(
        boundary_dict,
        max_poloidal_mode=MAX_POLOIDAL_MODE,
        max_toroidal_mode=MAX_TOROIDAL_MODE,
        n_field_periods=N_FIELD_PERIODS,
    )
    scale_np = np.asarray(scale, dtype=float)
    x = np.asarray(masked_coeffs, dtype=float) / scale_np
    return x, scale_np, unravel_and_unmask_fn


def build_fresh_own_lineage_boundary(rng: np.random.RandomState) -> dict[str, Any]:
    """A fresh own-lineage sample -- same `generate_nae` recipe B1's
    `scratch` arm uses, but with randomised own-recipe parameters (never
    derived from or approximating the leaderboard). Ranges are the
    problem's own feasibility-relevant bounds (own-lineage: our design
    intent, not a copy of anyone else's boundary)."""
    from constellaration import initial_guess as ig

    aspect_ratio = float(rng.uniform(6.0, 10.0))
    max_elongation = float(rng.uniform(3.0, 5.0))
    mirror_ratio = float(rng.uniform(0.15, 0.25))
    iota_over_n = float(rng.uniform(0.25, 0.45))
    surface = ig.generate_nae(
        aspect_ratio=aspect_ratio,
        max_elongation=max_elongation,
        rotational_transform=iota_over_n * N_FIELD_PERIODS,
        mirror_ratio=mirror_ratio,
        n_field_periods=N_FIELD_PERIODS,
        max_poloidal_mode=1,
        max_toroidal_mode=1,
    )
    return {
        "r_cos": np.asarray(surface.r_cos, dtype=float).tolist(),
        "z_sin": np.asarray(surface.z_sin, dtype=float).tolist(),
        "n_field_periods": N_FIELD_PERIODS,
    }


def build_seed_population(pop_size: int, seed: int) -> np.ndarray:
    """Own-lineage-only initial population, mixed per the pre-registration:
    our feasible P2 bank + Gaussian perturbations of it + fresh own-recipe
    samples. Returns an (pop_size, 288) array in the SAME scaled coordinate
    space every individual is optimised in."""
    rng = np.random.RandomState(seed)
    bank_records = load_jsonl(C8_CLIMB_BANK) + load_jsonl(B1_PROBE_BANK)
    if not bank_records:
        raise RuntimeError("no bank records found -- own-lineage seeding requires them")
    bank_x = []
    for rec in bank_records:
        x, _scale, _unravel = boundary_dict_to_scaled_x(
            {
                "r_cos": rec["boundary"]["r_cos"],
                "z_sin": rec["boundary"]["z_sin"],
                "n_field_periods": rec["n_field_periods"],
            }
        )
        bank_x.append(x)
    bank_x = np.stack(bank_x)  # (n_bank, 288)
    n_bank = bank_x.shape[0]

    population = []
    # 1. the bank itself, verbatim (as many as fit).
    for i in range(min(n_bank, pop_size)):
        population.append(bank_x[i])
    remaining = pop_size - len(population)
    if remaining <= 0:
        return np.stack(population[:pop_size])

    # 2. Gaussian perturbations of randomly-chosen bank members (own-lineage
    # -- every perturbation is centred on one of OUR OWN feasible designs).
    n_perturb = remaining // 2
    for _ in range(n_perturb):
        base = bank_x[rng.randint(n_bank)]
        population.append(base + rng.normal(scale=GAUSSIAN_PERTURBATION_SIGMA, size=base.shape))
    remaining = pop_size - len(population)

    # 3. fresh own-recipe samples (own RNG, own parameter ranges -- never
    # derived from any other design).
    for _ in range(remaining):
        bd = build_fresh_own_lineage_boundary(rng)
        x, _scale, _unravel = boundary_dict_to_scaled_x(bd)
        population.append(x)

    return np.stack(population[:pop_size])


def compute_bounds(seed_population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Data-derived box bounds (same idiom as `qi_polish.trust_region_radius`
    -- how far the seed population demonstrably spans, scaled up by
    `BOUND_FACTOR`, floored at `BOUND_MIN_RADIUS` per dimension -- rather
    than an arbitrary independent guess)."""
    center = seed_population.mean(axis=0)
    spread = np.abs(seed_population - center).max(axis=0)
    radius = np.maximum(BOUND_FACTOR * spread, BOUND_MIN_RADIUS)
    return center - radius, center + radius


# ---------------------------------------------------------------------------
# physics evaluation
# ---------------------------------------------------------------------------


def _eval_candidate_mhd(
    x: np.ndarray,
    scale: np.ndarray,
    unravel_and_unmask_fn: Any,
    settings: Any,
    constraint_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Top-level (picklable) worker: cheap-fidelity forward model on ONE
    candidate for `MHDStableQIStellarator`'s own objectives/constraints.
    `constraint_overrides` (E18 item 1) is threaded straight to
    `problem_with_overrides`; empty/None is the stock class."""
    from constellaration import forward_model

    problem = problem_with_overrides(constraint_overrides)
    boundary = unravel_and_unmask_fn(np.asarray(x) * scale)
    try:
        metrics, _eq = forward_model.forward_model(boundary=boundary, settings=settings)
    except Exception as e:  # noqa: BLE001
        return {"convergent": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    if metrics is None:
        return {"convergent": False, "error": "forward_model returned metrics=None"}
    objectives = problem.get_objectives(metrics)
    constraints = problem._normalized_constraint_violations(metrics)
    return {
        "convergent": True,
        "objectives": [float(objectives[0][0]), float(objectives[1][0])],
        "minimize_flags": [bool(objectives[0][1]), bool(objectives[1][1])],
        "constraints": [float(c) for c in constraints],
    }


def evaluate_population_cheap(
    X: np.ndarray,
    scale: np.ndarray,
    unravel_and_unmask_fn: Any,
    settings: Any,
    executor: futures.ProcessPoolExecutor,
    constraint_overrides: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    futs = [
        executor.submit(
            _eval_candidate_mhd,
            x,
            scale,
            unravel_and_unmask_fn,
            settings,
            constraint_overrides,
        )
        for x in X
    ]
    results = [f.result() for f in futs]
    F = np.zeros((len(X), 2))
    G = np.zeros((len(X), 5))
    for i, r in enumerate(results):
        if not r["convergent"]:
            F[i] = [NON_CONVERGENT_OBJECTIVE_PENALTY, NON_CONVERGENT_OBJECTIVE_PENALTY]
            G[i] = [NON_CONVERGENT_CONSTRAINT_PENALTY] * 5
            continue
        min_grad_len, aspect_ratio = r["objectives"]
        # `get_objectives` returns (value, minimize); minimize=False for
        # min_grad_len (maximise) -> minimise its negative; minimize=True
        # for aspect_ratio already.
        F[i, 0] = -min_grad_len if not r["minimize_flags"][0] else min_grad_len
        F[i, 1] = aspect_ratio if r["minimize_flags"][1] else -aspect_ratio
        G[i] = r["constraints"]
    return F, G, results


def real_evaluate_population(
    X: np.ndarray,
    scale: np.ndarray,
    unravel_and_unmask_fn: Any,
    constraint_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The EXACT leaderboard call -- `problem.evaluate(list[boundaries])`,
    real (`default_high_fidelity`) fidelity, hypervolume score. Capped by
    the caller at <=3 total calls per the pre-registration."""
    problem = problem_with_overrides(constraint_overrides)
    boundaries = [unravel_and_unmask_fn(np.asarray(x) * scale) for x in X]
    t0 = time.perf_counter()
    ev = problem.evaluate(boundaries)
    n_feasible = sum(1 for f in ev.feasibility if f <= 0)
    return {
        "score": float(ev.score),
        "n_feasible": int(n_feasible),
        "n_total": len(boundaries),
        "elapsed_s": time.perf_counter() - t0,
    }


def cheap_proxy_hypervolume(results: list[dict[str, Any]]) -> float | None:
    """Monitoring-only cheap-fidelity proxy hypervolume via the problem's
    own `_score`/`_hypervolume` machinery, reused (not reimplemented) --
    NOT the real leaderboard score and never counted against the
    <=3-real-evaluate() cap (no `problem.evaluate()` call here, just its
    private scorer applied to cheap-fidelity metrics we already computed
    is impossible without re-deriving ConstellarationMetrics, so this is
    approximated directly from feasibility + objective values instead)."""
    feasible_points = []
    for r in results:
        if not r["convergent"]:
            continue
        if all(c <= 0.0 for c in r["constraints"]):
            min_grad_len, aspect_ratio = r["objectives"]
            feasible_points.append((-min_grad_len, aspect_ratio))
    if not feasible_points:
        return 0.0
    from pymoo.indicators.hv import HV

    X = np.array(feasible_points)
    hv = HV(ref_point=np.array([1.0, 20.0]))
    return float(hv(X))


# ---------------------------------------------------------------------------
# E17 -- seeding an NSGA-II population from an existing confirmed front
# (extends the seeding/evaluation machinery above with an alternative
# starting point and fidelity: reuses p3_front_builder's anchor/representation
# loader, p3_feasibility_restoration's CanonicalEvaluator, and
# p3_promote_candidate's coefficient-space box; the from-scratch path above
# -- build_seed_population, evaluate_population_cheap, run() -- is untouched).
# ---------------------------------------------------------------------------
POP_SIZE_SEEDED_DEFAULT = 48
INIT_SCALE_DEFAULT = 0.02
SEEDED_TOTAL_HOURS_DEFAULT = 24.0
SEEDED_MAX_EVALS_DEFAULT = 3000
V_CURVE_EVERY = 100  # REPORT.json's v_curve: sampled every this many evaluations


def polynomial_mutate(
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    rng: np.random.RandomState,
    eta: float = 20.0,
) -> np.ndarray:
    """One real-coded polynomial-mutation sample of `x` within [`lo`, `hi`]
    (Deb & Agrawal 1999; pymoo's own default distribution index eta=20) --
    the standard formula pymoo's PolynomialMutation operator itself applies,
    reimplemented here (not called via pymoo) purely so E17's seeding stays
    host-importable and unit-testable without the pinned image, matching
    this file's own `build_seed_population`, which already hand-rolls its
    seed perturbation (Gaussian there) rather than using a pymoo operator.
    Elementwise; `lo` must be < `hi` in every coordinate."""
    x, lo, hi = (np.asarray(v, float) for v in (x, lo, hi))
    span = hi - lo
    u = rng.uniform(0.0, 1.0, size=x.shape)
    delta1 = (x - lo) / span
    delta2 = (hi - x) / span
    mut_pow = 1.0 / (eta + 1.0)
    lower = u <= 0.5
    xy_lo = 1.0 - delta1
    val_lo = 2.0 * u + (1.0 - 2.0 * u) * xy_lo ** (eta + 1.0)
    deltaq_lo = val_lo**mut_pow - 1.0
    xy_hi = 1.0 - delta2
    val_hi = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy_hi ** (eta + 1.0)
    deltaq_hi = 1.0 - val_hi**mut_pow
    deltaq = np.where(lower, deltaq_lo, deltaq_hi)
    return np.clip(x + deltaq * span, lo, hi)


def old_mask_from_anchor_xs(anchor_xs: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """True for a coordinate any anchor occupies with a nonzero value ('old'
    modes, in `p3_promote_candidate`'s sense: the modes the front's own
    designs actually populate); False where every anchor is exactly zero
    (padding from a mode-budget promotion, 'new'). Pure/data-derived -- needs
    no declared reference budget, only the anchors' own coordinates."""
    anchor_xs = np.asarray(anchor_xs, float)
    return np.any(np.abs(anchor_xs) > atol, axis=0)


def init_scale_box(scale: np.ndarray, old_mask: np.ndarray, init_scale: float) -> np.ndarray:
    """Per-coordinate seeding half-width: `init_scale` (the front builder's
    own probe/step units, TOL-adjacent) on every 'old' mode, widened on
    'new' modes by `p3_promote_candidate.coefficient_space_multiplier` so a
    zero-initialised, tiny-natural-scale mode gets a comparable PHYSICAL
    coefficient reach to an old mode instead of an arbitrarily tiny one.
    Degenerate case (no 'old' mode at all -- e.g. every anchor is exactly
    zero, so there is no reference scale to widen new modes against) falls
    back to isotropic `init_scale` everywhere rather than dividing by an
    empty median."""
    scale = np.asarray(scale, float)
    old_mask = np.asarray(old_mask, bool)
    if not old_mask.any():
        return float(init_scale) * np.ones_like(scale)
    return float(init_scale) * coefficient_space_multiplier(scale, old_mask)


def seed_population_from_anchors(
    anchor_xs: np.ndarray,
    pop_size: int,
    box: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generation 0 for the seeded NSGA-II run: every anchor verbatim (as
    many as fit), then polynomial-mutated anchors -- a uniformly-chosen
    anchor perturbed within its own [x-box, x+box] (clipped to the run's
    global [xl, xu]) -- until `pop_size` is reached. Raises if `pop_size` is
    smaller than the anchor set (nothing to drop safely -- fail loud)."""
    anchor_xs = np.asarray(anchor_xs, float)
    n_anchors, _dim = anchor_xs.shape
    if pop_size < n_anchors:
        raise ValueError(f"pop_size ({pop_size}) must be >= n_anchors ({n_anchors})")
    box = np.asarray(box, float)
    xl = np.asarray(xl, float)
    xu = np.asarray(xu, float)
    population = [anchor_xs[i] for i in range(n_anchors)]
    while len(population) < pop_size:
        base = anchor_xs[rng.randint(n_anchors)]
        lo = np.maximum(base - box, xl)
        hi = np.minimum(base + box, xu)
        population.append(polynomial_mutate(base, lo, hi, rng))
    return np.stack(population[:pop_size])


def load_seed_candidate(
    path: str, checkpoint_path: str | None, coeff_space_box: bool
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray | None]:
    """(anchors, representation, old_mask) for `--seed-candidate` (E18 item
    2): a single-element anchors list, so the existing
    `seed_population_from_anchors` pipeline seeds a population of this one
    point unchanged -- n_anchors=1 there already IS 'the seed verbatim plus
    pop_size - 1 polynomial mutations', no separate population-construction
    function needed.

    `representation` is built by `p3_final_restoration.representation_of`,
    reused rather than re-derived, exactly like `--start-candidate` does for
    every candidate this campaign has saved since E6 (all of which carry
    their own boundary_dict/mpol/ntor/nfp). `checkpoint_path` is the
    fallback representation source for a candidate that does not (refused
    if neither is available).

    `old_mask` is None unless `coeff_space_box`, in which case it is the
    E6b/E7 STRUCTURAL mask (`p3_promote_candidate.old_mode_mask`, keyed on
    the candidate's own 'source' budget) rather than the anchors pipeline's
    zero-value heuristic (`old_mask_from_anchor_xs`) -- needed because a
    fully-optimised candidate (like the 176-lineage 8x8 incumbent E18a
    seeds from) no longer has any exactly-zero 'new' mode for that
    heuristic to key on; refused if the candidate carries no 'source'."""
    cand = json.loads(Path(path).read_text())
    if "boundary_dict" in cand:
        representation = representation_of(cand)
    elif checkpoint_path:
        ck = json.loads(Path(checkpoint_path).read_text())
        representation = dict(
            boundary_dict=ck["boundary_dict"],
            mpol=int(ck["max_poloidal_mode"]),
            ntor=int(ck["max_toroidal_mode"]),
            nfp=int(ck["n_field_periods"]),
            scaling=float(ck.get("infinity_norm_spectrum_scaling", 1.0)),
        )
    else:
        raise SystemExit(
            "--checkpoint is required: the --seed-candidate JSON has no "
            "boundary_dict of its own to build a representation from"
        )
    anchor = {
        "tag": f"seed:{Path(path).name}",
        "r": [float(v) for v in cand["r"]],
        "aspect": float(cand["raw"]["aspect_ratio"]),
        "min_len": float(cand["raw"]["min_len"]),
        "x": [float(v) for v in cand["x"]],
    }
    old_mask = None
    if coeff_space_box:
        src_budget = cand.get("source")
        if not src_budget:
            raise SystemExit(
                "--coeff-space-box needs a --seed-candidate with a 'source' "
                "budget (the mode budget it was promoted from)"
            )
        old_mask, _scale = old_mode_mask(cand, int(src_budget["mpol"]), int(src_budget["ntor"]))
    return [anchor], representation, old_mask


def _representation_scale_and_unravel(representation: dict[str, Any]):
    """(scale, unravel_and_unmask_fn) for an ARBITRARY representation
    (boundary_dict, mpol, ntor, nfp, scaling) -- the general form of
    `boundary_dict_to_scaled_x`, which fixes those at this file's own 8x8
    from-scratch budget, parametrised instead by whatever `--representation`
    supplies (E13's 7x7 lineage, or any other budget)."""
    from constellaration_methods.engines.alm_ngopt_optimizer import (
        masked_coefficients_and_scale,
    )

    _boundary, _mask, _coeffs, unravel_and_unmask_fn, scale = masked_coefficients_and_scale(
        representation["boundary_dict"],
        max_poloidal_mode=int(representation["mpol"]),
        max_toroidal_mode=int(representation["ntor"]),
        n_field_periods=int(representation["nfp"]),
        infinity_norm_spectrum_scaling=float(representation.get("scaling", 1.0)),
    )
    return np.asarray(scale, float), unravel_and_unmask_fn


def _scale_for_representation(representation: dict[str, Any], dim: int) -> np.ndarray:
    """The representation's own per-coordinate scale array for
    `init_scale_box`, or an isotropic `np.ones(dim)` for a fake/toy
    representation (no real Fourier structure to derive one from -- mirrors
    `p3_front_builder._export_front`'s fake short-circuit)."""
    if _is_fake(representation):
        return np.ones(dim, dtype=float)
    scale, _unravel = _representation_scale_and_unravel(representation)
    return scale


def objectives_and_constraints_from_raws(
    raws: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """(F, G) for pymoo from a batch of canonical archive rows, in the exact
    shape `CanonicalEvaluator.evaluate`/`append_jsonl` already write them:
    F = [-min_len, aspect_ratio] (both minimised -- min_len maximised,
    aspect_ratio minimised, the same convention `p3_front_builder` uses); G
    = the five canonical residuals directly (already <=0-feasible, pymoo's
    own convention -- no re-derivation needed). A row with no 'raw'/'r'
    (non-convergent or errored) gets the existing large finite penalty,
    never nan/inf."""
    n = len(raws)
    F = np.zeros((n, 2))
    G = np.zeros((n, 5))
    for i, row in enumerate(raws):
        raw = row.get("raw")
        r = row.get("r")
        if raw is None or r is None:
            F[i] = [NON_CONVERGENT_OBJECTIVE_PENALTY, NON_CONVERGENT_OBJECTIVE_PENALTY]
            G[i] = [NON_CONVERGENT_CONSTRAINT_PENALTY] * 5
            continue
        F[i, 0] = -float(raw["min_len"])
        F[i, 1] = float(raw["aspect_ratio"])
        G[i] = [float(v) for v in r]
    return F, G


def evaluate_population_cheap_repr(
    X: np.ndarray,
    scale: np.ndarray,
    unravel_and_unmask_fn: Any,
    settings: Any,
    executor: futures.ProcessPoolExecutor,
    tags: list[str],
    archive_path: Path,
    constraint_overrides: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cheap-fidelity batch evaluation at an ARBITRARY representation --
    generalises `evaluate_population_cheap` (which fixes this file's own
    8x8 from-scratch space) to whatever budget `--representation` supplies,
    reusing the SAME worker (`_eval_candidate_mhd`). Every candidate is also
    logged to `archive_path` in the same row schema `CanonicalEvaluator`
    writes, so a seeded cheap-fidelity run's archive.jsonl is still
    `p3_front_from_archive`-compatible even though its residuals are cheap,
    not canonical, fidelity."""
    futs = [
        executor.submit(
            _eval_candidate_mhd,
            x,
            scale,
            unravel_and_unmask_fn,
            settings,
            constraint_overrides,
        )
        for x in X
    ]
    results = [f.result() for f in futs]
    F = np.zeros((len(X), 2))
    G = np.zeros((len(X), 5))
    for i, (x, r, tag) in enumerate(zip(X, results, tags)):
        if not r["convergent"]:
            F[i] = [NON_CONVERGENT_OBJECTIVE_PENALTY, NON_CONVERGENT_OBJECTIVE_PENALTY]
            G[i] = [NON_CONVERGENT_CONSTRAINT_PENALTY] * 5
            append_jsonl(
                archive_path,
                {
                    "tag": tag,
                    "x": [float(v) for v in x],
                    "non_convergent": True,
                    "elapsed_s": 0.0,
                },
            )
            continue
        min_grad_len, aspect_ratio = r["objectives"]
        F[i, 0] = -min_grad_len if not r["minimize_flags"][0] else min_grad_len
        F[i, 1] = aspect_ratio if r["minimize_flags"][1] else -aspect_ratio
        G[i] = r["constraints"]
        append_jsonl(
            archive_path,
            {
                "tag": tag,
                "x": [float(v) for v in x],
                "r": [float(v) for v in r["constraints"]],
                "raw": {
                    "aspect_ratio": float(aspect_ratio),
                    "min_len": float(min_grad_len),
                },
                "feasible": bool(max(r["constraints"]) <= TOL),
                "elapsed_s": 0.0,
            },
        )
    return F, G


def resolve_eval_fidelity(args: argparse.Namespace) -> str:
    """The effective --eval-fidelity: whatever was given explicitly; else
    'canonical' when seeding from --anchors or --seed-candidate (the front
    builder's own fidelity); else 'cheap' (the original from-scratch
    behaviour, unchanged by default)."""
    if args.eval_fidelity is not None:
        return args.eval_fidelity
    return "canonical" if (args.anchors or args.seed_candidate) else "cheap"


def validate_seeded_args(args: argparse.Namespace) -> None:
    """--anchors and --representation are a pair: fail loud if only one is
    given rather than silently falling back to the from-scratch path.
    Exactly one seeding source is required in seeded mode: --anchors or
    --seed-candidate (E18 item 2) -- fail loud if both, or neither, are
    given, rather than silently preferring one."""
    if bool(args.anchors) != bool(args.representation):
        raise SystemExit("--anchors and --representation must be given together")
    if bool(args.anchors) == bool(args.seed_candidate):
        raise SystemExit("exactly one of --anchors or --seed-candidate is required in seeded mode")


def apply_smoke_ceilings_seeded(args: argparse.Namespace) -> None:
    """--smoke forces the seeded runner's own toy ceilings (pop 4, one
    generation, 20 evaluations, 0.05h) regardless of any --pop-size/
    --max-generations/--max-evals/--total-hours also given -- mirrors
    `p3_front_builder.apply_smoke_ceilings`."""
    if getattr(args, "smoke", False):
        args.pop_size = 4
        args.max_generations = 1
        args.max_evals = 20
        args.total_hours = 0.05


def resolve_out_dir_seeded(args: argparse.Namespace) -> str:
    """args.out_dir if given; else the smoke default; else a hard error
    (--out-dir is required for a real seeded run) -- mirrors
    `p3_front_builder.resolve_out_dir`."""
    if args.out_dir is not None:
        return args.out_dir
    if args.smoke:
        return "results/b3-nsga2-e17-smoke"
    raise SystemExit("--out-dir is required unless --smoke")


def seeded_stop_reason(
    wall_elapsed_s: float,
    total_hours: float,
    generation: int,
    max_generations: int | None,
    evals_used: int,
    max_evals: int,
    pop_size: int,
) -> str | None:
    """The seeded runner's own ceiling check, evaluated once per generation
    BEFORE asking pymoo for the next population -- a generation's whole
    batch is charged atomically or not started at all, never partially."""
    if wall_elapsed_s >= total_hours * 3600:
        return "wall_cap"
    if max_generations is not None and generation >= max_generations:
        return "max_generations"
    if evals_used + pop_size > max_evals:
        return "max_evals"
    return None


def _make_seeded_evaluator(
    args: argparse.Namespace,
    representation: dict,
    archive_path: Path,
    budget: Budget,
    constraint_overrides: dict[str, float] | None = None,
):
    """(evaluate_batch, pool): the fidelity-specific batch evaluator for the
    seeded runner's ask/tell loop, and the process pool its lifetime is
    scoped to (a context manager; the caller opens it with `with`). Both
    fidelities log every evaluation to `archive_path` in the SAME row schema
    (tag, x, r, raw, feasible, elapsed_s), so `p3_front_from_archive.py
    --confirm` reads either kind of run unchanged. `constraint_overrides`
    (E18 item 1) reaches the canonical fidelity via `representation` (merged
    into every evaluation payload by `CanonicalEvaluator`) and the cheap
    fidelity via a direct parameter, since its worker rebuilds the problem
    per-candidate in a separate process."""
    import multiprocessing as mp

    workers = args.workers or len(os.sched_getaffinity(0))

    if args.eval_fidelity == "canonical":
        ctx = mp.get_context("spawn")
        pool = futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        rep_with_overrides = dict(representation, constraint_overrides=constraint_overrides)
        evaluator = CanonicalEvaluator(rep_with_overrides, pool, archive_path, budget)

        def evaluate_batch(X: np.ndarray, tags: list[str]) -> tuple[np.ndarray, np.ndarray]:
            _rs, raws = evaluator.evaluate(X, tags)
            return objectives_and_constraints_from_raws(raws)

        return evaluate_batch, pool

    if args.eval_fidelity == "cheap":
        from constellaration_methods.engines.alm_ngopt_optimizer import (
            _search_settings_for,
        )

        scale, unravel_and_unmask_fn = _representation_scale_and_unravel(representation)
        settings = _search_settings_for(problem_with_overrides(constraint_overrides), None)
        ctx = mp.get_context("forkserver")
        ctx.set_forkserver_preload([])
        pool = futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=thread_limits.pin_worker,
            initargs=(workers,),
        )

        def evaluate_batch(X: np.ndarray, tags: list[str]) -> tuple[np.ndarray, np.ndarray]:
            budget.charge(len(X))
            return evaluate_population_cheap_repr(
                X,
                scale,
                unravel_and_unmask_fn,
                settings,
                pool,
                tags,
                archive_path,
                constraint_overrides,
            )

        return evaluate_batch, pool

    raise SystemExit(f"unknown --eval-fidelity {args.eval_fidelity!r}")


def run_seeded_from_anchors(
    anchors: list[dict[str, Any]],
    representation: dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
    old_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """The seeded NSGA-II engine: pure enough to run on a hand-built
    `anchors`/`representation` (a `fake_front`/toy representation needs no
    VMEC) -- the E17 counterpart of `p3_front_builder.run_from_anchors`.
    Always writes REPORT.json, including on a KeyboardInterrupt.

    `old_mask`, when given, overrides the anchors pipeline's own zero-value
    heuristic (`old_mask_from_anchor_xs`) for the seeding box -- used by
    `--seed-candidate --coeff-space-box` (E18 item 2), whose single,
    already-optimised seed has no zero coordinate left to distinguish old
    modes by value."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / "archive.jsonl"
    events_path = out_dir / "events.jsonl"
    t0 = time.time()
    constraint_overrides = parse_constraint_overrides(getattr(args, "constraint_override", None))

    def log(**kw: Any) -> None:
        kw["t_h"] = round((time.time() - t0) / 3600, 4)
        append_jsonl(events_path, kw)

    (out_dir / "start_serialised.json").write_text(json.dumps(representation, indent=2))

    anchor_xs = np.array([a["x"] for a in anchors], dtype=float)
    dim = anchor_xs.shape[1]
    pop_size = args.pop_size
    if pop_size < len(anchors):
        raise SystemExit(
            f"--pop-size {pop_size} is smaller than the anchor count "
            f"({len(anchors)}); minimum --pop-size is {len(anchors)}"
        )

    scale = _scale_for_representation(representation, dim)
    if old_mask is None:
        old_mask = old_mask_from_anchor_xs(anchor_xs)
    box = init_scale_box(scale, old_mask, args.init_scale)
    xl, xu = compute_bounds(anchor_xs)
    rng = np.random.RandomState(args.seed)
    seed_population = seed_population_from_anchors(anchor_xs, pop_size, box, xl, xu, rng)

    anchor_points = [
        {"tag": a["tag"], "aspect_ratio": a["aspect"], "min_len": a["min_len"]} for a in anchors
    ]
    initial_hv = hypervolume(nondominated(anchor_points))
    log(
        event="seed",
        n_anchors=len(anchors),
        pop_size=pop_size,
        dim=dim,
        initial_hypervolume=initial_hv,
    )

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as PymooProblem

    class SeededPymooProblem(PymooProblem):
        def __init__(self) -> None:
            super().__init__(n_var=dim, n_obj=2, n_ieq_constr=5, xl=xl, xu=xu)

        def _evaluate(self, X: np.ndarray, out: dict[str, Any], *_a: Any, **_k: Any) -> None:
            raise RuntimeError(
                "SeededPymooProblem._evaluate should never be called directly -- "
                "evaluation is driven externally so every batch can be banked."
            )

    algo = NSGA2(pop_size=pop_size, sampling=seed_population)
    algo.setup(
        SeededPymooProblem(),
        termination=("n_gen", 10**9),
        seed=args.seed,
        verbose=False,
    )

    budget = Budget(cap=args.max_evals)
    generation = 0
    stop_reason = None
    wall_start = time.time()

    evaluate_batch, pool = _make_seeded_evaluator(
        args, representation, archive_path, budget, constraint_overrides
    )
    try:
        with pool:
            while True:
                wall_elapsed = time.time() - wall_start
                stop_reason = seeded_stop_reason(
                    wall_elapsed,
                    args.total_hours,
                    generation,
                    args.max_generations,
                    budget.used,
                    args.max_evals,
                    pop_size,
                )
                if stop_reason is not None:
                    break
                pop = algo.ask()
                X = pop.get("X")
                tags = [f"G{generation}:{i}" for i in range(len(X))]
                F, G = evaluate_batch(X, tags)
                pop.set("F", F)
                pop.set("G", G)
                algo.tell(infills=pop)
                generation += 1

                rows = load_jsonl(archive_path)
                feas = tolerance_feasible_rows(rows, tol=TOL)
                points = [dict(row["raw"], tag=row["tag"]) for row in feas if row.get("raw")]
                current_hv = hypervolume(nondominated(points)) if points else 0.0
                log(
                    event="generation",
                    generation=generation,
                    evaluations=budget.used,
                    hypervolume=current_hv,
                    best_v_so_far=best_v_of_rows(rows),
                    n_feasible_so_far=len(feas),
                    elapsed_h=wall_elapsed / 3600,
                )
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"

    rows = load_jsonl(archive_path)
    feas = tolerance_feasible_rows(rows, tol=TOL)
    points = [dict(row["raw"], tag=row["tag"]) for row in feas if row.get("raw")]
    final_hv = hypervolume(nondominated(points)) if points else 0.0

    report = {
        "seeded": True,
        "smoke": bool(args.smoke),
        "eval_fidelity": args.eval_fidelity,
        "n_anchors": len(anchors),
        "pop_size": pop_size,
        "dim": dim,
        "initial_hypervolume": initial_hv,
        "final_hypervolume": final_hv,
        "generations": generation,
        "evaluations": budget.used,
        "stop_reason": stop_reason,
        "constraint_overrides": constraint_overrides,
        "v_curve": v_curve_from_rows(rows, every=V_CURVE_EVERY),
        "ceilings": {
            "total_hours": args.total_hours,
            "max_evals": args.max_evals,
            "max_generations": args.max_generations,
            "pop_size": pop_size,
        },
    }
    (out_dir / "REPORT.json").write_text(json.dumps(report, indent=2))
    print(
        "\n=== E17 SEEDED NSGA-II REPORT ===\n"
        + json.dumps(
            {k: v for k, v in report.items() if k not in ("ceilings", "v_curve")},
            indent=2,
        )
        + "\n"
        + json.dumps(report["ceilings"], indent=2),
        flush=True,
    )
    return report


def run_seeded(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed_candidate:
        anchors, representation, old_mask = load_seed_candidate(
            args.seed_candidate, args.checkpoint, args.coeff_space_box
        )
        return run_seeded_from_anchors(
            anchors, representation, Path(args.out_dir), args, old_mask=old_mask
        )
    anchors, representation = load_anchors(args.anchors, args.representation)
    return run_seeded_from_anchors(anchors, representation, Path(args.out_dir), args)


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as PymooProblem

    constraint_overrides = parse_constraint_overrides(getattr(args, "constraint_override", None))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state_path = OUT_DIR / "state.json"
    generations_log = OUT_DIR / "generations.jsonl"
    real_eval_log = OUT_DIR / "real_evaluate_log.jsonl"

    pop_size = 12 if args.smoke else POP_SIZE_REAL
    max_generations = 3 if args.smoke else args.max_generations
    num_workers = 2 if args.smoke else (args.num_workers or len(os.sched_getaffinity(0)))
    mid_run_generation = None if args.smoke else args.mid_run_generation
    real_eval_cap = 0 if args.smoke else MAX_REAL_EVALUATE_CALLS

    seed_population = build_seed_population(pop_size, args.seed)
    xl, xu = compute_bounds(seed_population)
    dim = seed_population.shape[1]

    # scale/unravel come from ANY seed's own conversion (structure-only --
    # the mode budget/mask is identical for every individual at 8x8).
    _x0, scale, unravel_and_unmask_fn = boundary_dict_to_scaled_x(
        json.loads(
            (
                REPO_ROOT / "results" / "submission" / "best_feasible_simple_to_build.json"
            ).read_text()
        )
    )

    from constellaration_methods.engines.alm_ngopt_optimizer import (
        _search_settings_for,
    )

    mhd_problem = problem_with_overrides(constraint_overrides)
    search_settings = _search_settings_for(mhd_problem, None)

    class MHDPymooProblem(PymooProblem):
        def __init__(self) -> None:
            super().__init__(n_var=dim, n_obj=2, n_ieq_constr=5, xl=xl, xu=xu)

        def _evaluate(self, X: np.ndarray, out: dict[str, Any], *_a: Any, **_k: Any) -> None:
            raise RuntimeError(
                "MHDPymooProblem._evaluate should never be called directly -- "
                "evaluation is driven externally via evaluate_population_cheap "
                "so results can be banked per-generation."
            )

    pymoo_problem = MHDPymooProblem()
    algo = NSGA2(pop_size=pop_size, sampling=seed_population)
    algo.setup(pymoo_problem, termination=("n_gen", 10**9), seed=args.seed, verbose=False)

    generation = 0
    n_real_evaluate_calls = 0
    wall_start = time.perf_counter()
    stop_reason = None

    import multiprocessing

    mp_context = multiprocessing.get_context("forkserver")
    mp_context.set_forkserver_preload([])
    thread_limits.ensure_on_pythonpath(thread_limits.src_dir())

    print(
        f"[config] smoke={args.smoke} pop_size={pop_size} num_workers={num_workers} "
        f"max_generations={max_generations} dim={dim} real_eval_cap={real_eval_cap}",
        flush=True,
    )

    with futures.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp_context,
        initializer=thread_limits.pin_worker,
        initargs=(num_workers,),
    ) as executor:
        while True:
            wall_elapsed = time.perf_counter() - wall_start
            if wall_elapsed >= MAX_WALL_SECONDS:
                stop_reason = "wall_cap"
                break
            if max_generations is not None and generation >= max_generations:
                stop_reason = "max_generations"
                break

            gen_t0 = time.perf_counter()
            pop = algo.ask()
            X = pop.get("X")
            F, G, results = evaluate_population_cheap(
                X,
                scale,
                unravel_and_unmask_fn,
                search_settings,
                executor,
                constraint_overrides,
            )
            pop.set("F", F)
            pop.set("G", G)
            algo.tell(infills=pop)
            generation += 1
            gen_elapsed = time.perf_counter() - gen_t0

            n_feasible = sum(
                1 for r in results if r["convergent"] and all(c <= 0 for c in r["constraints"])
            )
            proxy_hv = cheap_proxy_hypervolume(results)
            print(
                f"[gen {generation}] elapsed={gen_elapsed:.1f}s n_feasible={n_feasible}"
                f"/{pop_size} cheap_proxy_hv={proxy_hv:.4f}",
                flush=True,
            )
            append_jsonl(
                generations_log,
                {
                    "generation": generation,
                    "elapsed_s": gen_elapsed,
                    "n_feasible": n_feasible,
                    "cheap_proxy_hypervolume": proxy_hv,
                    "X": X.tolist(),
                    "F": F.tolist(),
                    "G": G.tolist(),
                },
            )

            if (
                real_eval_cap > 0
                and n_real_evaluate_calls < real_eval_cap
                and mid_run_generation is not None
                and generation == mid_run_generation
            ):
                print("[real-evaluate] mid-run call...", flush=True)
                verdict = real_evaluate_population(
                    X, scale, unravel_and_unmask_fn, constraint_overrides
                )
                n_real_evaluate_calls += 1
                verdict["generation"] = generation
                verdict["call_label"] = "mid_run"
                append_jsonl(real_eval_log, verdict)
                print(f"[real-evaluate] {verdict}", flush=True)

            save_state(
                state_path,
                {
                    "generation": generation,
                    "wall_elapsed_seconds": time.perf_counter() - wall_start,
                    "n_real_evaluate_calls": n_real_evaluate_calls,
                    "n_feasible_this_gen": n_feasible,
                    "cheap_proxy_hypervolume": proxy_hv,
                    "stop_reason": stop_reason,
                },
            )

        # Final real evaluate() call on the last population, if the cap allows.
        if real_eval_cap > 0 and n_real_evaluate_calls < real_eval_cap:
            print("[real-evaluate] final call...", flush=True)
            final_X = algo.pop.get("X")
            verdict = real_evaluate_population(
                final_X, scale, unravel_and_unmask_fn, constraint_overrides
            )
            n_real_evaluate_calls += 1
            verdict["generation"] = generation
            verdict["call_label"] = "final"
            append_jsonl(real_eval_log, verdict)
            print(f"[real-evaluate] {verdict}", flush=True)

    # Re-save state with the ACTUAL stop reason and final real-evaluate
    # count: the per-generation save above runs before the loop's
    # next-iteration cap check sets `stop_reason` (and before any final
    # real-evaluate call above), so without this the persisted state.json
    # would understate both -- misleading on inspection.
    save_state(
        state_path,
        {
            "generation": generation,
            "wall_elapsed_seconds": time.perf_counter() - wall_start,
            "n_real_evaluate_calls": n_real_evaluate_calls,
            "stop_reason": stop_reason,
        },
    )

    print(
        f"[done] stop_reason={stop_reason} generation={generation} "
        f"n_real_evaluate_calls={n_real_evaluate_calls}",
        flush=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-generations", type=int, default=None)
    parser.add_argument("--mid-run-generation", type=int, default=DEFAULT_MID_RUN_GENERATION)
    parser.add_argument("--seed", type=int, default=1)
    # E17 -- seeding an initial population from an existing confirmed front.
    parser.add_argument("--anchors", default=None, help="a front REPORT.json (E15/E16 format)")
    parser.add_argument("--representation", default=None, help="an E13-style start_serialised.json")
    parser.add_argument(
        "--eval-fidelity",
        choices=["cheap", "canonical"],
        default=None,
        help="default: canonical when --anchors is given, else cheap (unchanged)",
    )
    parser.add_argument("--init-scale", type=float, default=INIT_SCALE_DEFAULT)
    parser.add_argument(
        "--pop-size",
        type=int,
        default=POP_SIZE_SEEDED_DEFAULT,
        help=(
            "NSGA-II population size; in seeded mode (--anchors) must be >= "
            "the anchor count, since every anchor seeds one population "
            "member -- refused otherwise"
        ),
    )
    parser.add_argument("--total-hours", type=float, default=SEEDED_TOTAL_HOURS_DEFAULT)
    parser.add_argument("--max-evals", type=int, default=SEEDED_MAX_EVALS_DEFAULT)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="canonical-fidelity evaluator pool size",
    )
    # E18 item 2 -- a single-candidate seed, as an alternative to --anchors.
    parser.add_argument(
        "--seed-candidate",
        default=None,
        help="a single candidate JSON (x/r/raw/representation), as an "
        "alternative to --anchors: the population is this one seed plus "
        "(pop_size - 1) polynomial mutations at --init-scale",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="--seed-candidate only: fallback representation source (a "
        "checkpoint JSON) for a candidate with no boundary_dict of its own",
    )
    parser.add_argument(
        "--coeff-space-box",
        action="store_true",
        help="--seed-candidate only (E6b/E7): seed the mutation box from "
        "the candidate's own structural old/new mode mask "
        "(p3_promote_candidate.old_mode_mask, keyed on its 'source' "
        "budget) instead of the anchors pipeline's zero-value heuristic",
    )
    # E18 item 1 -- override a constraint threshold for this run.
    parser.add_argument(
        "--constraint-override",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help="override a MHDStableQIStellarator constraint threshold "
        "(repeatable); applies to every evaluation site in this file. "
        "Valid names: "
        "constellaration_methods.problems.overrides.CONSTRAINT_OVERRIDE_NAMES",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.anchors or args.representation or args.seed_candidate:
        validate_seeded_args(args)
        args.eval_fidelity = resolve_eval_fidelity(args)
        apply_smoke_ceilings_seeded(args)
        args.out_dir = resolve_out_dir_seeded(args)
        run_seeded(args)
        return
    args.eval_fidelity = resolve_eval_fidelity(args)
    run(args)


if __name__ == "__main__":
    main()
