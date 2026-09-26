"""Promote a saved P3 candidate (x in scaled coordinates at one mode budget) to a
larger mode budget with the new Fourier modes at zero (E6, vault
p3-reopen-three-experiments-2026-09-16). The physical shape is unchanged; the
repeatability gate of the restoration runner verifies that canonically.

    python3 scripts/p3_promote_candidate.py IN.json OUT.json --mpol 8 --ntor 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pad_fourier(a, mpol_from: int, ntor_from: int, mpol_to: int, ntor_to: int):
    """Embed an (mpol_from+1) x (2 ntor_from+1) coefficient array into the
    (mpol_to+1) x (2 ntor_to+1) array at the same (m, n) positions, zeros elsewhere."""
    a = np.asarray(a, float)
    assert a.shape == (mpol_from + 1, 2 * ntor_from + 1), a.shape
    out = np.zeros((mpol_to + 1, 2 * ntor_to + 1))
    off = ntor_to - ntor_from
    out[: mpol_from + 1, off : off + 2 * ntor_from + 1] = a
    return out


def coefficient_space_multiplier(scale, old_mask):
    """Per-coordinate trust-radius multiplier. Old (source-budget) modes keep
    EXACTLY the isotropic box (mult 1: E6's control condition, so any
    difference from E6 is attributable to the new modes); each new mode gets
    the median old mode's reach in COEFFICIENT units, mult_j =
    median(scale[old]) / scale_j (~55x at 8x8; E6b, vault
    p3-reopen-three-experiments-2026-09-16)."""
    scale = np.asarray(scale, float)
    old_mask = np.asarray(old_mask, bool)
    s_ref = float(np.median(scale[old_mask]))
    return np.where(old_mask, 1.0, s_ref / scale)


def new_mode_energy_fraction(d, scale, old_mask) -> float:
    """Share of a step's displacement energy, in coefficient units
    (d_j * scale_j)^2, carried by the modes outside the source budget.
    E6's diagnosis was ~0.001 of the energy in new modes; E6b's inconclusive
    bar is < 0.05 (vault pre-registration)."""
    e = (np.asarray(d, float) * np.asarray(scale, float)) ** 2
    tot = float(e.sum())
    if tot == 0.0:
        return 0.0
    return float(e[~np.asarray(old_mask, bool)].sum() / tot)


def candidate_from_boundary_dict(bd: dict, r, raw=None, note: str = "") -> dict:
    """A restoration start candidate from a raw boundary (E11: an external
    design with no prior run): x = the engine's scaled coordinates at the
    boundary's own mode budget (scaling 1.0), r = its canonical residuals."""
    from constellaration_methods.engines.alm_ngopt_optimizer import (
        masked_coefficients_and_scale,
    )

    r_cos = np.asarray(bd["r_cos"], float)
    mpol, ntor = r_cos.shape[0] - 1, (r_cos.shape[1] - 1) // 2
    nfp = int(bd["n_field_periods"])
    _, _, coeffs, _, scale = masked_coefficients_and_scale(
        bd,
        max_poloidal_mode=mpol,
        max_toroidal_mode=ntor,
        n_field_periods=nfp,
        infinity_norm_spectrum_scaling=1.0,
    )
    x = np.asarray(coeffs, float) / np.asarray(scale, float)
    return {
        "note": note,
        "x": x.tolist(),
        "r": [float(v) for v in r],
        "raw": raw,
        "boundary_dict": bd,
        "mpol": mpol,
        "ntor": ntor,
        "nfp": nfp,
        "scaling": 1.0,
    }


def boundary_dict_of_candidate(cand: dict) -> dict:
    """The leaderboard-format boundary (r_cos, z_sin, r_sin, z_cos,
    n_field_periods, is_stellarator_symmetric) a candidate's scaled
    coordinates describe, via the engine's own unravel (close-out export)."""
    import jax.numpy as jnp

    from constellaration_methods.engines.alm_ngopt_optimizer import (
        masked_coefficients_and_scale,
    )

    _, _, _, unravel, scale = masked_coefficients_and_scale(
        cand["boundary_dict"],
        max_poloidal_mode=int(cand["mpol"]),
        max_toroidal_mode=int(cand["ntor"]),
        n_field_periods=int(cand["nfp"]),
        infinity_norm_spectrum_scaling=float(cand.get("scaling", 1.0)),
    )
    surf = unravel(jnp.asarray(cand["x"]) * jnp.asarray(scale))
    r_cos = np.asarray(surf.r_cos, float)
    z_sin = np.asarray(surf.z_sin, float)
    return {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        # the leaderboard loader requires these to be null for a symmetric surface
        "r_sin": None,
        "z_cos": None,
        "n_field_periods": int(cand["nfp"]),
        "is_stellarator_symmetric": True,
    }


def reflect_toroidally(bd: dict) -> dict:
    """phi -> -phi: the mirror image of a stellarator-symmetric boundary.
    Rotational transform changes sign; every other benchmark metric is
    unchanged (verified to 1e-5 in both library versions, results/
    p3-close-out). For m > 0 the toroidal index n maps to -n; for m = 0
    cos(n phi) is even (row unchanged) and sin(n phi) is odd (row negated),
    so the convention "m = 0 has no n < 0 entries" is preserved. The
    midplane reflection z -> -z is NOT a symmetry of the benchmark's flux
    metric; use this one."""
    r_cos = np.asarray(bd["r_cos"], float)
    z_sin = np.asarray(bd["z_sin"], float)
    r_out, z_out = r_cos.copy(), z_sin.copy()
    r_out[1:, :] = r_cos[1:, ::-1]
    z_out[1:, :] = z_sin[1:, ::-1]
    z_out[0, :] = -z_sin[0, :]
    return dict(bd, r_cos=r_out.tolist(), z_sin=z_out.tolist())


def old_mode_mask(cand: dict, mpol_old: int, ntor_old: int):
    """(old_mask, scale) for a candidate's scaled coordinates: True where the
    coordinate is a Fourier mode inside the old (mpol_old, ntor_old) budget.
    Found by probing the engine's unravel with unit vectors (no physics), so
    it cannot drift from the engine's own coordinate ordering."""
    import jax.numpy as jnp

    from constellaration_methods.engines.alm_ngopt_optimizer import (
        masked_coefficients_and_scale,
    )

    _, _, coeffs, unravel, scale = masked_coefficients_and_scale(
        cand["boundary_dict"],
        max_poloidal_mode=int(cand["mpol"]),
        max_toroidal_mode=int(cand["ntor"]),
        n_field_periods=int(cand["nfp"]),
        infinity_norm_spectrum_scaling=float(cand.get("scaling", 1.0)),
    )
    n = len(np.asarray(coeffs))
    base = unravel(jnp.zeros(n))
    ntor = int(cand["ntor"])
    old = np.zeros(n, bool)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        sj = unravel(jnp.asarray(e))
        dr = np.abs(np.asarray(sj.r_cos) - np.asarray(base.r_cos))
        dz = np.abs(np.asarray(sj.z_sin) - np.asarray(base.z_sin))
        arr = dr if dr.max() > dz.max() else dz
        m, col = np.unravel_index(int(arr.argmax()), arr.shape)
        old[j] = (int(m) <= mpol_old) and (abs(int(col) - ntor) <= ntor_old)
    return old, np.asarray(scale, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mpol", type=int, default=8)
    ap.add_argument("--ntor", type=int, default=8)
    ap.add_argument(
        "--to-boundary",
        action="store_true",
        help="write src's boundary (leaderboard format) to dst instead of promoting",
    )
    a = ap.parse_args()
    if a.to_boundary:
        cand = json.loads(Path(a.src).read_text())
        Path(a.dst).write_text(json.dumps(boundary_dict_of_candidate(cand), indent=1))
        print(f"wrote boundary {a.dst} from {a.src}")
        return
    import jax.numpy as jnp

    from constellaration_methods.engines.alm_ngopt_optimizer import (
        masked_coefficients_and_scale,
    )

    c = json.loads(Path(a.src).read_text())
    x = np.asarray(c["x"] if "x" in c else c["x_scaled"], float)
    mp_, nt_, nfp = int(c["mpol"]), int(c["ntor"]), int(c["nfp"])
    _, _, _, unravel, scale = masked_coefficients_and_scale(
        c["boundary_dict"],
        max_poloidal_mode=mp_,
        max_toroidal_mode=nt_,
        n_field_periods=nfp,
        infinity_norm_spectrum_scaling=float(c.get("scaling", 1.0)),
    )
    surf = unravel(jnp.asarray(x) * jnp.asarray(scale))
    r_cos = pad_fourier(np.asarray(surf.r_cos), mp_, nt_, a.mpol, a.ntor)
    z_sin = pad_fourier(np.asarray(surf.z_sin), mp_, nt_, a.mpol, a.ntor)
    bd = {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "r_sin": np.zeros_like(r_cos).tolist(),
        "z_cos": np.zeros_like(z_sin).tolist(),
        "n_field_periods": nfp,
        "is_stellarator_symmetric": True,
    }
    # returns (boundary, mask, masked_coefficients, unravel, scale); the
    # optimiser's x0 is masked_coefficients / scale (NOT the coefficients)
    _, _, coeffs_new, unravel2, scale2 = masked_coefficients_and_scale(
        bd,
        max_poloidal_mode=a.mpol,
        max_toroidal_mode=a.ntor,
        n_field_periods=nfp,
        infinity_norm_spectrum_scaling=1.0,
    )
    x_new = np.asarray(coeffs_new, float) / np.asarray(scale2, float)
    # round-trip check: the re-expressed boundary must reproduce the padded arrays
    back = unravel2(jnp.asarray(x_new) * jnp.asarray(scale2))
    err = max(
        float(np.abs(np.asarray(back.r_cos) - r_cos).max()),
        float(np.abs(np.asarray(back.z_sin) - z_sin).max()),
    )
    out = {
        "note": (
            f"promoted from {mp_}x{nt_} to {a.mpol}x{a.ntor} with new modes at "
            f"zero; source {a.src}; round-trip max abs error {err:.2e}"
        ),
        "x": np.asarray(x_new, float).tolist(),
        "r": c["r"] if "r" in c else c["residuals"],
        "boundary_dict": bd,
        "mpol": a.mpol,
        "ntor": a.ntor,
        "nfp": nfp,
        "scaling": 1.0,
        "source": {"path": a.src, "mpol": mp_, "ntor": nt_},
    }
    Path(a.dst).write_text(json.dumps(out, indent=1))
    print(
        f"wrote {a.dst}: {len(out['x'])} coordinates at {a.mpol}x{a.ntor}, "
        f"round-trip error {err:.2e}"
    )


if __name__ == "__main__":
    main()
