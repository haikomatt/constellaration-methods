"""Symmetry-canonicalisation helpers for boundary-diversity distances.

Promoted out of the family-search campaign's seed-construction module (it
originated as `qi_family.py`, which also holds campaign-specific seed
interpolation and novelty checks that are out of scope for this repository)
because `analysis.boundary_diversity` (the D1 diversity metric) needs the
symmetry-alignment primitives as its single source of truth: two boundaries
that differ only by a discrete stellarator symmetry (a theta shift, a phi
shift, or a helicity flip) must compare as identical, not merely similar.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def window_4x4(b: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(r_cos, z_sin) restricted to m<=4, |n|<=4, as 5x9 arrays -- the
    common comparison window whatever the design's native mode budget."""
    r = np.asarray(b["r_cos"], dtype=float)
    z = np.asarray(b["z_sin"], dtype=float)
    n_max = (r.shape[1] - 1) // 2
    cols = [n_max + n for n in range(-4, 5)]
    r4 = np.zeros((5, 9))
    z4 = np.zeros((5, 9))
    for m in range(min(5, r.shape[0])):
        for j, c in enumerate(cols):
            if 0 <= c < r.shape[1]:
                r4[m, j] = r[m, c]
                z4[m, j] = z[m, c]
    return r4, z4


def symmetry_variants(
    r: np.ndarray, z: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """The discrete transforms that preserve stellarator symmetry: theta
    shift by pi (factor (-1)^m), phi shift by pi/nfp (factor (-1)^n), and
    toroidal reversal n -> -n (helicity flip). 8 variants, identity
    included (sm=1, sn=1, flip=False).

    Shape-generic: `m_idx`/`n_idx` are derived from the input array's own
    (mpol+1, 2*ntor+1) shape rather than hardcoded to the 4x4 window, so a
    caller on a larger common grid (e.g. the 7x7 / 8x15 grid the P3
    diversity probe pads onto) gets the same transforms.
    """
    variants = []
    m_rows, n_cols = r.shape
    n_max = (n_cols - 1) // 2
    m_idx = np.arange(m_rows)[:, None]
    n_idx = np.arange(-n_max, n_max + 1)[None, :]
    for sm in (1, -1):
        for sn in (1, -1):
            for flip in (False, True):
                fm = (sm**m_idx) * (sn ** np.abs(n_idx))
                rv = r * fm
                zv = z * fm
                if flip:
                    rv = rv[:, ::-1].copy()
                    zv = zv[:, ::-1].copy()
                variants.append((rv, zv, dict(sm=sm, sn=sn, flip=flip)))
    return variants


def align_to(
    r_ref: np.ndarray, z_ref: np.ndarray, r: np.ndarray, z: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Best symmetry variant of (r, z) against the reference, by L2 on
    the shape coefficients (m=0,n=0 major radius excluded). Returns
    (distance^2, aligned_r, aligned_z, variant_tag).
    """
    m_rows, n_cols = r_ref.shape
    n_max = (n_cols - 1) // 2
    mask = np.ones((m_rows, n_cols), dtype=bool)
    mask[0, n_max] = False
    best = None
    for rv, zv, tag in symmetry_variants(r, z):
        d = float(np.sum((rv[mask] - r_ref[mask]) ** 2 + (zv[mask] - z_ref[mask]) ** 2))
        if best is None or d < best[0]:
            best = (d, rv, zv, tag)
    return best
