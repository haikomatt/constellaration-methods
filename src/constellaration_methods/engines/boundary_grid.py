"""Shared boundary-grid logic: zero-padding to a common (mpol, ntor) grid,
effective-mode / tail-energy characterization, and the seed-relative-distance
novelty metric. Single audited home for logic previously duplicated between
scripts/g1a_dataset_mode_audit.py and scripts/extract_best_own_design.py.
"""

import numpy as np


def _pad_one(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Zero-pad a Fourier coefficient array up to `shape`, keeping the n=0
    column centered (columns index toroidal mode n from -ntor..ntor)."""
    a = np.asarray(a, dtype=float)
    m, n = a.shape
    if m > shape[0] or n > shape[1]:
        raise ValueError(
            f"input shape {a.shape} exceeds target grid {shape} -- refusing to silently truncate"
        )
    out = np.zeros(shape)
    col_off = (shape[1] - n) // 2
    out[:m, col_off : col_off + n] = a
    return out


def pad_to_common_grid(
    r_cos: np.ndarray,
    z_sin: np.ndarray,
    target_mpol: int = 8,
    target_ntor: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad (r_cos, z_sin) up to a (target_mpol+1, 2*target_ntor+1) grid,
    centering the n=0 column. Raises on over-budget input rather than
    truncating."""
    shape = (target_mpol + 1, 2 * target_ntor + 1)
    return _pad_one(r_cos, shape), _pad_one(z_sin, shape)


def effective_modes(
    r_cos: np.ndarray,
    z_sin: np.ndarray,
    tol: float,
    m_cut: int = 4,
    n_cut: int = 4,
) -> tuple[int, int, float]:
    """Largest poloidal m and |toroidal n| with any coefficient above tol,
    plus the energy fraction in the (m>m_cut or |n|>n_cut) tail."""
    mag = np.sqrt(np.asarray(r_cos, dtype=float) ** 2 + np.asarray(z_sin, dtype=float) ** 2)
    mpol1, ncols = mag.shape
    ntor = (ncols - 1) // 2
    n_index = np.abs(np.arange(-ntor, ntor + 1))

    row_active = np.where((mag > tol).any(axis=1))[0]
    col_active = np.where((mag > tol).any(axis=0))[0]
    eff_m = int(row_active.max()) if row_active.size else 0
    eff_n = int(n_index[col_active].max()) if col_active.size else 0

    m_grid = np.arange(mpol1)[:, None] * np.ones((1, ncols))
    n_grid = n_index[None, :] * np.ones((mpol1, 1))
    tail = (m_grid > m_cut) | (n_grid > n_cut)
    e_total = float((mag**2).sum())
    e_tail = float((mag[tail] ** 2).sum()) if tail.any() else 0.0
    tail_frac = e_tail / e_total if e_total > 0 else 0.0
    return eff_m, eff_n, tail_frac


def relative_seed_distance(x: np.ndarray, seed: np.ndarray) -> float:
    """`rel_dist = ||x - seed||_2 / ||seed||_2`, the seed-locality / novelty
    metric. `x` and `seed` must already be on the same (padded) grid -- e.g.
    the concatenation of a design's r_cos/z_sin arrays."""
    x = np.asarray(x, dtype=float)
    seed = np.asarray(seed, dtype=float)
    dist = np.sqrt(((x - seed) ** 2).sum())
    seed_norm = np.sqrt((seed**2).sum())
    return float(dist / seed_norm)
