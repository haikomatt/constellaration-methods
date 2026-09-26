#!/usr/bin/env python3
"""Renders a real canonical-confirmed feasible stellarator boundary for a
social/LinkedIn-style hero image.

Standing rule: only render an actual `canonical_feasible=True` candidate
from `results/paper-baseline/confirmation.ndjson`, reconstructed with the
same construction logic the canonical check itself used (inlined below as
`_boundary_from_dict` -- see that function's docstring for why) -- never an
illustrative/unconfirmed design. Keeps the QI/Boozer-skip settings fix
(`default_high_fidelity_skip_qi()` for `GeometricalProblem`, which does not
need QI) for the bonus |B| re-solve.

Headless: Agg backend, saves PNGs to results/viz/, never calls plt.show().
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from constellaration import forward_model, problems
from constellaration.geometry import surface_rz_fourier

CONFIRMATION_PATH = Path("results/paper-baseline/confirmation.ndjson")
OUT_DIR = Path("results/viz")

DARK_BG = "#05070d"

# The P1 campaign's fixed search-space pin (task_template.N_FIELD_PERIODS /
# task_template.IS_STELLARATOR_SYMMETRIC in the source campaign): `task_template`
# itself is a family-search-campaign module out of scope for this repository,
# so only the two constants and the minimal boundary-dict constructor this
# script actually needs are inlined here, rather than importing the module.
_N_FIELD_PERIODS = 3
_MAX_POLOIDAL_MODE = 4
_MAX_TOROIDAL_MODE = 4
_IS_STELLARATOR_SYMMETRIC = True


def _boundary_from_dict(
    boundary_dict: dict[str, Any],
    max_poloidal_mode: int = _MAX_POLOIDAL_MODE,
    max_toroidal_mode: int = _MAX_TOROIDAL_MODE,
    n_field_periods: int | None = None,
) -> surface_rz_fourier.SurfaceRZFourier:
    """Minimal inlined replacement for the source campaign's
    `task_template.candidate_dict_to_boundary` (out of scope here, see
    module docstring): validates `r_cos`/`z_sin` against the given mode
    truncation and builds a SurfaceRZFourier at the fixed P1 symmetry pin
    (`n_field_periods=3`, stellarator-symmetric), unless a caller passes an
    explicit `n_field_periods` (e.g. an nfp=1 NAE-seeded boundary, which is
    baked into the boundary's real-space shape, not just a tag)."""
    r_cos = np.asarray(boundary_dict["r_cos"], dtype=float)
    z_sin = np.asarray(boundary_dict["z_sin"], dtype=float)

    expected_shape = (max_poloidal_mode + 1, 2 * max_toroidal_mode + 1)
    if r_cos.shape != expected_shape or z_sin.shape != expected_shape:
        raise ValueError(
            f"Expected r_cos/z_sin shape {expected_shape} for "
            f"max_poloidal_mode={max_poloidal_mode}/max_toroidal_mode="
            f"{max_toroidal_mode}, got r_cos={r_cos.shape} z_sin={z_sin.shape}"
        )

    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos,
        z_sin=z_sin,
        n_field_periods=(n_field_periods if n_field_periods is not None else _N_FIELD_PERIODS),
        is_stellarator_symmetric=_IS_STELLARATOR_SYMMETRIC,
    )


# ---------------------------------------------------------------------------
# 1. Pick and reconstruct the confirmed candidate
# ---------------------------------------------------------------------------


def load_confirmed_candidate(confirmation_path: Path = CONFIRMATION_PATH) -> dict:
    records = [json.loads(line) for line in open(confirmation_path) if line.strip()]
    feasible = [r for r in records if r.get("canonical_feasible")]
    if not feasible:
        raise RuntimeError(f"No canonical_feasible=True candidates in {confirmation_path}")
    # Default: best (lowest) canonical objective in the confirmed band.
    best = min(feasible, key=lambda r: r["canonical_objective_value"])
    return best


def load_boundary_json(path: Path) -> surface_rz_fourier.SurfaceRZFourier:
    """Loads a standalone boundary JSON (r_cos/z_sin arrays, e.g. the
    leaderboard-validation exports or extract_best_own_design.py output),
    inferring the mode budget from the array shape."""
    with open(path) as f:
        d = json.load(f)
    r_cos = np.asarray(d["r_cos"], dtype=float)
    return _boundary_from_dict(
        {"r_cos": d["r_cos"], "z_sin": d["z_sin"]},
        max_poloidal_mode=r_cos.shape[0] - 1,
        max_toroidal_mode=(r_cos.shape[1] - 1) // 2,
    )


def build_boundary(candidate: dict) -> surface_rz_fourier.SurfaceRZFourier:
    # Same construction logic the canonical check itself uses
    # (rescore_trajectory.rescore_step -> task_template.candidate_dict_to_boundary
    # in the source campaign; inlined here as `_boundary_from_dict`, see
    # module docstring), so this is exactly the `optimized_boundary` the
    # confirmation record refers to.
    boundary = _boundary_from_dict(candidate["boundary"])
    assert boundary.n_field_periods == _N_FIELD_PERIODS
    assert boundary.is_stellarator_symmetric == _IS_STELLARATOR_SYMMETRIC
    return boundary


# ---------------------------------------------------------------------------
# 2/3. Geometry grids
# ---------------------------------------------------------------------------


def hero_grid_xyz(boundary, n_theta=128, n_phi=256):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    theta_phi = np.stack([theta_grid, phi_grid], axis=-1)
    xyz = surface_rz_fourier.evaluate_points_xyz(boundary, theta_phi)
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]


def cross_section_rz(boundary, phi_value, n_theta=256):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    theta_phi = np.stack([theta, np.full_like(theta, phi_value)], axis=-1)
    rz = surface_rz_fourier.evaluate_points_rz(boundary, theta_phi)
    return rz[..., 0], rz[..., 1]


# ---------------------------------------------------------------------------
# 4. Bonus: |B| on the boundary from a high-fidelity re-solve
# ---------------------------------------------------------------------------


def solve_high_fidelity(boundary):
    problem = problems.GeometricalProblem()
    needs_qi = getattr(problem, "_does_it_require_qi", True)
    settings = (
        forward_model.ConstellarationSettings.default_high_fidelity()
        if needs_qi
        else forward_model.ConstellarationSettings.default_high_fidelity_skip_qi()
    )
    metrics, equilibrium = forward_model.forward_model(boundary, settings=settings)
    return metrics, equilibrium


def modb_on_boundary(equilibrium, theta_grid, phi_grid):
    """|B|(theta, phi) on the plasma boundary from the VMEC++ equilibrium's
    Nyquist Fourier coefficients, using the same half-mesh boundary
    extrapolation VMEC++/constellaration use internally
    (`mhd/magnetics_utils.py`'s `get_x_at_the_boundary` for
    `is_full_mesh=False` quantities): `1.75*x[-1] - x[-2] + 0.25*x[-3]`.
    """
    xm_nyq = np.asarray(equilibrium.xm_nyq)
    xn_nyq = np.asarray(equilibrium.xn_nyq)
    bmnc = np.asarray(equilibrium.bmnc).T[1:, :]  # drop the padding row at s=0
    bmnc_b = 1.75 * bmnc[-1, :] - bmnc[-2, :] + 0.25 * bmnc[-3, :]

    angle = (
        xm_nyq[:, None, None] * theta_grid[None, :, :]
        - xn_nyq[:, None, None] * phi_grid[None, :, :]
    )
    modb = np.sum(bmnc_b[:, None, None] * np.cos(angle), axis=0)
    return modb


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def tight_crop_dark(
    path: Path, bg_hex: str = DARK_BG, threshold: int = 14, margin_frac: float = 0.06
) -> None:
    """Crops away the excess dark margin matplotlib's 3D axes box leaves
    around the actual surface, so the subject fills the frame -- matters for
    a social hero shot, where a subject occupying a fraction of a mostly
    empty square reads as weak. Finds the bounding box of pixels that differ
    from the flat background color by more than `threshold`, then re-expands
    by `margin_frac` of the crop's own size so the crop doesn't clip
    anti-aliased edges.
    """
    from PIL import Image

    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img).astype(np.int16)
    bg = np.array(
        [int(bg_hex[1:3], 16), int(bg_hex[3:5], 16), int(bg_hex[5:7], 16)],
        dtype=np.int16,
    )
    diff = np.abs(arr[..., :3] - bg[None, None, :]).max(axis=-1)
    mask = diff > threshold
    if not mask.any():
        return  # nothing but background -- leave the file alone

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]

    h, w = bottom - top, right - left
    pad_h, pad_w = int(h * margin_frac), int(w * margin_frac)
    top = max(0, top - pad_h)
    bottom = min(arr.shape[0] - 1, bottom + pad_h)
    left = max(0, left - pad_w)
    right = min(arr.shape[1] - 1, right + pad_w)

    img.crop((left, top, right + 1, bottom + 1)).save(path)


def style_3d_axes(ax):
    ax.set_facecolor(DARK_BG)
    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))


def save_hero(x, y, z, out_path: Path, elev, azim, title_suffix, facecolors=None, cmap=None):
    fig = plt.figure(figsize=(12, 12), dpi=280, facecolor=DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if facecolors is not None:
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            antialiased=True,
            shade=False,
            linewidth=0,
        )
    else:
        ax.plot_surface(
            x,
            y,
            z,
            cmap="bone",
            rstride=1,
            cstride=1,
            antialiased=True,
            shade=True,
            linewidth=0,
            edgecolor="none",
            vmin=z.min(),
            vmax=z.max(),
        )
    ax.view_init(elev=elev, azim=azim)
    style_3d_axes(ax)

    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (
        (x.max() + x.min()) / 2,
        (y.max() + y.min()) / 2,
        (z.max() + z.min()) / 2,
    )
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    fig.savefig(out_path, dpi=280, facecolor=DARK_BG)
    plt.close(fig)
    tight_crop_dark(out_path)
    print(f"wrote {out_path} ({title_suffix})")


def save_cross_sections(boundary, out_path: Path):
    nfp = boundary.n_field_periods
    fractions = [0.0, 0.25, 0.5, 0.75]
    colors = ["#4fd1ff", "#a06bff", "#ff5fa2", "#ffd166"]

    fig, ax = plt.subplots(figsize=(9, 9), dpi=180, facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    for frac, color in zip(fractions, colors):
        phi_value = frac * (2 * np.pi / nfp)
        r, z = cross_section_rz(boundary, phi_value)
        ax.plot(r, z, color=color, linewidth=2.4, label=f"φ = {frac:g}·2π/nfp", alpha=0.95)
        ax.fill(r, z, color=color, alpha=0.05)

    ax.set_aspect("equal")
    ax.axis("off")
    legend = ax.legend(loc="upper right", frameon=False, fontsize=12)
    for text in legend.get_texts():
        text.set_color("white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=DARK_BG)
    plt.close(fig)
    print(f"wrote {out_path}")


def save_compare_heroes(boundaries, labels, out_path: Path, elev=22, azim=45) -> None:
    """Side-by-side 3D renders of two boundaries at the same view and scale."""
    fig = plt.figure(figsize=(20, 10), dpi=220, facecolor=DARK_BG)
    grids = [hero_grid_xyz(b) for b in boundaries]
    lims = np.array([[g.min(), g.max()] for x, y, z in grids for g in (x, y, z)])
    max_range = (lims[:, 1] - lims[:, 0]).max() / 2.0

    for i, ((x, y, z), label) in enumerate(zip(grids, labels)):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_surface(
            x,
            y,
            z,
            cmap="bone",
            rstride=1,
            cstride=1,
            antialiased=True,
            shade=True,
            linewidth=0,
            edgecolor="none",
            vmin=z.min(),
            vmax=z.max(),
        )
        ax.view_init(elev=elev, azim=azim)
        style_3d_axes(ax)
        mids = [(g.max() + g.min()) / 2 for g in (x, y, z)]
        ax.set_xlim(mids[0] - max_range, mids[0] + max_range)
        ax.set_ylim(mids[1] - max_range, mids[1] + max_range)
        ax.set_zlim(mids[2] - max_range, mids[2] + max_range)
        ax.set_title(label, color="white", fontsize=16, pad=0)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.96, wspace=0)
    fig.savefig(out_path, dpi=220, facecolor=DARK_BG)
    plt.close(fig)
    print(f"wrote {out_path} (side-by-side heroes)")


def save_compare_cross_sections(boundaries, labels, out_path: Path) -> None:
    """Overlaid poloidal cross-sections: first boundary solid, second dashed,
    at the same four toroidal angles, so shape differences are directly
    visible."""
    nfp = boundaries[0].n_field_periods
    fractions = [0.0, 0.25, 0.5, 0.75]
    colors = ["#4fd1ff", "#a06bff", "#ff5fa2", "#ffd166"]
    linestyles = ["-", "--"]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=180, facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    for boundary, ls, label in zip(boundaries, linestyles, labels):
        for frac, color in zip(fractions, colors):
            phi_value = frac * (2 * np.pi / nfp)
            r, z = cross_section_rz(boundary, phi_value)
            ax.plot(
                r,
                z,
                color=color,
                linewidth=2.0,
                linestyle=ls,
                alpha=0.95,
                label=f"{label}  φ = {frac:g}·2π/nfp",
            )

    ax.set_aspect("equal")
    ax.axis("off")
    legend = ax.legend(loc="upper right", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color("white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=DARK_BG)
    plt.close(fig)
    print(f"wrote {out_path} (overlaid cross-sections)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--boundary-json",
        type=Path,
        default=None,
        help="render this standalone boundary JSON instead of the best "
        "candidate from --confirmation",
    )
    p.add_argument("--confirmation", type=Path, default=CONFIRMATION_PATH)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--prefix", default="", help="filename prefix for outputs")
    p.add_argument("--label", default="candidate")
    p.add_argument(
        "--compare-boundary-json",
        type=Path,
        default=None,
        help="also render this boundary side-by-side with the primary one",
    )
    p.add_argument("--compare-label", default="reference")
    p.add_argument(
        "--skip-modb",
        action="store_true",
        help="skip the |B| high-fidelity re-solve render",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    if args.boundary_json is not None:
        boundary = load_boundary_json(args.boundary_json)
        provenance = {
            "boundary_json": str(args.boundary_json),
            "n_field_periods": boundary.n_field_periods,
            "is_stellarator_symmetric": boundary.is_stellarator_symmetric,
        }
    else:
        candidate = load_confirmed_candidate(args.confirmation)
        boundary = build_boundary(candidate)
        provenance = {
            "call_index": candidate["call_index"],
            "outer_iteration": candidate.get("outer_iteration"),
            "objective_agent_visible": candidate["objective"],
            "canonical_objective_value": candidate["canonical_objective_value"],
            "canonical_feasibility_margin": candidate["canonical_feasibility_margin"],
            "constraint_violation": candidate["constraint_violation"],
            "n_field_periods": boundary.n_field_periods,
            "is_stellarator_symmetric": boundary.is_stellarator_symmetric,
            "source": str(args.confirmation),
        }
    print("Provenance:", json.dumps(provenance, indent=2))
    with open(out_dir / f"{prefix}provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    # --- 2. 3D hero renders (plain surface, guaranteed baseline) ---
    x, y, z = hero_grid_xyz(boundary)
    save_hero(
        x,
        y,
        z,
        out_dir / f"{prefix}hero_oblique.png",
        elev=22,
        azim=45,
        title_suffix="oblique",
    )
    save_hero(
        x,
        y,
        z,
        out_dir / f"{prefix}hero_topdown.png",
        elev=89,
        azim=0,
        title_suffix="top-down",
    )

    # --- 3. Cross-section panel ---
    save_cross_sections(boundary, out_dir / f"{prefix}cross_sections.png")

    # --- 3b. Optional side-by-side comparison ---
    if args.compare_boundary_json is not None:
        other = load_boundary_json(args.compare_boundary_json)
        pair = [boundary, other]
        labels = [args.label, args.compare_label]
        save_compare_heroes(pair, labels, out_dir / f"{prefix}compare_heroes.png")
        save_compare_cross_sections(pair, labels, out_dir / f"{prefix}compare_cross_sections.png")

    if args.skip_modb:
        print("\nDone. Files in", out_dir)
        return

    # --- 4. Bonus: |B|-colored surface from a high-fidelity re-solve ---
    try:
        metrics, equilibrium = solve_high_fidelity(boundary)
        print(
            "High-fidelity re-solve converged. objective:",
            problems.GeometricalProblem().get_objective(metrics),
        )

        n_theta, n_phi = 128, 256
        theta = np.linspace(0, 2 * np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        modb = modb_on_boundary(equilibrium, theta_grid, phi_grid)

        norm = (modb - modb.min()) / (modb.max() - modb.min())
        cmap = plt.get_cmap("plasma")
        facecolors = cmap(norm)

        save_hero(
            x,
            y,
            z,
            out_dir / f"{prefix}hero_modB.png",
            elev=22,
            azim=45,
            title_suffix="|B| colored",
            facecolors=facecolors,
        )

        fig, ax = plt.subplots(figsize=(4, 6), dpi=180, facecolor=DARK_BG)
        fig.subplots_adjust(right=0.4)
        mappable = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=modb.min(), vmax=modb.max())
        )
        cbar = fig.colorbar(mappable, ax=ax, fraction=1.0)
        cbar.set_label("|B| [T]", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        ax.axis("off")
        fig.savefig(out_dir / f"{prefix}hero_modB_colorbar.png", dpi=180, facecolor=DARK_BG)
        plt.close(fig)
        print(f"wrote {out_dir / (prefix + 'hero_modB_colorbar.png')}")
    except Exception as e:
        print(f"Bonus |B|-colored render skipped (non-blocking): {e!r}")

    print("\nDone. Files in", out_dir)


if __name__ == "__main__":
    main()
