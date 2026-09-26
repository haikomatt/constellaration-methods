#!/usr/bin/env python3
"""Renders each leaderboard design (public top-8 + our own submission) as a
3D flux-surface PNG for a leaderboard gallery.

Host-side only, reused-artifacts only: pure-numpy Fourier synthesis of
R(theta, phi) / Z(theta, phi) from a design's `r_cos` / `z_sin` arrays --
the same formula the boundary cross-section helpers elsewhere in this repo
already use for 2D cross-sections, extended here to a full (theta, phi) grid
and to cartesian X/Y/Z. No `constellaration` import (verified not installed
on this host), no equilibrium solve, no container -- this is geometry only,
never physics.

Design list = a manifest of public top-8 winners + this repo's own
submission (score/objective/feasibility read from its own provenance file),
merged and re-ranked by score.

Usage:
    python -m constellaration_methods.analysis.render_boundary_3d
        --> writes one PNG per design + render_manifest.json to
            results/viz/leaderboard-gallery/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from constellaration_methods.analysis.dashboard_style import apply_mpl_style

apply_mpl_style()

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS_DIR = REPO_ROOT / "results" / "family-study" / "targets"
MANIFEST_PATH = TARGETS_DIR / "manifest.json"
SUBMISSION_PATH = REPO_ROOT / "results" / "submission" / "best_feasible_simple_to_build.json"
PROVENANCE_PATH = REPO_ROOT / "results" / "submission" / "PROVENANCE.json"
OUT_DIR = REPO_ROOT / "results" / "viz" / "leaderboard-gallery"

# Same dark stage every hero render in this repo uses
# (visualize_boundary.py's DARK_BG) -- kept identical so a screen-grab of
# this gallery sits flush against the rest of the campaign's renders.
DARK_BG = "#05070d"
CMAP_NAME = "plasma"  # perceptually-uniform, consistent across every design
# Rendered oversize then cropped tight to the surface (matplotlib's default
# 3D axes box leaves a lot of empty margin -- a subject filling only a third
# of the frame reads as weak for a social screen-grab). MIN_WIDTH_PX is the
# floor the final, cropped PNG must clear.
RENDER_WIDTH_PX = 2800
MIN_WIDTH_PX = 1600
DPI = 200


def _fourier_sum_rz(
    r_cos: np.ndarray,
    z_sin: np.ndarray,
    nfp: int,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
):
    """Shared inner loop for stellarator-symmetric Fourier synthesis:
        R(theta, phi) = sum_{m,n} r_cos[m, n] * cos(m*theta - nfp*n*phi)
        Z(theta, phi) = sum_{m,n} z_sin[m, n] * sin(m*theta - nfp*n*phi)
    `theta_grid`/`phi_grid` may be any matching-shape arrays -- a 2D
    (theta, phi) mesh for a full surface (evaluate_surface_xyz) or a 1D
    theta array paired with a phi array filled with one constant value for a
    single poloidal cross-section (evaluate_cross_section_rz). Extracted so
    both share one Fourier-sum implementation rather than two copies of the
    same nested loop.
    """
    m_max = r_cos.shape[0] - 1
    n_max = (r_cos.shape[1] - 1) // 2
    R = np.zeros_like(theta_grid, dtype=float)
    Z = np.zeros_like(theta_grid, dtype=float)
    for m in range(m_max + 1):
        for j, n in enumerate(range(-n_max, n_max + 1)):
            angle = m * theta_grid - nfp * n * phi_grid
            R += r_cos[m, j] * np.cos(angle)
            Z += z_sin[m, j] * np.sin(angle)
    return R, Z


def evaluate_surface_xyz(boundary: dict, n_theta: int = 150, n_phi: int = 320):
    """Stellarator-symmetric surface reconstruction, pure numpy (see
    `_fourier_sum_rz` for the Fourier sum itself). phi spans a full 0..2pi
    turn (not one field period) -- the nfp*n term already makes R, Z
    periodic every 2pi/nfp, so this reconstructs the whole torus in one
    pass, same as visualize_boundary.py's `hero_grid_xyz` did via the
    (unavailable here) constellaration surface evaluator.
    """
    r_cos = np.asarray(boundary["r_cos"], dtype=float)
    z_sin = np.asarray(boundary["z_sin"], dtype=float)
    nfp = int(boundary.get("n_field_periods", 3))

    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    th_grid, ph_grid = np.meshgrid(theta, phi, indexing="ij")
    R, Z = _fourier_sum_rz(r_cos, z_sin, nfp, th_grid, ph_grid)
    X = R * np.cos(ph_grid)
    Y = R * np.sin(ph_grid)
    return X, Y, Z


def evaluate_cross_section_rz(boundary: dict, phi_value: float, n_theta: int = 256):
    """Poloidal cross-section R(theta), Z(theta) at one fixed toroidal angle
    -- the same stellarator-symmetric Fourier synthesis as
    evaluate_surface_xyz (via the shared `_fourier_sum_rz` helper),
    restricted to a single phi. Pure numpy, no `constellaration` import
    (verified not installed on this host -- see module docstring); used by
    p3_front_gallery.py for its per-member cross-section strip.
    """
    r_cos = np.asarray(boundary["r_cos"], dtype=float)
    z_sin = np.asarray(boundary["z_sin"], dtype=float)
    nfp = int(boundary.get("n_field_periods", 3))
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi_grid = np.full_like(theta, phi_value)
    return _fourier_sum_rz(r_cos, z_sin, nfp, theta, phi_grid)


def tight_crop_dark(
    path: Path,
    bg_hex: str = DARK_BG,
    threshold: int = 14,
    margin_frac: float = 0.10,
    min_width_px: int = MIN_WIDTH_PX,
) -> None:
    """Crops away matplotlib's empty 3D-axes margin so the surface fills the
    frame, then re-expands by `margin_frac` so the crop doesn't clip
    anti-aliased edges. If the crop would fall below `min_width_px` on
    either side, it is upscaled back up to the floor -- the deliverable
    constraint (>=1600px wide) always wins over how tight the crop got.
    Adapted from visualize_boundary.py:tight_crop_dark (same approach,
    generalised with a size floor for this gallery's spec)."""
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
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(cols[0]), int(cols[-1])

    h, w = bottom - top, right - left
    pad_h, pad_w = int(h * margin_frac), int(w * margin_frac)
    top = max(0, top - pad_h)
    bottom = min(arr.shape[0] - 1, bottom + pad_h)
    left = max(0, left - pad_w)
    right = min(arr.shape[1] - 1, right + pad_w)

    cropped = img.crop((left, top, right + 1, bottom + 1))
    if cropped.width < min_width_px:
        scale = min_width_px / cropped.width
        cropped = cropped.resize((min_width_px, round(cropped.height * scale)), Image.LANCZOS)
    cropped.save(path)


def render_boundary_png(
    boundary: dict,
    out_path: Path,
    title: str,
    elev: float = 22,
    azim: float = 45,
    accent: bool = False,
) -> dict:
    """One PNG, >=1600px wide, dark background, shaded by local Z (height)
    on a perceptually-uniform colormap. Normalization is PER-DESIGN (each
    design's own Z range) -- documented in the returned dict so the gallery
    page can caption it honestly rather than imply a shared scale."""
    X, Y, Z = evaluate_surface_xyz(boundary)
    z_min, z_max = float(Z.min()), float(Z.max())
    span = max(z_max - z_min, 1e-12)
    norm = (Z - z_min) / span
    cmap = plt.get_cmap(CMAP_NAME)
    facecolors = cmap(norm)

    fig_side = RENDER_WIDTH_PX / DPI
    fig = plt.figure(figsize=(fig_side, fig_side), dpi=DPI, facecolor=DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        antialiased=True,
        shade=False,
        linewidth=0,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor(DARK_BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))

    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    mids = [(a.max() + a.min()) / 2 for a in (X, Y, Z)]
    ax.set_xlim(mids[0] - max_range, mids[0] + max_range)
    ax.set_ylim(mids[1] - max_range, mids[1] + max_range)
    ax.set_zlim(mids[2] - max_range, mids[2] + max_range)

    fig.savefig(out_path, dpi=DPI, facecolor=DARK_BG)
    plt.close(fig)
    # Crop to the surface alone (no title baked in yet), so the bounding
    # box can't be stretched by text sitting far from the geometry -- then
    # compose a title band on top. Keeps the subject filling the frame
    # AND the title crisp, instead of one bounding box trying to hold both.
    tight_crop_dark(out_path)
    compose_title_card(out_path, title, accent=accent)
    return {"z_min": z_min, "z_max": z_max, "cmap": CMAP_NAME}


def compose_title_card(
    path: Path, title: str, band_frac: float = 0.11, accent: bool = False
) -> None:
    """Adds a dark title band above the (already cropped) surface render,
    "user — score" per the spec. `accent` draws a thin accent-teal rule
    under the band, used for our own design's card so it reads as
    celebrated-but-tasteful rather than shouty."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(path).convert("RGB")
    w, h = img.size
    band_h = max(int(h * band_frac), 140)
    canvas = Image.new("RGB", (w, h + band_h), DARK_BG)
    canvas.paste(img, (0, band_h))

    draw = ImageDraw.Draw(canvas)
    font_size = max(int(band_h * 0.34), 28)
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/Roboto-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((w - tw) / 2, (band_h - th) / 2 - bbox[1]),
        title,
        font=font,
        fill="#F4F8F6",
    )
    if accent:
        rule_y = band_h - max(int(band_h * 0.06), 4)
        margin = int(w * 0.28)
        draw.rectangle(
            [margin, rule_y, w - margin, rule_y + max(int(band_h * 0.03), 3)],
            fill="#7FD8C6",
        )
    canvas.save(path)


def build_design_list() -> list[dict]:
    """Top-8 public winners (manifest.json) + our own submission
    (PROVENANCE.json's verified-on-saved-file score/objective/feasibility),
    merged and re-ranked by score descending. No recomputation -- every
    number here is read straight off existing artifacts."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    designs = []
    for row in manifest:
        designs.append(
            {
                "user": row["user"],
                "score": row["score"],
                "objective": row["objective"],
                "feasibility": row["feasibility"],
                "boundary_path": TARGETS_DIR / row["file"],
                "is_ours": False,
            }
        )

    provenance = json.loads(PROVENANCE_PATH.read_text())
    verified = provenance["verified_on_saved_file"]
    designs.append(
        {
            "user": "haikomatt (ours)",
            "score": verified["score"],
            "objective": verified["objective"],
            "feasibility": verified["feasibility_violation"],
            "boundary_path": SUBMISSION_PATH,
            "is_ours": True,
        }
    )

    designs.sort(key=lambda d: d["score"], reverse=True)
    for i, d in enumerate(designs, start=1):
        d["rank"] = i
        slug = d["user"].split(" ")[0].replace("(", "").replace(")", "")
        d["slug"] = f"{i:02d}_{slug}"
    return designs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    designs = build_design_list()
    manifest_out = []
    for d in designs:
        boundary = json.loads(Path(d["boundary_path"]).read_text())
        title = f"{d['user']} — {d['score']:.4f}"
        png_path = OUT_DIR / f"{d['slug']}_boundary.png"
        norm_info = render_boundary_png(boundary, png_path, title, accent=d["is_ours"])
        print(f"wrote {png_path} ({title})")
        manifest_out.append(
            {
                "rank": d["rank"],
                "user": d["user"],
                "score": d["score"],
                "objective": d["objective"],
                "feasibility": d["feasibility"],
                "is_ours": d["is_ours"],
                "boundary_source": str(d["boundary_path"].relative_to(REPO_ROOT)),
                "boundary_png": str(png_path.relative_to(REPO_ROOT)),
                "z_min": norm_info["z_min"],
                "z_max": norm_info["z_max"],
                "cmap": norm_info["cmap"],
            }
        )

    (OUT_DIR / "render_manifest.json").write_text(json.dumps(manifest_out, indent=2))
    print(f"\nwrote {OUT_DIR / 'render_manifest.json'} ({len(manifest_out)} designs)")


if __name__ == "__main__":
    main()
