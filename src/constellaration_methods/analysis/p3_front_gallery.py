#!/usr/bin/env python3
"""Self-contained static HTML gallery for a P3 (MHDStableQIStellarator)
Pareto front, in the spirit of the competition leader's public "stellarator
atlas" dashboard: per-member 3D isometric surface renders, a front plot, and
a margins table.

Member/file matching rule (Phase 0 evidence: read against the real E15 run
directory before writing this): REPORT.json's `members` list gives each
member a `tag` (e.g. "P:L3:t1.0", "JR:141:+"). Its boundary export lives at
`<run-dir>/<tag with ':' replaced by '_'>.boundary.json` -- e.g.
"JR_141_+.boundary.json" for tag "JR:141:+". A `.phireflected.boundary.json`
sibling exists for every member (a phi-mirrored duplicate export); the
gallery always renders the primary (non-reflected) file, never the sibling.
Optional per-member raw metric values (currently just the raw vacuum-well
value) are looked up from `verification.jsonl` if present, matched by tag
suffix rather than a hard-coded run label -- see `find_raw_well` -- so the
same code runs unchanged against a differently-labelled run directory (e.g.
E16 on Monday).

Host-side only, reused-artifacts only: pure-numpy Fourier surface
reconstruction via `render_boundary_3d.py`'s `evaluate_surface_xyz` and
`evaluate_cross_section_rz` (no `constellaration` import, no VMEC, no
equilibrium re-solve -- geometry only). Isometric renders reuse
`render_boundary_3d.render_boundary_png` unchanged; page chrome reuses
`dashboard_style.py`'s CSS and matplotlib rcParams.

Usage:
    python -m constellaration_methods.analysis.p3_front_gallery \\
        --run-dir results/p3-e15-front \\
        --out results/p3-e15-front/gallery \\
        [--references path/to/p3_front_gallery_references.json]

Writes a single self-contained <out>/index.html (PNGs embedded as base64
data URIs -- no other files written, no network references).
"""

# File-level: the CSS/HTML template strings below are single-line-per-rule
# content (data strings, not code), so several lines legitimately exceed the
# line-length limit -- wrapping them would be re-authoring the markup/CSS,
# not a real fix.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import base64
import io
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from constellaration_methods.analysis.dashboard_style import COLOR, CSS
from constellaration_methods.analysis.render_boundary_3d import (
    DARK_BG,
    evaluate_cross_section_rz,
    render_boundary_png,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Order of REPORT.json's 5-element `confirmed_r` residual arrays -- the same
# order scripts/h4_p3_frontier_eval.py's CONSTRAINT_NAMES uses for
# `problem._normalized_constraint_violations` (positive = violated).
CONSTRAINT_NAMES = ("iota", "log10_qi", "mirror", "flux", "well")

# solo_hypervolume = (1 + min_len) * (20 - aspect); RESULT.md documents the
# hypervolume reference point as "ref (1.0, 20.0)" -- that "1.0" is the +1
# shift applied to min_len (i.e. min_len_reference = -1.0), "20.0" is the
# aspect reference used directly.
ASPECT_REFERENCE = 20.0
MIN_LEN_REFERENCE = -1.0

VIEW_NAMES = ("iso_oblique", "iso_topdown", "cross_section")
CROSS_SECTION_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
CROSS_SECTION_COLORS = ("#4fd1ff", "#a06bff", "#ff5fa2", "#ffd166")

# Default references file lives alongside this module (previously resolved
# relative to a scripts/ directory via a repo-root sys.path hack; this
# module is now importable normally, so the default is just its own
# sibling file).
DEFAULT_REFERENCES_PATH = Path(__file__).resolve().parent / "p3_front_gallery_references.json"
DEFAULT_REFERENCES = [
    {"label": "phanerozoic (2026-09-15, 668 members)", "hypervolume": 207.53},
    {"label": "canyon-ant (2026-09-09, 43 members)", "hypervolume": 205.82},
    {"label": "davidkh (2026-07-10, 200 members)", "hypervolume": 187.91},
    {"label": "ExLLM", "hypervolume": 133.6},
    {"label": "ALM-NGOpt", "hypervolume": 129.8},
]

FOOTER_LINE = (
    "Derived from a public leaderboard boundary; comparison and study only; "
    "not for submission without reuse terms."
)


@dataclass
class FrontMember:
    """One confirmed P3 front member, as loaded from REPORT.json plus its
    matched boundary export (and, optionally, verification.jsonl)."""

    tag: str
    aspect: float
    min_len: float
    residuals: dict[str, float]
    raw_well: float | None
    solo_hv: float
    retained: bool
    boundary_path: Path


# ---------------------------------------------------------------------------
# Matching: member tag -> boundary export file / raw verification record
# ---------------------------------------------------------------------------


def member_file_stem(tag: str) -> str:
    """A report member's `tag` (e.g. "JR:141:+") maps to its boundary
    export filename stem by replacing ':' with '_' (e.g. "JR_141_+") --
    verified against every one of the 9 real E15 front members' filenames
    before writing this function."""
    return tag.replace(":", "_")


def find_boundary_file(run_dir: Path, tag: str) -> Path:
    """The primary (non phi-reflected) boundary export for `tag`. Fails
    loud -- a missing export for a reported front member is a real data
    problem, never silently skipped."""
    stem = member_file_stem(tag)
    path = run_dir / f"{stem}.boundary.json"
    if not path.exists():
        raise FileNotFoundError(f"no boundary export for member tag {tag!r} (expected {path})")
    return path


def find_raw_well(run_dir: Path, tag: str) -> float | None:
    """Looks up the raw (un-normalised) vacuum-well metric for `tag` from
    verification.jsonl, if present. That file's own tag format is
    "CONF:<run-label>:<file-stem>:<confirmation-index>" -- matched here on
    the file-stem (second-from-last colon-separated segment) so the lookup
    is correct whatever the run label is (E15, E16, ...), not hard-coded to
    this run. Returns None if the file or a matching row is absent: a
    missing *raw* value is optional provenance, not required data, so it
    must never block the gallery."""
    path = run_dir / "verification.jsonl"
    if not path.exists():
        return None
    stem = member_file_stem(tag)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        parts = record.get("tag", "").split(":")
        if len(parts) >= 2 and parts[-2] == stem:
            return record.get("raw", {}).get("vacuum_well")
    return None


def load_front_members(run_dir: Path, report: dict) -> list[FrontMember]:
    """Builds one FrontMember per REPORT.json `members` entry, matching each
    to its boundary export and (optionally) its raw verification record."""
    members = []
    for entry in report["members"]:
        tag = entry["tag"]
        residual_values = entry["confirmed_r"]
        if len(residual_values) != len(CONSTRAINT_NAMES):
            raise ValueError(
                f"member {tag!r} has {len(residual_values)} residuals, "
                f"expected {len(CONSTRAINT_NAMES)} {CONSTRAINT_NAMES}"
            )
        aspect = entry["confirmed_aspect_ratio"]
        min_len = entry["confirmed_min_len"]
        members.append(
            FrontMember(
                tag=tag,
                aspect=aspect,
                min_len=min_len,
                residuals=dict(zip(CONSTRAINT_NAMES, residual_values)),
                raw_well=find_raw_well(run_dir, tag),
                solo_hv=entry.get("solo_hypervolume", solo_hypervolume(aspect, min_len)),
                retained=entry.get("retained", True),
                boundary_path=find_boundary_file(run_dir, tag),
            )
        )
    return members


# ---------------------------------------------------------------------------
# Pure logic: front ordering, residual colouring, hypervolume, feasibility
# ---------------------------------------------------------------------------


def nondominated_front(
    points: list[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    """Filters (tag, aspect, min_len) triples to the nondominated subset
    (aspect minimised, min_len maximised) and returns it sorted ascending by
    aspect. REPORT.json's own `members` list is expected to already be the
    nondominated front, but this never trusts that blindly -- a defensive
    recompute before the front plot joins any points with a line."""
    kept = []
    for tag, aspect, min_len in points:
        dominated = any(
            other_aspect <= aspect
            and other_min_len >= min_len
            and (other_aspect < aspect or other_min_len > min_len)
            for other_tag, other_aspect, other_min_len in points
            if other_tag != tag
        )
        if not dominated:
            kept.append((tag, aspect, min_len))
    return sorted(kept, key=lambda point: point[1])


def residual_color(value: float, tol: float = 0.01) -> str:
    """green: strictly negative (safely inside the bound); amber: within
    tolerance (0 <= value <= tol -- the margin the confirmation step itself
    still treats as feasible); red: beyond tolerance (a genuine violation)."""
    if value < 0:
        return "green"
    if value <= tol:
        return "amber"
    return "red"


def solo_hypervolume(
    aspect: float,
    min_len: float,
    aspect_reference: float = ASPECT_REFERENCE,
    min_len_reference: float = MIN_LEN_REFERENCE,
) -> float:
    """(min_len - min_len_reference) * (aspect_reference - aspect), i.e.
    (1 + min_len) * (20 - aspect) with the module defaults -- the same
    formula REPORT.json's own `solo_hypervolume` field uses (verified
    against P:L3:t1.0's reported 108.3063115831926)."""
    return (min_len - min_len_reference) * (aspect_reference - aspect)


def benchmark_feasible(residuals: dict[str, float]) -> bool:
    """True only if every P3 normalised residual is strictly negative --
    the same "every nominal bound strictly" condition REPORT.json's
    `strict_nominal` field records per member."""
    return all(value < 0 for value in residuals.values())


# ---------------------------------------------------------------------------
# REPORT.json layout normalisation
# ---------------------------------------------------------------------------


# Phase 0 evidence (read against the real p3-e15-front and p3-e16-front
# REPORT.json files before writing this): two layouts exist.
#
# E15 (scripts/p3_front_from_archive.py): top-level `members` list, each
# entry carrying `tag`, `archived_*`, `confirmation_1`/`confirmation_2`,
# `retained`, `confirmed_r`, `confirmed_V`, `confirmed_aspect_ratio`,
# `confirmed_min_len`, `strict_nominal`, `solo_hypervolume`; plus top-level
# `confirmed_front_tags`, `confirmed_hypervolume`, and
# `canonical_evaluations_used` (that script's own docstring: candidates are
# archived cheaply, then the nondominated set is "evaluate[d] ... twice in
# fresh canonical processes" -- `canonical_evaluations_used = len(xs)` is
# the count of those fresh high-fidelity confirmation runs, "<= 30").
#
# E16 (scripts/p3_front_builder.py): top-level `confirmed_front` list
# instead, each entry only `tag`, `r`, `aspect_ratio`, `min_len` (already
# confirmed -- there is no separate retained/rejected set to distinguish).
# No `strict_nominal` or `solo_hypervolume` per member, and no top-level
# `confirmed_hypervolume` -- use `final_hypervolume` instead (the same
# hypervolume-of-the-confirmed-front quantity, per that script's own
# `hypervolume_gain = final_hypervolume - initial_hypervolume` framing).
# No `canonical_evaluations_used` either; `p3_front_builder.py` runs two
# budgets, `opt_evals` (cheap optimisation-loop evaluations, comment "step
# is accepted only if the canonical evaluation ...") and `verif_evals`
# (the high-fidelity confirmation budget, `Budget(cap=args.max_verif_evals)`
# -- the same "canonical/high-fidelity confirmation" concept E15 counts as
# `canonical_evaluations_used`) -- so `verif_evals` is the analogous field,
# not `opt_evals` or their sum.
def normalize_report(report: dict) -> dict:
    """Returns `report` unchanged if it is already E15-shaped (has a
    top-level `members` list), otherwise builds an equivalent E15-shaped
    dict from an E16-shaped `confirmed_front` list so the rest of the
    script (which only ever reads the E15 field names) runs unmodified
    against either layout. Refuses -- ValueError naming the top-level keys
    actually seen -- rather than guessing at an unrecognised third layout."""
    if "members" in report:
        return report
    if "confirmed_front" in report:
        members = []
        for entry in report["confirmed_front"]:
            residuals = entry["r"]
            members.append(
                {
                    "tag": entry["tag"],
                    "confirmed_r": residuals,
                    "confirmed_aspect_ratio": entry["aspect_ratio"],
                    "confirmed_min_len": entry["min_len"],
                    "retained": True,
                    "strict_nominal": max(residuals) < 0,
                    "solo_hypervolume": solo_hypervolume(entry["aspect_ratio"], entry["min_len"]),
                }
            )
        return {
            **report,
            "members": members,
            "confirmed_front_tags": [member["tag"] for member in members],
            "confirmed_hypervolume": report["final_hypervolume"],
            "canonical_evaluations_used": report["verif_evals"],
        }
    raise ValueError(
        "unrecognised REPORT.json layout: expected a top-level 'members' "
        "(E15) or 'confirmed_front' (E16) list, found neither -- top-level "
        f"keys seen: {sorted(report.keys())}"
    )


# ---------------------------------------------------------------------------
# Rendering (I/O-heavy; exercised for real only inside the pinned image, not
# by unit tests -- see module docstring / tests/analysis/test_p3_front_gallery.py)
# ---------------------------------------------------------------------------


def _png_to_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_cross_section_strip(boundary: dict, out_path: Path) -> None:
    """Poloidal cross-section strip at 4 toroidal angles (0, 1/4, 1/2, 3/4
    of one field period), pure-numpy via
    render_boundary_3d.evaluate_cross_section_rz -- same visual language
    (dark stage, one colour per angle) as
    visualize_boundary.py:save_cross_sections, rewritten against the
    boundary-dict export schema so it needs no `constellaration` import."""
    nfp = int(boundary.get("n_field_periods", 3))
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180, facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    for fraction, color in zip(CROSS_SECTION_FRACTIONS, CROSS_SECTION_COLORS):
        phi_value = fraction * (2 * np.pi / nfp)
        r, z = evaluate_cross_section_rz(boundary, phi_value)
        ax.plot(
            r,
            z,
            color=color,
            linewidth=2.2,
            alpha=0.95,
            label=f"phi = {fraction:g}*2pi/nfp",
        )
        ax.fill(r, z, color=color, alpha=0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=DARK_BG)
    plt.close(fig)


GALLERY_IMAGE_MAX_WIDTH = 900


def _downsize_for_embedding(path: Path, max_width: int = GALLERY_IMAGE_MAX_WIDTH) -> None:
    """Shrinks an already-rendered PNG to `max_width` before it is
    base64-embedded. render_boundary_png's >=1600px (here ~2800px)
    leaderboard-hero output is sized for one social hero shot, not for the
    28 thumbnails a 9-member gallery grid embeds -- downsizing the finished
    PNG here (rather than changing render_boundary_png's own resolution)
    keeps that function reused exactly as-is."""
    from PIL import Image

    with Image.open(path) as img:
        if img.width <= max_width:
            return
        scale = max_width / img.width
        resized = img.resize((max_width, round(img.height * scale)), Image.LANCZOS)
    resized.save(path)


def render_member_images(boundary: dict, tag: str) -> dict[str, str]:
    """Renders the 3 views for one member (2 isometric angles + a poloidal
    cross-section strip) to temporary PNGs and returns each as a base64
    `data:` URI, keyed by VIEW_NAMES."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        oblique_path = tmp_dir / "iso_oblique.png"
        topdown_path = tmp_dir / "iso_topdown.png"
        cross_path = tmp_dir / "cross_section.png"

        render_boundary_png(boundary, oblique_path, title=tag, elev=22, azim=45)
        render_boundary_png(boundary, topdown_path, title=tag, elev=89, azim=0)
        render_cross_section_strip(boundary, cross_path)
        for path in (oblique_path, topdown_path, cross_path):
            _downsize_for_embedding(path)

        return {
            "iso_oblique": _png_to_data_uri(oblique_path),
            "iso_topdown": _png_to_data_uri(topdown_path),
            "cross_section": _png_to_data_uri(cross_path),
        }


def render_front_plot(members: list[FrontMember], references: list[dict]) -> str:
    """Aspect ratio (x, lower better) vs min normalised L_grad_B (y, higher
    better): the nondominated front joined by a line, a schematic shaded
    "hypervolume direction" wedge (the true reference point is far off this
    panel's own data range, so the shading is drawn to the panel's own
    corner and annotated, not to true coordinate scale), and any
    `{label, aspect, min_len}` references plotted as points. Returns a
    base64 PNG `data:` URI."""
    front = nondominated_front([(m.tag, m.aspect, m.min_len) for m in members])
    xs = [point[1] for point in front]
    ys = [point[2] for point in front]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=160)
    xlim = (min(xs) - 0.02, max(xs) + 0.05)
    ylim = (min(ys) - 0.05, max(ys) + 0.02)

    prev_y = ylim[0]
    for x, y in zip(xs, ys):
        ax.fill_between([x, xlim[1]], prev_y, y, color=COLOR["soft"], alpha=0.6, zorder=0)
        prev_y = y

    ax.plot(
        xs,
        ys,
        "-o",
        color=COLOR["accent"],
        linewidth=1.8,
        markersize=6,
        zorder=3,
        label="confirmed front",
    )
    for reference in references:
        if "aspect" in reference and "min_len" in reference:
            ax.scatter(
                [reference["aspect"]],
                [reference["min_len"]],
                marker="D",
                color=COLOR["ochre"],
                zorder=4,
                label=reference["label"],
            )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("aspect ratio (lower is better)")
    ax.set_ylabel("min normalised L_grad_B (higher is better)")
    ax.set_title("P3 confirmed front")
    # The front runs bottom-left to top-right, so the legend sits at
    # upper-left (near the low-aspect/low-min_len end) and the reference
    # annotation at bottom-right (near the high-aspect/high-min_len end,
    # closest to the shaded hypervolume direction) -- neither overlaps data
    # or the other.
    ax.legend(loc="upper left", fontsize=7.5)
    ax.annotate(
        f"hypervolume reference point (aspect={ASPECT_REFERENCE:.1f}, "
        f'min_len={MIN_LEN_REFERENCE:.1f}) -- RESULT.md\'s "ref (1.0, 20.0)".\n'
        "Shaded to this panel's own corner for legibility; not to true scale.",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        fontsize=6.5,
        color=COLOR["muted"],
        ha="right",
        va="bottom",
    )

    hv_only = [r for r in references if "aspect" not in r and "hypervolume" in r]
    if hv_only:
        note = "Reference hypervolumes (front not ours to plot): " + "; ".join(
            f"{r['label']} = {r['hypervolume']:.1f}" for r in hv_only
        )
        fig.text(0.02, 0.005, note, fontsize=7, color=COLOR["muted"], ha="left")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# HTML assembly (pure string templating -- no I/O; unit tested directly)
# ---------------------------------------------------------------------------


def _member_card_html(member: FrontMember, images: dict[str, str]) -> str:
    return f"""
  <article class="mcard">
    <div class="mcard-imgs">
      <img src="{images["iso_oblique"]}" alt="isometric oblique view, {member.tag}">
      <img src="{images["iso_topdown"]}" alt="isometric top-down view, {member.tag}">
      <img class="cross-section" src="{images["cross_section"]}" alt="poloidal cross-section strip, {member.tag}">
    </div>
    <div class="mcard-body">
      <b>{member.tag}</b>
      <table class="mini">
        <tr><td>aspect</td><td class="num">{member.aspect:.4f}</td></tr>
        <tr><td>min_len</td><td class="num">{member.min_len:.4f}</td></tr>
        <tr><td>solo HV</td><td class="num">{member.solo_hv:.3f}</td></tr>
      </table>
    </div>
  </article>"""


def _margin_row_html(member: FrontMember, tol: float) -> str:
    cells = "".join(
        f'<td class="num cell-{residual_color(member.residuals[name], tol)}">'
        f"{member.residuals[name]:.4f}</td>"
        for name in CONSTRAINT_NAMES
    )
    raw_well = "n/a" if member.raw_well is None else f"{member.raw_well:.5f}"
    feasible = "yes" if benchmark_feasible(member.residuals) else "no"
    return (
        f"<tr><td>{member.tag}</td>{cells}"
        f'<td class="num">{raw_well}</td>'
        f'<td class="num">{member.aspect:.4f}</td>'
        f'<td class="num">{member.min_len:.4f}</td>'
        f'<td class="num">{member.solo_hv:.3f}</td>'
        f"<td>{feasible}</td></tr>"
    )


EXTRA_CSS = """
.gallery-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.1rem; }
.mcard { background: var(--paper-2); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.mcard-imgs { display: grid; grid-template-columns: 1fr 1fr; background: #05070d; }
.mcard-imgs img { width: 100%; display: block; }
.mcard-imgs img.cross-section { grid-column: 1 / -1; }
.mcard-body { padding: .7rem .85rem .85rem; }
.mcard-body b { font-family: var(--serif); font-size: 1rem; }
.front-plot { max-width: 100%; border-radius: 10px; border: 1px solid var(--line); background: var(--paper-2); }
.cell-green { color: var(--accent); font-weight: 600; }
.cell-amber { color: var(--ochre); font-weight: 600; }
.cell-red { color: var(--alert); font-weight: 600; }
"""


def build_html(
    *,
    run_name: str,
    report: dict,
    manifest: dict,
    members: list[FrontMember],
    member_images: dict[str, dict[str, str]],
    front_plot_uri: str,
    references: list[dict],
    footer_text: str = FOOTER_LINE,
) -> str:
    """Pure string templating -- no I/O, no rendering. `member_images` maps
    each member's tag to its VIEW_NAMES -> data-URI dict (see
    `render_member_images`). `footer_text` defaults to the module's own
    generic, non-identifying `FOOTER_LINE` but is a parameter (not a
    hardcoded literal in the returned HTML) so a caller can supply its own
    footer without editing this function."""
    tol = report.get("tol", 0.01)
    provenance_rows = "".join(
        f"<tr><td>{key}</td><td>{value}</td></tr>"
        for key, value in sorted(manifest.items())
        if key != "files"
    )
    cards = "\n".join(_member_card_html(member, member_images[member.tag]) for member in members)
    table_rows = "".join(_margin_row_html(member, tol) for member in members)
    hv_reference_rows = "".join(
        f"<tr><td>{reference['label']}</td>"
        f'<td class="num">{reference.get("hypervolume", "-")}</td></tr>'
        for reference in references
    )
    confirmed_hv = report.get("confirmed_hypervolume", float("nan"))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>P3 front gallery -- {run_name}</title>
<style>{CSS}{EXTRA_CSS}</style>
</head>
<body>
<div class="wrap">
<header class="hero">
  <div class="eyebrow">MHDStableQIStellarator (P3)</div>
  <h1>Confirmed front gallery -- {run_name}</h1>
  <p class="sub">Every render below reconstructs a real, on-disk boundary
  export from this run directory -- pure-numpy Fourier surface geometry, no
  equilibrium re-solve. Study and comparison only.</p>
  <div class="status-row">
    <div class="status-tile"><div class="k">Members</div><div class="v">{len(members)}</div></div>
    <div class="status-tile good"><div class="k">Confirmed hypervolume</div><div class="v">{confirmed_hv:.4f}</div></div>
    <div class="status-tile"><div class="k">Tolerance</div><div class="v">{tol}</div></div>
    <div class="status-tile"><div class="k">Canonical evaluations</div><div class="v">{report.get("canonical_evaluations_used", "-")}</div></div>
  </div>
</header>

<section class="panel">
  <h2>1. Front plot</h2>
  <p class="desc">Aspect ratio (x, lower is better) against minimum
  normalised L_grad_B (y, higher is better), the {len(members)} confirmed
  members joined along the nondominated front. See the plot's own
  annotation for how the hypervolume reference point is shown.</p>
  <img class="front-plot" src="{front_plot_uri}" alt="P3 front plot: aspect ratio vs min normalised L_grad_B">
  <table class="mini" style="max-width:420px;margin-top:.8rem;">
    <tr><th>Reference (public P3 leaderboard)</th><th class="num">Hypervolume</th></tr>
    {hv_reference_rows}
  </table>
</section>

<section class="panel">
  <h2>2. Per-member renders</h2>
  <p class="desc">Two isometric views of the last closed flux surface plus a
  poloidal cross-section strip at four toroidal angles, per confirmed front
  member.</p>
  <div class="gallery-grid">
{cards}
  </div>
</section>

<section class="panel">
  <h2>3. Margins table</h2>
  <p class="desc">The five P3 normalised residuals (positive = violated);
  green = strictly negative, amber = within the {tol} confirmation
  tolerance, red = beyond it.</p>
  <div style="overflow-x:auto;">
  <table>
    <thead><tr>
      <th>tag</th><th class="num">iota</th><th class="num">log10_qi</th><th class="num">mirror</th><th class="num">flux</th><th class="num">well</th>
      <th class="num">raw well</th><th class="num">aspect</th><th class="num">min_len</th><th class="num">solo HV</th><th>benchmark-feasible</th>
    </tr></thead>
    <tbody>
{table_rows}
    </tbody>
  </table>
  </div>
</section>

<section class="panel">
  <h2>4. Provenance</h2>
  <table style="max-width:520px;">
{provenance_rows}
  </table>
</section>

<footer class="meta">
  {footer_text}
</footer>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    report = normalize_report(json.loads((run_dir / "REPORT.json").read_text()))
    manifest_path = run_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    references = json.loads(args.references.read_text())

    members = load_front_members(run_dir, report)

    member_images = {}
    for member in members:
        boundary = json.loads(member.boundary_path.read_text())
        member_images[member.tag] = render_member_images(boundary, member.tag)
    front_plot_uri = render_front_plot(members, references)

    html = build_html(
        run_name=run_dir.name,
        report=report,
        manifest=manifest,
        members=members,
        member_images=member_images,
        front_plot_uri=front_plot_uri,
        references=references,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "index.html"
    out_path.write_text(html)
    print(f"wrote {out_path} ({len(members)} members)")


if __name__ == "__main__":
    main()
