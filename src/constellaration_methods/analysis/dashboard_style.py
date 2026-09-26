"""Single source of truth for the science dashboard's visual language.

A quiet, editorial dashboard style built for this project: a quiet,
editorial palette (paper + deep teal + ochre-as-reference + a single
accent), serif display type over a plain sans body, wide letter-spaced
small-caps "eyebrow" labels, and a dark "statement" band reserved for the
one place the deck uses it -- title/divider moments -- with everything
else on light paper. Chart color decisions follow the `dataviz` skill's
method (categorical hue order fixed and validated, sequential = one hue
light->dark, status colors reserved for state, never for "series N"); the
hexes below are simply the parameters fed into that method, validated
against this file's own light/dark chart surfaces (see the bottom of this
file).

Import once, reuse everywhere:
    from constellaration_methods.analysis.dashboard_style import (
        COLOR, CSS, apply_mpl_style, CAT_COLORS,
    )

Nothing consuming this module should hardcode a hex color or a
font-family string -- if a new one is needed, add it here first.
"""

# File-level: the CSS block below is single-line-per-rule content (a data
# string, not code), so several lines legitimately exceed the line-length
# limit -- wrapping them would be re-authoring the CSS, not a real fix.
# ruff: noqa: E501

from __future__ import annotations

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Palette -- a quiet, editorial set of light/content tokens plus a matching
# dark/statement-band override, following the same custom-property naming
# on both sides.
# ---------------------------------------------------------------------------

COLOR = {
    # light / content-slide tokens (the default :root)
    "paper": "#F1F3F1",
    "paper_2": "#FBFCFB",
    "ink": "#16211E",
    "muted": "#5E6E68",
    "line": "#D7DEDA",
    "accent": "#0E6E60",
    "soft": "#DCEBE7",
    "ochre": "#B7791F",
    "alert": "#B4302B",
    # dark / title-slide tokens (the .slide.dark overrides)
    "deep": "#0A3A33",
    "deep_2": "#0E4A41",
    "dark_ink": "#F4F8F6",
    "dark_muted": "#A9C3BC",
    "dark_line": "#2C5A51",
    "dark_accent": "#7FD8C6",
    "dark_soft": "#123F37",
}

# Typography -- serif/sans stacks for the page chrome. Chart text stays in
# the plain sans (dataviz: "everything stays in the system sans, no
# display/serif face anywhere in a chart") -- serif is a page-chrome
# (headline) choice only, applied in CSS, never in matplotlib.
SERIF = '"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, ui-serif, serif'
SANS = 'ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
# matplotlib has no access to the web font stack above; "Roboto" is both
# this host's installed sans AND already the page's own --sans fallback
# (see SANS above), so it is not a substitution -- DejaVu Sans is the final
# bundled-with-matplotlib fallback if Roboto is ever unavailable.
MPL_SANS = ["Roboto", "DejaVu Sans", "sans-serif"]

# ---------------------------------------------------------------------------
# Chart color roles. Charts render on light paper (the dashboard's data
# sections are the deck's *content*-slide treatment, not its title-slide
# treatment -- see CSS below), so all chart colors are validated against the
# light surface.
#
# Semantic roles (not raw hexes) so a render_* function reads intent:
#   OURS      -- our own live/current design or series (accent teal)
#   REFERENCE -- an external, study-only comparator, e.g. davidkh (ochre --
#                the same hue the deck itself uses for "provisional /
#                placeholder / reference" text, .placeholder)
#   PRIOR     -- a closed/earlier phase of our own lineage (categorical
#                "blue" slot -- informational, not alarming)
#   HIGHLIGHT -- a derived marker over raw data, e.g. an incumbent-best step
#                line (ochre's warm partner in the deck, --ochre reused)
#   LIMIT     -- a hard threshold/boundary a design must clear or fails
#                (alert red -- a genuine status color, reserved, never used
#                for "series N")
#   GRID/AXIS/TEXT -- chart chrome, from the deck's --line/--muted/--ink
# ---------------------------------------------------------------------------

ROLE = {
    "ours": COLOR["accent"],
    "reference": COLOR["ochre"],
    "prior": "#2a78d6",  # dataviz default categorical slot 1 ("blue"), informational
    "highlight": COLOR["ochre"],
    "limit": COLOR["alert"],
    "grid": COLOR["line"],
    "axis": COLOR["muted"],
    "text": COLOR["ink"],
    "surface": COLOR["paper_2"],
}

# Categorical palette for charts with >2 identity series (the fleet seed
# descent panel: up to ~9 fan seeds). This is the dataviz skill's validated
# default 8-hue categorical theme -- fixed order, never cycled/reassigned --
# re-validated against THIS file's own light chart surface (paper_2) rather
# than the skill's generic default surface; see validation note at bottom of
# file. A 9th series folds into "other" (muted), per the skill's rule, not a
# 9th generated hue.
CAT_COLORS = [
    "#2a78d6",  # 1 blue   ("prior"/informational role above)
    "#008300",  # 2 green
    "#e87ba4",  # 3 magenta
    "#eda100",  # 4 yellow
    "#1baf7a",  # 5 aqua   (closest of the eight to the deck's own accent)
    "#eb6834",  # 6 orange
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
CAT_OTHER = COLOR["muted"]  # 9th+ series fold here, not a generated hue

# One-hue sequential ramp (soft mint -> accent -> deep teal) for continuous
# magnitude fields (the |B| Boozer maps currently use a generic viridis --
# dataviz: "sequential = one hue, light->dark", so this replaces it with the
# deck's own hue family instead of an off-brand rainbow-adjacent default).
SEQUENTIAL_STOPS = [COLOR["soft"], COLOR["accent"], COLOR["deep"]]


def sequential_cmap(name: str = "materiality_teal"):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, SEQUENTIAL_STOPS, N=256)


# ---------------------------------------------------------------------------
# matplotlib rcParams -- chart chrome (surface, ink, gridlines, spines) in
# the deck's light/content-slide tokens. Call once at import time.
# ---------------------------------------------------------------------------


def apply_mpl_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": COLOR["paper_2"],
            "axes.facecolor": COLOR["paper_2"],
            "savefig.facecolor": COLOR["paper_2"],
            "axes.edgecolor": COLOR["line"],
            "axes.linewidth": 0.9,
            "axes.labelcolor": COLOR["ink"],
            "axes.titlecolor": COLOR["ink"],
            "axes.grid": True,
            "grid.color": COLOR["line"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "xtick.color": COLOR["muted"],
            "ytick.color": COLOR["muted"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": COLOR["ink"],
            "font.family": MPL_SANS,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.solid_capstyle": "round",
        }
    )


# ---------------------------------------------------------------------------
# CSS -- the page chrome. Structure: a small uppercase letter-spaced
# "eyebrow" label with an accent tick before it, serif h1/h2 display type, a
# --accent left-rule table treatment, and a light-content / dark-statement
# split -- here used deliberately (not decoratively): the campaign-status
# header is the one dark "statement" band (the title/divider treatment),
# every data section beneath it is light paper (the content-slide
# treatment), so charts rendered on paper_2 sit flush against their section
# background with no color seam.
# ---------------------------------------------------------------------------

CSS = f"""
:root {{
  --paper: {COLOR["paper"]};
  --paper-2: {COLOR["paper_2"]};
  --ink: {COLOR["ink"]};
  --muted: {COLOR["muted"]};
  --line: {COLOR["line"]};
  --accent: {COLOR["accent"]};
  --soft: {COLOR["soft"]};
  --ochre: {COLOR["ochre"]};
  --alert: {COLOR["alert"]};
  --deep: {COLOR["deep"]};
  --deep-2: {COLOR["deep_2"]};
  --serif: {SERIF};
  --sans: {SANS};
}}
* {{ box-sizing: border-box; }}
html {{ background: var(--deep); }}
body {{
  font-family: var(--sans); color: var(--ink); background: var(--paper);
  margin: 0; -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1320px; margin: 0 auto; padding: 0 0 4rem; }}

/* -- header: the dark "statement" band -- */
header.hero {{
  background: linear-gradient(180deg, var(--deep) 0%, var(--deep-2) 100%);
  color: #F4F8F6; padding: 2.6rem clamp(1.4rem, 4vw, 3rem) 2.2rem;
  margin-bottom: 2.2rem;
}}
.eyebrow {{
  font-size: .78rem; font-weight: 600; letter-spacing: .16em; text-transform: uppercase;
  color: #7FD8C6; display: flex; align-items: center; gap: .7rem; margin-bottom: 1rem;
}}
.eyebrow::before {{ content: ""; width: 2rem; height: 2px; background: #7FD8C6; display: inline-block; }}
header.hero h1 {{
  font-family: var(--serif); font-weight: 600; font-size: clamp(1.6rem, 3.2vw, 2.4rem);
  line-height: 1.08; letter-spacing: -.01em; margin: 0 0 .6rem;
}}
header.hero .sub {{ color: #CDE0DA; font-size: 1rem; line-height: 1.5; max-width: 62ch; margin: 0 0 1.6rem; }}
.status-row {{ display: flex; flex-wrap: wrap; gap: 1.6rem 2.4rem; margin-top: 1.2rem; }}
.status-tile {{ min-width: 10rem; }}
.status-tile .k {{ font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; color: #8FB0A8; margin-bottom: .3rem; }}
.status-tile .v {{ font-family: var(--serif); font-size: 1.5rem; color: #F4F8F6; font-variant-numeric: tabular-nums; }}
.status-tile .v .unit {{ font-family: var(--sans); font-size: .85rem; color: #A9C3BC; margin-left: .3rem; }}
.status-tile.good .v {{ color: #7FD8C6; }}
.status-tile.warn .v {{ color: #f0c479; }}
.status-tile.bad .v  {{ color: #f0908c; }}

/* -- content sections: light paper, the content-slide treatment -- */
section.panel {{ padding: 0 clamp(1.4rem, 4vw, 3rem); margin-bottom: 2.6rem; }}
section.panel h2 {{
  font-family: var(--serif); font-weight: 600; font-size: 1.3rem; color: var(--ink);
  margin: 0 0 .3rem; letter-spacing: -.005em;
}}
section.panel .desc {{ color: var(--muted); font-size: .92rem; line-height: 1.55; max-width: 80ch; margin: 0 0 1.1rem; }}
h3 {{ font-family: var(--sans); font-size: .78rem; letter-spacing: .1em; text-transform: uppercase; color: var(--accent); margin: 1.4rem 0 .6rem; }}

/* -- cards (figures, physics views) -- */
.card {{
  background: var(--paper-2); border: 1px solid var(--line); border-radius: 10px;
  padding: 1rem 1.1rem 1.1rem; margin: 0 0 1rem;
}}
.card img {{ max-width: 100%; border-radius: 6px; display: block; }}
.pcard {{
  display: inline-block; vertical-align: top; background: var(--paper-2);
  border-top: 2px solid var(--accent); border-radius: 0 0 8px 8px;
  padding: .8rem .9rem .9rem; margin: 0 .8rem .8rem 0; font-size: .85rem; max-width: 300px;
}}
.pcard b {{ font-family: var(--serif); font-size: .98rem; font-weight: 600; }}
.pcard img {{ max-width: 100%; border-radius: 4px; margin-top: .5rem; }}
.pcard.reference {{ border-top-color: var(--ochre); }}

/* -- tables -- */
table {{ border-collapse: collapse; width: 100%; font-size: .86rem; }}
th {{
  text-align: left; font-weight: 600; font-size: .68rem; letter-spacing: .08em;
  text-transform: uppercase; color: var(--accent); border-bottom: 2px solid var(--line);
  padding: .5rem .7rem; white-space: nowrap;
}}
td {{ padding: .48rem .7rem; border-bottom: 1px solid var(--line); color: var(--ink); vertical-align: top; }}
td:first-child {{ color: var(--muted); }}
tbody tr:hover td {{ background: var(--soft); }}
table.leaderboard tbody tr:first-child td {{ background: var(--soft); font-weight: 600; }}
table.leaderboard tbody tr:first-child {{ border-left: 3px solid var(--accent); }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.cost {{ color: var(--muted); font-variant-numeric: tabular-nums; }}

.note {{ color: var(--muted); font-size: .85rem; line-height: 1.5; }}
.good {{ color: var(--accent); font-weight: 600; }}
.warn {{ color: var(--ochre); font-weight: 600; }}
.bad  {{ color: var(--alert); font-weight: 600; }}
.tag {{
  display: inline-block; font-size: .66rem; letter-spacing: .07em; text-transform: uppercase;
  color: var(--accent); background: var(--soft); padding: .12rem .55rem; border-radius: 999px;
}}
.stamp {{ color: var(--muted); font-size: .78rem; }}
a {{ color: var(--accent); }}

footer.meta {{
  padding: 1.4rem clamp(1.4rem, 4vw, 3rem) 0; border-top: 1px solid var(--line);
  margin: 0 clamp(1.4rem, 4vw, 3rem); color: var(--muted); font-size: .78rem;
}}
"""

# ---------------------------------------------------------------------------
# Validation note (dataviz skill step 3 -- run, don't eyeball):
#
#   node scripts/validate_palette.js \
#     "#2a78d6,#008300,#e87ba4,#eda100,#1baf7a,#eb6834,#4a3aa7,#e34948" \
#     --mode light --surface "#FBFCFB"
#
# -> ALL CHECKS PASS against this file's own paper_2 chart surface (lightness
#    band, chroma floor, adjacent CVD dE >= 8, normal-vision floor 19.6,
#    worst-4 contrast WARN -> mitigated here by the existing per-seed legend
#    + direct line labels, which the dashboard already carries). Re-run
#    against paper_2 specifically (not the skill's generic #fcfcfb default)
#    any time COLOR["paper_2"] changes.
# ---------------------------------------------------------------------------
