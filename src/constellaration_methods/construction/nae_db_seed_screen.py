#!/usr/bin/env python3
"""E21 (generator route A', second pass): the near-axis QI database seed
screen. For an entry selected from the Rodriguez and Plunk near-axis QI
database v3 (arXiv 2601.08400; dataset: Zenodo, doi:10.5281/zenodo.18220133,
CC BY 4.0, Max Planck Institute for Plasma Physics), rebuild the analytic
near-axis field with the authors' pyQIC fork, apply their MHD magnetic-well
completion at a requested relative well depth, export a boundary at a
requested aspect ratio, and gate it on a minimum-cylindrical-radius geometry
check before it is handed to the canonical evaluator -- plus a D1
minimum-distance placement report for the confirmed best boundaries.

**Not shipped.** Neither the database (803,532 rows, roughly 16 GB with its
per-configuration sweep files) nor the authors' pyQIC fork's pinned
environment is distributed with this repository. `docs/external-data.md`
states exactly what each is, where to obtain it, its licence, and the local
layout this module expects. Every function that touches them
(`rebuild_and_export`, and therefore `build_batch`'s real path) fails with a
clear, actionable error when they are absent rather than a bare traceback.
`build_batch` and `placement_report` both accept an injected callable in
place of the real rebuild/leaders-load, so this module's own tests -- and
any host without the data or the fork installed -- never need either.

Ported from the working campaign's `scripts/nae_db_p3_generator.py`
(branch `phase1/generator-build`), folded together with the two helper
scripts it built on so the whole pipeline is one importable, tested module
rather than three: `qi_h5prime_build_seed_boundaries.py` (`config_position`,
`load_raw_config`, `digest_config`, `ell_grid`, `fork_qic_kwargs`,
`verify_rebuild`, `boundary_fourier`) and `phase0_nae_db_p3_probe.py`
(`inv_normalise_vpp`). Both are themselves adapted from the database
authors' own `Paper_scripts/Scripts/utils.py` (`make_stel_object` /
`construct_qic`, model/smooth branches dropped -- this module always
rebuilds from the stored grids, never a reduced model) -- the d/sqrt(2),
dbar/sqrt(2) and helicity-sign conventions there are theirs, not a bug to
fix. `inv_normalise_rc` and `inv_normalise_vpp` are copied verbatim from
the authors' `Paper_scripts/Scripts/plot_extremes.py` (same Zenodo record,
CC BY 4.0) because that module imports plotting libraries at import time
and cannot itself be imported headless.

`nearest_family_distance` (the D1 placement metric) is reused, not copied,
from this package's own `construction.p3_construction_proxies`.

Every seed carries its own provenance (database file, row ID, input
features, well target, before/after completion state, near-axis fidelity
fields) so a downstream REPORT.json can cite the database on a per-seed
basis, not just once in a docstring.

Subcommand `build` (runs OUTSIDE any container, in the pinned fork
environment -- see docs/external-data.md -- on whichever cores the campaign
convention reserves for host-side work, e.g.:

    taskset -c 14,15 path/to/fork-venv/bin/python \\
        -m constellaration_methods.construction.nae_db_seed_screen build \\
        --selection selection_primary.json --selection-extra selection_extra.json \\
        --replace-failures --mag-well 0.02 --aspect 12.0 --aspect-by estimate \\
        --min-rmin-ratio 0.5 --workers 2 --out-dir results/seeds

Subcommand `placement` (D1 minimum-distance-to-leaders'-family placement of
the confirmed boundaries; host-testable via an injected `load_leaders_fn` --
the real load of a leaders' archive costs on the order of an hour and is
never run by this module's own tests):

    python -m constellaration_methods.construction.nae_db_seed_screen placement \\
        --boundary-dir results/seeds/boundaries --labels db00538_n2_s231,db00538_n2_s364 \\
        --out results/placement/REPORT.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from constellaration_methods.construction import p3_construction_proxies as pcp

REPO = Path(__file__).resolve().parents[3]

FORK_COMMIT = "2a2e443d"
ZENODO_DOI = "10.5281/zenodo.18220133"
_trapz = getattr(np, "trapezoid", None) or np.trapz

D_RAND = 0.388

IOTA_MATCH_RTOL = 1e-3
MIN_IOTA_OVER_NFP = 0.25
MIN_RMIN_RATIO = 0.5


# --------------------------------------------------------------------------- #
# external-data guard
# --------------------------------------------------------------------------- #


def _import_qic():
    from qic import Qic

    return Qic


def _require_qic(import_fn: Callable[[], Any] | None = None):
    """Import the authors' pyQIC fork's `Qic` class, or raise a clear,
    actionable error if it -- or the right fork -- is not installed. See
    `docs/external-data.md` for exactly what to install and how.
    `import_fn` is injected for testing (no real `qic` package needed);
    the default imports the real one."""
    try:
        qic_cls = (import_fn or _import_qic)()
    except ImportError as e:
        raise RuntimeError(
            "the authors' pyQIC fork ('qic', GitHub SebereX, branch "
            f"'change_structure', commit {FORK_COMMIT}) is not importable "
            f"({e}). This module needs it to rebuild near-axis database "
            "configurations and apply the MHD well completion. The vanilla "
            "PyPI 'qic' package will not work here -- see docs/external-data.md "
            "for exactly what to install and how."
        ) from e
    if not hasattr(qic_cls, "mag_well_reshape"):
        raise RuntimeError(
            "found a 'qic' package, but it is not the authors' fork: it has "
            "no Qic.mag_well_reshape (the MHD well completion this module "
            "applies). See docs/external-data.md."
        )
    return qic_cls


# --------------------------------------------------------------------------- #
# database rebuild (adapted from the database authors' own utils.py,
# Rodriguez & Plunk, Zenodo 10.5281/zenodo.18220133, CC BY 4.0)
# --------------------------------------------------------------------------- #


def config_position(entry: dict[str, Any]) -> int:
    """File position of a dataframe row within its sweep JSON. The authors'
    own `plot_extremes.py` uses `config_id = row["ID"] - 1`: the `ID` column
    is the 1-BASED position in the file (`id_within`/`configurationIndex`
    are different enumerations and do NOT locate the item)."""
    return int(entry["ID"]) - 1


def load_raw_config(db_file: str | Path, config_id: int) -> dict[str, Any]:
    """Stream the `config_id`-th item of a sweep JSON (files are up to
    roughly 50 MB; `ijson` keeps it incremental). Same enumeration the
    authors' `load_config_id_from_db` uses: plain item order within the
    file. Requires the `ijson` package (an ordinary, freely installable
    dependency -- see docs/external-data.md, `pip install -e ".[generator]"`)
    and a local copy of the database's sweep file at `db_file`; raises a
    clear error if either is missing."""
    db_file = Path(db_file)
    if not db_file.exists():
        raise FileNotFoundError(
            f"{db_file} does not exist -- this module needs a local copy of "
            "the Rodriguez and Plunk near-axis QI database (CC BY 4.0, "
            f"Zenodo doi:{ZENODO_DOI}); see docs/external-data.md for the "
            "exact layout and how to obtain it."
        )
    try:
        import ijson
    except ImportError as e:
        raise RuntimeError(
            "the 'ijson' package is required to stream the database's sweep "
            'files. Install it with pip install -e ".[generator]" -- see '
            "docs/external-data.md."
        ) from e
    with open(db_file, "rb") as fh:
        for current_index, obj in enumerate(ijson.items(fh, "item", use_float=True)):
            if current_index == config_id:
                return {name: np.array(value) for name, value in obj}
    raise IndexError(f"{db_file} has no item {config_id}")


def digest_config(config: dict[str, Any]) -> dict[str, Any]:
    """Adapted from the authors' `make_stel_object`: strip the periodic
    endpoint from the grids and apply their d/sqrt(2), dbar/sqrt(2) and
    helicity-sign conventions. Returns a plain dict (host-testable)."""
    return {
        "nphi": int(config["nphi"]),
        "nfp": int(config["nfp"]),
        "iota": float(config["iota"]),
        "helicity": -float(config["helicity"]),
        "varphi": np.asarray(config["varphi"], dtype=float)[:-1],
        "varphi_end": float(np.asarray(config["varphi"], dtype=float)[-1]),
        "d_l_d_varphi": np.asarray(config["d_l_d_varphi"], dtype=float)[:-1],
        "B0": np.asarray(config["B0"], dtype=float)[:-1],
        "d_bar": (np.asarray(config["dbar"], dtype=float) / np.sqrt(2))[:-1],
        "alpha": np.asarray(config["alpha"], dtype=float)[:-1],
        "curvature": np.asarray(config["curvature"], dtype=float)[:-1],
        "torsion": np.asarray(config["torsion"], dtype=float)[:-1],
    }


def ell_grid(digested: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Arc length along the axis over one field period (the authors'
    cumulative trapezoid over the periodically-closed varphi grid);
    returns `(ell[:-1], L)`."""
    from scipy.integrate import cumulative_trapezoid

    dl = np.append(digested["d_l_d_varphi"], digested["d_l_d_varphi"][0])
    varphi = np.append(digested["varphi"], digested["varphi"][0] + 2 * np.pi / digested["nfp"])
    ell = cumulative_trapezoid(dl, varphi, initial=0.0)
    return ell[:-1], float(ell[-1])


def fork_qic_kwargs(digested: dict[str, Any]) -> dict[str, Any]:
    """The authors' `construct_qic` input block (grid inputs, minimal
    second-order shaping X2c=X2s=0, simple-fourier buffers, vacuum)."""
    ell, length = ell_grid(digested)
    return dict(
        Raxis=None,
        Zaxis=None,
        B0={"type": "grid", "input_value": digested["B0"]},
        d=None,
        d_over_curvature={"type": "grid", "input_value": digested["d_bar"]},
        X2c={"type": "scalar", "input_value": 0},
        X2s={"type": "scalar", "input_value": 0},
        omn_buffer={"omn_method": "simple-fourier", "k_buffer": 3},
        sigma0=0.0,
        nfp=digested["nfp"],
        p2=0,
        I2=0,
        order="r2",
        omn=True,
        helicity=digested["helicity"],
        alpha_tilde={
            "type": "grid",
            "input_value": digested["alpha"]
            + digested["varphi"] * digested["helicity"] * digested["nfp"],
        },
        varphi=digested["varphi"],
        L=length,
        ell={"type": "grid", "input_value": ell},
        torsion={"type": "grid", "input_value": digested["torsion"]},
        curvature={"type": "grid", "input_value": digested["curvature"]},
        frenet=True,
        nphi=len(ell),
        phi_shift=0,
        diff_finite=None,
        axis_complete=False,
        solve_geo=True,
        Bbar=1,
    )


def verify_rebuild(qic_iota: float, expected_iota: float, nfp: int) -> str | None:
    """None if the rebuilt config is the selected one and QI-viable, else
    the reason string (recorded in the manifest, never raised silently
    past): never descend from an unchecked rebuild."""
    if not np.isfinite(qic_iota):
        return f"rebuilt iota is not finite: {qic_iota}"
    if abs(qic_iota - expected_iota) > IOTA_MATCH_RTOL * max(1.0, abs(expected_iota)):
        return (
            f"iota mismatch: rebuilt {qic_iota:.6f} vs dataframe {expected_iota:.6f}"
            " -- wrong config_id or construction drift"
        )
    if abs(qic_iota) / nfp < MIN_IOTA_OVER_NFP:
        return f"|iota|/nfp={abs(qic_iota) / nfp:.3f} < {MIN_IOTA_OVER_NFP}"
    return None


def boundary_fourier(q, minor_radius: float, ntheta: int = 64, mmax: int = 10):
    """Finite-r boundary of the near-axis field as stellarator-symmetric
    Fourier arrays, shaped for `constellaration`'s `SurfaceRZFourier`
    (`rc.T` / `zs.T`)."""
    import qic.util

    r2d, z2d, _ = q.Frenet_to_cylindrical(r=minor_radius, ntheta=ntheta)
    rc, _, _, zs = qic.util.to_Fourier(
        R_2D=r2d, Z_2D=z2d, nfp=q.nfp, mpol=mmax, ntor=mmax, lasym=False
    )
    return rc.T, zs.T


# --------------------------------------------------------------------------- #
# authors' aspect/well helpers (verbatim, CC BY 4.0 -- see module docstring)
# --------------------------------------------------------------------------- #


def inv_normalise_rc(stel, aspect: float) -> float:
    """Minor radius achieving `aspect` at `stel`'s major radius: the
    authors' `inv_normalise_rc` (`Paper_scripts/Scripts/plot_extremes.py`,
    Rodriguez and Plunk, Zenodo doi:10.5281/zenodo.18220133, CC BY 4.0),
    copied verbatim (r = R/A, R computed exactly as their helper does)
    because that module imports plotting at import time and cannot itself
    be imported headless."""
    phi = np.append(stel.varphi, 2 * np.pi / stel.nfp + stel.varphi[0])
    big_r = stel.G0 / (2 * np.pi) * (stel.nfp * _trapz(np.append(1 / stel.B0, 1 / stel.B0[0]), phi))
    return float(big_r / aspect)


def inv_normalise_vpp(stel, w: float, a_ref: float = 10.0) -> float:
    """V'' from the relative well depth W (positive = well), the authors'
    `inv_normalise_Vpp` (`Paper_scripts/Scripts/plot_extremes.py`, same
    Zenodo record, CC BY 4.0) verbatim."""
    phi = np.append(stel.varphi, 2 * np.pi / stel.nfp + stel.varphi[0])
    big_r = stel.G0 / (2 * np.pi) * (stel.nfp * _trapz(np.append(1 / stel.B0, 1 / stel.B0[0]), phi))
    r_ref = big_r / a_ref
    factor = (
        stel.Bbar
        * r_ref**2
        / (4 * np.pi * stel.G0)
        / (stel.nfp * _trapz(np.append(1 / stel.B0**2, 1 / stel.B0[0] ** 2), phi))
    )
    return float(w / factor)


# --------------------------------------------------------------------------- #
# geometry gate (E21 amendment 1: axis-excursion corner of the database that
# VMEC rejects outright)
# --------------------------------------------------------------------------- #


def surface_rmin_over_r00(r_cos, nfp: int, ntheta: int = 48, nphi: int = 32) -> float:
    """The minimum cylindrical radius over the exported surface divided by
    its mean major radius (`r_cos[0][ntor]`). One corner of the database
    (axis excursions to roughly 0.16-0.33 of the mean major radius) exports
    surfaces VMEC fails outright on; well-behaved seeds sat at 0.53-0.76.
    `MIN_RMIN_RATIO` is the gate `rebuild_and_export` enforces."""
    rc = np.asarray(r_cos, dtype=float)
    ntor = (rc.shape[1] - 1) // 2
    th = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
    ph = np.linspace(0.0, 2 * np.pi / nfp, nphi, endpoint=False)
    tt, pp = np.meshgrid(th, ph, indexing="ij")
    big_r = np.zeros_like(tt)
    for m in range(rc.shape[0]):
        for j in range(rc.shape[1]):
            n = j - ntor
            big_r += rc[m, j] * np.cos(m * tt - n * nfp * pp)
    r00 = float(rc[0, ntor])
    return float(big_r.min() / r00) if r00 else float("nan")


# --------------------------------------------------------------------------- #
# build: rebuild + MHD completion + aspect export for one selection entry
# --------------------------------------------------------------------------- #


def rebuild_and_export(
    entry: dict[str, Any],
    *,
    mag_well: float,
    aspect: float,
    data_repo: Path,
    min_rmin_ratio: float = MIN_RMIN_RATIO,
    aspect_by: str = "estimate",
) -> dict[str, Any]:
    """Rebuild one selected database entry, apply the authors' MHD
    completion (`Qic.mag_well_reshape`) at relative well `mag_well`, and
    export the boundary at `aspect` via `inv_normalise_rc`. Returns every
    manifest field the batch `build` step needs plus the boundary dict
    (popped off by the caller before writing the manifest). Raises a clear
    error when the database or the pyQIC fork are absent (see
    docs/external-data.md), and on any other failure -- the caller records
    it per-seed, never drops it silently."""
    t0 = time.perf_counter()
    db_file = Path(entry["db_file"])
    if not db_file.is_absolute():
        db_file = Path(data_repo) / db_file
    if not db_file.exists():
        raise FileNotFoundError(
            f"{db_file} does not exist -- this module needs a local copy of "
            "the Rodriguez and Plunk near-axis QI database (CC BY 4.0, "
            f"Zenodo doi:{ZENODO_DOI}); see docs/external-data.md for the "
            "exact layout and how to obtain it."
        )
    qic_cls = _require_qic()

    raw = load_raw_config(db_file, config_position(entry))
    digested = digest_config(raw)
    q = qic_cls(**fork_qic_kwargs(digested))
    problem = verify_rebuild(float(q.iota), float(entry["iota"]), digested["nfp"])
    if problem is not None:
        raise ValueError(problem)
    vpp_before = float(q.d2_volume_d_psi2)
    vpp_target = inv_normalise_vpp(q, mag_well)
    q = qic_cls.mag_well_reshape(q, well=vpp_target)
    vpp_after = float(q.d2_volume_d_psi2)
    minor_radius = inv_normalise_rc(q, aspect)
    r_cos, z_sin = boundary_fourier(q, minor_radius)
    r00 = float(r_cos[0][(r_cos.shape[1] - 1) // 2])
    if aspect_by == "estimate":
        # The authors' helper R overstates the mean major radius of
        # excursive axes; the boundary's own r00/r tracked the canonical
        # aspect much more closely. Re-export once at the radius that makes
        # the boundary's own estimate equal `aspect`.
        minor_radius = r00 / aspect
        r_cos, z_sin = boundary_fourier(q, minor_radius)
        r00 = float(r_cos[0][(r_cos.shape[1] - 1) // 2])
    rmin_ratio = surface_rmin_over_r00(r_cos, digested["nfp"])
    if rmin_ratio < min_rmin_ratio:
        raise ValueError(
            f"geometry gate: rmin/r00 {rmin_ratio:.3f} < {min_rmin_ratio} "
            "(axis-excursion corner, VMEC rejects such surfaces)"
        )
    features = entry.get("features", {})
    return {
        "seed_name": entry["seed_name"],
        "n_field_periods": digested["nfp"],
        "db_file": entry["db_file"],
        "ID": entry["ID"],
        "features": features,
        "W_target": mag_well,
        "vpp_target": vpp_target,
        "d2_volume_d_psi2_before": vpp_before,
        "d2_volume_d_psi2_after": vpp_after,
        "near_axis_iota_over_nfp": abs(float(q.iota)) / digested["nfp"],
        "mirr_ratio": entry.get("mirr_ratio"),
        "L_grad_B": entry.get("L_grad_B"),
        "delta_B": entry.get("delta_B"),
        "rho_c0": features.get("ρc_0", entry.get("ρc_0")),
        "aspect_requested": aspect,
        "aspect_estimate": r00 / minor_radius,
        "aspect_by": aspect_by,
        "rmin_over_r00": rmin_ratio,
        "minor_radius": minor_radius,
        "elapsed_s": time.perf_counter() - t0,
        "fork_commit": FORK_COMMIT,
        "boundary": {
            "r_cos": np.asarray(r_cos, dtype=float).tolist(),
            "z_sin": np.asarray(z_sin, dtype=float).tolist(),
            "n_field_periods": digested["nfp"],
            "is_stellarator_symmetric": True,
        },
    }


def _rebuild_worker(
    args: tuple[dict[str, Any], float, float, Path, float, str],
) -> dict[str, Any]:
    """Top-level (picklable) wrapper for the real `rebuild_and_export` in a
    spawned worker process -- `ProcessPoolExecutor` needs a module-level
    callable, and this is the only entry point that ever runs inside one
    (an injected `rebuild_fn` always runs sequentially, see `build_batch`).
    Records a failure instead of letting it propagate and kill the pool."""
    entry, mag_well, aspect, data_repo, min_rmin_ratio, aspect_by = args
    try:
        return rebuild_and_export(
            entry,
            mag_well=mag_well,
            aspect=aspect,
            data_repo=data_repo,
            min_rmin_ratio=min_rmin_ratio,
            aspect_by=aspect_by,
        )
    except Exception as e:  # noqa: BLE001 -- record, never drop silently
        return {"seed_name": entry["seed_name"], "error": f"{type(e).__name__}: {e}"}


def build_batch(
    entries: list[dict[str, Any]],
    *,
    mag_well: float,
    aspect: float,
    data_repo: Path,
    out_dir: Path,
    rebuild_fn: Callable[..., dict[str, Any]] | None = None,
    workers: int = 2,
    replace_failures: bool = False,
    spare_entries: list[dict[str, Any]] | None = None,
    min_rmin_ratio: float = MIN_RMIN_RATIO,
    aspect_by: str = "estimate",
) -> dict[str, Any]:
    """Rebuild+export every entry in `entries`. Writes `<seed_name>.
    boundary.json` per success into `out_dir/boundaries/` and
    `out_dir/manifest.json` (a dict with `seeds`: one record per requested
    slot, success or failure -- failures are never dropped -- plus
    `replacements` and `n_requested`/`n_success`).

    `rebuild_fn=None` (the default) uses the real `rebuild_and_export`,
    which needs the database and the pyQIC fork (see docs/external-data.md);
    with `workers>1` it runs in a spawned `ProcessPoolExecutor` (host
    multiprocessing -- this function is meant to be called host-side, in
    the fork environment, never inside a container). An injected
    `rebuild_fn` (as this module's own tests use) always runs sequentially
    in this process regardless of `workers` -- test fakes need not be
    picklable, and this is exactly what lets this function, and this
    module's tests, run without the external data at all.

    `replace_failures`: whenever an entry fails, the next unused entry (by
    `seed_name`) from `spare_entries` is queued in its place, so the
    original request count (`len(entries)`) is still reached whenever
    enough spares exist -- `replacements` records each substitution."""
    out_dir = Path(out_dir)
    bdir = out_dir / "boundaries"
    bdir.mkdir(parents=True, exist_ok=True)
    target = len(entries)
    fn = rebuild_fn or rebuild_and_export
    real_pool = rebuild_fn is None and workers > 1
    used_names = {e["seed_name"] for e in entries}
    spare_queue = list(spare_entries or [])

    def run_one(entry: dict[str, Any]) -> dict[str, Any]:
        try:
            result = fn(entry, mag_well=mag_well, aspect=aspect, data_repo=data_repo)
            boundary = result.pop("boundary")
            path = bdir / f"{result['seed_name']}.boundary.json"
            path.write_text(json.dumps(boundary))
            result["file"] = str(path)
            return result
        except Exception as e:  # noqa: BLE001 -- record, never drop silently
            return {
                "seed_name": entry["seed_name"],
                "error": f"{type(e).__name__}: {e}",
            }

    manifest: list[dict[str, Any]] = []
    replacements: list[dict[str, str]] = []
    queue = list(entries)
    n_success = 0
    while queue:
        batch, queue = queue, []
        if real_pool and len(batch) > 1:
            ctx = mp.get_context("spawn")
            args = [(e, mag_well, aspect, data_repo, min_rmin_ratio, aspect_by) for e in batch]
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                raw_results = list(pool.map(_rebuild_worker, args))
            results = []
            for entry, raw in zip(batch, raw_results):
                if "error" in raw:
                    results.append(raw)
                    continue
                boundary = raw.pop("boundary")
                path = bdir / f"{raw['seed_name']}.boundary.json"
                path.write_text(json.dumps(boundary))
                raw["file"] = str(path)
                results.append(raw)
        else:
            results = [run_one(e) for e in batch]
        for entry, rec in zip(batch, results):
            if "error" not in rec:
                n_success += 1
                manifest.append(rec)
                continue
            if replace_failures and n_success < target:
                while spare_queue:
                    spare = spare_queue.pop(0)
                    if spare["seed_name"] in used_names:
                        continue
                    used_names.add(spare["seed_name"])
                    rec["replaced_by"] = spare["seed_name"]
                    replacements.append(
                        {
                            "failed": entry["seed_name"],
                            "replacement": spare["seed_name"],
                        }
                    )
                    queue.append(spare)
                    break
            manifest.append(rec)

    out = {
        "seeds": manifest,
        "replacements": replacements,
        "n_requested": target,
        "n_success": n_success,
    }
    (out_dir / "manifest.json").write_text(json.dumps(out, indent=2, default=float))
    return out


# --------------------------------------------------------------------------- #
# placement: D1 minimum-distance-to-leaders'-family report
# --------------------------------------------------------------------------- #


def _as_n_field_periods_family(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`nearest_family_distance` needs `n_field_periods`; the analysis
    package's own atlas/board loaders key their records `nfp`. Small
    adapter, no data copy beyond the two arrays."""
    return [
        {
            "n_field_periods": r["nfp"],
            "r_cos": r["r_cos"],
            "z_sin": r["z_sin"],
        }
        for r in records
    ]


def _default_load_leaders(atlas_dir: Path | None, board_dir: Path | None) -> list[dict[str, Any]]:
    """Real leaders' load: the public phanerozoic atlas plus the public
    board, reusing `analysis.p3_design_diversity_probe`'s own loaders
    (extend before create). Neither directory ships with this repository
    (they are large public-record downloads, not this project's own
    artefacts); a missing or unset directory simply contributes no
    records -- `placement_report` already reports "n/a" (n=0) for an empty
    family, which is exactly what the real E21 run found (see
    docs/results/p3/E21/STATUS.md). Never called by this module's own
    tests, which inject `load_leaders_fn`."""
    from constellaration_methods.analysis.p3_design_diversity_probe import (
        load_atlas,
        load_board,
    )

    leaders: list[dict[str, Any]] = []
    if atlas_dir is not None and Path(atlas_dir).exists():
        leaders.extend(_as_n_field_periods_family(load_atlas(Path(atlas_dir))))
    if board_dir is not None and Path(board_dir).exists():
        leaders.extend(_as_n_field_periods_family(load_board(Path(board_dir))))
    return leaders


def load_reference_boundary(path: Path | None) -> dict[str, Any] | None:
    """A single fixed reference boundary (in the original study, a
    davidkh-lineage feasible P3 boundary) to compare placement against, or
    `None` if `path` is `None` or does not exist -- this is a study-only
    comparison and this repository ships no such file by default."""
    if path is None or not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    return {
        "n_field_periods": int(d["n_field_periods"]),
        "r_cos": d["r_cos"],
        "z_sin": d["z_sin"],
    }


def _summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "median": None}
    s = sorted(values)
    return {"n": len(s), "min": s[0], "median": float(np.median(s))}


def placement_report(
    boundary_dir: Path,
    labels: list[str],
    *,
    load_leaders_fn: Callable[[], list[dict[str, Any]]],
    reference_boundary: dict[str, Any] | None,
    d_rand: float = D_RAND,
) -> dict[str, Any]:
    """D1 minimum distance of each `label`'s boundary (`boundary_dir/
    <label>.boundary.json`) to the nearest same-nfp leaders'-family member,
    and to `reference_boundary` (a single fixed boundary, treated as a
    family of one) -- reusing `construction.p3_construction_proxies.
    nearest_family_distance` for both (extend-before-create). Reports
    min/median over `labels` for each set, alongside the study-only
    reference `d_rand`."""
    leaders = load_leaders_fn()
    reference_family = [reference_boundary] if reference_boundary is not None else []
    per_label = []
    for label in labels:
        path = Path(boundary_dir) / f"{label}.boundary.json"
        boundary = json.loads(path.read_text())
        d_leaders = pcp.nearest_family_distance(boundary, leaders)
        d_reference = pcp.nearest_family_distance(boundary, reference_family)
        per_label.append({"label": label, "d_leaders": d_leaders, "d_reference": d_reference})
    return {
        "d_rand": d_rand,
        "n_labels": len(labels),
        "leaders": _summarize([r["d_leaders"] for r in per_label if r["d_leaders"] is not None]),
        "reference": _summarize(
            [r["d_reference"] for r in per_label if r["d_reference"] is not None]
        ),
        "per_label": per_label,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="rebuild+complete+export a selection of database seeds")
    b.add_argument(
        "--selection",
        type=Path,
        required=True,
        help="a selection JSON (seed entries with db_file/ID/iota/... -- see "
        "docs/external-data.md)",
    )
    b.add_argument("--selection-extra", type=Path, default=None)
    b.add_argument("--mag-well", type=float, default=0.02)
    b.add_argument("--aspect", type=float, default=11.5)
    b.add_argument(
        "--data-repo",
        type=Path,
        default=REPO,
        help="directory holding data/qi_nae_db/ (see docs/external-data.md); "
        "defaults to this repository's own root",
    )
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--workers", type=int, default=2)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--replace-failures", action="store_true")
    b.add_argument("--min-rmin-ratio", type=float, default=MIN_RMIN_RATIO)
    b.add_argument("--aspect-by", choices=("estimate", "helper"), default="estimate")

    pl = sub.add_parser("placement", help="D1 minimum-distance placement of confirmed boundaries")
    pl.add_argument("--boundary-dir", type=Path, required=True)
    pl.add_argument("--labels", type=str, required=True)
    pl.add_argument("--atlas-dir", type=Path, default=None)
    pl.add_argument("--board-dir", type=Path, default=None)
    pl.add_argument("--reference-boundary", type=Path, default=None)
    pl.add_argument("--out", type=Path, required=True)
    return p


def cmd_build(args: argparse.Namespace) -> None:
    selection = json.loads(args.selection.read_text())
    entries = selection["seeds"]
    if args.limit is not None:
        entries = entries[: args.limit]
    spares = None
    if args.selection_extra is not None:
        spares = json.loads(args.selection_extra.read_text())["seeds"]
    manifest = build_batch(
        entries,
        mag_well=args.mag_well,
        min_rmin_ratio=args.min_rmin_ratio,
        aspect_by=args.aspect_by,
        aspect=args.aspect,
        data_repo=args.data_repo,
        out_dir=args.out_dir,
        workers=args.workers,
        replace_failures=args.replace_failures,
        spare_entries=spares,
    )
    print(f"built {manifest['n_success']}/{manifest['n_requested']} -> {args.out_dir}")


def cmd_placement(args: argparse.Namespace) -> None:
    labels = args.labels.split(",")
    reference_boundary = load_reference_boundary(args.reference_boundary)
    report = placement_report(
        args.boundary_dir,
        labels,
        load_leaders_fn=lambda: _default_load_leaders(args.atlas_dir, args.board_dir),
        reference_boundary=reference_boundary,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.cmd == "build":
        cmd_build(args)
    else:
        cmd_placement(args)


if __name__ == "__main__":
    main()
