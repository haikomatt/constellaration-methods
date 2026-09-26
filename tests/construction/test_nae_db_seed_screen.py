"""E21 (generator route A', second pass): `construction.nae_db_seed_screen`
-- the database rebuild+completion+export driver (`build_batch`) and the D1
placement report (`placement_report`), plus the external-data guards.

Host-testable throughout: `build_batch` takes an injected `rebuild_fn` (no
pyQIC fork or database needed) and `placement_report` takes an injected
`load_leaders_fn` (no atlas/board load needed). `inv_normalise_rc` and
`inv_normalise_vpp` are checked against a closed-form flat-B0 case -- no
`qic.Qic` import needed. `_require_qic` and `rebuild_and_export`'s missing-
data path are checked without either the fork or the database installed."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from constellaration_methods.construction import nae_db_seed_screen as gen

# --------------------------------------------------------------------------- #
# inv_normalise_rc / inv_normalise_vpp: verbatim copies of the authors'
# helpers (CC BY 4.0) -- checked against the closed form for flat B0 = 1,
# G0 = 1.
# --------------------------------------------------------------------------- #


def _flat_stel(nfp: int = 2, nphi: int = 32) -> SimpleNamespace:
    return SimpleNamespace(
        varphi=np.linspace(0.0, 2 * np.pi / nfp, nphi, endpoint=False),
        nfp=nfp,
        G0=1.0,
        Bbar=1.0,
        B0=np.full(nphi, 1.0),
    )


def test_inv_normalise_rc_matches_closed_form_for_flat_b0():
    stel = _flat_stel()
    # R = G0/(2 pi) * nfp * trapz(1/B0, varphi) = 1/(2 pi) * 2 * 2 pi = 1.0
    r = gen.inv_normalise_rc(stel, aspect=10.0)
    assert r == pytest.approx(1.0 / 10.0, rel=1e-6)


def test_inv_normalise_rc_scales_inversely_with_aspect():
    stel = _flat_stel()
    r10 = gen.inv_normalise_rc(stel, aspect=10.0)
    r20 = gen.inv_normalise_rc(stel, aspect=20.0)
    assert r20 == pytest.approx(r10 / 2.0, rel=1e-9)


def test_inv_normalise_vpp_is_linear_in_w_and_signed():
    stel = _flat_stel()
    vpp_pos = gen.inv_normalise_vpp(stel, 0.02)
    vpp_neg = gen.inv_normalise_vpp(stel, -0.02)
    vpp_zero = gen.inv_normalise_vpp(stel, 0.0)
    assert vpp_zero == pytest.approx(0.0, abs=1e-12)
    assert vpp_pos == pytest.approx(-vpp_neg, rel=1e-9)


# --------------------------------------------------------------------------- #
# surface_rmin_over_r00: the E21 geometry gate
# --------------------------------------------------------------------------- #


def test_surface_rmin_over_r00_gate_values():
    """A circular-cross-section torus of major radius 1 and minor radius
    0.1 has rmin/r00 = 0.9; a shifted (n=1) axis excursion of amplitude 0.6
    drops it below the 0.5 gate."""
    rc = [[0.0, 1.0, 0.0], [0.0, 0.1, 0.0]]  # ntor=1: columns n=-1,0,1
    assert abs(gen.surface_rmin_over_r00(rc, 2) - 0.9) < 1e-6
    rc2 = [[0.0, 1.0, 0.6], [0.0, 0.1, 0.0]]
    assert gen.surface_rmin_over_r00(rc2, 2) < gen.MIN_RMIN_RATIO


# --------------------------------------------------------------------------- #
# external-data guards: clear, actionable errors with neither the database
# nor the pyQIC fork present -- never a bare traceback.
# --------------------------------------------------------------------------- #


class TestExternalDataGuards:
    def test_require_qic_raises_clear_error_when_not_importable(self):
        def boom():
            raise ImportError("no module named qic")

        with pytest.raises(RuntimeError, match="docs/external-data.md"):
            gen._require_qic(import_fn=boom)

    def test_require_qic_raises_clear_error_when_not_the_fork(self):
        class VanillaQic:
            pass  # no mag_well_reshape

        with pytest.raises(RuntimeError, match="not the authors' fork"):
            gen._require_qic(import_fn=lambda: VanillaQic)

    def test_require_qic_accepts_fork_shaped_class(self):
        class ForkQic:
            @classmethod
            def mag_well_reshape(cls, stel, well):
                return stel

        assert gen._require_qic(import_fn=lambda: ForkQic) is ForkQic

    def test_rebuild_and_export_raises_file_not_found_before_importing_qic(self, tmp_path):
        """The database-missing check happens before any pyQIC import, so
        this is deterministic on a host with no fork installed at all."""
        entry = {
            "seed_name": "dbMissing",
            "N": 2,
            "db_file": "data/qi_nae_db/does_not_exist.json",
            "ID": 1,
            "iota": 0.5,
        }
        with pytest.raises(FileNotFoundError, match="docs/external-data.md"):
            gen.rebuild_and_export(entry, mag_well=0.02, aspect=11.5, data_repo=tmp_path)

    def test_load_raw_config_raises_file_not_found_with_pointer(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="docs/external-data.md"):
            gen.load_raw_config(tmp_path / "nope.json", 0)


# --------------------------------------------------------------------------- #
# build_batch: injected rebuild_fn, manifest writing, --replace-failures
# --------------------------------------------------------------------------- #


def _entry(seed_name: str, *, n: int = 2, id_: int = 1) -> dict:
    return {
        "seed_name": seed_name,
        "N": n,
        "db_file": f"data/qi_nae_db/{seed_name}.json",
        "ID": id_,
        "iota": 0.66,
        "mirr_ratio": 0.15,
        "features": {"ρc_0": 4.0},
    }


def _ok_rebuild_fn(entry, *, mag_well, aspect, data_repo):
    return {
        "seed_name": entry["seed_name"],
        "n_field_periods": entry["N"],
        "db_file": entry["db_file"],
        "ID": entry["ID"],
        "features": entry.get("features", {}),
        "W_target": mag_well,
        "vpp_target": 1.0,
        "d2_volume_d_psi2_before": 500.0,
        "d2_volume_d_psi2_after": 0.0,
        "near_axis_iota_over_nfp": 0.3,
        "mirr_ratio": entry.get("mirr_ratio"),
        "L_grad_B": 0.3,
        "delta_B": 0.1,
        "rho_c0": entry.get("features", {}).get("ρc_0"),
        "aspect_requested": aspect,
        "aspect_estimate": aspect - 0.1,
        "minor_radius": 0.1,
        "elapsed_s": 0.01,
        "fork_commit": gen.FORK_COMMIT,
        "boundary": {
            "r_cos": [[0.0, 1.0, 0.0]],
            "z_sin": [[0.0, 0.0, 0.0]],
            "n_field_periods": entry["N"],
            "is_stellarator_symmetric": True,
        },
    }


def _failing_rebuild_fn(entry, *, mag_well, aspect, data_repo):
    raise ValueError(f"synthetic failure for {entry['seed_name']}")


class TestBuildBatch:
    def test_writes_one_boundary_file_per_success(self, tmp_path):
        entries = [_entry(f"seed{i}") for i in range(3)]
        out_dir = tmp_path / "out"
        manifest = gen.build_batch(
            entries,
            mag_well=0.02,
            aspect=11.5,
            data_repo=tmp_path,
            out_dir=out_dir,
            rebuild_fn=_ok_rebuild_fn,
            workers=1,
        )
        assert manifest["n_success"] == 3
        assert manifest["n_requested"] == 3
        for i in range(3):
            p = out_dir / "boundaries" / f"seed{i}.boundary.json"
            assert p.exists()
            b = json.loads(p.read_text())
            assert b["n_field_periods"] == 2

    def test_manifest_json_written_and_carries_required_fields(self, tmp_path):
        entries = [_entry("seedA")]
        out_dir = tmp_path / "out"
        gen.build_batch(
            entries,
            mag_well=0.02,
            aspect=11.5,
            data_repo=tmp_path,
            out_dir=out_dir,
            rebuild_fn=_ok_rebuild_fn,
            workers=1,
        )
        manifest = json.loads((out_dir / "manifest.json").read_text())
        row = manifest["seeds"][0]
        for key in (
            "seed_name",
            "n_field_periods",
            "db_file",
            "ID",
            "features",
            "W_target",
            "vpp_target",
            "d2_volume_d_psi2_before",
            "d2_volume_d_psi2_after",
            "near_axis_iota_over_nfp",
            "mirr_ratio",
            "L_grad_B",
            "delta_B",
            "rho_c0",
            "aspect_requested",
            "aspect_estimate",
            "minor_radius",
            "elapsed_s",
            "fork_commit",
            "file",
        ):
            assert key in row, f"missing manifest field {key!r}"
        assert "boundary" not in row  # boundary goes to its own file, not the manifest

    def test_failure_is_recorded_never_dropped(self, tmp_path):
        entries = [_entry("bad_seed")]
        out_dir = tmp_path / "out"
        manifest = gen.build_batch(
            entries,
            mag_well=0.02,
            aspect=11.5,
            data_repo=tmp_path,
            out_dir=out_dir,
            rebuild_fn=_failing_rebuild_fn,
            workers=1,
        )
        assert manifest["n_success"] == 0
        row = manifest["seeds"][0]
        assert row["seed_name"] == "bad_seed"
        assert "ValueError" in row["error"]

    def test_replace_failures_pulls_from_spares(self, tmp_path):
        def flaky(entry, *, mag_well, aspect, data_repo):
            if entry["seed_name"] == "primary":
                raise ValueError("primary always fails")
            return _ok_rebuild_fn(entry, mag_well=mag_well, aspect=aspect, data_repo=data_repo)

        entries = [_entry("primary")]
        spares = [_entry("spare1"), _entry("spare2")]
        out_dir = tmp_path / "out"
        manifest = gen.build_batch(
            entries,
            mag_well=0.02,
            aspect=11.5,
            data_repo=tmp_path,
            out_dir=out_dir,
            rebuild_fn=flaky,
            workers=1,
            replace_failures=True,
            spare_entries=spares,
        )
        assert manifest["n_requested"] == 1
        assert manifest["n_success"] == 1
        assert manifest["replacements"] == [{"failed": "primary", "replacement": "spare1"}]
        names_written = {
            p.stem.replace(".boundary", "") for p in (out_dir / "boundaries").glob("*.json")
        }
        assert names_written == {"spare1"}

    def test_replace_failures_off_leaves_gap(self, tmp_path):
        entries = [_entry("primary")]
        out_dir = tmp_path / "out"
        manifest = gen.build_batch(
            entries,
            mag_well=0.02,
            aspect=11.5,
            data_repo=tmp_path,
            out_dir=out_dir,
            rebuild_fn=_failing_rebuild_fn,
            workers=1,
            replace_failures=False,
            spare_entries=[_entry("spare1")],
        )
        assert manifest["n_success"] == 0
        assert manifest["replacements"] == []


def test_rebuild_worker_records_failure_instead_of_raising(monkeypatch):
    """A missing sweep file (or any other rebuild failure) must not raise
    out of a spawned pool worker and kill the whole batch -- it is
    recorded like the sequential path does."""

    def boom(entry, **kw):
        raise FileNotFoundError("no such sweep file")

    monkeypatch.setattr(gen, "rebuild_and_export", boom)
    rec = gen._rebuild_worker(({"seed_name": "dbX"}, 0.02, 11.5, Path("/nowhere"), 0.5, "estimate"))
    assert rec["seed_name"] == "dbX" and rec["error"].startswith("FileNotFoundError")


# --------------------------------------------------------------------------- #
# placement: D1 min-distance report, injected leaders loader
# --------------------------------------------------------------------------- #


def _torus_boundary(nfp: int, aspect: float) -> dict:
    a = 1.0 / aspect
    return {
        "r_cos": [[0.0, 1.0, 0.0], [0.0, a, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, a, 0.0]],
        "n_field_periods": nfp,
    }


class TestPlacementReport:
    def test_reports_min_and_median_to_leaders_and_reference(self, tmp_path):
        boundary_dir = tmp_path / "boundaries"
        boundary_dir.mkdir()
        labels = ["a", "b"]
        for i, label in enumerate(labels):
            (boundary_dir / f"{label}.boundary.json").write_text(
                json.dumps(_torus_boundary(nfp=3, aspect=9.0 + i))
            )
        leaders = [
            _torus_boundary(nfp=3, aspect=9.5),
            _torus_boundary(nfp=4, aspect=9.5),
        ]
        reference = _torus_boundary(nfp=3, aspect=9.2)
        report = gen.placement_report(
            boundary_dir,
            labels,
            load_leaders_fn=lambda: leaders,
            reference_boundary=reference,
        )
        assert report["n_labels"] == 2
        assert report["d_rand"] == pytest.approx(gen.D_RAND)
        assert report["leaders"]["n"] == 2
        assert report["leaders"]["min"] >= 0.0
        assert report["leaders"]["median"] >= report["leaders"]["min"]
        assert report["reference"]["n"] == 2
        assert {r["label"] for r in report["per_label"]} == set(labels)

    def test_reference_none_reports_zero_n(self, tmp_path):
        boundary_dir = tmp_path / "boundaries"
        boundary_dir.mkdir()
        (boundary_dir / "a.boundary.json").write_text(
            json.dumps(_torus_boundary(nfp=3, aspect=9.0))
        )
        report = gen.placement_report(
            boundary_dir,
            ["a"],
            load_leaders_fn=lambda: [_torus_boundary(nfp=3, aspect=9.5)],
            reference_boundary=None,
        )
        assert report["reference"]["n"] == 0

    def test_no_same_nfp_leaders_reports_zero_n(self, tmp_path):
        boundary_dir = tmp_path / "boundaries"
        boundary_dir.mkdir()
        (boundary_dir / "a.boundary.json").write_text(
            json.dumps(_torus_boundary(nfp=3, aspect=9.0))
        )
        report = gen.placement_report(
            boundary_dir,
            ["a"],
            load_leaders_fn=lambda: [_torus_boundary(nfp=4, aspect=9.5)],
            reference_boundary=None,
        )
        assert report["leaders"]["n"] == 0


def test_load_reference_boundary_none_when_path_none():
    assert gen.load_reference_boundary(None) is None


def test_load_reference_boundary_none_when_missing(tmp_path):
    assert gen.load_reference_boundary(tmp_path / "nope.json") is None


def test_load_reference_boundary_loads_existing_file(tmp_path):
    p = tmp_path / "ref.boundary.json"
    p.write_text(json.dumps(_torus_boundary(nfp=3, aspect=9.0)))
    ref = gen.load_reference_boundary(p)
    assert ref["n_field_periods"] == 3


def test_default_load_leaders_empty_when_dirs_unset():
    assert gen._default_load_leaders(None, None) == []


def test_default_load_leaders_empty_when_dirs_missing(tmp_path):
    assert gen._default_load_leaders(tmp_path / "no-atlas", tmp_path / "no-board") == []


# --------------------------------------------------------------------------- #
# CLI parsing (both subcommands)
# --------------------------------------------------------------------------- #


def test_cli_build_parses_flags():
    parser = gen.build_arg_parser()
    args = parser.parse_args(
        [
            "build",
            "--selection",
            "selection_p3.json",
            "--out-dir",
            "results/p3-e21-nae-db",
            "--mag-well",
            "0.02",
            "--aspect",
            "11.5",
            "--workers",
            "2",
        ]
    )
    assert args.cmd == "build"
    assert args.mag_well == pytest.approx(0.02)
    assert args.aspect == pytest.approx(11.5)
    assert args.workers == 2
    assert args.replace_failures is False


def test_cli_placement_parses_flags():
    parser = gen.build_arg_parser()
    args = parser.parse_args(
        [
            "placement",
            "--boundary-dir",
            "results/p3-e21/boundaries",
            "--labels",
            "a,b,c",
            "--out",
            "results/p3-e21-placement/REPORT.json",
        ]
    )
    assert args.cmd == "placement"
    assert args.labels == "a,b,c"
    assert args.atlas_dir is None
    assert args.board_dir is None
    assert args.reference_boundary is None
