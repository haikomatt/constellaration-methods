import json

from strip_boundaries import BOUNDARY_KEYS, strip_boundaries, strip_file


def test_strip_boundaries_removes_top_level_boundary_key():
    data = {"label": "E13", "boundary": {"r_cos": [[1.0]], "z_sin": [[0.0]]}}

    result = strip_boundaries(data)

    assert result == {"label": "E13"}


def test_strip_boundaries_removes_all_banned_keys():
    data = {k: "value" for k in BOUNDARY_KEYS}
    data["kept"] = "value"

    result = strip_boundaries(data)

    assert result == {"kept": "value"}


def test_strip_boundaries_recurses_into_nested_lists_and_dicts():
    data = {
        "members": [
            {"tag": "G0:26", "archived_r": [1, 2, 3], "boundary_dict": {"r_cos": [[1.0]]}},
            {"tag": "G0:25", "x": [0.1, 0.2]},
        ]
    }

    result = strip_boundaries(data)

    assert result == {
        "members": [
            {"tag": "G0:26", "archived_r": [1, 2, 3]},
            {"tag": "G0:25"},
        ]
    }


def test_strip_boundaries_preserves_non_boundary_numeric_fields():
    data = {"confirmed_hypervolume": 124.23, "n_front_members": 25, "r": [0.01, -0.02]}

    result = strip_boundaries(data)

    # "r" (the residual vector) is not in BOUNDARY_KEYS -- only the
    # boundary-coordinate-carrying keys are stripped.
    assert result == data


def test_strip_boundaries_does_not_mutate_the_input():
    data = {"boundary": {"r_cos": [[1.0]]}, "kept": 1}

    strip_boundaries(data)

    assert "boundary" in data  # original untouched


def test_strip_file_writes_a_stripped_copy(tmp_path):
    src = tmp_path / "REPORT.json"
    src.write_text(json.dumps({"label": "E15", "boundary": {"r_cos": [[1.0]]}}))
    dst = tmp_path / "out" / "REPORT.json"

    strip_file(src, dst)

    written = json.loads(dst.read_text())
    assert written == {"label": "E15"}


def test_strip_file_creates_parent_directories(tmp_path):
    src = tmp_path / "REPORT.json"
    src.write_text(json.dumps({"a": 1}))
    dst = tmp_path / "a" / "b" / "c" / "REPORT.json"

    strip_file(src, dst)

    assert dst.exists()
