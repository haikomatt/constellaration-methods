#!/usr/bin/env python3
"""Anchor render check (E28 guard, 2026-10-26): every anchor in an anchors REPORT must be
reproduced by its verbatim generation-0 evaluation in the front builder's archive.

E25 seeded an anchor's exact x into a run whose template had a different fixed major radius,
so the same x was evaluated as a different shape (aspect 7.49 instead of 7.37) and nothing
flagged it. This compares, for each anchor, its stored confirmed residuals with the archive
row whose x equals the anchor's x.

    python scripts/p3_anchor_render_check.py ANCHORS_REPORT.json ARCHIVE.jsonl [--tol 1e-4]

Exit 0: every anchor found and matched. Exit 1: a mismatch (printed as MISMATCH). Exit 2: an
anchor not yet in the archive (printed as PENDING; the caller waits and retries).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a partly written last line
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument(
        "--source-archive",
        type=Path,
        default=None,
        help=(
            "archive of the run whose confirmed front is the anchors REPORT: "
            "members carry a tag but no x"
        ),
    )
    a = ap.parse_args()
    members = [m for m in json.loads(a.report.read_text())["members"] if m.get("retained")]
    rows = _read_jsonl(a.archive)
    source_x = {}
    if a.source_archive is not None:
        source_x = {r["tag"]: r["x"] for r in _read_jsonl(a.source_archive) if r.get("x")}
    status = 0
    for m in members:
        x = m.get("x") or source_x.get(m.get("tag"))
        if x is None:
            print(f"MISMATCH {m.get('tag', '<untagged>')}: no coordinates to match")
            status = 1
            continue
        m = dict(m, x=x)
        hit = next(
            (
                r
                for r in rows
                if r.get("x") is not None
                and len(r["x"]) == len(m["x"])
                and max(abs(p - q) for p, q in zip(r["x"], m["x"], strict=True)) <= 1e-12
            ),
            None,
        )
        if hit is None:
            print(f"PENDING {m.get('tag', '<untagged>')}: no archive row with its x yet")
            status = max(status, 2) if status != 1 else 1
            continue
        if hit.get("r") is None:
            print(
                f"MISMATCH {m.get('tag', '<untagged>')}: its evaluation "
                f"{hit['tag']} did not converge"
            )
            status = 1
            continue
        d = max(abs(p - q) for p, q in zip(hit["r"], m["confirmed_r"], strict=True))
        if d > a.tol:
            print(
                f"MISMATCH {m.get('tag', '<untagged>')}: {hit['tag']} residuals differ "
                f"by {d:.3g} (> {a.tol})"
            )
            status = 1
        else:
            print(f"OK {m.get('tag', '<untagged>')}: {hit['tag']} reproduces it (max diff {d:.2g})")
    return status


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
