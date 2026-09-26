#!/usr/bin/env bash
# Snapshot a pre-registration note into the repo (dated, committed) so a
# lock's pre-launch immutability is checkable from git history.
# Usage: scripts/prereg_snapshot.sh <label> <vault-note-path>
set -euo pipefail
label="${1:?label, e.g. E12-lock}"
src="${2:?path to the pre-registration note to snapshot, e.g. ~/Documents/.../p3-reopen-three-experiments.md}"
dst="docs/preregistration/$(date -u +%Y%m%dT%H%M%SZ)-${label}.md"
cp "$src" "$dst"
sha=$(sha256sum "$dst" | cut -c1-16)
echo "snapshot $dst sha256 $sha"
git add "$dst"
git commit -q -m "prereg: snapshot ${label} (sha256 ${sha})" -- "$dst"
git log --oneline -1
