#!/usr/bin/env bash
# CI-style leak check: fail the build if any tracked file mentions a private
# client/project name or a personal path/email that should never appear in
# this public repository. Run from the repo root.
set -euo pipefail

PATTERN='Radek|Klarrio|Kinly|bossyk|narwhal|aissured|simmons|auditk|/home/matt|@gmail'

cd "$(git rev-parse --show-toplevel)"

hits=$(git grep -niE "$PATTERN" -- . ':!tools/check_leaks.sh' || true)

if [ -n "$hits" ]; then
    echo "LEAK CHECK FAILED -- matched pattern: $PATTERN" >&2
    echo "$hits" >&2
    exit 1
fi

echo "OK: no leak-pattern hits in tracked files"
