FROM python:3.10-slim

# git: several modules stamp REPORT.json provenance with the running repo's
# commit by reading .git directly (no subprocess, see
# analysis/p3_design_diversity_probe.py's _git_commit docstring) -- no git
# *binary* is required for that, but tests that exercise it against this
# repo's own real .git directory need the mounted checkout to be a normal
# git repository (it always is; this comment documents why .git is not
# excluded from the build/run context, not a runtime dependency on git(1)).
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pinned to the exact versions every result under docs/results/ was produced
# with. constellaration==0.2.6 transitively pins vmecpp==0.4.11 and
# scikit-learn==1.5.2; the explicit scikit-learn pin is defensive so a future
# constellaration bump cannot silently drag it along (see pyproject.toml).
COPY pyproject.toml README.md ./
COPY src/ src/

# Editable install: several modules resolve their own repo root from
# __file__ (e.g. to read .git for provenance stamping, or to default
# --out paths under results/). A non-editable install copies the package
# into site-packages, disconnected from the checkout mounted at /workspace,
# which breaks that resolution. Editable keeps __file__ pointing at
# /workspace/src/... , matching the checkout bind-mounted here at run time.
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

CMD ["pytest", "tests/", "-q"]
