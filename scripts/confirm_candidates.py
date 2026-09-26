#!/usr/bin/env python3
"""Thin CLI wrapper -- see `constellaration_methods.evaluation.confirm_candidates` for the
implementation. Runnable without installing the package
(`python scripts/confirm_candidates.py ...`); the same entry point is also installed as
a console script (see pyproject.toml) once the package is pip-installed.
"""

from constellaration_methods.evaluation.confirm_candidates import main

if __name__ == "__main__":
    main()
