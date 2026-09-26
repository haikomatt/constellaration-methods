#!/usr/bin/env python3
"""Thin CLI wrapper -- see `constellaration_methods.evaluation.score_mhd_stable` for the
implementation. Runnable without installing the package
(`python scripts/score_mhd_stable.py ...`); the same entry point is also installed as
a console script (see pyproject.toml) once the package is pip-installed.
"""

from constellaration_methods.evaluation.score_mhd_stable import main

if __name__ == "__main__":
    main()
