#!/usr/bin/env python3
"""Thin CLI wrapper -- see `constellaration_methods.fronts.b3_nsga2_mhd` for the
implementation. Runnable without installing the package
(`python scripts/b3_nsga2_mhd.py ...`); the same entry point is also installed as
a console script (see pyproject.toml) once the package is pip-installed.
"""

from constellaration_methods.fronts.b3_nsga2_mhd import main

if __name__ == "__main__":
    main()
