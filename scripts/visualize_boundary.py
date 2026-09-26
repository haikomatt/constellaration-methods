#!/usr/bin/env python3
"""Thin CLI wrapper -- see `constellaration_methods.analysis.visualize_boundary` for the
implementation. Runnable without installing the package
(`python scripts/visualize_boundary.py ...`); the same entry point is also installed as
a console script (see pyproject.toml) once the package is pip-installed.
"""

from constellaration_methods.analysis.visualize_boundary import main

if __name__ == "__main__":
    main()
