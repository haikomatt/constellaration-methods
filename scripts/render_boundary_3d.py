#!/usr/bin/env python3
"""Thin CLI wrapper -- see `constellaration_methods.analysis.render_boundary_3d` for the
implementation. Runnable without installing the package
(`python scripts/render_boundary_3d.py ...`); the same entry point is also installed as
a console script (see pyproject.toml) once the package is pip-installed.
"""

from constellaration_methods.analysis.render_boundary_3d import main

if __name__ == "__main__":
    main()
