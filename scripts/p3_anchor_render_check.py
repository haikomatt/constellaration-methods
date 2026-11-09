#!/usr/bin/env python3
"""Thin CLI wrapper -- see `constellaration_methods.fronts.p3_anchor_render_check` for the
implementation. Runnable without installing the package
(`python scripts/p3_anchor_render_check.py ...`); the same entry point is also installed as
a console script (see pyproject.toml) once the package is pip-installed.
"""

from constellaration_methods.fronts.p3_anchor_render_check import cli

if __name__ == "__main__":
    cli()
