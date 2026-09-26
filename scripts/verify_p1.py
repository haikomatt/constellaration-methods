#!/usr/bin/env python3
"""Thin CLI wrapper -- see `constellaration_methods.evaluation.verify_p1` for the
implementation. Runnable without installing the package
(`python scripts/verify_p1.py ...`); the same entry point is also installed as
a console script (see pyproject.toml) once the package is pip-installed.
"""

from constellaration_methods.evaluation.verify_p1 import main

if __name__ == "__main__":
    main()
