"""tools/ is a flat script directory, not an installed package -- add it to
sys.path for these tests only, mirroring how the scripts themselves resolve
sibling imports when run directly (`python tools/check_no_boundaries.py`,
which gets its own directory on sys.path[0] for free)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
