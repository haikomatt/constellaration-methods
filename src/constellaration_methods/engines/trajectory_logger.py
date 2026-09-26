"""Build step 4: NDJSON trajectory logger with run-level metadata.

One file per run. First line is run metadata; every subsequent line
(including the step-0 seed candidates) is one JSON step record. Canonical
(`default_high_fidelity`) score fields default to null here and are filled
in by the mandatory re-score pass (build step 6) -- per the plan's Phase 2
judging rule: drift is judged against agent-visible scores, outcome quality
against canonical scores, and one must never overwrite the other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

_CANONICAL_FIELD_DEFAULTS = {
    "canonical_objective_value": None,
    "canonical_feasible": None,
    "canonical_feasibility_margin": None,
    "canonical_non_convergent": None,
}


class TrajectoryLogger:
    def __init__(self, path: Path, run_metadata: dict[str, Any]):
        self.path = Path(path)
        self._file: TextIO = open(self.path, "w")
        self._write({"type": "run_metadata", **run_metadata})

    def log_step(self, step_record: dict[str, Any]) -> None:
        record = dict(_CANONICAL_FIELD_DEFAULTS)
        record.update(step_record)
        self._write({"type": "step", **record})

    def _write(self, obj: dict[str, Any]) -> None:
        self._file.write(json.dumps(obj, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> TrajectoryLogger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_trajectory(path: Path) -> list[dict[str, Any]]:
    """Reads an NDJSON trajectory file back into a list of dicts, in order
    (run_metadata first, then steps)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
