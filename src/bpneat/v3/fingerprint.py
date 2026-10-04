"""v3's own fingerprint set, disjoint from v2's.

v2 binds seven modules to its release (``cfdf1fa3…``). v3 may not edit any of
them, so it declares its own set: the modules that decide what a v3 run
computes. Both fingerprints are recorded in every v3 record, so a reader can
check that the frozen v2 code really was frozen while v3 ran.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..record import SCIENCE_MODULES as V2_SCIENCE_MODULES
from ..record import code_fingerprint as v2_code_fingerprint
from ..record import sha256_file

#: Modules that determine a v3 run's output. Suite, analysis, figure and CLI
#: code is excluded for the same reason it is excluded from v2's set: it cannot
#: change a run's result, and including it makes the fingerprint move whenever
#: a plot is added.
V3_SCIENCE_MODULES = (
    "conditions.py",
    "datasets.py",
    "learners.py",
    "dense.py",
    "protocol.py",
    "search.py",
    "selection.py",
)


def v3_fingerprint() -> dict:
    root = Path(__file__).resolve().parent
    per_file = {name: sha256_file(root / name) for name in V3_SCIENCE_MODULES}
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(per_file.items())).encode()
    ).hexdigest()
    return {"files": per_file, "combined": combined}


def fingerprints() -> dict:
    """Both fingerprints: v3's own, and the frozen v2 set it builds on."""
    return {
        "v3": v3_fingerprint()["combined"],
        "v2_frozen": v2_code_fingerprint()["combined"],
        "v2_modules": list(V2_SCIENCE_MODULES),
        "v3_modules": list(V3_SCIENCE_MODULES),
    }
