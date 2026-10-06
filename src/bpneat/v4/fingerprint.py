"""v4's own fingerprint set, disjoint from v2's and v3's.

Three fingerprints go into every v4 record: v4's own science modules, the v3
set (``8438c9e8…`` as released), and the frozen v2 set (``cfdf1fa3…``). v4 may
not edit either of the older sets, so recording all three lets a reader check
that both really were frozen while v4 ran — and lets the release firewall refuse
a sealed-test pass if any of the three has moved.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..record import SCIENCE_MODULES as V2_SCIENCE_MODULES
from ..record import code_fingerprint as v2_code_fingerprint
from ..record import sha256_file
from ..v3.fingerprint import V3_SCIENCE_MODULES, v3_fingerprint

#: Modules that determine a v4 run's output. Suite, analysis, figure and CLI
#: code is excluded for the same reason it is excluded from v2's and v3's sets:
#: it cannot change a run's result, and including it makes the fingerprint move
#: whenever a plot is added.
V4_SCIENCE_MODULES = (
    "cgp.py",
    "conditions.py",
    "learners.py",
    "protocol.py",
    "search.py",
)


def v4_fingerprint() -> dict:
    root = Path(__file__).resolve().parent
    per_file = {name: sha256_file(root / name) for name in V4_SCIENCE_MODULES}
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(per_file.items())).encode()
    ).hexdigest()
    return {"files": per_file, "combined": combined}


def fingerprints() -> dict:
    """All three: v4's own, the released v3 set, and the frozen v2 set."""
    return {
        "v4": v4_fingerprint()["combined"],
        "v3_frozen": v3_fingerprint()["combined"],
        "v2_frozen": v2_code_fingerprint()["combined"],
        "v2_modules": list(V2_SCIENCE_MODULES),
        "v3_modules": list(V3_SCIENCE_MODULES),
        "v4_modules": list(V4_SCIENCE_MODULES),
    }
