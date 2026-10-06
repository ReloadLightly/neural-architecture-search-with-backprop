"""v5's own fingerprint set, disjoint from v2's, v3's and v4's.

Four fingerprints go into every v5 record. v5 may edit none of the three older
sets, so recording all of them lets a reader check they really were frozen while
v5 ran, and lets the release firewall refuse a sealed-test pass if any has moved.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..record import code_fingerprint as v2_code_fingerprint
from ..record import sha256_file
from ..v3.fingerprint import v3_fingerprint
from ..v4.fingerprint import v4_fingerprint

#: Modules that determine a v5 run's output.
V5_SCIENCE_MODULES = ("conditions.py", "protocol.py", "search.py")


def v5_fingerprint() -> dict:
    root = Path(__file__).resolve().parent
    per_file = {name: sha256_file(root / name) for name in V5_SCIENCE_MODULES}
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(per_file.items())).encode()
    ).hexdigest()
    return {"files": per_file, "combined": combined}


def fingerprints() -> dict:
    return {
        "v5": v5_fingerprint()["combined"],
        "v4_frozen": v4_fingerprint()["combined"],
        "v3_frozen": v3_fingerprint()["combined"],
        "v2_frozen": v2_code_fingerprint()["combined"],
        "v5_modules": list(V5_SCIENCE_MODULES),
    }
