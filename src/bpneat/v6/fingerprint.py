"""v6's own fingerprint set, disjoint from v2's, v3's, v4's and v5's.

Five fingerprints go into every v6 record. v6 may edit none of the four older
sets — its ``search`` arm is v5's search module imported unchanged, its ``fixed``
arm is v3's matched multistart, and both rest on v2's frozen primitives — so
recording all of them lets a reader check they really were frozen while v6 ran,
and lets the release firewall refuse a sealed-test pass if any has moved.

v6 has no search module of its own. That is the point of the design: the thing
under study is the budget, so the searcher is the released one, byte for byte.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..record import code_fingerprint as v2_code_fingerprint
from ..record import sha256_file
from ..v3.fingerprint import v3_fingerprint
from ..v4.fingerprint import v4_fingerprint
from ..v5.fingerprint import v5_fingerprint

#: Modules that determine a v6 run's output.
V6_SCIENCE_MODULES = ("conditions.py", "protocol.py")


def v6_fingerprint() -> dict:
    root = Path(__file__).resolve().parent
    per_file = {name: sha256_file(root / name) for name in V6_SCIENCE_MODULES}
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(per_file.items())).encode()
    ).hexdigest()
    return {"files": per_file, "combined": combined}


def fingerprints() -> dict:
    return {
        "v6": v6_fingerprint()["combined"],
        "v5_frozen": v5_fingerprint()["combined"],
        "v4_frozen": v4_fingerprint()["combined"],
        "v3_frozen": v3_fingerprint()["combined"],
        "v2_frozen": v2_code_fingerprint()["combined"],
        "v6_modules": list(V6_SCIENCE_MODULES),
    }
