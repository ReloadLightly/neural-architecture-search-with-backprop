"""The n-dimensional core's own fingerprint, and the frozen one it rests on.

The nd core is not a protocol and makes no claim. It is the encoding a protocol
may be built on, so what it records is: its own hash, and the hash of the frozen
v2 modules it generalises. A protocol built on it records both, and a reader can
then check that the frozen core had not moved while the general one was written.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..record import code_fingerprint as v2_code_fingerprint
from ..record import sha256_file

#: Modules that decide what a graph here computes.
ND_SCIENCE_MODULES = ("encoding.py",)


def nd_fingerprint() -> dict:
    root = Path(__file__).resolve().parent
    per_file = {name: sha256_file(root / name) for name in ND_SCIENCE_MODULES}
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(per_file.items())).encode()
    ).hexdigest()
    return {"files": per_file, "combined": combined}


def fingerprints() -> dict:
    return {
        "nd": nd_fingerprint()["combined"],
        "v2_frozen": v2_code_fingerprint()["combined"],
        "nd_modules": list(ND_SCIENCE_MODULES),
    }
