"""v8's fingerprint set: its own, the n-dimensional core's, and the frozen ones.

v8 is the first protocol built on :mod:`bpneat.nd` rather than directly on the
frozen modules, so a record has to pin both: the encoding that decides what a
graph computes at any width, and the frozen set that encoding is proved
byte-identical to at two inputs. A reader checking a v8 release can then confirm
that neither had moved while it ran.

The data is fingerprinted too, and not by hashing the loader. The datasets are
committed CSVs, and what matters is the bytes of those files — a release built
on a dataset that later changed would otherwise look sound.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..nd.datasets import DATA_DIR
from ..nd.fingerprint import nd_fingerprint
from ..record import code_fingerprint as v2_code_fingerprint
from ..record import sha256_file
from ..v3.fingerprint import v3_fingerprint
from ..v5.fingerprint import v5_fingerprint

#: Modules that determine a v8 run's output.
V8_SCIENCE_MODULES = ("conditions.py", "protocol.py")


def v8_fingerprint() -> dict:
    root = Path(__file__).resolve().parent
    per_file = {name: sha256_file(root / name) for name in V8_SCIENCE_MODULES}
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(per_file.items())).encode()
    ).hexdigest()
    return {"files": per_file, "combined": combined}


def data_fingerprint() -> dict:
    """The committed datasets themselves, by content, not by loader."""
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    per_file = {d["name"]: d["sha256"] for d in manifest["datasets"]}
    for name, recorded in per_file.items():
        live = sha256_file(DATA_DIR / f"{name}.csv")
        if live != recorded:
            raise AssertionError(
                f"{name}.csv does not match the manifest: the data has changed "
                "since it was vendored"
            )
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(per_file.items())).encode()
    ).hexdigest()
    return {"files": per_file, "combined": combined}


def fingerprints() -> dict:
    return {
        "v8": v8_fingerprint()["combined"],
        "data": data_fingerprint()["combined"],
        "nd": nd_fingerprint()["combined"],
        "v5_frozen": v5_fingerprint()["combined"],
        "v3_frozen": v3_fingerprint()["combined"],
        "v2_frozen": v2_code_fingerprint()["combined"],
        "v8_modules": list(V8_SCIENCE_MODULES),
    }
