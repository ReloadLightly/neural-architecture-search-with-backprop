"""Durable, self-describing run records.

The first attempt at this study lost completed results because nothing promoted
them out of the execution environment. Everything here is built around one rule:
a finished run is written atomically, fingerprinted, and independently
reconstructable, before anything else happens.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .genome import OP_NAMES, Genome

SCHEMA_VERSION = 1


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json_atomic(path: Path, payload: dict) -> str:
    """Write JSON via a temp file + rename, and return its sha256.

    A crash mid-write leaves the previous file intact rather than a truncated
    one, which is what makes a partially completed shard safe to resume.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# The modules that determine what a run computes. Analysis, figure, CLI,
# suite, record, checkpoint and final-test code is deliberately excluded: it
# cannot change a run's result, and including it made the fingerprint move four
# times during the v1 suite purely because plotting code was added mid-run.
SCIENCE_MODULES = (
    "baselines.py",
    "conditions.py",
    "datasets.py",
    "evolve.py",
    "genome.py",
    "learn.py",
    "protocol.py",
)


def code_fingerprint() -> dict:
    """Hash only the source that determines scientific output."""
    root = Path(__file__).resolve().parent
    per_file = {name: sha256_file(root / name) for name in SCIENCE_MODULES}
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(per_file.items())).encode()
    ).hexdigest()
    return {"files": per_file, "combined": combined}


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "git_commit": git_commit(),
        "code_fingerprint": code_fingerprint()["combined"],
    }


def serialise_genome(g: Genome, weights: np.ndarray) -> dict:
    """Everything needed to rebuild a champion exactly, without the search.

    ``weights`` is the live vector; ``Genome.weight`` is only the initial one
    and is deliberately not persisted.
    """
    return {
        "ops": list(g.ops),
        "op_names": [OP_NAMES[o] for o in g.ops],
        "src": list(g.src),
        "dst": list(g.dst),
        "active": [bool(a) for a in g.active],
        "innovation": list(g.innovation),
        "weights": [float(w) for w in weights],
    }


def deserialise_genome(d: dict) -> tuple[Genome, np.ndarray]:
    g = Genome(
        ops=list(d["ops"]),
        src=list(d["src"]),
        dst=list(d["dst"]),
        weight=[float(w) for w in d["weights"]],
        active=[bool(a) for a in d["active"]],
        innovation=list(d["innovation"]),
    )
    return g, np.array(d["weights"], dtype=np.float64)


@dataclass
class RunRecord:
    """One (task, condition, replicate) run. Never contains test metrics."""

    run_id: str
    task: str
    condition: str
    replicate: int
    dataset_seed: int
    search_seed: int
    track: str
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    compute: dict = field(default_factory=dict)
    champion: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    environment: dict = field(default_factory=environment)
    schema_version: int = SCHEMA_VERSION
    test_evaluated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def run_id(task: str, condition: str, replicate: int) -> str:
    return f"{task}__{condition}__r{replicate:02d}"


def assert_no_test_metrics(payload: dict) -> None:
    """Structural guard: a search-stage record may not carry test numbers."""
    blob = json.dumps(payload, default=_json_default)
    for banned in ('"test_accuracy"', '"test_loss"', '"test_'):
        if banned in blob and '"test_evaluated"' not in banned:
            offending = [k for k in _walk_keys(payload) if k.startswith("test_")]
            offending = [k for k in offending if k != "test_evaluated"]
            if offending:
                raise AssertionError(f"record carries sealed-test metrics: {offending}")


def _walk_keys(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)
