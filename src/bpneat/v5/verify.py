"""Prove a committed v5 release regenerates from its own raw records.

Three checks: the derived tables must rebuild byte-identically from
``raw/runs/*.json``; every recorded hash must match and every file must be
listed; and the manifest's four fingerprints must still equal the live ones,
including the three older sets v5 is forbidden to touch.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from ..record import sha256_file
from .fingerprint import fingerprints


def verify(release_dir: Path, progress=print) -> int:
    from .analysis import build
    from .release import verify_checksums

    release_dir = Path(release_dir)
    progress(f"verifying {release_dir}")
    failures: list[str] = []

    derived = [
        "summary.csv", "summary.json", "paired-effects.csv", "budget-table.csv",
        "operator-usage.csv", "complexity.csv", "hypotheses.csv",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "release"
        scratch.mkdir()
        shutil.copytree(release_dir / "raw", scratch / "raw")
        for extra in ("final-test.json", "manifest.json"):
            if (release_dir / extra).exists():
                shutil.copy2(release_dir / extra, scratch / extra)
        build(scratch, progress=lambda *_: None)
        rebuilt = 0
        for name in derived:
            a, b = release_dir / name, scratch / name
            if not a.exists():
                continue
            if not b.exists() or sha256_file(a) != sha256_file(b):
                failures.append(f"regeneration: {name} does not rebuild byte-identically")
            else:
                rebuilt += 1
    n_runs = len(list((release_dir / "raw" / "runs").glob("*.json")))
    progress(f"  regeneration  {rebuilt} derived files rebuilt from {n_runs} raw records")

    bad = verify_checksums(release_dir)
    if bad:
        failures.extend(f"checksums: {b}" for b in bad)
    else:
        progress("  checksums     all recorded hashes match, no unlisted files")

    mpath = release_dir / "manifest.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        live = fingerprints()
        for key in ("v5", "v4_frozen", "v3_frozen", "v2_frozen"):
            if m.get("fingerprints", {}).get(key) != live[key]:
                failures.append(f"fingerprint: {key} has moved since the suite ran")
        progress(
            f"  manifest      protocol {m.get('protocol_version')}, "
            f"complete={m.get('complete')}, test_evaluated={m.get('test_evaluated')}"
        )
        progress(f"  fingerprint   v5 {live['v5']}")
        for k, label in (("v4_frozen", "v4"), ("v3_frozen", "v3"), ("v2_frozen", "v2")):
            progress(f"                {label} {live[k]} (released, frozen)")
    else:
        failures.append("manifest: manifest.json is missing")

    for f in failures:
        progress(f"  FAIL  {f}")
    if failures:
        progress(f"FAILED — {len(failures)} check(s) did not hold")
        return 1
    progress("OK — the release regenerates from its own raw records")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bpneat-v5-verify")
    ap.add_argument("--dir", required=True)
    return verify(Path(ap.parse_args(argv).dir))


if __name__ == "__main__":
    raise SystemExit(main())
