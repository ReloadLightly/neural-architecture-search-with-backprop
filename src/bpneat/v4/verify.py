"""Prove a committed v4 release regenerates from its own raw records.

Four checks, in the order that matters: the derived tables must rebuild
byte-identically from ``raw/runs/*.json``; every recorded hash must match and
every file must be listed; the manifest's three fingerprints must still equal
the live ones — including the v3 and v2 sets, which v4 is forbidden to touch;
and the bridge, if present, must still report that every v3 cell it re-ran
reproduced bit-for-bit.
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
        "operator-usage.csv", "sign-matrix.csv", "cross-algorithm.csv",
        "hypotheses.csv",
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
        for key in ("v4", "v3_frozen", "v2_frozen"):
            if m.get("fingerprints", {}).get(key) != live[key]:
                failures.append(f"fingerprint: {key} has moved since the suite ran")
        progress(
            f"  manifest      protocol {m.get('protocol_version')}, "
            f"complete={m.get('complete')}, test_evaluated={m.get('test_evaluated')}"
        )
        progress(f"  fingerprint   v4 {live['v4']}")
        progress(f"                v3 {live['v3_frozen']} (released, frozen)")
        progress(f"                v2 {live['v2_frozen']} (frozen)")
    else:
        failures.append("manifest: manifest.json is missing")

    bpath = release_dir / "bridge" / "replication.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())
        if not b.get("all_identical"):
            failures.append(
                f"bridge: only {b.get('n_identical')}/{b.get('n_runs')} v3 cells reproduce"
            )
        else:
            progress(
                f"  bridge        {b['n_identical']}/{b['n_runs']} v3 cells reproduce bit-for-bit"
            )

    for f in failures:
        progress(f"  FAIL  {f}")
    if failures:
        progress(f"FAILED — {len(failures)} check(s) did not hold")
        return 1
    progress("OK — the release regenerates from its own raw records")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bpneat-v4-verify")
    ap.add_argument("--dir", required=True)
    return verify(Path(ap.parse_args(argv).dir))


if __name__ == "__main__":
    raise SystemExit(main())
