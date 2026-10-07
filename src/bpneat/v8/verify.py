"""Prove a committed v8 release regenerates from its own raw records.

Three checks: the derived tables must rebuild byte-identically from
``raw/runs/*.json``; every recorded hash must match and every file must be
listed; and the manifest's six fingerprints must still equal the live
ones — v8's own, the committed datasets by content, the n-dimensional core, and
the three released sets that core is proved identical to at two inputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from ..record import sha256_file
from .fingerprint import fingerprints

DERIVED = (
    "summary.csv",
    "summary.json",
    "arm-contrasts.csv",
    "equivalence.csv",
    "champion-size.csv",
    "budget-table.csv",
    "hypotheses.csv",
)


def verify(release_dir: Path, progress=print) -> int:
    from .analysis import build
    from .release import verify_checksums

    release_dir = Path(release_dir)
    progress(f"verifying {release_dir}")
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "release"
        scratch.mkdir()
        shutil.copytree(release_dir / "raw", scratch / "raw")
        for extra in ("final-test.json", "manifest.json"):
            if (release_dir / extra).exists():
                shutil.copy2(release_dir / extra, scratch / extra)
        build(scratch, progress=lambda *_: None)
        rebuilt = 0
        for name in DERIVED:
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
        for key in ("v8", "data", "nd", "v5_frozen", "v3_frozen", "v2_frozen"):
            if m.get("fingerprints", {}).get(key) != live[key]:
                failures.append(f"fingerprint: {key} has moved since the suite ran")
        progress(
            f"  manifest      protocol {m.get('protocol_version')}, "
            f"complete={m.get('complete')}, test_evaluated={m.get('test_evaluated')}, "
            f"extension={m.get('extension_present', 0)}/{m.get('extension_planned', 0)}"
        )
        progress(f"  fingerprint   v8 {live['v8']}")
        progress(f"                data {live['data']} (committed CSVs, by content)")
        progress(f"                nd   {live['nd']} (the n-dimensional core)")
        for k, label in (("v5_frozen", "v5"), ("v3_frozen", "v3"), ("v2_frozen", "v2")):
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
    ap = argparse.ArgumentParser(prog="bpneat-v8-verify")
    ap.add_argument("--dir", required=True)
    return verify(Path(ap.parse_args(argv).dir))


if __name__ == "__main__":
    raise SystemExit(main())
