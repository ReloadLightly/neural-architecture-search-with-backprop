"""Package a v5 release — in the only safe order.

v2 learned this the hard way: `final-test` wrote the checksum manifest as its
last act, then the tables were regenerated to add the sealed-test columns, and
the manifest ended up describing files that no longer existed in that form.
Here tables and figures are rebuilt first and checksums are sealed last, in one
command, so the ordering cannot be got wrong by hand.
"""

from __future__ import annotations

from pathlib import Path

from ..record import sha256_file
from .analysis import build as build_tables
from .figures import build_all as build_figures


def write_checksums(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    lines = [
        f"{sha256_file(p)}  {p.relative_to(out_dir)}"
        for p in sorted(out_dir.rglob("*"))
        if p.is_file() and p.name != "sha256sums.txt"
    ]
    target = out_dir / "sha256sums.txt"
    target.write_text("\n".join(lines) + "\n")
    return target


def verify_checksums(out_dir: Path) -> list[str]:
    out_dir = Path(out_dir)
    manifest = out_dir / "sha256sums.txt"
    if not manifest.exists():
        return ["sha256sums.txt (missing)"]
    bad, listed = [], set()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        want, rel = line.split("  ", 1)
        listed.add(rel)
        target = out_dir / rel
        if not target.exists():
            bad.append(f"{rel} (missing)")
        elif sha256_file(target) != want:
            bad.append(rel)
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "sha256sums.txt":
            rel = str(p.relative_to(out_dir))
            if rel not in listed:
                bad.append(f"{rel} (unlisted)")
    return bad


def build_release(out_dir: Path, progress=print) -> dict:
    out_dir = Path(out_dir)
    payload = build_tables(out_dir, progress=progress)
    build_figures(out_dir, progress=progress)
    write_checksums(out_dir)
    bad = verify_checksums(out_dir)
    progress(f"checksums: {'OK' if not bad else 'MISMATCH ' + ', '.join(bad)}")
    if bad:
        raise SystemExit(1)
    return payload
