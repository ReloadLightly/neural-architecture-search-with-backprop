"""One entry point for the v3 suite: run a shard, check status, finalise.

``make v3-run`` launches every shard in the background; this module is what
each shard executes, and what reports on them.

Thread pinning happens here, before NumPy is imported, and deliberately not in
the Makefile. Each shard is single-threaded Python doing many tiny matrix
products, so a BLAS thread pool buys nothing — but with one pool per shard,
four shards oversubscribe a four-core machine four-fold. Measured on this
container: 16 threads on 4 cores ran each shard about 4x slower than running
one shard alone. Setting it in code means the pinning cannot be lost by
invoking the module directly.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from .protocol import cells, planned_runs  # noqa: E402
from .suite import run_cells  # noqa: E402


def _status(out_dir: Path) -> int:
    runs = out_dir / "raw" / "runs"
    present = len(list(runs.glob("*.json"))) if runs.exists() else 0
    planned = len(planned_runs())
    pct = 100.0 * present / planned if planned else 0.0
    print(f"{present}/{planned} runs ({pct:.1f}%), {len(cells())} cells planned")
    mpath = out_dir / "manifest.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        print(f"complete={m['complete']} failed={len(m['failed'])} missing={len(m['missing'])}")
        for f in m["failed"][:5]:
            print(f"  FAILED {f['run_id']}: {f['error'][:120]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bpneat-v3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-total", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--release", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)

    if args.status:
        return _status(out)
    if args.final_test:
        from .finaltest import run_final_test

        run_final_test(out)
        return 0
    if args.release:
        from .release import build_release

        build_release(out)
        return 0

    m = run_cells(out, args.shard_index, args.shard_total, force=args.force)
    print(f"\n{m['present']}/{m['planned']} present, complete={m['complete']}, "
          f"failed={len(m['failed'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
