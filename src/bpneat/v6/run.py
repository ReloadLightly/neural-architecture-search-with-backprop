"""One entry point for the v6 suite: run a shard, check status, finalise.

``make v6-run`` launches every shard in the background; this module is what each
shard executes, and what reports on them.

Thread pinning happens here, before NumPy is imported, and deliberately not in
the Makefile — same reason as v3, v4 and v5: each shard is single-threaded
Python doing many small matrix products, so a BLAS thread pool buys nothing, but
one pool per shard oversubscribes the machine by the shard count.
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
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

from .protocol import (  # noqa: E402
    ALL_BUDGETS,
    ALL_TASKS,
    ARMS,
    BUDGETS,
    N_REPLICATES,
    cells,
    planned_runs,
)
from .suite import run_cells  # noqa: E402


def _status(out_dir: Path) -> int:
    runs = out_dir / "raw" / "runs"
    paths = sorted(runs.glob("*.json")) if runs.exists() else []
    planned = len(planned_runs())
    pct = 100.0 * len(paths) / planned if planned else 0.0
    print(f"{len(paths)}/{planned} confirmatory runs ({pct:.1f}%), "
          f"{len(cells())} cells planned")

    # Per-rung, because the rungs do not cost the same: an overall percentage
    # says nothing about how much work is left.
    by_budget: dict[str, int] = defaultdict(int)
    by_task: dict[str, int] = defaultdict(int)
    for p in paths:
        task, cond, _ = p.stem.split("__")
        by_budget[cond.rpartition("_")[2]] += 1
        by_task[task] += 1
    per_rung = len(ARMS) * len(ALL_TASKS) * N_REPLICATES
    for b in ALL_BUDGETS:
        tag = "" if b in BUDGETS else "  (extension)"
        print(f"  {b.label:>8}  {by_budget.get(b.label, 0):4d}/{per_rung}{tag}")
    print("  tasks: " + ", ".join(f"{t}={by_task.get(t, 0)}" for t in ALL_TASKS))

    mpath = out_dir / "manifest.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        print(f"complete={m['complete']} failed={len(m['failed'])} "
              f"missing={len(m['missing'])}")
        for f in m["failed"][:5]:
            print(f"  FAILED {f['run_id']}: {f['error'][:120]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bpneat-v6")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-total", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extension", action="store_true",
                    help="also run the declared b16800 rung")
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

    m = run_cells(out, args.shard_index, args.shard_total, force=args.force,
                  include_extension=args.extension)
    print(f"\n{m['present']}/{m['planned']} present, complete={m['complete']}, "
          f"failed={len(m['failed'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
