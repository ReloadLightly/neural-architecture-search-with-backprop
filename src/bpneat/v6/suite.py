"""Sharded, resumable execution of the v6 plan.

The shard unit is a **cell** — one (task, replicate) — because each fixed arm
needs the gradient budget the search arm spent at the same rung in that same
cell. Running a cell as a unit keeps that dependency local, so shards stay
independent and any shard can be re-run.

Within a cell the order is budget-ascending, and inside a rung
search -> null -> fixed. Two reasons, both about what a truncated run leaves
behind: the fixed arm cannot start before its search arm, and a cell that is cut
off mid-way then holds complete low rungs rather than three half-finished ones.

Each finished run is written atomically the moment it completes, so an
interrupted shard loses at most the run in flight.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..record import write_json_atomic
from .conditions import CONDITIONS, run_condition
from .fingerprint import fingerprints
from .protocol import (
    ARM_ORDER,
    MATCHED_TO,
    PROTOCOL_VERSION,
    cells,
    planned_runs,
    split_condition,
)


def shard_cells(index: int, total: int) -> list[tuple[str, int]]:
    if not (0 <= index < total):
        raise ValueError(f"shard {index} out of range for {total}")
    return [c for i, c in enumerate(cells()) if i % total == index]


def _plan_by_cell(include_extension: bool) -> dict[tuple[str, int], list[dict]]:
    out: dict[tuple[str, int], list[dict]] = {}
    for p in planned_runs(include_extension):
        out.setdefault((p["task"], p["replicate"]), []).append(p)
    return out


def _order_key(p: dict) -> tuple:
    """Cheap rungs first; inside a rung, the arm others are matched to first."""
    arm, budget = split_condition(p["condition"])
    return (budget.candidates, ARM_ORDER[arm])


def run_cells(
    out_dir: Path,
    shard_index: int = 0,
    shard_total: int = 1,
    force: bool = False,
    include_extension: bool = False,
    progress=print,
) -> dict:
    out_dir = Path(out_dir)
    runs_dir = out_dir / "raw" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    fp = fingerprints()
    by_cell = _plan_by_cell(include_extension)
    mine = shard_cells(shard_index, shard_total)
    started = time.time()
    done = skipped = 0
    failed: list[dict] = []

    for ci, cell in enumerate(mine, 1):
        task, rep = cell
        items = sorted(by_cell[cell], key=_order_key)
        budgets: dict[str, int] = {}

        for p in items:
            cond = p["condition"]
            rid = f"{task}__{cond}__r{rep:02d}"
            path = runs_dir / f"{rid}.json"

            if path.exists() and not force:
                try:
                    rec = json.loads(path.read_text())
                    if rec.get("fingerprints", {}).get("v6") == fp["v6"]:
                        budgets[cond] = rec["compute"]["gradient_steps"]
                        skipped += 1
                        continue
                except json.JSONDecodeError:
                    pass

            needs = MATCHED_TO.get(cond)
            if needs is not None and needs not in budgets:
                src = runs_dir / f"{task}__{needs}__r{rep:02d}.json"
                if src.exists():
                    budgets[needs] = json.loads(src.read_text())["compute"][
                        "gradient_steps"
                    ]
                else:
                    failed.append({"run_id": rid, "error": f"{needs} budget unavailable"})
                    progress(f"      SKIP {rid}: no {needs} budget yet")
                    continue

            try:
                rec = run_condition(
                    condition=cond, task=task, replicate=rep,
                    dataset_seed=p["dataset_seed"], search_seed=p["search_seed"],
                    matched_steps=budgets if needs is not None else None,
                )
                rec["protocol_version"] = PROTOCOL_VERSION
                rec["fingerprints"] = fp
                write_json_atomic(path, rec)
                budgets[cond] = rec["compute"]["gradient_steps"]
                done += 1
                progress(
                    f"[{ci}/{len(mine)}] {rid}: val={rec['metrics']['validation_accuracy']:.3f} "
                    f"causal={rec['metrics']['causal_hidden_nodes']}n "
                    f"steps={rec['compute']['gradient_steps']:,} "
                    f"{rec['compute']['wall_time_seconds']:.0f}s"
                )
            except Exception as exc:  # a bad run must not kill the shard
                failed.append({"run_id": rid, "error": repr(exc)})
                progress(f"      FAILED {rid}: {exc!r}")

        _manifest(out_dir, runs_dir, fp, failed, started, done, skipped)

    return _manifest(out_dir, runs_dir, fp, failed, started, done, skipped)


def _rid(p: dict) -> str:
    return f"{p['task']}__{p['condition']}__r{p['replicate']:02d}"


def _manifest(out_dir, runs_dir, fp, failed, started, done, skipped) -> dict:
    expected = {_rid(p) for p in planned_runs()}
    extension = {_rid(p) for p in planned_runs(include_extension=True)} - expected
    present = {p.stem for p in runs_dir.glob("*.json")}
    # ``complete`` is about the confirmatory ladder alone. The extension rung is
    # counted and reported, and its absence is not an incompleteness, because
    # the contract scores no hypothesis on it.
    m = {
        "protocol_version": PROTOCOL_VERSION,
        "planned": len(expected),
        "present": len(present & expected),
        "missing": sorted(expected - present),
        "unexpected": sorted(present - expected - extension),
        "failed": failed,
        "complete": not (expected - present) and not failed,
        "extension_planned": len(extension),
        "extension_present": len(present & extension),
        "extension_complete": bool(extension) and not (extension - present),
        "test_evaluated": False,
        "fingerprints": fp,
        "conditions": sorted(CONDITIONS),
        "elapsed_seconds": time.time() - started,
        "this_shard": {"completed": done, "skipped": skipped},
    }
    write_json_atomic(Path(out_dir) / "manifest.json", m)
    return m
