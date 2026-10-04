"""Sharded, resumable execution of the v3 plan.

The shard unit is a **cell** — one (task, replicate) — not a single run, because
the matched fixed-architecture conditions need the gradient budget that
``backprop_neat`` spent in that same cell. Running a cell as a unit keeps that
dependency local, so shards stay independent and any shard can be re-run.

Each finished run is written atomically the moment it completes, so an
interrupted shard loses at most the run in flight. A container restart during
the v2 suite cost nothing; the same design carries over.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..record import write_json_atomic
from .conditions import CONDITIONS, run_condition
from .fingerprint import fingerprints
from .protocol import PROTOCOL_VERSION, cells, planned_runs

REFERENCE = "backprop_neat"


def shard_cells(index: int, total: int) -> list[tuple[str, int]]:
    if not (0 <= index < total):
        raise ValueError(f"shard {index} out of range for {total}")
    return [c for i, c in enumerate(cells()) if i % total == index]


def _plan_by_cell() -> dict[tuple[str, int], list[dict]]:
    out: dict[tuple[str, int], list[dict]] = {}
    for p in planned_runs():
        out.setdefault((p["task"], p["replicate"]), []).append(p)
    return out


def run_cells(
    out_dir: Path,
    shard_index: int = 0,
    shard_total: int = 1,
    force: bool = False,
    progress=print,
) -> dict:
    out_dir = Path(out_dir)
    runs_dir = out_dir / "raw" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    fp = fingerprints()
    by_cell = _plan_by_cell()
    mine = shard_cells(shard_index, shard_total)
    started = time.time()
    done = skipped = 0
    failed: list[dict] = []

    for ci, cell in enumerate(mine, 1):
        items = by_cell[cell]
        task, rep = cell
        # The reference cell first: the matched arms are paired to its budget.
        items = sorted(items, key=lambda p: (p["condition"] != REFERENCE, p["condition"]))
        matched_steps: int | None = None

        for p in items:
            rid = f"{task}__{p['condition']}__r{rep:02d}"
            path = runs_dir / f"{rid}.json"

            if path.exists() and not force:
                try:
                    rec = json.loads(path.read_text())
                    if rec.get("fingerprints", {}).get("v3") == fp["v3"]:
                        if p["condition"] == REFERENCE:
                            matched_steps = rec["compute"]["gradient_steps"]
                        skipped += 1
                        continue
                except json.JSONDecodeError:
                    pass

            needs_budget = CONDITIONS[p["condition"]].kind == "matched_multistart"
            if needs_budget and matched_steps is None:
                ref_path = runs_dir / f"{task}__{REFERENCE}__r{rep:02d}.json"
                if ref_path.exists():
                    matched_steps = json.loads(ref_path.read_text())["compute"]["gradient_steps"]
                else:
                    failed.append({"run_id": rid, "error": "reference run unavailable"})
                    progress(f"      SKIP {rid}: no reference budget yet")
                    continue

            try:
                rec = run_condition(
                    condition=p["condition"], task=task, replicate=rep,
                    dataset_seed=p["dataset_seed"], search_seed=p["search_seed"],
                    matched_steps=matched_steps if needs_budget else None,
                )
                rec["protocol_version"] = PROTOCOL_VERSION
                rec["fingerprints"] = fp
                write_json_atomic(path, rec)
                if p["condition"] == REFERENCE:
                    matched_steps = rec["compute"]["gradient_steps"]
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


def _manifest(out_dir, runs_dir, fp, failed, started, done, skipped) -> dict:
    expected = {f"{p['task']}__{p['condition']}__r{p['replicate']:02d}" for p in planned_runs()}
    present = {p.stem for p in runs_dir.glob("*.json")}
    m = {
        "protocol_version": PROTOCOL_VERSION,
        "planned": len(expected),
        "present": len(present & expected),
        "missing": sorted(expected - present),
        "unexpected": sorted(present - expected),
        "failed": failed,
        "complete": not (expected - present) and not failed,
        "test_evaluated": False,
        "fingerprints": fp,
        "elapsed_seconds": time.time() - started,
        "this_shard": {"completed": done, "skipped": skipped},
    }
    write_json_atomic(Path(out_dir) / "manifest.json", m)
    return m
