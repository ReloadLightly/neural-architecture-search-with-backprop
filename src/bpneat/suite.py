"""Sharded execution with durable, resumable artifacts.

Each run owns its own file and is written atomically the moment it finishes, so
an interrupted shard loses at most the run in flight. A shard is safely
re-runnable: completed runs are skipped by fingerprint unless the code that
produced them has changed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .conditions import run_condition
from .datasets import make_bundle
from .protocol import PROTOCOL_VERSION, TRACKS, planned_runs
from .record import (
    assert_no_test_metrics,
    code_fingerprint,
    environment,
    run_id,
    write_json_atomic,
)


def shard(plan: list[dict], index: int, total: int) -> list[dict]:
    """Deterministic interleaved split, so shards stay balanced by cost."""
    if not (0 <= index < total):
        raise ValueError(f"shard {index} out of range for {total}")
    return [p for i, p in enumerate(plan) if i % total == index]


def run_plan(
    out_dir: Path,
    track: str = "B",
    conditions=None,
    tasks=None,
    replicates=None,
    shard_index: int = 0,
    shard_total: int = 1,
    checkpoints: bool = True,
    force: bool = False,
    progress=print,
) -> dict:
    """Execute a shard of the plan, writing one durable record per run."""
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}")
    spec = TRACKS[track]

    out_dir = Path(out_dir)
    runs_dir = out_dir / "raw" / "runs"
    ckpt_dir = out_dir / "checkpoints"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    plan = planned_runs(track, conditions, tasks, replicates)
    mine = shard(plan, shard_index, shard_total)
    fingerprint = code_fingerprint()["combined"]

    completed, skipped, failed = 0, 0, []
    started = time.time()

    for i, item in enumerate(mine, 1):
        rid = run_id(item["task"], item["condition"], item["replicate"])
        path = runs_dir / f"{rid}.json"

        if path.exists() and not force:
            try:
                existing = json.loads(path.read_text())
                if existing.get("environment", {}).get("code_fingerprint") == fingerprint:
                    skipped += 1
                    progress(f"[{i}/{len(mine)}] skip {rid} (already complete)")
                    continue
            except json.JSONDecodeError:
                pass  # truncated or corrupt: re-run it

        progress(f"[{i}/{len(mine)}] run  {rid}")
        try:
            bundle = make_bundle(item["task"], seed=item["dataset_seed"])
            record = run_condition(
                condition=item["condition"],
                bundle=bundle,
                replicate=item["replicate"],
                dataset_seed=item["dataset_seed"],
                search_seed=item["search_seed"],
                propagation=spec.propagation,
                track=track,
                fitness_split=spec.fitness_split,
            )
            payload = record.to_dict()
            payload["protocol_version"] = PROTOCOL_VERSION
            assert_no_test_metrics(payload)
            write_json_atomic(path, payload)
            completed += 1
            progress(
                f"      val_acc={record.metrics['validation_accuracy']:.3f} "
                f"causal={record.metrics['causal_hidden_nodes']}n "
                f"cands={record.compute['candidate_evaluations']} "
                f"{record.compute['wall_time_seconds']:.1f}s"
            )
        except Exception as exc:  # a failed run must not kill the shard
            failed.append({"run_id": rid, "error": repr(exc)})
            progress(f"      FAILED {exc!r}")

        _write_manifest(out_dir, track, plan, runs_dir, fingerprint, failed, started)

    return _write_manifest(out_dir, track, plan, runs_dir, fingerprint, failed, started)


def _write_manifest(out_dir, track, plan, runs_dir, fingerprint, failed, started) -> dict:
    """A live manifest, refreshed after every run so progress is never guessed."""
    present = {p.stem for p in runs_dir.glob("*.json")}
    expected = {run_id(p["task"], p["condition"], p["replicate"]) for p in plan}
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "track": track,
        "planned": len(expected),
        "present": len(present & expected),
        "missing": sorted(expected - present),
        "unexpected": sorted(present - expected),
        "failed": failed,
        "complete": not (expected - present) and not failed,
        "test_evaluated": False,
        "code_fingerprint": fingerprint,
        "environment": environment(),
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(Path(out_dir) / "manifest.json", manifest)
    return manifest
