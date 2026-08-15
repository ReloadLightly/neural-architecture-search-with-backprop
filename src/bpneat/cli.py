"""Command line entry points.

``search`` and ``suite`` are structurally incapable of reading the sealed test
split; ``final-test`` is the only command that can, and it refuses anything that
is not a complete, frozen, unmodified confirmatory release.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .conditions import CONDITIONS, CORE_CONDITIONS, run_condition
from .datasets import make_bundle
from .protocol import PROTOCOL_VERSION, REPLICATES, TASKS, TRACKS, planned_runs
from .record import write_json_atomic
from .suite import run_plan


def _cmd_search(args) -> int:
    bundle = make_bundle(args.task, seed=args.dataset_seed)
    spec = TRACKS[args.track]
    rec = run_condition(
        condition=args.condition,
        bundle=bundle,
        replicate=args.replicate,
        dataset_seed=args.dataset_seed,
        search_seed=args.search_seed,
        propagation=spec.propagation,
        track=args.track,
        fitness_split=spec.fitness_split,
    )
    print(json.dumps({"metrics": rec.metrics, "compute": rec.compute}, indent=2))
    if args.out:
        write_json_atomic(Path(args.out), rec.to_dict())
        print(f"wrote {args.out}")
    return 0


def _cmd_suite(args) -> int:
    manifest = run_plan(
        out_dir=Path(args.out),
        track=args.track,
        conditions=args.conditions or CORE_CONDITIONS,
        tasks=args.tasks or TASKS,
        replicates=REPLICATES[: args.replicates] if args.replicates else None,
        shard_index=args.shard_index,
        shard_total=args.shard_total,
        force=args.force,
    )
    print(
        f"\n{manifest['present']}/{manifest['planned']} runs present, "
        f"complete={manifest['complete']}, failed={len(manifest['failed'])}"
    )
    return 0 if not manifest["failed"] else 1


def _cmd_plan(args) -> int:
    plan = planned_runs(
        args.track,
        args.conditions or CORE_CONDITIONS,
        args.tasks or TASKS,
        REPLICATES[: args.replicates] if args.replicates else None,
    )
    print(f"protocol {PROTOCOL_VERSION}, track {args.track}: {len(plan)} runs")
    for task in args.tasks or TASKS:
        n = sum(1 for p in plan if p["task"] == task)
        print(f"  {task:8s} {n:4d} runs")
    return 0


def _cmd_final_test(args) -> int:
    from .finaltest import FinalTestRefused, run_final_test, write_checksums

    try:
        run_final_test(Path(args.dir), allow_draft=args.allow_draft)
    except FinalTestRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"checksums: {write_checksums(Path(args.dir))}")
    return 0


def _cmd_report(args) -> int:
    from .analysis import build

    payload = build(Path(args.dir))
    print(f"\n{payload['n_runs']} runs, tracks={payload['tracks']}, "
          f"test_evaluated={payload['test_evaluated']}")
    print(f"\n{'task':8s} {'condition':18s} {'val_acc':>9} {'sd':>6} {'succ':>6} "
          f"{'causal_n':>9} {'steps':>9}")
    for r in payload["summary"]:
        print(f"{r['task']:8s} {r['condition']:18s} "
              f"{r['validation_accuracy_mean']:9.3f} {r['validation_accuracy_sd']:6.3f} "
              f"{r['success_rate']:6.2f} {r['causal_hidden_nodes_mean']:9.1f} "
              f"{r['gradient_steps_mean']:9.0f}")
    return 0


def _cmd_figures(args) -> int:
    from .figures import build_all

    paths = build_all(Path(args.dir))
    for p in paths:
        print(f"wrote {p}")
    return 0


def _cmd_conditions(args) -> int:
    for name, c in CONDITIONS.items():
        print(f"{'*' if c.core else ' '} {name:18s} {c.kind:14s} {c.mechanism}")
    print("\n* = core confirmatory condition")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bpneat", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_matrix_args(sp):
        sp.add_argument("--track", choices=sorted(TRACKS), default="B")
        sp.add_argument("--tasks", nargs="*", choices=TASKS)
        sp.add_argument("--conditions", nargs="*", choices=sorted(CONDITIONS))
        sp.add_argument("--replicates", type=int, help="use the first N replicates")

    s = sub.add_parser("search", help="one run, printed and optionally saved")
    s.add_argument("--task", choices=TASKS, required=True)
    s.add_argument("--condition", choices=sorted(CONDITIONS), default="backprop_neat")
    s.add_argument("--track", choices=sorted(TRACKS), default="B")
    s.add_argument("--dataset-seed", type=int, default=8101)
    s.add_argument("--search-seed", type=int, default=18101)
    s.add_argument("--replicate", type=int, default=0)
    s.add_argument("--out")
    s.set_defaults(func=_cmd_search)

    s = sub.add_parser("suite", help="run a shard of the matrix")
    add_matrix_args(s)
    s.add_argument("--out", required=True)
    s.add_argument("--shard-index", type=int, default=0)
    s.add_argument("--shard-total", type=int, default=1)
    s.add_argument("--force", action="store_true", help="re-run completed records")
    s.set_defaults(func=_cmd_suite)

    s = sub.add_parser("plan", help="print the run plan without executing it")
    add_matrix_args(s)
    s.set_defaults(func=_cmd_plan)

    s = sub.add_parser("final-test", help="one-shot sealed-test evaluation")
    s.add_argument("--dir", required=True)
    s.add_argument(
        "--allow-draft",
        action="store_true",
        help="permit a -draft protocol (calibration rehearsal only)",
    )
    s.set_defaults(func=_cmd_final_test)

    s = sub.add_parser("report", help="rebuild every table from the raw records")
    s.add_argument("--dir", required=True)
    s.set_defaults(func=_cmd_report)

    s = sub.add_parser("figures", help="render the publication figures")
    s.add_argument("--dir", required=True)
    s.set_defaults(func=_cmd_figures)

    s = sub.add_parser("conditions", help="list the comparison matrix")
    s.set_defaults(func=_cmd_conditions)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
