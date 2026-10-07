"""Measure what each v8 arm costs, on the pilot seeds, before the freeze.

v8's shape is a cost decision as much as a design one: one search at the
reference budget takes minutes on three of the datasets and nearly an hour on
the fourth, which is why that fourth is a declared extension rather than a
confirmatory dataset. That number has to be measured rather than assumed, and
the measurement has to be on the record, so this script writes one.

Pilot seeds only (9005, 19005), burned in ``v8.protocol.BURNED_SEEDS`` before
the freeze. One replicate per dataset. Scores no hypothesis and never reads a
sealed test split.

The search arm is run at a *reduced* generation count on purpose and the cost of
the full budget is extrapolated from its realised per-candidate cost, because
measuring the full thing on `digits` would cost an hour of the machine the
confirmatory suite is running on. The extrapolation is linear in candidates,
which is right for the three non-evolutionary arms and an *under*-estimate for
the search, whose cost grows as its networks do — v6's pilot measured that
growth directly. The record says which numbers are measured and which are
scaled, and by what.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat.nd.baselines import (  # noqa: E402
    make_linear,
    make_mixed_mlp,
    matched_multistart,
)
from bpneat.nd.datasets import make_bundle  # noqa: E402
from bpneat.nd.evolve import NdConfig, layout_for, search  # noqa: E402
from bpneat.nd.learn import accuracy  # noqa: E402
from bpneat.record import environment, write_json_atomic  # noqa: E402
from bpneat.v8.conditions import config_for, random_architecture_search  # noqa: E402
from bpneat.v8.fingerprint import fingerprints  # noqa: E402
from bpneat.v8.protocol import (  # noqa: E402
    BURNED_SEEDS,
    EVERY_TASK,
    FIXED_HIDDEN,
    MULTISTART_RESTARTS,
    PROTOCOL_VERSION,
    REFERENCE_CANDIDATES,
)

SPLIT_SEED, SEARCH_SEED = 9005, 19005
OUT = Path("results/backprop-neat-v8/pilot/cost.json")

#: How much of the real budget the pilot actually spends. Small enough that the
#: pilot does not compete with a confirmatory suite for a machine, large enough
#: that the per-candidate cost it measures is not dominated by start-up.
PILOT_GENERATIONS = 4
PILOT_MULTISTART_STEPS = 20_000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="v8-pilot-cost")
    ap.add_argument("--tasks", nargs="*", default=list(EVERY_TASK))
    args = ap.parse_args(argv)
    assert SPLIT_SEED in BURNED_SEEDS and SEARCH_SEED in BURNED_SEEDS

    rows: list[dict] = []
    for task in args.tasks:
        bundle = make_bundle(task, seed=SPLIT_SEED)
        layout = layout_for(bundle)
        full = config_for(task, bundle)

        # --- search, at a reduced generation count -------------------------
        cfg = NdConfig(
            task=task, layout=layout, generations=PILOT_GENERATIONS,
            population=full.population, n_species=full.n_species,
            inner_steps=full.inner_steps, batch_size=full.batch_size,
        )
        t0 = time.time()
        res = search(bundle, cfg, seed=SEARCH_SEED)
        search_wall = time.time() - t0
        per_candidate = search_wall / res.candidates
        steps_per_candidate = res.gradient_steps / res.candidates
        full_steps = int(steps_per_candidate * REFERENCE_CANDIDATES)
        rows.append({
            "task": task, "arm": "search", "measured": True,
            "n_features": bundle.n_features, "n_classes": bundle.n_classes,
            "candidates": res.candidates, "wall_time_seconds": search_wall,
            "seconds_per_candidate": per_candidate,
            "gradient_steps": res.gradient_steps,
            "steps_per_candidate": steps_per_candidate,
            "validation_accuracy": res.metrics["validation_accuracy"],
            "causal_hidden_nodes": res.metrics["causal_hidden_nodes"],
            "full_budget_minutes_linear_scaling": per_candidate
            * REFERENCE_CANDIDATES / 60.0,
            "full_budget_gradient_steps_estimate": full_steps,
        })

        # --- null, at the same reduced candidate count ----------------------
        t0 = time.time()
        g, w, cands, steps = random_architecture_search(
            bundle, cfg, SEARCH_SEED, res.candidates
        )
        null_wall = time.time() - t0
        rows.append({
            "task": task, "arm": "null", "measured": True,
            "candidates": cands, "wall_time_seconds": null_wall,
            "seconds_per_candidate": null_wall / cands,
            "gradient_steps": steps,
            "steps_per_candidate": steps / cands,
            "validation_accuracy": accuracy(g, w, bundle.validation.X, bundle.validation.y),
            "full_budget_minutes_linear_scaling": null_wall / cands
            * REFERENCE_CANDIDATES / 60.0,
        })

        # --- the two matched arms, at a reduced step budget -----------------
        for arm, build in (
            ("linear", lambda rng: make_linear(rng, layout)),
            ("fixed", lambda rng: make_mixed_mlp(FIXED_HIDDEN, rng, layout)),
        ):
            t0 = time.time()
            g, w, spent, _ = matched_multistart(
                build, bundle, PILOT_MULTISTART_STEPS, MULTISTART_RESTARTS,
                seed=SEARCH_SEED,
            )
            wall = time.time() - t0
            rows.append({
                "task": task, "arm": arm, "measured": True,
                "candidates": MULTISTART_RESTARTS,
                "wall_time_seconds": wall,
                "gradient_steps": spent,
                "seconds_per_gradient_step": wall / max(spent, 1),
                "validation_accuracy": accuracy(
                    g, w, bundle.validation.X, bundle.validation.y
                ),
                "causal_hidden_nodes": int(
                    np.sum([1 for i in range(layout.n_structural, g.n_nodes)])
                ),
                "full_budget_minutes_linear_scaling": wall / max(spent, 1)
                * full_steps / 60.0,
            })
        for row in rows[-4:]:
            print(json.dumps(row), flush=True)

    write_json_atomic(OUT, {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "cost calibration before the v8 freeze; decides which datasets "
                   "are confirmatory and which is a declared extension",
        "scores": "nothing; pilot seeds, one replicate per dataset",
        "method": {
            "search_and_null": f"measured at {PILOT_GENERATIONS} generations "
                               f"({(PILOT_GENERATIONS + 1) * 100} candidates) and "
                               f"scaled linearly to {REFERENCE_CANDIDATES}; linear "
                               "scaling under-estimates the search, whose cost grows "
                               "with its networks, and is right for the null",
            "linear_and_fixed": f"measured at {PILOT_MULTISTART_STEPS:,} gradient "
                                "steps and scaled linearly to the search's estimated "
                                "full-budget step count; linear in steps is right for "
                                "a fixed architecture",
        },
        "split_seed": SPLIT_SEED,
        "search_seed": SEARCH_SEED,
        "rows": rows,
        "fingerprints": fingerprints(),
        "environment": environment(),
    })
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
