"""Measure what each v6 rung costs, on the pilot seeds, before the freeze.

The v6 ladder is a cost decision as much as a design decision: the search's
expense is superlinear in its generation count, and how superlinear decides
whether the top rung is confirmatory or an extension. That number has to be
measured rather than assumed, and the measurement has to be on the record, so
this script writes one.

Pilot seeds only (9004, 19004), burned in ``v6.protocol.BURNED_SEEDS`` before the
freeze. Spirals only — the most expensive of the three geometries, so it bounds
the others. Scores no hypothesis and never reads a sealed test split.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpneat.record import environment, write_json_atomic  # noqa: E402
from bpneat.v6.conditions import run_condition  # noqa: E402
from bpneat.v6.fingerprint import fingerprints  # noqa: E402
from bpneat.v6.protocol import BURNED_SEEDS, PROTOCOL_VERSION  # noqa: E402

TASK = "spiral"
DATA_SEED, SEARCH_SEED = 9004, 19004
REPLICATE = 0  # not a replicate of the plan; a pilot

#: The rungs whose cost decides the ladder, plus one of each other arm so the
#: three arms' per-candidate costs are on the record too.
PLAN = ("search_b500", "search_b1000", "search_b2100", "search_b6300",
        "null_b500", "null_b1000", "fixed_b2100")

OUT = Path("results/backprop-neat-v6/pilot/cost.json")


def main() -> int:
    assert DATA_SEED in BURNED_SEEDS and SEARCH_SEED in BURNED_SEEDS, "pilot seeds must be burned"
    rows: list[dict] = []
    steps_by_condition: dict[str, int] = {}
    for cond in PLAN:
        matched = {k: v for k, v in steps_by_condition.items()} or None
        rec = run_condition(cond, TASK, REPLICATE, DATA_SEED, SEARCH_SEED,
                            matched_steps=matched)
        steps_by_condition[cond] = rec["compute"]["gradient_steps"]
        row = {
            "condition": cond,
            "arm": rec["arm"],
            "budget": rec["budget"],
            "task": TASK,
            "candidate_budget": rec["candidate_budget"],
            "generations": rec["config"]["generations"],
            "gradient_steps": rec["compute"]["gradient_steps"],
            "steps_per_candidate": rec["compute"]["gradient_steps"]
            / rec["candidate_budget"],
            "wall_time_seconds": rec["compute"]["wall_time_seconds"],
            "validation_accuracy": rec["metrics"]["validation_accuracy"],
            "causal_hidden_nodes": rec["metrics"]["causal_hidden_nodes"],
            "matched_steps": rec["config"].get("matched_steps"),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    write_json_atomic(
        OUT,
        {
            "protocol_version": PROTOCOL_VERSION,
            "purpose": "cost calibration before the v6 freeze; decides which rungs "
                       "are confirmatory and which is a declared extension",
            "scores": "nothing; pilot seeds, one geometry, one pilot replicate",
            "task": TASK,
            "dataset_seed": DATA_SEED,
            "search_seed": SEARCH_SEED,
            "rows": rows,
            "fingerprints": fingerprints(),
            "environment": environment(),
        },
    )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
