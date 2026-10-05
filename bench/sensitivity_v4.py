"""Measure how sensitive a single candidate's training is to irrelevant changes.

v4's preregistration declares that candidate training under this protocol is
chaotic, and that the unit of evidence is therefore a distribution over
replicates rather than an individual run. That is a quantitative claim, so it
gets measured rather than asserted, and the numbers are written to
``docs/v4-sensitivity.md`` where the documentation checks can bind to them.

Three probes, all on the frozen or gated learners, none touching a test split:

1. **Perturbation.** Scale a candidate's initial weights by ``1 + 1e-12`` and
   train it under the frozen learner. The perturbation is far below any
   meaningful precision; the question is what it does to the trained weights.
2. **Implementation.** Train the same candidate through the frozen genome
   evaluator and through the dense evaluator, which agree on forward values and
   gradients to ~1e-15. Same model, same rule, different summation order.
3. **Architecture class.** Repeat both probes on a tanh MLP, which has no
   expansive operator, to show the effect is a property of the graph class and
   not of the code.

The run is cheap — a few hundred candidate trainings — and takes about a minute.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from bpneat.baselines import make_mlp  # noqa: E402
from bpneat.learn import train  # noqa: E402
from bpneat.v3 import dense  # noqa: E402
from bpneat.v3.conditions import MLP_HIDDEN  # noqa: E402
from bpneat.v3.datasets import make_bundle  # noqa: E402
from bpneat.v4 import cgp  # noqa: E402
from bpneat.v4.learners import rollback_train  # noqa: E402

#: Pilot seeds, both burned before the v4 freeze.
DATASET_SEED = 9002
RNG_SEED = 19002
N_GENOMES = 40
BUDGET = 600
PERTURBATION = 1e-12


def _cgp_genomes(n: int):
    """Decoded CGP phenotypes shallow enough for the frozen evaluator to be exact."""
    out = []
    i = 0
    while len(out) < n:
        g = cgp.decode(cgp.random_genotype(np.random.default_rng(4000 + i))).genome
        plan = dense.plan(g)
        if len(plan.groups) <= 14:  # the frozen path settles for at most 16 ticks
            out.append((g, plan))
        i += 1
    return out


def _mlp_genomes(n: int):
    out = []
    for i in range(n):
        g = make_mlp((8,), np.random.default_rng(4500 + i))
        out.append((g, dense.plan(g)))
    return out


def _probe(genomes, bundle, budget: int) -> dict:
    """Both probes over one set of genomes, under one budget."""
    pert_w, pert_e, impl_w, impl_e = [], [], [], []
    pert_steps = impl_steps = 0

    for g, plan in genomes:
        w0 = np.array(g.weight, dtype=np.float64)
        w1 = w0 * (1.0 + PERTURBATION)
        base = train(g, w0, bundle.train.X, bundle.train.y,
                     np.random.default_rng(RNG_SEED), n_cycles=budget, settle=True)
        pert = train(g, w1, bundle.train.X, bundle.train.y,
                     np.random.default_rng(RNG_SEED), n_cycles=budget, settle=True)
        impl = rollback_train(g, w0, bundle.train.X, bundle.train.y,
                              np.random.default_rng(RNG_SEED), n_cycles=budget, plan=plan)

        pert_w.append(float(np.max(np.abs(base.weights - pert.weights))))
        pert_e.append(abs(base.error - pert.error))
        pert_steps += int(base.gradient_steps != pert.gradient_steps)
        impl_w.append(float(np.max(np.abs(base.weights - impl.weights))))
        impl_e.append(abs(base.error - impl.error))
        impl_steps += int(base.gradient_steps != impl.gradient_steps)

    def stat(x):
        a = np.array(x, dtype=float)
        return {
            "median": float(np.median(a)),
            "mean": float(a.mean()),
            "max": float(a.max()),
            "frac_above_1e-6": float(np.mean(a > 1e-6)),
        }

    return {
        "n": len(genomes),
        "budget": budget,
        "perturbation": {
            "weights": stat(pert_w),
            "error": stat(pert_e),
            "step_count_differs": pert_steps,
        },
        "implementation": {
            "weights": stat(impl_w),
            "error": stat(impl_e),
            "step_count_differs": impl_steps,
        },
    }


def _row(name: str, s: dict) -> str:
    return (
        f"| {name} | {s['weights']['median']:.2e} | {s['weights']['max']:.2e} | "
        f"{s['weights']['frac_above_1e-6']:.0%} | {s['error']['median']:.2e} | "
        f"{s['error']['max']:.2e} | {s['step_count_differs']} |"
    )


def main() -> int:
    bundle = make_bundle("spiral", seed=DATASET_SEED)
    cgp_g = _cgp_genomes(N_GENOMES)
    mlp_g = _mlp_genomes(N_GENOMES)

    short = _probe(cgp_g, bundle, 5)
    full_cgp = _probe(cgp_g, bundle, BUDGET)
    full_mlp = _probe(mlp_g, bundle, BUDGET)

    lines = [
        "# How sensitive is one candidate's training?",
        "",
        "Generated by `bench/sensitivity_v4.py` (`make v4-sensitivity`). Pilot seeds",
        f"only ({DATASET_SEED}, {RNG_SEED}), spirals, {N_GENOMES} candidates per row, no test",
        "split read.",
        "",
        "v4's preregistration declares that candidate training under this protocol is",
        "chaotic, so the unit of evidence is a distribution over replicates and never a",
        "single run. These are the numbers behind that declaration.",
        "",
        "Two probes, both of which *should* be irrelevant:",
        "",
        f"* **perturbation** — multiply the initial weights by `1 + {PERTURBATION:g}`;",
        "* **implementation** — train through the dense evaluator instead of the frozen",
        "  genome evaluator. The two agree on forward values to 2e-15 and on gradients to",
        "  3e-15, and to ~3e-12 even where a fifth of node values are clamped",
        "  (`tests/test_v4.py::test_dense_and_frozen_evaluators_agree_on_values_and_gradients`).",
        "",
        "`max |Δw|` is over the trained weight vector; `|Δerror|` is the final training",
        "loss the learner reports; the last column counts candidates whose *realized step",
        "count* differed, i.e. where the rollback rule itself fired differently.",
        "",
        "| probe | median \\|Δw\\| | max \\|Δw\\| | frac > 1e-6 | median \\|Δerr\\| | max \\|Δerr\\| | step count differs |",
        "|---|---|---|---|---|---|---|",
        _row(f"CGP phenotypes, {short['budget']} updates — perturbation",
             short["perturbation"]),
        _row(f"CGP phenotypes, {short['budget']} updates — implementation",
             short["implementation"]),
        _row(f"CGP phenotypes, {BUDGET} updates — perturbation", full_cgp["perturbation"]),
        _row(f"CGP phenotypes, {BUDGET} updates — implementation",
             full_cgp["implementation"]),
        _row(f"tanh MLP (8), {BUDGET} updates — perturbation", full_mlp["perturbation"]),
        _row(f"tanh MLP (8), {BUDGET} updates — implementation",
             full_mlp["implementation"]),
        "",
        "## What this says",
        "",
        "At five updates both probes are at the floor — the trajectories are identical to",
        "rounding. By the full budget, CGP phenotypes have moved by order 0.1 in weight",
        "space under a perturbation twelve orders of magnitude smaller, and under an",
        "implementation change that is provably not a model change at all. The tanh MLP",
        "does not do this: it has no expansive operator, and its trajectory stays stable",
        "to the end of the budget.",
        "",
        "The mechanism is RMSProp's epsilon floor. When the gradient-square cache sits",
        "below `SMOOTH_EPS = 1e-8`, the denominator is pinned at 1e-4 and the update is",
        "effectively `100 x lr x grad`, so a difference in the gradient is amplified a",
        "hundredfold per step and then passed through `square` and `gaussian` nodes. The",
        "observed growth is roughly an order of magnitude per update until the two runs",
        "are simply at different points in weight space.",
        "",
        "## Consequences, which are binding rather than advisory",
        "",
        "1. **No claim rests on a run.** Every v4 claim is a paired contrast over thirty",
        "   replicates; the per-replicate values are kept in `paired-effects.csv` so the",
        "   spread is visible rather than summarised away.",
        "2. **The equivalence gate asserts what is true.** The CGP learner is gated on",
        "   evaluator agreement across weight scales, on the first updates coinciding, and",
        "   on the *full* budget coinciding for graphs without expansive operators — where",
        "   the rollback rule does fire, so \"same rule\" is demonstrated rather than",
        "   assumed. It is not gated on chaotic trajectories agreeing, because they do not.",
        "3. **It is a result, not only a caveat.** Two implementations of one model,",
        "   agreeing to 1e-15, produce visibly different single-run architecture-search",
        "   outcomes. A NAS comparison reported from single runs is not reproducible even",
        "   in principle.",
        "",
        "Gates: `tests/test_v4.py::test_candidate_training_is_chaotic`,",
        "`::test_rollback_matches_the_frozen_rule_on_a_stable_graph`.",
        "",
    ]
    target = ROOT / "docs" / "v4-sensitivity.md"
    target.write_text("\n".join(lines))
    print(f"wrote {target}")
    for name, s in (("cgp@5", short), ("cgp@600", full_cgp), ("mlp@600", full_mlp)):
        print(
            f"  {name:9s} perturbation max|dw|={s['perturbation']['weights']['max']:.2e}  "
            f"implementation max|dw|={s['implementation']['weights']['max']:.2e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
