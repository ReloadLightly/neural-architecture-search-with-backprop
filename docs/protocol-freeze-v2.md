# Protocol freeze — `v2`

> **Written after the fact.** The v2 freeze existed in code, not as a document:
> it was enforced by `PROTOCOL_VERSION = "v2"` in `src/bpneat/protocol.py` and
> by the seven-module fingerprint `cfdf1fa3198adc0e369466800ede0ea5afa24d99af18000a2161357ae5ec0d75`,
> which the final-test firewall checks against every manifest. That mechanism
> is what actually governed the run; this file reconstructs it in prose from
> `protocol.py`, `conditions.py`, `evolve.py` and `learn.py` at the
> fingerprinted commit, so a reader does not have to read code to audit it.
>
> Nothing here was changed to match the results. Every value below is machine
> readable from the committed modules, and `bpneat verify` checks the
> fingerprint on every push.
>
> The errata that narrow what these results support are in
> [`v2-errata.md`](v2-errata.md). Read both.

## Questions

**Primary.** Under matched and fully reported search budgets, can Backprop-NEAT
discover task-specialised computation graphs that generalise across
independently generated XOR, circle and spiral datasets better than minimal,
fixed, randomly sampled, or evolution-only alternatives?

> **Erratum E1.** The word *matched* in that sentence is not true of the budgets
> as run, and the budgets section below says so plainly. The two statements
> contradicted each other in the original freeze; the budgets section is the
> accurate one. See [`v2-errata.md`](v2-errata.md).

**Propagation.** Does Ha's exact propagation break rule change what evolution
can discover, relative to settling the same propagation to a fixed point?

> **Erratum E3.** As run, this question is not answerable: the two tracks differ
> in propagation *and* in fitness split.

## Tracks — never pooled

| Track | Propagation | Fitness split | Purpose |
|---|---|---|---|
| A | `ha2016` — stop once every node is touched | training loss, source rollback | historical reconstruction |
| B | `settled` — tick to a topology-determined fixed point | validation loss | mechanism comparison |

## Core conditions

| Condition | Mechanism removed | Kind |
|---|---|---|
| `backprop_neat` | — (full mechanism) | evolution |
| `homogeneous_tanh` | operator diversity | evolution |
| `evolution_only` | gradient learning | evolution |
| `random_search` | evolutionary selection | 60 sampled architectures |
| `fixed_mlp` | topology search | 60 restarts, 32×32 tanh |
| `logistic` | all nonlinearity (linear floor) | 60 restarts |

`no_penalty` and `baldwinian` exist in the code as secondary conditions and
were **not** run. Conditions were not dropped after their results were seen.

## Search settings

Population 100, five K-medoids subpopulations, **no elite**, subpopulation
extinction rate 0.5. Parents by fitness-proportionate roulette with weight
`1/(-fitness + 0.01)` over the whole subpopulation — Ha's `pickRandomIndex`.
Mutation: weight rate 0.2 / size 0.5, add-connection 0.5, add-node 0.2.
Ten generations for XOR and circles, twenty for spirals.

## Inner learner

600 nominal minibatch updates per evaluated candidate, batch size 10, RMSProp
with learning rate 0.01, decay 0.999, L2 1e-3 and gradient clipping at 5.0.
Rollback is checked every 20 updates against the full training split and breaks
at the first increase. Fitness is `-error × (1 + 0.03·√connections)`.

> **Erratum E1.** That rollback rule is the starvation mechanism. It is a mild
> regulariser for an 8-node evolved graph and a hard stop for a 69-node fixed
> network, which it halts after ~42 updates on spirals.

## Data

`ha2016` generators, raw unstandardised coordinates, noise 0.5, 200 training /
200 validation / 200 sealed test examples per replicate, independently drawn.

## Replicates

Ten paired replicates; both the dataset seed and the search seed vary.

| Replicate | Dataset seed | Search seed |
|---|---|---|
| 1 | 8101 | 18101 |
| 2 | 8202 | 18202 |
| 3 | 8303 | 18303 |
| 4 | 8404 | 18404 |
| 5 | 8505 | 18505 |
| 6 | 8606 | 18606 |
| 7 | 8707 | 18707 |
| 8 | 8808 | 18808 |
| 9 | 8909 | 18909 |
| 10 | 9010 | 19010 |

All conditions within a task and replicate share the same data. Seeds burned on
pilots or calibration — 1103, 2203, 3301, 4409, 5501, 6607, 7103, 17103 — are
excluded, and `protocol.validate()` asserts it.

## Budgets — *not* a compute match

Evolutionary conditions evaluate `population × (generations + 1)` candidates:
1,100 on XOR and circles, 2,100 on spirals. Multistart and random-search
controls use **60 restarts**. This is a candidate count, not a compute match,
and candidate evaluations, realized gradient steps and wall time are reported
separately for exactly that reason.

> **Erratum E1.** Reporting the asymmetry is not the same as controlling for
> it. Measured on track B, spirals: `fixed_mlp` spent 2,506 realized gradient
> steps in total against `backprop_neat`'s 132,138 — a factor of 53. Claims 1
> and 4 of the v2 claim ladder do not survive this. Protocol v3 adds
> budget-matched controls.

## Outcomes

**Primary:** sealed-test binary cross-entropy of the validation-selected
champion, paired within task and replicate.

> **Erratum E2.** Mean BCE is fragile. On XOR it is dominated by two
> overconfident replicates while Backprop-NEAT is perfect in 7/10 and ahead of
> `fixed_mlp` 10/10 on accuracy. v3 makes accuracy primary and reports median
> and clipped-mean loss beside it.

**Secondary:** sealed-test accuracy; validation accuracy and loss;
generalisation gap; success at frozen thresholds (XOR 0.90, circles 0.90,
spirals 0.80); represented nodes and connections; causally active hidden nodes
and connections; causal operator frequencies; candidate evaluations, realized
gradient steps and wall time; across-replicate dispersion.

## Analysis

Every replicate reported, not only means. Mean and median paired difference
with a paired bootstrap 95% interval. Success counts and rates. Tracks never
pooled; tasks never collapsed into one headline number.

## Sealed test

Evaluated exactly once per track, by `bpneat final-test`, after the suite is
complete and the fingerprint matches. Champions are loaded as validation
selected them and are never retrained or reselected.

## Compute as run

360 runs (2 tracks × 6 conditions × 3 tasks × 10 replicates), 268,800 candidate
evaluations, 11,653,669 realized gradient steps, 1.87 core-hours.
