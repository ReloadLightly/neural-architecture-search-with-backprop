> **This document describes protocol v1, which is INVALIDATED.**
>
> It is kept because the v1 freeze is part of the record, not because it
> governs anything. Its "elitism 1" selection setting is precisely the defect
> that invalidated v1 — see [`v1-invalidation.md`](v1-invalidation.md). The
> contract that actually governed the committed results is
> [`protocol-freeze-v2.md`](protocol-freeze-v2.md), and the live contract for
> new work is [`v3-preregistration.md`](v3-preregistration.md).
>
> No number may be cited from a v1 run.

# Protocol freeze — `v1`

Frozen before the confirmatory run. Nothing below may change without a new
protocol version and a new result identity. Fixes that alter scientific output
invalidate this version rather than amending it.

## Questions

**Primary.** Under matched and fully reported search budgets, can Backprop-NEAT
discover task-specialised computation graphs that generalise across
independently generated XOR, circle and spiral datasets better than minimal,
fixed, randomly sampled, or evolution-only alternatives?

**Propagation.** Does Ha's exact propagation break rule change what evolution can
discover, relative to settling the same propagation to a fixed point?

## Tracks — never pooled

| Track | Propagation | Fitness | Purpose |
|---|---|---|---|
| A | `ha2016` — stop once every node is touched | training loss, source rollback | historical reconstruction |
| B | `settled` — tick to a topology-determined fixed point | validation loss | mechanism comparison |

Under `ha2016` the output node holds id 3 and is recomputed before every hidden
node, so it reads its operands from the previous tick. A genome that keeps a
direct input or bias edge into the output therefore has any hidden structure
added on top of it ignored by fitness. Seed genomes are exactly that shape.
This is a property of the source algorithm and is measured, not corrected away.

## Core conditions

| Condition | Mechanism removed |
|---|---|
| `backprop_neat` | — (full mechanism) |
| `homogeneous_tanh` | operator diversity |
| `evolution_only` | gradient learning |
| `random_search` | evolutionary selection |
| `fixed_mlp` | topology search |
| `logistic` | all nonlinearity (linear floor) |

`no_penalty` and `baldwinian` are declared secondary and are not part of this
freeze. Conditions may not be dropped after their validation results are seen.

## Budgets

Population 100, five K-medoids species, elitism 1. Ten generations for XOR and
circles, twenty for spirals. Six hundred inner minibatch updates per evaluated
candidate, batch size 10, RMSProp lr 0.01 / decay 0.999 / regc 1e-3 / clip 5.0,
rollback checked every 20 updates against the full training split. Fitness
`-error x (1 + 0.03*sqrt(connections))`. Multistart and random-search controls
use 60 restarts; this is a candidate count, not a compute match, and candidate
evaluations, realized gradient steps and wall time are reported separately.

## Data

`ha2016` generators, raw unstandardised coordinates, noise 0.5, 200 training /
200 validation / 200 sealed test examples per replicate, independently drawn.

## Replicates

Ten paired replicates. Both the dataset seed and the search seed vary, so
across-replicate spread reflects data sampling as well as search stochasticity.

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
pilots, calibration or smoke runs — 1103, 2203, 3301, 4409, 5501, 6607, 7103,
17103 — are excluded and the contract asserts it.

## Outcomes

**Primary:** sealed-test binary cross-entropy of the validation-selected
champion, analysed as paired differences within task and replicate.

**Secondary:** sealed-test accuracy; validation accuracy and loss;
generalisation gap; success at frozen thresholds (XOR 0.90, circles 0.90,
spirals 0.80); represented nodes and connections; causally active hidden nodes
and connections; causal operator frequencies; candidate evaluations, realized
gradient steps and wall time; across-replicate dispersion.

## Analysis

Every replicate reported, not only means. Mean and median paired difference with
a paired bootstrap 95% interval. Success counts and rates. Effect size and
uncertainty over binary significance. Negative and null results reported.
Tracks are never pooled; tasks are never collapsed into one headline number.

## Sealed test

Evaluated exactly once, by `bpneat.cli final-test`, after the suite is complete
and every fingerprint matches this freeze commit. Champions are loaded as
validation selected them and are never retrained or reselected. Test results may
change interpretation; they may not trigger retuning under this version.

## Expected compute

360 runs (2 tracks x 6 conditions x 3 tasks x 10 replicates). Measured at 6.6
minutes per 18-run replicate slice on 4 cores, so roughly 2.2 hours serial and
about 35 minutes across four shards.
