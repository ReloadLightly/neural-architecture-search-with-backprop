# Protocol v3 preregistration — "When the evaluator decides"

**Frozen at tag `v3-freeze`, before any confirmatory compute.**
Protocol version `v3`; v3 science fingerprint over
`conditions, datasets, dense, learners, protocol, search, selection`.
The seven v2 modules are imported, never edited: a v3 gate asserts the v2
fingerprint is still `cfdf1fa3198adc0e369466800ede0ea5afa24d99af18000a2161357ae5ec0d75`.

Nothing below may change once this is tagged. A change that alters scientific
output requires a new version and a new result identity; results are never
patched under an unchanged version.

## Thesis

Implementation details absent from the published description of an
evolution-with-learning algorithm — the learner's stopping rule, forward-pass
scheduling, the selection operator, weight inheritance — can each decide the
study's conclusions. v3 measures each as a controlled factor.

This is not a hypothetical worry. Protocol v1 and protocol v2 of this same
repository differed only in the selection operator and reached *opposite*
conclusions about Ha's propagation rule
([`v1-invalidation.md`](v1-invalidation.md)), and v2's headline spirals result
rests on a control that its own learner stopped after ~42 gradient steps
([`audit-2026-10.md`](audit-2026-10.md)).

## Blocks

`backprop_neat` is the shared reference cell: it is the (settled, validation)
corner of Block B, the `roulette_s0.01` level of Block C, and the Lamarckian
level of Block D. It is run once per cell and referenced by all four blocks, so
the blocks are analysis views over one run plan.

### Block A — controls and budgets (5 tasks)

| Condition | What it isolates |
|---|---|
| `backprop_neat` | the full mechanism |
| `fixed_mlp_tanh_ha` | v2's control, kept to exhibit the artifact |
| `fixed_mlp_tanh_matched` | same net, same compute, no rollback |
| `fixed_mlp_sin_matched` | operator bias without search |
| `fixed_mlp_mixed_matched` | heterogeneous operators without search |
| `random_search_matched` | selection removed, matched on candidates |
| `homogeneous_tanh` | operator diversity removed |
| `evolution_only` | gradient learning removed |

**Budget matching is paired within replicate.** The matched conditions receive
the gradient budget `backprop_neat` actually spent *in that same cell*, spread
over 60 restarts, selected on validation. The cell is therefore the shard unit.
`random_search_matched` is matched on candidate evaluations
(`population × (generations + 1)` = 1,100 or 2,100) and reported against
gradient steps, because the two budgets cannot be matched simultaneously.

The matched learner is Ha's RMSProp with **one thing removed**: the rollback
stopping rule. Learning rate 0.01, decay 0.999, L2 1e-3, clip 5.0, batch 10 —
all identical to v2.

### Block B — propagation × fitness split (3 tasks)

A 2×2 on `backprop_neat`: propagation ∈ {`ha2016`, `settled`} × fitness split ∈
{train, validation}. v2 confounded these two factors; this separates them.

### Block C — selection-pressure dose response (xor, spiral)

Seven levels: roulette with slack ∈ {1.0, 0.1, **0.01 = Ha's own**, 0.001},
tournament k ∈ {2, 4}, and the v1 operator (elitism 1 + truncation to the
better half) at the strong end.

Levels are plotted against the **realized standardised selection intensity**
`I = (Σ pᵢfᵢ − mean f) / sd f`, recorded each generation from the actual
population. Breeding odds of best over median are undefined for truncation
(the median can have probability zero), so intensity is the comparable axis.

### Block D — Lamarckian vs Baldwinian (3 tasks)

`backprop_neat` (weights inherited) against `baldwinian` (weights
re-initialised each generation, so selection still favours learnable structure
but nothing learned is passed on).

## Tasks

Ha's `xor`, `circle`, `spiral` (noise 0.5, 200/200/200 independent splits),
plus two harder geometries defined in `bpneat.v3.datasets`:

- **`checkerboard`** — 4×4 parity grid on [−5, 5]², cell side 2.5, Gaussian
  noise 0.5. Disconnected and parity-structured: no single radial or sinusoidal
  feature expresses it.
- **`spiral3`** — three-arm, 2.5-turn spiral on radius 6. Arm 0 carries class 1
  with half the points; arms 1 and 2 carry class 0 with a quarter each, so
  labels stay balanced while the boundary is strictly harder than the two-arm
  original.

**Real datasets: skipped, and here is why.** The genome encoding fixes two
input nodes (`IN_X`, `IN_Y`) in `bpneat.genome`, which is a frozen v2 module.
Supporting d > 2 would mean editing it, which R3 forbids. The stretch goal is
therefore not attempted rather than attempted badly.

## Replicates

30 paired replicates. Dataset seed `30001 + 7i`, search seed `40001 + 7i`,
i = 0…29. All conditions within a task and replicate share the same data.
Every v1/v2 confirmatory and calibration seed, and the v3 pilot seeds
(9001/19001), are burned; `protocol.validate()` asserts no overlap.

## Outcomes

**Primary: sealed-test accuracy**, paired within replicate. v2 used mean BCE,
which erratum E2 shows is fragile — on XOR it inverted the ranking because of
two overconfident replicates.

**Statistics.** Wilcoxon signed-rank on paired differences; bootstrap 95%
interval of the **median** paired difference (10,000 resamples, seed 20261004);
Holm correction across the pre-declared family of 35 comparisons
(`backprop_neat` against each of the other seven Block A conditions, on each of
the five tasks). Blocks B, C and D are reported with intervals and are
explicitly **exploratory** — not in the Holm family.

**Secondary:** median and 5%-clipped mean BCE; causal hidden nodes and
connections; collapse rate (0 causally active hidden nodes); operator usage;
realized gradient steps; candidate evaluations; wall time; selection intensity.

## Hypotheses, with directions

- **H1.** Under budget-matched plain training, the fixed tanh MLP's spirals
  deficit against `backprop_neat` disappears or reverses.
- **H2.** `fixed_mlp_sin_matched` matches or beats `backprop_neat` on spirals —
  operator bias, not search.
- **H3.** Collapse rate depends on propagation, not on the fitness split.
- **H4.** Collapse rate and causal size vary monotonically, or in an inverted
  U, with selection intensity; the v1 operator sits at the collapse end.
- **H5.** Lamarckian vs Baldwinian — two-sided, no direction predicted.

A hypothesis that fails is reported as failed. H1 and H2 failing would
*restore* v2's claims; that outcome is as publishable as the other.

## Central figure — the conclusion stability matrix

Rows: the five v2 headline claims. Columns: evaluator settings (v2 as run;
budget-matched; sin-operator control; each propagation × fitness cell; the
selection-pressure extremes). Cells: supported / reversed / not significant.

## Engineering gates — all passed before this freeze

- **Dense fast path.** A vectorized evaluator for acyclic sum-aggregating
  genomes, grouping nodes by topological depth. Agreement with the frozen
  genome path is asserted to 1e-10 on forward values *and* gradients across
  tanh/sin/relu/gaussian and mixed-operator networks, and training through
  either path from the same seed gives identical weights. Measured speedup 41×
  (16.3 ms → 0.40 ms per step on a 69-node MLP); a 132k-step matched run falls
  from 36 min to 0.9 min. `mult` aggregates by product and is not expressible
  as one matrix product, so such genomes raise and fall back.
- **v3 fingerprinting**, disjoint from v2's, recorded in every record.
- **Sealed-test isolation**: poisoning the test split with NaN leaves search
  output unchanged; no record carries a test metric.
- **Resume exactness**: re-running a finished shard changes no bytes.
- **Final-test firewall**: refuses an incomplete suite, an already-evaluated
  release, a changed v3 fingerprint, or a changed v2 fingerprint.

29 v3 gates, 87 in total, ruff clean.

## Measured compute forecast

From a complete pilot cell (spiral, replicate 1, 13 of 18 conditions in 679 s
wall), per-run costs are 2–115 s; the dominant arms are
`random_search_matched` (115 s), `fixed_mlp_mixed_matched` (109 s) and
`fixed_mlp_sin_matched` (76 s).

| Cell | Conditions | Est. wall |
|---|---:|---:|
| spiral | 18 | ~930 s |
| xor | 18 | ~600 s |
| circle | 12 | ~380 s |
| checkerboard | 8 | ~480 s |
| spiral3 | 8 | ~480 s |
| **per replicate** | | **~2,870 s** |

**1,920 distinct runs, 150 cells, ≈ 24 core-hours, ≈ 6 h wall across 4 shards.**
Shards are cell-sharded, resumable and idempotent; a container restart costs at
most the run in flight.

## Sealed test

Evaluated exactly once, by `bpneat.v3.finaltest`, after the suite is complete
and both fingerprints match. Champions are loaded as validation selected them
and are never retrained or reselected. Test results may change the
interpretation; they may not trigger retuning under this version.

---

## Freeze record

This preregistration was committed and pushed as
**`a343c665b37ddb0d5a7ae97b0bc562c6fce73d34`** on `main`, before any
confirmatory run started. That commit is the freeze.

An annotated tag `v3-freeze` exists locally but could not be published: this
session's GitHub proxy refuses both tag pushes and writes to the git-refs API.
The commit SHA is the stronger record in any case — a tag can be moved, a
commit hash cannot — and every v3 record carries the v3 fingerprint that binds
it to this contract.
