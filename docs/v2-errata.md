# Errata — protocol v2

An external audit in October 2026 found that the v2 engineering is sound and
the v2 *science* is confounded. Every finding was reproduced from the committed
release before anything was changed here; the reproduction script is
`bench/audit_2026_10.py` and its output is
[`audit-2026-10.md`](audit-2026-10.md).

**Nothing in `results/backprop-neat-v2/` has been altered.** The runs are
correct and they regenerate byte-identically; what was wrong is what was
*claimed* about them. These errata narrow the claims. Protocol v3
([`v3-preregistration.md`](v3-preregistration.md)) turns the confounds into the
object of study.

| | Finding | Effect on the claim ladder |
|---|---|---|
| **E1** | Controls were starved of gradient budget | Claims 1 and 4 **withdrawn as stated** |
| **E2** | Mean BCE inverted the XOR ranking | Primary outcome **changed** for v3 |
| **E3** | Propagation and fitness split confounded | Claim 5 **withdrawn** |
| **E4** | The v1 retraction overcorrected | A narrowed v1 phenomenon **restored** |
| **E5** | Protocol documents had drifted | Documents corrected |
| **E6** | "Periodicity for the periodic task" overreads | Claim 3 **narrowed** |

---

## E1 — The controls were starved of gradient budget

**What was claimed.** That on spirals Backprop-NEAT beat a fixed 32×32 MLP in
10/10 paired replicates, and that it did so "with almost no structure: 4.6
causally active hidden nodes against the fixed MLP's 65."

**What the data show.** The comparison is not between two architectures. It is
between two *training budgets*, measured on track B:

| Task | `backprop_neat` steps | `fixed_mlp` steps | ratio |
|---|---:|---:|---:|
| XOR | 92,303 | 6,408 | 14× |
| Circles | 84,621 | 6,566 | 13× |
| Spirals | 132,138 | 2,506 | **53×** |

The cause is Ha's rollback rule in `learn.train`: training breaks at the first
full-batch loss increase, checked every 20 updates. For an 8-node evolved graph
that is a mild regulariser. For a 69-node MLP it is a hard stop after ~42
updates on spirals.

A probe on the validation split alone (never the sealed test), over four
dataset seeds, isolates the learner from the architecture:

| Arm | Realized steps | Validation accuracy |
|---|---:|---|
| 32×32 tanh, Ha's learner | 61–101 | 0.60–0.62 |
| 32×32 tanh, plain RMSProp | 20,000 | 0.78–0.91 |
| 32×32 **sin**, plain RMSProp | 600 | **0.94–0.96** |

The same architecture crosses from near-chance to competent when only the
stopping rule changes. A fixed sin-MLP, in 600 plain steps, beats every
Backprop-NEAT champion in the release on the same data.

**Consequence.** Claims 1 and 4 are **withdrawn as stated**. v2 cannot
distinguish "architecture search helps on deceptive geometry" from "the
evaluator starved the baseline". The *surviving* clean comparisons are the two
evolutionary ablations, which share the candidate budget by construction:

| Spirals, sealed-test accuracy | Mean | Backprop-NEAT wins |
|---|---|---|
| `backprop_neat` | 0.787 | — |
| `homogeneous_tanh` (1,100/2,100 candidates, 124k steps) | 0.745 | 7/10 |
| `evolution_only` (same candidates, 0 steps by construction) | 0.639 | 9/10 |

So *operator diversity* and *gradient learning* remain supported contributions.
*Beating a fixed architecture* does not.

## E2 — Mean BCE inverted the XOR ranking

**What was claimed.** That on XOR "a fixed MLP or random architecture search
matches or beats Backprop-NEAT".

**What the data show.** On sealed-test accuracy Backprop-NEAT beats `fixed_mlp`
in **10/10** replicates, and is perfect (1.000) in **7/10**. It wins on loss in
8/10. Its mean BCE is worse only because two replicates are confidently wrong:
losses of 0.686 and 1.154 against a median of **0.000**.

**Consequence.** The sentence was wrong on its own data. v2's primary outcome —
mean BCE — is fragile under confident errors. v3 makes **accuracy** primary and
reports median and clipped-mean loss beside it.

## E3 — Propagation and fitness split were confounded

**What was claimed.** That "Ha's exact propagation rule measurably changes what
evolution discovers (XOR sealed-test accuracy 0.751 under `ha2016` against
0.996 settled)".

**What the data show.** Track A is `ha2016` **and** training-loss fitness;
track B is `settled` **and** validation fitness. Two factors move together, so
no difference between the tracks is attributable to either one.

**Consequence.** Claim 5 is **withdrawn**. Block B of v3 runs the 2×2 that
would answer it.

## E4 — The v1 retraction overcorrected

**What was claimed.** That v1's finding — Ha's propagation rule collapsing
evolution to logistic regression — "was wrong and is retracted".

**What the data show.** It was *overstated*, not wrong. In track A,
`backprop_neat` champions with **zero** causally active hidden nodes:

| Task | Collapse rate | Collapsed accuracy | Non-collapsed |
|---|---|---|---|
| XOR | **3/10** | 0.535–0.560 (logistic floor) | 0.630–1.000 |
| Spirals | **1/10** | 0.605 | 0.690–0.815 |

Track A's XOR mean of 0.751 is therefore a **bimodal mixture**, not a central
tendency. The phenomenon is real at a rate; v1's error was treating it as
universal, and the retraction's error was denying it entirely.

**Consequence.** The phenomenon is restored in narrowed form: *under `ha2016`
propagation, a minority of runs collapse to the linear floor.* Whether that
depends on propagation or on the fitness split is exactly E3's confound, and
Block B of v3 separates them. Collapse rate is a pre-declared v3 outcome.

## E5 — Protocol documents had drifted

- `docs/protocol-freeze.md` was titled **v1** and specified `elitism 1` — the
  defect that invalidated v1 — yet `docs/writeup.md` cited it as the frozen
  protocol. It is now [`protocol-freeze-v1.md`](protocol-freeze-v1.md) with an
  invalidation banner, and [`protocol-freeze-v2.md`](protocol-freeze-v2.md)
  reconstructs the contract that actually governed the results, labelled as
  written after the fact.
- `docs/v1-invalidation.md` presented single calibration runs as if they were
  v2 results (`ha2016` XOR 1.000; "`ha2016` outperforms settled on spirals,
  0.795 vs 0.730"). The confirmatory values are 0.751 and 0.738 vs 0.787 — the
  spirals sentence runs the other way and is withdrawn.
- The "where this lands" table in `docs/reference-targets.md` was computed
  mid-run over partially completed replicates. It is recomputed from the
  finished release.

## E6 — "Periodicity for the periodic task" overreads the result

**What was claimed.** That evolution "discovered periodicity for the periodic
task", from `sin` being 0.52 of causal operator usage on spirals.

**What the data show.** The operator statistic is correct. The interpretation
is not supported by the boundaries the champions actually draw: the median
spirals champion draws roughly horizontal sine stripes that extend into regions
with no data, rather than a spiral. A periodic operator is being used to tile
the plane, which happens to cut the spiral arms, not to represent the spiral's
rotational structure.

**Consequence.** Claim 3 is narrowed to what is measured: *causal operator
usage differs by task* — `mult` on XOR (0.46), `square`+`gaussian` on circles
(0.53), `sin` on spirals (0.52). No claim is made that the evolved feature
*matches the generative structure* of the task.

---

## What survives

1. Gradient learning and topology search are complementary: `evolution_only`
   loses on all three tasks (XOR 9/10, circles 10/10, spirals 10/10), at
   identical candidate budget.
2. Operator diversity contributes: `homogeneous_tanh` loses on circles (10/10)
   and spirals (7/10), at matched candidate budget.
3. Causal operator usage differs by task.
4. Represented structure is not computation — the causal-subgraph measurement
   stands on its own, independent of any between-condition comparison.
5. The release reproduces: every derived file rebuilds byte-identically from
   the raw records, and `bpneat verify` asserts it in CI on every push.

## What does not

1. That architecture search beats fixed architectures (E1).
2. That Backprop-NEAT is structure-efficient *relative to a fairly trained
   baseline* (E1).
3. That the propagation rule alone changes what evolution finds (E3).
4. That evolution discovered a representation matching the task's generative
   structure (E6).
