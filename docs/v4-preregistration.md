# Protocol v4 — preregistration

**Status: frozen before compute.** Written, gated and committed before any v4
confirmatory run existed. The freeze record is at the end of this file.

## The question

v3 established that Backprop-NEAT's measured advantage over fixed
architectures on these five geometries is produced by the control's training
budget, not by the architectures the search finds. Under v2's protocol a 32×32
tanh network reaches 0.635 on spirals; given the gradient budget the search
actually spent, the same network reaches 0.896, and a fixed network with
heterogeneous operators reaches 0.966 against the search's 0.791.

That is a result about one algorithm. It leaves the obvious objection open: that
Backprop-NEAT is unusually weak, or unusually dependent on an undertrained
control, and that a different topology search would not behave this way.

**v4 asks whether the sign of a search-versus-fixed comparison is set by the
budget protocol or by the search algorithm.**

## The second algorithm

Cartesian Genetic Programming with gradient-trained candidates
(`src/bpneat/v4/cgp.py`, `src/bpneat/v4/search.py`). It was chosen because it
differs from Backprop-NEAT in both places an evolutionary algorithm can differ.

| | Backprop-NEAT | CGP |
|---|---|---|
| Genotype | variable length, grows by mutation | fixed length, 48 function nodes |
| Phenotype | the whole genome | the subgraph the output gene reaches |
| Structural history | innovation numbers | none |
| Recombination | crossover within species | none |
| Population | 100, five species, whole-population replacement | 1 parent + 4 offspring |
| Selection | fitness-proportionate roulette, slack 0.01 | best offspring replaces parent on `>=` |
| Drift mechanism | weak selection on a large population | neutral acceptance of equal-fitness offspring |
| Seed | logistic regression | a uniform random graph |

Everything that is *not* the search algorithm is held identical to v3's
`backprop_neat`: the task geometries, the inner learner (RMSProp with Ha's
rollback rule, 600 nominal updates, batch 10), the penalised validation fitness,
Lamarckian weight inheritance, and the candidate budget
(`population × (generations + 1)`, i.e. 1100 on xor/circle and 2100 on the three
hard geometries).

Two restrictions on CGP, both forced by the evaluator and declared here rather
than discovered later:

1. **Operators.** The eight operators the dense evaluator can express:
   `sigmoid, tanh, relu, gaussian, sin, abs, square, add`. `mult` aggregates by
   product and is excluded. This matters because a 48-node CGP row can decode to
   a chain deeper than the frozen evaluator's 16-tick settling bound, which
   would silently truncate; the dense evaluator is exact at any depth.
2. **Evaluation.** Settled propagation only — the same setting v3's reference
   condition uses. Ha's asynchronous rule is not used by any v4 condition. (It
   is not safe to assume it could be: on 300 random genotypes, 11 decoded
   phenotypes return a constant zero under `ha2016`, and all 300 are alive under
   `settled`.)

## Fresh test data

v3's sealed test was opened once, on the splits its dataset seeds generate.
Reusing those splits would make v4's test numbers a second look at data already
spent confirming a result.

v4 therefore draws **thirty entirely new replicates** — dataset seeds
`50001 + 11i`, search seeds `60001 + 11i` — none of which appears in the burned
set, which now contains every seed v1, v2 and v3 spent plus v4's own pilots
(9002, 19002). The cost of this choice is that v4 and v3 are **not paired**: no
v4 number may be compared with a v3 number by a statistical test. The benefit is
that v4's headline is an out-of-sample replication rather than a reanalysis.
Every v4 claim is a within-v4 paired contrast. Comparison to v3 is qualitative
— does the same thing happen — and is written as such.

## Design

Eight conditions, five tasks, thirty replicates: **1200 runs, 150 cells.** A
cell is one (task, replicate) and is the unit of sharding, because the matched
arms need the budget a reference actually spent in that same cell.

| Condition | Role |
|---|---|
| `bpneat` | released v3 Backprop-NEAT, re-run unchanged. Reference 1. |
| `cgp` | CGP (1+4) with neutral drift. Reference 2. |
| `fixed_tanh_ha` | 32×32 tanh MLP, 60 restarts, Ha's rollback rule — the unmatched control that produced the artifact. Shared by both blocks. |
| `fixed_tanh_matched_bpneat` | same MLP at `bpneat`'s realized gradient budget, no rollback |
| `fixed_mixed_matched_bpneat` | heterogeneous-operator MLP at `bpneat`'s realized budget |
| `fixed_tanh_matched_cgp` | same MLP at `cgp`'s realized budget |
| `fixed_mixed_matched_cgp` | heterogeneous-operator MLP at `cgp`'s realized budget |
| `cgp_random_matched` | CGP genotypes drawn rather than selected, matched on candidates |

The two algorithms get **separate matched arms** because they do not spend the
same budget; a shared matched arm would silently favour whichever algorithm
spent less.

### Blocks

- **E** — does v3's reversal replicate for Backprop-NEAT on fresh data?
  `bpneat`, `fixed_tanh_ha`, `fixed_tanh_matched_bpneat`,
  `fixed_mixed_matched_bpneat`, all tasks.
- **F** — does the same reversal hold for CGP? `cgp`, `fixed_tanh_ha`,
  `fixed_tanh_matched_cgp`, `fixed_mixed_matched_cgp`, `cgp_random_matched`,
  all tasks.
- **G** — do the two algorithms differ from each other? `bpneat`, `cgp`, all
  tasks.

## Analysis, fixed in advance

Primary outcome: **sealed-test accuracy**, paired within (task, replicate).
Each pair is tested with the Wilcoxon signed-rank test; the median paired
difference carries a 10 000-sample bootstrap 95% interval.

Two pre-declared families, Holm-corrected **within** each and never pooled:

- **P1**, anchored on `bpneat`: seven contrasts × five tasks = **35 tests**.
- **P2**, anchored on `cgp`: five contrasts × five tasks = **25 tests**.

Significance means Holm-corrected *p* < 0.05 in the test's own family. Raw
Wilcoxon *p* is reported but is never the basis of a verdict — v3's stability
matrix had one claim that read "supported" on the raw *p* and failed at Holm
*p* = 0.385.

## Hypotheses

**H1 — the artifact replicates.** On every task, `fixed_tanh_matched_bpneat`
beats `fixed_tanh_ha`, significantly in P1. *Fails* if the matched arm does not
beat the unmatched one on at least four of five tasks.

**H2 — the reversal replicates for Backprop-NEAT on fresh data.**
`bpneat` beats `fixed_tanh_ha`, and `fixed_tanh_matched_bpneat` beats `bpneat`,
on at least four of five tasks. *Fails* if the matched arm does not beat
`bpneat` on at least three tasks.

**H3 — the reversal is not Backprop-NEAT-specific.** The same two statements
hold for `cgp` against `fixed_tanh_ha` and `fixed_tanh_matched_cgp`. This is
the study's primary hypothesis. *Fails* if CGP's matched arm does not beat
`cgp` on at least three of five tasks.

**H4 — the sign is set by the protocol, not the algorithm.** For both
algorithms, `search − fixed_unmatched > 0` and `search − fixed_matched < 0`.
Operationalised as: the four signs agree in direction for both algorithms on at
least four of five tasks. *Fails* if the two algorithms disagree in sign on more
than one task.

**H5 — the two algorithms are close to each other relative to the effect of
the protocol.** `|median(cgp − bpneat)| < 0.5 × median(fixed_tanh_matched_cgp −
cgp)` on at least three of five tasks. *Fails* otherwise, in which case the
algorithms differ enough that H4's agreement would need a separate explanation.

**H6 — CGP's selection adds little over its own null.** `cgp` does not beat
`cgp_random_matched` significantly in P2 on more than two of five tasks,
mirroring v3's finding for Backprop-NEAT (where the contrast against
candidate-matched random search failed at Holm *p* = 0.385). *Fails* if CGP
beats its null on four or more tasks — which would be a real difference between
the algorithms and is reportable either way.

No hypothesis is a success criterion for the study. H3 reversing, holding or
failing are all publishable outcomes; what is not permitted is choosing which to
report after seeing the data.

## Firewalls

- **Sealed test.** Touched exactly once, by `make v4-finaltest`, which refuses
  an incomplete release, a release already marked `test_evaluated`, or one whose
  v4, v3 or v2 fingerprint has moved since the suite ran. Champions are loaded
  from the stored records exactly as validation selected them; nothing is
  retrained or reselected.
- **Frozen code.** v4 carries its own fingerprint over
  `cgp, conditions, learners, protocol, search`, and records the released v3
  set (`8438c9e8…`) and the frozen v2 set (`cfdf1fa3…`) in every run. Editing
  either older set invalidates the release.
- **Pilots.** Seeds 9002 and 19002 only, both burned before the freeze.
- **The bridge.** v3's `backprop_neat` is re-run on the first ten v3 replicates
  of every task and compared to the committed records field by field. This is a
  reproducibility gate, not a result; it reads no test split and enters no
  table.

## A known property of the protocol, declared here

Training a single candidate under this learner is **chaotic**. Perturbing a
candidate's initial weights by 1e-12 relative, or evaluating the identical model
through two implementations that agree on values and gradients to 1e-15, moves
the trained weights by order 0.1 within about twenty updates. The mechanism is
RMSProp's epsilon floor: when the gradient-square cache sits below
`SMOOTH_EPS = 1e-8` the update is effectively `100 × lr × grad`, so gradient
differences are amplified a hundredfold per step and then passed through
expansive operators (`square`, `gaussian`).

Two consequences, both binding on v4:

1. The unit of evidence is a distribution over replicates, never an individual
   run. Every v4 claim is a paired contrast over thirty replicates.
2. The equivalence gate on the CGP learner asserts what is true — that the
   evaluators agree on values and gradients to 1e-10 across weight scales
   including the clamped regime, that the first updates coincide, and that the
   full 600-step budget coincides on graphs without expansive operators, where
   the trajectory is stable and the rollback rule actually fires. It does not
   assert agreement of chaotic trajectories, because that would be false.

Measured numbers: `docs/v4-sensitivity.md`. Gates:
`tests/test_v4.py::test_candidate_training_is_chaotic`,
`::test_rollback_matches_the_frozen_rule_on_a_stable_graph`.

## Freeze record

Committed with the v4 gates passing and no v4 run record in existence. The
commit that carries this file and the five fingerprinted modules is the freeze
commit; the tag is `v4-freeze`.

| | |
|---|---|
| v4 science modules | `cgp.py`, `conditions.py`, `learners.py`, `protocol.py`, `search.py` |
| v4 fingerprint | recorded in `results/backprop-neat-v4/manifest.json` |
| v3 fingerprint (frozen) | `8438c9e89c7c72a3…` |
| v2 fingerprint (frozen) | `cfdf1fa3198adc0e…` |
| Planned runs | 1200 across 150 cells |
| Replicates | 30, dataset seeds `50001 + 11i`, search seeds `60001 + 11i` |
