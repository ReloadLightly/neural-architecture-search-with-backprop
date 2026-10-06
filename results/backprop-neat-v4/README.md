# Protocol v4 — a second search algorithm through the same controls

**Question.** v3 showed that Backprop-NEAT's measured advantage over fixed
architectures comes from the control's training budget. That is a result about
one algorithm. v4 asks whether the *sign* of a search-versus-fixed comparison is
set by the budget protocol or by the search algorithm, by running a second,
independently derived topology search — Cartesian Genetic Programming with
gradient-trained candidates — through the same controls, on dataset seeds no
earlier release has opened.

**Answer, in one line.** The artifact is universal; the reversal is not.

| | |
|---|---|
| Runs | 1200 across 150 cells, zero failures |
| Design | 8 conditions × 5 geometries × 30 paired replicates |
| Candidate evaluations | 809,550 |
| Gradient updates | 101,852,179 |
| Compute | 10.93 core-hours |
| Sealed test | evaluated once, on dataset seeds `50001 + 11i` — splits no earlier release opened |
| v4 fingerprint | `cd9d0c1f60b44c8d…`, identical across all 1200 records |
| v3 fingerprint | `8438c9e89c7c72a3…` (released, unchanged throughout) |
| v2 fingerprint | `cfdf1fa3198adc0e…` (frozen, unchanged throughout) |
| Reproducibility bridge | 50/50 v3 cells reproduce bit-for-bit |

Preregistration: [`docs/v4-preregistration.md`](../../docs/v4-preregistration.md),
committed and tagged `v4-freeze` before any run record existed.

## The preregistered scorecard

Scored by [`hypotheses.csv`](hypotheses.csv), using the decision rules written
down before the suite ran. **Four of six fail.**

| | Hypothesis | Rule | Observed | |
|---|---|---|---|---|
| H1 | the unmatched control is starved, not weak | matched beats unmatched on ≥4 of 5 tasks | 5/5 | **holds** |
| H2 | Backprop-NEAT's advantage reverses under a matched budget | search>unmatched on ≥4 and matched>search on ≥3 | beats unmatched 3/5, loses to matched 1/5 | **fails** |
| H3 | the same for CGP | as H2 | beats unmatched 2/5, loses to matched 2/5 | **fails** |
| H4 | the sign is set by the budget protocol, not the algorithm | both signs agree on ≥4 of 5 tasks | 2/5 | **fails** |
| H5 | the algorithms differ less than the protocol does | gap < 0.5× the matching effect on ≥3 of 5 | 2/5 | **fails** |
| H6 | CGP's selection adds little over its candidate-matched null | CGP beats its null on ≤2 of 5 | 0/5 | **holds** |

No hypothesis was a success criterion. H3 reversing, holding or failing were all
publishable; what the preregistration forbids is choosing which to report after
seeing the data.

## What the sealed test says

Mean accuracy over 30 paired replicates, from [`summary.csv`](summary.csv).
Search conditions are marked \*.

| condition | XOR | circles | spirals | checkerboard | 3-arm spiral |
|---|---|---|---|---|---|
| Backprop-NEAT \* | 0.990 | 0.984 | 0.791 | 0.621 | 0.590 |
| CGP (1+4) \* | 0.989 | 0.986 | 0.741 | 0.584 | 0.558 |
| fixed tanh, **unmatched** | 0.983 | 0.971 | 0.640 | 0.570 | 0.528 |
| fixed tanh @ BP-NEAT budget | 0.986 | 0.979 | 0.898 | 0.686 | 0.583 |
| fixed tanh @ CGP budget | 0.986 | 0.980 | 0.867 | 0.695 | 0.572 |
| fixed **mixed** @ BP-NEAT budget | 0.977 | 0.985 | **0.964** | **0.714** | **0.783** |
| fixed **mixed** @ CGP budget | 0.978 | 0.985 | 0.960 | 0.713 | 0.753 |
| random CGP, candidate-matched | 0.989 | 0.985 | 0.757 | 0.694 | 0.601 |

### 1. The artifact replicates perfectly (H1)

On every one of the five geometries, the same 32×32 tanh network scores higher
once it is given the gradient budget the search actually spent. On spirals that
is **0.640 → 0.898**, and its success rate goes from 0 of 30 replicates to 30 of
30. v3's diagnosis holds out of sample, on splits v3 never opened.

The budget gap is not subtle and is published in
[`budget-table.csv`](budget-table.csv): on spirals the unmatched control spends
**2,586** gradient updates where Backprop-NEAT spends **131,739**. Ha's rollback
rule stops a 69-node network after about 43 updates per restart.

### 2. The reversal is geometry-dependent, not general (H2, H3 fail)

Search losing to a fairly trained control happens on spirals for both
algorithms, and on checkerboard for CGP. It does not happen elsewhere. On XOR
and circles every condition sits above 0.97 and nothing separates. The
thresholds of ≥4 and ≥3 tasks were set in advance; the data gave 1 and 2.

This is the main correction v4 makes to how v3's result should be read. v3's
headline is a true statement about the deceptive geometry, not a law about
architecture search.

### 3. The two algorithms agree less than predicted (H4, H5 fail)

H4 asks for the unmatched and matched-tanh verdicts to agree across the two
algorithms; they do on 2 of 5 tasks. Column by column, from
[`sign-matrix.csv`](sign-matrix.csv): unmatched 4/5, matched tanh 3/5, matched
mixed 5/5.

H5 fails on its own magnitude rule (2/5), but the significance picture differs
and both belong in the record: from
[`cross-algorithm.csv`](cross-algorithm.csv), the two algorithms are
statistically indistinguishable on four of five geometries, and the one
significant difference is the 3-arm spiral (median −0.032, Holm *p* = 0.037).
H5 fails because on tasks where the matching effect is near zero, any gap
exceeds half of it — not because the algorithms are far apart.

> **Post-hoc, and labelled as such.** Against the *strongest* control — mixed
> operators at a matched budget — the two algorithms' verdicts are identical on
> all five geometries: both win on XOR, both tie on circles, both lose on all
> three hard ones. H4's preregistered rule reads the matched-**tanh** column, so
> this 5/5 agreement is an observation for a future study to test, not a result
> of this one.

### 4. Selection is the component that does not pay (H6)

CGP never beats its candidate-matched null on any geometry. On the two hardest
the null **beats it significantly**: checkerboard median −0.090 (Holm
*p* = 0.000) and 3-arm spiral −0.047 (*p* = 0.011). Unselected random CGP
genotypes are the better search arm on both.

The mechanism is visible in the phenotypes. Selection drives CGP to small active
graphs — 3.9 active function nodes on spirals against 8.0 for the unselected
control, and 0.8 against 7.8 on checkerboard — in an encoding that is *not*
structurally capped, since all 48 function nodes are addressable from the first
generation. Both algorithms inherit Ha's complexity penalty
(`1 + 0.03·√connections`), which is the plausible shared cause. **v4 contains no
penalty-off arm, so nothing here isolates it.**

Together with v3, where Backprop-NEAT did not beat candidate-matched random
search after correction, that is two independently derived algorithms whose
search component contributes nothing measurable on these tasks.

### 5. The operator prior dominates

The largest effects in the release belong to no search at all. A fixed 32×32
network whose hidden operators are drawn from the reference set reaches 0.964 on
spirals and 0.783 on the 3-arm spiral, against the best search's 0.791 and
0.590. This is not a claim that heterogeneous operators are the answer; it is a
demonstration that the operator prior can account for an outcome that would
otherwise be attributed to search.

## Reproducibility

The bridge ([`bridge/replication.json`](bridge/replication.json)) re-runs v3's
`backprop_neat` condition on the first ten v3 replicates of every geometry,
under v4's checkout, and compares champion topology, champion weights, realized
gradient steps and every validation metric with the committed v3 records. **All
50 are identical.** The bridge reads no test split, enters no table and supports
no claim; it exists so that "v3 is frozen" is checked rather than asserted.

```
python -m bpneat.v4.verify --dir results/backprop-neat-v4
```

rebuilds all eight derived tables from `raw/runs/*.json` and byte-compares them,
recomputes every recorded hash, re-checks all three fingerprints, and re-reads
the bridge. CI runs it on every push once the release is sealed.

## Limits of this release

- **Not paired with v3.** Fresh dataset seeds mean no v4 number may be compared
  with a v3 number by a statistical test. Comparisons to v3 here are
  qualitative, as the preregistration requires.
- **Two easy geometries carry no information.** XOR and circles saturate above
  0.97 for every condition, which is most of why H2 and H3 fail their task
  counts. A future design should drop them or make them harder.
- **No penalty-off arm.** The best explanation for §4 is untested here.
- **The analysis code is not fingerprinted.** `analysis.py` decides how
  hypotheses are scored, and it was present in the freeze commit `4fa3009`
  before any run record existed — verifiable with
  `git ls-tree -r --name-only 4fa3009 src/bpneat/v4/`. But that rests on git
  history rather than on a hash, unlike the five science modules. Closing that
  gap is a change for the next protocol.
- **Candidate training is chaotic for a minority of candidates.** See
  [`docs/v4-sensitivity.md`](../../docs/v4-sensitivity.md): at the full budget,
  5% of CGP candidates diverge by more than 1e-6 under a 1e-12 perturbation, and
  the median does not. Every claim here is a distribution over 30 replicates for
  that reason.

## Files

| | |
|---|---|
| `raw/runs/*.json` | 1200 atomic run records, each with all three fingerprints |
| `final-test.json` | the single sealed-test pass |
| `summary.csv` | per task × condition, validation and sealed-test |
| `paired-effects.csv` | Wilcoxon, bootstrap CI, Holm *p*, per-replicate pairs |
| `sign-matrix.csv` | the headline table: which side wins, per algorithm and control |
| `hypotheses.csv` | the preregistered scorecard |
| `cross-algorithm.csv` | between-algorithm gap against the matching effect |
| `budget-table.csv` | what every arm actually spent |
| `operator-usage.csv` | causal operator frequencies |
| `figures/` | 7 figures, each regenerated by the verifier |
| `bridge/replication.json` | the v3 reproducibility gate |
| `sha256sums.txt` | written last, after tables and figures |
