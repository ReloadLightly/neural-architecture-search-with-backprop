# Protocol v5 — NEAT's own machinery as the experiment

**Question.** v3 and v4 asked what the *evaluator* does to a comparison. Neither
touched NEAT's own mechanisms: every condition in both ran the reference settings
of the structural operators, the complexity penalty, the speciation and the
crossover. For an algorithm called *NeuroEvolution of Augmenting Topologies*,
nothing had tested whether the augmenting happens or whether the machinery around
it earns its place.

**Answer, in one line.** It happens, two of NEAT's own defaults suppress it, and
releasing them helps — but not enough to catch a fixed network.

| | |
|---|---|
| Runs | 720 across 90 cells, zero failures |
| Design | 8 conditions × 3 geometries × 30 paired replicates |
| Candidate evaluations | 1,328,400 |
| Gradient updates | 85,498,011 |
| Compute | 7.34 core-hours |
| Sealed test | evaluated once, on dataset seeds `70001 + 13i` — splits no earlier release opened |
| v5 fingerprint | identical across all 720 records |
| Ancestors | v4 `cd9d0c1f…`, v3 `8438c9e8…`, v2 `cfdf1fa3…`, all unchanged throughout |

Preregistration: [`docs/v5-preregistration.md`](../../docs/v5-preregistration.md),
committed before any run record existed (freeze commit
`6f01738bbc6bfb4eef613415eba08917114d4b3f`).

## The preregistered scorecard

**Six of seven hold.** Scored by [`hypotheses.csv`](hypotheses.csv) using rules
written before the suite ran.

| | Hypothesis | Rule | Observed | |
|---|---|---|---|---|
| v5-H1 | the complexity penalty suppresses complexification | grows champions on ≥2 of 3 | 2/3 | **holds** |
| v5-H2 | the structural mutation rate is a binding constraint | grows champions on ≥2 of 3 | 2/3 | **holds** |
| v5-H3 | the two act together | grows champions on ≥2 of 3 | 3/3 | **holds** |
| v5-H4 | bigger topologies buy accuracy | some arm beats the reference in ≥2 of 9 cells | 2/9 | **holds** |
| v5-H5 | speciation is load-bearing | removing it loses on ≥2 of 3 | 2/3 | **holds** |
| v5-H6 | crossover is load-bearing | removing it loses on ≥2 of 3 | 1/3 | **fails** |
| v5-H7 | no NEAT variant beats a budget-matched fixed network | best variant wins 0 of 3 | 0/3 | **holds** |

## What the sealed test says

Causal hidden units and sealed-test accuracy, mean over 30 paired replicates, on
spirals / checkerboard / 3-arm spiral. From [`summary.csv`](summary.csv) and
[`complexity.csv`](complexity.csv).

| condition | causal units | sealed-test accuracy |
|---|---|---|
| `neat_reference` | 4.2 / 2.1 / 1.6 | 0.791 / 0.662 / 0.597 |
| `neat_complexify` | 7.8 / 5.0 / 1.8 | 0.830 / 0.685 / 0.606 |
| `neat_no_penalty` | 4.9 / 4.8 / 4.0 | 0.807 / 0.650 / 0.616 |
| `neat_complexify_no_penalty` | **8.0 / 8.2 / 7.5** | **0.835 / 0.688 / 0.621** |
| `neat_no_speciation` | 3.6 / 0.8 / 0.5 | 0.781 / 0.580 / 0.568 |
| `neat_no_crossover` | 5.0 / 2.2 / 1.3 | 0.760 / 0.640 / 0.593 |
| `neat_deep_narrow` | 10.3 / 4.1 / 1.4 | **0.864** / 0.632 / 0.599 |
| `fixed_mixed_matched` | 65 | **0.964 / 0.712 / 0.765** |

### 1. Complexification is real, and NEAT's own defaults hold it down

Every arm adds nodes at the same opportunity rate the reference does — about 404
node-addition events per run at `p_add_node = 0.2`, and about 993 at 0.5 — but
what *survives* into the champion is set by the fitness. Removing the complexity
penalty `1 + 0.03·√connections` grows champions significantly (v5-H1); raising
the mutation rate grows them (v5-H2); removing both grows them on every geometry
(v5-H3), from 1.6 causal units to 7.5 on the 3-arm spiral.

The penalty comes from Ha's reference fitness and is doing considerably more than
regularising.

### 2. The bigger topologies do buy accuracy

v5-H4 is the hypothesis the protocol existed to test. Had it failed while H1–H3
held, the reading would have been that NEAT's networks are small because the
fitness asks them to be and growing them is pointless. It did not fail: the
unconstrained arm improves on the reference on spirals and checkerboard, and the
width-for-depth trade reaches 0.864 on spirals against the reference's 0.791.

### 3. Speciation earns its place; crossover does not

Removing speciation loses significantly on two of three geometries (v5-H5) and
collapses the checkerboard champion to **0.8** causal units — the smallest
network in the release. Removing historical-marking crossover loses on one
geometry only (v5-H6, **fails**). Of the five mechanisms tested, crossover is the
one this study cannot show is load-bearing.

### 4. The ceiling is not NEAT's to raise

No variant beats the budget-matched fixed network on any geometry (v5-H7). The
best result anywhere in v5 — 0.864 on spirals — closes about a third of the gap
to 0.964. Freeing complexification moves NEAT part of the way across a distance a
better-chosen fixed architecture crosses for nothing.

## Limits of this release

- **Not paired with v3 or v4.** Fresh dataset seeds; comparisons to earlier
  releases are qualitative by construction.
- **`neat_deep_narrow` is compound.** Quartering the population at fixed
  `n_species` also shrinks each species from ~20 members to ~5. The two cannot be
  separated without a third change, so that arm answers "does trading width for
  depth help" and does **not** isolate depth. Declared in the preregistration
  before the run.
- **One penalty setting, one raised rate.** v5 tests penalty on/off and
  `p_add_node` at 0.2/0.5, not a dose-response over either.
- **The analysis code is not fingerprinted.** `analysis.py` decides how these
  hypotheses are scored; it is in the freeze commit before any run record
  existed, which `git ls-tree -r --name-only 6f01738 src/bpneat/v5/` shows, but
  that rests on git history rather than a hash.
- **Three geometries.** Chosen in the contract because v4 showed XOR and circles
  saturate; the trade-off is that each hypothesis is scored out of three.

## Files

| | |
|---|---|
| `raw/runs/*.json` | 720 atomic run records, each with all four fingerprints |
| `final-test.json` | the single sealed-test pass |
| `summary.csv` | per task × condition, validation and sealed-test, size and compute |
| `complexity.csv` | size against the reference, and what it bought |
| `paired-effects.csv` | accuracy *and* size contrasts, Wilcoxon, bootstrap CI, Holm *p* |
| `hypotheses.csv` | the preregistered scorecard |
| `budget-table.csv` | what every arm spent, and the held candidate budget |
| `figures/` | 5 figures, each regenerated by the verifier |
| `sha256sums.txt` | written last, after tables and figures |

```
python -m bpneat.v5.verify --dir results/backprop-neat-v5
```

rebuilds every derived table from `raw/runs/*.json` and byte-compares it,
recomputes every hash, and re-checks all four fingerprints. CI runs it on every
push once the release is sealed.
