# v1 is invalidated

Protocol v1 completed 180 track-A runs and 53 track-B runs before a fidelity
defect was found in the search itself. The defect changes scientific output, so
under the freeze rule v1 is invalidated and replaced by a new protocol identity
rather than patched. Its raw records are retained under
`results/backprop-neat-v1/` so this can be audited; **no number in that
directory may be cited.**

## The defect

`_reproduce` used elitism plus truncation selection — one elite kept per
subpopulation, and parents drawn only from the better half. The reference does
neither. `NEATTrainer.evolve` in `ml/neat.js` replaces the entire population
with offspring every generation, and `pickRandomIndex` selects parents by
fitness-proportionate roulette over the *whole* subpopulation:

```js
g.normFitness = 1/(-g.fitness + slack);   // slack = 0.01
```

Fitness is `-error × penalty`, so the weight rises as error falls, but only
gently: at slack 0.01 an error-0.3 genome is roughly 2.3× as likely to breed as
an error-0.7 one. That weakness is functional. It lets structural additions that
are neutral or mildly harmful survive long enough to combine into something
useful. Truncation selection plus an elite removes exactly that drift, so
topologies stayed near the minimal seed.

## What it caused

The suppressed drift produced a false conclusion. v1 reported that under Ha's
exact propagation rule evolution collapses to logistic regression — champions
with **zero** causally active hidden nodes on XOR and spirals — and this was
written up as a property of the source algorithm.

It is not — or at least, it is not the whole story, and the version of this
section written in August overcorrected in the other direction. See
**Erratum E4** below.

### What the single calibration runs showed

> **These are single runs on one calibration seed, not results.** They were
> recorded before the v2 confirmatory suite existed, and they are kept here
> only to show what motivated the fix. Do not cite them.

| Task | v1 `ha2016` | v2 `ha2016` *(1 calibration run)* | Ha (Fig. 10.3, test) |
|---|---|---|---|
| XOR | 0.650 | 1.000 | 94.3% |
| Circles | 0.870 | 0.960 | 96.3% |
| Spirals | 0.590 | 0.795 | 81.5% |

### What the confirmatory suite actually shows

Ten paired replicates, sealed test, from
`results/backprop-neat-v2/track-a/final-test.json`:

| Task | track A `ha2016` mean | median | collapse rate |
|---|---|---|---|
| XOR | **0.751** | 0.782 | 3/10 |
| Circles | **0.929** | 0.955 | 0/10 |
| Spirals | **0.738** | 0.730 | 1/10 |

The calibration run's XOR 1.000 was a lucky draw: the confirmatory mean is
0.751, and it is **bimodal** — three of ten champions collapse to zero causally
active hidden nodes at the logistic floor (0.535–0.560) while the rest reach
0.630–1.000. So the v1 phenomenon is real at a *rate*, not absent. v1's error
was treating it as universal; this document's error was then calling it simply
"wrong and retracted".

The sentence this section used to carry — "`ha2016` now outperforms settled
propagation on spirals (0.795 vs 0.730)" — **is withdrawn**. In the
confirmatory data the comparison runs the other way: track A spirals 0.738
against track B 0.787. It was also never a clean comparison, because the two
tracks differ in the fitness split as well as the propagation rule
(**Erratum E3**). Protocol v3 separates those factors.

Champion sizes do move toward the reference, and that part stands: XOR 7 nodes
/ 10 connections against Ha's 8 / 12, circles 8 / 13 against 11 / 20. Spirals
remain far smaller (9 / 18 against 34 / 96), so the size question in
[`reference-targets.md`](reference-targets.md) stays open.

## A second, lesser defect

`code_fingerprint` hashed every module in the package, including `analysis.py`,
`figures.py` and `cli.py` — code that cannot change what a run computes. Adding
the plotting pipeline mid-suite moved the fingerprint four times across the v1
track-A runs, which would have made the final-test firewall refuse a release
whose science code had in fact never changed. `git diff` between the freeze
commit and the end of the run confirms all seven science modules were
byte-identical throughout.

The fingerprint now covers only `baselines, conditions, datasets, evolve,
genome, learn, protocol`.

## What v1 still demonstrates

The infrastructure worked. A container restart killed all four shards mid-run
and cost nothing: every completed run was on disk, atomically written and
fingerprinted, and the manifest reported exactly what was missing. That is the
failure mode the whole project was built to prevent, and it was survived
unplanned.

## Consequence for claims

Nothing about propagation, architecture size, or condition ranking may be
carried over from v1. Every claim must come from the v2 release.
