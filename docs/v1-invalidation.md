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

It is not. It was an artifact of the over-strong selection. With reference
selection restored, the same propagation rule reaches the published regime:

| Task | v1 `ha2016` | v2 `ha2016` | Ha (Fig. 10.3, test) |
|---|---|---|---|
| XOR | 0.650 | **1.000** | 94.3% |
| Circles | 0.870 | **0.960** | 96.3% |
| Spirals | 0.590 | **0.795** | 81.5% |

Champion sizes also move toward the reference: XOR 7 nodes / 10 connections
against Ha's 8 / 12, circles 8 / 13 against 11 / 20. Spirals remain smaller
(9 / 18 against 34 / 96) while reaching comparable accuracy, so the size
question in `docs/reference-targets.md` stays open.

On spirals, `ha2016` now *outperforms* settled propagation (0.795 vs 0.730),
which is the opposite of what v1 concluded.

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
