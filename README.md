# Backprop-NEAT — Summer

Evolving computation graphs that learn by backpropagation, reconstructed from
David Ha's [Backprop-NEAT](https://blog.otoro.net/2016/05/07/backprop-neat/)
(2016) for Chapter 4 of *Japan's Search for a Novel Foreign Policy after 2022*.

**Status: protocol v2 frozen; the confirmatory suite is running. Protocol v1 is
invalidated for a selection-fidelity defect and its records are retained only
for audit — see [`docs/v1-invalidation.md`](docs/v1-invalidation.md). No number
here is reportable evidence until the v2 release completes.**

## What is implemented

A NumPy reconstruction built against the published reference
([`hardmaru/backprop-neat-js`](https://github.com/hardmaru/backprop-neat-js)),
not from memory:

- `ha2016` task geometries — XOR, circles, and the 1.75-turn radius-6 spiral,
  raw and unstandardised, noise 0.5 — with independent train/validation/sealed
  test partitions;
- arbitrary directed computation graphs over the reference operator set
  (sigmoid, tanh, relu, gaussian, sin, abs, mult, square, add), including
  recurrent connections;
- backpropagation through the *executed* trace, RMSProp (lr 0.01, decay 0.999,
  regc 1e-3, clip 5.0), minibatch 10, 600 inner steps, rollback every 20
  against the full training split;
- fitness `-error × (1 + 0.03·√connections)`, as in `datafit-neat.js`;
- NEAT structural mutation, union crossover under an innovation registry,
  K-medoids subpopulations, elitism and a hall of fame;
- inherited learned weights (Lamarckian) with a Baldwinian switch;
- fixed-MLP, logistic, and random-architecture controls built as ordinary
  genomes, so every condition shares one learner;
- causal subgraph extraction — the nodes, connections and operators that
  actually reached the returned output.

## What the gates caught

**A naive baseline is silently zeroed by Ha's propagation rule.** The rule stops
propagation once every node has been touched. The output node holds id 3 and is
recomputed before every hidden node on each tick, so it reads its operands from
the previous tick. Wire bias straight into *both* the output and the hidden
units — the obvious way to build an MLP — and every node is touched on tick 0,
the loop stops, and the network returns a constant. A fixed 32×32 MLP built that
way outputs identically zero: 0.500 accuracy, a dead control. Every baseline here
therefore routes its output bias through a carrier node, and is verified alive
under Ha's exact rule as well as under settling. A comparison against a control
crippled this way would be void.

**Settling must be bounded and weight-independent.** Recurrent cycles carrying
`square`/`mult` diverge to inf within a few ticks, so node values are clamped;
and the tick count is derived from topology by BFS, because a value-dependent
stopping rule makes the traced function discontinuous. Finite-difference
gradient error fell 4.5e14 → 6.6e-2 → 3.7e-7 as each cause was removed.

**Selection fidelity decides whether topologies grow at all.** The reference
replaces the whole population with offspring each generation and picks parents by
fitness-proportionate roulette over the entire subpopulation, weight
`1/(-fitness + 0.01)` — an error-0.3 genome breeds only ~2.3× as often as an
error-0.7 one. That weakness is functional: it lets neutral or mildly harmful
structural additions survive long enough to combine. An earlier version of this
code used elitism plus truncation to the better half, which pinned topologies at
the minimal seed and produced a false finding — that Ha's propagation rule traps
evolution in logistic regression. It does not. See
[`docs/v1-invalidation.md`](docs/v1-invalidation.md).

## Fidelity against the published reference

With reference selection restored, track A (`ha2016`, Ha's exact rule) reaches
the regime published in *Neuroevolution* §10.1, Figure 10.3:

| Task | This reconstruction (validation) | Ha, Fig. 10.3 (test) | Nodes here / Ha | Conns here / Ha |
|---|---|---|---|---|
| XOR | 1.000 | 94.3% | 7 / 8 | 10 / 12 |
| Circles | 0.960 | 96.3% | 8 / 11 | 13 / 20 |
| Spirals | 0.795 | 81.5% | 9 / 34 | 18 / 96 |

Single seed, reference budget. Accuracy and champion size both land close on
XOR and circles. Spirals reach comparable accuracy at far smaller size, which
remains an open question — see
[`docs/reference-targets.md`](docs/reference-targets.md).

## Calibration, one replicate (not evidence)

The full core matrix at reference budget — population 100, five species, 10
generations (20 for spirals), 600 inner steps — on replicate 1 of track B.
Single seed, so this is calibration and a cost measurement, not a result:

| Task | Backprop-NEAT | Homog. tanh | Evolution only | Random arch. | Fixed MLP | Logistic |
|---|---|---|---|---|---|---|
| XOR | 0.965 | 1.000 | 0.970 | 0.990 | 0.990 | 0.470 |
| Circles | 0.990 | 0.955 | 0.880 | 0.995 | 0.990 | 0.595 |
| Spirals | **0.775** | 0.755 | 0.705 | 0.730 | 0.685 | 0.630 |

Validation accuracy of the validation-selected champion. Backprop-NEAT leads
every control on spirals — the deceptive geometry — and the separable tasks are
at ceiling for everything except the linear floor, which is exactly where a
linear model should sit. One seed proves none of this.

## Measured compute forecast

One replicate of the six core conditions across all three tasks: **18 runs,
6.6 minutes** wall clock on 4 cores, measured. So the frozen ten-replicate core
matrix is **180 runs, ≈66 minutes serial**, or roughly 20 minutes across four
shards. Track A costs the same again. Cost is dominated by the fixed-MLP
multistart (60 restarts × 600 steps ≈ 95 s/run), not by evolution.

## Running

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy matplotlib pytest
.venv/bin/python -m pytest tests/ -q                    # 39 gates

export PYTHONPATH=src
python -m bpneat.cli conditions                         # the comparison matrix
python -m bpneat.cli plan --track B                     # 180 runs
python -m bpneat.cli suite --track B --out results/run  # shardable, resumable
python -m bpneat.cli final-test --dir results/run       # refuses a draft protocol
```

Shard a long run with `--shard-index i --shard-total n`; shards never overwrite
each other, completed runs are skipped by code fingerprint, and `manifest.json`
is refreshed after every run so progress is read rather than guessed.

## Test isolation

`bpneat.finaltest` is the only module that may read `DatasetBundle.test`. It
refuses to run against an incomplete, draft-protocol, already-evaluated, or
code-modified release, loads champions exactly as validation selected them, and
never retrains or reselects. The gates poison the test split with NaN and assert
that search output is unchanged.

## Not yet built

The v2 release README with its final claim ladder, and the decision-boundary and
topology figures for individual champions.
