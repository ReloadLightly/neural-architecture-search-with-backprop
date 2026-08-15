# Backprop-NEAT — Summer

Evolving computation graphs that learn by backpropagation, reconstructed from
David Ha's [Backprop-NEAT](https://blog.otoro.net/2016/05/07/backprop-neat/)
(2016) for Chapter 4 of *Japan's Search for a Novel Foreign Policy after 2022*.

**Status: implementation, infrastructure and correctness gates complete; the
full core matrix runs end-to-end at reference budget. The confirmatory study has
not been run — nothing here is reportable evidence yet.**

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

## Two findings from the gates

**Ha's propagation break rule can silently zero a working network.** The output
node holds id 3 and is recomputed before every hidden node on each tick, while
the loop stops as soon as every node has been touched. Any graph wired directly
from bias or the inputs into *both* the output and the hidden units therefore
returns an output computed before a single hidden node has run. A fixed 32×32
MLP outputs identically zero under that rule — 0.500 accuracy, a dead network.
Settled, the same weights under the same RMSProp settings reach 0.745 train /
0.715 validation on spirals. Any comparison against a baseline crippled this way
would be void — so every baseline routes its output bias through a carrier node
and is verified alive under Ha's exact rule as well as under settling.

**Settling must be bounded and weight-independent.** Recurrent cycles carrying
`square`/`mult` diverge to inf within a few ticks, so node values are clamped;
and the tick count is derived from topology by BFS, because a value-dependent
stopping rule makes the traced function discontinuous. Finite-difference
gradient error fell 4.5e14 → 6.6e-2 → 3.7e-7 as each cause was removed.

## Propagation is a declared factor, not a default

`ha2016` is Ha's exact break rule; `settled` ticks the same propagation to a
topology-determined fixed point. Head-to-head at population 50, identical seeds:

| Task | `ha2016` | `settled` | Causal hidden nodes under `ha2016` |
|---|---|---|---|
| XOR | 0.650 | 0.980 | **0** |
| Circles | 0.870 | 0.990 | 4 |
| Spirals | 0.590 | 0.715 | **0** |

The zero-causal-hidden champions are logistic regressions: evolution never
escaped its seed topology, because hidden structure added on top of a direct
input→output edge is invisible to fitness. This reproduces the "near chance
across conditions" symptom of the earlier attempt, and it belongs in Chapter 4
as a finding about the source algorithm — so the two modes are separate tracks
and are never pooled. The fixed MLP scores identically under both modes, which
is what makes the comparison legitimate.

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
.venv/bin/python -m pytest tests/ -q                    # 33 gates

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

Protocol freeze (the version is `v1-draft`, which the firewall refuses),
the paired-effects and Pareto analysis, and the figure pipeline.
