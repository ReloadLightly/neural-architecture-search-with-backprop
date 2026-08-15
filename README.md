# Backprop-NEAT — Summer

Evolving computation graphs that learn by backpropagation, reconstructed from
David Ha's [Backprop-NEAT](https://blog.otoro.net/2016/05/07/backprop-neat/)
(2016) for Chapter 4 of *Japan's Search for a Novel Foreign Policy after 2022*.

**Status: implementation and correctness gates complete. No study has been run
yet — there are no result artifacts, and nothing here is reportable evidence.**

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
would be void, so `settle=True` is the default and both modes are tested.

**Settling must be bounded and weight-independent.** Recurrent cycles carrying
`square`/`mult` diverge to inf within a few ticks, so node values are clamped;
and the tick count is derived from topology by BFS, because a value-dependent
stopping rule makes the traced function discontinuous. Finite-difference
gradient error fell 4.5e14 → 6.6e-2 → 3.7e-7 as each cause was removed.

## Smoke result (not evidence)

One seed, population 50, 6–10 generations — far below the reference budget, and
reported only to show the mechanism is alive:

| Task | Backprop-NEAT val. acc. | Fixed MLP 32×32 | Causal operators evolved |
|---|---|---|---|
| XOR | 0.950 | 0.965 | tanh, sin, mult |
| Circles | 0.945 | 0.965 | square, gaussian |
| Spirals | 0.730 | 0.640 | sin, gaussian, relu |

Circles evolved radial operators (`square`, `gaussian`) unprompted, which is the
qualitative behaviour Ha describes. Champions stayed small (6–7 represented
nodes, 2–3 causally active hidden nodes). These are single-seed numbers at a
fraction of the reference budget; they are calibration, not results.

## Running

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy matplotlib pytest
.venv/bin/python -m pytest tests/ -q      # 17 correctness gates
.venv/bin/python -u bench/final_probe.py  # smoke run across all three tasks
```

## Not yet built

Checkpoint/resume, the one-shot final-test firewall, paired dataset/search
replicates, sharded execution with durable artifacts, and the analysis and
figure pipeline. The sealed test split has never been read; `tests/test_core.py`
enforces that by poisoning it with NaN and asserting search output is unchanged.
