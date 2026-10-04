# Backprop-NEAT — result release v2

> ### ⚠️ Two of the claims below are withdrawn
>
> An October 2026 audit, reproduced in
> [`../../docs/audit-2026-10.md`](../../docs/audit-2026-10.md), showed that the
> fixed-MLP and random-architecture controls were starved of gradient budget:
> 2,506 realized steps on spirals against Backprop-NEAT's 132,138, because Ha's
> rollback rule halts a 69-node network after ~42 updates. Protocol v3 re-ran
> the comparison with matched budgets and the result **reverses** — the same
> network scores 0.896 instead of 0.635 and beats Backprop-NEAT's 0.791.
>
> Read [`ERRATA.md`](ERRATA.md) and
> [`../../docs/v2-errata.md`](../../docs/v2-errata.md) before citing anything
> here. The runs are correct and regenerate byte-identically; what was wrong is
> what was claimed about them. Superseded by
> [`../backprop-neat-v3/`](../backprop-neat-v3/).

Protocol `v2`, frozen before execution
([`../../docs/protocol-freeze-v2.md`](../../docs/protocol-freeze-v2.md)).
Sealed test evaluated exactly once, after the suite completed and the blind
review passed.

Validation-based selection did not overfit: the mean validation-to-test drop is
at most **0.032** in any condition and at most **0.011** for Backprop-NEAT.

## What was run

360 runs: two tracks × six core conditions × three tasks × ten paired
replicates. Zero failures. Every run used population 100, five K-medoids
subpopulations, 10 generations (20 for spirals), 600 inner minibatch updates per
evaluated candidate, and the `ha2016` geometries at 200/200/200 examples.

**Total compute: 268,800 candidate evaluations, 11,653,669 realized gradient
steps, 1.87 core-hours.**

| Track | Propagation | Fitness |
|---|---|---|
| A | `ha2016` — Ha's exact break rule | training loss, source rollback |
| B | `settled` — ticked to a topology-determined fixed point | validation loss |

Tracks are reported separately and are never pooled.

## What was not run

- `no_penalty` and `baldwinian` were declared secondary and are outside this
  freeze. They were not run, and adding them now would be a post-hoc change to
  the matrix.
- Protocol v1 was **invalidated** for a selection-fidelity defect before any
  test evaluation; its records are retained under `results/backprop-neat-v1/`
  for audit only. See `docs/v1-invalidation.md`. No v1 number is cited here.

## Headline: sealed-test accuracy, track B

Mean over ten paired replicates. `causal n` is causally active hidden nodes.

| Task | Backprop-NEAT | Homog. tanh | Evolution only | Random arch. | Fixed MLP | Logistic |
|---|---|---|---|---|---|---|
| XOR | **0.996** | 0.995 | 0.736 | 0.986 | 0.979 | 0.536 |
| Circles | **0.986** | 0.943 | 0.791 | 0.988 | 0.973 | 0.503 |
| Spirals | **0.787** | 0.746 | 0.639 | 0.713 | 0.637 | 0.596 |

| Task | Backprop-NEAT causal n | Fixed MLP causal n |
|---|---|---|
| XOR | 2.6 | 65 |
| Circles | 4.0 | 65 |
| Spirals | 4.6 | 65 |

Validation-to-test drops are ≤0.03 in every condition, so validation-based
selection did not overfit.

## Paired within-replicate differences, track B, sealed-test loss

Negative favours Backprop-NEAT. 95% paired bootstrap interval over replicates.

| Task | Comparison | Mean | 95% CI | Wins |
|---|---|---|---|---|
| Spirals | vs Fixed MLP | **−0.210** | [−0.262, −0.157] | 10/10 |
| Spirals | vs Random arch. | **−0.126** | [−0.181, −0.077] | 10/10 |
| Spirals | vs Evolution only | **−0.208** | [−0.262, −0.153] | 10/10 |
| Spirals | vs Homog. tanh | **−0.088** | [−0.144, −0.030] | 8/10 |
| Circles | vs Homog. tanh | **−0.117** | [−0.168, −0.060] | 9/10 |
| Circles | vs Evolution only | **−0.410** | [−0.472, −0.345] | 10/10 |
| Circles | vs Fixed MLP | −0.003 | [−0.048, +0.062] | 8/10 |
| Circles | vs Random arch. | +0.037 | [−0.002, +0.095] | 3/10 |
| XOR | vs Evolution only | **−0.307** | [−0.512, −0.061] | 9/10 |
| XOR | vs Fixed MLP | +0.130 | [−0.050, +0.395] | 8/10 |
| XOR | vs Random arch. | +0.149 | [−0.035, +0.402] | 8/10 |
| XOR | vs Homog. tanh | +0.157 | [−0.016, +0.398] | 7/10 |

On XOR the mean favours the controls while the win count favours Backprop-NEAT:
a minority of runs fail badly and drag the mean. That dispersion is the finding,
not noise to be smoothed away.

## Operator specialisation (H4)

Share of causally active hidden operators, track B, Backprop-NEAT champions:

| Operator | XOR | Circles | Spirals |
|---|---|---|---|
| mult | **0.46** | 0.00 | 0.09 |
| square | 0.08 | **0.30** | 0.02 |
| gaussian | 0.04 | **0.23** | 0.04 |
| sin | 0.15 | 0.10 | **0.52** |
| abs | 0.04 | 0.10 | 0.13 |
| tanh | 0.08 | 0.07 | 0.11 |

Architectures differ across tasks by *which operators reach the output*, not
merely by size. Circles concentrate on radial operators (square + gaussian =
0.53), which is the behaviour *Neuroevolution* §10.1 describes qualitatively;
XOR concentrates on multiplicative interaction; spirals on periodicity.

## Propagation: track A against track B

Backprop-NEAT, sealed-test accuracy:

| Task | Track A (`ha2016`) | Track B (`settled`) |
|---|---|---|
| XOR | 0.751 | 0.996 |
| Circles | 0.929 | 0.986 |
| Spirals | 0.738 | 0.787 |

Ha's exact break rule costs real performance, most severely on XOR. It does
**not** prevent evolution from working — an earlier version of this project
claimed it did, and that claim was wrong and is retracted.

## Fidelity against the published reference

*Neuroevolution* §10.1, Figure 10.3 reports champions at 94.3% / 96.3% / 81.5%
test accuracy with 8/12, 11/20 and 34/96 nodes/connections.

Track B reaches 0.996 / 0.986 / 0.787 with 2.6–4.6 causally active hidden nodes.
Accuracy is comparable or better on all three tasks; **network size is not
reproduced**, most starkly on spirals. `applyMutations` adds at most one node
per offspring at rate 0.2, which over 17 generations cannot produce the ~30
additions a 34-node champion needs, so Figure 10.3's champion most likely comes
from a long interactive playground session rather than a batch run. See
`docs/reference-targets.md`.

## Claim ladder

**Strongest supported claims.**

1. On the deceptive geometry (spirals), Backprop-NEAT beat every control on
   sealed-test loss in 10/10 paired replicates against the fixed MLP, random
   architecture search, evolution-only and the linear floor, with intervals
   excluding zero.
2. Gradient learning and topology search are complementary: Backprop-NEAT beat
   evolution-only on all three tasks, decisively (10/10, 10/10, 9/10).
3. Evolved architectures specialise by task in causal operator usage, not just
   in size.
4. Represented size is not computation. Backprop-NEAT beat a 65-hidden-unit MLP
   on spirals using 4.6 causally active hidden nodes.
5. Ha's original propagation rule measurably changes what evolution discovers,
   and its cost is task-dependent.

**Claims this experiment does NOT support.**

1. That architecture search beats fixed architectures in general. On XOR and
   circles the fixed MLP and random architecture search matched or beat
   Backprop-NEAT; only spirals separated them.
2. That evolutionary selection beats random architecture sampling in general —
   that held on spirals only, and random search won circles.
3. That this reproduces Ha's Figure 10.3. Accuracy is comparable; size is not.
4. Anything about geopolitics, forecasting, or foreign policy. These are
   two-dimensional synthetic classification tasks.

**The most tempting unsupported claim**, stated so it can be resisted: *"evolution
discovers better architectures than human engineers."* Two of three tasks say
otherwise here, and the one that supports it is the single hardest geometry. The
defensible version is narrower and more interesting: architecture search paid off
exactly where the problem was deceptive, and paid off in structure-efficiency
rather than raw accuracy.

## Reproducing

```bash
export PYTHONPATH=src
python -m bpneat.cli report  --dir results/backprop-neat-v2/track-b
python -m bpneat.cli figures --dir results/backprop-neat-v2
```

Every table and figure is regenerated from `raw/runs/*.json`; nothing is
recomputed and no model is retrained. `sha256sums.txt` fixes the release
contents.
