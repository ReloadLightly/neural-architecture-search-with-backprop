# Protocol v3 — "When the evaluator decides"

1,920 runs, 150 cells, 30 paired replicates, five geometries, zero failures.
Preregistered and frozen at commit `a343c66`
([`../../docs/v3-preregistration.md`](../../docs/v3-preregistration.md))
before any confirmatory compute. Sealed test opened exactly once.

- **2,143,500** candidate evaluations, **173,623,974** realized gradient steps,
  **23.78 core-hours** (forecast: 24).
- v3 fingerprint `8438c9e89c7c72a3…`, identical across all 1,920 records.
- Frozen v2 fingerprint `cfdf1fa3198adc0e…` unchanged throughout.

## The question

Can implementation details absent from an algorithm's published description
decide a study's conclusions? Four are measured as factors: the inner learner's
stopping rule, forward-pass scheduling, the selection operator, and weight
inheritance.

## Conclusion stability matrix

Which v2 headline claim survives which evaluator. Verdicts use Wilcoxon
signed-rank with **Holm correction** over each source's pre-declared family
(v3: 35 comparisons; v2: its analogous 15).

| | Claim | v2 as run | v3 re-run | budget-matched tanh | sin control | mixed control | candidate-matched |
|---|---|---|---|---|---|---|---|
| C1 | Backprop-NEAT beats a fixed MLP on spirals | supported | — | **reversed** | **reversed** | **reversed** | — |
| C2 | Gradient learning complements topology search | not significant | supported | — | — | — | — |
| C3 | Operator diversity contributes | not significant | supported | — | — | — | — |
| C4 | Few causal nodes beat a 65-unit MLP | supported | — | **reversed** | **reversed** | **reversed** | — |
| C5 | Evolutionary selection beats random sampling | not significant | — | — | — | — | not significant |

Two readings, both uncomfortable. The two claims v2 stated most confidently
(C1, C4) **reverse** under any fairly trained control. The other three were
never supported by v2's own data at the correction level it should have used;
v3's larger sample recovers two of them (C2, C3) and leaves C5 unsupported in
both protocols.

![stability matrix](figures/stability-matrix.png)

## Block A — controls and budgets

Sealed-test accuracy, mean over 30 paired replicates.

| Condition | XOR | Circles | Spirals | Checkerboard | 3-arm spiral |
|---|---|---|---|---|---|
| Backprop-NEAT | 0.995 | 0.986 | 0.791 | 0.662 | 0.614 |
| Fixed MLP tanh, *v2's Ha learner* | 0.985 | 0.977 | **0.635** | 0.590 | 0.536 |
| Fixed MLP tanh, budget-matched | 0.990 | 0.982 | **0.896** | 0.700 | 0.589 |
| Fixed MLP sin, budget-matched | 0.972 | 0.982 | **0.962** | 0.693 | 0.746 |
| Fixed MLP mixed, budget-matched | 0.982 | 0.984 | **0.966** | **0.728** | **0.773** |
| Random architecture, candidate-matched | 0.996 | 0.988 | 0.768 | 0.698 | 0.604 |
| Homogeneous tanh | 0.996 | 0.931 | 0.715 | 0.541 | 0.542 |
| Evolution only | 0.792 | 0.803 | 0.623 | 0.546 | 0.536 |

Realized gradient steps on spirals make the mechanism plain: Backprop-NEAT
spends **134,449**, each budget-matched control **134,428** — and v2's control
**2,633**.

The same 32×32 tanh network scores **0.635** under v2's evaluator and **0.896**
under a matched budget. The architecture is identical; only the stopping rule
changes. That 0.261 swing is larger than any difference v2 reported between
conditions.

Paired differences on spirals (Backprop-NEAT − control), Holm-corrected:

| Control | Median diff | 95% CI | Wins | Holm *p* |
|---|---|---|---|---|
| Fixed MLP mixed, matched | **−0.182** | [−0.197, −0.152] | 0/30 | 6.0e-05 |
| Fixed MLP sin, matched | **−0.175** | [−0.195, −0.145] | 0/30 | 6.0e-05 |
| Fixed MLP tanh, matched | **−0.110** | [−0.120, −0.090] | 1/30 | 6.3e-05 |
| Random arch., candidate-matched | +0.025 | [+0.003, +0.045] | 20/30 | 0.385 |
| Homogeneous tanh | +0.085 | [+0.043, +0.117] | 23/30 | 0.018 |
| Fixed MLP tanh, *Ha learner* | +0.158 | [+0.125, +0.190] | 30/30 | 6.0e-05 |
| Evolution only | +0.170 | [+0.140, +0.188] | 30/30 | 6.0e-05 |

![accuracy against realized compute](figures/budget-vs-accuracy.png)
![Block A paired effects](figures/block-a-paired-effects.png)

## Block B — propagation × fitness split

v2 varied both at once. Separated, sealed-test accuracy / collapse rate:

| Propagation | Fitness | XOR | Circles | Spirals |
|---|---|---|---|---|
| `ha2016` | train | 0.811 / **0.17** | 0.940 / 0.03 | 0.720 / **0.20** |
| `ha2016` | validation | 0.951 / 0.00 | 0.959 / 0.00 | 0.726 / 0.03 |
| `settled` | train | 0.993 / 0.00 | 0.984 / 0.00 | 0.796 / 0.00 |
| `settled` | validation | 0.995 / 0.00 | 0.986 / 0.00 | 0.791 / 0.00 |

Collapse — a champion with zero causally active hidden nodes — needs **both**
`ha2016` propagation *and* training-loss fitness. Either alone produces almost
none. **H3 is wrong**: it predicted collapse would depend on propagation and
not on the fitness split; it is an interaction.

![propagation grid](figures/propagation-grid.png)

## Block C — selection-pressure dose response

Against realized standardised selection intensity, spirals:

| Selector | Intensity | Collapse | Causal nodes | Test acc |
|---|---|---|---|---|
| `roulette_s1.0` | 0.021 | 0.00 | 4.5 | 0.797 |
| `roulette_s0.1` | 0.046 | 0.00 | 4.4 | 0.812 |
| `roulette_s0.001` | 0.053 | 0.00 | 4.7 | 0.803 |
| `roulette_s0.01` (**Ha's own**) | 0.054 | 0.00 | 4.3 | 0.791 |
| `tournament_k2` | 0.495 | 0.00 | 4.0 | 0.796 |
| `v1_truncation` (**v1's operator**) | 0.690 | 0.00 | 4.3 | 0.809 |
| `tournament_k4` | 0.802 | 0.00 | 4.8 | 0.833 |

**H4 is wrong, and informatively so.** Collapse is zero at *every* selection
level, on both tasks, and causal size is flat (4.0–4.8). Selection pressure
does not drive collapse. Worse for the v1 story: v1's own truncation operator
collapses nothing and scores **above** Ha's setting (0.809 vs 0.791).

Taken with Block B, this revises the v1 invalidation itself. v1 was
invalidated for being unfaithful to the reference selection operator, and the
collapse it reported was attributed to that infidelity. v3 says the attribution
was wrong: collapse is produced by the propagation × fitness interaction, which
v1 also had. v1's selection operator was unfaithful — and not the cause.

![selection dose response](figures/selection-dose-response.png)

## Block D — Lamarckian vs Baldwinian

| Task | Lamarckian | Baldwinian |
|---|---|---|
| XOR | 0.995 | 0.726 |
| Circles | 0.986 | 0.567 |
| Spirals | 0.791 | 0.532 |

**H5 answered decisively** in the Lamarckian direction, and not for lack of
effort: the Baldwinian condition spent *more* gradient steps on spirals
(159,083 against 134,449) and still landed near chance. When evolution rewires
a graph every generation, inherited weights are what makes a topology's
learning transferable to its descendants.

## Hypotheses

| | Prediction | Outcome |
|---|---|---|
| H1 | matched tanh MLP's spirals deficit disappears or reverses | **reverses** (−0.110, 1/30, *p*=6.3e-05) |
| H2 | sin MLP matches or beats on spirals | **holds** (−0.175, 0/30, *p*=6.0e-05) |
| H3 | collapse depends on propagation, not fitness split | **fails** — interaction of both |
| H4 | collapse/causal size vary with selection intensity | **fails** — flat; v1's operator collapses nothing |
| H5 | Lamarckian vs Baldwinian, two-sided | **Lamarckian**, on all three tasks |

Three of five predictions failed. They are reported as failed.

## What this release supports

1. An undocumented stopping rule moved a fixed baseline by 0.261 on spirals and
   reversed the study's headline comparison.
2. Operator prior, not search, accounts for the spirals result: a fixed sin or
   mixed-operator network at the same budget beats every evolved champion
   (0.962/0.966 against 0.791), losing 0 of 30 replicates.
3. Gradient learning and operator diversity are real contributions of
   Backprop-NEAT, now supported at 30 replicates where v2's 10 could not.
4. Evolutionary selection does not beat candidate-matched random sampling on
   any task here.
5. Collapse to the linear floor is an interaction of propagation and fitness
   split, not a consequence of selection pressure.

## What it does not

- That architecture search is useless. It is not better *than a fairly trained
  fixed network on these five 2-D geometries*, which is a narrow claim.
- That sin activations are a general answer. They are an operator prior that
  happens to suit spirals.
- Anything about tasks with more than two input dimensions; the genome encoding
  fixes two, in a frozen module.
- Anything about geopolitics or forecasting.

## Reproduce

```bash
make v3-release                                 # rebuild tables and figures, reseal
python -m bpneat.v3.verify --dir results/backprop-neat-v3
```

`verify` rebuilds every derived table from `raw/runs/*.json` in a scratch
directory and byte-compares, rechecks every hash, and confirms both
fingerprints. CI runs it on every push.
