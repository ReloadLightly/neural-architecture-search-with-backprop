---
title: "When the Evaluator Decides: Implementation Details Determine the Conclusions of an Evolution-with-Learning Study"
author: Roland Löchli
venue: GECCO 2027 (draft)
---

# Abstract

We reconstruct David Ha's Backprop-NEAT (2016) — NEAT topology search in which
backpropagation trains every evaluated candidate — and run it five times under
increasingly careful protocols. The reconstruction is faithful: it is audited
against the published source, frozen behind a module fingerprint, and every
published number regenerates byte-identically from committed raw records.

Its conclusions still changed twice, and both times the cause was an
implementation detail absent from the algorithm's published description.

In protocol v1, a selection operator we chose ourselves — elitism plus
truncation, where the reference uses near-neutral fitness-proportionate
roulette — suppressed structural drift and produced the finding that Ha's
asynchronous propagation rule traps evolution at the linear floor. In protocol
v2, with the reference operator restored, that finding vanished; but v2's
headline result (architecture search beats a fixed network on the deceptive
geometry) rests on a control that the *learner's stopping rule* halted after 42
gradient steps against the evolutionary condition's 132,138. The same fixed
network, given a matched budget, is competitive; given a sinusoidal activation
and 600 plain steps, it beats every evolved champion in the release.

Protocol v3 therefore treats the implementation details as the experiment.
We measure four factors — the learner's stopping rule, forward-pass scheduling,
the selection operator, and weight inheritance — across 1,920 preregistered
runs, 30 paired replicates, and five 2-D geometries, and report a *conclusion
stability matrix*: which of the five v2 headline claims survives which
evaluator setting. Three of five preregistered hypotheses failed, and two of
v2's five claims reverse under every fairly trained control.

Protocol v4 then asks whether any of this is specific to Backprop-NEAT, by
putting a second, independently derived topology search — Cartesian Genetic
Programming with gradient-trained candidates, sharing no encoding, no selection
mechanism and no code with the first — through the same controls, across 1,200
further runs on dataset splits no earlier release had opened. **Four of its six
preregistered hypotheses failed, including the one we most expected to hold.**

The artifact is universal and the reversal is not. Restoring the budget the
control was denied raises it on all five geometries (0.640 → 0.898 on spirals,
0 of 30 successes to 30 of 30), for both algorithms. But *search losing to a
fairly trained control* proves geometry-dependent rather than general, and the
hypotheses asserting otherwise failed at thresholds set in advance. What does
replicate across both algorithms is that the search component never beats its
own candidate-matched null — and on the two hardest geometries CGP is
significantly beaten by it, while a fixed network with heterogeneous operators
beats every evolved champion in the release.

Protocol v5 then turns the algorithm's own machinery into the experiment — the
complexification rate, the complexity penalty that opposes it, speciation,
crossover, and width against depth at a fixed budget — across 720 further runs.
**Six of its seven hypotheses hold, and they are the most favourable results in
the project.** NEAT's complexification is real and is suppressed by two of its
own reference settings; removing both grows champions on every geometry, from 1.6
causally active units to 7.5 on the hardest, and that growth does buy accuracy.
Speciation earns its place; historical-marking crossover cannot be shown to. But
no variant beats the budget-matched fixed network on any geometry: freeing
complexification closes about a third of a gap a better-chosen fixed architecture
crosses for nothing.

We argue that for evolution-with-learning algorithms, where an inner learner
and an outer search interact, the evaluator is not a neutral measuring device
but a component of the method, and that reporting a single budget point is
closer to reporting a hyperparameter than a result. A second finding limits even
that: two implementations of one model, agreeing on values and gradients to
1e-15, train 5% of candidates to visibly different networks, so a
neural-architecture-search result reported from single runs is not reproducible
in principle.

**Reproduction:** one command rebuilds every table and figure from raw records;
CI asserts byte-identity on every push.

# 1 Introduction

Neuroevolution papers report that a search procedure found a good architecture.
The claim has a hidden conjunct: *under the evaluation we ran*. For algorithms
that pair an outer evolutionary loop with an inner gradient learner, that
conjunct carries unusual weight, because the inner learner is applied to every
candidate — including the baselines — and a learner tuned for one kind of graph
can silently cripple another.

This paper is a case study in how far that goes. It is not a critique of
Backprop-NEAT; the algorithm is a good one and our reconstruction reaches the
accuracy regime of the published demonstration. It is a report on three
attempts to measure it honestly, two of which produced confident, wrong
conclusions, and on what the third attempt had to do differently.

**Contributions.**

1. A faithful, fingerprint-frozen NumPy reconstruction of Backprop-NEAT with a
   sealed-test firewall, exact resume, and byte-identical regeneration of every
   published artefact (§3).
2. Two documented failures of our own study, each traced to a specific
   implementation detail, each with the invalidated results retained for audit
   (§4).
3. A preregistered factorial study treating those details as factors, with a
   dense evaluator that makes budget-matched controls affordable while provably
   computing the same function as the reference path (§5).
4. A conclusion stability matrix: an explicit statement of which claims are
   properties of the algorithm and which are properties of the evaluator (§6).
5. A second, independently derived search algorithm run through the same
   controls on fresh test splits, separating the part of the result that
   generalises — the starved control, and a search component that never beats
   its own null — from the part that does not, which includes the reversal our
   own previous protocol led with (§7).
6. A measurement showing that numerical equivalence does not imply run-level
   reproducibility: two evaluators agreeing to 1e-15 on values and gradients
   train a minority of candidates to visibly different networks (§7.5).
7. A factorial study of NEAT's own mechanisms, separating what the algorithm
   cannot do from what its default hyperparameters prevent it from doing, and
   showing that the former is a smaller set than our own earlier protocols
   implied (§8).

# 2 Background and related work

**Backprop-NEAT.** NEAT [Stanley & Miikkulainen 2002] evolves network topology
and weights together under historical markings and speciation. Ha's 2016
Backprop-NEAT [Ha 2016; `hardmaru/backprop-neat-js`] replaces weight evolution
with backpropagation: every evaluated genome is trained under a fixed inner
budget, so fitness measures how well a topology *can learn*, not how good its
inherited weights happen to be. It is described in a blog post and a browser
demo, and discussed in the *Neuroevolution* book, §10.1 [Risi, Tang, Ha &
Miikkulainen 2025], whose Figure 10.3 gives champion topologies and accuracies
that we use as a fidelity target.

**Efficient backprop over NEAT genomes.** PropNEAT [Merry, Riddle & Warren
2024] maps NEAT genomes bidirectionally to a layer-based architecture to make
GPU backpropagation efficient, evaluating on 58 PMLB binary-classification
datasets. Our dense evaluator (§5.2) solves a related problem — a genome graph
is slow to evaluate node-by-node — but with a different goal: PropNEAT seeks
throughput for training, while ours must be *provably the same function* as a
frozen reference implementation, because it is used to give baselines a matched
budget inside a comparison. We assert agreement to 1e-10 on values and
gradients rather than benchmarking speed alone.

**Implementation details as confounds.** The closest precedent is in deep RL.
Henderson et al. [2018] showed that reported RL results vary with seeds,
codebases and undocumented choices; Engstrom et al. [2020] went further and
attributed most of PPO's advantage over TRPO to "code-level optimizations" —
augmentations found only in implementations or described as auxiliary details.
Our finding is the neuroevolution analogue, with an additional twist: because
the inner learner is shared between the evolved condition and its controls, one
undocumented rule can advantage one arm of the comparison over another.

The same pattern has now appeared in LLM-driven evolutionary search. Oved et
al. [2026] evaluate three strategies over a 40-seed × 200-iteration grid and
find that the ranking depends on the budget: on one task the strategy that
looks worst at one seed is best at forty. Their recommendation — report the
frontier, not a point — is ours, transposed from LLM proposal budgets to
gradient budgets.

**A sister case in the same research line.** A companion study of competitive
coevolution on Slime Volleyball
[`competitive-coevolution-of-slimes`] reports, in its own release, that most of
the instability it set out to attribute to coevolution is injected at the last
step by the *champion-export rule*: Ha's rule exports an individual ranking on
average 60 of 128 in its own population, with a lineage counter uncorrelated
with skill (ρ = +0.06), and replacing it with a 1,024-game internal tournament
raised the exported champion's score from −2.33 to −2.16 in a preregistered
test on fresh runs, higher in 9 of 12 runs. Two independent reconstructions of
two different Ha demos thus failed in the same way: an undocumented selection
or export detail, not the headline mechanism, determined what was measured.

**Lamarckian and Baldwinian learning.** Backprop-NEAT inherits learned weights,
which makes it Lamarckian. Benito, Lutzeyer & Doerr [2026] revisit the
comparison with modern empirical and theoretical methods on Maximum Independent
Set and Maximum Cut over the GraphBench datasets, finding that Baldwinian and
Lamarckian evolution consistently outperform Darwinian evolution, and proving
runtime bounds on an extended DeceptiveLeadingBlocks benchmark. Our Block D
tests the Lamarckian/Baldwinian contrast inside Backprop-NEAT, where the
"local search" is gradient descent on a graph that evolution is simultaneously
rewiring.

# 3 The reconstruction

## 3.1 The algorithm as implemented

Genomes are arbitrary directed graphs over nine operators (sigmoid, tanh, relu,
gaussian, sin, abs, mult, square, add), seeded as minimal logistic regressions
and grown by add-node and add-connection mutation under an innovation registry,
with union crossover and K-medoids speciation. Recurrent edges are permitted.

Every evaluated candidate is trained by backpropagation through the *executed*
trace: RMSProp (lr 0.01, decay 0.999, L2 1e-3, clip 5.0), minibatch 10, 600
nominal updates, with a rollback check every 20 updates against the full
training split that reverts and stops at the first increase. Fitness is
`-error × (1 + 0.03·√connections)`. All of this follows `datafit-neat.js` and
`ml/neat.js`; the parameters were read from the source, not recalled.

## 3.2 Propagation, and why it needs a name

Ha's forward pass is asynchronous. A node becomes *touched* when any operand is
touched, every touched node is recomputed in node-id order each tick, and
ticking stops once every node has been touched. The output node holds id 3, so
it is recomputed *before* every hidden node and reads its operands from the
previous tick.

This has a consequence that bites any reimplementation. Wire a bias directly
into both the output and the hidden units — the obvious way to build an MLP
baseline — and every node is touched on tick 0, the loop stops, and the network
returns a constant. A 32×32 MLP built that way scores exactly 0.500. Our
baselines therefore route the output bias through a carrier node, and the test
suite asserts every baseline is alive under Ha's exact rule *and* under a
settled variant that ticks to a topology-determined fixed point.

We treat the two propagation modes as a declared factor (`ha2016`, `settled`)
rather than choosing one silently.

## 3.3 Infrastructure

Seven modules determine what a run computes and are bound to the release by a
SHA-256 fingerprint; the final-test command refuses to run if it has moved.
Each run record is written atomically and carries its own configuration,
metrics, compute accounting and serialized champion. Checkpoints restore the
generator state, innovation registry and counters, and a gate asserts that a
resumed run is bit-identical to an uninterrupted one. The sealed test split is
readable by exactly one module, and a gate poisons it with NaN and asserts that
search output does not change.

This machinery is what made the failures in §4 discoverable rather than
invisible, and it is the reason the invalidated results can still be audited.

# 4 Two failures, documented

Both failures were found after a confirmatory suite had completed. Neither
was a coding error in the usual sense: the code did what it said, and the tests
passed.

## 4.1 v1: a selection operator we chose ourselves

Our first `_reproduce` kept one elite per subpopulation and drew parents from
the better half. The reference does neither. `NEATTrainer.evolve` replaces the
whole population with offspring each generation, and `pickRandomIndex` samples
parents by fitness-proportionate roulette over the *entire* subpopulation with
weight `1/(-fitness + 0.01)`. At that slack, an error-0.3 genome breeds only
about 2.3× as often as an error-0.7 one.

That weakness is functional. It lets structural additions that are neutral or
mildly harmful survive long enough to combine. Truncation plus an elite removes
exactly that drift, so topologies stayed pinned at the minimal seed — and we
concluded, from champions with zero causally active hidden nodes, that Ha's
propagation rule traps evolution in logistic regression.

With the reference operator restored, the same propagation rule reached the
published accuracy regime. v1 was invalidated under its own freeze rule rather
than patched; its records are retained and marked.

## 4.2 v2: a stopping rule that starved the controls

v2's headline was that on the deceptive geometry, architecture search beat a
fixed 32×32 MLP in 10/10 paired replicates using 4.6 causally active hidden
nodes against the baseline's 65.

An external audit recomputed the compute each arm actually spent:

| Task | Evolved, gradient steps | Fixed MLP, gradient steps | Ratio |
|---|---:|---:|---:|
| XOR | 92,303 | 6,408 | 14× |
| Circles | 84,621 | 6,566 | 13× |
| Spirals | 132,138 | 2,506 | **53×** |

The cause is the rollback rule of §3.1. For an 8-node evolved graph, stopping
at the first full-batch increase is a mild regulariser. For a 69-node fixed
network it is a hard stop after roughly 42 updates. The comparison was between
two training budgets, not two architectures.

A probe on the validation split isolates the learner from the architecture:

| Arm | Steps | Validation accuracy |
|---|---:|---|
| 32×32 tanh, Ha's learner | 61–101 | 0.60–0.62 |
| 32×32 tanh, plain RMSProp | 20,000 | 0.78–0.91 |
| 32×32 **sin**, plain RMSProp | 600 | **0.94–0.96** |

The same architecture crosses from near-chance to competent when only the
stopping rule changes, and a fixed sinusoidal network beats every evolved
champion in the release after 600 plain updates.

Two of the five v2 claims were withdrawn and two narrowed. The comparisons that
survive are the two evolutionary ablations, which share the candidate budget by
construction: removing operator diversity costs accuracy on spirals (7/10
replicates) and circles (10/10), and removing gradient learning costs it on all
three tasks (9/10, 10/10, 10/10).

## 4.3 What the two failures have in common

In both cases the defect was a choice that the algorithm's published
description does not fix, that no test could fail on, and that changed the
*direction* of a headline conclusion. In v1 we chose wrongly relative to the
reference. In v2 we inherited the reference faithfully and it disadvantaged a
baseline the reference never had. Fidelity is not sufficient; neither is
reproducibility. Both failures reproduce perfectly.

# 5 Protocol v3: the evaluator as the experiment

## 5.1 Design

v3 is preregistered and frozen before any confirmatory compute (commit
`a343c66`): 1,920 runs over 150 cells, 30 paired replicates, five 2-D
geometries — Ha's XOR, circles and spirals, plus a 4×4 parity checkerboard and
a balanced three-arm spiral. Four blocks:

- **A — controls and budgets** (5 tasks, 8 conditions). Backprop-NEAT against
  v2's starved control (kept deliberately, to exhibit the artifact), the same
  network under a matched budget with no rollback, a sin-operator and a
  mixed-operator variant under the same budget, candidate-matched random
  architecture search, and the two evolutionary ablations.
- **B — propagation × fitness split** (3 tasks, 2×2). v2 varied both at once;
  this separates them.
- **C — selection-pressure dose response** (2 tasks, 7 levels). Roulette slack
  ∈ {1.0, 0.1, 0.01 = Ha's, 0.001}, tournament k ∈ {2, 4}, and the v1 operator.
- **D — Lamarckian vs Baldwinian** (3 tasks).

Budget matching is **paired within replicate**: a matched control receives the
gradient budget that Backprop-NEAT actually spent in that same cell, spread
over 60 restarts and selected on validation. The shard unit is therefore a cell
rather than a run.

Selection levels are compared on *realized standardised selection intensity*,
`I = (Σ pᵢfᵢ − mean f)/sd f`, recorded from the live population each
generation. Breeding odds of best over median are undefined for truncation,
where the median can have probability zero; intensity is finite for every
operator here.

Primary outcome is sealed-test **accuracy**, paired within replicate, with
Wilcoxon signed-rank, a bootstrap interval on the median paired difference, and
Holm correction over a pre-declared family of 35 Block A comparisons. Blocks B,
C and D are reported with intervals and marked exploratory. v2's mean-BCE
primary was abandoned because it inverted the XOR ranking under two confident
errors.

## 5.2 Making matched budgets affordable

A matched control needs ~132,000 gradient steps. The reference-faithful genome
evaluator walks one node per Python call and costs 16.3 ms per step on a
69-node network — 36 minutes per run, which would make the study impossible.

We add a dense evaluator that groups nodes by topological depth and computes
each depth with a single matrix product, preserving the clamping and operator
semantics of the genome path. Measured speedup is 41× (0.40 ms/step), taking a
matched run from 36 minutes to 53 seconds.

Because this evaluator is used *inside* a comparison, being fast is not enough;
it must be the same function. We assert agreement with the frozen genome path
to 1e-10 on forward values **and** gradients across tanh, sin, relu, gaussian
and mixed-operator networks, and assert that training through either path from
the same seed yields identical weights. Genomes containing `mult`, which
aggregates by product and is not expressible as one matrix product, raise and
fall back to the genome path.

# 6 Results

## 6.1 The headline: one rule, 0.261 accuracy

On spirals, the identical 32×32 tanh network scores **0.635** under v2's
evaluator and **0.896** under a matched budget. Nothing about the architecture
changes; only the learner's stopping rule does. That 0.261 swing is larger than
any difference v2 reported between any two conditions.

Sealed-test accuracy, 30 paired replicates:

| Condition | XOR | Circles | Spirals | Checkerboard | 3-arm spiral |
|---|---|---|---|---|---|
| Backprop-NEAT | 0.995 | 0.986 | 0.791 | 0.662 | 0.614 |
| Fixed MLP tanh, *v2's learner* | 0.985 | 0.977 | 0.635 | 0.590 | 0.536 |
| Fixed MLP tanh, matched | 0.990 | 0.982 | **0.896** | 0.700 | 0.589 |
| Fixed MLP sin, matched | 0.972 | 0.982 | **0.962** | 0.693 | 0.746 |
| Fixed MLP mixed, matched | 0.982 | 0.984 | **0.966** | **0.728** | **0.773** |
| Random arch., candidate-matched | 0.996 | 0.988 | 0.768 | 0.698 | 0.604 |
| Homogeneous tanh | 0.996 | 0.931 | 0.715 | 0.541 | 0.542 |
| Evolution only | 0.792 | 0.803 | 0.623 | 0.546 | 0.536 |

Paired differences on spirals (Backprop-NEAT − control), Wilcoxon with Holm
correction over the pre-declared family of 35:

| Control | Median diff | 95% CI | Wins | Holm *p* |
|---|---|---|---|---|
| Fixed MLP mixed, matched | −0.182 | [−0.197, −0.152] | 0/30 | 6.0e-05 |
| Fixed MLP sin, matched | −0.175 | [−0.195, −0.145] | 0/30 | 6.0e-05 |
| Fixed MLP tanh, matched | −0.110 | [−0.120, −0.090] | 1/30 | 6.3e-05 |
| Random arch., candidate-matched | +0.025 | [+0.003, +0.045] | 20/30 | 0.385 |
| Homogeneous tanh | +0.085 | [+0.043, +0.117] | 23/30 | 0.018 |
| Fixed MLP tanh, *Ha learner* | +0.158 | [+0.125, +0.190] | 30/30 | 6.0e-05 |
| Evolution only | +0.170 | [+0.140, +0.188] | 30/30 | 6.0e-05 |

The pattern holds on the two harder geometries we added. On the 3-arm spiral,
the budget-matched *tanh* network is worse than Backprop-NEAT (0.589 vs 0.614)
while the *sin* and *mixed* networks are far better (0.746, 0.773). The
operative variable is the operator prior, not the search: give a fixed
architecture the right activation and the evolved advantage disappears.

## 6.2 The conclusion stability matrix

| | Claim | v2 as run | v3 re-run | matched tanh | sin | mixed | cand.-matched |
|---|---|---|---|---|---|---|---|
| C1 | Beats a fixed MLP on spirals | supported | — | **reversed** | **reversed** | **reversed** | — |
| C2 | Gradient learning complements search | n.s. | supported | — | — | — | — |
| C3 | Operator diversity contributes | n.s. | supported | — | — | — | — |
| C4 | Few causal nodes beat a 65-unit MLP | supported | — | **reversed** | **reversed** | **reversed** | — |
| C5 | Selection beats random sampling | n.s. | — | — | — | — | n.s. |

Read down the first column: under Holm correction over its own analogous
family, v2's data supports only C1 and C4 — exactly the two that reverse. Read
across: the claims v2 stated most confidently are the ones that do not survive,
while the two that do survive (C2, C3) needed v3's 30 replicates to reach
significance at all.

## 6.3 Propagation and fitness split are not separable

Sealed-test accuracy / collapse rate, where *collapse* means a champion with
zero causally active hidden nodes:

| Propagation | Fitness | XOR | Circles | Spirals |
|---|---|---|---|---|
| `ha2016` | train | 0.811 / **0.17** | 0.940 / 0.03 | 0.720 / **0.20** |
| `ha2016` | validation | 0.951 / 0.00 | 0.959 / 0.00 | 0.726 / 0.03 |
| `settled` | train | 0.993 / 0.00 | 0.984 / 0.00 | 0.796 / 0.00 |
| `settled` | validation | 0.995 / 0.00 | 0.986 / 0.00 | 0.791 / 0.00 |

**v3-H3 is wrong.** It predicted collapse would depend on propagation and not on
the fitness split. Collapse requires *both*: Ha's propagation rule with
training-loss fitness. Either factor alone produces almost none. This is why v2
could not attribute its propagation result — the two factors interact, and v2
varied them together.

## 6.4 Selection pressure does not cause collapse

| Selector | Intensity | Collapse | Causal nodes | Test acc (spirals) |
|---|---|---|---|---|
| `roulette_s1.0` | 0.021 | 0.00 | 4.5 | 0.797 |
| `roulette_s0.1` | 0.046 | 0.00 | 4.4 | 0.812 |
| `roulette_s0.001` | 0.053 | 0.00 | 4.7 | 0.803 |
| `roulette_s0.01` (Ha's) | 0.054 | 0.00 | 4.3 | 0.791 |
| `tournament_k2` | 0.495 | 0.00 | 4.0 | 0.796 |
| `v1_truncation` | 0.690 | 0.00 | 4.3 | 0.809 |
| `tournament_k4` | 0.802 | 0.00 | 4.8 | 0.833 |

**v3-H4 is wrong, and its failure corrects our own earlier correction.** Collapse
is zero at every level of selection pressure, on both tasks, and causal size is
flat at 4.0–4.8. Protocol v1 was invalidated because its selection operator was
unfaithful to the reference, and we attributed the collapse v1 reported to that
infidelity. Here v1's own truncation operator collapses nothing and scores
*above* Ha's setting (0.809 vs 0.791). The operator was unfaithful; it was not
the cause. The cause was the propagation × fitness interaction of §6.3, which
v1 also had and which survived into v2's track A at a rate of 3/10 and 1/10.

We therefore reversed a conclusion, and then reversed our explanation of why we
had reversed it. Both corrections were only possible because the invalidated
records were retained.

## 6.5 Inherited weights are load-bearing

| Task | Lamarckian | Baldwinian |
|---|---|---|
| XOR | 0.995 | 0.726 |
| Circles | 0.986 | 0.567 |
| Spirals | 0.791 | 0.532 |

**v3-H5 answers decisively**, and not through effort: the Baldwinian condition
spent *more* gradient steps on spirals (159,083 against 134,449) and still
landed near chance. Benito et al. [2026] find Baldwinian and Lamarckian
evolution both beating Darwinian on combinatorial graph problems; our setting
differs in that the structure being searched is rewired every generation, so a
topology's learned weights are the only channel by which its learnability
reaches its descendants.

## 6.6 Hypotheses (v3)

| | Prediction | Outcome |
|---|---|---|
| v3-H1 | matched MLP's spirals deficit disappears or reverses | **reverses** |
| v3-H2 | sin MLP matches or beats on spirals | **holds** |
| v3-H3 | collapse depends on propagation, not fitness split | **fails** — interaction |
| v3-H4 | collapse/size vary with selection intensity | **fails** — flat |
| v3-H5 | Lamarckian vs Baldwinian, two-sided | **Lamarckian** |

Three of five failed. Had v3-H1 and v3-H2 failed instead, v2's claims would have been
restored and this paper would report that; the preregistration was written to
make either outcome publishable.

# 7 Protocol v4: is it the algorithm, or the protocol?

§6 is a result about one algorithm, and it invites one objection above all
others: that Backprop-NEAT is simply weak, or unusually dependent on an
undertrained control, and that a better topology search would not behave this
way. The objection is reasonable. It is also testable, and v4 tests it.

v4 asks a single question. **Is the sign of a search-versus-fixed comparison set
by the budget protocol, or by the search algorithm?** If the protocol sets it,
then a second, independently derived search put through the same controls
reverses in the same places. If the algorithm sets it, the two disagree.

## 7.1 The second algorithm

We use Cartesian Genetic Programming (Miller & Thomson 2000; Miller 2020) with
gradient-trained candidates. CGP was chosen because it differs from
Backprop-NEAT in both of the places an evolutionary algorithm can differ — how a
solution is represented, and how the population moves — rather than in one.

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

The `>=` in that table is the mechanism, not a detail. An offspring whose active
phenotype is unchanged but whose inactive genes have moved scores exactly the
same and is accepted, so the genotype random-walks through neutral space at
constant fitness and a single later mutation can switch on a region that took
many mutations to assemble. This is CGP's answer to the problem NEAT solves with
innovation numbers and weak selection, and it is a different answer: the
genotype-phenotype map leaves most genes inactive by design (Goldman & Punch
2015), and the drift through them measurably aids escape from local optima
(Turner & Miller 2015). Attaching a weight to every connection gene follows the
CGPANN line (Khan et al. 2013), and makes the weights part of the genotype, so an
offspring inherits its parent's *trained* weights for every gene it does not
mutate — the direct analogue of Backprop-NEAT's Lamarckian inheritance.

Everything that is not the search algorithm is held identical to §6's reference
condition: the five geometries, the inner learner (RMSProp under Ha's rollback
rule, 600 nominal updates, batch 10), the penalised validation fitness, weight
inheritance, and the candidate budget — `population × (generations + 1)`, which
is 1100 on XOR and circles and 2100 on the three hard geometries. CGP spends
exactly λ = 4 candidate evaluations per generation and never re-evaluates its
parent, so the budgets are comparable rather than nominally equal.

Two restrictions, declared in the preregistration rather than discovered
afterwards. CGP's operator set is the eight operators our dense evaluator can
express, excluding `mult`, which aggregates by product; this matters because a
48-node CGP row can decode to a chain deeper than the frozen evaluator's
16-tick settling bound, where it would silently return a value taken before the
deepest nodes had run, and the dense evaluator is exact at any depth. And every
v4 condition is evaluated under settled propagation, as §6's reference is. We do
not assume Ha's asynchronous rule would be safe here: on 300 random genotypes,
11 decoded phenotypes return a constant zero under it and all 300 are alive
under settling.

## 7.2 Design, and why the test data is new

Eight conditions, five geometries, thirty replicates: **1200 runs across 150
cells**, where a cell is one (task, replicate) and is the unit of sharding,
because the matched arms need the budget a reference actually spent in that same
cell.

| Condition | Role |
|---|---|
| `bpneat` | the released v3 Backprop-NEAT, re-run unchanged. Reference 1 |
| `cgp` | CGP (1+4) with neutral drift. Reference 2 |
| `fixed_tanh_ha` | 32×32 tanh MLP, 60 restarts, Ha's rollback rule — the unmatched control of §4.2. Shared |
| `fixed_tanh_matched_bpneat` | the same MLP at `bpneat`'s realized budget, no rollback |
| `fixed_mixed_matched_bpneat` | heterogeneous operators at `bpneat`'s realized budget |
| `fixed_tanh_matched_cgp` | the same MLP at `cgp`'s realized budget |
| `fixed_mixed_matched_cgp` | heterogeneous operators at `cgp`'s realized budget |
| `cgp_random_matched` | CGP genotypes drawn rather than selected, matched on candidates |

Each algorithm gets **its own** matched arms. The two do not spend the same
gradient budget, and a single shared matched arm would quietly favour whichever
algorithm spent less — the same class of mistake as §4.2, one level up.

v3's sealed test was opened once, on the splits its dataset seeds generate.
Reusing them here would make v4's test numbers a second look at data already
spent confirming a result, so v4 draws thirty entirely new replicates and the
burned-seed set now contains every seed v1, v2 and v3 spent. The cost is real
and we accept it explicitly: **v4 and v3 are not paired, and no v4 number may be
compared with a v3 number by a statistical test.** Every v4 claim is a within-v4
paired contrast. The comparison to v3 is qualitative — does the same thing
happen — and is written that way throughout. The benefit is that v4's headline is
an out-of-sample replication rather than a reanalysis.

Inference is pre-declared as two families, Holm-corrected within each and never
pooled: 35 tests anchored on `bpneat`, 25 anchored on `cgp`. A reproducibility
bridge re-runs v3's `backprop_neat` on the first ten v3 replicates of every task
under v4's code and compares champion topology, champion weights, realized
gradient steps and every validation metric to the committed records. The bridge
reads no test split, enters no table, and is a gate rather than a result.

## 7.3 Results: the artifact is universal, the reversal is not

**Four of our six preregistered hypotheses failed.** The scorecard
(`results/backprop-neat-v4/hypotheses.csv`) is computed from the release by the
rules written down before the suite ran, not narrated afterwards.

| | Hypothesis | Rule | Observed | |
|---|---|---|---|---|
| v4-H1 | the unmatched control is starved, not weak | matched beats unmatched on ≥4 of 5 | 5/5 | **holds** |
| v4-H2 | Backprop-NEAT's advantage reverses | search>unmatched ≥4, matched>search ≥3 | 3/5, 1/5 | **fails** |
| v4-H3 | the same for CGP | as v4-H2 | 2/5, 2/5 | **fails** |
| v4-H4 | the protocol sets the sign, not the algorithm | signs agree on ≥4 of 5 | 2/5 | **fails** |
| v4-H5 | the algorithms differ less than the protocol does | gap < 0.5× matching effect on ≥3 of 5 | 2/5 | **fails** |
| v4-H6 | CGP's selection adds little over its null | beats null on ≤2 of 5 | 0/5 | **holds** |

Sealed-test accuracy, mean over 30 paired replicates; search conditions marked \*:

| condition | XOR | circles | spirals | checkerboard | 3-arm spiral |
|---|---|---|---|---|---|
| Backprop-NEAT \* | 0.990 | 0.984 | 0.791 | 0.621 | 0.590 |
| CGP (1+4) \* | 0.989 | 0.986 | 0.741 | 0.584 | 0.558 |
| fixed tanh, unmatched | 0.983 | 0.971 | 0.640 | 0.570 | 0.528 |
| fixed tanh @ BP-NEAT budget | 0.986 | 0.979 | 0.898 | 0.686 | 0.583 |
| fixed tanh @ CGP budget | 0.986 | 0.980 | 0.867 | 0.695 | 0.572 |
| fixed mixed @ BP-NEAT budget | 0.977 | 0.985 | **0.964** | **0.714** | **0.783** |
| fixed mixed @ CGP budget | 0.978 | 0.985 | 0.960 | 0.713 | 0.753 |
| random CGP, candidate-matched | 0.989 | 0.985 | 0.757 | 0.694 | 0.601 |

**v4-H1 holds on every geometry.** The same 32×32 tanh network scores higher once
it is given the gradient budget the search actually spent — on spirals
0.640 → 0.898, with its success rate going from 0 of 30 replicates to 30 of 30.
§4.2's diagnosis replicates out of sample, on splits no earlier release opened.
The budget gap is the whole of it: on spirals the unmatched control spends 2,586
gradient updates where Backprop-NEAT spends 131,739, because Ha's rollback rule
stops a 69-node network after about 43 updates per restart.

**v4-H2 and v4-H3 fail, and that is the paper's main correction to its own §6.**
Search losing to a fairly trained control is *geometry-dependent*. It happens on
spirals for both algorithms and on checkerboard for CGP, and nowhere else: on
XOR and circles every condition sits above 0.97 and nothing can separate. We set
thresholds of ≥4 and ≥3 geometries in advance and got 1 and 2. §6's headline is
a true statement about the deceptive geometry; it is not a law about
architecture search, and v4 is what establishes the difference.

**v4-H4 and v4-H5 fail.** The unmatched and matched-tanh verdicts agree across the two
algorithms on 2 of 5 geometries. Column by column the agreement is 4/5 on the
unmatched control and 3/5 on matched tanh. v4-H5 fails its magnitude rule at 2/5,
but the significance picture is not the same thing and both belong in the
record: the two algorithms are statistically indistinguishable on four of five
geometries, and the single significant difference is the 3-arm spiral (median
−0.032, Holm *p* = 0.037). v4-H5 fails because where the matching effect is near
zero, any gap exceeds half of it.

> **Post-hoc.** Against the strongest control — mixed operators at a matched
> budget — the two algorithms' verdicts are identical on all five geometries:
> both win on XOR, both tie on circles, both lose on all three hard ones. v4-H4's
> preregistered rule reads the matched-*tanh* column, so this 5/5 agreement is
> an observation for a future study to test, not a result of this one. We report
> it because suppressing it would be the mirror image of the error the
> preregistration exists to prevent.

## 7.4 Selection is the component that does not pay

v4-H6 is the sharpest result in the release. **CGP never beats its
candidate-matched null on any geometry**, and on the two hardest the null beats
*it*: checkerboard median −0.090 (Holm *p* = 0.000), 3-arm spiral −0.047
(*p* = 0.011). Drawing CGP genotypes at random and training them under the same
learner is the better arm.

The mechanism is visible in the phenotypes. Selection drives CGP to small active
graphs — 3.9 active function nodes on spirals against 8.0 for the unselected
control, and 0.8 against 7.8 on checkerboard — in an encoding that is **not**
structurally capped, since all 48 function nodes are addressable from the first
generation. Backprop-NEAT reaches 4.2 causally active hidden units on spirals,
and there the cap *is* real: the reference's `new_node_rate = 0.2` over twenty
generations adds about four nodes to a lineage, which is why neither this
reconstruction nor v4 approaches the 34-node champion of the published
demonstration (§3). What v4 adds is that a search without that cap converges to
the same scale anyway, and does so *because of* selection rather than despite
it. Both algorithms inherit Ha's complexity penalty, `1 + 0.03·√connections`,
which is the plausible shared cause. **v4 contains no penalty-off arm, so
nothing here isolates it.**

Together with §6.4, where Backprop-NEAT did not beat candidate-matched random
search after correction, this is two independently derived algorithms whose
search component contributes nothing measurable on these tasks, while the inner
learner carries essentially all of the compute — on spirals, 131,739 gradient
updates against a few thousand mutation and selection operations.

## 7.5 Numerical equivalence is not run-level reproducibility

Building the CGP arm produced a result we did not go looking for. Its candidates
are trained on a vectorized evaluator that agrees with the frozen genome
evaluator to 2e-15 on forward values and 3e-15 on gradients, and to 3e-12 even
where a fifth of node values are clamped — they are the same model, and the test
suite pins it across weight scales.

They do not produce the same trained network. At the full 600-update budget, 5%
of CGP candidates diverge by more than 1e-6 in weight space, with a maximum of
4.8e-01, under either a 1e-12 relative perturbation of the initial weights or
merely the change of summation order between two provably equivalent
implementations. The median candidate does not move; the effect is bimodal, and
calling it "chaotic training" without that qualification would overstate a
minority effect as a universal one. The mechanism is RMSProp's epsilon floor:
when the gradient-square cache falls below `SMOOTH_EPS = 1e-8` the denominator
is pinned at 1e-4 and the update becomes `100 × lr × grad`, so a difference in
the gradient is amplified a hundredfold per step and then passed through
`square` and `gaussian` nodes. A tanh MLP does not do this at any rate
(1.6e-10 worst case over the same budget), which is why our learner-equivalence
gate can be held to the whole budget there and only to the first updates on CGP
phenotypes.

Two consequences. First, it is why every claim in this paper is a distribution
over thirty replicates: an architecture-search result reported from single runs
is not reproducible across numerically equivalent implementations of the same
model, let alone across machines. Second, it is a limit on the reproducibility
guarantee we offer. Our releases regenerate byte-identically from committed raw
records because the *records* are fixed; re-running the search from the seeds on
a different BLAS would not reproduce them candidate for candidate. Measurements:
`docs/v4-sensitivity.md`.

# 8 Protocol v5: NEAT's own machinery

§6 and §7 are about the evaluator. Both answers concern the inner learner and the
controls, and neither touched the algorithm's own mechanisms: every condition in
both studies ran the reference settings of the structural operators, the
complexity penalty, the speciation and the crossover. For a method named
*NeuroEvolution of Augmenting Topologies*, nothing so far had tested whether the
augmenting happens, or whether the machinery around it earns its place.

The motivating observation is on the record across all four earlier protocols.
This reconstruction's champions reach about four causally active hidden units on
spirals, against the 34 nodes and 96 connections of the champion in Figure 10.3
of the published description (§3). Part of that is arithmetic: `new_node_rate =
0.2` over twenty generations adds roughly four nodes to a lineage. But §7.4
showed something arithmetic does not explain — CGP, whose 48 function nodes are
all addressable from the first generation and which is under no growth cap,
*also* converges small, and smaller than its own unselected control. Selection
shrank the network in an encoding that could not have been prevented from
growing it.

## 8.1 Design

Five mechanisms as factors, on the three geometries that separate conditions,
with the budget-matched fixed network kept as the yardstick: **720 runs across 90
cells**, 30 paired replicates, 1,328,400 candidate evaluations, 85,498,011
gradient updates, 7.34 core-hours, zero failures. Dataset seeds are fresh again
(`70001 + 13i`), so v5 is paired with no earlier release.

XOR and circles are excluded **by the contract**, not by a later choice: §7.3
established that they saturate above 0.97 for every condition and contribute
nothing except to drag hypotheses below their task thresholds. Declaring the
exclusion in advance is the difference between a design decision and a reported
subset.

`P_ADD_NODE` and `P_ADD_CONNECTION` are module constants in the frozen v2 code,
which is why v5 needs new code at all: the rates cannot be varied without it, and
the frozen module may not be edited. A gate asserts that at the reference rates
v5's `mutate` produces byte-identical genomes to the frozen one, so a changed
rate changes a rate and nothing else.

One arm is **compound and is declared as such**. Holding the candidate budget
while quartering the population necessarily quadruples the generations *and*
shrinks each species from about twenty members to about five; the two cannot be
separated without a third change. `neat_deep_narrow` therefore answers "does
trading width for depth help at a fixed evaluation budget" and does not isolate
depth.

## 8.2 Results

**Six of seven preregistered hypotheses hold**, and the picture is considerably
more favourable to NEAT than §6 and §7 alone would suggest. Causally active
hidden units and sealed-test accuracy, means over 30 replicates, on spirals /
checkerboard / 3-arm spiral:

| condition | causal units | sealed-test accuracy |
|---|---|---|
| reference | 4.2 / 2.1 / 1.6 | 0.791 / 0.662 / 0.597 |
| `p_add_node` 0.2 → 0.5 | 7.8 / 5.0 / 1.8 | 0.830 / 0.685 / 0.606 |
| − complexity penalty | 4.9 / 4.8 / 4.0 | 0.807 / 0.650 / 0.616 |
| **both** | **8.0 / 8.2 / 7.5** | **0.835 / 0.688 / 0.621** |
| − speciation | 3.6 / 0.8 / 0.5 | 0.781 / 0.580 / 0.568 |
| − crossover | 5.0 / 2.2 / 1.3 | 0.760 / 0.640 / 0.593 |
| ¼ population, 4× generations | 10.3 / 4.1 / 1.4 | **0.864** / 0.632 / 0.599 |
| fixed network, matched budget | 65 | **0.964 / 0.712 / 0.765** |

**Complexification is real, and two reference settings suppress it.** Every arm
is offered node additions at the same rate the reference is — about 404
node-addition events per run at `p_add_node = 0.2` and about 993 at 0.5 — but
what survives into a champion is decided by the fitness. Removing the complexity
penalty grows champions significantly (v5-H1, 2/3); raising the structural
mutation rate grows them (v5-H2, 2/3); removing both grows them on every geometry
(v5-H3, 3/3), from 1.6 causal units to 7.5 on the 3-arm spiral. The penalty is
`1 + 0.03·√connections`, inherited from the reference fitness, and it is doing
considerably more than regularising.

**And the larger topologies buy accuracy** (v5-H4, holds). This is the hypothesis
the protocol existed to test. Had it failed while v5-H1 to v5-H3 held, the reading
would have been that NEAT's networks are small because the fitness asks them to be
and that growing them is pointless — a tidy result, and the one we expected. It
did not fail: the unconstrained arm improves on the reference on two geometries,
and the width-for-depth trade reaches 0.864 on spirals against 0.791.

**Speciation earns its place; crossover does not.** Removing speciation loses
significantly on two of three geometries (v5-H5, holds) and collapses the
checkerboard champion to 0.8 causal units — the smallest network in the release,
and a direct demonstration of what speciation is for. Removing historical-marking
crossover loses on one geometry only (v5-H6, **fails**). Of the five mechanisms
tested, crossover is the one this study cannot show is load-bearing, which is
notable because it is among the mechanisms the original NEAT paper argues for
most explicitly.

**The ceiling is not NEAT's to raise** (v5-H7, holds, 0/3). No variant beats the
budget-matched fixed network on any geometry. The best result anywhere in v5 —
0.864 on spirals — closes about a third of the distance to 0.964. Freeing
complexification moves the algorithm part of the way across a gap that a
better-chosen fixed architecture crosses for nothing.

## 8.3 What this changes about §6 and §7

It narrows them, in NEAT's favour, without overturning them. §7.4 reported that
the search component contributes nothing measurable; v5 shows that a substantial
part of that is self-inflicted by two reference hyperparameters rather than
intrinsic to the method, and that the mechanism NEAT is named for does work once
the fitness stops opposing it. What survives unchanged is the comparison that
motivated the whole project: even an unhobbled NEAT does not reach a fixed
network with a well-chosen operator set at the same gradient budget. The search
space prior remains where the performance is.

# 9 Limitations

**Two-dimensional synthetic tasks.** All five geometries are 2-D binary
classification. The genome encoding fixes two input nodes in a module frozen by
the v2 fingerprint, so extending to higher-dimensional real datasets would have
required editing frozen code; we skipped that rather than do it badly. Our
claims are about how evaluation choices move conclusions on these tasks, and we
do not know the magnitudes transfer.

**Two algorithms, one reference implementation.** §7 answers the first version
of this limitation: Cartesian Genetic Programming, which shares no code, no
encoding and no selection mechanism with Backprop-NEAT, shows the same starved
control (v4-H1, 5/5) and the same failure of its search component to beat its own
null (v4-H6, 0/5). What v4 also shows is that the *reversal* does not generalise as
cleanly as §6 suggests — it is geometry-dependent, and v4-H2 and v4-H3 both failed.
Two algorithms is still two. Whether the pattern holds for methods with a
different inner learner — the learner, not the search, is what carries the
compute here — remains untested.

**Two of our five geometries carry no information.** XOR and circles saturate
above 0.97 for every condition in v4, which is most of why v4-H2 and v4-H3 fail their
task counts: a geometry on which nothing can lose is a geometry on which nothing
can be learned. v5 excludes them by contract, which fixes the problem going
forward and leaves every v5 hypothesis scored out of three — a coarse
denominator, and the price of the fix.

**v5 varies each mechanism at two settings, not along a dose-response.** The
penalty is on or off and `p_add_node` is 0.2 or 0.5. We can say that both
constrain complexification and that releasing them helps; we cannot say where
either optimum lies, or whether a penalty *tuned* rather than removed would beat
both. The width-against-depth arm is additionally compound by construction
(§8.1).

**The analysis code is not fingerprinted.** Five science modules per protocol
are hash-bound to their release. `analysis.py`, which decides how the
preregistered hypotheses are *scored*, is not — it cannot change what a run
computes, which is why it was excluded, but it can change what a run is taken to
mean. It was present in the v4 freeze commit before any run record existed and
that is checkable (`git ls-tree -r --name-only 4fa3009 src/bpneat/v4/`), but it
rests on git history rather than on a hash. A scoring fingerprint, recorded at
freeze time alongside the science one, is the obvious fix.

**Matched on gradient steps, not on everything.** Candidate count and gradient
steps cannot both be matched between an evolutionary condition and a multistart
control. We match one, report the other, and present accuracy against realized
compute rather than a single budget point.

**Thirty replicates.** Enough for the paired tests we report, not enough to
characterise tails. Collapse rates in particular are proportions estimated from
30 runs per cell.

**A sin-MLP beating evolved champions is not a claim that sin is the answer.**
It is a demonstration that the operator prior, not the search, can account for
a result that was attributed to search.

# 10 Conclusion

We set out to measure a neuroevolution algorithm and twice measured our own
evaluator instead. Both times the responsible choice — a selection operator, a
learner's stopping rule — was absent from the algorithm's published
description, invisible to the test suite, and sufficient to flip a headline
conclusion.

The remedy we propose is not more care. It is to promote the suspect details to
factors and report the surface rather than a point: which conclusions hold
across evaluator settings, and which are artifacts of one. For
evolution-with-learning methods the inner learner is shared between the method
and its baselines, so a single undocumented rule can advantage one arm of a
comparison; that makes the stability matrix a minimum, not a luxury.

The uncomfortable corollary is that our v2 release was reproducible,
fingerprinted, preregistered, sealed-test-once, byte-identical on
regeneration — and wrong in two of five claims. Reproducibility guarantees that
others can obtain the same numbers. It says nothing about whether the numbers
answer the question.

v4 applies the same instrument to v3 and finds the same kind of fault, which is
the point of building it. §6's headline — that a fairly trained fixed network
beats the search — survives on the deceptive geometry and fails as a general
claim; two of our preregistered hypotheses said otherwise and were wrong. We
report that in the same voice we used to report v2's errors, because a method
for catching one's own mistakes is worth nothing if it is retired once it starts
finding them.

v5 then corrects us in the other direction, which is worth saying plainly because
it is the only correction in this paper that makes the studied algorithm look
better rather than worse. Having reported across two protocols that the search
component contributes nothing measurable, we found on turning NEAT's own
mechanisms into factors that a substantial part of that is inflicted by two of
its reference hyperparameters and not by the method: complexification works once
the fitness stops opposing it, the larger topologies it then builds do score
better, and speciation is load-bearing. A critique that had stopped before §8
would have been accurate about what we measured and wrong about why.

What did survive two independently derived algorithms, and NEAT's own machinery
set free, is narrower and more useful than what we set out to show. The control that looks weak is starved, on
every geometry and under both algorithms. The search component — the part of
these methods that the papers are about — never beats a candidate-matched null
drawn from the same space, and on the hardest geometries it is beaten by one.
The compute goes almost entirely into the inner learner, and so does the result.
For an evolution-with-learning method, the honest question is not whether the
search found something, but whether anything would have been lost by not
searching at all; answering it costs one extra arm, and we would now treat a
paper without it the way we treat one without a seed.

# References

- Benito, I., Lutzeyer, J. F., & Doerr, B. (2026). *A Fresh Look at Lamarckian
  Evolution and the Baldwin Effect.* LIX, CNRS, École Polytechnique.
  arXiv:2605.28703.
- Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L.,
  & Madry, A. (2020). *Implementation Matters in Deep Policy Gradients: A Case
  Study on PPO and TRPO.* ICLR 2020. arXiv:2005.12729.
- Goldman, B. W., & Punch, W. F. (2015). *Analysis of Cartesian Genetic
  Programming's Evolutionary Mechanisms.* IEEE Transactions on Evolutionary
  Computation 19(3), 359–373.
- Ha, D. (2016). *Neural Network Evolution Playground with Backprop NEAT.*
  blog.otoro.net. Source: `github.com/hardmaru/backprop-neat-js`.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D.
  (2018). *Deep Reinforcement Learning that Matters.* AAAI 2018.
- Khan, M. M., Ahmad, A. M., Khan, G. M., & Miller, J. F. (2013). *Fast learning
  neural networks using Cartesian genetic programming.* Neurocomputing 121,
  274–289. doi:10.1016/j.neucom.2013.04.005
- Löchli, R. (2026). *Competitive coevolution of slimes.*
  `github.com/ReloadLightly/competitive-coevolution-of-slimes`. Cited only from
  that repository's own release files.
- Merry, M., Riddle, P., & Warren, J. (2024). *PropNEAT — Efficient
  GPU-Compatible Backpropagation over Neuroevolutionary Augmenting Topology
  Networks.* arXiv:2411.03726.
- Miller, J. F. (2020). *Cartesian Genetic Programming: its status and future.*
  Genetic Programming and Evolvable Machines 21(1–2), 129–168.
  doi:10.1007/s10710-019-09360-6
- Miller, J. F., & Thomson, P. (2000). *Cartesian Genetic Programming.* EuroGP
  2000, LNCS 1802, 121–132. doi:10.1007/978-3-540-46239-2_9
- Oved, T., Pony, R., Naparstek, O., & Barzelay, U. (2026). *Evolution or
  Illusion? Rethinking Evaluation in LLM Evolutionary Search.* IBM Research.
  arXiv:2609.19799.
- Risi, S., Tang, Y., Ha, D., & Miikkulainen, R. (2025). *Neuroevolution:
  Harnessing Creativity in AI Agent Design*, §10.1. neuroevolutionbook.com.
- Stanley, K. O., & Miikkulainen, R. (2002). *Evolving Neural Networks through
  Augmenting Topologies.* Evolutionary Computation 10(2), 99–127.
- Turner, A. J., & Miller, J. F. (2015). *Neutral genetic drift: an investigation
  using Cartesian Genetic Programming.* Genetic Programming and Evolvable
  Machines 16(4), 531–558. doi:10.1007/s10710-015-9244-6
