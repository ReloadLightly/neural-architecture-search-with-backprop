---
title: "When the Evaluator Decides: Implementation Details Determine the Conclusions of an Evolution-with-Learning Study"
author: Roland Löchli
venue: GECCO 2027 (draft)
---

# Abstract

We reconstruct David Ha's Backprop-NEAT (2016) — NEAT topology search in which
backpropagation trains every evaluated candidate — and run it three times under
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
evaluator setting.

<!-- RESULTS-SUMMARY -->

We argue that for evolution-with-learning algorithms, where an inner learner
and an outer search interact, the evaluator is not a neutral measuring device
but a component of the method, and that reporting a single budget point is
closer to reporting a hyperparameter than a result.

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

<!-- V3-RESULTS -->

# 7 Limitations

**Two-dimensional synthetic tasks.** All five geometries are 2-D binary
classification. The genome encoding fixes two input nodes in a module frozen by
the v2 fingerprint, so extending to higher-dimensional real datasets would have
required editing frozen code; we skipped that rather than do it badly. Our
claims are about how evaluation choices move conclusions on these tasks, and we
do not know the magnitudes transfer.

**One algorithm, one reference.** We study Backprop-NEAT because it has a
public reference implementation to audit against. Whether the specific
sensitivities here generalise to other evolution-with-learning methods is
untested; the deep-RL precedents suggest the *pattern* does.

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

# 8 Conclusion

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

# References

- Benito, I., Lutzeyer, J. F., & Doerr, B. (2026). *A Fresh Look at Lamarckian
  Evolution and the Baldwin Effect.* LIX, CNRS, École Polytechnique.
  arXiv:2605.28703.
- Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L.,
  & Madry, A. (2020). *Implementation Matters in Deep Policy Gradients: A Case
  Study on PPO and TRPO.* ICLR 2020. arXiv:2005.12729.
- Ha, D. (2016). *Neural Network Evolution Playground with Backprop NEAT.*
  blog.otoro.net. Source: `github.com/hardmaru/backprop-neat-js`.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D.
  (2018). *Deep Reinforcement Learning that Matters.* AAAI 2018.
- Löchli, R. (2026). *Competitive coevolution of slimes.*
  `github.com/ReloadLightly/competitive-coevolution-of-slimes`. Cited only from
  that repository's own release files.
- Merry, M., Riddle, P., & Warren, J. (2024). *PropNEAT — Efficient
  GPU-Compatible Backpropagation over Neuroevolutionary Augmenting Topology
  Networks.* arXiv:2411.03726.
- Oved, T., Pony, R., Naparstek, O., & Barzelay, U. (2026). *Evolution or
  Illusion? Rethinking Evaluation in LLM Evolutionary Search.* IBM Research.
  arXiv:2609.19799.
- Risi, S., Tang, Y., Ha, D., & Miikkulainen, R. (2025). *Neuroevolution:
  Harnessing Creativity in AI Agent Design*, §10.1. neuroevolutionbook.com.
- Stanley, K. O., & Miikkulainen, R. (2002). *Evolving Neural Networks through
  Augmenting Topologies.* Evolutionary Computation 10(2), 99–127.
