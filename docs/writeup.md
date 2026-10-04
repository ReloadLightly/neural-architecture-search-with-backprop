# A Network That Must See Together

*Backprop-NEAT as the second experiment of* After 2022

> Draft for Chapter 4 of *After 2022: Japan's Search for a Novel Foreign
> Policy*. Every number in this document traces to
> [`results/backprop-neat-v2/`](../results/backprop-neat-v2/) — protocol v2,
> frozen before execution, sealed test evaluated exactly once.

## 1. The sentence and the mechanism

In December 2022, Japan's National Security Strategy committed the country to
"build a multilayer network among its ally and like-minded countries, expand
it, and strengthen deterrence."

Read as policy, the sentence is unremarkable — every foreign ministry promises
networks. Read as a specification, it is startlingly precise, because each of
its three verbs names a distinct operation on a graph. *Build a multilayer
network among like-minded countries* is a topology statement: choose the nodes,
lay the connections. *Expand it* is a growth statement — and growth is the
operative word, because the strategy does not imagine designing the final
network in advance; it imagines adding structure incrementally to a network
that already works. *Strengthen* is a parameter statement: leave the topology
as it is and optimise the weights on the edges you already have.

There is an algorithm whose entire identity is this division of labour. NEAT —
NeuroEvolution of Augmenting Topologies — begins with the smallest network that
can function and complexifies it: structural mutations add nodes and
connections one at a time, and the additions survive only if the grown network
still performs. Backprop-NEAT, David Ha's 2016 variant, adds the third verb:
every candidate topology is trained by backpropagation under a fixed budget, so
selection judges architectures by what they can *learn to do*, not by what
their initial weights happen to compute. Build, expand, strengthen — as
literally as an algorithm can enact a sentence.

And the correspondence runs one level deeper, to the reason a network is wanted
at all. The tasks Backprop-NEAT is tested on — XOR, concentric circles, twin
spirals — are chosen because they defeat any single decision line. No
individual edge of a trained network classifies a spiral; only the ensemble
does. That is the situation the strategy describes: no single bilateral
relationship classifies a regional contingency as tolerable or intolerable,
deterrable or not. The judgment the network exists to make is collective or it
is not made.

A correspondence this exact deserves an experiment — and a fence. So the fence
first, in the words of the release's own claim ladder: this experiment supports
no claim about geopolitics, forecasting, or foreign policy. It is
two-dimensional synthetic classification. What it earns is not evidence about
Japan but *design knowledge* about the mechanism the sentence assumes — where
growing-and-strengthening pays, what it costs, and what silently breaks it.
That knowledge is spent in the third experiment of this series, which points an
evolutionary search at the open world the sentence actually lives in.

## 2. Why reconstruct a 2016 browser demo

The series this chapter belongs to asks one question three times, in worlds of
increasing openness: what must evolution hold fixed for what it discovers to be
trustworthy? Experiment 1 held an environment fixed and let two populations
train against each other; its sharpest lesson was that competence could be
*reached but not held* — champion quality oscillated once selection lost its
memory. Experiment 3 will hold a model of Japan's diplomatic world fixed and
let a large language model propose the mutations. Between them, this experiment
holds the learner fixed — same optimiser, same budget, same data, same
propagation, for every condition — and lets only structure evolve, on the
smallest problems that still have the property that matters: deceptive
geometry.

Reconstruction, rather than invention, is the point. A new toy environment
would prove nothing about our instruments, because failure could always be
blamed on the toy. Ha's Backprop-NEAT is a published reference with published
behaviour: an operator set, a fitness function, published champions with known
accuracies. Rebuilding it exactly — audited against `backprop-neat-js`
line by line, not from memory — turns every deviation into information. Either
the reconstruction reaches the reference regime, which calibrates the
instrument, or it fails to, and the specific failure teaches us what the
published account left unsaid. Both happened here, and the second kind of
result turned out to be the valuable one.

Cheapness is also a feature, not an embarrassment. The full confirmatory
matrix — 360 runs, 268,800 candidate evaluations, 11.65 million realized
gradient steps — costs 1.87 core-hours. An instrument that cheap can afford
the disciplines that expensive instruments skip: a protocol frozen before
execution, ten paired replicates, matched controls for every mechanism, and a
sealed test evaluated exactly once.

## 3. What was actually run

The protocol (frozen in `docs/protocol-freeze-v2.md` before the confirmatory run)
crosses six conditions with three tasks and ten paired replicates, on two
tracks that are never pooled. Track A runs Ha's exact propagation rule and
training-loss fitness: the historical reconstruction. Track B runs the same
graphs settled to a topology-determined fixed point, selected on validation:
the mechanism comparison. The six conditions are an ablation ladder, each
removing one mechanism — operator diversity (homogeneous tanh), gradient
learning (evolution only), evolutionary selection (random architecture
search), topology search (a fixed 32×32 MLP), and everything nonlinear
(logistic floor) — with every condition, evolved or fixed, expressed as a
genome and trained by the same inner learner. A difference between two
conditions is therefore attributable to the mechanism named, not to two
different pieces of training code.

The discipline was tested in the way that matters: by having something to
lose. Protocol v1 completed 233 runs before a fidelity audit found that its
selection operator did not match the reference — elitism plus truncation where
Ha uses gentle fitness-proportionate roulette with no elite. The difference
sounds cosmetic. It is not: the reference's weak selection is what lets
neutral structural additions survive long enough to combine, and the stricter
operator suppressed exactly that drift. v1 had "found" that Ha's propagation
rule traps evolution in logistic regression; with reference selection
restored, the same rule reaches the published regime, and the finding reversed
sign. Under the freeze rule a fix that changes scientific output invalidates
the version rather than amending it, so all 233 runs were declared dead,
retained for audit, and barred from citation. The false conclusion died before
it could be published, which is the whole purpose of the machinery.

## 4. What the experiment found

Four results survive the sealed test, and one welcome negative.

**Search pays exactly where the geometry is deceptive.** On spirals — the one
task where no simple boundary exists — Backprop-NEAT beat the fixed MLP, random
architecture search, evolution-only and the linear floor on sealed-test loss in
ten out of ten paired replicates, with bootstrap intervals excluding zero. On
XOR and circles, the controls matched or beat it: a fixed MLP is entirely
adequate for easy geometry, and random architecture sampling won circles
outright. The honest summary is conditional, and stronger for it: architecture
search earned its budget only where the problem deceived, and that is precisely
the regime one would deploy it in.

**The mechanisms are complementary, not redundant.** Evolution without gradient
learning lost on every task, decisively (XOR 9/10, circles 10/10, spirals
10/10). Gradient learning without topology search — the fixed MLP — lost
exactly where deception began. Neither half of Backprop-NEAT's name is
decorative.

**Architectures specialise in operators, not just size.** The causally active
hidden units of the spiral champions are 52% `sin`; the circle champions are
53% `square` and `gaussian`; the XOR champions are 46% `mult`. Evolution
discovered periodicity for the periodic task, radial symmetry for the radial
task, and multiplicative interaction for the parity task — the qualitative
behaviour the *Neuroevolution* book describes for this system, here measured
rather than narrated.

**Represented structure is not computation.** The champions that beat a
65-hidden-unit MLP on spirals did it with 4.6 causally active hidden nodes —
measured on the executed trace, counting only structure whose output actually
reaches the decision. The gap between what a network *contains* and what it
*computes* is not a rounding error; it is nearly an order of magnitude, and it
is the difference between counting a network and understanding one.

**And the propagation rule mattered.** Ha's exact break rule — a scheduling
detail of the forward pass, invisible in any published description of the
algorithm — cost a quarter of the achievable accuracy on XOR (0.751 sealed,
against 0.996 settled) and changed what evolution could discover on every
task. This is the welcome negative: the "same" algorithm under two defensible
readings of its forward pass is two different experiments.

## 5. Three readings for Experiment 3

The findings are about toy geometry. The readings are about the instrument,
and they transfer.

**First: deception is the regime that justifies the budget.** Where the
landscape is separable, a fixed design plus optimisation wins on simplicity;
search pays rent only where good solutions look bad from a distance. This
sharpens the question Experiment 3 must answer before it spends money: not
"can evolution find good foreign-policy portfolios?" but "is the portfolio
landscape deceptive enough that anything simpler would be blind?" The design
therefore freezes what is settled — the pillars of policy where the answer is
already institutionalised — and evolves only where settledness is lowest and
the combinatorial space is genuinely open. What must evolution hold fixed? is
answered here at toy scale: hold fixed everything you already trust; evolve
where trust runs out.

**Second: count what binds, not what exists.** The causal-subgraph measure is
this experiment's quiet methodological contribution: it refuses to credit a
network for structure that never reaches the output. The corresponding failure
in the target domain is endemic — a score that counts partnerships cannot tell
fourteen shallow agreements from four that operate, any more than a node count
can tell the fixed MLP's 65 hidden units from the champion's 4.6 causal ones.
Composite power indices that aggregate counted instruments inherit exactly
this blindness, and an evolutionary search rewarded on such a metric will
learn to *represent* rather than to *compute* — reward-hacking, in the
domain's own vocabulary. Experiment 3's evaluator must therefore measure the
policy network the way the trace measures the graph: by what demonstrably
reaches the outcome.

**Third: the evaluator is part of the experiment.** The propagation-rule
result and the v1 invalidation are the same lesson at two depths.
An implementation detail of the world model changed what evolution could
discover; a plausible "improvement" to selection sterilised the search
entirely — the reference's gentleness, an error-0.3 genome breeding barely
2.3× as often as an error-0.7 one, turned out to be load-bearing, because
drift is what novelty survives on (the open-endedness literature has argued
this on principle since Lehman and Stanley; here it has a measured cost).
For Experiment 3, where the evaluator is an open-world construction rather
than a labelled dataset, this is the central warning: the evaluator's
mechanics are not scaffolding around the experiment, they *are* the
experiment, and they deserve the same freezes, the same fidelity audits, and
the same willingness to invalidate that the search itself gets. A search over
policies can fail in its world model just as silently as a control network
zeroed by a propagation rule — and without a Figure 10.3 to be checked
against.

## 6. What this does not show

The tasks are two-dimensional synthetic classification; nothing here forecasts
anything. The reference is a single implementation by a single author, and
this reconstruction reaches its accuracy regime without reproducing its
published champion sizes — the discrepancy is understood, documented in
[`reference-targets.md`](reference-targets.md), and stands as a caution about
what "reproducing" a published demo even means. Ten replicates support paired
uncertainty summaries, not universal laws. And the epigraph of this chapter
remains an analogy, deliberately spent as motivation rather than evidence: the
experiment shows that build–expand–strengthen is a mechanism that can be made
to work and made to fail for measurable reasons, in the smallest world that
has one. Whether the mechanism serves in the world the sentence was written
for is precisely what the next experiment is designed to ask, with these
lessons priced in.
