# Reference targets from the published sources

Fidelity benchmarks this reconstruction is measured against. They are *targets*,
not results, and they are not pooled with anything produced here.

## Sources

- David Ha, ["Neural Network Evolution Playground with Backprop
  NEAT"](https://blog.otoro.net/2016/05/07/backprop-neat/) (2016) — the
  interactive playground and its narrative.
- `hardmaru/backprop-neat-js` — the source this implementation was audited
  against.
- Risi, Tang, Ha & Miikkulainen, [*Neuroevolution: Harnessing Creativity in AI
  Agent Design*](https://neuroevolutionbook.com/), §10.1 "Neural Architecture
  Search with NEAT", Figures 10.1 and 10.3.

## Operator set (Figure 10.1)

input, output, bias, sigmoid, tanh, relu, gaussian, sine, abs, mult, add,
square. This matches `bpneat.genome.ACTIVATIONS` exactly — nine evolvable
operators plus the three structural node types.

## Champion targets (Figure 10.3)

| Task | Gen | Nodes | Connections | Fitness | Train | Test |
|---|---|---|---|---|---|---|
| XOR | 6 | 8 | 12 | −0.1451 | 96.0% | 94.3% |
| Circles | 6 | 11 | 20 | −0.266 | 97.5% | 96.3% |
| Spirals | 17 | 34 | 96 | −0.5017 | 87.0% | 81.5% |

The book's qualitative reading: XOR relies on `abs` and `relu` to form long
lines with sharp corners; circles exploit `sine`, `square` and `gaussian` for a
radially symmetric feature; spirals need a large topology to approximate a
complex boundary. Treat this as a hypothesis to test (H4), never as a criterion
for selecting which runs to report.

## Where this reconstruction currently lands

> The table below was stale: it reported means over *partially completed*
> replicates taken mid-run. It is now recomputed from the finished release.

Track A (`ha2016`, Ha's exact break rule), `backprop_neat`, ten paired
replicates, from `results/backprop-neat-v2/track-a/`:

| Task | Nodes (ref) | Nodes (here) | Conns (ref) | Conns (here) | Accuracy (ref, test) | Accuracy (here, **sealed test**) |
|---|---|---|---|---|---|---|
| XOR | 8 | 5.8 | 12 | 7.5 | 94.3% | 75.1% |
| Circles | 11 | 7.6 | 20 | 11.0 | 96.3% | 92.9% |
| Spirals | 34 | 9.2 | 96 | 14.9 | 81.5% | 73.8% |

Every task undershoots on size, spirals badly: 9.2 nodes against 34, 14.9
connections against 96. On accuracy, circles come closest (92.9% against
96.3%); XOR undershoots most, and its 75.1% is a **bimodal mixture** — three of
ten champions collapse to the logistic floor and the other seven reach
0.630–1.000 — so the mean does not describe any individual run.

For comparison, track B (`settled` propagation, validation fitness) reaches
0.996 / 0.986 / 0.787 on the same tasks. Neither track reproduces the
reference *champion sizes*.

## The size discrepancy is understood, and unresolved

`NEATTrainer.applyMutations` in `ml/neat.js` adds **at most one node per
offspring**, at `new_node_rate = 0.2`:

```js
applyMutations: function(g) {
  if (Math.random() < this.new_node_rate) g.addRandomNode();
  if (Math.random() < this.new_connection_rate) g.addRandomConnection();
  g.mutateWeights(this.mutation_rate, this.mutation_size);
}
```

Over 17 generations that is ≈3.4 added nodes per lineage, i.e. a champion of
roughly 7–8 nodes — which is what this reconstruction produces. A 34-node
champion needs about thirty node additions, which that rate cannot deliver in
seventeen generations.

The most likely explanation is that Figure 10.3's champion comes from the
interactive playground, where a session accumulates far more mutation events
than a clean seventeen-generation batch run, and where the displayed generation
counter need not correspond to the number of mutation events a lineage has
seen. Alternative explanations not yet excluded: a different mutation
configuration behind the demo UI, or the effect of `extinction_rate = 0.5`
combined with the hall of fame on which genome ends up displayed.

**Consequence for claims.** This reconstruction may not be described as
reproducing Ha's Figure 10.3 champions. It reaches the reference *accuracy*
regime on circles, approaches it on XOR under settled propagation, and falls
short on spirals. Network size is not reproduced on any task. Anything said
about "what Backprop-NEAT discovers" is a statement about this implementation
under the frozen protocol, not about the published demo.
