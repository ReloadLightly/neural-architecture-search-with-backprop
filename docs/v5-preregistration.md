# Protocol v5 — preregistration

**Status: frozen before compute.** Written, gated and committed before any v5
confirmatory run existed. The freeze record is at the end of this file.

## The question

v3 and v4 asked what the *evaluator* does to a comparison, and both answers were
about the inner learner and the controls. Neither touched NEAT's own machinery:
every condition in both studies ran the reference configuration of the structural
operators, the complexity penalty, the speciation and the crossover.

That is a gap, and it is the gap a reader interested in neuroevolution rather
than in optimisers will notice first. The algorithm is called *NeuroEvolution of
Augmenting Topologies*. Nothing so far has tested whether the augmenting
happens, whether the speciation that is supposed to protect it does anything, or
whether the crossover that is supposed to recombine it earns its place.

**v5 makes NEAT's five mechanisms the factors.**

The motivating observation is on the record and is not new here. Across v2, v3
and v4 this reconstruction's champions reach about four causally active hidden
units on spirals, against the **34 nodes and 96 connections** of the champion in
Figure 10.3 of the published description (`docs/reference-targets.md`). The
reference's `new_node_rate = 0.2` over twenty generations adds roughly four nodes
to a lineage, so the arithmetic alone explains part of it. v4 then showed
something the arithmetic does not explain: Cartesian Genetic Programming, whose
48 function nodes are all addressable from the first generation and which is
therefore under no such cap, *also* converges to small phenotypes — 3.9 active
nodes on spirals against 8.0 for its own unselected control, and 0.8 against 7.8
on checkerboard. Selection shrank the network in an encoding that could not have
been prevented from growing it.

Both algorithms inherit the same complexity penalty from the reference fitness,
`1 + 0.03·√connections`. v5 tests whether that penalty, the mutation rate, or
neither is what keeps these topologies small, and whether any of it costs
accuracy.

## Design

Eight conditions, three geometries, thirty replicates: **720 runs across 90
cells.** A cell is one (task, replicate) and is the unit of sharding, because
the fixed yardstick needs the budget the reference actually spent in that cell.

| Condition | What it changes against the reference |
|---|---|
| `neat_reference` | nothing. `p_add_node` 0.2, `p_add_connection` 0.5, penalty on, 5 species, crossover on, population 100. The anchor. |
| `neat_complexify` | `p_add_node` 0.2 → **0.5**, `p_add_connection` 0.5 → **0.8** |
| `neat_no_penalty` | the complexity penalty **off** |
| `neat_complexify_no_penalty` | both of the above — a planned interaction, declared, not discovered |
| `neat_no_speciation` | `n_species` 5 → **1** |
| `neat_no_crossover` | mutation only |
| `neat_deep_narrow` | population 100 → **25**, generations ×4, candidate budget held. **Compound:** see below |
| `fixed_mixed_matched` | the yardstick: a fixed 32×32 heterogeneous-operator network at the reference's realized gradient budget |

### Three geometries, not five

v4 ran five and learned that XOR and circles saturate above 0.97 for every
condition, contributing nothing except to drag two hypotheses below their task
thresholds. v5 keeps spirals, checkerboard and the 3-arm spiral — the three on
which v3 and v4 separated conditions at all — and says so in the contract rather
than discovering it again afterwards.

### `neat_deep_narrow` is a compound arm, and is declared as one

Holding the candidate budget while quartering the population necessarily
quadruples the generations, and it also shrinks each species: with
`n_species = 5`, a population of 100 gives about twenty individuals per species
and a population of 25 gives about five. Species-level competition, the
extinction draw and the per-species quota all behave differently at that size.

The two cannot be separated — holding species size constant would mean changing
`n_species` with the population, which is a third change. So this arm answers
"does trading width for depth help, at a fixed evaluation budget" and **does
not** isolate depth from species granularity. Any effect it shows is attributed
to the trade as a whole, and the preregistered hypotheses do not name it as
evidence about depth alone.

### Everything else is held

The genome encoding, the structural operators themselves (`add_node`,
`add_connection`, `mutate_weights`), the compatibility metric, the crossover
implementation, the inner learner, the fitness function, settled propagation,
Lamarckian inheritance, the selection operator (v3 Block C established that the
selection *level* moves nothing here, so v5 does not re-ask that), and the
candidate budget of 2100 evaluations. Only the five named factors move.

The frozen primitives are imported unchanged from v2. v5 reimplements only the
mutation wrapper and the reproduction loop, because `P_ADD_NODE` and
`P_ADD_CONNECTION` are module constants in frozen code and cannot be varied
without new code. A gate asserts that at the reference rates v5's `mutate`
produces byte-identical genomes to the frozen one.

### Fresh test data, again

Dataset seeds `70001 + 13i`, search seeds `80001 + 13i`, with every seed v1–v4
spent added to the burned set. v5's sealed test is therefore drawn from splits no
earlier release has opened, and **v5 is not paired with any earlier release** —
no v5 number may be compared with a v3 or v4 number by a statistical test.

## Analysis, fixed in advance

Two primary outcomes, both paired within (task, replicate):

1. **Sealed-test accuracy** — as in v3 and v4.
2. **Champion size**, measured as *causally active* hidden nodes rather than
   represented ones, because this reconstruction has repeatedly found represented
   structure that never reaches the output.

One pre-declared family: every condition against `neat_reference`, on every
geometry — **21 tests**, Holm-corrected within the family. Wilcoxon signed-rank
with a 10,000-sample bootstrap interval on the median paired difference.
Significance means Holm-corrected *p* < 0.05.

## Hypotheses

**v5-H1 — the penalty is what suppresses complexification.** Removing it grows
champions significantly on ≥2 of 3 geometries.

**v5-H2 — the mutation rate is a binding constraint.** Raising `p_add_node` to
0.5 grows champions significantly on ≥2 of 3.

**v5-H3 — the two act together.** Removing both grows champions significantly on
≥2 of 3.

**v5-H4 — bigger topologies buy accuracy.** Some complexification arm beats the
reference on accuracy in ≥2 of the 9 arm-task cells. *This is the hypothesis the
study exists to test.* If it fails while H1–H3 hold, then NEAT's topologies are
small because the fitness asks them to be, and growing them does not help — which
would say that the search space, not the search, is where the performance is.

**v5-H5 — speciation is load-bearing.** Removing it loses significantly on ≥2 of
3.

**v5-H6 — crossover is load-bearing.** Removing it loses significantly on ≥2 of
3.

**v5-H7 — no NEAT variant beats a budget-matched fixed network.** The best
variant beats `fixed_mixed_matched` on 0 of 3 geometries. Stated in the direction
we expect from v3 and v4; a single geometry where a NEAT variant wins falsifies
it, and that would be the most interesting outcome in the release.

As in v3 and v4, no hypothesis is a success criterion. Each may hold or fail;
what is forbidden is choosing which to report afterwards.

## Firewalls

- **Sealed test.** Touched once, by `make v5-finaltest`, which refuses an
  incomplete release, one already marked `test_evaluated`, or one whose v5, v4,
  v3 or v2 fingerprint has moved.
- **Frozen code.** v5 fingerprints `conditions.py`, `protocol.py`, `search.py`
  and records the released v4 (`cd9d0c1f…`), v3 (`8438c9e8…`) and v2
  (`cfdf1fa3…`) sets in every run.
- **Pilots.** Seeds 9003 and 19003 only, burned before the freeze.

## A known limitation, declared here

`analysis.py` decides how these hypotheses are *scored* and is not covered by any
fingerprint — the same gap v4's release README records. It is committed in the
freeze commit, before any run record exists, and that is checkable with
`git ls-tree`; but it rests on git history rather than on a hash. A separate
scoring fingerprint is the fix and is deliberately **not** being introduced
mid-protocol, because changing the firewall between the freeze and the run is
exactly the kind of move these documents exist to prevent.

## Freeze record

| | |
|---|---|
| v5 science modules | `conditions.py`, `protocol.py`, `search.py` |
| v4 fingerprint (released) | `cd9d0c1f60b44c8d…` |
| v3 fingerprint (released) | `8438c9e89c7c72a3…` |
| v2 fingerprint (frozen) | `cfdf1fa3198adc0e…` |
| Planned runs | 720 across 90 cells |
| Replicates | 30, dataset seeds `70001 + 13i`, search seeds `80001 + 13i` |
| Freeze commit | `6f01738bbc6bfb4eef613415eba08917114d4b3f` |
| Tag | `v5-freeze` |

> **The freeze tags exist only locally.** This session's GitHub proxy refuses tag pushes (HTTP 403), so the commit hash above is the resolvable record; the tag is a convenience that a maintainer with push rights can add with `git tag v5-freeze 6f01738bbc6b && git push origin v5-freeze`.

