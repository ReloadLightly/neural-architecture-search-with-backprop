# Protocol v8 — preregistration

**Status: frozen before compute.** Written, gated and committed before any v8
run record existed. The freeze record is at the end of this file.

## The question

Every number this repository has produced is about **two-dimensional synthetic
geometries**. The README says why, and says it as a limitation: the frozen
genome fixes two input nodes in a module no protocol may edit, so
higher-dimensional real datasets were "skipped rather than attempted badly".

That is the ceiling on the whole project. A claim about neural architecture
search tested only on spirals is a claim about spirals.
[`src/bpneat/nd/`](../src/bpneat/nd/) lifts it — a general encoding proved
**byte-identical** to the frozen one at two inputs and one output, so that
earlier results transfer rather than merely resemble — and v8 is the first
protocol to use it.

## What the data turned out to be like, measured before anything was designed

Four real tabular datasets are vendored into [`data/tabular/`](../data/tabular/)
as CSV, with checksums and a provenance chain back to UCI. Before writing a line
of this contract, each was measured against the simplest possible model: the
**seed genome** — bias and every input wired straight to the outputs, no hidden
units at all — trained by the frozen inner learner, best of six restarts, on
three splits.

| dataset | features | classes | majority class | **linear model, validation** |
|---|---|---|---|---|
| `iris` | 4 | 3 | 0.333 | **0.967** |
| `wine` | 13 | 3 | 0.399 | **0.972** |
| `breast_cancer` | 30 | 2 | 0.627 | **0.982** |
| `digits` | 64 | 10 | 0.102 | **0.946** |

A linear model is within two to five points of a ceiling on every one of them.
And the search agrees from the other direction: a first look at its champions on
`iris` and `breast_cancer` returned **zero causally active hidden units**.

So v8 does not ask whether the search beats a fixed network. It asks the sharper
question this data makes available, and the one the project's thesis actually
implies:

> **Does architecture search beat having no architecture?**

## Design

Four arms, three datasets, thirty replicates: **360 confirmatory runs across 90
cells**, plus a declared extension of 120 more. A cell is one (dataset,
replicate) and is the unit of sharding, because two of the four arms need the
gradient budget the search arm actually spent in that same cell.

### The arms

| Arm | What it is | Where it comes from |
|---|---|---|
| `search` | NEAT at v5's reference configuration, 2,100 candidate evaluations | `bpneat.nd.evolve`, which is v5's reproduction loop over a general layout |
| `null` | the same candidate count **sampled** instead of searched | v3's random-architecture sampler, byte-identical at two inputs, so a null here is the null v3 and v6 ran |
| `linear` | **no architecture at all** — the seed genome, at the search's realised gradient budget across 60 restarts | the encoding's own `logistic_genome` |
| `fixed` | a 32×32 mixed-operator network, same matched budget, same 60 restarts | v3's `mixed_mlp`, byte-identical at two inputs |

The budget is 2,100 candidates, which is what v2 through v6 all anchor on.
Whether *that budget* is the right one is v6's question and is not re-asked here.

### Why `linear` is an arm and not a footnote

Because on this data it is the thing to beat, and because "the champion had no
hidden units" is an observation about a genome while "the search did not beat a
model with no hidden units, paired, on compute" is a measurement. The arm is
given the same matched gradient budget and the same restart count as the fixed
network, so it is the fairest version of itself rather than a straw man.

It also makes a failure interpretable. If the fixed network cannot beat `linear`
either, then nothing about architecture helps on these datasets and the result
is a statement about the data rather than about the search. That is why it is
**v8-H4** below and not a line in a discussion section.

### Three datasets and one declared extension

`digits` is 64 features and ten classes — the widest and the hardest of the four,
and the only one with real headroom above linear. It is also the expensive one:
one search at the reference budget costs minutes on the other three and nearly
an hour on it. The contract therefore names the three the hypotheses are scored
on and names `digits` separately: it is run **only** if the confirmatory suite
finishes, **only** before the sealed test, and is reported as a supplement
scoring no hypothesis. The firewall enforces it — `final-test` refuses a
partially-present extension dataset.

### The split, and the resolution it buys

A replicate is a **different stratified partition of the same finite rows**, not
a fresh draw from a generator. Five tenths train, two validation, three sealed
test. Features are centred and scaled by the **training split's own** mean and
deviation; fitting that on all three would leak the sealed split through the
mean, and a gate checks that the held-out splits are *not* centred.

Three tenths to the sealed test rather than two is a decision about what can be
claimed, not a convention. A finite test split resolves accuracy only to one row
in its size:

| dataset | sealed test rows | finest difference it can express |
|---|---|---|
| `iris` | 45 | 0.022 |
| `wine` | 52 | 0.019 |
| `breast_cancer` | 172 | 0.006 |
| `digits` | 540 | 0.002 |

The equivalence margin below is 0.03 — between one and two rows of the smallest
sealed split. `validate()` refuses a margin finer than the smallest dataset can
resolve, and it **refused 0.03 at the previous two-tenths split**, which is why
the split changed. The margin was not chosen first and justified afterwards.

### Everything else is held

The inner learner (600 cycles of RMSProp with Ha's rollback, batch 10), the
complexity penalty, settled propagation, Lamarckian inheritance, the selection
operator, the operator set, the structural rates, the population of 100, the
speciation into 5, the restart count of 60, and the sampler's size range. Only
the data and the fourth arm are new.

## Analysis, fixed in advance

Primary outcome: **sealed-test accuracy of the validation-selected champion**,
paired within (dataset, replicate). The thirty replicates are independent of
each other because their splits are drawn independently; the pairing is within a
split.

Paired differences on a finite test split are heavily tied — they take values in
multiples of one row — so the signed-rank test is run with
`zero_method="zsplit"`, and the number of ties is reported in every contrast.

### The five declared families

| family | cells | what it is |
|---|---|---|
| `vs_linear` | 9 | each of search, null and fixed minus linear, on 3 datasets |
| `vs_null` | 3 | search − null |
| `vs_fixed` | 3 | search − fixed |
| `equivalence_with_linear` | 3 | TOST of search against linear |
| `champion_size` | 3 | does the search's champion carry a hidden unit at all |

Holm within each family, at its **declared** size, so a missing cell cannot buy
the survivors more power. Significance means Holm-corrected *p* < 0.05.
Bootstrap intervals on the median paired difference use 10,000 resamples, seed
`20261008`. A contrast on any metric other than the primary outcome is
descriptive, carries `scored = False`, and gets no corrected *p*-value.

### Equivalence

(search − linear) is tested for equivalence within δ = **0.03 accuracy** by two
one-sided signed-rank tests. Equivalence is declared when both reject. Holm is
applied to the larger of the two *p*-values, which makes declaring equivalence
*harder* — the conservative direction for a claim that two arms are the same.

### Champion size

Whether the search returns a linear model is a claim about a proportion, not
about the mean of a count that is zero most of the time. It is tested with an
exact binomial against one half: *more often than not, the champion carries no
causally active hidden unit*.

## Hypotheses

**v8-H1 — architecture search beats having no architecture.** search − linear is
significantly positive on ≥2 of 3 datasets. *This is the hypothesis the study
exists to test,* and it is stated in the direction this project's five earlier
releases say is false.

**v8-H2 — selecting among architectures beats sampling them.** search − null is
significantly positive on ≥2 of 3. The question v3, v4 and v6 asked on synthetic
geometries, asked again on real data.

**v8-H3 — the search beats a budget-matched fixed network.** search − fixed is
significantly positive on ≥2 of 3.

**v8-H4 — some architecture beats no architecture on these datasets.**
fixed − linear is significantly positive on ≥2 of 3. *The control for the whole
study.* If H1 fails and H4 fails too, then nothing about architecture helps here
and v8 has measured a property of the data; if H1 fails while H4 holds, then
architecture helps and the search is not the way to find it. Those are different
results and this hypothesis is what separates them.

**v8-H5 — the search and the linear model are equivalent.** Both one-sided tests
reject at δ = 0.03 on ≥2 of 3. Stated as equivalence rather than as a failure to
reject: "we could not tell them apart" and "they are the same to within one row
of the sealed test" are different claims, and this one tries to earn the second.

**v8-H6 — the search returns a linear model.** The champion carries no causally
active hidden unit more often than not, by an exact binomial, on ≥2 of 3. If
this holds alongside H5, the finding is not that the search *failed* to find a
useful architecture but that it *declined* to build one — which, under a fitness
with a complexity penalty and data a line already fits, is the algorithm working
as specified.

As in every earlier protocol, no hypothesis is a success criterion. Each may
hold or fail; what is forbidden is choosing which to report afterwards.

## Firewalls

- **Sealed test.** Touched once, by `make v8-finaltest`, which refuses an
  incomplete suite, one already marked `test_evaluated`, one carrying unplanned
  runs, one with a partially-present extension dataset, or one whose v8, data,
  nd, v5, v3 or v2 fingerprint has moved. A static gate asserts that no module
  in `src/bpneat/v8/` other than the firewall mentions `.test` **at all** — not
  even its length.
- **Frozen code.** v8 fingerprints `conditions.py` and `protocol.py`, and
  records the n-dimensional core, the committed datasets **by content**, and the
  released v5, v3 and v2 sets in every run. The data is hashed rather than the
  loader because a release built on a dataset that later changed would otherwise
  look sound.
- **Champion width.** A champion is rebuilt through the layout its own record
  carries. The frozen deserialiser returns a two-input graph, which would score
  silently and wrongly on a thirty-feature dataset.
- **Pilots.** Seeds 9005 and 19005 only, burned before the freeze.

## Declared limitations

**These are small, classic benchmarks.** Four datasets between 150 and 1,797
rows, all of them decades old and all of them nearly saturated by a linear
model. v8 can support a claim about *this class of data* — small, tabular,
mostly linearly separable — and not about modern machine learning. It is a step
off two-dimensional spirals, not an arrival at ImageNet.

**Near-saturation limits the power to detect a small real advantage.** With a
linear model at 0.97 there is little room for anything to win in, which is
exactly why the equivalence test is here and why its margin is pinned to one
test row rather than chosen for convenience. A failure of H1 under these
conditions is weaker evidence than a failure would be on data with headroom —
and that is what the `digits` extension is for.

**One inner learner.** Every number in this repository, v8 included, is measured
under 600 cycles of RMSProp with Ha's rollback. A different trainer could change
which architecture wins. This is a real confounder and it is not tested here.

**The null samples a different space from the one the search explores.** The
sampler draws 2–15 hidden nodes immediately while the search starts minimal.
v6's preregistration declares this limitation and v8 inherits it unchanged, so
that v8's null is comparable to v3's and v6's. Fixing it is v7's job.

**`analysis.py` is in the freeze commit but is not fingerprinted** — the same
gap v4, v5 and v6 record. It is committed before any run record exists, which is
checkable with `git ls-tree`, but it rests on git history rather than a hash.

**`figures.py` is not in the freeze commit,** and is named here so its absence
is not mistaken for an omission. It draws what `analysis.py` has already
computed and can change no verdict.

## Cost, measured before the freeze

COST_TABLE

## Freeze record

| | |
|---|---|
| v8 science modules | `conditions.py`, `protocol.py` |
| v8 fingerprint | `FINGERPRINT_V8` |
| data fingerprint (committed CSVs, by content) | `FINGERPRINT_DATA` |
| n-dimensional core | `FINGERPRINT_ND` |
| v5 fingerprint (released) | `33f2c177be4b5335…` |
| v3 fingerprint (released) | `8438c9e89c7c72a3…` |
| v2 fingerprint (frozen) | `cfdf1fa3198adc0e…` |
| Confirmatory runs | 360 across 90 cells |
| Declared extension | 120 runs on `digits`, scoring nothing |
| Replicates | 30, split seeds `110001 + 13i`, search seeds `120001 + 13i` |
| Freeze commit | `FREEZE_COMMIT` |

> **The freeze tags exist only locally.** This session's GitHub proxy refuses tag
> pushes (HTTP 403), so the commit hash above is the resolvable record; the tag
> is a convenience that a maintainer with push rights can add with
> `git tag v8-freeze FREEZE_COMMIT && git push origin v8-freeze`.
