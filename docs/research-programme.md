# Where this is going

Six protocols in, this project has one finding and one ceiling.

**The finding.** On three 2-D geometries, at 2,100 candidate evaluations, the
search component of Backprop-NEAT does not beat sampling the same space at
random, and does not beat a fixed 32×32 network given the same gradient budget.
v4 found the same for an independently-encoded second search algorithm. v5 found
that none of NEAT's five named mechanisms rescues it. v6, running now, asks
whether any of that is a statement about the budget.

**The ceiling.** Every one of those results is about *two-dimensional synthetic
geometries*, and two of the three remaining open objections are about the
controls rather than the conclusion. This document names the three things that
would have to be true for the finding to be worth believing outside this
repository, and the protocol that tests each. It is a plan, not a result; nothing
in it may be cited.

---

## v6 — does the search pay at any budget? *(running)*

Frozen at `aa7e9f36`, preregistered in
[`v6-preregistration.md`](v6-preregistration.md). Five rungs from 500 to 16,800
candidates, three arms at each. The single-budget objection, answered.

---

## v7 — is the null fair, and is NEAT's machinery worth anything at all?

**The objection v7 exists to answer.** The candidate-matched null in v3 and v6
samples architectures with 2–15 hidden nodes drawn uniformly, while NEAT starts
minimal and reaches about four causally active units. So "search ≈ random
sampling" compares NEAT against a sampler with a *different* size distribution,
and a reader is entitled to ask whether the result is about search or about that
mismatch. v6's preregistration declares this limitation; v7 removes it.

Planned arms, all at the reference budget and at 3× it:

| Arm | What it removes | What it would show |
|---|---|---|
| `neat_reference` | nothing | the anchor |
| `null_uniform` | selection; v3's and v6's sampler, kept for comparability | the existing result, re-measured |
| `null_size_matched` | selection **only** — architecture sizes are drawn from the empirical distribution of v6's search champions at the same rung, read from the committed v6 release | whether the existing null's advantage was its size distribution |
| `null_generation_zero` | all reproduction — NEAT's own initial population, resampled to the matched candidate count | the purest "no search" control, inside NEAT's own distribution |
| `hillclimb_1plus1` | the population, speciation and crossover — a 1+1 hill climber under NEAT's own mutation operators | whether anything NEAT adds beats the simplest search that could work |

`null_size_matched` is the arm that could overturn the project's headline: if the
search beats a null matched on size, then selection *is* doing work and the
earlier nulls were straw men. `hillclimb_1plus1` is the arm that could sharpen
it: if a 1+1 hill climber matches NEAT, then the population, the speciation and
the crossover are decoration, which is a stronger statement than "NEAT does not
beat random".

v7 cannot be preregistered until v6 is released, because one of its arms is
defined by a distribution measured from v6. The *procedure* for reading that
distribution is what the contract will fix, not the numbers.

---

## v8 — does any of this survive outside two dimensions?

**The objection v8 exists to answer, and the repository's own stated limit.**
Every number here comes from five 2-D synthetic geometries. The README's
Limitations section says higher-dimensional real datasets were "skipped rather
than attempted badly", because the frozen genome fixes two input nodes in a
module no protocol may edit. That is the honest reason, and it is also the
ceiling on the whole project: a claim about neural architecture search that has
only ever been tested on spirals is a claim about spirals.

v8 needs new code, and the new code has to earn the frozen core's credibility
rather than ask for it:

1. **A generalised encoding** — `d` inputs and `k` outputs, with the frozen
   propagation scheme, operators, gradients and fitness unchanged in substance.
2. **An equivalence proof, as a gate.** At `d = 2, k = 1` the generalised
   encoding must produce *byte-identical* genomes, forward values, gradients and
   champions to the frozen one, from identical seeds. Not "agrees to 1e-10" —
   identical, because the frozen path and the general path are then the same
   computation and every earlier result transfers. v3 did exactly this for its
   dense learner against the frozen genome path, and the technique is the same.
3. **Real tabular data**, with the same control battery: search, candidate-matched
   null, and a budget-matched fixed network. No new conclusions are available
   until the controls come too.

Until (2) is green, (3) is not attempted. If (2) cannot be made green, that is
reported as a failure of the generalisation and v8 stops; it is not worked
around by relaxing the gate.

### What the data turned out to be like, measured before anything was designed

(1) and (2) are done — `src/bpneat/nd/`, 29 gates. (3) now has four real
datasets vendored into `data/tabular/`, and measuring them changed the design.

| dataset | features | classes | majority class | **linear model, validation** |
|---|---|---|---|---|
| `iris` | 4 | 3 | 0.333 | **0.967** |
| `wine` | 13 | 3 | 0.389 | **0.972** |
| `breast_cancer` | 30 | 2 | 0.628 | **0.982** |
| `digits` | 64 | 10 | 0.103 | **0.946** |

The "linear model" there is the *seed genome* — bias and every input wired
straight to the outputs, no hidden units at all — trained by the frozen inner
learner, best of six restarts, on three splits. It is within two to five points
of a ceiling on every one of them. A first look at the search agrees from the
other direction: on `iris` and `breast_cancer` its champions come back with
**zero causally active hidden units**.

Two consequences, both of which belong in the design rather than in a later
discussion section:

**The linear model has to be an arm.** Not a footnote — an arm, at the same
matched gradient budget as everything else. On data like this the sharpest
version of this project's question is not "does the search beat a fixed
network" but *does architecture search beat having no architecture*, and that
has to be measured rather than inferred from a champion's node count.

**`digits` is the only one with real headroom, and it is the expensive one.**
Measured on this machine, one search at the reference budget of 2,100
candidates costs 3.8 minutes on `iris`, 5.3 on `wine`, 1.6 on `breast_cancer`
and **58 on `digits`**. So v8 will follow v6's shape: three confirmatory
datasets and `digits` as a declared extension, run only if the confirmatory
suite completes, scoring no hypothesis. Saying that in advance is the point —
discovering it halfway through and dropping the expensive one is the move these
documents exist to prevent.

---

## Three things that are not protocols

**A scoring fingerprint.** v4's, v5's and v6's release notes all record the same
gap: `analysis.py` decides how hypotheses are scored and is not covered by any
fingerprint. It is committed in each freeze commit before any run record exists,
which is checkable with `git ls-tree`, but it rests on git history rather than a
hash. The fix is a second fingerprint over the scoring modules, recorded at
freeze time. It is deliberately not introduced mid-protocol — changing a firewall
between a freeze and a run is the move these documents exist to prevent — so it
lands with v7.

**The inner learner.** Every result in this repository is measured under one
inner learner: 600 cycles of RMSProp with Ha's rollback check, batch 10. A
different trainer could change which architecture wins. This is a real
confounder and it is not yet tested; it belongs in v8 as a robustness arm rather
than as a protocol of its own.

**The repository's own metadata.** The GitHub description still advertises the
withdrawn v2 claim, and the default branch is a working branch rather than
`main`. Both need repository-settings permissions this session does not have
(the proxy returns `403 Repository settings writes are not permitted`), so they
are listed here as outstanding rather than silently left.
