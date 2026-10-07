# Audit: the fast evaluator and node-id order

**Status: a latent defect in `src/bpneat/v3/dense.py`, found while generalising
it, reached by no committed result. No number in this repository changes.**

## What was found

`bpneat.v3.dense` is an optimisation: it evaluates an acyclic genome with one
matrix product per topological level instead of one node at a time, and its
module docstring says it is "exact to the genome path" with equivalence asserted
to 1e-10. That claim is true of every graph v3 and v4 ever handed it, and false
in general.

The genome path recomputes every touched node in **node-id order** within a
tick, reading values in place. It therefore reaches the topological fixed point
the dense evaluator computes only when the node ids are themselves a topological
order — with one exception the frozen code does account for: the output holds
the lowest id after the inputs, is always recomputed from the previous tick's
hidden values, and `settle_ticks` adds exactly one tick for it.

An active edge running from a higher id *into a hidden node* is a second
inversion, and nothing compensates for it. The two paths then settle to
different values.

**Measured, at two inputs, with the frozen random-architecture sampler and v3's
own dense evaluator:** of 300 sampled graphs, 166 contain a cycle and are
refused; of the remaining 134, **29 disagree with the genome path**, and the
worst disagreement is **0.47 in a logit**. That is not a tolerance question.

## Why no result is affected

`v3.dense` is reached from exactly two places, and both hand it graphs whose ids
are a topological order:

| caller | what it plans | id-ordered? |
|---|---|---|
| `v3.learners.plain_train`, via `matched_multistart` and used by v3, v4, v5 and v6's fixed arms | `make_mlp` / `mixed_mlp` — built layer by layer, each layer's nodes appended after the previous layer's | yes, by construction |
| `v4.search` and `v4.learners`, for every CGP candidate | a decoded CGP genotype; CGP nodes may only read earlier nodes, which `v4.cgp` states and relies on | yes, by construction |

Checked rather than argued: **400 decoded CGP genotypes were decoded and
examined. Zero carried a backwards edge into a hidden node, zero disagreed with
the genome path, and the worst gap was 4.3e-14** — float noise.

The random-architecture controls — v3's `random_search_matched` and v6's `null`
arm, the graphs that *do* wire backwards — never touch the dense path at all.
They are trained by `bpneat.learn.train`, which uses the genome path only.

So the defect is real, and it is latent: nothing asserted that the dense path
was only ever given graphs it is exact on, so a future protocol that planned a
sampled graph would have got silently different numbers.

## What was done

**The frozen module was not edited.** v3 is released and fingerprinted
(`8438c9e8…`), and editing it would invalidate four releases to fix a bug none
of them has.

1. **`bpneat.nd.dense` refuses what it cannot reproduce.** The general version
   raises `NotDenseable` on any active edge that runs from a higher id into a
   hidden node, so such a graph falls back to the genome path. Three gates in
   `tests/test_nd_dense.py` cover it: that forward-wired sampled graphs agree on
   both paths, that backwards-wired ones are refused, and that the disagreement
   being guarded against is still measurable — if the two paths ever start
   agreeing, the gate fails and tells you to delete the guard rather than leave
   it as folklore.

2. **A gate holds the released protocols to the safe side.**
   `tests/test_dense_ordering.py` asserts that every architecture v3, v4, v5 and
   v6 hand to `dense.plan` — the fixed networks and the decoded CGP genomes — is
   id-ordered. If a future change makes one of them wire backwards, CI fails
   before a release is built on it.

3. **This note.** The claim "exact to the genome path" in `v3/dense.py`'s
   docstring is too strong, and since that file cannot be edited, the correction
   lives here and is linked from the audit index.
