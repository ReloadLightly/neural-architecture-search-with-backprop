# Protocol v6 — preregistration

**Status: frozen before compute.** Written, gated and committed before any v6
run record existed. The freeze record is at the end of this file.

## The question

Every release so far compared the search against its controls at **one** budget:
2100 candidate evaluations, which is what a population of 100 run for twenty
generations costs. That number is inherited from the published description of
Backprop-NEAT and has never been varied.

Three releases then reported, in three different ways, that the search does not
beat a fixed network given the same compute — v3 block A, v4 H2–H5, v5 H7 — and
v4 added that Cartesian Genetic Programming's selection adds little over its own
candidate-matched null. Those are the headline findings of this project.

**They are statements about one point on a curve.** The obvious reading of them,
"architecture search does not pay", is not licensed by a single budget, and the
budget in question is small. Three things could be true instead:

1. the search is indistinguishable from random sampling at 2100 candidates but
   separates from it with room to work;
2. the fixed network wins at 2100 and stops winning once the search is given
   room;
3. the opposite of both — more search on a validation split is also more
   opportunity to overfit it, so the sealed-test gap could *widen* in the fixed
   network's favour as the budget grows.

v6 turns the budget into the factor so that the project's own headline can be
stated with a range attached, or withdrawn.

## Design

Three arms at every rung of a budget ladder, three geometries, thirty
replicates. **1,080 confirmatory runs across 90 cells**, plus a declared
extension rung of 270 more. A cell is one (geometry, replicate) and is the unit
of sharding, because each fixed arm needs the gradient budget the search arm
spent at the same rung in that same cell.

### The arms

| Arm | What it is | Where it comes from |
|---|---|---|
| `search` | NEAT at the reference configuration, run to this rung's generation count | `bpneat.v5.search`, imported unchanged. At the reference rung it is v5's `neat_reference` exactly. |
| `null` | the same candidate count **sampled** instead of searched: random architectures under the same inner learner, selection removed | v3's `random_search_matched`, transcribed into v6 so that v6 owns every line its fingerprint covers. A gate asserts the two produce the identical champion from identical inputs. |
| `fixed` | a fixed 32×32 mixed-operator network at the search arm's **realised gradient budget** in the same cell, 60 restarts | v3's `matched_multistart`; v5's `fixed_mixed_matched`, parameterised by rung. |

Nothing about *what* is compared is new. Only how much compute each side is
given.

### The ladder

Population is held at 100 and the budget is moved by generations alone, so a
budget change is one change and not a width/depth trade as well — v5's
`neat_deep_narrow` already asked that question and v6 does not re-ask it.

| label | generations | candidates | × reference | part of |
|---|---|---|---|---|
| `b500` | 4 | 500 | 0.24 | confirmatory |
| `b1000` | 9 | 1,000 | 0.48 | confirmatory |
| `b2100` | 20 | 2,100 | 1.00 | confirmatory |
| `b6300` | 62 | 6,300 | 3.00 | confirmatory |
| `b16800` | 167 | 16,800 | 8.00 | **extension** |

`b2100` is the budget v2, v3, v4 and v5 all ran at, so the third rung is the
point those releases already measured — reached again on fresh splits with fresh
seeds, which makes it a replication as well as a rung.

### Four confirmatory rungs and one declared extension

The search's cost is superlinear in its generation count, because the networks
grow and each later candidate costs more than an earlier one. Measured on this
machine before the freeze, with the pilot seeds, on spirals:

| rung | `search` wall time | gradient steps | steps per candidate |
|---|---|---|---|
| `b500` | 2.8 s | 24,680 | 49.4 |
| `b1000` | 8.0 s | 56,860 | 56.9 |
| `b2100` | 27.9 s | 135,880 | 64.7 |
| `b6300` | 269.2 s | 518,540 | 82.3 |

That is close to quadratic in the candidate count — the search spends more
gradient steps on each later candidate as well as evaluating more of them — which
puts `b16800` at roughly half an hour per run and the top rung alone at about
three times the cost of the whole confirmatory ladder. The other two arms are
close to linear in their budget, at 0.0345 s per sampled candidate for the `null` arm and 0.3718 ms per gradient step for the `fixed` arm. Rather than discover that halfway through and
quietly drop it, the contract names the four rungs the hypotheses are scored on
and names `b16800` separately. The extension is run **only** if the confirmatory
ladder finishes, **only** before the sealed test, and is reported as a
supplement. **No hypothesis is scored on it.** If it is incomplete when the
sealed test is taken it is discarded rather than reported partially, because a
sealed test that covered some of its replicates could never be completed
afterwards — the split is spent. The firewall enforces this: `final-test` refuses
a partially-present extension rung.

The confirmatory ladder still spans 12.6× in candidates and about a decade in
gradient steps, which is what the slope hypotheses need.

### The ladder is nested by construction, and that has a consequence

Within one cell every `search` arm is given the same search seed, and the
search's random stream does not depend on how many generations it will run for.
So the champion of `search_b500` **is** the champion of `search_b16800` as it
stood at generation four. The same holds for the `null` arm, whose running best
over a stream of sampled architectures is a prefix of the longer stream. A gate
asserts both.

This is deliberate: it makes the budget contrast paired on the strongest
possible terms — the arms differ only in where they stopped — and it removes
between-rung seed noise from every slope.

It also has a consequence that has to be stated in advance rather than
discovered. The champion is selected on *validation* fitness, so validation
fitness is non-decreasing in budget **by construction** and carries no evidence
whatsoever about whether more search helps. **Every hypothesis below is scored on
sealed-test accuracy**, which selection never saw and which is free to fall as
the budget grows. Validation accuracy appears in `summary.csv` as a diagnostic
and in no decision rule.

### Everything else is held

The genome encoding, the structural operators and their rates, the compatibility
metric, the complexity penalty, the speciation, the crossover, the selection
operator, the operator set, the inner learner (600 cycles, batch 10), settled
propagation, Lamarckian inheritance, the population of 100, the restart count of
60, the random-architecture sampler's range, and the three geometries. Only the
generation count moves, and the two quantities derived from it: the candidate
count the null is matched to, and the gradient budget the fixed arm is matched
to.

### Fresh test data, again

Dataset seeds `90001 + 13i`, search seeds `100001 + 13i`, with every seed v1–v5
spent added to the burned set. v6's sealed test is therefore drawn from splits no
earlier release has opened, and **v6 is not paired with any earlier release** —
no v6 number may be compared with a v3, v4 or v5 number by a statistical test.
The reference rung is a *distributional* replication of v5's reference cell, not
a paired one.

## Analysis, fixed in advance

Primary outcome: **sealed-test accuracy of the validation-selected champion**,
paired within (geometry, rung, replicate).

### Contrasts

Wilcoxon signed-rank on the paired difference, with a 10,000-sample bootstrap
interval on the median (seed `20261007`). Significance means Holm-corrected
*p* < 0.05 **within the declared family**, at the family's declared size — so a
cell that is missing does not buy the surviving cells more power.

### Slopes

A slope is computed **per replicate** and then tested across replicates, rather
than fitted once to the group medians. The rungs within a replicate are not
independent — the ladder is nested — and a per-replicate slope keeps that
dependence inside the unit instead of pretending four rungs are four
independent points. The resulting thirty slopes *are* independent of each other,
because the replicates are.

Each slope is an ordinary least-squares fit of accuracy on the base-10 logarithm
of a compute axis, and each term uses the axis on which its arms are **matched**:

| term | what it measures | axis |
|---|---|---|
| `search` | accuracy per decade of gradient compute | log₁₀ gradient steps |
| `null` | same, for the unselected control | log₁₀ gradient steps |
| `fixed` | same, for the fixed network | log₁₀ gradient steps |
| `edge_vs_null` | how the search's margin over its candidate-matched null moves | log₁₀ **candidates** |
| `edge_vs_fixed` | how the search's margin over the budget-matched fixed net moves | log₁₀ gradient steps |

Gradient steps for the single-arm terms because it is the one axis on which all
three arms are commensurable and the axis the fixed arm is matched on; candidates
for `edge_vs_null` because candidates is what those two arms are matched on.

### Equivalence

At the reference rung only, (search − null) is tested for equivalence within
δ = **0.02 accuracy** by two one-sided signed-rank tests: the paired median is
above −δ and below +δ. Equivalence is declared when both reject. Holm is applied
to the larger of the two *p*-values, which makes declaring equivalence *harder*,
the conservative direction for a claim that two arms are the same.

Two points of accuracy is below the smallest difference any earlier release
treated as meaningful, and about a quarter of the spread between the best and
worst arm at the reference budget in v5.

### The four declared families

| family | cells | what it is |
|---|---|---|
| `search_vs_null` | 12 | search − null, at each of 4 rungs × 3 geometries |
| `search_vs_fixed` | 12 | search − fixed, at each of 4 rungs × 3 geometries |
| `scaling_slopes` | 15 | 5 slope terms × 3 geometries |
| `equivalence_at_reference` | 3 | TOST at the reference rung, per geometry |
| `size_slopes` | 3 | the search champion's size against budget, per geometry |

Five families rather than one, because they answer five different questions and a
single family of 45 would penalise each of them for the others' size. Every
family is enumerated in `protocol.py` before any run exists, and a gate asserts
the sizes and that the extension rung appears in none of them.

A contrast on any metric other than sealed-test accuracy — champion size against
the controls, for instance — is **descriptive**. Those rows carry
`scored = False` in `arm-contrasts.csv` and get no corrected *p*-value, so they
cannot be read as a test this contract declared.

## Hypotheses

**v6-H1 — the search's edge over its candidate-matched null grows with budget.**
The per-replicate slope of (search − null) against log₁₀ candidates is
significantly positive on ≥2 of 3 geometries. *This is the hypothesis that would
rescue the search:* if its advantage is real but needs room, it should appear
here.

**v6-H2 — at the reference budget the search and its null are equivalent.** Both
one-sided tests reject at δ = 0.02 on ≥2 of 3 geometries. Stated as equivalence,
not as a failure to reject: v3 and v4 found no significant difference between a
search and its candidate-matched null, and "we could not tell them apart" and
"they are the same to within two points of accuracy" are different claims. This
one tries to earn the second.

**v6-H3 — at some budget the search beats the budget-matched fixed network.**
(search − fixed) is significantly positive at ≥1 rung, on ≥2 of 3 geometries.
Stated in the direction three releases say is *false*; a budget at which it
becomes true is the most interesting outcome available here.

**v6-H4 — the search buys less per decade of compute than the fixed network
does.** The per-replicate slope of (search − fixed) against log₁₀ gradient steps
is significantly negative on ≥2 of 3 geometries. If this holds, the fixed
network's lead is not a constant to be out-run but a gap that widens with
compute.

**v6-H5 — sampling more architectures helps even with selection removed.** The
null's own slope against log₁₀ gradient steps is significantly positive on ≥2 of
3 geometries. If this holds while H1 fails, then the budget buys accuracy through
*sampling* more architectures rather than through *selecting* among them, which
is a sharper statement than anything in v3 or v4.

**v6-H6 — the reference-budget verdict is not an artefact of the budget.** No
geometry has one rung where the search significantly wins against the fixed
network and another where it significantly loses. A flip would mean this
project's headline conclusion is budget-specific and has to be restated with the
budget attached, in v3's, v4's and v5's READMEs as well as v6's. The rule
requires *both* sides of the flip to be significant, so noise near zero cannot
manufacture one.

**v6-H7 — the champion's size is set by the budget, not only by the fitness.**
The per-replicate slope of the search champion's causally active hidden nodes
against log₁₀ candidates is significantly positive on ≥2 of 3 geometries.

*This hypothesis was suggested by the cost pilot above, and that is stated here
rather than hidden.* On the burned pilot seed, on spirals, the champion grew from
1 causally active hidden unit at 500 candidates to 12 at 6,300, while validation accuracy rose from
0.700 to 0.920. One seed is not
evidence and the pilot scores nothing. But if that pattern survives thirty
replicates on three geometries, then v5's finding — champions reach about four
active units because the fitness asks them to — is a statement about the
*budget* as much as about the fitness, and v5's conclusion has to be restated
with the budget attached. The pilot record is committed at
`results/backprop-neat-v6/pilot/cost.json` and every number in it is bound by a
gate.

As in v3, v4 and v5, no hypothesis is a success criterion. Each may hold or fail;
what is forbidden is choosing which to report afterwards.

## Firewalls

- **Sealed test.** Touched once, by `make v6-finaltest`, which refuses an
  incomplete confirmatory suite, one already marked `test_evaluated`, one
  carrying unplanned runs, one with a partially-present extension rung, or one
  whose v6, v5, v4, v3 or v2 fingerprint has moved.
- **Frozen code.** v6 fingerprints `conditions.py` and `protocol.py` — it has no
  searcher of its own, which is the point of the design — and records the
  released v5 (`33f2c177…`), v4 (`cd9d0c1f…`), v3 (`8438c9e8…`) and v2
  (`cfdf1fa3…`) sets in every run.
- **Pilots.** Seeds 9004 and 19004 only, burned before the freeze. Every number
  in the cost table above comes from those seeds.

## Declared limitations

**The null searches a different space from the one NEAT explores.** The random
architecture sampler draws 2–15 hidden nodes and up to 11 extra connections
immediately, while NEAT starts minimal and grows slowly, reaching about four
causally active hidden units. So the null is not a sample from the distribution
NEAT's population occupies; it is a sample from the space NEAT's encoding can
represent. v3 froze that choice and v6 keeps it unchanged, so that v6's null is
comparable to v3's. Designing a null matched to NEAT's *realised* size
distribution is a different study, and this one does not claim to be it.

**`analysis.py` is in the freeze commit but is not fingerprinted.** The same gap
v4's and v5's release notes record. It decides how these hypotheses are *scored*,
it is committed before any run record exists, and that is checkable with
`git ls-tree` — but it rests on git history rather than on a hash. A separate
scoring fingerprint is the fix and is deliberately **not** being introduced
mid-protocol, because changing the firewall between the freeze and the run is
exactly the kind of move these documents exist to prevent.

**`figures.py` is not in the freeze commit,** and is named here so that its
absence is not mistaken for an omission. It draws the tables that
`analysis.py` has already computed and can change no verdict; `release.py`
imports it lazily for that reason.

## Cost, measured before the freeze

From the pilot timings above and the realised per-condition costs in the
committed v3 and v5 releases:

| | per replicate (3 geometries) | × 30 replicates |
|---|---|---|
| `search`, four rungs | ≈ 690 s | 5.8 core-hours |
| `null`, four rungs | ≈ 1,030 s | 8.6 core-hours |
| `fixed`, four rungs | ≈ 850 s | 7.1 core-hours |
| **confirmatory total** | **≈ 2,570 s** | **≈ 21 core-hours** |

About **5–6 hours of wall time** on this four-core machine with `SHARDS=4`. The
extension rung would add roughly 65 core-hours — about 17 wall hours — which is
why it is declared separately rather than assumed.

## Freeze record

| | |
|---|---|
| v6 science modules | `conditions.py`, `protocol.py` |
| v6 fingerprint | `f1205038f33ad77e9706d8a51fe04cd32de268ea19f060f03c891a536c493cc0` |
| v5 fingerprint (released) | `33f2c177be4b5335…` |
| v4 fingerprint (released) | `cd9d0c1f60b44c8d…` |
| v3 fingerprint (released) | `8438c9e89c7c72a3…` |
| v2 fingerprint (frozen) | `cfdf1fa3198adc0e…` |
| Confirmatory runs | 1,080 across 90 cells |
| Declared extension | 270 runs at `b16800`, scoring nothing |
| Replicates | 30, dataset seeds `90001 + 13i`, search seeds `100001 + 13i` |
| Freeze commit | `FREEZE_COMMIT` |

> **The freeze tags exist only locally.** This session's GitHub proxy refuses tag
> pushes (HTTP 403), so the commit hash above is the resolvable record; the tag is
> a convenience that a maintainer with push rights can add with
> `git tag v6-freeze FREEZE_COMMIT && git push origin v6-freeze`.
