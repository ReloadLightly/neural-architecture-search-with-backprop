# Neural architecture search with backprop

A NumPy reconstruction of David Ha's
[Backprop-NEAT](https://blog.otoro.net/2016/05/07/backprop-neat/) (2016) — NEAT
topology search with backpropagation training every candidate — run as a frozen,
preregistered experiment, and then audited until it broke.

[![CI](https://github.com/ReloadLightly/neural-architecture-search-with-backprop/actions/workflows/ci.yml/badge.svg)](https://github.com/ReloadLightly/neural-architecture-search-with-backprop/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> ### ⚠️ Read the errata first
>
> An external audit (October 2026) reproduced every v2 number and showed that
> the headline comparison is **confounded**: v2's fixed-MLP control received
> 2,506 gradient steps on spirals against Backprop-NEAT's 132,138, because Ha's
> rollback rule stops a 69-node network after ~42 updates. Two of the five v2
> claims are withdrawn and two are narrowed.
> **[`docs/v2-errata.md`](docs/v2-errata.md)** · reproduction:
> **[`docs/audit-2026-10.md`](docs/audit-2026-10.md)**
>
> The runs themselves are correct and regenerate byte-identically. What was
> wrong is what was claimed about them. Protocol **v3**
> ([`docs/v3-preregistration.md`](docs/v3-preregistration.md)) turns the
> confounds into the object of study: *when does the evaluator, rather than the
> algorithm, decide the conclusion?*

## What this is

A NumPy reconstruction of David Ha's
[Backprop-NEAT](https://blog.otoro.net/2016/05/07/backprop-neat/) (2016),
audited against the published source
([`hardmaru/backprop-neat-js`](https://github.com/hardmaru/backprop-neat-js))
rather than rebuilt from memory, and run as a frozen, preregistered experiment:
six conditions that each remove one mechanism, ten paired replicates per task,
two propagation tracks that are never pooled, and a sealed test evaluated
exactly once.

| Track | Propagation | Fitness | Purpose |
|---|---|---|---|
| A | `ha2016` — Ha's exact break rule | training loss | historical reconstruction |
| B | `settled` — topology-determined fixed point | validation loss | mechanism comparison |

This repository is the second of three experiments in a line on evolutionary
search for open-world decision problems; the motivating argument, which is a
policy-research argument and **not** evidence produced here, lives in
[`docs/writeup.md`](docs/writeup.md). Nothing in this README depends on it.

## Status

Protocol **v3 is running** — 1,920 runs over 150 cells, preregistered and
frozen at commit `a343c66` before any compute
([`docs/v3-preregistration.md`](docs/v3-preregistration.md)).

Protocol **v2 is released**: 360 runs, zero failures, sealed test evaluated
once — [`results/backprop-neat-v2/`](results/backprop-neat-v2/), with the full
claim ladder in the [release README](results/backprop-neat-v2/README.md).
Protocol **v1 is invalidated** for a selection-fidelity defect and retained for
audit ([`docs/v1-invalidation.md`](docs/v1-invalidation.md)); no v1 number may
be cited.

## Headline: sealed-test accuracy, track B

Mean over ten paired replicates. **Read the budget column before the accuracy
columns** — it is the whole of erratum E1.

| Task | Backprop-NEAT | Homog. tanh | Evolution only | Random arch. | Fixed MLP | Logistic |
|---|---|---|---|---|---|---|
| XOR | **0.996** | 0.995 | 0.736 | 0.986 | 0.978 | 0.536 |
| Circles | **0.986** | 0.943 | 0.791 | 0.988 | 0.973 | 0.504 |
| Spirals | **0.787** | 0.745 | 0.639 | 0.713 | 0.637 | 0.595 |

| Spirals, realized gradient steps | | |
|---|---:|---|
| `backprop_neat` | 132,138 | — |
| `homogeneous_tanh` | 124,044 | comparable |
| `evolution_only` | 0 | zero *by construction*; that is the ablation |
| `random_search` | 4,018 | 33× less |
| `fixed_mlp` | 2,506 | **53× less** |

The fixed-MLP and random-architecture columns are **not** fair comparisons.
Ha's rollback rule halts a 69-node network after ~42 updates, so those two
conditions were starved rather than outperformed. Given a matched budget and no
rollback, the same 32×32 network reaches 0.78–0.91 on spirals, and a fixed
*sin* network reaches 0.94–0.96 in 600 plain steps — above every champion here.
See [`docs/v2-errata.md`](docs/v2-errata.md).

### The comparisons that survive

Both evolutionary ablations share the candidate budget by construction, so they
are clean. Paired within replicate, sealed test:

| Spirals, vs | Accuracy | Backprop-NEAT wins | Mean loss diff | 95% CI |
|---|---|---|---|---|
| Homogeneous tanh | 0.787 vs 0.745 | 7/10 | −0.088 | [−0.144, −0.030] |
| Evolution only | 0.787 vs 0.639 | 9/10 | −0.208 | [−0.262, −0.153] |

Operator diversity and gradient learning each contribute. Beating a fixed
architecture is not shown.

Validation-based selection did not overfit: mean validation→test drop is ≤0.032
in every condition and ≤0.011 for Backprop-NEAT. Total compute: 268,800
candidate evaluations, 11,653,669 realized gradient steps, **1.87 core-hours**.

## What is supported, and what is not

**Supported.**

1. **Gradient learning and topology search are complementary.** `evolution_only`
   loses on all three tasks at identical candidate budget — XOR 9/10, circles
   10/10, spirals 10/10.
2. **Operator diversity contributes.** `homogeneous_tanh` loses on circles
   (10/10) and spirals (7/10) at matched candidate budget.
3. **Causal operator usage differs by task** — `mult` on XOR (0.46),
   `square`+`gaussian` on circles (0.53), `sin` on spirals (0.52). This is a
   statement about which operators reach the output, and nothing more; see
   erratum E6 against the stronger reading.
4. **Represented structure is not computation.** Champions carry 2.6–4.6
   causally active hidden nodes against 6.6–8.7 represented ones. This is a
   measurement of the evolved graphs themselves and does not depend on any
   between-condition comparison.
5. **The release reproduces.** Every derived file rebuilds byte-identically
   from the raw records; CI asserts it on every push.

**Withdrawn** (see [`docs/v2-errata.md`](docs/v2-errata.md)).

- *That architecture search beats fixed architectures on deceptive geometry*
  (E1 — the control was starved 53×).
- *That 4.6 causal nodes beat a 65-unit MLP* (E1 — same cause; the comparison
  measures budgets, not architectures).
- *That Ha's propagation rule alone changes what evolution discovers* (E3 —
  the two tracks differ in propagation **and** fitness split).

**Narrowed.**

- The XOR sentence "controls match or beat Backprop-NEAT" was wrong on its own
  data: Backprop-NEAT wins 10/10 on accuracy and is perfect in 7/10. Mean BCE
  inverted the ranking because of two confident errors (E2).
- v1's collapse phenomenon is restored at a *rate*, not retracted: 3/10 XOR and
  1/10 spiral track-A champions collapse to zero causal hidden nodes at the
  logistic floor, making track-A XOR's 0.751 a bimodal mixture (E4).

**Never supported, by construction.** Anything about geopolitics, forecasting
or foreign policy. These are two-dimensional synthetic classification tasks.

The most tempting unsupported claim, stated so it can be resisted: *"evolution
designs better architectures than human engineers."* After E1, this repository
has no evidence for even the weak form of that. What it does have is a
measurement of how much the evaluator decided — which is what v3 is for.

## Figures

![The three task geometries](results/backprop-neat-v2/figures/task-geometries.png)

![Decision boundaries of the selected champions](results/backprop-neat-v2/figures/champion-boundaries.png)

![Champion topologies: causal structure solid, merely represented structure ghosted](results/backprop-neat-v2/figures/champion-topologies.png)

![Paired sealed-test-loss differences, track B](results/backprop-neat-v2/figures/paired-test-loss-track-b.png)

![Which operators reach the output, by task](results/backprop-neat-v2/figures/operator-usage-track-b.png)

Every figure and every table is derived from `raw/runs/*.json` in the release —
nothing is recomputed, no model is retrained. The champion drawings use, per
task, the replicate with the median sealed-test accuracy: a display choice, not
a selection claim.

## What the gates caught

Three findings about the *reference algorithm* surfaced before any
confirmatory compute was spent, and shaped the protocol:

**A naive baseline is silently zeroed by Ha's propagation rule.** The rule
stops propagation once every node has been touched, and the output node is
recomputed before every hidden node. Wire bias directly into both the output
and the hidden units — the obvious way to build an MLP — and the network
returns a constant: a 32×32 MLP built that way scores 0.500, a dead control.
Every baseline here routes its output bias through a carrier node and is
verified alive under both propagation modes. A comparison against a control
crippled this way would be void.

**Settling must be bounded and weight-independent.** Recurrent cycles carrying
`square`/`mult` diverge within ticks, so node values are clamped and the tick
count derives from topology by BFS — a value-dependent stopping rule would make
the traced function discontinuous. Finite-difference gradient error fell
4.5e14 → 6.6e-2 → 3.7e-7 as each cause was removed.

**Selection fidelity decides whether topologies grow at all.** The reference
selects parents by fitness-proportionate roulette over the whole subpopulation,
with no elite — an error-0.3 genome breeds only ~2.3× as often as an error-0.7
one. That weakness is functional: it lets neutral structure survive long enough
to combine. An earlier version of this code used elitism plus truncation, which
pinned topologies at the minimal seed and produced a false finding. Protocol v1
was invalidated for it rather than patched
([`docs/v1-invalidation.md`](docs/v1-invalidation.md)).

## Verify and reproduce

```bash
git clone https://github.com/ReloadLightly/neural-architecture-search-with-backprop
cd neural-architecture-search-with-backprop
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

pytest -q                                               # correctness gates
bpneat verify --dir results/backprop-neat-v2/track-a    # the release regenerates
bpneat verify --dir results/backprop-neat-v2/track-b    #   byte-identically
```

`bpneat verify` rebuilds every derived table from the raw run records in a
temporary directory and byte-compares it against the committed release, checks
every file hash in `sha256sums.txt`, and confirms the code fingerprint still
matches the one the suite ran under. The release is not something to take on
trust; it is something this command proves on your machine.

To regenerate tables and figures, or re-run the study:

```bash
bpneat report   --dir results/backprop-neat-v2/track-b
bpneat figures  --dir results/backprop-neat-v2
bpneat champions --dir results/backprop-neat-v2          # boundary + topology drawings

bpneat plan --track B                                    # 180 runs
bpneat suite --track B --out results/rerun/track-b       # shardable, resumable
bpneat final-test --dir results/rerun/track-b            # refuses drafts and modified code
```

Measured budget for the full 360-run matrix: 1.87 core-hours (32 minutes
across four shards). Shards never overwrite each other, completed runs
are skipped by code fingerprint, and `manifest.json` is refreshed after every
run so progress is read rather than guessed. `bpneat final-test` runs exactly
once per release: it refuses an incomplete, draft, already-evaluated or
code-modified suite.

## Repository map

| Path | What it is |
|---|---|
| `src/bpneat/` | the reconstruction and the experiment machinery; seven science modules are fingerprint-frozen to the release |
| `tests/` | the gates — dataset independence, gradient checks, propagation semantics, sealed-test isolation, resume-exactness, the final-test firewall |
| `bench/` | feasibility probes that sized the compute budget from measurement |
| `docs/` | the frozen protocol, the v1 invalidation, reference targets, and the write-up |
| `results/backprop-neat-v2/` | the citable release: raw records, tables, figures, `final-test.json`, checksums |
| `results/backprop-neat-v1/` | invalidated, retained so the invalidation can be audited |
| `run_shard.sh` | how the v2 suite was actually executed |

## Lineage

- David Ha, ["Neural Network Evolution Playground with Backprop
  NEAT"](https://blog.otoro.net/2016/05/07/backprop-neat/) (2016), and
  [`hardmaru/backprop-neat-js`](https://github.com/hardmaru/backprop-neat-js) —
  the reference this implementation was audited against.
- Risi, Tang, Ha & Miikkulainen, [*Neuroevolution: Harnessing Creativity in AI
  Agent Design*](https://neuroevolutionbook.com/), §10.1 "Neural Architecture
  Search with NEAT", Figures 10.1 and 10.3.
- Stanley & Miikkulainen, "Evolving Neural Networks through Augmenting
  Topologies", *Evolutionary Computation* 10(2), 2002 — NEAT itself.

On fidelity: this reconstruction reaches the reference accuracy regime but does
not reproduce the published champion sizes on any task; the discrepancy is
understood and documented in
[`docs/reference-targets.md`](docs/reference-targets.md), and no claim of
reproducing Figure 10.3 is made.

## License and citation

Apache-2.0 (see [`LICENSE`](LICENSE)). To cite this repository, use
[`CITATION.cff`](CITATION.cff) and cite the **v2 release** — v1 numbers are
invalidated and exist only for audit.
