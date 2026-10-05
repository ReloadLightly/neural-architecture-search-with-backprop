"""Cartesian Genetic Programming with weighted connections, decoded to a genome.

The second search algorithm. It is chosen because it differs from Backprop-NEAT
in both of the places an evolutionary algorithm can differ:

* **Encoding.** A fixed-length genotype over a row of ``n_func`` function nodes.
  Node ``j`` reads two addresses strictly below its own, so the graph is
  feed-forward by construction. The *phenotype* is the subgraph reachable from
  the single output gene, so most genes are typically inactive — the
  genotype-phenotype map that CGP is built around. There are no innovation
  numbers, no species, and no crossover.
* **Search.** ``(1+lambda)`` with neutral drift: an offspring replaces the
  parent when its fitness is greater *or equal*, which lets the inactive region
  of the genotype drift for free. See :mod:`bpneat.v4.search`.

Weights live in the genotype, one per connection gene plus one bias per node
(weighted CGP, as in CGPANN). They are therefore inherited by an offspring for
every gene it does not mutate, which is the direct analogue of Backprop-NEAT's
Lamarckian inheritance.

Two deliberate restrictions, both declared in the preregistration:

* The operator set is :data:`bpneat.v4.cgp.OPS` — the eight operators the dense
  evaluator can express. ``mult`` aggregates by product and is excluded, so
  every CGP phenotype is evaluated exactly by one matrix product per
  topological depth no matter how deep it is. The frozen genome evaluator
  settles for at most ``SETTLE_MAX_TICK = 16`` ticks, which a 48-node CGP row
  can exceed; excluding ``mult`` removes that truncation hazard entirely.
* The output bias is routed through an ``add`` carrier node, following the
  convention in :mod:`bpneat.baselines`, so the output bias is a learned weight
  on a node rather than a direct ``bias -> output`` edge. This is a convention,
  not a guarantee: v4 evaluates every phenotype under **settled** propagation —
  the same setting v3's ``backprop_neat`` reference uses — and under Ha's
  asynchronous rule a decoded phenotype can still return a constant. Measured on
  300 random genotypes: 11 return zero under ``ha2016`` and all 300 are alive
  under ``settled``. No v4 condition uses ``ha2016``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..baselines import _bias_carrier, _connect
from ..genome import BIAS, N_STRUCTURAL, OP_NULL, OUT, Genome
from ..v3 import dense

#: Operators a CGP node may carry. ``mult`` is excluded; see the module docstring.
OPS: tuple[int, ...] = tuple(dense.DENSE_OPS)

#: Addressable terminals: bias, input x, input y. They map to genome nodes 0-2.
N_TERMINALS = 3
#: Inputs per function node. Classic CGP arity for a two-input operator set.
ARITY = 2

#: Function nodes in the row. Large enough that most genes are inactive, which
#: is the regime neutral drift needs.
N_FUNC = 48

#: Per-gene mutation probabilities, applied independently to every gene.
STRUCTURAL_RATE = 0.06
WEIGHT_RATE = 0.10
WEIGHT_SIZE = 0.5


@dataclass
class Genotype:
    """A fixed-length CGP genotype. ``op``/``src`` are structure, the rest weights."""

    op: np.ndarray  # (N_FUNC,) int64, values drawn from OPS
    src: np.ndarray  # (N_FUNC, ARITY) int64, addresses < N_TERMINALS + j
    w: np.ndarray  # (N_FUNC, ARITY) float64, one weight per input gene
    wb: np.ndarray  # (N_FUNC,) float64, per-node bias weight
    out_src: int  # address the output gene reads
    out_w: float  # weight on that edge
    out_wb: float  # output bias, carried by the add node
    out_cw: float  # carrier -> output weight (1.0 at birth, then learned)

    @property
    def n_func(self) -> int:
        return len(self.op)

    def copy(self) -> Genotype:
        return Genotype(
            op=self.op.copy(),
            src=self.src.copy(),
            w=self.w.copy(),
            wb=self.wb.copy(),
            out_src=self.out_src,
            out_w=self.out_w,
            out_wb=self.out_wb,
            out_cw=self.out_cw,
        )


def random_genotype(rng: np.random.Generator, n_func: int = N_FUNC) -> Genotype:
    """A uniform random legal genotype.

    CGP is conventionally seeded at random rather than from a minimal network.
    Backprop-NEAT instead starts every individual at logistic regression, which
    a single output gene cannot express; the difference in starting point is
    part of what makes this a second algorithm rather than a reparameterisation.
    """
    op = rng.choice(np.array(OPS, dtype=np.int64), size=n_func)
    src = np.empty((n_func, ARITY), dtype=np.int64)
    for j in range(n_func):
        src[j] = rng.integers(0, N_TERMINALS + j, size=ARITY)
    return Genotype(
        op=op.astype(np.int64),
        src=src,
        w=rng.normal(0.0, 1.0, (n_func, ARITY)),
        wb=rng.normal(0.0, 1.0, n_func),
        out_src=int(rng.integers(0, N_TERMINALS + n_func)),
        out_w=float(rng.normal(0.0, 1.0)),
        out_wb=0.0,
        out_cw=1.0,
    )


def mutate(gt: Genotype, rng: np.random.Generator) -> Genotype:
    """Probabilistic point mutation over every gene, structure and weight.

    If nothing changed, one structural gene is forced to change, so an offspring
    is never a bit-identical copy of its parent — that would make neutral drift
    indistinguishable from stagnation in the candidate count.
    """
    child = gt.copy()
    n = child.n_func
    changed = False

    for j in range(n):
        if rng.random() < STRUCTURAL_RATE:
            child.op[j] = int(rng.choice(np.array(OPS, dtype=np.int64)))
            changed = True
        for k in range(ARITY):
            if rng.random() < STRUCTURAL_RATE:
                child.src[j, k] = int(rng.integers(0, N_TERMINALS + j))
                changed = True
    if rng.random() < STRUCTURAL_RATE:
        child.out_src = int(rng.integers(0, N_TERMINALS + n))
        changed = True

    wmask = rng.random((n, ARITY)) < WEIGHT_RATE
    child.w = child.w + wmask * rng.normal(0.0, WEIGHT_SIZE, (n, ARITY))
    bmask = rng.random(n) < WEIGHT_RATE
    child.wb = child.wb + bmask * rng.normal(0.0, WEIGHT_SIZE, n)
    if rng.random() < WEIGHT_RATE:
        child.out_w += float(rng.normal(0.0, WEIGHT_SIZE))
    if rng.random() < WEIGHT_RATE:
        child.out_wb += float(rng.normal(0.0, WEIGHT_SIZE))
    if rng.random() < WEIGHT_RATE:
        child.out_cw += float(rng.normal(0.0, WEIGHT_SIZE))
    changed = changed or bool(wmask.any()) or bool(bmask.any())

    if not changed:
        j = int(rng.integers(0, n))
        child.op[j] = int(rng.choice(np.array(OPS, dtype=np.int64)))
    return child


def active_nodes(gt: Genotype) -> tuple[int, ...]:
    """Function-node indices reachable from the output gene, in row order."""
    seen: set[int] = set()
    stack = [gt.out_src]
    while stack:
        addr = stack.pop()
        if addr < N_TERMINALS:
            continue
        j = addr - N_TERMINALS
        if j in seen:
            continue
        seen.add(j)
        stack.extend(int(s) for s in gt.src[j])
    return tuple(sorted(seen))


@dataclass
class Phenotype:
    """The decoded genome, plus the map back to the genotype's weight slots."""

    genome: Genome
    #: Per genome connection, the genotype slot its weight came from. One of
    #: ``("wb", j, 0)``, ``("w", j, k)``, ``("out_wb", -1, 0)``,
    #: ``("out_w", -1, 0)``, ``("out_cw", -1, 0)``.
    slots: tuple[tuple[str, int, int], ...]
    active: tuple[int, ...]


def decode(gt: Genotype) -> Phenotype:
    """Build the phenotype: the subgraph the output gene actually reaches.

    Inactive function nodes are not represented at all, so graph size reported
    for a CGP champion is phenotype size. The row order is already topological,
    so the decoded genome is acyclic and :func:`bpneat.v3.dense.plan`
    always succeeds on it.
    """
    order = active_nodes(gt)
    node_of = {addr: addr for addr in range(N_TERMINALS)}  # bias, x, y -> 0, 1, 2
    for i, j in enumerate(order):
        node_of[N_TERMINALS + j] = N_STRUCTURAL + i

    g = Genome(ops=[OP_NULL] * N_STRUCTURAL)
    slots: list[tuple[str, int, int]] = []

    for j in order:
        g.ops.append(int(gt.op[j]))
        nid = g.n_nodes - 1
        _connect(g, BIAS, nid, float(gt.wb[j]))
        slots.append(("wb", j, 0))
        for k in range(ARITY):
            _connect(g, node_of[int(gt.src[j, k])], nid, float(gt.w[j, k]))
            slots.append(("w", j, k))

    carrier = _bias_carrier(g)
    # _bias_carrier appends the bias edge with weight 0.0; the output bias is
    # the learned value, so overwrite that slot rather than adding a second edge.
    g.weight[-1] = float(gt.out_wb)
    slots.append(("out_wb", -1, 0))
    _connect(g, node_of[int(gt.out_src)], OUT, float(gt.out_w))
    slots.append(("out_w", -1, 0))
    _connect(g, carrier, OUT, float(gt.out_cw))
    slots.append(("out_cw", -1, 0))

    assert len(slots) == g.n_connections
    return Phenotype(genome=g, slots=tuple(slots), active=order)


def write_back(gt: Genotype, pheno: Phenotype, weights: np.ndarray) -> None:
    """Copy trained weights back into the genotype, in place.

    This is what makes CGP Lamarckian here: an offspring inherits the parent's
    learned weights for every gene it does not mutate, exactly as a
    Backprop-NEAT child inherits its parent's trained connection weights.
    """
    for ci, (kind, j, k) in enumerate(pheno.slots):
        val = float(weights[ci])
        if kind == "wb":
            gt.wb[j] = val
        elif kind == "w":
            gt.w[j, k] = val
        elif kind == "out_wb":
            gt.out_wb = val
        elif kind == "out_w":
            gt.out_w = val
        elif kind == "out_cw":
            gt.out_cw = val
        else:  # pragma: no cover - slots are closed over the five kinds above
            raise AssertionError(kind)


def genotype_summary(gt: Genotype) -> dict:
    """Structure-only description, for the record. No weights, no data."""
    order = active_nodes(gt)
    return {
        "n_func": gt.n_func,
        "active_nodes": len(order),
        "active_fraction": len(order) / gt.n_func,
        "output_is_terminal": bool(gt.out_src < N_TERMINALS),
    }
