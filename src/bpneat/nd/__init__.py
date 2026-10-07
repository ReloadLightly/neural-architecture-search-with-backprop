"""The n-dimensional core: the frozen encoding, with the two inputs unfixed.

Every result in this repository is about five two-dimensional synthetic
geometries, and the README says plainly why: the frozen genome fixes two input
nodes in a module no protocol may edit, so higher-dimensional real datasets were
skipped rather than attempted badly. That is the ceiling on the whole project —
a claim about neural architecture search tested only on spirals is a claim about
spirals.

This package lifts the ceiling, and it has to earn the frozen core's credibility
rather than ask for it. The rule it is built under:

    At ``d = 2`` inputs and ``k = 1`` output, every function here must produce
    **byte-identical** results to the frozen module — the same genome, the same
    tape, the same forward values, the same gradients, the same champion, from
    the same seed. Not "agrees to 1e-10". Identical, because then the frozen
    path and the general path are the same computation and every earlier result
    transfers unchanged.

That is a gate (``tests/test_nd_equivalence.py``), not an aspiration. If it
cannot be made green, the generalisation has failed and is reported as failed.

Nothing here is a protocol. It is the encoding a protocol may be built on, with
its own fingerprint, and no claim may be made from it until one is.
"""

from __future__ import annotations

__all__ = ["ENCODING_VERSION"]

ENCODING_VERSION = "nd1"
