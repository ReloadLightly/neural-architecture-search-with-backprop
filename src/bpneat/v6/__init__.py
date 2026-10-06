"""Protocol v6 — does the architecture search pay at any budget?

Every release so far compared the search against its controls at **one** budget:
2100 candidate evaluations, which is what a population of 100 run for twenty
generations costs. Three releases then reported, in three different ways, that
the search does not beat a fixed network given the same compute (v3 block A, v4
H2-H5, v5 H7), and v4 added that Cartesian Genetic Programming's selection adds
little over its own candidate-matched null.

All of those are statements about one point on a curve. The obvious reading of
them — "search does not pay" — is not licensed by a single budget, because the
budget we happened to inherit from the published description is small. A search
that is indistinguishable from random sampling at 2100 candidates may separate
from it at 16800; a fixed network that wins at 2100 may stop winning when the
search is given room; and the reverse of both is equally possible, since more
search on a validation split is also more opportunity to overfit it.

v6 turns the budget into the factor. Five budgets spanning 33x, three arms at
every budget — the search, its candidate-matched null, and the budget-matched
fixed network — on the three geometries where conditions separate at all.

The arms are not new. ``search`` is v5's reference configuration, ``null`` is
v3's candidate-matched random architecture search, ``fixed`` is the 32x32
mixed-operator network at the search's realised gradient budget. Nothing about
*what* is compared changes; only how much compute each side is given.
"""

from __future__ import annotations

__all__ = ["PROTOCOL_VERSION"]

PROTOCOL_VERSION = "v6"
