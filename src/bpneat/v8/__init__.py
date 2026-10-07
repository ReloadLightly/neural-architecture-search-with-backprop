"""Protocol v8 — does the search pay on real data, or is linear the answer?

Every number v1 through v7 produced is about two-dimensional synthetic
geometries, because the frozen genome fixes two input nodes. :mod:`bpneat.nd`
lifts that, proving itself byte-identical to the frozen core at two inputs, and
v8 is the first protocol to use it.

The design changed once the data was measured, and the measurement is on the
record (`docs/v8-preregistration.md`). On all four real datasets a *linear*
model — the seed genome, bias and every input wired straight to the outputs, no
hidden units at all — lands within two to five points of everything else. The
search agrees from the other side: its champions come back with zero causally
active hidden units.

So v8 does not ask "does the search beat a fixed network". It asks the sharper
question that data makes available:

    **Does architecture search beat having no architecture?**

Four arms, every one of them at the same realised gradient budget: the search,
its candidate-matched null, a budget-matched fixed 32x32 network, and the linear
seed model. If the fixed network cannot beat linear either, that is the finding
and it is a statement about these datasets rather than about the search — which
is why it is a preregistered hypothesis of its own rather than a caveat.
"""

from __future__ import annotations

__all__ = ["PROTOCOL_VERSION"]

PROTOCOL_VERSION = "v8"
