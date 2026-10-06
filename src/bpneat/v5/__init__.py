"""Protocol v5 — NEAT's own machinery, as the experiment.

v3 and v4 asked what the *evaluator* does to a comparison. Both answers were
about the inner learner and the controls, and both left NEAT's own mechanisms
untouched at their reference settings. That is a gap: the algorithm is called
*NeuroEvolution of Augmenting Topologies*, and nothing so far has tested whether
the augmenting, the speciation that is supposed to protect it, or the crossover
that is supposed to recombine it does any work.

v5 turns those into factors. Five mechanisms, each ablated or amplified against
the reference configuration, on the three geometries where conditions can
actually separate, with the budget-matched fixed network kept as the yardstick.

The frozen primitives — ``add_node``, ``add_connection``, ``mutate_weights``,
``crossover``, ``kmedoids``, ``evaluate`` — are imported unchanged from v2. Only
the *rates* and the reproduction loop are new, because the rates are module
constants in frozen code and cannot be varied without new code.
"""

from __future__ import annotations

__all__ = ["PROTOCOL_VERSION"]

PROTOCOL_VERSION = "v5"
