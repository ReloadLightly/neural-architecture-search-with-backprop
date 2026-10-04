"""Protocol v3 — "When the evaluator decides".

v3 asks whether implementation details absent from the published description of
an evolution-with-learning algorithm — the learner's stopping rule, forward-pass
scheduling, the selection operator, weight inheritance — each decide the study's
conclusions.

All v2 code is imported, never edited: the seven v2 science modules are
fingerprint-bound to the v2 release. v3 carries its own fingerprint set
(:mod:`bpneat.v3.fingerprint`) and its own protocol identity.
"""

V3_VERSION = "v3"
