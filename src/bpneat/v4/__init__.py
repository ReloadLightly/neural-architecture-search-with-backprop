"""Protocol v4 — is the matched-budget reversal specific to Backprop-NEAT?

v3 showed that Backprop-NEAT's measured advantage over fixed architectures is
produced by the control's training budget, not by the architectures evolution
finds. That result is about one algorithm. v4 asks whether the *sign* of a
search-versus-fixed comparison is set by the budget protocol or by the search
algorithm, by running a second, independently-derived topology search —
Cartesian Genetic Programming with gradient-trained candidates — through the
same controls, on dataset seeds no earlier release has touched.

Nothing here edits frozen v2 or released v3 code. Both are reused by import,
and v4 carries its own fingerprint set and its own protocol identity.
"""

from __future__ import annotations

__all__ = ["PROTOCOL_VERSION"]

PROTOCOL_VERSION = "v4"
